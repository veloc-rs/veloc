use alloc::vec;
use alloc::vec::Vec;
use core::ptr;
use libc;
use mmap_rs::MmapOptions;
use wasmparser::ValType;

const WASM_PAGE_SIZE: usize = 65536;
const WASM_RESERVATION_SIZE: usize = 4 * 1024 * 1024 * 1024; // 4GB static memory

/// 传递给 JIT 代码的底层内存视图
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VMMemory {
    pub base: *mut u8,
    pub current_length: usize,
    pub maximum_pages: u32,
}

impl VMMemory {
    pub fn new(initial_pages: u32, maximum_pages: Option<u32>) -> crate::error::Result<Self> {
        let initial_size = (initial_pages as usize) * WASM_PAGE_SIZE;
        let maximum_pages = maximum_pages.unwrap_or(65536); // Default 4GB

        // 1. 预留完整的 4GB 虚拟地址空间
        // 我们先分配 PROT_NONE 的内存以预留空间
        let mut mmap = MmapOptions::new(WASM_RESERVATION_SIZE)
            .map_err(|e| crate::error::Error::Memory(format!("MmapOptions failed: {:?}", e)))?
            .map_mut()
            .map_err(|e| {
                crate::error::Error::Memory(format!("Memory reservation failed: {:?}", e))
            })?;

        let base = mmap.as_mut_ptr();

        // 初始时将整个区域设为不可访问，然后再 commit 初始大小的部分
        unsafe {
            libc::mprotect(base as *mut _, WASM_RESERVATION_SIZE, libc::PROT_NONE);
            if initial_size > 0 {
                libc::mprotect(
                    base as *mut _,
                    initial_size,
                    libc::PROT_READ | libc::PROT_WRITE,
                );
            }
        }

        // 泄漏 mmap 以保持其在 Instance 生命周期内有效
        std::mem::forget(mmap);

        Ok(Self {
            base,
            current_length: initial_size,
            maximum_pages,
        })
    }

    pub fn grow(&mut self, delta_pages: u32) -> Option<u32> {
        let old_size = self.current_length;
        let old_pages = (old_size / WASM_PAGE_SIZE) as u32;
        let new_pages = old_pages.checked_add(delta_pages)?;

        if new_pages > self.maximum_pages {
            return None;
        }

        let new_size = (new_pages as usize) * WASM_PAGE_SIZE;

        if delta_pages > 0 {
            unsafe {
                let grow_ptr = self.base.add(old_size);
                let grow_size = new_size - old_size;

                if libc::mprotect(
                    grow_ptr as *mut _,
                    grow_size,
                    libc::PROT_READ | libc::PROT_WRITE,
                ) != 0
                {
                    return None;
                }
            }
        }

        self.current_length = new_size;
        Some(old_pages)
    }

    pub const fn size() -> u32 {
        core::mem::size_of::<Self>() as u32
    }

    pub const fn base_offset() -> u32 {
        0
    }

    pub const fn current_length_offset() -> u32 {
        core::mem::size_of::<*mut u8>() as u32
    }

    pub const fn maximum_pages_offset() -> u32 {
        (core::mem::size_of::<*mut u8>() + core::mem::size_of::<usize>()) as u32
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VMFuncRef {
    /// Pointer to native code
    pub native_call: *const core::ffi::c_void,
    /// Pointer to function's own VMContext
    pub vmctx: *mut VMContext,
    /// Type index (canonicalized hash)
    pub type_index: u32,
    /// Optional offset or metadata
    pub offset: u32,
    /// Pointer to caller's VMContext (optional)
    pub caller: *mut VMContext,
}

impl VMFuncRef {
    pub fn null() -> Self {
        Self {
            native_call: ptr::null(),
            vmctx: ptr::null_mut(),
            type_index: u32::MAX,
            offset: 0,
            caller: ptr::null_mut(),
        }
    }

    pub fn is_null(&self) -> bool {
        self.native_call.is_null()
    }

    /// Size of the VMFuncRef structure
    pub const fn size() -> u32 {
        core::mem::size_of::<Self>() as u32
    }

    /// Offset of the native_call field
    pub const fn func_ptr_offset() -> u32 {
        0
    }

    /// Offset of the vmctx field
    pub const fn vmctx_offset() -> u32 {
        core::mem::size_of::<*const core::ffi::c_void>() as u32
    }

    /// Offset of the type_index field
    pub const fn type_index_offset() -> u32 {
        (core::mem::size_of::<*const core::ffi::c_void>() + core::mem::size_of::<*mut VMContext>())
            as u32
    }

    /// Offset of the offset field
    pub const fn offset_offset() -> u32 {
        (core::mem::size_of::<*const core::ffi::c_void>()
            + core::mem::size_of::<*mut VMContext>()
            + core::mem::size_of::<u32>()) as u32
    }

    /// Offset of the caller field
    pub const fn caller_offset() -> u32 {
        (core::mem::size_of::<*const core::ffi::c_void>()
            + core::mem::size_of::<*mut VMContext>()
            + core::mem::size_of::<u32>()
            + core::mem::size_of::<u32>()) as u32
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VMTable {
    pub base: *mut *mut VMFuncRef,
    pub current_elements: usize,
    pub maximum_elements: u32,
    pub element_type: wasmparser::RefType,
}

impl VMTable {
    pub fn new(size: u32, maximum: Option<u32>, element_type: wasmparser::RefType) -> Self {
        let mut elements = vec![ptr::null_mut::<VMFuncRef>(); size as usize];
        let base = elements.as_mut_ptr();
        let current_elements = elements.len();
        // 泄漏 Vec 以保持内存有效
        std::mem::forget(elements);
        Self {
            base,
            current_elements,
            maximum_elements: maximum.unwrap_or(u32::MAX),
            element_type,
        }
    }

    pub fn grow(&mut self, delta: u32, init_val: *mut VMFuncRef) -> i32 {
        let old_size = self.current_elements;
        let new_size = old_size + delta as usize;

        if new_size > self.maximum_elements as usize {
            return -1;
        }

        // We use from_raw_parts/resize/forget to manage the allocation
        // Note: This assumes the original table was also created via a Vec and leaked.
        unsafe {
            let mut elements = Vec::from_raw_parts(self.base, old_size, old_size);
            elements.resize(new_size, init_val);
            self.base = elements.as_mut_ptr();
            self.current_elements = elements.len();
            std::mem::forget(elements);
        }

        old_size as i32
    }

    pub fn set(&mut self, index: u32, func_ref_ptr: *mut VMFuncRef) {
        if (index as usize) < self.current_elements {
            unsafe {
                *self.base.add(index as usize) = func_ref_ptr;
            }
        }
    }

    pub const fn size() -> u32 {
        core::mem::size_of::<Self>() as u32
    }

    pub const fn base_offset() -> u32 {
        0
    }

    pub const fn current_elements_offset() -> u32 {
        core::mem::size_of::<*mut VMFuncRef>() as u32
    }
}

#[repr(C, align(16))]
pub struct VMGlobal {
    pub value: VMGlobalValue,
    pub ty: ValType,
    pub mutable: bool,
}

#[repr(C)]
pub union VMGlobalValue {
    pub i32: i32,
    pub i64: i64,
    pub f32: f32,
    pub f64: f64,
    pub u128: [u8; 16],
}

impl VMGlobal {
    pub fn new(ty: ValType, mutable: bool) -> Self {
        Self {
            value: VMGlobalValue { i64: 0 },
            ty,
            mutable,
        }
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrapCode {
    Unreachable = 0,
    TableOutOfBounds = 1,
    IndirectCallNull = 2,
    IntegerDivideByZero = 3,
    MemoryOutOfBounds = 4,
    IntegerOverflow = 5,
    IndirectCallBadSig = 6,
    InvalidConversionToInteger = 7,
    NullReference = 8,
}

impl TrapCode {
    pub fn from_u32(code: u32) -> Option<Self> {
        if code <= 8 {
            Some(unsafe { core::mem::transmute(code) })
        } else {
            None
        }
    }
}

impl core::fmt::Display for TrapCode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            TrapCode::Unreachable => "unreachable",
            TrapCode::TableOutOfBounds => "table out of bounds",
            TrapCode::IndirectCallNull => "indirect call null",
            TrapCode::IntegerDivideByZero => "integer divide by zero",
            TrapCode::MemoryOutOfBounds => "memory out of bounds",
            TrapCode::IntegerOverflow => "integer overflow",
            TrapCode::IndirectCallBadSig => "indirect call bad sig",
            TrapCode::InvalidConversionToInteger => "invalid conversion to integer",
            TrapCode::NullReference => "null reference",
        };
        write!(f, "{}", s)
    }
}

/// VMContext 是 JIT 代码执行时的上下文。
/// 固定头部包含陷阱处理和运行时函数，后面紧跟动态生成的各类定义。
#[repr(C, align(16))]
pub struct VMContext {
    pub(crate) _maker: core::marker::PhantomPinned,
}

unsafe extern "C" {
    pub fn __sigsetjmp(env: *mut u8, savemask: i32) -> i32;
    pub fn siglongjmp(env: *mut u8, val: i32) -> !;
}

impl VMContext {
    /// 获取 jmp_buf 指针
    pub unsafe fn jmp_buf_ptr(&self, offsets: &VMOffsets) -> *mut u8 {
        unsafe { (self as *const Self as *mut u8).add(offsets.jmp_buf as usize) }
    }

    /// 获取 memories 指针的切片
    pub unsafe fn memories_mut(
        &mut self,
        offsets: &VMOffsets,
        count: usize,
    ) -> &mut [*mut VMMemory] {
        if count == 0 {
            return &mut [];
        }
        let ptr = unsafe {
            (self as *mut Self as *mut u8).add(offsets.memories as usize) as *mut *mut VMMemory
        };
        unsafe { core::slice::from_raw_parts_mut(ptr, count) }
    }

    /// 获取 tables 指针的切片
    pub unsafe fn tables_mut(&mut self, offsets: &VMOffsets, count: usize) -> &mut [*mut VMTable] {
        if count == 0 {
            return &mut [];
        }
        let ptr = unsafe {
            (self as *mut Self as *mut u8).add(offsets.tables as usize) as *mut *mut VMTable
        };
        unsafe { core::slice::from_raw_parts_mut(ptr, count) }
    }

    /// 获取 globals 指针的切片
    pub unsafe fn globals_mut(
        &mut self,
        offsets: &VMOffsets,
        count: usize,
    ) -> &mut [*mut VMGlobal] {
        if count == 0 {
            return &mut [];
        }
        let ptr = unsafe {
            (self as *mut Self as *mut u8).add(offsets.globals as usize) as *mut *mut VMGlobal
        };
        unsafe { core::slice::from_raw_parts_mut(ptr, count) }
    }

    /// 获取 functions 定义的切片
    pub unsafe fn functions_mut(&mut self, offsets: &VMOffsets, count: usize) -> &mut [VMFuncRef] {
        if count == 0 {
            return &mut [];
        }
        let ptr = unsafe {
            (self as *mut Self as *mut u8).add(offsets.functions as usize) as *mut VMFuncRef
        };
        unsafe { core::slice::from_raw_parts_mut(ptr, count) }
    }

    /// 获取 signature_hashes 的切片
    pub unsafe fn signature_hashes_mut(&mut self, offsets: &VMOffsets, count: usize) -> &mut [u32] {
        if count == 0 {
            return &mut [];
        }
        let ptr = unsafe {
            (self as *mut Self as *mut u8).add(offsets.signature_hashes as usize) as *mut u32
        };
        unsafe { core::slice::from_raw_parts_mut(ptr, count) }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VMOffsets {
    pub memories: u32,
    pub tables: u32,
    pub globals: u32,
    pub functions: u32,
    pub signature_hashes: u32,
    pub jmp_buf: u32,
    pub total_size: u32,
}

impl VMOffsets {
    pub fn new(
        num_memories: u32,
        num_tables: u32,
        num_globals: u32,
        num_functions: u32,
        num_signatures: u32,
    ) -> Self {
        fn align(offset: u32, alignment: u32) -> u32 {
            (offset + alignment - 1) & !(alignment - 1)
        }

        // VMContext 结构体
        let mut offset = 0;

        // 1. Memories (Hot，紧跟在 Header 后面, 指向 VMMemory 的指针)
        offset = align(offset, 16);
        let memories = offset;
        offset += num_memories * Self::memory_elem_size();

        // 2. Tables (指向 VMTable 的指针)
        offset = align(offset, 16);
        let tables = offset;
        offset += num_tables * Self::table_elem_size();

        // 3. Globals (指向 VMGlobalDefinition 的指针)
        offset = align(offset, 16);
        let globals = offset;
        offset += num_globals * Self::global_elem_size();

        // 4. Functions (Storing VMFuncRef structs directly)
        offset = align(offset, core::mem::align_of::<VMFuncRef>() as u32);
        let functions = offset;
        offset += num_functions * VMFuncRef::size();

        // 5. Signature Hashes
        offset = align(offset, 16);
        let signature_hashes = offset;
        offset += num_signatures * core::mem::size_of::<u32>() as u32;

        // 6. jmp_buf (放在最后，对齐到 16 字节)
        offset = align(offset, 16);
        let jmp_buf = offset;
        // libc::jmp_buf 在通常系统上是 200 字节左右，我们预留更大并对齐
        offset += 256;

        Self {
            memories,
            tables,
            globals,
            functions,
            signature_hashes,
            jmp_buf,
            total_size: offset,
        }
    }

    fn memory_elem_size() -> u32 {
        core::mem::size_of::<*mut VMMemory>() as u32
    }

    fn table_elem_size() -> u32 {
        core::mem::size_of::<*mut VMTable>() as u32
    }

    fn global_elem_size() -> u32 {
        core::mem::size_of::<*mut VMGlobal>() as u32
    }

    pub fn jmp_buf(&self) -> u32 {
        self.jmp_buf
    }

    pub fn memory_offset(&self, index: u32) -> u32 {
        self.memories + index * Self::memory_elem_size()
    }

    pub fn table_offset(&self, index: u32) -> u32 {
        self.tables + index * Self::table_elem_size()
    }

    pub fn global_offset(&self, index: u32) -> u32 {
        self.globals + index * Self::global_elem_size()
    }

    pub fn function_offset(&self, index: u32) -> u32 {
        self.functions + index * VMFuncRef::size()
    }
}
