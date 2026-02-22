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

    /// 获取 imported memories 指针的切片（用于实例化时填充）
    pub unsafe fn imported_memories_mut(
        &mut self,
        offsets: &VMOffsets,
        count: usize,
    ) -> &mut [*mut VMMemory] {
        if count == 0 {
            return &mut [];
        }
        let ptr = unsafe {
            (self as *mut Self as *mut u8).add(offsets.imported_memories as usize)
                as *mut *mut VMMemory
        };
        unsafe { core::slice::from_raw_parts_mut(ptr, count) }
    }

    /// 获取 imported tables 指针的切片（用于实例化时填充）
    pub unsafe fn imported_tables_mut(
        &mut self,
        offsets: &VMOffsets,
        count: usize,
    ) -> &mut [*mut VMTable] {
        if count == 0 {
            return &mut [];
        }
        let ptr = unsafe {
            (self as *mut Self as *mut u8).add(offsets.imported_tables as usize)
                as *mut *mut VMTable
        };
        unsafe { core::slice::from_raw_parts_mut(ptr, count) }
    }

    /// 获取 imported globals 指针的切片（用于实例化时填充）
    pub unsafe fn imported_globals_mut(
        &mut self,
        offsets: &VMOffsets,
        count: usize,
    ) -> &mut [*mut VMGlobal] {
        if count == 0 {
            return &mut [];
        }
        let ptr = unsafe {
            (self as *mut Self as *mut u8).add(offsets.imported_globals as usize)
                as *mut *mut VMGlobal
        };
        unsafe { core::slice::from_raw_parts_mut(ptr, count) }
    }

    /// 获取本地定义的 memories 的切片（直接内联存储）
    pub unsafe fn local_memories_mut(
        &mut self,
        offsets: &VMOffsets,
        count: usize,
    ) -> &mut [VMMemory] {
        if count == 0 {
            return &mut [];
        }
        let ptr = unsafe {
            (self as *mut Self as *mut u8).add(offsets.local_memories as usize) as *mut VMMemory
        };
        unsafe { core::slice::from_raw_parts_mut(ptr, count) }
    }

    /// 获取本地定义的 tables 的切片（直接内联存储）
    pub unsafe fn local_tables_mut(&mut self, offsets: &VMOffsets, count: usize) -> &mut [VMTable] {
        if count == 0 {
            return &mut [];
        }
        let ptr = unsafe {
            (self as *mut Self as *mut u8).add(offsets.local_tables as usize) as *mut VMTable
        };
        unsafe { core::slice::from_raw_parts_mut(ptr, count) }
    }

    /// 获取本地定义的 globals 的切片（直接内联存储）
    pub unsafe fn local_globals_mut(
        &mut self,
        offsets: &VMOffsets,
        count: usize,
    ) -> &mut [VMGlobal] {
        if count == 0 {
            return &mut [];
        }
        let ptr = unsafe {
            (self as *mut Self as *mut u8).add(offsets.local_globals as usize) as *mut VMGlobal
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

    /// 通过索引获取 memory（区分导入和本地）
    pub unsafe fn get_memory(
        &self,
        offsets: &VMOffsets,
        index: u32,
        num_imported: u32,
    ) -> *const VMMemory {
        let base = self as *const Self as *const u8;
        if index < num_imported {
            // 导入的：通过指针访问
            let ptr_array =
                unsafe { base.add(offsets.imported_memories as usize) } as *const *const VMMemory;
            unsafe { *ptr_array.add(index as usize) }
        } else {
            // 本地的：直接内联访问
            let local_idx = index - num_imported;
            let local_base =
                unsafe { base.add(offsets.local_memories as usize) } as *const VMMemory;
            unsafe { local_base.add(local_idx as usize) }
        }
    }

    /// 通过索引获取 memory 的可变指针（区分导入和本地）
    pub unsafe fn get_memory_mut(
        &mut self,
        offsets: &VMOffsets,
        index: u32,
        num_imported: u32,
    ) -> *mut VMMemory {
        let base = self as *mut Self as *mut u8;
        if index < num_imported {
            // 导入的：通过指针访问
            let ptr_array =
                unsafe { base.add(offsets.imported_memories as usize) } as *mut *mut VMMemory;
            unsafe { *ptr_array.add(index as usize) }
        } else {
            // 本地的：直接内联访问
            let local_idx = index - num_imported;
            let local_base = unsafe { base.add(offsets.local_memories as usize) } as *mut VMMemory;
            unsafe { local_base.add(local_idx as usize) }
        }
    }

    /// 通过索引获取 table（区分导入和本地）
    pub unsafe fn get_table(
        &self,
        offsets: &VMOffsets,
        index: u32,
        num_imported: u32,
    ) -> *const VMTable {
        let base = self as *const Self as *const u8;
        if index < num_imported {
            let ptr_array =
                unsafe { base.add(offsets.imported_tables as usize) } as *const *const VMTable;
            unsafe { *ptr_array.add(index as usize) }
        } else {
            let local_idx = index - num_imported;
            let local_base = unsafe { base.add(offsets.local_tables as usize) } as *const VMTable;
            unsafe { local_base.add(local_idx as usize) }
        }
    }

    /// 通过索引获取 table 的可变指针（区分导入和本地）
    pub unsafe fn get_table_mut(
        &mut self,
        offsets: &VMOffsets,
        index: u32,
        num_imported: u32,
    ) -> *mut VMTable {
        let base = self as *mut Self as *mut u8;
        if index < num_imported {
            let ptr_array =
                unsafe { base.add(offsets.imported_tables as usize) } as *mut *mut VMTable;
            unsafe { *ptr_array.add(index as usize) }
        } else {
            let local_idx = index - num_imported;
            let local_base = unsafe { base.add(offsets.local_tables as usize) } as *mut VMTable;
            unsafe { local_base.add(local_idx as usize) }
        }
    }

    /// 通过索引获取 global（区分导入和本地）
    pub unsafe fn get_global(
        &self,
        offsets: &VMOffsets,
        index: u32,
        num_imported: u32,
    ) -> *const VMGlobal {
        let base = self as *const Self as *const u8;
        if index < num_imported {
            let ptr_array =
                unsafe { base.add(offsets.imported_globals as usize) } as *const *const VMGlobal;
            unsafe { *ptr_array.add(index as usize) }
        } else {
            let local_idx = index - num_imported;
            let local_base = unsafe { base.add(offsets.local_globals as usize) } as *const VMGlobal;
            unsafe { local_base.add(local_idx as usize) }
        }
    }

    /// 通过索引获取 global 的可变指针（区分导入和本地）
    pub unsafe fn get_global_mut(
        &mut self,
        offsets: &VMOffsets,
        index: u32,
        num_imported: u32,
    ) -> *mut VMGlobal {
        let base = self as *mut Self as *mut u8;
        if index < num_imported {
            let ptr_array =
                unsafe { base.add(offsets.imported_globals as usize) } as *mut *mut VMGlobal;
            unsafe { *ptr_array.add(index as usize) }
        } else {
            let local_idx = index - num_imported;
            let local_base = unsafe { base.add(offsets.local_globals as usize) } as *mut VMGlobal;
            unsafe { local_base.add(local_idx as usize) }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VMOffsets {
    // 导入的定义：存储指针
    pub imported_memories: u32,
    pub imported_tables: u32,
    pub imported_globals: u32,
    // 本地定义：直接内联存储
    pub local_memories: u32,
    pub local_tables: u32,
    pub local_globals: u32,
    // 函数和签名哈希始终内联
    pub functions: u32,
    pub signature_hashes: u32,
    pub jmp_buf: u32,
    pub total_size: u32,
    // 统计信息
    pub num_imported_memories: u32,
    pub num_imported_tables: u32,
    pub num_imported_globals: u32,
    pub num_local_memories: u32,
    pub num_local_tables: u32,
    pub num_local_globals: u32,
}

impl VMOffsets {
    pub fn new(
        num_imported_memories: u32,
        num_imported_tables: u32,
        num_imported_globals: u32,
        num_local_memories: u32,
        num_local_tables: u32,
        num_local_globals: u32,
        num_functions: u32,
        num_signatures: u32,
    ) -> Self {
        fn align(offset: u32, alignment: u32) -> u32 {
            (offset + alignment - 1) & !(alignment - 1)
        }

        let mut offset = 0;

        // 1. Imported Memories (存储指向外部 VMMemory 的指针)
        offset = align(offset, 16);
        let imported_memories = offset;
        offset += num_imported_memories * Self::imported_memory_elem_size();

        // 2. Imported Tables (存储指向外部 VMTable 的指针)
        offset = align(offset, 16);
        let imported_tables = offset;
        offset += num_imported_tables * Self::imported_table_elem_size();

        // 3. Imported Globals (存储指向外部 VMGlobal 的指针)
        offset = align(offset, 16);
        let imported_globals = offset;
        offset += num_imported_globals * Self::imported_global_elem_size();

        // 4. Local Memories (直接内联存储 VMMemory 结构体)
        offset = align(offset, 16);
        let local_memories = offset;
        offset += num_local_memories * VMMemory::size();

        // 5. Local Tables (直接内联存储 VMTable 结构体)
        offset = align(offset, 16);
        let local_tables = offset;
        offset += num_local_tables * VMTable::size();

        // 6. Local Globals (直接内联存储 VMGlobal 结构体)
        offset = align(offset, 16);
        let local_globals = offset;
        offset += num_local_globals * core::mem::size_of::<VMGlobal>() as u32;

        // 7. Functions (直接内联存储 VMFuncRef 结构体)
        offset = align(offset, core::mem::align_of::<VMFuncRef>() as u32);
        let functions = offset;
        offset += num_functions * VMFuncRef::size();

        // 8. Signature Hashes
        offset = align(offset, 16);
        let signature_hashes = offset;
        offset += num_signatures * core::mem::size_of::<u32>() as u32;

        // 9. jmp_buf (放在最后，对齐到 16 字节)
        offset = align(offset, 16);
        let jmp_buf = offset;
        // libc::jmp_buf 在通常系统上是 200 字节左右，我们预留更大并对齐
        offset += 256;

        Self {
            imported_memories,
            imported_tables,
            imported_globals,
            local_memories,
            local_tables,
            local_globals,
            functions,
            signature_hashes,
            jmp_buf,
            total_size: offset,
            num_imported_memories,
            num_imported_tables,
            num_imported_globals,
            num_local_memories,
            num_local_tables,
            num_local_globals,
        }
    }

    fn imported_memory_elem_size() -> u32 {
        core::mem::size_of::<*mut VMMemory>() as u32
    }

    fn imported_table_elem_size() -> u32 {
        core::mem::size_of::<*mut VMTable>() as u32
    }

    fn imported_global_elem_size() -> u32 {
        core::mem::size_of::<*mut VMGlobal>() as u32
    }

    pub fn jmp_buf(&self) -> u32 {
        self.jmp_buf
    }

    /// 获取导入 memory 的偏移（存储的是指针）
    pub fn imported_memory_offset(&self, index: u32) -> u32 {
        self.imported_memories + index * Self::imported_memory_elem_size()
    }

    /// 获取导入 table 的偏移（存储的是指针）
    pub fn imported_table_offset(&self, index: u32) -> u32 {
        self.imported_tables + index * Self::imported_table_elem_size()
    }

    /// 获取导入 global 的偏移（存储的是指针）
    pub fn imported_global_offset(&self, index: u32) -> u32 {
        self.imported_globals + index * Self::imported_global_elem_size()
    }

    /// 获取本地 memory 的偏移（直接存储 VMMemory）
    pub fn local_memory_offset(&self, index: u32) -> u32 {
        self.local_memories + index * VMMemory::size()
    }

    /// 获取本地 table 的偏移（直接存储 VMTable）
    pub fn local_table_offset(&self, index: u32) -> u32 {
        self.local_tables + index * VMTable::size()
    }

    /// 获取本地 global 的偏移（直接存储 VMGlobal）
    pub fn local_global_offset(&self, index: u32) -> u32 {
        self.local_globals + index * core::mem::size_of::<VMGlobal>() as u32
    }

    /// 获取任意 memory 的访问偏移（供 JIT 使用）
    /// 返回 (is_imported, offset) 元组
    pub fn memory_access_info(&self, index: u32) -> (bool, u32) {
        if index < self.num_imported_memories {
            (true, self.imported_memory_offset(index))
        } else {
            (
                false,
                self.local_memory_offset(index - self.num_imported_memories),
            )
        }
    }

    /// 获取任意 table 的访问信息
    pub fn table_access_info(&self, index: u32) -> (bool, u32) {
        if index < self.num_imported_tables {
            (true, self.imported_table_offset(index))
        } else {
            (
                false,
                self.local_table_offset(index - self.num_imported_tables),
            )
        }
    }

    /// 获取任意 global 的访问信息
    pub fn global_access_info(&self, index: u32) -> (bool, u32) {
        if index < self.num_imported_globals {
            (true, self.imported_global_offset(index))
        } else {
            (
                false,
                self.local_global_offset(index - self.num_imported_globals),
            )
        }
    }

    pub fn function_offset(&self, index: u32) -> u32 {
        self.functions + index * VMFuncRef::size()
    }
}
