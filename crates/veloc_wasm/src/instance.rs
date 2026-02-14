use crate::Val;
use crate::engine::Strategy;
use crate::module::{GlobalInit, Module, ModuleArtifact};
use crate::store::{Instance, Store};
use crate::vm::{__sigsetjmp, TrapCode, VMContext, VMFuncRef};
use crate::{Extern, Result};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::ops::{Deref, DerefMut};
use hashbrown::HashMap;
use std::mem;
use veloc::interpreter::{Interpreter, InterpreterValue, MemoryRegion, Program, VM};
use veloc::ir::FuncId;
use wasmparser::{ExternalKind, ValType};

pub type ExternMap = HashMap<String, Extern>;

pub(crate) struct InstanceHandle(pub *mut VMInstance);

impl Deref for InstanceHandle {
    type Target = VMInstance;
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.0 }
    }
}

impl DerefMut for InstanceHandle {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.0 }
    }
}

impl Drop for InstanceHandle {
    fn drop(&mut self) {
        unsafe {
            let instance = &*self.0;
            let offset_of_vmctx = std::mem::offset_of!(VMInstance, vmctx);
            let total_size = (offset_of_vmctx as u32) + instance.module.vm_offsets().total_size;
            let layout = std::alloc::Layout::from_size_align(total_size as usize, 16).unwrap();

            std::ptr::drop_in_place(self.0);
            std::alloc::dealloc(self.0 as *mut u8, layout);
        }
    }
}

/// Instance 是 Module 绑定到对应 Store 后的运行实体
#[repr(C)]
pub(crate) struct VMInstance {
    pub(crate) module: Module,
    pub(crate) interpreter: Option<Interpreter>,
    pub(crate) vm: Option<veloc::interpreter::VM>,
    pub(crate) host_state: Box<dyn core::any::Any + Send + Sync>,
    pub(crate) element_lengths: Vec<usize>,
    pub(crate) data_lengths: Vec<usize>,
    pub(crate) vmctx_self_reference: *mut VMContext,
    pub(crate) vmctx: VMContext,
}

impl VMInstance {
    pub(crate) unsafe fn from_vmctx(ptr: *mut VMContext) -> &'static mut VMInstance {
        unsafe {
            let offset = std::mem::offset_of!(VMInstance, vmctx);
            let instance_ptr = (ptr as *mut u8).sub(offset) as *mut VMInstance;
            &mut *instance_ptr
        }
    }

    pub(crate) fn handle_trap(&self, trap_val: u32) -> crate::error::Error {
        let code = trap_val.saturating_sub(1);
        if let Some(trap) = TrapCode::from_u32(code) {
            crate::error::Error::Trap(trap)
        } else {
            crate::error::Error::Message(format!("Unknown trap code: {}", code))
        }
    }

    #[inline]
    fn vmctx_ptr(&self) -> *mut VMContext {
        self.vmctx_self_reference
    }

    #[inline]
    fn memories(&self) -> &mut [*mut crate::vm::VMMemory] {
        let meta = self.module.metadata();
        let offsets = self.module.vm_offsets();
        unsafe { (*self.vmctx_ptr()).memories_mut(offsets, meta.memories.len()) }
    }

    #[inline]
    fn tables(&self) -> &mut [*mut crate::vm::VMTable] {
        let meta = self.module.metadata();
        let offsets = self.module.vm_offsets();
        unsafe { (*self.vmctx_ptr()).tables_mut(offsets, meta.tables.len()) }
    }

    #[inline]
    fn globals(&self) -> &mut [*mut crate::vm::VMGlobal] {
        let meta = self.module.metadata();
        let offsets = self.module.vm_offsets();
        unsafe { (*self.vmctx_ptr()).globals_mut(offsets, meta.globals.len()) }
    }

    #[inline]
    fn functions(&self) -> &mut [crate::vm::VMFuncRef] {
        let meta = self.module.metadata();
        let offsets = self.module.vm_offsets();
        unsafe { (*self.vmctx_ptr()).functions_mut(offsets, meta.functions.len()) }
    }

    pub(crate) fn call_init(&mut self, program: &Program) -> crate::error::Result<()> {
        let init_func_id = self.module.init_func_id();

        let vmctx_ptr = self.vmctx_self_reference;
        let offsets = self.module.vm_offsets();
        let jmp_buf_ptr = unsafe { (*vmctx_ptr).jmp_buf_ptr(offsets) };

        let trap_val = unsafe { crate::vm::__sigsetjmp(jmp_buf_ptr, 0) };
        if trap_val != 0 {
            return Err(self.handle_trap(trap_val as u32));
        }

        if self.module.strategy() == Strategy::Interpreter {
            let vm = self
                .vm
                .as_mut()
                .ok_or_else(|| crate::error::Error::Message("VM not found".to_string()))?;
            let interpreter = self
                .interpreter
                .as_mut()
                .ok_or_else(|| crate::error::Error::Message("Interpreter not found".to_string()))?;

            let args = vec![InterpreterValue::I64(vmctx_ptr as i64)];

            interpreter.run_function(program, vm, init_func_id, &args);
        } else {
            let loaded = self.module.loaded().ok_or_else(|| {
                crate::error::Error::Message("Loaded object not found".to_string())
            })?;
            let init_ptr = unsafe {
                loaded.get::<*const ()>("__veloc_init").ok_or_else(|| {
                    crate::error::Error::Message("Failed to find init func".to_string())
                })?
            };
            let init_func: extern "C" fn(*mut VMContext) =
                unsafe { std::mem::transmute(*init_ptr) };
            init_func(vmctx_ptr);
        }
        Ok(())
    }

    pub(crate) fn get_memory(&self, index: u32) -> Option<&[u8]> {
        let mem = unsafe { &**self.memories().get(index as usize)? };
        Some(unsafe { core::slice::from_raw_parts(mem.base, mem.current_length) })
    }

    pub(crate) fn get_memory_mut(&mut self, index: u32) -> Option<&mut [u8]> {
        let mem = unsafe { &mut **self.memories().get(index as usize)? };
        Some(unsafe { core::slice::from_raw_parts_mut(mem.base, mem.current_length) })
    }

    pub(crate) fn memory_size(&self, index: u32) -> u32 {
        self.memories()
            .get(index as usize)
            .map(|&m| unsafe { (*m).current_length as u32 / 65536 })
            .unwrap_or(0)
    }

    pub(crate) fn memory_grow(&mut self, index: u32, delta: u32) -> crate::error::Result<u32> {
        let mem = unsafe {
            &mut **self.memories().get(index as usize).ok_or_else(|| {
                crate::error::Error::Message("Memory index out of bounds".to_string())
            })?
        };
        mem.grow(delta)
            .ok_or_else(|| crate::error::Error::Message("Memory grow failed".to_string()))
    }

    pub(crate) fn table_size(&self, index: u32) -> u32 {
        self.tables()
            .get(index as usize)
            .map(|&t| unsafe { (*t).current_elements as u32 })
            .unwrap_or(0)
    }

    pub(crate) fn table_grow(
        &self,
        index: u32,
        init_val: *mut VMFuncRef,
        delta: u32,
    ) -> crate::error::Result<i32> {
        let table = unsafe {
            &mut **self.tables().get(index as usize).ok_or_else(|| {
                crate::error::Error::Message("Table index out of bounds".to_string())
            })?
        };
        Ok(table.grow(delta, init_val))
    }

    pub(crate) fn table_fill(
        &self,
        index: u32,
        dst: u32,
        val: *mut VMFuncRef,
        len: u32,
    ) -> crate::error::Result<()> {
        let table = unsafe {
            &mut **self.tables().get(index as usize).ok_or_else(|| {
                crate::error::Error::Message("Table index out of bounds".to_string())
            })?
        };

        if (dst as usize)
            .checked_add(len as usize)
            .map_or(true, |end| end > table.current_elements)
        {
            return Err(crate::error::Error::Trap(TrapCode::TableOutOfBounds));
        }

        unsafe {
            let slice = core::slice::from_raw_parts_mut(table.base.add(dst as usize), len as usize);
            slice.fill(val);
        }
        Ok(())
    }

    pub(crate) fn table_copy(
        &self,
        dst_idx: u32,
        src_idx: u32,
        dst: u32,
        src: u32,
        len: u32,
    ) -> crate::error::Result<()> {
        let tables = self.tables();
        let dst_table = unsafe {
            &mut **tables.get(dst_idx as usize).ok_or_else(|| {
                crate::error::Error::Message("Table index out of bounds".to_string())
            })?
        };
        let src_table = unsafe {
            &**tables.get(src_idx as usize).ok_or_else(|| {
                crate::error::Error::Message("Table index out of bounds".to_string())
            })?
        };

        if (dst as usize)
            .checked_add(len as usize)
            .map_or(true, |end| end > dst_table.current_elements)
            || (src as usize)
                .checked_add(len as usize)
                .map_or(true, |end| end > src_table.current_elements)
        {
            return Err(crate::error::Error::Trap(TrapCode::TableOutOfBounds));
        }

        unsafe {
            core::ptr::copy(
                src_table.base.add(src as usize),
                dst_table.base.add(dst as usize),
                len as usize,
            );
        }
        Ok(())
    }

    pub(crate) fn table_init(
        &self,
        table_idx: u32,
        elem_idx: u32,
        dst: u32,
        src: u32,
        len: u32,
    ) -> crate::error::Result<()> {
        let meta = self.module.metadata();
        let table = unsafe {
            &mut **self.tables().get(table_idx as usize).ok_or_else(|| {
                crate::error::Error::Message("Table index out of bounds".to_string())
            })?
        };

        let elements = &meta.elements[elem_idx as usize];
        let elem_len = self.element_lengths[elem_idx as usize];

        if (dst as usize)
            .checked_add(len as usize)
            .map_or(true, |end| end > table.current_elements)
            || (src as usize)
                .checked_add(len as usize)
                .map_or(true, |end| end > elem_len)
        {
            return Err(crate::error::Error::Trap(TrapCode::TableOutOfBounds));
        }

        let functions = self.functions();
        let globals = self.globals();

        for i in 0..len {
            let ops = &elements.items[(src + i) as usize];
            let mut func_ref_ptr: *mut VMFuncRef = std::ptr::null_mut();
            for op in ops {
                match op {
                    GlobalInit::RefFunc(func_idx) => {
                        func_ref_ptr = &mut functions[*func_idx as usize] as *mut VMFuncRef;
                    }
                    GlobalInit::RefNull => {
                        func_ref_ptr = std::ptr::null_mut();
                    }
                    GlobalInit::GlobalGet(global_idx) => {
                        let val_ptr = globals[*global_idx as usize] as *mut *mut VMFuncRef;
                        func_ref_ptr = unsafe { *val_ptr };
                    }
                    _ => {}
                }
            }
            unsafe {
                *table.base.add((dst + i) as usize) = func_ref_ptr;
            }
        }
        Ok(())
    }

    pub(crate) fn elem_drop(&mut self, elem_idx: u32) {
        if let Some(len) = self.element_lengths.get_mut(elem_idx as usize) {
            *len = 0;
        }
    }

    pub(crate) fn memory_init(
        &self,
        mem_idx: u32,
        data_idx: u32,
        dst: u32,
        src: u32,
        len: u32,
    ) -> crate::error::Result<()> {
        let meta = self.module.metadata();
        let memory = unsafe {
            &mut **self.memories().get(mem_idx as usize).ok_or_else(|| {
                crate::error::Error::Message("Memory index out of bounds".to_string())
            })?
        };

        let data = &meta.data[data_idx as usize];
        let data_len = self.data_lengths[data_idx as usize];

        if (dst as usize)
            .checked_add(len as usize)
            .map_or(true, |end| end > memory.current_length)
            || (src as usize)
                .checked_add(len as usize)
                .map_or(true, |end| end > data_len)
        {
            return Err(crate::error::Error::Trap(TrapCode::MemoryOutOfBounds));
        }

        unsafe {
            core::ptr::copy_nonoverlapping(
                data.data.as_ptr().add(src as usize),
                memory.base.add(dst as usize),
                len as usize,
            );
        }
        Ok(())
    }

    pub(crate) fn data_drop(&mut self, data_idx: u32) {
        if let Some(len) = self.data_lengths.get_mut(data_idx as usize) {
            *len = 0;
        }
    }

    pub(crate) fn memory_copy(
        &self,
        dst_idx: u32,
        src_idx: u32,
        dst: u32,
        src: u32,
        len: u32,
    ) -> crate::error::Result<()> {
        let memories = self.memories();
        let dst_mem = unsafe {
            &mut **memories.get(dst_idx as usize).ok_or_else(|| {
                crate::error::Error::Message("Memory index out of bounds".to_string())
            })?
        };
        let src_mem = unsafe {
            &**memories.get(src_idx as usize).ok_or_else(|| {
                crate::error::Error::Message("Memory index out of bounds".to_string())
            })?
        };

        if (dst as usize)
            .checked_add(len as usize)
            .map_or(true, |end| end > dst_mem.current_length)
            || (src as usize)
                .checked_add(len as usize)
                .map_or(true, |end| end > src_mem.current_length)
        {
            return Err(crate::error::Error::Trap(TrapCode::MemoryOutOfBounds));
        }

        unsafe {
            core::ptr::copy(
                src_mem.base.add(src as usize),
                dst_mem.base.add(dst as usize),
                len as usize,
            );
        }
        Ok(())
    }

    pub(crate) fn memory_fill(
        &self,
        mem_idx: u32,
        dst: u32,
        val: u32,
        len: u32,
    ) -> crate::error::Result<()> {
        let memory = unsafe {
            &mut **self.memories().get(mem_idx as usize).ok_or_else(|| {
                crate::error::Error::Message("Memory index out of bounds".to_string())
            })?
        };

        if (dst as usize)
            .checked_add(len as usize)
            .map_or(true, |end| end > memory.current_length)
        {
            return Err(crate::error::Error::Trap(TrapCode::MemoryOutOfBounds));
        }

        unsafe {
            core::ptr::write_bytes(memory.base.add(dst as usize), val as u8, len as usize);
        }
        Ok(())
    }

    pub(crate) fn get_global(&self, index: u32) -> Option<crate::Val> {
        let global = unsafe { &**self.globals().get(index as usize)? };
        unsafe {
            match global.ty {
                ValType::I32 => Some(crate::Val::I32(global.value.i32)),
                ValType::I64 => Some(crate::Val::I64(global.value.i64)),
                ValType::F32 => Some(crate::Val::F32(global.value.f32)),
                ValType::F64 => Some(crate::Val::F64(global.value.f64)),
                _ => None,
            }
        }
    }

    pub(crate) fn set_global(&mut self, index: u32, val: crate::Val) -> crate::error::Result<()> {
        let global = unsafe {
            &mut **self.globals().get(index as usize).ok_or_else(|| {
                crate::error::Error::Message("Global index out of bounds".to_string())
            })?
        };
        if !global.mutable {
            return Err(crate::error::Error::Message(
                "Global is immutable".to_string(),
            ));
        }
        match (val, global.ty) {
            (crate::Val::I32(v), ValType::I32) => global.value.i32 = v,
            (crate::Val::I64(v), ValType::I64) => global.value.i64 = v,
            (crate::Val::F32(v), ValType::F32) => global.value.f32 = v,
            (crate::Val::F64(v), ValType::F64) => global.value.f64 = v,
            _ => {
                return Err(crate::error::Error::Message(
                    "Global type mismatch".to_string(),
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn get_export(&self, name: &str) -> Option<Extern> {
        let (kind, idx) = self.module.metadata().exports.get(name)?;
        let idx = *idx as usize;

        let export = match kind {
            ExternalKind::Func => Extern::Function(self.functions()[idx]),
            ExternalKind::Memory => Extern::Memory(self.memories()[idx]),
            ExternalKind::Table => Extern::Table(self.tables()[idx]),
            ExternalKind::Global => Extern::Global(self.globals()[idx]),
            _ => return None,
        };
        Some(export)
    }

    pub(crate) unsafe fn init_table_segment(
        &self,
        segment_idx: u32,
        offset: u32,
    ) -> crate::error::Result<()> {
        let meta = self.module.metadata();
        let element = &meta.elements[segment_idx as usize];
        let table = unsafe {
            &mut **self
                .tables()
                .get(element.table_index as usize)
                .ok_or_else(|| {
                    crate::error::Error::Message("Table index out of bounds".to_string())
                })?
        };

        if (offset as usize)
            .checked_add(element.items.len())
            .map_or(true, |end| end > table.current_elements)
        {
            return Err(crate::error::Error::Trap(TrapCode::TableOutOfBounds));
        }

        let functions = self.functions();
        let globals = self.globals();

        for (i, ops) in element.items.iter().enumerate() {
            let mut func_ref_ptr: *mut VMFuncRef = std::ptr::null_mut();
            for op in ops {
                match op {
                    GlobalInit::RefFunc(func_idx) => {
                        func_ref_ptr = &mut functions[*func_idx as usize] as *mut VMFuncRef;
                    }
                    GlobalInit::RefNull => {
                        func_ref_ptr = std::ptr::null_mut();
                    }
                    GlobalInit::GlobalGet(global_idx) => {
                        let val_ptr = globals[*global_idx as usize] as *mut *mut VMFuncRef;
                        func_ref_ptr = unsafe { *val_ptr };
                    }
                    _ => {}
                }
            }
            unsafe {
                *table.base.add((offset as usize) + i) = func_ref_ptr;
            }
        }
        Ok(())
    }

    pub(crate) unsafe fn init_memory_segment(
        &self,
        segment_idx: u32,
        offset: u32,
    ) -> crate::error::Result<()> {
        let data = &self.module.metadata().data[segment_idx as usize];
        let memory = unsafe {
            &mut **self
                .memories()
                .get(data.memory_index as usize)
                .ok_or_else(|| {
                    crate::error::Error::Message("Memory index out of bounds".to_string())
                })?
        };

        if (offset as usize)
            .checked_add(data.data.len())
            .map_or(true, |end| end > memory.current_length)
        {
            return Err(crate::error::Error::Trap(TrapCode::MemoryOutOfBounds));
        }

        unsafe {
            core::ptr::copy_nonoverlapping(
                data.data.as_ptr(),
                memory.base.add(offset as usize),
                data.data.len(),
            );
        }
        Ok(())
    }
}

impl VMInstance {
    pub(crate) fn new(
        store: &mut Store,
        module: Module,
        extern_imports: &[Extern],
    ) -> Result<Instance> {
        let meta = module.metadata();
        let offsets = module.vm_offsets();

        if extern_imports.len() != meta.imports.len() {
            return Err(crate::error::Error::Message(format!(
                "Import count mismatch: expected {}, got {}",
                meta.imports.len(),
                extern_imports.len()
            )));
        }

        // 分组导入
        let mut imported_funcs = Vec::new();
        let mut imported_tables = Vec::new();
        let mut imported_memories = Vec::new();
        let mut imported_globals = Vec::new();

        for (i, ext) in extern_imports.iter().enumerate() {
            let import_meta = &meta.imports[i];
            match (import_meta.kind, ext) {
                (ExternalKind::Func, Extern::Function(f)) => imported_funcs.push(*f),
                (ExternalKind::Table, Extern::Table(t)) => imported_tables.push(*t),
                (ExternalKind::Memory, Extern::Memory(m)) => imported_memories.push(*m),
                (ExternalKind::Global, Extern::Global(g)) => imported_globals.push(*g),
                (kind, _) => {
                    return Err(crate::error::Error::IncompatibleImport {
                        module: import_meta.module.clone(),
                        field: import_meta.field.clone(),
                        expected: format!("{:?}", kind),
                        actual: "mismatched type".into(),
                    });
                }
            }
        }

        // 1. 如果是解释器模式，记录模块 ID
        let interp_module_id = if module.strategy() == Strategy::Interpreter {
            match module.artifact() {
                ModuleArtifact::Interpreter(ir) => Some(store.program.register_module(ir.clone())),
                _ => None,
            }
        } else {
            None
        };

        // 2. 分配并初始化 joined block
        let offset_of_vmctx = std::mem::offset_of!(VMInstance, vmctx);
        let total_size = (offset_of_vmctx as u32) + offsets.total_size;
        let layout = std::alloc::Layout::from_size_align(total_size as usize, 16).unwrap();
        let base_ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if base_ptr.is_null() {
            return Err(crate::error::Error::Memory("Allocation failed".to_string()));
        }

        let instance_ptr = base_ptr as *mut VMInstance;
        let vmctx_ptr = unsafe { base_ptr.add(offset_of_vmctx) as *mut VMContext };

        unsafe {
            let mut vm = (module.strategy() == Strategy::Interpreter).then(VM::new);
            if let Some(vm) = vm.as_mut() {
                vm.register_region(MemoryRegion::new(
                    0,
                    0,
                    usize::MAX,
                    "host_memory".to_string(),
                ));
            }

            let instance = VMInstance {
                module: module.clone(),
                interpreter: interp_module_id.map(Interpreter::new),
                vm,
                host_state: Box::new(()),
                vmctx_self_reference: vmctx_ptr,
                vmctx: VMContext {
                    _maker: core::marker::PhantomPinned,
                },
                element_lengths: meta.elements.iter().map(|e| e.items.len()).collect(),
                data_lengths: meta.data.iter().map(|d| d.data.len()).collect(),
            };
            std::ptr::write(instance_ptr, instance);
        }

        // 3. 实例化资源并填充 VMContext
        unsafe {
            let instance = &mut *instance_ptr;

            // Signature Hashes
            let vm_hashes = (*vmctx_ptr).signature_hashes_mut(&offsets, meta.signatures.len());
            for (i, sig) in meta.signatures.iter().enumerate() {
                vm_hashes[i] = sig.hash;
            }

            // Globals
            let vm_globals = instance.globals();
            for i in 0..meta.num_imported_globals {
                let def_ptr = imported_globals[i];

                // Type check
                let actual_ty = (*def_ptr).ty;
                let actual_mutable = (*def_ptr).mutable;
                let expected_ty = meta.globals[i].ty;
                let expected_mutable = meta.globals[i].mutable;
                if actual_ty != expected_ty || actual_mutable != expected_mutable {
                    let import = meta
                        .imports
                        .iter()
                        .filter(|imp| imp.kind == ExternalKind::Global)
                        .nth(i)
                        .unwrap();
                    return Err(crate::error::Error::IncompatibleImport {
                        module: import.module.clone(),
                        field: import.field.clone(),
                        expected: format!("{:?} (mut:{})", expected_ty, expected_mutable),
                        actual: format!("{:?} (mut:{})", actual_ty, actual_mutable),
                    });
                }
                vm_globals[i] = def_ptr;
            }
            for i in meta.num_imported_globals..meta.globals.len() {
                let global_meta = &meta.globals[i];
                let default_val = match global_meta.ty {
                    wasmparser::ValType::I32 => crate::Val::I32(0),
                    wasmparser::ValType::I64 => crate::Val::I64(0),
                    wasmparser::ValType::F32 => crate::Val::F32(0.0),
                    wasmparser::ValType::F64 => crate::Val::F64(0.0),
                    _ => crate::Val::I64(0),
                };
                let id = store.alloc_global(default_val, global_meta.mutable);
                vm_globals[i] = store.get_global(id);
            }

            // Functions
            let vm_functions = instance.functions();
            for i in 0..meta.num_imported_funcs {
                let mut func_ref = imported_funcs[i];

                // If it's a host function (vmctx is null), set it to the current vmctx
                if func_ref.vmctx.is_null() {
                    func_ref.vmctx = vmctx_ptr;
                }

                // Type check
                let ty_idx = meta.functions[i].type_index;
                let expected_hash = meta.signatures[ty_idx as usize].hash;
                if func_ref.type_index != expected_hash {
                    let import = meta
                        .imports
                        .iter()
                        .filter(|imp| imp.kind == ExternalKind::Func)
                        .nth(i)
                        .unwrap();
                    return Err(crate::error::Error::IncompatibleImport {
                        module: import.module.clone(),
                        field: import.field.clone(),
                        expected: format!("sig hash {}", expected_hash),
                        actual: format!("sig hash {}", func_ref.type_index),
                    });
                }
                vm_functions[i] = func_ref;
            }
            for i in meta.num_imported_funcs..meta.functions.len() {
                let func_meta = &meta.functions[i];
                let ptr = if let Some(loaded) = module.loaded() {
                    let entry_sym = loaded.get::<*const ()>(&func_meta.name).ok_or_else(|| {
                        crate::error::Error::Message(format!("Symbol {} not found", func_meta.name))
                    })?;
                    *entry_sym as *const u8
                } else {
                    let mid = interp_module_id.expect("Interpreter module not registered");
                    store
                        .program
                        .get_interpreter_func_ptr(mid, func_meta.func_id)
                };

                vm_functions[i] = VMFuncRef {
                    native_call: ptr as *const core::ffi::c_void,
                    vmctx: vmctx_ptr,
                    type_index: meta.signatures[func_meta.type_index as usize].hash,
                    offset: 0,
                    caller: core::ptr::null_mut(),
                };
            }

            // Memories
            let vm_memories = instance.memories();
            for i in 0..meta.num_imported_memories {
                let def_ptr = imported_memories[i];

                // Type check
                let actual_pages = (*def_ptr).current_length / 65536;
                let expected_pages = meta.memories[i].initial as usize;
                if actual_pages < expected_pages {
                    let import = meta
                        .imports
                        .iter()
                        .filter(|imp| imp.kind == ExternalKind::Memory)
                        .nth(i)
                        .unwrap();
                    return Err(crate::error::Error::IncompatibleImport {
                        module: import.module.clone(),
                        field: import.field.clone(),
                        expected: format!("at least {} pages", expected_pages),
                        actual: format!("{} pages", actual_pages),
                    });
                }
                vm_memories[i] = def_ptr;
            }
            for i in meta.num_imported_memories..meta.memories.len() {
                let initial = meta.memories[i].initial as u32;
                let maximum = meta.memories[i].maximum.map(|m| m as u32);
                let id = store.alloc_memory(initial, maximum)?;
                vm_memories[i] = store.get_memory(id);
            }

            // Tables
            let vm_tables = instance.tables();
            for i in 0..meta.num_imported_tables {
                let def_ptr = imported_tables[i];
                vm_tables[i] = def_ptr;
            }
            for i in meta.num_imported_tables..meta.tables.len() {
                let table_meta = &meta.tables[i];
                let id = store.alloc_table(
                    table_meta.initial,
                    table_meta.maximum,
                    table_meta.element_type,
                )?;
                vm_tables[i] = store.get_table(id);

                if let Some(init_ops) = &table_meta.init {
                    let table = &mut *vm_tables[i];
                    let mut func_ref_ptr: *mut VMFuncRef = std::ptr::null_mut();
                    for op in init_ops {
                        match op {
                            GlobalInit::RefFunc(func_idx) => {
                                func_ref_ptr =
                                    &mut vm_functions[*func_idx as usize] as *mut VMFuncRef;
                            }
                            GlobalInit::RefNull => {
                                func_ref_ptr = std::ptr::null_mut();
                            }
                            GlobalInit::GlobalGet(global_idx) => {
                                let val_ptr =
                                    vm_globals[*global_idx as usize] as *mut *mut VMFuncRef;
                                func_ref_ptr = *val_ptr;
                            }
                            _ => {}
                        }
                    }
                    core::slice::from_raw_parts_mut(table.base, table.current_elements)
                        .fill(func_ref_ptr);
                }
            }
        }

        unsafe {
            // 4. 调用初始化函数 (初始化 globals, tables, memories)
            (&mut *instance_ptr).call_init(&store.program)?;
        }

        Ok(store.push_instance(InstanceHandle(instance_ptr)))
    }
}

impl Instance {
    pub fn get_export(self, store: &Store, name: &str) -> Option<Extern> {
        store.get_instance(self).get_export(name)
    }

    pub fn exports(self, store: &Store) -> ExternMap {
        let instance = store.get_instance(self);
        let meta = instance.module.metadata();
        let mut exports = HashMap::new();
        for name in meta.exports.keys() {
            if let Some(export) = instance.get_export(name) {
                exports.insert(name.clone(), export);
            }
        }
        exports
    }

    pub fn get_func(self, store: &Store, name: &str) -> Option<TypedFunc> {
        let instance = store.get_instance(self);
        let export = instance.get_export(name)?;

        if let Extern::Function(func_ref) = export {
            let meta = instance.module.metadata();
            let export_info = meta.exports.get(name)?;
            let func_idx = export_info.1 as usize;
            let func_id = meta.functions[func_idx].func_id;
            let ty_idx = meta.functions[func_idx].type_index;
            let sig = &meta.signatures[ty_idx as usize];

            let kind = if instance.module.strategy() == Strategy::Interpreter {
                TypedFuncKind::Interpreter {
                    target_func_id: func_id,
                }
            } else {
                let tramp_name = format!("{}_trampoline", name);
                let tramp_ptr = instance.module.loaded().and_then(|loaded| unsafe {
                    loaded
                        .get::<*const ()>(&tramp_name)
                        .map(|s| *s as *const u8)
                });

                TypedFuncKind::JIT {
                    trampoline_ptr: tramp_ptr?,
                }
            };

            Some(TypedFunc {
                params: sig.params.clone(),
                results: sig.results.clone(),
                func_ref,
                kind,
            })
        } else {
            None
        }
    }

    pub fn get_memory<'a>(self, store: &'a Store, index: u32) -> Option<&'a [u8]> {
        store.get_instance(self).get_memory(index)
    }

    pub fn get_memory_mut<'a>(self, store: &'a mut Store, index: u32) -> Option<&'a mut [u8]> {
        store.get_instance_mut(self).get_memory_mut(index)
    }

    pub fn memory_size(self, store: &Store, index: u32) -> u32 {
        store.get_instance(self).memory_size(index)
    }

    pub fn memory_grow(
        self,
        store: &mut Store,
        index: u32,
        delta: u32,
    ) -> crate::error::Result<u32> {
        store.get_instance_mut(self).memory_grow(index, delta)
    }

    pub fn get_global(self, store: &Store, index: u32) -> Option<crate::Val> {
        store.get_instance(self).get_global(index)
    }

    pub fn set_global(
        self,
        store: &mut Store,
        index: u32,
        val: crate::Val,
    ) -> crate::error::Result<()> {
        store.get_instance_mut(self).set_global(index, val)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum TypedFuncKind {
    JIT { trampoline_ptr: *const u8 },
    Interpreter { target_func_id: FuncId },
}

/// 类型化函数句柄
/// 可用于从宿主代码直接调用 Wasm 函数
pub struct TypedFunc {
    params: Box<[ValType]>,
    results: Box<[ValType]>,
    func_ref: VMFuncRef,
    kind: TypedFuncKind,
}

impl TypedFunc {
    pub fn call(&self, store: &mut Store, args: &[Val]) -> crate::error::Result<Vec<Val>> {
        let vmctx_ptr = self.func_ref.vmctx;

        unsafe {
            let instance = VMInstance::from_vmctx(vmctx_ptr);
            let offsets = &instance.module.vm_offsets();
            let jmp_buf_ptr = (*vmctx_ptr).jmp_buf_ptr(offsets);

            let trap_val = __sigsetjmp(jmp_buf_ptr, 0);
            if trap_val != 0 {
                return Err(instance.handle_trap(trap_val as u32));
            }

            if args.len() != self.params.len() {
                return Err(crate::error::Error::Message(format!(
                    "Argument count mismatch: expected {}, got {}",
                    self.params.len(),
                    args.len()
                )));
            }

            for (i, (arg, &expected)) in args.iter().zip(self.params.iter()).enumerate() {
                if arg.ty() != expected {
                    return Err(crate::error::Error::Message(format!(
                        "Argument {} type mismatch: expected {:?}, got {:?}",
                        i,
                        expected,
                        arg.ty()
                    )));
                }
            }

            let mut results_raw = vec![0i64; self.results.len()];

            let res_bits = match self.kind {
                TypedFuncKind::JIT { trampoline_ptr } => {
                    let mut storage = [0i64; 17];
                    for (i, arg) in args.iter().enumerate() {
                        storage[i] = arg.as_i64();
                    }
                    let mut actual_len = args.len();
                    if self.results.len() > 1 {
                        storage[actual_len] = results_raw.as_mut_ptr() as i64;
                        actual_len += 1;
                    }

                    if actual_len > 16 {
                        return Err(crate::error::Error::Message("Too many arguments".into()));
                    }

                    let trampoline: extern "C" fn(*const VMContext, *const i64) -> i64 =
                        mem::transmute(trampoline_ptr);
                    trampoline(vmctx_ptr, storage.as_ptr())
                }
                TypedFuncKind::Interpreter { target_func_id } => {
                    let mut int_args = Vec::with_capacity(args.len() + 2);
                    int_args.push(InterpreterValue::I64(vmctx_ptr as i64));
                    for arg in args {
                        int_args.push(arg.to_interpreter_val());
                    }

                    if self.results.len() > 1 {
                        int_args.push(InterpreterValue::I64(results_raw.as_mut_ptr() as i64));
                    }

                    let interpreter = instance.interpreter.as_mut().unwrap();
                    let vm = instance.vm.as_mut().unwrap();
                    interpreter
                        .run_function(&store.program, vm, target_func_id, &int_args)
                        .to_i64_bits()
                }
            };

            let results = if self.results.len() == 1 {
                vec![Val::from_i64(res_bits, self.results[0])]
            } else {
                self.results
                    .iter()
                    .enumerate()
                    .map(|(i, &ty)| Val::from_i64(results_raw[i], ty))
                    .collect()
            };

            Ok(results)
        }
    }
}
