use crate::Extern;
use crate::Val;
use crate::engine::Strategy;
use crate::module::{GlobalInit, Module, ModuleArtifact};
use crate::store::{InstanceId, Store};
use crate::vm::{__sigsetjmp, TrapCode, VMContext, VMFuncRef};
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

pub struct InstanceHandle(pub *mut Instance);

impl Deref for InstanceHandle {
    type Target = Instance;
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
            let offset_of_vmctx = std::mem::offset_of!(Instance, vmctx);
            let total_size = (offset_of_vmctx as u32) + instance.module.inner.offsets.total_size;
            let layout = std::alloc::Layout::from_size_align(total_size as usize, 16).unwrap();

            std::ptr::drop_in_place(self.0);
            std::alloc::dealloc(self.0 as *mut u8, layout);
        }
    }
}

/// Instance 是 Module 绑定到对应 Store 后的运行实体
#[repr(C)]
pub struct Instance {
    pub(crate) module: Module,
    pub(crate) interpreter: Option<Interpreter>,
    pub(crate) vm: Option<veloc::interpreter::VM>,
    pub(crate) host_state: Box<dyn core::any::Any + Send + Sync>,
    pub(crate) element_lengths: Vec<usize>,
    pub(crate) data_lengths: Vec<usize>,
    pub(crate) vmctx_self_reference: *mut VMContext,
    pub(crate) vmctx: VMContext,
}

impl Instance {
    pub(crate) unsafe fn from_vmctx(ptr: *mut VMContext) -> &'static mut Instance {
        unsafe {
            let offset = std::mem::offset_of!(Instance, vmctx);
            let instance_ptr = (ptr as *mut u8).sub(offset) as *mut Instance;
            &mut *instance_ptr
        }
    }

    pub(crate) fn handle_trap(&self, trap_val: u32) -> crate::error::Error {
        let code = (trap_val - 1) as u32;
        match code {
            x if x == TrapCode::Unreachable as u32 => {
                crate::error::Error::Trap(TrapCode::Unreachable)
            }
            x if x == TrapCode::TableOutOfBounds as u32 => {
                crate::error::Error::Trap(TrapCode::TableOutOfBounds)
            }
            x if x == TrapCode::IndirectCallNull as u32 => {
                crate::error::Error::Trap(TrapCode::IndirectCallNull)
            }
            x if x == TrapCode::IntegerDivideByZero as u32 => {
                crate::error::Error::Trap(TrapCode::IntegerDivideByZero)
            }
            x if x == TrapCode::MemoryOutOfBounds as u32 => {
                crate::error::Error::Trap(TrapCode::MemoryOutOfBounds)
            }
            x if x == TrapCode::IntegerOverflow as u32 => {
                crate::error::Error::Trap(TrapCode::IntegerOverflow)
            }
            x if x == TrapCode::IndirectCallBadSig as u32 => {
                crate::error::Error::Trap(TrapCode::IndirectCallBadSig)
            }
            _ => crate::error::Error::Message("Unknown trap code".to_string()),
        }
    }

    pub(crate) fn call_init(&mut self, program: &Program) -> crate::error::Result<()> {
        let init_func_id = self.module.inner.init_func_id;

        let vmctx_ptr = self.vmctx_self_reference;
        let offsets = &self.module.inner.offsets;
        let jmp_buf_ptr = unsafe { (*vmctx_ptr).jmp_buf_ptr(offsets) };

        let trap_val = unsafe { crate::vm::__sigsetjmp(jmp_buf_ptr, 0) };
        if trap_val != 0 {
            return Err(self.handle_trap(trap_val as u32));
        }

        if self.module.inner.strategy == Strategy::Interpreter {
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
        let meta = self.module.metadata();
        if index as usize >= meta.memories.len() {
            return None;
        }
        let offsets = &self.module.inner.offsets;
        let vmctx = self.vmctx_self_reference;
        let vm_memories = unsafe { (*vmctx).memories_mut(offsets, meta.memories.len()) };
        let mem = unsafe { &*vm_memories[index as usize] };
        Some(unsafe { core::slice::from_raw_parts(mem.base, mem.current_length) })
    }

    pub(crate) fn get_memory_mut(&mut self, index: u32) -> Option<&mut [u8]> {
        let meta = self.module.metadata();
        if index as usize >= meta.memories.len() {
            return None;
        }
        let offsets = &self.module.inner.offsets;
        let vmctx = self.vmctx_self_reference;
        let vm_memories = unsafe { (*vmctx).memories_mut(offsets, meta.memories.len()) };
        let mem = unsafe { &mut *vm_memories[index as usize] };
        Some(unsafe { core::slice::from_raw_parts_mut(mem.base, mem.current_length) })
    }

    pub(crate) fn memory_size(&self, index: u32) -> u32 {
        let meta = self.module.metadata();
        if index as usize >= meta.memories.len() {
            return 0;
        }
        let offsets = &self.module.inner.offsets;
        let vmctx = self.vmctx_self_reference;
        let vm_memories = unsafe { (*vmctx).memories_mut(offsets, meta.memories.len()) };
        let def = unsafe { &*vm_memories[index as usize] };
        (def.current_length / 65536) as u32
    }

    pub(crate) fn memory_grow(&mut self, index: u32, delta: u32) -> crate::error::Result<u32> {
        let meta = self.module.metadata();
        if index as usize >= meta.memories.len() {
            return Err(crate::error::Error::Message(
                "Memory index out of bounds".to_string(),
            ));
        }
        let engine = &self.module.inner.engine;
        let offsets = self.module.inner.offsets;
        let vmctx = self.vmctx_self_reference;

        engine
            .grow_memory(vmctx, index, meta.memories.len() as u32, delta, &offsets)
            .ok_or_else(|| crate::error::Error::Message("Memory grow failed".to_string()))
    }

    pub(crate) fn table_size(&self, index: u32) -> u32 {
        let meta = self.module.metadata();
        let offsets = &self.module.inner.offsets;
        let vmctx = self.vmctx_self_reference;
        let vm_tables = unsafe { (*vmctx).tables_mut(offsets, meta.tables.len()) };
        let table = unsafe { &*vm_tables[index as usize] };
        table.current_elements as u32
    }

    pub(crate) fn table_grow(
        &self,
        index: u32,
        init_val: *mut VMFuncRef,
        delta: u32,
    ) -> crate::error::Result<i32> {
        let meta = self.module.metadata();
        let offsets = &self.module.inner.offsets;
        let vmctx = self.vmctx_self_reference;
        let vm_tables = unsafe { (*vmctx).tables_mut(offsets, meta.tables.len()) };
        let table = unsafe { &mut *vm_tables[index as usize] };

        let old_size = table.current_elements;
        let new_size = old_size + delta as usize;

        if let Some(max) = meta.tables[index as usize].maximum {
            if new_size > max as usize {
                return Ok(-1);
            }
        }

        let mut new_vec = Vec::with_capacity(new_size);
        unsafe {
            let old_ptr = table.base;
            core::ptr::copy_nonoverlapping(old_ptr, new_vec.as_mut_ptr(), old_size);
            new_vec.set_len(old_size);
            for _ in 0..delta {
                new_vec.push(init_val);
            }

            let new_ptr = new_vec.as_mut_ptr();
            core::mem::forget(new_vec);
            table.base = new_ptr;
            table.current_elements = new_size;
        }

        Ok(old_size as i32)
    }

    pub(crate) fn table_fill(
        &self,
        index: u32,
        dst: u32,
        val: *mut VMFuncRef,
        len: u32,
    ) -> crate::error::Result<()> {
        let meta = self.module.metadata();
        let offsets = &self.module.inner.offsets;
        let vmctx = self.vmctx_self_reference;
        let vm_tables = unsafe { (*vmctx).tables_mut(offsets, meta.tables.len()) };
        let table = unsafe { &mut *vm_tables[index as usize] };

        if (dst as usize)
            .checked_add(len as usize)
            .map_or(true, |end| end > table.current_elements)
        {
            return Err(crate::error::Error::Trap(TrapCode::TableOutOfBounds));
        }

        for i in 0..len {
            unsafe {
                *table.base.add((dst + i) as usize) = val;
            }
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
        let meta = self.module.metadata();
        let offsets = &self.module.inner.offsets;
        let vmctx = self.vmctx_self_reference;
        let vm_tables = unsafe { (*vmctx).tables_mut(offsets, meta.tables.len()) };

        let dst_table = unsafe { &mut *vm_tables[dst_idx as usize] };
        let src_table = unsafe { &mut *vm_tables[src_idx as usize] };

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
        let offsets = &self.module.inner.offsets;
        let vmctx = self.vmctx_self_reference;
        let vm_tables = unsafe { (*vmctx).tables_mut(offsets, meta.tables.len()) };
        let table = unsafe { &mut *vm_tables[table_idx as usize] };

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

        let vm_functions = unsafe { (*vmctx).functions_mut(offsets, meta.functions.len()) };

        for i in 0..len {
            let ops = &elements.items[(src + i) as usize];
            let mut func_ref_ptr: *mut VMFuncRef = std::ptr::null_mut();
            for op in ops {
                match op {
                    GlobalInit::RefFunc(func_idx) => {
                        func_ref_ptr = &mut vm_functions[*func_idx as usize] as *mut VMFuncRef;
                    }
                    GlobalInit::RefNull => {
                        func_ref_ptr = std::ptr::null_mut();
                    }
                    GlobalInit::GlobalGet(global_idx) => {
                        let vm_globals =
                            unsafe { (*vmctx).globals_mut(offsets, meta.globals.len()) };
                        let val_ptr = vm_globals[*global_idx as usize] as *mut *mut VMFuncRef;
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
        self.element_lengths[elem_idx as usize] = 0;
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
        let offsets = &self.module.inner.offsets;
        let vmctx = self.vmctx_self_reference;
        let vm_memories = unsafe { (*vmctx).memories_mut(offsets, meta.memories.len()) };
        let memory = unsafe { &mut *vm_memories[mem_idx as usize] };

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
        self.data_lengths[data_idx as usize] = 0;
    }

    pub(crate) fn memory_copy(
        &self,
        dst_idx: u32,
        src_idx: u32,
        dst: u32,
        src: u32,
        len: u32,
    ) -> crate::error::Result<()> {
        let meta = self.module.metadata();
        let offsets = &self.module.inner.offsets;
        let vmctx = self.vmctx_self_reference;
        let vm_memories = unsafe { (*vmctx).memories_mut(offsets, meta.memories.len()) };

        let dst_mem = unsafe { &mut *vm_memories[dst_idx as usize] };
        let src_mem = unsafe { &mut *vm_memories[src_idx as usize] };

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
        let meta = self.module.metadata();
        let offsets = &self.module.inner.offsets;
        let vmctx = self.vmctx_self_reference;
        let vm_memories = unsafe { (*vmctx).memories_mut(offsets, meta.memories.len()) };
        let memory = unsafe { &mut *vm_memories[mem_idx as usize] };

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
        let meta = self.module.metadata();
        if index as usize >= meta.globals.len() {
            return None;
        }
        let offsets = &self.module.inner.offsets;
        let vmctx = self.vmctx_self_reference;
        let vm_globals = unsafe { (*vmctx).globals_mut(offsets, meta.globals.len()) };
        let global_ptr = vm_globals[index as usize];
        unsafe {
            let global = &*global_ptr;
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
        let meta = self.module.metadata();
        if index as usize >= meta.globals.len() {
            return Err(crate::error::Error::Message(
                "Global index out of bounds".to_string(),
            ));
        }
        let offsets = &self.module.inner.offsets;
        let vmctx = self.vmctx_self_reference;
        let vm_globals = unsafe { (*vmctx).globals_mut(offsets, meta.globals.len()) };
        let global_ptr = vm_globals[index as usize];
        unsafe {
            let global = &mut *global_ptr;
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
        }
        Ok(())
    }

    pub(crate) fn get_export(&self, name: &str) -> Option<Extern> {
        let meta = self.module.metadata();
        let offsets = &self.module.inner.offsets;
        let (kind, idx) = meta.exports.get(name)?;
        let idx = *idx as usize;

        unsafe {
            let vmctx_ptr = self.vmctx_self_reference;
            let export = match kind {
                ExternalKind::Func => {
                    let func_refs = (*vmctx_ptr).functions_mut(offsets, meta.functions.len());
                    Extern::Function(func_refs[idx])
                }
                ExternalKind::Memory => {
                    let memories = (*vmctx_ptr).memories_mut(offsets, meta.memories.len());
                    Extern::Memory(memories[idx])
                }
                ExternalKind::Table => {
                    let tables = (*vmctx_ptr).tables_mut(offsets, meta.tables.len());
                    Extern::Table(tables[idx])
                }
                ExternalKind::Global => {
                    let globals = (*vmctx_ptr).globals_mut(offsets, meta.globals.len());
                    Extern::Global(globals[idx])
                }
                _ => return None,
            };
            Some(export)
        }
    }

    pub(crate) unsafe fn init_table_segment(
        &self,
        segment_idx: u32,
        offset: u32,
    ) -> crate::error::Result<()> {
        let meta = self.module.metadata();
        let offsets = &self.module.inner.offsets;
        let vmctx = self.vmctx_self_reference;

        let element = &meta.elements[segment_idx as usize];
        unsafe {
            let vm_tables = (*vmctx).tables_mut(offsets, meta.tables.len());
            let table_ptr = vm_tables[element.table_index as usize];
            let table_def = &mut *table_ptr;

            if (offset as usize)
                .checked_add(element.items.len())
                .map_or(true, |end| end > table_def.current_elements)
            {
                return Err(crate::error::Error::Trap(TrapCode::TableOutOfBounds));
            }

            let vm_functions = (*vmctx).functions_mut(offsets, meta.functions.len());

            for (i, ops) in element.items.iter().enumerate() {
                let mut func_ref_ptr: *mut VMFuncRef = std::ptr::null_mut();
                for op in ops {
                    match op {
                        GlobalInit::RefFunc(func_idx) => {
                            func_ref_ptr = &mut vm_functions[*func_idx as usize] as *mut VMFuncRef;
                        }
                        GlobalInit::RefNull => {
                            func_ref_ptr = std::ptr::null_mut();
                        }
                        GlobalInit::GlobalGet(global_idx) => {
                            let vm_globals = (*vmctx).globals_mut(offsets, meta.globals.len());
                            let val_ptr = vm_globals[*global_idx as usize] as *mut *mut VMFuncRef;
                            func_ref_ptr = *val_ptr;
                        }
                        _ => {}
                    }
                }

                let dest = table_def.base.add((offset as usize) + i);
                *dest = func_ref_ptr;
            }
        }
        Ok(())
    }

    pub(crate) unsafe fn init_memory_segment(
        &self,
        segment_idx: u32,
        offset: u32,
    ) -> crate::error::Result<()> {
        let meta = self.module.metadata();
        let offsets = &self.module.inner.offsets;
        let vmctx = self.vmctx_self_reference;

        let data = &meta.data[segment_idx as usize];
        let data_len = data.data.len();

        unsafe {
            let vm_memories = (*vmctx).memories_mut(offsets, meta.memories.len());
            let memory_ptr = vm_memories[data.memory_index as usize];
            let memory_def = &mut *memory_ptr;

            if (offset as usize) + data_len <= memory_def.current_length {
                core::ptr::copy_nonoverlapping(
                    data.data.as_ptr(),
                    memory_def.base.add(offset as usize),
                    data_len,
                );
                Ok(())
            } else {
                Err(crate::error::Error::Trap(TrapCode::MemoryOutOfBounds))
            }
        }
    }
}

impl Instance {
    pub fn new(
        store: &mut Store,
        module: Module,
        extern_imports: &HashMap<String, ExternMap>,
    ) -> crate::error::Result<InstanceId> {
        let meta = module.metadata();
        let offsets = module.inner.offsets;

        // 1. 如果是解释器模式，记录模块 ID
        let interp_module_id = if module.inner.strategy == Strategy::Interpreter {
            match &module.inner.artifact {
                ModuleArtifact::Interpreter(ir) => Some(store.program.register_module(ir.clone())),
                _ => None,
            }
        } else {
            None
        };

        // 2. 分配并初始化 joined block
        let offset_of_vmctx = std::mem::offset_of!(Instance, vmctx);
        let total_size = (offset_of_vmctx as u32) + offsets.total_size;
        let layout = std::alloc::Layout::from_size_align(total_size as usize, 16).unwrap();
        let base_ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if base_ptr.is_null() {
            return Err(crate::error::Error::Memory("Allocation failed".to_string()));
        }

        let instance_ptr = base_ptr as *mut Instance;
        let vmctx_ptr = unsafe { base_ptr.add(offset_of_vmctx) as *mut VMContext };

        unsafe {
            let mut vm = (module.inner.strategy == Strategy::Interpreter).then(VM::new);
            if let Some(vm) = vm.as_mut() {
                vm.register_region(MemoryRegion::new(
                    0,
                    0,
                    usize::MAX,
                    "host_memory".to_string(),
                ));
            }

            let instance = Instance {
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
            // Signature Hashes
            let vm_hashes = (*vmctx_ptr).signature_hashes_mut(&offsets, meta.signatures.len());
            for i in 0..meta.signatures.len() {
                vm_hashes[i] = meta.signatures[i].hash;
            }

            // Globals
            let vm_globals = (*vmctx_ptr).globals_mut(&offsets, meta.globals.len());
            for i in 0..meta.num_imported_globals {
                let def_ptr = Self::resolve_import(
                    meta,
                    extern_imports,
                    ExternalKind::Global,
                    i,
                    |e| match e {
                        Extern::Global(ptr) => Some(*ptr),
                        _ => None,
                    },
                )?;
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
            let vm_functions = (*vmctx_ptr).functions_mut(&offsets, meta.functions.len());
            for i in 0..meta.num_imported_funcs {
                let func_ref = Self::resolve_import(
                    meta,
                    extern_imports,
                    ExternalKind::Func,
                    i,
                    |e| match e {
                        Extern::Function(f) => Some(*f),
                        _ => None,
                    },
                )?;
                vm_functions[i] = func_ref;
            }
            for i in meta.num_imported_funcs..meta.functions.len() {
                let func_name = &meta.functions[i].name;
                let ptr = if let Some(loaded) = module.loaded() {
                    let entry_sym = loaded.get::<*const ()>(func_name).ok_or_else(|| {
                        crate::error::Error::Message(format!("Symbol {} not found", func_name))
                    })?;
                    *entry_sym as *const u8
                } else {
                    let mid = interp_module_id.expect("Interpreter module not registered");
                    store
                        .program
                        .get_interpreter_func_ptr(mid, meta.functions[i].func_id)
                };

                let ty_idx = meta.functions[i].type_index;
                let sig_hash = meta.signatures[ty_idx as usize].hash_u64();

                vm_functions[i] = VMFuncRef {
                    native_call: ptr as *const core::ffi::c_void,
                    vmctx: vmctx_ptr,
                    type_index: sig_hash as u32,
                    offset: 0,
                    caller: core::ptr::null_mut(),
                };
            }

            // Memories
            let vm_memories = (*vmctx_ptr).memories_mut(&offsets, meta.memories.len());
            for i in 0..meta.num_imported_memories {
                let def_ptr = Self::resolve_import(
                    meta,
                    extern_imports,
                    ExternalKind::Memory,
                    i,
                    |e| match e {
                        Extern::Memory(ptr) => Some(*ptr),
                        _ => None,
                    },
                )?;
                vm_memories[i] = def_ptr;
            }
            for i in meta.num_imported_memories..meta.memories.len() {
                let initial = meta.memories[i].initial as u32;
                let maximum = meta.memories[i].maximum.map(|m| m as u32);
                let id = store.alloc_memory(initial, maximum)?;
                vm_memories[i] = store.get_memory(id);
            }

            // Tables
            let vm_tables = (*vmctx_ptr).tables_mut(&offsets, meta.tables.len());
            for i in 0..meta.num_imported_tables {
                let def_ptr = Self::resolve_import(
                    meta,
                    extern_imports,
                    ExternalKind::Table,
                    i,
                    |e| match e {
                        Extern::Table(ptr) => Some(*ptr),
                        _ => None,
                    },
                )?;
                vm_tables[i] = def_ptr;
            }
            for i in meta.num_imported_tables..meta.tables.len() {
                let initial = meta.tables[i].initial;
                let element_type = meta.tables[i].element_type;
                let id = store.alloc_table(initial, element_type)?;
                vm_tables[i] = store.get_table(id);

                // 如果表有默认初始化值，应用它
                if let Some(init_ops) = &meta.tables[i].init {
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

                    for j in 0..table.current_elements {
                        *table.base.add(j) = func_ref_ptr;
                    }
                }
            }
        }

        unsafe {
            // 4. 调用初始化函数 (初始化 globals, tables, memories)
            (&mut *instance_ptr).call_init(&store.program)?;
        }

        Ok(store.push_instance(InstanceHandle(instance_ptr)))
    }

    fn resolve_import<T>(
        meta: &crate::module::ModuleMetadata,
        extern_imports: &HashMap<String, ExternMap>,
        kind: ExternalKind,
        index: usize,
        resolver: impl FnOnce(&Extern) -> Option<T>,
    ) -> crate::error::Result<T> {
        let import = meta
            .imports
            .iter()
            .filter(|imp| imp.kind == kind)
            .nth(index)
            .unwrap();
        let resolved = extern_imports
            .get(&import.module)
            .and_then(|m| m.get(&import.field));

        resolved.and_then(resolver).ok_or_else(|| {
            crate::error::Error::Message(format!(
                "{:?} import not found: {}.{}",
                kind, import.module, import.field
            ))
        })
    }
}

impl InstanceId {
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

            let tramp_ptr = if let Some(loaded) = instance.module.loaded() {
                let tramp_name = format!("{}_trampoline", name);
                unsafe {
                    loaded
                        .get::<*const ()>(&tramp_name)
                        .map(|s| *s as *const u8)
                }
            } else {
                None
            };

            Some(TypedFunc {
                params: sig.params.clone(),
                results: sig.results.clone(),
                instance_id: self,
                func_ref,
                trampoline_ptr: tramp_ptr,
                func_id,
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

/// 类型化函数句柄
/// 可用于从宿主代码直接调用 Wasm 函数
pub struct TypedFunc {
    params: Box<[ValType]>,
    results: Box<[ValType]>,
    instance_id: InstanceId,
    func_ref: VMFuncRef,
    trampoline_ptr: Option<*const u8>,
    func_id: FuncId,
}

impl TypedFunc {
    pub fn call(&self, store: &mut Store, args: &[Val]) -> crate::error::Result<Vec<Val>> {
        let instance = store.get_instance(self.instance_id);

        if instance.module.inner.strategy == Strategy::Interpreter {
            return self.call_interpreter(store, args);
        }

        let vmctx_ptr = self.func_ref.vmctx;

        unsafe {
            let instance = Instance::from_vmctx(vmctx_ptr);
            let offsets = &instance.module.inner.offsets;
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

            let mut results_raw = vec![0i64; self.results.len()];
            let mut storage = [0i64; 17]; // max 16 args + potential results ptr
            let mut actual_len = args.len();

            if self.results.len() > 1 {
                storage[actual_len] = results_raw.as_mut_ptr() as i64;
                actual_len += 1;
            }

            if actual_len > 16 {
                return Err(crate::error::Error::Message(format!(
                    "Too many arguments: {}",
                    actual_len
                )));
            }

            for (i, arg) in args.iter().enumerate() {
                // Type check (keep for safety, can be optimized out if trusted)
                let expected = self.params[i];
                let actual = match arg {
                    Val::I32(_) => ValType::I32,
                    Val::I64(_) => ValType::I64,
                    Val::F32(_) => ValType::F32,
                    Val::F64(_) => ValType::F64,
                };
                if actual != expected {
                    return Err(crate::error::Error::Message(format!(
                        "Argument {} type mismatch: expected {:?}, got {:?}",
                        i, expected, actual
                    )));
                }
                storage[i] = arg.as_i64();
            }

            let tramp_ptr = self.trampoline_ptr.ok_or_else(|| {
                crate::error::Error::Message("JIT trampoline not found".to_string())
            })?;

            let trampoline: extern "C" fn(*const VMContext, *const i64) -> i64 =
                mem::transmute(tramp_ptr);
            let res = trampoline(vmctx_ptr, storage.as_ptr());

            let mut results_val = Vec::with_capacity(self.results.len());
            if self.results.len() == 1 {
                results_val.push(Val::from_i64(res, self.results[0]));
            } else if self.results.len() > 1 {
                for (i, &ty) in self.results.iter().enumerate() {
                    results_val.push(Val::from_i64(results_raw[i], ty));
                }
            }

            Ok(results_val)
        }
    }

    pub fn call_interpreter(
        &self,
        store: &mut Store,
        args: &[Val],
    ) -> crate::error::Result<Vec<Val>> {
        let vmctx_ptr = self.func_ref.vmctx;

        unsafe {
            let instance = Instance::from_vmctx(vmctx_ptr);
            let offsets = &instance.module.inner.offsets;
            let jmp_buf_ptr = (*vmctx_ptr).jmp_buf_ptr(offsets);

            let trap_val = __sigsetjmp(jmp_buf_ptr, 0);
            if trap_val != 0 {
                return Err(instance.handle_trap(trap_val as u32));
            }

            let interpreter = instance.interpreter.as_mut().ok_or_else(|| {
                crate::error::Error::Message("Interpreter not initialized".to_string())
            })?;

            let mut int_args = Vec::with_capacity(args.len() + 1);
            int_args.push(InterpreterValue::I64(vmctx_ptr as i64));

            for &arg in args {
                int_args.push(match arg {
                    Val::I32(v) => InterpreterValue::I32(v),
                    Val::I64(v) => InterpreterValue::I64(v),
                    Val::F32(v) => InterpreterValue::F32(v),
                    Val::F64(v) => InterpreterValue::F64(v),
                });
            }

            let mut results_raw = vec![0i64; self.results.len()];
            if self.results.len() > 1 {
                int_args.push(InterpreterValue::I64(results_raw.as_mut_ptr() as i64));
            }

            let vm = instance
                .vm
                .as_mut()
                .ok_or_else(|| crate::error::Error::Message("VM not initialized".to_string()))?;

            let res = interpreter.run_function(&store.program, vm, self.func_id, &int_args);

            let mut results_val = Vec::with_capacity(self.results.len());
            if self.results.len() == 1 {
                results_val.push(Val::from_i64(res.to_i64_bits(), self.results[0]));
            } else if self.results.len() > 1 {
                for (i, &ty) in self.results.iter().enumerate() {
                    results_val.push(Val::from_i64(results_raw[i], ty));
                }
            }

            Ok(results_val)
        }
    }
}
