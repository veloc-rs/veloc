use crate::instance::InstanceHandle;
use crate::vm::{VMGlobal, VMMemory, VMTable};
use alloc::sync::Arc;
use cranelift_entity::{PrimaryMap, entity_impl};
use std::sync::Mutex;
use veloc::interpreter::{InterpreterValue, Program, host::HostFunction};
use wasi_common::WasiCtx;
use wasmparser::ValType;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Memory(u32);
entity_impl!(Memory, "mem");

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Table(u32);
entity_impl!(Table, "table");

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Global(u32);
entity_impl!(Global, "global");

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FuncRef(u32);
entity_impl!(FuncRef, "funcref");

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Instance(u32);
entity_impl!(Instance, "instance");

/// Store 保存运行时的资源状态
pub struct Store {
    memories: PrimaryMap<Memory, Box<VMMemory>>,
    tables: PrimaryMap<Table, Box<VMTable>>,
    globals: PrimaryMap<Global, Box<VMGlobal>>,
    instances: PrimaryMap<Instance, InstanceHandle>,
    pub(crate) program: veloc::interpreter::Program,
    pub(crate) wasi_ctx: Option<Arc<Mutex<WasiCtx>>>,
}

impl Store {
    pub fn new() -> Self {
        let mut program = Program::new();
        register_builtins(&mut program);
        Self {
            memories: PrimaryMap::new(),
            tables: PrimaryMap::new(),
            globals: PrimaryMap::new(),
            instances: PrimaryMap::new(),
            program,
            wasi_ctx: None,
        }
    }

    pub fn set_wasi(&mut self, wasi: WasiCtx) {
        self.wasi_ctx = Some(Arc::new(Mutex::new(wasi)));
    }

    pub fn alloc_memory(
        &mut self,
        initial: u32,
        maximum: Option<u32>,
    ) -> crate::error::Result<Memory> {
        let definition = Box::new(VMMemory::new(initial, maximum)?);
        Ok(self.memories.push(definition))
    }

    pub fn get_memory(&self, id: Memory) -> *mut VMMemory {
        self.memories[id].as_ref() as *const _ as *mut _
    }

    pub fn alloc_table(
        &mut self,
        initial: u32,
        maximum: Option<u32>,
        element_type: wasmparser::RefType,
    ) -> crate::error::Result<Table> {
        let definition = Box::new(VMTable::new(initial, maximum, element_type));
        Ok(self.tables.push(definition))
    }

    pub fn get_table(&self, id: Table) -> *mut VMTable {
        self.tables[id].as_ref() as *const _ as *mut _
    }

    pub fn alloc_global(&mut self, val: crate::Val, mutable: bool) -> Global {
        let ty = val.ty();
        let mut definition = Box::new(VMGlobal::new(ty, mutable));
        match val {
            crate::Val::I32(v) => {
                definition.value.i32 = v;
            }
            crate::Val::I64(v) => {
                definition.value.i64 = v;
            }
            crate::Val::F32(v) => {
                definition.value.f32 = v;
            }
            crate::Val::F64(v) => {
                definition.value.f64 = v;
            }
            crate::Val::Externref(v) => {
                definition.value.i64 = v as i64;
            }
            crate::Val::Funcref(_v) => {
                // This is a bit tricky, we need a stable pointer.
                // For now, let's just use 0 if we can't get one.
                definition.value.i64 = 0;
            }
        };

        self.globals.push(definition)
    }

    pub fn get_global(&self, id: Global) -> *mut VMGlobal {
        self.globals[id].as_ref() as *const _ as *mut _
    }

    pub(crate) fn push_instance(&mut self, handle: InstanceHandle) -> Instance {
        self.instances.push(handle)
    }

    pub(crate) fn get_instance(&self, id: Instance) -> &InstanceHandle {
        &self.instances[id]
    }

    pub(crate) fn get_instance_mut(&mut self, id: Instance) -> &mut InstanceHandle {
        &mut self.instances[id]
    }

    /// 获取全局变量的当前值
    pub fn global_get(&self, id: Global) -> crate::Val {
        let definition = &self.globals[id];
        match definition.ty {
            ValType::I32 => unsafe { crate::Val::I32(definition.value.i32) },
            ValType::I64 => unsafe { crate::Val::I64(definition.value.i64) },
            ValType::F32 => unsafe { crate::Val::F32(definition.value.f32) },
            ValType::F64 => unsafe { crate::Val::F64(definition.value.f64) },
            _ => panic!("Unsupported global type: {:?}", definition.ty),
        }
    }

    /// 设置全局变量的值
    pub fn global_set(&mut self, id: Global, val: crate::Val) -> crate::error::Result<()> {
        let definition = &mut self.globals[id];
        if !definition.mutable {
            return Err(crate::error::Error::Message(
                "Cannot set immutable global".to_string(),
            ));
        }
        // 类型检查
        match (definition.ty, val) {
            (ValType::I32, crate::Val::I32(v)) => definition.value.i32 = v,
            (ValType::I64, crate::Val::I64(v)) => definition.value.i64 = v,
            (ValType::F32, crate::Val::F32(v)) => definition.value.f32 = v,
            (ValType::F64, crate::Val::F64(v)) => definition.value.f64 = v,
            _ => {
                return Err(crate::error::Error::Message(format!(
                    "Type mismatch setting global: expected {:?}, got {:?}",
                    definition.ty, val
                )));
            }
        }
        Ok(())
    }
}

pub(crate) fn register_builtins(program: &mut Program) {
    // 注册默认 Wasm 宿主函数
    program.register_raw(
        "wasm_trap_handler".to_string(),
        Arc::new(|args: &[InterpreterValue]| {
            let vmctx_ptr = args[0].unwarp_i64() as *mut crate::vm::VMContext;
            let code = if args.len() > 1 {
                args[1].unwarp_i32()
            } else {
                0
            };
            crate::module::wasm_trap_handler(vmctx_ptr, code as u32);
            #[allow(unreachable_code)]
            InterpreterValue::none()
        }) as HostFunction,
    );
    program.register_raw(
        "wasm_memory_size".to_string(),
        Arc::new(|args: &[InterpreterValue]| {
            let vmctx_ptr = args[0].unwarp_i64() as *mut crate::vm::VMContext;
            let mem_idx = args[1].unwarp_i32() as u32;
            let res = crate::module::wasm_memory_size(vmctx_ptr, mem_idx);
            InterpreterValue::i32(res as i32)
        }) as HostFunction,
    );
    program.register_raw(
        "wasm_memory_grow".to_string(),
        Arc::new(|args: &[InterpreterValue]| {
            let vmctx_ptr = args[0].unwarp_i64() as *mut crate::vm::VMContext;
            let mem_idx = args[1].unwarp_i32() as u32;
            let delta = args[2].unwarp_i32() as u32;
            let res = crate::module::wasm_memory_grow(vmctx_ptr, mem_idx, delta);
            InterpreterValue::i32(res as i32)
        }) as HostFunction,
    );
    program.register_raw(
        "wasm_table_size".to_string(),
        Arc::new(|args: &[InterpreterValue]| {
            let vmctx_ptr = args[0].unwarp_i64() as *mut crate::vm::VMContext;
            let table_idx = args[1].unwarp_i32() as u32;
            let res = crate::module::wasm_table_size(vmctx_ptr, table_idx);
            InterpreterValue::i32(res as i32)
        }) as HostFunction,
    );
    program.register_raw(
        "wasm_table_grow".to_string(),
        Arc::new(|args: &[InterpreterValue]| {
            let vmctx_ptr = args[0].unwarp_i64() as *mut crate::vm::VMContext;
            let table_idx = args[1].unwarp_i32() as u32;
            let init_val = args[2].unwarp_i64() as *mut crate::vm::VMFuncRef;
            let delta = args[3].unwarp_i32() as u32;
            let res = crate::module::wasm_table_grow(vmctx_ptr, table_idx, init_val, delta);
            InterpreterValue::i32(res)
        }) as HostFunction,
    );
    program.register_raw(
        "wasm_table_fill".to_string(),
        Arc::new(|args: &[InterpreterValue]| {
            let vmctx_ptr = args[0].unwarp_i64() as *mut crate::vm::VMContext;
            let table_idx = args[1].unwarp_i32() as u32;
            let dst = args[2].unwarp_i32() as u32;
            let val = args[3].unwarp_i64() as *mut crate::vm::VMFuncRef;
            let len = args[4].unwarp_i32() as u32;
            crate::module::wasm_table_fill(vmctx_ptr, table_idx, dst, val, len);
            InterpreterValue::none()
        }) as HostFunction,
    );
    program.register_raw(
        "wasm_table_copy".to_string(),
        Arc::new(|args: &[InterpreterValue]| {
            let vmctx_ptr = args[0].unwarp_i64() as *mut crate::vm::VMContext;
            let dst_idx = args[1].unwarp_i32() as u32;
            let src_idx = args[2].unwarp_i32() as u32;
            let dst = args[3].unwarp_i32() as u32;
            let src = args[4].unwarp_i32() as u32;
            let len = args[5].unwarp_i32() as u32;
            crate::module::wasm_table_copy(vmctx_ptr, dst_idx, src_idx, dst, src, len);
            InterpreterValue::none()
        }) as HostFunction,
    );
    program.register_raw(
        "wasm_table_init".to_string(),
        Arc::new(|args: &[InterpreterValue]| {
            let vmctx_ptr = args[0].unwarp_i64() as *mut crate::vm::VMContext;
            let table_idx = args[1].unwarp_i32() as u32;
            let elem_idx = args[2].unwarp_i32() as u32;
            let dst = args[3].unwarp_i32() as u32;
            let src = args[4].unwarp_i32() as u32;
            let len = args[5].unwarp_i32() as u32;
            crate::module::wasm_table_init(vmctx_ptr, table_idx, elem_idx, dst, src, len);
            InterpreterValue::none()
        }) as HostFunction,
    );
    program.register_raw(
        "wasm_elem_drop".to_string(),
        Arc::new(|args: &[InterpreterValue]| {
            let vmctx_ptr = args[0].unwarp_i64() as *mut crate::vm::VMContext;
            let elem_idx = args[1].unwarp_i32() as u32;
            crate::module::wasm_elem_drop(vmctx_ptr, elem_idx);
            InterpreterValue::none()
        }) as HostFunction,
    );
    program.register_raw(
        "wasm_memory_init".to_string(),
        Arc::new(|args: &[InterpreterValue]| {
            let vmctx_ptr = args[0].unwarp_i64() as *mut crate::vm::VMContext;
            let mem_idx = args[1].unwarp_i32() as u32;
            let data_idx = args[2].unwarp_i32() as u32;
            let dst = args[3].unwarp_i32() as u32;
            let src = args[4].unwarp_i32() as u32;
            let len = args[5].unwarp_i32() as u32;
            crate::module::wasm_memory_init(vmctx_ptr, mem_idx, data_idx, dst, src, len);
            InterpreterValue::none()
        }) as HostFunction,
    );
    program.register_raw(
        "wasm_data_drop".to_string(),
        Arc::new(|args: &[InterpreterValue]| {
            let vmctx_ptr = args[0].unwarp_i64() as *mut crate::vm::VMContext;
            let data_idx = args[1].unwarp_i32() as u32;
            crate::module::wasm_data_drop(vmctx_ptr, data_idx);
            InterpreterValue::none()
        }) as HostFunction,
    );
    program.register_raw(
        "wasm_memory_copy".to_string(),
        Arc::new(|args: &[InterpreterValue]| {
            let vmctx_ptr = args[0].unwarp_i64() as *mut crate::vm::VMContext;
            let dst_idx = args[1].unwarp_i32() as u32;
            let src_idx = args[2].unwarp_i32() as u32;
            let dst = args[3].unwarp_i32() as u32;
            let src = args[4].unwarp_i32() as u32;
            let len = args[5].unwarp_i32() as u32;
            crate::module::wasm_memory_copy(vmctx_ptr, dst_idx, src_idx, dst, src, len);
            InterpreterValue::none()
        }) as HostFunction,
    );
    program.register_raw(
        "wasm_memory_fill".to_string(),
        Arc::new(|args: &[InterpreterValue]| {
            let vmctx_ptr = args[0].unwarp_i64() as *mut crate::vm::VMContext;
            let mem_idx = args[1].unwarp_i32() as u32;
            let dst = args[2].unwarp_i32() as u32;
            let val = args[3].unwarp_i32() as u32;
            let len = args[4].unwarp_i32() as u32;
            crate::module::wasm_memory_fill(vmctx_ptr, mem_idx, dst, val, len);
            InterpreterValue::none()
        }) as HostFunction,
    );

    program.register_raw(
        "wasm_init_table_element".to_string(),
        Arc::new(|args: &[InterpreterValue]| {
            let vmctx_ptr = args[0].unwarp_i64() as *mut crate::vm::VMContext;
            let element_idx = args[1].unwarp_i32() as u32;
            let offset = args[2].unwarp_i32() as u32;
            unsafe {
                crate::module::wasm_init_table_element(vmctx_ptr, element_idx, offset);
            }
            InterpreterValue::none()
        }) as HostFunction,
    );
    program.register_raw(
        "wasm_init_memory_data".to_string(),
        Arc::new(|args: &[InterpreterValue]| {
            let vmctx_ptr = args[0].unwarp_i64() as *mut crate::vm::VMContext;
            let data_idx = args[1].unwarp_i32() as u32;
            let offset = args[2].unwarp_i32() as u32;
            unsafe {
                crate::module::wasm_init_memory_data(vmctx_ptr, data_idx, offset);
            }
            InterpreterValue::none()
        }) as HostFunction,
    );
    program.register_raw(
        "wasm_init_table".to_string(),
        Arc::new(|args: &[InterpreterValue]| {
            let vmctx_ptr = args[0].unwarp_i64() as *mut crate::vm::VMContext;
            let table_idx = args[1].unwarp_i32() as u32;
            let val = args[2].unwarp_i64() as *mut crate::vm::VMFuncRef;
            unsafe {
                crate::module::wasm_init_table(vmctx_ptr, table_idx, val);
            }
            InterpreterValue::none()
        }) as HostFunction,
    );
}
