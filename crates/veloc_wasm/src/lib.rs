extern crate alloc;

pub mod engine;
pub mod error;
pub mod func;
pub mod instance;
pub mod linker;
pub mod module;
pub mod store;
pub mod translator;
pub mod vm;
pub mod wasi;

pub use crate::engine::Engine;
pub use crate::func::{Caller, IntoFunc};
pub use crate::instance::TypedFunc;
pub use crate::linker::Linker;
pub use crate::module::Module;
pub use crate::store::{Global, Instance, Memory, Store, Table};

use crate::vm::{VMFuncRef, VMGlobal, VMMemory, VMTable};

#[derive(Clone, Copy)]
pub enum Extern {
    Function(VMFuncRef),
    Table(*mut VMTable),
    Memory(*mut VMMemory),
    Global(*mut VMGlobal),
}

impl core::fmt::Debug for Extern {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Extern::Function(fr) => f.debug_tuple("Function").field(fr).finish(),
            Extern::Table(t) => f.debug_tuple("Table").field(t).finish(),
            Extern::Memory(m) => f.debug_tuple("Memory").field(m).finish(),
            Extern::Global(g) => f.debug_tuple("Global").field(g).finish(),
        }
    }
}

impl From<VMFuncRef> for Extern {
    fn from(f: VMFuncRef) -> Self {
        Extern::Function(f)
    }
}

impl From<*mut VMMemory> for Extern {
    fn from(m: *mut VMMemory) -> Self {
        Extern::Memory(m)
    }
}

impl From<*mut VMTable> for Extern {
    fn from(t: *mut VMTable) -> Self {
        Extern::Table(t)
    }
}

impl From<*mut VMGlobal> for Extern {
    fn from(g: *mut VMGlobal) -> Self {
        Extern::Global(g)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Val {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Externref(u32),
    Funcref(Option<crate::vm::VMFuncRef>),
}

impl Val {
    pub fn as_i64(&self) -> i64 {
        match *self {
            Val::I32(v) => v as i64,
            Val::I64(v) => v,
            Val::F32(v) => v.to_bits() as i64,
            Val::F64(v) => v.to_bits() as i64,
            Val::Externref(v) => v as i64,
            Val::Funcref(v) => match v {
                Some(f) => &f as *const _ as i64, // This is not quite right but used for return mapping
                None => 0,
            },
        }
    }

    pub fn from_i64(v: i64, ty: wasmparser::ValType) -> Self {
        match ty {
            wasmparser::ValType::I32 => Val::I32(v as i32),
            wasmparser::ValType::I64 => Val::I64(v),
            wasmparser::ValType::F32 => Val::F32(f32::from_bits(v as u32)),
            wasmparser::ValType::F64 => Val::F64(f64::from_bits(v as u64)),
            wasmparser::ValType::Ref(rt) => {
                if rt.is_extern_ref() {
                    if v == 0 {
                        Val::Externref(0)
                    } else {
                        Val::Externref(v as u32)
                    }
                } else {
                    if v == 0 {
                        Val::Funcref(None)
                    } else {
                        unsafe { Val::Funcref(Some(*(v as *const crate::vm::VMFuncRef))) }
                    }
                }
            }
            _ => panic!("Unsupported type: {:?}", ty),
        }
    }

    pub fn ty(&self) -> wasmparser::ValType {
        match *self {
            Val::I32(_) => wasmparser::ValType::I32,
            Val::I64(_) => wasmparser::ValType::I64,
            Val::F32(_) => wasmparser::ValType::F32,
            Val::F64(_) => wasmparser::ValType::F64,
            Val::Externref(_) => wasmparser::ValType::EXTERNREF,
            Val::Funcref(_) => wasmparser::ValType::FUNCREF,
        }
    }

    pub fn to_interpreter_val(&self) -> veloc::interpreter::InterpreterValue {
        match *self {
            Val::I32(v) => veloc::interpreter::InterpreterValue::i32(v),
            Val::I64(v) => veloc::interpreter::InterpreterValue::i64(v),
            Val::F32(v) => veloc::interpreter::InterpreterValue::f32(v),
            Val::F64(v) => veloc::interpreter::InterpreterValue::f64(v),
            Val::Externref(v) => veloc::interpreter::InterpreterValue::i64(v as i64),
            Val::Funcref(v) => match v {
                Some(f) => veloc::interpreter::InterpreterValue::i64(&f as *const _ as i64),
                None => veloc::interpreter::InterpreterValue::i64(0),
            },
        }
    }
}

// 重新导出常用 API，模仿 Wasmtime 的组织方式
pub use error::{Error, Result};
pub use vm::VMContext;
