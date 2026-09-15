#![feature(explicit_tail_calls)]
#![feature(macro_metavar_expr)]
#![feature(rust_preserve_none_cc)]
#![expect(incomplete_features)]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

#[macro_use]
pub mod bytecode;
pub mod error;
pub mod host;
pub mod interpreter;
pub mod runtime;
pub mod value;

pub use bytecode::printer::FuncPrinter;
pub use error::{Error, Result};
pub use host::{HostFuncId, HostFunction};
pub use interpreter::{Interpreter, VirtualMemory};
pub use runtime::{CallTarget, FunctionRef, Program, ProgramBuilder};
pub use value::InterpreterValue;
pub use veloc_mir::ModuleId;

/// Native byte-addressed memory representation implemented by this interpreter.
pub const DATA_LAYOUT: veloc_types::DataLayout = {
    use veloc_types::{Type, TypeLayout};
    const PTR_BYTES: u32 = core::mem::size_of::<usize>() as u32;
    veloc_types::DataLayout {
        pointer_size: PTR_BYTES as u8,
        little_endian: cfg!(target_endian = "little"),
        types: &[
            (Type::BOOL, TypeLayout::fixed(1, 1)),
            (Type::I8, TypeLayout::fixed(1, 1)),
            (Type::I16, TypeLayout::fixed(2, 2)),
            (Type::I32, TypeLayout::fixed(4, 4)),
            (Type::I64, TypeLayout::fixed(8, 8)),
            (Type::F32, TypeLayout::fixed(4, 4)),
            (Type::F64, TypeLayout::fixed(8, 8)),
            (Type::PTR, TypeLayout::fixed(PTR_BYTES, PTR_BYTES)),
        ],
    }
};
