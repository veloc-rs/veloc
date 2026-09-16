#![no_std]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod backend;
pub mod driver;
mod emitter;
pub mod error;
pub use emitter::{EmittedCode, Emitter, ExternalRelocation, Target as FixupTarget};
pub mod isle {
    pub use crate::target::x86_64::isle::*;
}
// Exported macros use the dependency's canonical name for hygienic paths.
#[doc(hidden)]
pub use veloc_lir;
pub mod object;
pub mod passes;
pub mod pipeline;
pub mod regalloc;
pub mod target;
pub mod translate;

pub use crate::passes::isel;

pub use backend::Backend;
pub use driver::{CodegenOptions, CodegenPipeline, CodegenStats};
pub use target::arch::{
    CallConv, RewriteResult, SelectResult, TargetArch, TargetConfig, TargetEmitter,
    TargetFrameLowering, TargetInstructionSelector, TargetLegalizer, TargetMachine,
    TargetOperandLowering, TargetPassConfig, TargetPostIsel,
};

/// 根据目标配置创建对应的目标机器
pub fn create_target_machine(config: TargetConfig) -> Option<alloc::boxed::Box<dyn TargetMachine>> {
    use target::arch::TargetArch;
    match config.arch {
        TargetArch::X86_64 => Some(alloc::boxed::Box::new(
            target::x86_64::X86_64TargetMachine::new(config),
        )),
        _ => None,
    }
}

pub use error::{Error, Result};

pub use alloc::format;
pub use alloc::string::String;
pub use alloc::vec::Vec;
pub use veloc_lir::SymbolId;
