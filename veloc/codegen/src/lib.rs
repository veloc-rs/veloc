// Shared Spec generators emit alloc paths even in std-only consumers.
extern crate alloc;

pub mod backend;
pub mod driver;
mod emitter;
pub mod error;
pub use emitter::{EmittedCode, Emitter, ExternalRelocation, Target as FixupTarget};
// Exported macros use the dependency's canonical name for hygienic paths.
#[doc(hidden)]
pub use veloc_lir;
pub mod object;
pub mod passes;
pub mod pipeline;
pub mod regalloc;
pub mod target;
pub mod translate;

pub mod analysis;
pub mod isel;
pub mod verify;

pub use backend::Backend;
pub use driver::{CodegenOptions, CodegenPipeline, CodegenStats};
pub use target::{
    CallConv, RewriteResult, SelectResult, TargetArch, TargetConfig, TargetEmitter,
    TargetFrameLowering, TargetInstructionSelector, TargetLegalizer, TargetMachine,
    TargetOperandLowering, TargetPassConfig, TargetPostIsel,
};

/// 根据目标配置创建对应的目标机器
pub fn create_target_machine(config: TargetConfig) -> Result<std::boxed::Box<dyn TargetMachine>> {
    use target::TargetArch;
    match config.arch {
        TargetArch::X86_64 => Ok(std::boxed::Box::new(
            target::x86_64::X86_64TargetMachine::new(config)?,
        )),
        _ => Err(Error::target_machine_unavailable(config.arch)),
    }
}

pub use error::{Error, Result};

pub use std::format;
pub use std::string::String;
pub use std::vec::Vec;
pub use veloc_lir::SymbolId;
