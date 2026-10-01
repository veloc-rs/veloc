use super::{FunctionSession, ModulePassContext};
use crate::error::Result;

/// Representation invariants at function-pass boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionStage {
    Generic,
    Legal,
    Selected,
    Allocated,
    Framed,
}
impl FunctionStage {
    pub(crate) fn verify(
        self,
        function: &veloc_lir::MachineFunction,
        target: &dyn crate::TargetMachine,
    ) -> Result<()> {
        use crate::verify::{verify, verify_allocated, verify_selected};
        match self {
            Self::Generic => verify(function, target),
            Self::Legal => {
                verify(function, target)?;
                crate::passes::lowering::Legalizer::new(target.legalizer()).verify(function)
            }
            Self::Selected => verify_selected(function, target),
            Self::Allocated => verify_allocated(function, target),
            Self::Framed => {
                verify_allocated(function, target)?;
                if function.stack_frame.layout().is_none() {
                    return Err(crate::Error::codegen("framed function has no stack layout"));
                }
                Ok(())
            }
        }
    }
}

/// Immutable pass configuration; scratch state belongs to each run.
pub trait FunctionPass {
    fn name(&self) -> &'static str;
    fn input_stage(&self) -> FunctionStage;
    fn output_stage(&self) -> FunctionStage {
        self.input_stage()
    }
    fn run(&self, session: &mut FunctionSession<'_>) -> Result<()>;
}

/// M is the representation this pass may modify. In particular, an emission
/// pass has no MachineFunction to mutate behind already-generated fragments.
pub trait ModuleCodegenPass<M> {
    fn name(&self) -> &'static str;
    fn run(&self, module: &mut M, ctx: &mut ModulePassContext<'_>) -> Result<()>;
}
