//! Register allocation stage of the code generation pipeline.
use crate::error::Result;
use crate::pipeline::{ChangeSet, FunctionPassContext, PassEffect, StageTransformPass};
use crate::regalloc::RegisterAllocator;
use veloc_lir::stages::{PostIselOptimized, RegAllocated};

pub struct RegisterAllocationPass<'a> {
    target: &'a dyn crate::target::arch::TargetMachine,
}

impl<'a> RegisterAllocationPass<'a> {
    pub fn new(target: &'a dyn crate::target::arch::TargetMachine) -> Self {
        Self { target }
    }
}

impl<'a> StageTransformPass<PostIselOptimized, RegAllocated> for RegisterAllocationPass<'a> {
    fn name(&self) -> &'static str {
        "regalloc"
    }

    fn run(
        &self,
        mfunc: veloc_lir::MachineFunction<PostIselOptimized>,
        ctx: &mut FunctionPassContext<'_, PostIselOptimized>,
    ) -> Result<(veloc_lir::MachineFunction<RegAllocated>, PassEffect)> {
        let allocation = RegisterAllocator::new(self.target).allocate(
            mfunc,
            ctx.func_sig.call_conv,
            ctx.function_analyses,
        )?;
        let mfunc = allocation.materialize();
        ctx.stats.final_inst_count = mfunc.blocks.iter().map(|b| b.insts.len()).sum();
        ctx.stats.stack_slot_count = mfunc.stack_frame.slots.len();
        Ok((
            mfunc,
            PassEffect::new(
                ChangeSet::REGALLOC
                    | ChangeSet::PHYSICAL_REGS
                    | ChangeSet::INST_OPERANDS
                    | ChangeSet::INST_SEMANTICS
                    | ChangeSet::CFG
                    | ChangeSet::STACK_FRAME,
            ),
        ))
    }
}
