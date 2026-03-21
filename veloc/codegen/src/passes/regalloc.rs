use crate::error::Result;
use crate::pipeline::stages::{PostIselOptimized, RegAllocated};
use crate::pipeline::{ChangeSet, FunctionPassContext, PassEffect, StageTransformPass};
use crate::regalloc::RegisterAllocator;

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
        mut mfunc: crate::mir::MachineFunction<PostIselOptimized>,
        ctx: &mut FunctionPassContext<'_, PostIselOptimized>,
    ) -> Result<(crate::mir::MachineFunction<RegAllocated>, PassEffect)> {
        RegisterAllocator::new(self.target)
            .allocate(mfunc.as_untyped_mut(), ctx.func_sig.call_conv);
        ctx.stats.final_inst_count = mfunc.blocks.iter().map(|b| b.insts.len()).sum();
        ctx.stats.stack_slot_count = mfunc.stack_frame.slots.len();
        Ok((
            mfunc.into_stage(),
            PassEffect::new(
                ChangeSet::REGALLOC | ChangeSet::PHYSICAL_REGS | ChangeSet::INST_OPERANDS,
            ),
        ))
    }
}
