use crate::error::Result;
use crate::isel::InstructionSelector;
use crate::isel::generic_egraph::run_generic_pre_isel_egraph_combine;
use crate::pipeline::stages::{BankSelected, PostIselOptimized, PreIselPrepared, SelectedMir};
use crate::pipeline::{ChangeSet, FunctionPassContext, PassEffect, StageTransformPass};
use crate::target::arch::OperandConstraintStage;

pub struct PreIselPreparePass<'a> {
    lowering: &'a dyn crate::target::arch::TargetLowering,
}

impl<'a> PreIselPreparePass<'a> {
    pub fn new(lowering: &'a dyn crate::target::arch::TargetLowering) -> Self {
        Self { lowering }
    }
}

impl<'a> StageTransformPass<BankSelected, PreIselPrepared> for PreIselPreparePass<'a> {
    fn name(&self) -> &'static str {
        "pre-isel-prepared"
    }

    fn run(
        &self,
        mut mfunc: crate::mir::MachineFunction<BankSelected>,
        ctx: &mut FunctionPassContext<'_, BankSelected>,
    ) -> Result<(crate::mir::MachineFunction<PreIselPrepared>, PassEffect)> {
        run_generic_pre_isel_egraph_combine(mfunc.as_untyped_mut(), ctx.function_analyses);
        crate::passes::OperandConstraintPass::new(self.lowering, OperandConstraintStage::PreSelect)
            .run(&mut mfunc)?;
        Ok((
            mfunc.into_stage(),
            PassEffect::new(ChangeSet::INST_SEMANTICS | ChangeSet::INST_OPERANDS),
        ))
    }
}

pub struct InstructionSelectionPass<'a> {
    lowering: &'a dyn crate::target::arch::TargetLowering,
}

impl<'a> InstructionSelectionPass<'a> {
    pub fn new(lowering: &'a dyn crate::target::arch::TargetLowering) -> Self {
        Self { lowering }
    }
}

impl<'a> StageTransformPass<PreIselPrepared, SelectedMir> for InstructionSelectionPass<'a> {
    fn name(&self) -> &'static str {
        "selected"
    }

    fn run(
        &self,
        mut mfunc: crate::mir::MachineFunction<PreIselPrepared>,
        ctx: &mut FunctionPassContext<'_, PreIselPrepared>,
    ) -> Result<(crate::mir::MachineFunction<SelectedMir>, PassEffect)> {
        InstructionSelector::new(self.lowering).select(mfunc.as_untyped_mut())?;
        ctx.stats.selected_inst_count = mfunc.blocks.iter().map(|b| b.insts.len()).sum();
        Ok((
            mfunc.into_stage(),
            PassEffect::new(
                ChangeSet::SELECTED_OPCODES | ChangeSet::INST_SEMANTICS | ChangeSet::INST_OPERANDS,
            ),
        ))
    }
}

pub struct PostIselOptimizePass<'a> {
    lowering: &'a dyn crate::target::arch::TargetLowering,
}

impl<'a> PostIselOptimizePass<'a> {
    pub fn new(lowering: &'a dyn crate::target::arch::TargetLowering) -> Self {
        Self { lowering }
    }
}

impl<'a> StageTransformPass<SelectedMir, PostIselOptimized> for PostIselOptimizePass<'a> {
    fn name(&self) -> &'static str {
        "post-isel-optimized"
    }

    fn run(
        &self,
        mut mfunc: crate::mir::MachineFunction<SelectedMir>,
        ctx: &mut FunctionPassContext<'_, SelectedMir>,
    ) -> Result<(crate::mir::MachineFunction<PostIselOptimized>, PassEffect)> {
        self.lowering.combine_instructions(mfunc.as_untyped_mut());
        ctx.stats.combined_inst_count = mfunc.blocks.iter().map(|b| b.insts.len()).sum();
        crate::passes::OperandConstraintPass::new(
            self.lowering,
            OperandConstraintStage::PostSelect,
        )
        .run(&mut mfunc)?;
        Ok((
            mfunc.into_stage(),
            PassEffect::new(ChangeSet::INST_SEMANTICS | ChangeSet::INST_OPERANDS),
        ))
    }
}
