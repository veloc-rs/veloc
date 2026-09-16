use crate::error::Result;
use crate::pipeline::{ChangeSet, FunctionPass, FunctionPassContext, PassEffect};
use crate::target::arch::{FixedUseConstraint, TargetOperandLowering};
use core::marker::PhantomData;
use veloc_lir::InstBuild;
use veloc_lir::{InstId, MachineFunction, Reg, Writable};

/// Shared rewriting with separate generic and target instruction copy policies.
struct OperandConstraintPassImpl<'a, Policy> {
    lowering: &'a dyn TargetOperandLowering,
    _policy: PhantomData<Policy>,
}

trait ConstraintPolicy {
    fn operand_constraints(
        lowering: &dyn TargetOperandLowering,
        inst: &veloc_lir::InstRef<'_>,
        mfunc: &MachineFunction,
    ) -> crate::target::arch::OperandConstraintSet;

    fn build_copy(
        lowering: &dyn TargetOperandLowering,
        mfunc: &mut MachineFunction,
        dst: Reg,
        src: Reg,
    ) -> InstId;
}

struct PreSelectConstraints;
struct PostSelectConstraints;

impl ConstraintPolicy for PreSelectConstraints {
    fn operand_constraints(
        lowering: &dyn TargetOperandLowering,
        inst: &veloc_lir::InstRef<'_>,
        mfunc: &MachineFunction,
    ) -> crate::target::arch::OperandConstraintSet {
        lowering.preselect_operand_constraints(inst, mfunc)
    }

    fn build_copy(
        lowering: &dyn TargetOperandLowering,
        mfunc: &mut MachineFunction,
        dst: Reg,
        src: Reg,
    ) -> InstId {
        if dst.is_vreg() && src.is_vreg() {
            mfunc.writer().copy(Writable(dst), src)
        } else {
            lowering
                .build_preselect_reg_copy(mfunc, dst, src)
                .unwrap_or_else(|err| {
                    panic!(
                        "failed to build pre-select reg copy for {:?} <- {:?}: {}",
                        dst, src, err
                    )
                })
        }
    }
}

impl ConstraintPolicy for PostSelectConstraints {
    fn operand_constraints(
        lowering: &dyn TargetOperandLowering,
        inst: &veloc_lir::InstRef<'_>,
        mfunc: &MachineFunction,
    ) -> crate::target::arch::OperandConstraintSet {
        lowering.postselect_operand_constraints(inst, mfunc)
    }

    fn build_copy(
        lowering: &dyn TargetOperandLowering,
        mfunc: &mut MachineFunction,
        dst: Reg,
        src: Reg,
    ) -> InstId {
        lowering
            .build_postselect_reg_copy(mfunc, dst, src)
            .unwrap_or_else(|err| {
                panic!(
                    "failed to build post-select reg copy for {:?} <- {:?}: {}",
                    dst, src, err
                )
            })
    }
}

impl<'a, Policy> OperandConstraintPassImpl<'a, Policy>
where
    Policy: ConstraintPolicy,
{
    pub fn new(lowering: &'a dyn TargetOperandLowering) -> Self {
        Self {
            lowering,
            _policy: PhantomData,
        }
    }

    pub fn run(&self, mfunc: &mut MachineFunction) -> Result<()> {
        let _ = self.apply(mfunc)?;
        Ok(())
    }

    fn run_with_effect(&self, mfunc: &mut MachineFunction) -> Result<PassEffect> {
        let changed = self.apply(mfunc)?;
        if changed == 0 {
            Ok(PassEffect::NONE)
        } else {
            Ok(PassEffect::new(
                ChangeSet::INST_SEMANTICS | ChangeSet::INST_OPERANDS,
            ))
        }
    }

    fn apply(&self, mfunc: &mut MachineFunction) -> Result<usize> {
        let num_blocks = mfunc.num_blocks();
        let mut changed = 0usize;
        for block_idx in 0..num_blocks {
            mfunc.rewrite_block(block_idx, |cursor| self.rewrite_block(cursor, &mut changed))?;
        }

        Ok(changed)
    }

    fn rewrite_block(
        &self,
        cursor: &mut veloc_lir::BlockRewriteCursor<'_>,
        changed: &mut usize,
    ) -> Result<()> {
        if cursor.current_inst().is_invalid() {
            cursor.remove_current();
            *changed += 1;
            return Ok(());
        }

        let constraints =
            Policy::operand_constraints(self.lowering, &cursor.current_inst(), cursor.mfunc());
        if constraints.is_empty() {
            cursor.keep_current();
            return Ok(());
        }

        let inst_changed = self.apply_constraints(cursor, &constraints)?;
        if inst_changed {
            *changed += 1;
            cursor.keep_current();
        } else {
            cursor.keep_current();
        }
        Ok(())
    }

    fn apply_constraints(
        &self,
        cursor: &mut veloc_lir::BlockRewriteCursor<'_>,
        constraints: &crate::target::arch::OperandConstraintSet,
    ) -> Result<bool> {
        let mut changed = false;
        for fixed in constraints.fixed_uses.iter() {
            if self.apply_fixed_use_constraint(cursor, fixed)? {
                changed = true;
            }
        }

        Ok(changed)
    }

    fn apply_fixed_use_constraint(
        &self,
        cursor: &mut veloc_lir::BlockRewriteCursor<'_>,
        fixed: &FixedUseConstraint,
    ) -> Result<bool> {
        let inst = cursor.current_inst();
        let index = fixed.use_operand;
        let current = inst.inputs()[index];
        if current == fixed.reg {
            return Ok(false);
        }

        self.emit_constraint_copy(cursor, fixed.reg, current)?;
        let id = cursor.current_inst_id();
        cursor.mfunc_mut().set_inst_input(id, index, fixed.reg);
        Ok(true)
    }

    fn emit_constraint_copy(
        &self,
        cursor: &mut veloc_lir::BlockRewriteCursor<'_>,
        dst: Reg,
        src: Reg,
    ) -> Result<()> {
        let copy_inst = Policy::build_copy(self.lowering, cursor.mfunc_mut(), dst, src);
        cursor.emit(copy_inst);
        Ok(())
    }
}

pub struct PreSelectOperandConstraintPass<'a> {
    inner: OperandConstraintPassImpl<'a, PreSelectConstraints>,
}

impl<'a> PreSelectOperandConstraintPass<'a> {
    pub fn new(lowering: &'a dyn TargetOperandLowering) -> Self {
        Self {
            inner: OperandConstraintPassImpl::new(lowering),
        }
    }

    pub fn run(&self, mfunc: &mut MachineFunction) -> Result<()> {
        self.inner.run(mfunc)
    }
}

impl<'a> FunctionPass for PreSelectOperandConstraintPass<'a> {
    fn name(&self) -> &'static str {
        "operand-constraints"
    }

    fn run(
        &self,
        mfunc: &mut MachineFunction,
        _ctx: &mut FunctionPassContext<'_>,
    ) -> Result<PassEffect> {
        self.inner.run_with_effect(mfunc)
    }
}

pub struct PostSelectOperandConstraintPass<'a> {
    inner: OperandConstraintPassImpl<'a, PostSelectConstraints>,
}

impl<'a> PostSelectOperandConstraintPass<'a> {
    pub fn new(lowering: &'a dyn TargetOperandLowering) -> Self {
        Self {
            inner: OperandConstraintPassImpl::new(lowering),
        }
    }

    pub fn run(&self, mfunc: &mut MachineFunction) -> Result<()> {
        self.inner.run(mfunc)
    }
}

impl<'a> FunctionPass for PostSelectOperandConstraintPass<'a> {
    fn name(&self) -> &'static str {
        "operand-constraints"
    }

    fn run(
        &self,
        mfunc: &mut MachineFunction,
        _ctx: &mut FunctionPassContext<'_>,
    ) -> Result<PassEffect> {
        self.inner.run_with_effect(mfunc)
    }
}

#[cfg(test)]
mod tests {
    use super::PreSelectOperandConstraintPass;
    use crate::target::arch::{FixedUseConstraint, OperandConstraintSet, TargetOperandLowering};
    use alloc::vec;
    use veloc_lir::{InstBuild, InstRead};
    use veloc_lir::{InstId, MachineBlock, MachineFunction, Reg, Writable};

    struct DummyLowering {
        constraints: OperandConstraintSet,
    }

    impl DummyLowering {
        fn new(constraints: OperandConstraintSet) -> Self {
            Self { constraints }
        }
    }

    impl TargetOperandLowering for DummyLowering {
        fn preselect_operand_constraints(
            &self,
            _inst: &veloc_lir::InstRef<'_>,
            _mfunc: &MachineFunction,
        ) -> OperandConstraintSet {
            self.constraints.clone()
        }

        fn build_preselect_reg_copy(
            &self,
            mfunc: &mut MachineFunction,
            dst: Reg,
            src: Reg,
        ) -> Result<InstId, crate::error::Error> {
            Ok(mfunc.writer().copy(Writable(dst), src))
        }
    }

    fn make_function_with_inst(
        build: impl FnOnce(veloc_lir::InstWriter<'_>) -> InstId,
    ) -> (MachineFunction, veloc_lir::InstId) {
        let mut mfunc = MachineFunction::new("test".into());
        mfunc
            .blocks
            .push(MachineBlock::new(veloc_mir::Block::from_u32(0)));
        let inst_id = build(mfunc.writer());
        mfunc.append_inst_id_to_block(0, inst_id);
        (mfunc, inst_id)
    }

    #[test]
    fn fixed_use_inserts_copy_and_rewrites_operand() {
        let dst = Reg::new_vreg(0);
        let src = Reg::new_vreg(1);
        let fixed = Reg::new_preg(7);
        let inst = |writer: veloc_lir::InstWriter<'_>| {
            writer.unary(
                veloc_lir::MachineOpcode::Generic(veloc_lir::GenericOpcode::Neg),
                Writable(dst),
                src,
            )
        };
        let (mut mfunc, inst_id) = make_function_with_inst(inst);
        let lowering = DummyLowering::new(OperandConstraintSet {
            fixed_uses: vec![FixedUseConstraint {
                use_operand: 0,
                reg: fixed,
            }]
            .into(),
        });

        PreSelectOperandConstraintPass::new(&lowering)
            .run(&mut mfunc)
            .unwrap();

        assert_eq!(mfunc.blocks[0].insts.len(), 2);
        let veloc_lir::InstView::UnaryReg(copy) = mfunc.inst(mfunc.blocks[0].insts[0]).view()
        else {
            panic!("expected UnaryReg");
        };
        assert_eq!(copy.dst, fixed);
        assert_eq!(copy.src, src);

        let veloc_lir::InstView::UnaryReg(lowered) = mfunc.inst(inst_id).view() else {
            panic!("expected UnaryReg");
        };
        assert_eq!(lowered.src, fixed);
    }
}
