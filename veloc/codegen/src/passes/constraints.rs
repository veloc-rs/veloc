use crate::analysis::{ChangeSet, PassEffect};
use crate::error::Result;
use crate::pipeline::{FunctionPass, FunctionPassContext};
use crate::target::TargetOperandLowering;
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
    ) -> crate::target::OperandConstraintSet;

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
    ) -> crate::target::OperandConstraintSet {
        lowering.preselect_operand_constraints(inst, mfunc)
    }

    fn build_copy(
        lowering: &dyn TargetOperandLowering,
        mfunc: &mut MachineFunction,
        dst: Reg,
        src: Reg,
    ) -> InstId {
        if dst.is_vreg() && src.is_vreg() {
            mfunc.editor().writer().copy(Writable(dst), src)
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
    ) -> crate::target::OperandConstraintSet {
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
        let mut changed = 0usize;
        let ids: alloc::vec::Vec<_> = mfunc.blocks().flat_map(|b| mfunc.block_insts(b)).collect();
        for id in ids {
            if mfunc.inst(id).is_invalid() {
                continue;
            }
            let constraints = Policy::operand_constraints(self.lowering, &mfunc.inst(id), mfunc);
            let mut inst_changed = false;
            for fixed in constraints.fixed_uses.iter() {
                let current = mfunc.inst(id).inputs()[fixed.use_operand];
                if current == fixed.reg {
                    continue;
                }
                let copy = Policy::build_copy(self.lowering, mfunc, fixed.reg, current);
                let mut edit = mfunc.editor();
                edit.insert_before(id, copy);
                edit.set_inst_input(id, fixed.use_operand, fixed.reg);
                inst_changed = true;
            }
            changed += usize::from(inst_changed);
        }
        Ok(changed)
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
    use crate::target::{FixedUseConstraint, OperandConstraintSet, TargetOperandLowering};
    use alloc::vec;
    use veloc_lir::{InstBuild, InstRead};
    use veloc_lir::{InstId, MachineFunction, Reg, Writable};

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
            Ok(mfunc.editor().writer().copy(Writable(dst), src))
        }
    }

    fn make_function_with_inst(
        build: impl FnOnce(veloc_lir::InstWriter<'_>) -> InstId,
    ) -> (MachineFunction, veloc_lir::InstId) {
        let mut mfunc = MachineFunction::new("test".into());
        mfunc.editor().create_block();
        let inst_id = build(mfunc.editor().writer());
        mfunc
            .editor()
            .append_inst(veloc_lir::BlockId::from_u32(0), inst_id);
        (mfunc, inst_id)
    }

    #[test]
    fn fixed_use_inserts_copy_and_rewrites_operand() {
        let dst = Reg::new_vreg(0);
        let src = Reg::new_vreg(1);
        let fixed = Reg::new_preg(7);
        let inst = |writer: veloc_lir::InstWriter<'_>| writer.neg(Writable(dst), src);
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

        assert_eq!(
            mfunc
                .block_insts(veloc_lir::BlockId::from_u32(0))
                .collect::<alloc::vec::Vec<_>>()
                .len(),
            2
        );
        let veloc_lir::InstView::UnaryReg(copy) = mfunc
            .inst(
                mfunc
                    .block_insts(veloc_lir::BlockId::from_u32(0))
                    .collect::<alloc::vec::Vec<_>>()[0],
            )
            .view()
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
