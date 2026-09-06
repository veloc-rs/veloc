use crate::error::Result;
use crate::pipeline::{ChangeSet, FunctionPass, FunctionPassContext, PassEffect};
use crate::target::arch::{FixedUseConstraint, TargetOperandLowering, TiedOperandConstraint};
use core::marker::PhantomData;
use veloc_lir::MachineOperand;
use veloc_lir::stages::{PreIselPrepared, SelectedLir};
use veloc_lir::{MachineFunction, MachineInst, Reg, Writable};

/// 在给定阶段应用 target/指令元数据定义的操作数约束。
struct OperandConstraintPassImpl<'a, Stage> {
    lowering: &'a dyn TargetOperandLowering,
    _stage: PhantomData<Stage>,
}

trait ConstraintStageSpec {
    type Stage;

    fn operand_constraints(
        lowering: &dyn TargetOperandLowering,
        inst: &MachineInst,
        mfunc: &MachineFunction<Self::Stage>,
    ) -> crate::target::arch::OperandConstraintSet;

    fn build_copy(
        lowering: &dyn TargetOperandLowering,
        mfunc: &MachineFunction<Self::Stage>,
        dst: Reg,
        src: Reg,
    ) -> MachineInst;
}

struct PreSelectConstraintStage;
struct PostSelectConstraintStage;

impl ConstraintStageSpec for PreSelectConstraintStage {
    type Stage = PreIselPrepared;

    fn operand_constraints(
        lowering: &dyn TargetOperandLowering,
        inst: &MachineInst,
        mfunc: &MachineFunction<PreIselPrepared>,
    ) -> crate::target::arch::OperandConstraintSet {
        lowering.preselect_operand_constraints(inst, mfunc)
    }

    fn build_copy(
        lowering: &dyn TargetOperandLowering,
        mfunc: &MachineFunction<PreIselPrepared>,
        dst: Reg,
        src: Reg,
    ) -> MachineInst {
        if dst.is_vreg() && src.is_vreg() {
            MachineInst::build_copy(Writable(dst), src)
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

impl ConstraintStageSpec for PostSelectConstraintStage {
    type Stage = SelectedLir;

    fn operand_constraints(
        lowering: &dyn TargetOperandLowering,
        inst: &MachineInst,
        mfunc: &MachineFunction<SelectedLir>,
    ) -> crate::target::arch::OperandConstraintSet {
        lowering.postselect_operand_constraints(inst, mfunc)
    }

    fn build_copy(
        lowering: &dyn TargetOperandLowering,
        mfunc: &MachineFunction<SelectedLir>,
        dst: Reg,
        src: Reg,
    ) -> MachineInst {
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

impl<'a, Stage> OperandConstraintPassImpl<'a, Stage>
where
    Stage: ConstraintStageSpec,
{
    pub fn new(lowering: &'a dyn TargetOperandLowering) -> Self {
        Self {
            lowering,
            _stage: PhantomData,
        }
    }

    pub fn run(&self, mfunc: &mut MachineFunction<Stage::Stage>) -> Result<()> {
        let _ = self.apply(mfunc)?;
        Ok(())
    }

    fn run_with_effect(&self, mfunc: &mut MachineFunction<Stage::Stage>) -> Result<PassEffect> {
        let changed = self.apply(mfunc)?;
        if changed == 0 {
            Ok(PassEffect::NONE)
        } else {
            Ok(PassEffect::new(
                ChangeSet::INST_SEMANTICS | ChangeSet::INST_OPERANDS,
            ))
        }
    }

    fn apply(&self, mfunc: &mut MachineFunction<Stage::Stage>) -> Result<usize> {
        let num_blocks = mfunc.num_blocks();
        let mut changed = 0usize;
        for block_idx in 0..num_blocks {
            mfunc.rewrite_block(block_idx, |cursor| self.rewrite_block(cursor, &mut changed))?;
        }

        Ok(changed)
    }

    fn rewrite_block(
        &self,
        cursor: &mut veloc_lir::BlockRewriteCursor<'_, Stage::Stage>,
        changed: &mut usize,
    ) -> Result<()> {
        if cursor.current_inst().is_invalid() {
            cursor.remove_current();
            *changed += 1;
            return Ok(());
        }

        let mut inst = cursor.current_inst_clone();
        let constraints = Stage::operand_constraints(self.lowering, &inst, cursor.mfunc());
        if constraints.is_empty() {
            cursor.keep_current();
            return Ok(());
        }

        let inst_changed = self.apply_constraints(cursor, &mut inst, &constraints)?;
        if inst_changed {
            *changed += 1;
            cursor.replace_current(inst);
        } else {
            cursor.keep_current();
        }
        Ok(())
    }

    fn apply_constraints(
        &self,
        cursor: &mut veloc_lir::BlockRewriteCursor<'_, Stage::Stage>,
        inst: &mut MachineInst,
        constraints: &crate::target::arch::OperandConstraintSet,
    ) -> Result<bool> {
        let mut changed = false;
        for tied in &constraints.tied_operands {
            if self.apply_tied_operand_constraint(
                cursor,
                inst,
                tied,
                &constraints.commute_operand_pairs,
            )? {
                changed = true;
            }
        }

        for fixed in &constraints.fixed_uses {
            if self.apply_fixed_use_constraint(cursor, inst, fixed)? {
                changed = true;
            }
        }

        Ok(changed)
    }

    fn apply_tied_operand_constraint(
        &self,
        cursor: &mut veloc_lir::BlockRewriteCursor<'_, Stage::Stage>,
        inst: &mut MachineInst,
        tied: &TiedOperandConstraint,
        commute_pairs: &[(usize, usize)],
    ) -> Result<bool> {
        let def_reg = expect_operand_reg(inst, tied.def_operand, "def", operand_reg);
        let mut source_reg = expect_operand_reg(inst, tied.use_operand, "use", use_reg);
        let mut commuted = false;

        if source_reg != def_reg
            && try_commute_tied_use(inst, tied.use_operand, def_reg, commute_pairs)
        {
            source_reg = def_reg;
            commuted = true;
        }

        if source_reg == def_reg {
            return Ok(commuted);
        }

        self.emit_constraint_copy(cursor, def_reg, source_reg)?;
        set_use_reg(inst, tied.use_operand, def_reg);
        Ok(true)
    }

    fn apply_fixed_use_constraint(
        &self,
        cursor: &mut veloc_lir::BlockRewriteCursor<'_, Stage::Stage>,
        inst: &mut MachineInst,
        fixed: &FixedUseConstraint,
    ) -> Result<bool> {
        let current = expect_operand_reg(inst, fixed.use_operand, "fixed use", use_reg);
        if current == fixed.reg {
            return Ok(false);
        }

        self.emit_constraint_copy(cursor, fixed.reg, current)?;
        set_use_reg(inst, fixed.use_operand, fixed.reg);
        Ok(true)
    }

    fn emit_constraint_copy(
        &self,
        cursor: &mut veloc_lir::BlockRewriteCursor<'_, Stage::Stage>,
        dst: Reg,
        src: Reg,
    ) -> Result<()> {
        let copy_inst = Stage::build_copy(self.lowering, cursor.mfunc(), dst, src);
        cursor.emit_before(copy_inst);
        Ok(())
    }
}

pub struct PreSelectOperandConstraintPass<'a> {
    inner: OperandConstraintPassImpl<'a, PreSelectConstraintStage>,
}

impl<'a> PreSelectOperandConstraintPass<'a> {
    pub fn new(lowering: &'a dyn TargetOperandLowering) -> Self {
        Self {
            inner: OperandConstraintPassImpl::new(lowering),
        }
    }

    pub fn run(&self, mfunc: &mut MachineFunction<PreIselPrepared>) -> Result<()> {
        self.inner.run(mfunc)
    }
}

impl<'a> FunctionPass<PreIselPrepared> for PreSelectOperandConstraintPass<'a> {
    fn name(&self) -> &'static str {
        "operand-constraints"
    }

    fn run(
        &self,
        mfunc: &mut MachineFunction<PreIselPrepared>,
        _ctx: &mut FunctionPassContext<'_, PreIselPrepared>,
    ) -> Result<PassEffect> {
        self.inner.run_with_effect(mfunc)
    }
}

pub struct PostSelectOperandConstraintPass<'a> {
    inner: OperandConstraintPassImpl<'a, PostSelectConstraintStage>,
}

impl<'a> PostSelectOperandConstraintPass<'a> {
    pub fn new(lowering: &'a dyn TargetOperandLowering) -> Self {
        Self {
            inner: OperandConstraintPassImpl::new(lowering),
        }
    }

    pub fn run(&self, mfunc: &mut MachineFunction<SelectedLir>) -> Result<()> {
        self.inner.run(mfunc)
    }
}

impl<'a> FunctionPass<SelectedLir> for PostSelectOperandConstraintPass<'a> {
    fn name(&self) -> &'static str {
        "operand-constraints"
    }

    fn run(
        &self,
        mfunc: &mut MachineFunction<SelectedLir>,
        _ctx: &mut FunctionPassContext<'_, SelectedLir>,
    ) -> Result<PassEffect> {
        self.inner.run_with_effect(mfunc)
    }
}

fn operand_reg(operand: &MachineOperand) -> Option<Reg> {
    match operand {
        MachineOperand::Def(w) | MachineOperand::TiedDefUse(w) => Some(w.to_reg()),
        MachineOperand::Use(reg) => Some(*reg),
        _ => None,
    }
}

fn use_reg(operand: &MachineOperand) -> Option<Reg> {
    match operand {
        MachineOperand::Use(reg) => Some(*reg),
        MachineOperand::TiedDefUse(w) => Some(w.to_reg()),
        _ => None,
    }
}

fn missing_operand_error(inst: &MachineInst, role: &str, operand_idx: usize) -> ! {
    panic!(
        "missing {} operand {} for instruction {:?} while applying operand constraints",
        role, operand_idx, inst.opcode
    )
}

fn expect_operand_reg(
    inst: &MachineInst,
    operand_idx: usize,
    role: &str,
    reg_of: fn(&MachineOperand) -> Option<Reg>,
) -> Reg {
    inst.operands
        .get(operand_idx)
        .and_then(reg_of)
        .unwrap_or_else(|| missing_operand_error(inst, role, operand_idx))
}

fn set_use_reg(inst: &mut MachineInst, operand_idx: usize, reg: Reg) {
    let operand = inst.operands.get_mut(operand_idx).unwrap_or_else(|| {
        panic!(
            "operand index {} out of bounds while applying operand constraints",
            operand_idx
        )
    });
    match operand {
        MachineOperand::Use(slot) => *slot = reg,
        MachineOperand::TiedDefUse(slot) => *slot = Writable(reg),
        _ => {
            panic!(
                "operand {} is not a use operand while applying operand constraints",
                operand_idx
            );
        }
    }
}

fn try_commute_tied_use(
    inst: &mut MachineInst,
    use_operand: usize,
    required_reg: Reg,
    commute_pairs: &[(usize, usize)],
) -> bool {
    for &(lhs, rhs) in commute_pairs {
        let other_idx = if lhs == use_operand {
            rhs
        } else if rhs == use_operand {
            lhs
        } else {
            continue;
        };

        if inst.operands.get(other_idx).and_then(use_reg) == Some(required_reg) {
            inst.operands.swap(use_operand, other_idx);
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::PreSelectOperandConstraintPass;
    use crate::target::arch::{
        FixedUseConstraint, OperandConstraintSet, TargetOperandLowering, TiedOperandConstraint,
    };
    use alloc::vec;
    use alloc::vec::Vec;
    use veloc_lir::stages::PreIselPrepared;
    use veloc_lir::{MachineBlock, MachineFunction, MachineInst, Reg, Writable};

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
            _inst: &MachineInst,
            _mfunc: &MachineFunction<PreIselPrepared>,
        ) -> OperandConstraintSet {
            self.constraints.clone()
        }

        fn build_preselect_reg_copy(
            &self,
            _mfunc: &MachineFunction<PreIselPrepared>,
            dst: Reg,
            src: Reg,
        ) -> Result<MachineInst, crate::error::Error> {
            Ok(MachineInst::build_copy(Writable(dst), src))
        }
    }

    fn make_function_with_inst(
        inst: MachineInst,
    ) -> (MachineFunction<PreIselPrepared>, veloc_lir::InstId) {
        let mut mfunc = MachineFunction::<PreIselPrepared>::new("test".into());
        mfunc
            .blocks
            .push(MachineBlock::new(veloc_mir::Block::from_u32(0)));
        let inst_id = mfunc.alloc_inst(inst);
        mfunc.append_inst_id_to_block(0, inst_id);
        (mfunc, inst_id)
    }

    #[test]
    fn tied_use_already_satisfied_inserts_no_copy() {
        let reg = Reg::new_vreg(0);
        let inst = MachineInst::build_binary(
            veloc_lir::MachineOpcode::Generic(veloc_lir::GenericOpcode::G_ADD),
            Writable(reg),
            reg,
            Reg::new_vreg(1),
        );
        let (mut mfunc, inst_id) = make_function_with_inst(inst);
        let lowering = DummyLowering::new(OperandConstraintSet {
            tied_operands: vec![TiedOperandConstraint {
                def_operand: 0,
                use_operand: 1,
            }],
            commute_operand_pairs: vec![(1, 2)],
            fixed_uses: Vec::new(),
        });

        PreSelectOperandConstraintPass::new(&lowering)
            .run(&mut mfunc)
            .unwrap();

        assert_eq!(mfunc.blocks[0].insts, vec![inst_id]);
    }

    #[test]
    fn tied_use_prefers_commutation_over_copy() {
        let dst = Reg::new_vreg(0);
        let lhs = Reg::new_vreg(1);
        let inst = MachineInst::build_binary(
            veloc_lir::MachineOpcode::Generic(veloc_lir::GenericOpcode::G_ADD),
            Writable(dst),
            lhs,
            dst,
        );
        let (mut mfunc, inst_id) = make_function_with_inst(inst);
        let lowering = DummyLowering::new(OperandConstraintSet {
            tied_operands: vec![TiedOperandConstraint {
                def_operand: 0,
                use_operand: 1,
            }],
            commute_operand_pairs: vec![(1, 2)],
            fixed_uses: Vec::new(),
        });

        PreSelectOperandConstraintPass::new(&lowering)
            .run(&mut mfunc)
            .unwrap();

        assert_eq!(mfunc.blocks[0].insts, vec![inst_id]);
        let lowered = mfunc.dfg[inst_id].as_binary_reg().unwrap();
        assert_eq!(lowered.lhs, dst);
        assert_eq!(lowered.rhs, lhs);
    }

    #[test]
    fn fixed_use_inserts_copy_and_rewrites_operand() {
        let dst = Reg::new_vreg(0);
        let src = Reg::new_vreg(1);
        let fixed = Reg::new_preg(7);
        let inst = MachineInst::build_unary(
            veloc_lir::MachineOpcode::Generic(veloc_lir::GenericOpcode::G_NEG),
            Writable(dst),
            src,
        );
        let (mut mfunc, inst_id) = make_function_with_inst(inst);
        let lowering = DummyLowering::new(OperandConstraintSet {
            tied_operands: Vec::new(),
            commute_operand_pairs: Vec::new(),
            fixed_uses: vec![FixedUseConstraint {
                use_operand: 1,
                reg: fixed,
            }],
        });

        PreSelectOperandConstraintPass::new(&lowering)
            .run(&mut mfunc)
            .unwrap();

        assert_eq!(mfunc.blocks[0].insts.len(), 2);
        let copy = mfunc.dfg[mfunc.blocks[0].insts[0]].as_unary_reg().unwrap();
        assert_eq!(copy.dst, fixed);
        assert_eq!(copy.src, src);

        let lowered = mfunc.dfg[inst_id].as_unary_reg().unwrap();
        assert_eq!(lowered.src, fixed);
    }
}
