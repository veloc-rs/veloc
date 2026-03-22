use crate::error::Result;
use crate::mir::{MachineFunction, MachineInst, Reg, Writable};
use crate::pipeline::{ChangeSet, FunctionPass, FunctionPassContext, PassEffect};
use crate::target::arch::{
    FixedUseConstraint, OperandConstraintStage, TargetLowering, TiedOperandConstraint,
};

/// 在给定阶段应用 target/指令元数据定义的操作数约束。
pub struct OperandConstraintPass<'a> {
    lowering: &'a dyn TargetLowering,
    stage: OperandConstraintStage,
}

impl<'a> OperandConstraintPass<'a> {
    pub fn new(lowering: &'a dyn TargetLowering, stage: OperandConstraintStage) -> Self {
        Self { lowering, stage }
    }

    pub fn run<S>(&self, mfunc: &mut MachineFunction<S>) -> Result<()> {
        let _ = self.apply(mfunc)?;
        Ok(())
    }

    fn apply<S>(&self, mfunc: &mut MachineFunction<S>) -> Result<usize> {
        let num_blocks = mfunc.num_blocks();
        let mut changed = 0usize;
        for block_idx in 0..num_blocks {
            mfunc.rewrite_block(block_idx, |cursor| -> Result<()> {
                if cursor.current_inst().is_invalid() {
                    cursor.remove_current();
                    changed += 1;
                    return Ok(());
                }

                let mut inst = cursor.current_inst_clone();
                let constraints = self.lowering.operand_constraints(
                    self.stage,
                    &inst,
                    cursor.mfunc().as_untyped(),
                );
                if constraints.is_empty() {
                    cursor.keep_current();
                    return Ok(());
                }

                let mut inst_changed = false;
                for tied in &constraints.tied_operands {
                    if self.apply_tied_constraint(
                        cursor,
                        &mut inst,
                        tied,
                        &constraints.commute_operand_pairs,
                    )? {
                        inst_changed = true;
                    }
                }

                for fixed in &constraints.fixed_uses {
                    if self.apply_fixed_use_constraint(cursor, &mut inst, fixed)? {
                        inst_changed = true;
                    }
                }

                if !inst_changed {
                    cursor.keep_current();
                } else {
                    changed += 1;
                    cursor.replace_current(inst);
                }
                Ok(())
            })?;
        }

        Ok(changed)
    }

    fn apply_tied_constraint<S>(
        &self,
        cursor: &mut crate::mir::BlockRewriteCursor<'_, S>,
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

    fn apply_fixed_use_constraint<S>(
        &self,
        cursor: &mut crate::mir::BlockRewriteCursor<'_, S>,
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

    fn emit_constraint_copy<S>(
        &self,
        cursor: &mut crate::mir::BlockRewriteCursor<'_, S>,
        dst: Reg,
        src: Reg,
    ) -> Result<()> {
        let copy_inst = build_constraint_copy(
            self.lowering,
            cursor.mfunc().as_untyped(),
            self.stage,
            dst,
            src,
        );
        cursor.emit_before(copy_inst);
        Ok(())
    }
}

impl<'a, S> FunctionPass<S> for OperandConstraintPass<'a> {
    fn name(&self) -> &'static str {
        "operand-constraints"
    }

    fn run(
        &self,
        mfunc: &mut MachineFunction<S>,
        _ctx: &mut FunctionPassContext<'_, S>,
    ) -> Result<PassEffect> {
        let changed = self.apply(mfunc)?;
        if changed == 0 {
            Ok(PassEffect::NONE)
        } else {
            Ok(PassEffect::new(
                ChangeSet::INST_SEMANTICS | ChangeSet::INST_OPERANDS,
            ))
        }
    }
}

use crate::mir::MachineOperand;

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

fn build_constraint_copy(
    lowering: &dyn TargetLowering,
    mfunc: &MachineFunction,
    stage: OperandConstraintStage,
    dst: Reg,
    src: Reg,
) -> MachineInst {
    if dst.is_preg() || src.is_preg() || stage == OperandConstraintStage::PostSelect {
        lowering
            .build_reg_copy(mfunc, dst, src)
            .unwrap_or_else(|err| {
                panic!(
                    "failed to build target reg copy for {:?} <- {:?}: {}",
                    dst, src, err
                )
            })
    } else {
        MachineInst::build_copy(Writable(dst), src)
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
    use super::OperandConstraintPass;
    use crate::isel::{LegalizerInfo, SelectResult, SelectionContext};
    use crate::mir::{MachineBlock, MachineFunction, MachineInst, Reg, Writable};
    use crate::target::arch::{
        FixedUseConstraint, OperandConstraintSet, OperandConstraintStage, TargetLowering,
        TiedOperandConstraint,
    };
    use alloc::vec;
    use alloc::vec::Vec;

    struct DummyLowering {
        legalizer_info: LegalizerInfo,
        constraints: OperandConstraintSet,
    }

    impl DummyLowering {
        fn new(constraints: OperandConstraintSet) -> Self {
            Self {
                legalizer_info: LegalizerInfo::new(),
                constraints,
            }
        }
    }

    impl TargetLowering for DummyLowering {
        fn finalize_stack_frame(
            &self,
            _mfunc: &mut MachineFunction,
            _call_conv: crate::target::arch::CallConv,
        ) {
        }

        fn insert_prologue_epilogue(&self, _mfunc: &mut MachineFunction) {}

        fn legalize_instruction(
            &self,
            _inst_id: crate::mir::InstId,
            _mfunc: &mut MachineFunction,
            _output: &mut Vec<crate::mir::InstId>,
        ) {
        }

        fn select_instruction(
            &self,
            _ctx: &mut SelectionContext,
        ) -> Result<SelectResult, crate::error::Error> {
            unreachable!("selection is not used in operand constraint tests")
        }

        fn operand_constraints(
            &self,
            _stage: OperandConstraintStage,
            _inst: &MachineInst,
            _mfunc: &MachineFunction,
        ) -> OperandConstraintSet {
            self.constraints.clone()
        }

        fn build_reg_copy(
            &self,
            _mfunc: &MachineFunction,
            dst: Reg,
            src: Reg,
        ) -> Result<MachineInst, crate::error::Error> {
            Ok(MachineInst::build_copy(Writable(dst), src))
        }

        fn legalizer_info(&self) -> &LegalizerInfo {
            &self.legalizer_info
        }
    }

    fn make_function_with_inst(inst: MachineInst) -> (MachineFunction, crate::mir::InstId) {
        let mut mfunc = MachineFunction::new("test".into());
        mfunc
            .blocks
            .push(MachineBlock::new(veloc_ir::Block::from_u32(0)));
        let inst_id = mfunc.alloc_inst(inst);
        mfunc.append_inst_id_to_block(0, inst_id);
        (mfunc, inst_id)
    }

    #[test]
    fn tied_use_already_satisfied_inserts_no_copy() {
        let reg = Reg::new_vreg(0);
        let inst = MachineInst::build_binary(
            crate::mir::MachineOpcode::Generic(crate::mir::GenericOpcode::G_ADD),
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

        OperandConstraintPass::new(&lowering, OperandConstraintStage::PreSelect)
            .run(&mut mfunc)
            .unwrap();

        assert_eq!(mfunc.blocks[0].insts, vec![inst_id]);
    }

    #[test]
    fn tied_use_prefers_commutation_over_copy() {
        let dst = Reg::new_vreg(0);
        let lhs = Reg::new_vreg(1);
        let inst = MachineInst::build_binary(
            crate::mir::MachineOpcode::Generic(crate::mir::GenericOpcode::G_ADD),
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

        OperandConstraintPass::new(&lowering, OperandConstraintStage::PreSelect)
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
            crate::mir::MachineOpcode::Generic(crate::mir::GenericOpcode::G_NEG),
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

        OperandConstraintPass::new(&lowering, OperandConstraintStage::PreSelect)
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
