//! Validation at pass boundaries, independent of instruction construction.
use super::FunctionAnalysisCtx;
use crate::{
    Error, Result,
    target::arch::{TargetInstructions, ValidationMode},
};
use alloc::{format, vec::Vec};
use hashbrown::{HashMap, HashSet};
use veloc_lir::BlockId as Block;
#[cfg(test)]
use veloc_lir::InstBuild;
use veloc_lir::InstRead;
use veloc_lir::{ControlFlow, InstField, MachineFunction, Reg};

/// Selected code must still be SSA and contain no generic instructions.
pub fn verify_selected(f: &MachineFunction, target: &dyn TargetInstructions) -> Result<()> {
    verify(f, target)?;
    for block in f.blocks() {
        for id in f.block_insts(block) {
            if !matches!(f.inst(id).opcode(), veloc_lir::MachineOpcode::Target(_)) {
                return Err(Error::codegen(format!("unselected instruction {id:?}")));
            }
        }
    }
    Ok(())
}

/// Allocation removes SSA block parameters and all executable virtual registers.
/// This check is explicit: no mutable phase flag can cause it to be skipped.
pub fn verify_allocated(f: &MachineFunction, target: &dyn TargetInstructions) -> Result<()> {
    f.check_refs().map_err(|e| Error::codegen(e))?;
    if !f.params.is_empty() {
        return Err(Error::codegen(
            "function parameters remain after allocation",
        ));
    }
    for block in f.blocks() {
        if !f.block_params(block).unwrap().is_empty() {
            return Err(Error::codegen("block parameters remain after allocation"));
        }
        for id in f.block_insts(block) {
            let inst = f.inst(id);
            if !matches!(inst.opcode(), veloc_lir::MachineOpcode::Target(_)) {
                return Err(Error::codegen(format!("unselected instruction {id:?}")));
            }
            if inst.defs().chain(inst.uses()).any(|reg| reg.is_vreg()) {
                return Err(Error::codegen(format!(
                    "virtual register remains in {id:?}"
                )));
            }
            target.validate_instruction(&inst, ValidationMode::Allocated)?;
        }
    }
    Ok(())
}

pub fn verify(f: &MachineFunction, target: &dyn TargetInstructions) -> Result<()> {
    let fail = |message| Error::codegen(format!("machine SSA in {}: {message}", f.name));
    f.check_refs().map_err(|e| fail(e.into()))?;
    let mut defs = HashMap::new();
    let mut blocks = HashSet::new();
    let mut instructions = HashSet::new();
    for block in f.blocks() {
        if !blocks.insert(block) {
            return Err(fail(format!("duplicate block {}", block)));
        }
        let mut define = |reg: Reg, pos| -> Result<()> {
            let Some(value) = reg.as_vreg() else {
                return Ok(());
            };
            if f.vregs().get(value).is_none() {
                return Err(fail(format!("unknown value {reg:?}")));
            }
            if let Some(previous) = defs.insert(reg, (block, pos)) {
                return Err(fail(format!(
                    "multiple definitions of {reg:?}: {previous:?} and ({:?}, {pos})",
                    block
                )));
            }
            Ok(())
        };
        for &param in f.block_params(block).unwrap() {
            if param.is_preg() {
                return Err(fail("physical block parameter before allocation".into()));
            }
            define(param, 0)?;
        }
        let mut transferred = false;
        for (index, id) in f.block_insts(block).enumerate() {
            if !instructions.insert(id) {
                return Err(fail(format!("instruction {id:?} occurs twice in layout")));
            }
            let inst = f.inst(id);
            if inst.is_generic() {
                inst.validate()?;
            } else {
                target.validate_instruction(&inst, ValidationMode::Virtual)?;
            }
            for reg in inst.defs() {
                if transferred && reg.is_vreg() {
                    return Err(fail(format!(
                        "SSA definition after conditional block exit: {id:?}"
                    )));
                }
                define(reg, index + 1)?;
            }
            match target.control_flow(&inst) {
                ControlFlow::Branch => transferred = true,
                ControlFlow::Jump | ControlFlow::Return | ControlFlow::Trap
                    if f.layout().next_inst(id).is_some() =>
                {
                    return Err(fail(format!("instruction after unconditional exit {id:?}")));
                }
                _ => {}
            }
        }
    }
    let mut analyses = FunctionAnalysisCtx::default();
    let cfg = analyses.cfg(f, target).clone();
    let mut reachable = HashSet::new();
    let mut pending: Vec<_> = f.entry_block().into_iter().collect();
    while let Some(block) = pending.pop() {
        if reachable.insert(block) {
            pending.extend_from_slice(cfg.succs(block));
        }
    }
    let dom = analyses.dominators(f, target);
    for block in f.blocks() {
        let falls_through = f.layout().last_inst(block).is_none_or(|id| {
            matches!(
                target.control_flow(&f.inst(id)),
                ControlFlow::Next | ControlFlow::Call | ControlFlow::Branch
            )
        });
        if falls_through
            && f.layout()
                .next_block(block)
                .is_some_and(|next| !f.block_params(next).unwrap().is_empty())
        {
            return Err(fail(format!(
                "fallthrough from {} cannot supply block parameters",
                block
            )));
        }
        for (index, id) in f.block_insts(block).enumerate() {
            let inst = f.inst(id);
            for reg in inst.uses().filter(|r| r.is_vreg()) {
                let &(owner, pos) = defs
                    .get(&reg)
                    .ok_or_else(|| fail(format!("undefined {reg:?} used by {id:?}")))?;
                if (owner == block && pos >= index + 1)
                    || (owner != block
                        && reachable.contains(&block)
                        && !dom.dominates(owner, block))
                {
                    return Err(fail(format!(
                        "definition of {reg:?} does not dominate {id:?}"
                    )));
                }
            }
            let targets: Vec<_> = inst
                .fields()
                .iter()
                .filter_map(|op| {
                    if let InstField::Block(block) = op {
                        Some(*block)
                    } else {
                        None
                    }
                })
                .collect();
            let check_edge = |block: Block, args: &[Reg]| -> Result<()> {
                let params = f
                    .block_params(block)
                    .ok_or_else(|| fail(format!("unknown successor {block}")))?;
                if params.len() != args.len() {
                    return Err(fail(format!(
                        "edge to {block} has {} arguments for {} parameters",
                        args.len(),
                        params.len()
                    )));
                }
                for (&param, &arg) in params.iter().zip(args) {
                    if !arg.is_vreg() || f.vreg_data(param).ty != f.vreg_data(arg).ty {
                        return Err(fail(format!("edge type mismatch for {param:?} <- {arg:?}")));
                    }
                }
                Ok(())
            };
            match f.inst_extra(id) {
                Some(veloc_lir::InstExtraRef::Branch(info)) => {
                    let [target] = targets.as_slice() else {
                        return Err(fail("invalid single-edge shape".into()));
                    };
                    check_edge(*target, &info.args)?;
                }
                Some(veloc_lir::InstExtraRef::BranchCond(info)) => {
                    let [yes, no] = targets.as_slice() else {
                        return Err(fail("invalid conditional-edge shape".into()));
                    };
                    check_edge(*yes, &info.then_args)?;
                    check_edge(*no, &info.else_args)?;
                }
                Some(veloc_lir::InstExtraRef::BrTable(info)) => {
                    for edge in info.targets() {
                        check_edge(edge.block, &edge.args)?;
                    }
                }
                _ => {
                    for block in targets {
                        check_edge(block, &[])?;
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use veloc_lir::{BranchCondInfo, BranchInfo, InstExtra, Type, Writable};

    #[test]
    fn checks_representation_invariants_without_phase_tags() {
        use crate::target::x86_64::isle::{REG_RAX, TargetInst};
        use veloc_lir::{InstField, MachineOpcode};
        let target =
            crate::target::x86_64::X86_64TargetMachine::new(crate::TargetConfig::default());
        let mut f = MachineFunction::new("boundaries".into());
        f.editor().create_block();
        let value = f.editor().alloc_vreg(Type::I64);
        let constant = f.editor().writer().constant(Writable(value), 42);
        f.editor()
            .append_inst(veloc_lir::BlockId::from_u32(0), constant);
        let ret = f.editor().writer().ret(&[value]);
        f.editor().append_inst(veloc_lir::BlockId::from_u32(0), ret);
        verify(&f, &target).unwrap();
        assert!(verify_selected(&f, &target).is_err());

        f.editor().rewriter(constant).write(
            MachineOpcode::Target(TargetInst::X86Mov64Imm64.as_u32()),
            &[value],
            &[],
            &[InstField::Imm(42)],
        );
        f.editor().rewriter(ret).write(
            MachineOpcode::Target(TargetInst::X86Ret.as_u32()),
            &[],
            &[],
            &[],
        );
        verify_selected(&f, &target).unwrap();
        assert!(verify_allocated(&f, &target).is_err());

        f.editor().rewriter(constant).write(
            MachineOpcode::Target(TargetInst::X86Mov64Imm64.as_u32()),
            &[REG_RAX],
            &[],
            &[InstField::Imm(42)],
        );
        verify_allocated(&f, &target).unwrap();
        f.params.push(value);
        assert!(verify_allocated(&f, &target).is_err());
        f.params.clear();
        f.editor().append_block_param(Block::from_u32(0), value);
        assert!(verify_allocated(&f, &target).is_err());
    }

    #[test]
    fn verifies_definitions_dominance_and_edge_contracts() {
        let target =
            crate::target::x86_64::X86_64TargetMachine::new(crate::TargetConfig::default());
        let mut f = MachineFunction::new("diamond".into());
        let blocks: Vec<_> = (0..4).map(|_| f.editor().create_block()).collect();
        let x = f.editor().alloc_vreg(Type::I64);
        let y = f.editor().alloc_vreg(Type::I64);
        let p = f.editor().alloc_vreg(Type::I64);
        let c = f.editor().alloc_vreg(Type::BOOL);
        let r = f.editor().alloc_vreg(Type::I64);
        f.editor().append_block_param(blocks[3], p);
        let a = f.editor().writer().constant(Writable(x), 1);
        f.editor().append_inst(veloc_lir::BlockId::from_u32(0), a);
        let b = f.editor().writer().constant(Writable(c), 1);
        f.editor().append_inst(veloc_lir::BlockId::from_u32(0), b);
        let branch = f.editor().writer().brcond(c, blocks[1], blocks[2]);
        f.editor().set_inst_extra(
            branch,
            InstExtra::BranchCond(BranchCondInfo {
                then_args: Default::default(),
                else_args: Default::default(),
            }),
        );
        f.editor()
            .append_inst(veloc_lir::BlockId::from_u32(0), branch);
        let left = f.editor().writer().br(blocks[3]);
        f.editor().set_inst_extra(
            left,
            InstExtra::Branch(BranchInfo {
                args: smallvec::smallvec![x],
            }),
        );
        f.editor()
            .append_inst(veloc_lir::BlockId::from_u32(1), left);
        let def_y = f.editor().writer().constant(Writable(y), 2);
        f.editor()
            .append_inst(veloc_lir::BlockId::from_u32(2), def_y);
        let right = f.editor().writer().br(blocks[3]);
        f.editor().set_inst_extra(
            right,
            InstExtra::Branch(BranchInfo {
                args: smallvec::smallvec![y],
            }),
        );
        f.editor()
            .append_inst(veloc_lir::BlockId::from_u32(2), right);
        let copy = f.editor().writer().copy(Writable(r), p);
        f.editor()
            .append_inst(veloc_lir::BlockId::from_u32(3), copy);
        let ret = f.editor().writer().ret(&[r]);
        f.editor().append_inst(veloc_lir::BlockId::from_u32(3), ret);
        verify(&f, &target).unwrap();

        let mut broken = f.clone();
        broken.editor().rewriter(copy).copy(Writable(p), x);
        assert!(
            verify(&broken, &target)
                .unwrap_err()
                .to_string()
                .contains("multiple definitions")
        );
        let mut broken = f.clone();
        broken.editor().set_inst_extra(
            left,
            InstExtra::Branch(BranchInfo {
                args: smallvec::smallvec![y],
            }),
        );
        assert!(
            verify(&broken, &target)
                .unwrap_err()
                .to_string()
                .contains("does not dominate")
        );
        let mut broken = f.clone();
        broken.editor().clear_inst_extra(left);
        assert!(
            verify(&broken, &target)
                .unwrap_err()
                .to_string()
                .contains("arguments for")
        );
        let mut broken = f.clone();
        broken.editor().set_inst_extra(
            left,
            InstExtra::Branch(BranchInfo {
                args: smallvec::smallvec![c],
            }),
        );
        assert!(
            verify(&broken, &target)
                .unwrap_err()
                .to_string()
                .contains("type mismatch")
        );
        let mut broken = f.clone();
        broken.editor().rewriter(a).copy(Writable(x), x);
        assert!(
            verify(&broken, &target)
                .unwrap_err()
                .to_string()
                .contains("does not dominate")
        );
    }
}
