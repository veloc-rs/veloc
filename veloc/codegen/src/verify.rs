//! Validation at pass boundaries, independent of instruction construction.
use crate::analysis::FunctionAnalysisCtx;
use crate::{
    Error, Result,
    target::{TargetInstructions, ValidationMode},
};
use hashbrown::{HashMap, HashSet};
use std::{format, vec::Vec};
use veloc_lir::BlockId as Block;
#[cfg(test)]
use veloc_lir::InstBuild;
use veloc_lir::InstRead;
use veloc_lir::{ControlFlow, MachineFunction, Reg};

/// Selected code is SSA with target instructions and symbolic call-frame boundaries.
pub fn verify_selected(f: &MachineFunction, target: &dyn TargetInstructions) -> Result<()> {
    verify(f, target)?;
    for block in f.blocks() {
        for id in f.block_insts(block) {
            if !matches!(f.inst(id).opcode(), veloc_lir::MachineOpcode::Target(_))
                && !f.inst(id).is_call_frame()
            {
                return Err(Error::codegen(format!("unselected instruction {id:?}")));
            }
        }
    }
    Ok(())
}

/// Allocation removes SSA block parameters and all executable virtual registers.
/// This check is explicit: no mutable phase flag can cause it to be skipped.
pub fn verify_allocated(f: &MachineFunction, target: &dyn TargetInstructions) -> Result<()> {
    verify_call_frames(f, target)?;
    f.check_refs().map_err(|e| Error::codegen(e))?;
    if !f.params().is_empty() {
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
            if inst.is_call_frame() {
                inst.validate()?;
                continue;
            }
            if !matches!(inst.opcode(), veloc_lir::MachineOpcode::Target(_)) {
                return Err(Error::codegen(format!("unselected instruction {id:?}")));
            }
            if inst.defs().chain(inst.uses()).any(|reg| reg.is_vreg()) {
                return Err(Error::codegen(format!(
                    "virtual register remains in {id:?}"
                )));
            }
            target.validate_instruction(f, &inst, ValidationMode::Allocated)?;
        }
    }
    Ok(())
}

pub fn verify(f: &MachineFunction, target: &dyn TargetInstructions) -> Result<()> {
    verify_call_frames(f, target)?;
    let fail = |message| Error::codegen(format!("machine SSA in {}: {message}", f.name));
    f.check_refs().map_err(|e| fail(e.into()))?;
    let mut defs = HashMap::new();
    for &param in f.params() {
        if param.as_vreg().is_none_or(|v| f.vregs().get(v).is_none()) {
            return Err(fail(format!("invalid function parameter {param:?}")));
        }
        if defs.insert(param, (f.entry_block(), 0)).is_some() {
            return Err(fail(format!("duplicate function parameter {param:?}")));
        }
    }
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
                target.validate_instruction(f, &inst, ValidationMode::Virtual)?;
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
    if !f.params().is_empty() && !cfg.preds(f.entry_block()).is_empty() {
        return Err(fail(
            "function parameters require a dedicated call entry without predecessors".into(),
        ));
    }
    let mut reachable = HashSet::new();
    let mut pending: Vec<_> = std::vec![f.entry_block()];
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
            for edge in f.successors(id) {
                check_edge(edge.block, edge.args)?;
            }
        }
    }
    Ok(())
}

/// The reserved-frame implementation supports nonnested, block-local call
/// sequences. Reject wider lifetimes until stack-state propagation is supported.
pub(crate) fn verify_call_frames(
    f: &MachineFunction,
    target: &dyn TargetInstructions,
) -> Result<()> {
    use veloc_lir::{FieldValueRef, GenericOpcode, MachineOpcode, StackObject};
    let finalized = f.stack_frame.layout().is_some();
    let fail = |message: &str| Error::codegen(format!("call frame in {}: {message}", f.name));
    let mut seen = HashSet::new();
    for block in f.blocks() {
        let mut active = None;
        let mut called = false;
        for id in f.block_insts(block) {
            let inst = f.inst(id);
            if inst.is_call_frame() {
                if finalized {
                    return Err(fail("boundary remains after frame lowering"));
                }
                inst.validate()?;
                let FieldValueRef::CallFrame(frame) = inst.fields().read(0) else {
                    return Err(fail("missing frame identity"));
                };
                if f.stack_frame.call(*frame).is_none() {
                    return Err(fail("unknown frame"));
                }
                if inst.opcode() == MachineOpcode::Generic(GenericOpcode::CallFrameSetup) {
                    if active.is_some() || !seen.insert(*frame) {
                        return Err(fail("nested or repeated frame setup"));
                    }
                    active = Some(*frame);
                    called = false;
                } else {
                    if active != Some(*frame) || !called {
                        return Err(fail("unmatched frame destroy"));
                    }
                    active = None;
                }
                continue;
            }
            if active.is_some()
                && !matches!(
                    target.control_flow(&inst),
                    ControlFlow::Next | ControlFlow::Call
                )
            {
                return Err(fail("control transfer inside a block-local call frame"));
            }
            for index in 0..inst.fields().len() {
                if let FieldValueRef::StackSlot(slot) = inst.fields().read(index) {
                    let slot = f
                        .stack_frame
                        .slots()
                        .get(*slot)
                        .ok_or_else(|| fail("unknown stack slot"))?;
                    if let StackObject::Outgoing { frame, offset } = slot.object {
                        let area = f
                            .stack_frame
                            .call(frame)
                            .ok_or_else(|| fail("unknown outgoing frame"))?;
                        if offset
                            .checked_add(slot.size)
                            .is_none_or(|end| end > area.size)
                            || slot.align > area.align
                            || offset % slot.align != 0
                        {
                            return Err(fail("outgoing object exceeds call frame constraints"));
                        }
                        if !finalized && (active != Some(frame) || called) {
                            return Err(fail("outgoing address outside argument preparation"));
                        }
                    }
                }
            }
            if let Some(info) = f.try_call_info(id) {
                if let Some(frame) = info.frame {
                    if f.stack_frame.call(frame).is_none() {
                        return Err(fail("unknown call frame"));
                    }
                    if !finalized && (active != Some(frame) || called) {
                        return Err(fail("call outside its frame"));
                    }
                    for slot in &info.stack_args {
                        if !matches!(f.stack_frame.slots().get(*slot).map(|s| s.object),
                            Some(StackObject::Outgoing { frame: owner, .. }) if owner == frame)
                        {
                            return Err(fail("call argument belongs to another frame"));
                        }
                    }
                    called = true;
                } else if active.is_some() {
                    return Err(fail("unlowered call inside a call frame"));
                }
            }
        }
        if active.is_some() {
            return Err(fail("cross-block call frames are not supported yet"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::string::ToString;
    use veloc_lir::Type;

    #[test]
    fn checks_representation_invariants_without_phase_tags() {
        use crate::target::x86_64::inst::{REG_RAX, TargetInst};
        use veloc_lir::{FieldValue, MachineOpcode};
        let target =
            crate::target::x86_64::X86_64TargetMachine::new(crate::TargetConfig::default())
                .unwrap();
        let mut f = MachineFunction::new("boundaries".into());
        let value = f.editor().alloc_vreg(Type::I64);
        let constant = f.editor().writer().constant(value, 42);
        f.editor()
            .append_inst(veloc_lir::BlockId::from_u32(0), constant);
        let ret = f.editor().writer().ret(&[value]);
        f.editor().append_inst(veloc_lir::BlockId::from_u32(0), ret);
        verify(&f, &target).unwrap();
        assert!(verify_selected(&f, &target).is_err());

        f.editor().replace(constant).write(
            MachineOpcode::Target(TargetInst::X86Mov64Imm64.as_u32()),
            &[value],
            &[],
            [FieldValue::Imm(42)],
        );
        f.editor().replace(ret).write(
            MachineOpcode::Target(TargetInst::X86Ret.as_u32()),
            &[],
            &[],
            [],
        );
        verify_selected(&f, &target).unwrap();
        assert!(verify_allocated(&f, &target).is_err());

        f.editor().replace(constant).write(
            MachineOpcode::Target(TargetInst::X86Mov64Imm64.as_u32()),
            &[REG_RAX],
            &[],
            [FieldValue::Imm(42)],
        );
        verify_allocated(&f, &target).unwrap();
        f.editor().append_param(value);
        assert!(verify_allocated(&f, &target).is_err());
        f.editor().take_params();
        f.editor().append_block_param(Block::from_u32(0), value);
        assert!(verify_allocated(&f, &target).is_err());
    }

    #[test]
    fn verifies_definitions_dominance_and_edge_contracts() {
        let target =
            crate::target::x86_64::X86_64TargetMachine::new(crate::TargetConfig::default())
                .unwrap();
        let mut f = MachineFunction::new("diamond".into());
        let blocks: Vec<_> = (0..4)
            .map(|i| {
                if i == 0 {
                    f.entry_block()
                } else {
                    f.editor().create_block()
                }
            })
            .collect();
        let x = f.editor().alloc_vreg(Type::I64);
        let y = f.editor().alloc_vreg(Type::I64);
        let p = f.editor().alloc_vreg(Type::I64);
        let c = f.editor().alloc_vreg(Type::BOOL);
        let r = f.editor().alloc_vreg(Type::I64);
        f.editor().append_block_param(blocks[3], p);
        let a = f.editor().writer().constant(x, 1);
        f.editor().append_inst(veloc_lir::BlockId::from_u32(0), a);
        let b = f.editor().writer().constant(c, 1);
        f.editor().append_inst(veloc_lir::BlockId::from_u32(0), b);
        let yes = f.editor().create_edge(blocks[1], &[]);
        let no = f.editor().create_edge(blocks[2], &[]);
        let branch = f.editor().writer().brcond(c, yes, no);
        f.editor()
            .append_inst(veloc_lir::BlockId::from_u32(0), branch);
        let edge = f.editor().create_edge(blocks[3], &[]);
        let left = f.editor().writer().br(edge);
        f.editor().set_edge_args(edge, &[x]);
        f.editor()
            .append_inst(veloc_lir::BlockId::from_u32(1), left);
        let def_y = f.editor().writer().constant(y, 2);
        f.editor()
            .append_inst(veloc_lir::BlockId::from_u32(2), def_y);
        let edge = f.editor().create_edge(blocks[3], &[]);
        let right = f.editor().writer().br(edge);
        f.editor().set_edge_args(edge, &[y]);
        f.editor()
            .append_inst(veloc_lir::BlockId::from_u32(2), right);
        let copy = f.editor().writer().copy(r, p);
        f.editor()
            .append_inst(veloc_lir::BlockId::from_u32(3), copy);
        let ret = f.editor().writer().ret(&[r]);
        f.editor().append_inst(veloc_lir::BlockId::from_u32(3), ret);
        verify(&f, &target).unwrap();

        let mut broken = f.clone();
        broken.editor().replace(copy).copy(p, x);
        assert!(
            verify(&broken, &target)
                .unwrap_err()
                .to_string()
                .contains("multiple definitions")
        );
        let mut broken = f.clone();
        let edge = broken.inst(left).edge_ids().next().unwrap();
        broken.editor().set_edge_args(edge, &[y]);
        assert!(
            verify(&broken, &target)
                .unwrap_err()
                .to_string()
                .contains("does not dominate")
        );
        let mut broken = f.clone();
        broken.editor().clear_successor_args(left);
        assert!(
            verify(&broken, &target)
                .unwrap_err()
                .to_string()
                .contains("arguments for")
        );
        let mut broken = f.clone();
        let edge = broken.inst(left).edge_ids().next().unwrap();
        broken.editor().set_edge_args(edge, &[c]);
        assert!(
            verify(&broken, &target)
                .unwrap_err()
                .to_string()
                .contains("type mismatch")
        );
        let mut broken = f.clone();
        broken.editor().replace(a).copy(x, x);
        assert!(
            verify(&broken, &target)
                .unwrap_err()
                .to_string()
                .contains("does not dominate")
        );
    }
}
