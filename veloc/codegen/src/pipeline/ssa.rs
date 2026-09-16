//! Validation at pass boundaries, independent of instruction construction.
use super::FunctionAnalysisCtx;
use crate::{Error, Result, target::arch::TargetMachine};
use alloc::{format, vec::Vec};
use hashbrown::{HashMap, HashSet};
#[cfg(test)]
use veloc_lir::InstBuild;
use veloc_lir::InstRead;
use veloc_lir::{ControlFlow, InstExtra, InstField, MachineFunction, Reg};
use veloc_mir::Block;

pub fn verify<S>(f: &MachineFunction<S>, target: &dyn TargetMachine) -> Result<()> {
    let fail = |message| Error::codegen(format!("machine SSA in {}: {message}", f.name));
    f.check_refs().map_err(|e| fail(e.into()))?;
    if f.is_regallocated {
        for block in &f.blocks {
            for &id in &block.insts {
                target.validate_instruction(&f.inst(id), true)?;
            }
        }
        return Ok(());
    }
    let mut defs = HashMap::new();
    let mut blocks = HashSet::new();
    let mut instructions = HashSet::new();
    for block in &f.blocks {
        if !blocks.insert(block.id) {
            return Err(fail(format!("duplicate block {}", block.id)));
        }
        let mut define = |reg: Reg, pos| -> Result<()> {
            let Some(value) = reg.as_vreg() else {
                return Ok(());
            };
            if f.vregs.get(value).is_none() {
                return Err(fail(format!("unknown value {reg:?}")));
            }
            if let Some(previous) = defs.insert(reg, (block.id, pos)) {
                return Err(fail(format!(
                    "multiple definitions of {reg:?}: {previous:?} and ({:?}, {pos})",
                    block.id
                )));
            }
            Ok(())
        };
        for &param in &block.params {
            if param.is_preg() {
                return Err(fail("physical block parameter before allocation".into()));
            }
            define(param, 0)?;
        }
        let mut transferred = false;
        for (index, &id) in block.insts.iter().enumerate() {
            if !instructions.insert(id) {
                return Err(fail(format!("instruction {id:?} occurs twice in layout")));
            }
            let inst = f.inst(id);
            if inst.is_generic() {
                inst.validate()?;
            } else {
                target.validate_instruction(&inst, false)?;
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
                    if index + 1 != block.insts.len() =>
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
    let mut pending: Vec<_> = f.blocks.first().map(|b| b.id).into_iter().collect();
    while let Some(block) = pending.pop() {
        if reachable.insert(block) {
            pending.extend_from_slice(cfg.succs(block));
        }
    }
    let dom = analyses.dominators(f, target);
    for (block_index, block) in f.blocks.iter().enumerate() {
        let falls_through = block.insts.last().is_none_or(|&id| {
            matches!(
                target.control_flow(&f.inst(id)),
                ControlFlow::Next | ControlFlow::Call | ControlFlow::Branch
            )
        });
        if falls_through
            && f.blocks
                .get(block_index + 1)
                .is_some_and(|next| !next.params.is_empty())
        {
            return Err(fail(format!(
                "fallthrough from {} cannot supply block parameters",
                block.id
            )));
        }
        for (index, &id) in block.insts.iter().enumerate() {
            let inst = f.inst(id);
            for reg in inst.uses().filter(|r| r.is_vreg()) {
                let &(owner, pos) = defs
                    .get(&reg)
                    .ok_or_else(|| fail(format!("undefined {reg:?} used by {id:?}")))?;
                if (owner == block.id && pos >= index + 1)
                    || (owner != block.id
                        && reachable.contains(&block.id)
                        && !dom.dominates(owner, block.id))
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
                Some(InstExtra::Branch(info)) => {
                    let [target] = targets.as_slice() else {
                        return Err(fail("invalid single-edge shape".into()));
                    };
                    check_edge(*target, &info.args)?;
                }
                Some(InstExtra::BranchCond(info)) => {
                    let [yes, no] = targets.as_slice() else {
                        return Err(fail("invalid conditional-edge shape".into()));
                    };
                    check_edge(*yes, &info.then_args)?;
                    check_edge(*no, &info.else_args)?;
                }
                Some(InstExtra::BrTable(info)) => {
                    for edge in &info.targets {
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
    use veloc_lir::{BranchCondInfo, BranchInfo, Type, Writable, stages::RawLir};

    #[test]
    fn verifies_definitions_dominance_and_edge_contracts() {
        let target =
            crate::target::x86_64::X86_64TargetMachine::new(crate::TargetConfig::default());
        let mut f = MachineFunction::<RawLir>::new("diamond".into());
        let blocks: Vec<_> = (0..4).map(|_| f.create_synthetic_block()).collect();
        let x = f.alloc_vreg(Type::I64);
        let y = f.alloc_vreg(Type::I64);
        let p = f.alloc_vreg(Type::I64);
        let c = f.alloc_vreg(Type::BOOL);
        let r = f.alloc_vreg(Type::I64);
        f.blocks[3].params.push(p);
        let a = f.writer().constant(Writable(x), 1);
        f.append_inst_id_to_block(0, a);
        let b = f.writer().constant(Writable(c), 1);
        f.append_inst_id_to_block(0, b);
        let branch = f.writer().brcond(c, blocks[1], blocks[2]);
        f.set_inst_extra(
            branch,
            InstExtra::BranchCond(BranchCondInfo {
                then_args: Default::default(),
                else_args: Default::default(),
            }),
        );
        f.append_inst_id_to_block(0, branch);
        let left = f.writer().br(blocks[3]);
        f.set_inst_extra(
            left,
            InstExtra::Branch(BranchInfo {
                args: smallvec::smallvec![x],
            }),
        );
        f.append_inst_id_to_block(1, left);
        let def_y = f.writer().constant(Writable(y), 2);
        f.append_inst_id_to_block(2, def_y);
        let right = f.writer().br(blocks[3]);
        f.set_inst_extra(
            right,
            InstExtra::Branch(BranchInfo {
                args: smallvec::smallvec![y],
            }),
        );
        f.append_inst_id_to_block(2, right);
        let copy = f.writer().copy(Writable(r), p);
        f.append_inst_id_to_block(3, copy);
        let ret = f.writer().ret(&[r]);
        f.append_inst_id_to_block(3, ret);
        verify(&f, &target).unwrap();

        let mut broken = f.clone();
        broken.rewriter(copy).copy(Writable(p), x);
        assert!(
            verify(&broken, &target)
                .unwrap_err()
                .to_string()
                .contains("multiple definitions")
        );
        let mut broken = f.clone();
        broken.set_inst_extra(
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
        broken.clear_inst_extra(left);
        assert!(
            verify(&broken, &target)
                .unwrap_err()
                .to_string()
                .contains("arguments for")
        );
        let mut broken = f.clone();
        broken.set_inst_extra(
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
        broken.rewriter(a).copy(Writable(x), x);
        assert!(
            verify(&broken, &target)
                .unwrap_err()
                .to_string()
                .contains("does not dominate")
        );
    }
}
