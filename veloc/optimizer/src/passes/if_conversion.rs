//! Convert cheap conditional computations to boolean expressions and selects.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use std::collections::HashMap;
use veloc_analyzer::AnalysisManager;
use veloc_mir::{
    FuncBody, Inst, InstView, Opcode, ScalarConst, Successor, SuccessorData, Type, Value,
};
pub struct IfConversionPass;
impl FunctionPass for IfConversionPass {
    fn reuse_key(&self) -> Option<core::any::TypeId> {
        Some(core::any::TypeId::of::<Self>())
    }
    fn name(&self) -> &'static str {
        "IfConversionPass"
    }
    fn run(&self, am: &mut AnalysisManager<'_>, _: &OptConfig, metrics: &Profile) -> PassOutcome {
        let f = am.function_mut();
        let blocks: Vec<_> = f.layout().block_order().collect();
        let mut conditions = 0;
        let mut diamonds = 0;
        for block in blocks {
            let Some(last) = f.layout().last_inst(block) else {
                continue;
            };
            if common_destination(f, last) {
                conditions += 1;
            }
            if diamond(f, last) {
                diamonds += 1;
            }
        }
        metrics.count("if_conversion.conditions", conditions);
        metrics.count("if_conversion.diamonds", diamonds);
        if conditions + diamonds == 0 {
            PassOutcome::Unchanged
        } else {
            PassOutcome::Changed
        }
    }
}

/// Merge two branches sharing a complete edge. The budget allows one extra
/// comparison to execute unconditionally; algebraic simplification follows in
/// ExpressionPass and does not decide whether speculation is worthwhile.
fn common_destination(f: &mut FuncBody, branch: Inst) -> bool {
    let InstView::Br {
        condition,
        then_dest,
        else_dest,
    } = f.dfg().inst(branch)
    else {
        return false;
    };
    let edges = [
        SuccessorData::new(then_dest.block, then_dest.args),
        SuccessorData::new(else_dest.block, else_dest.args),
    ];
    let parent = f.layout().inst_block(branch).unwrap();
    for common_index in 0..2 {
        let common = &edges[common_index];
        let next = &edges[1 - common_index];
        if next.block == parent
            || next.block == common.block
            || !next.args.is_empty()
            || f.cfg().preds(next.block) != [parent]
        {
            continue;
        }
        let (compare, tail) = {
            let mut body = f.layout().block_insts(next.block);
            let (Some(compare), Some(tail), None) = (body.next(), body.next(), body.next()) else {
                continue;
            };
            (compare, tail)
        };
        let view = f.dfg().inst(compare);
        if !matches!(
            view,
            InstView::IntCompare { .. } | InstView::FloatCompare { .. }
        ) || !view.can_speculate()
        {
            continue;
        }
        let InstView::Br {
            condition: other,
            then_dest,
            else_dest,
        } = f.dfg().inst(tail)
        else {
            continue;
        };
        if f.dfg().value_inst(other) != Some(compare)
            || f.dfg().uses(other).any(|site| site.inst() != tail)
        {
            continue;
        }
        let (other_index, miss) = if common.as_view() == then_dest {
            (0, else_dest)
        } else if common.as_view() == else_dest {
            (1, then_dest)
        } else {
            continue;
        };
        let miss = SuccessorData::new(miss.block, miss.args);
        // The sole predecessor and absence of block parameters make the
        // comparison's inputs available here. Moving keeps any edge uses valid.
        f.edit().move_before(compare, branch);
        let mut rhs = other;
        if common_index != other_index {
            let yes = f.edit().constant(ScalarConst::from(true).into());
            let inverted = f.edit().insert_before(
                branch,
                |w| w.binary(Opcode::IXor, [rhs, yes]),
                &[Type::BOOL],
            );
            rhs = f.dfg().first_result(inverted).unwrap();
        }
        let opcode = if common_index == 0 {
            Opcode::IOr
        } else {
            Opcode::IAnd
        };
        let combined = f.edit().insert_before(
            branch,
            |w| w.binary(opcode, [condition, rhs]),
            &[Type::BOOL],
        );
        let condition = f.dfg().first_result(combined).unwrap();
        let (yes, no) = if common_index == 0 {
            (common, &miss)
        } else {
            (&miss, common)
        };
        f.edit()
            .replace_inst(branch, |w| w.br(condition, yes.as_view(), no.as_view()));
        return true;
    }
    false
}

fn diamond(f: &mut FuncBody, branch: Inst) -> bool {
    let InstView::Br {
        condition,
        then_dest,
        else_dest,
    } = f.dfg().inst(branch)
    else {
        return false;
    };
    let condition = condition;
    let yes = SuccessorData::new(then_dest.block, then_dest.args);
    let no = SuccessorData::new(else_dest.block, else_dest.args);
    let parent = f.layout().inst_block(branch).unwrap();
    // Either both edges already meet, or each arm is a small pure block with
    // this branch as its only predecessor and a jump to a common merge.
    let arms = if yes.block == no.block {
        None
    } else {
        let arm = |edge: &SuccessorData| {
            if edge.block == parent || f.cfg().preds(edge.block) != [parent] {
                return None;
            }
            let insts: Vec<_> = f.layout().block_insts(edge.block).collect();
            let (&last, body) = insts.split_last()?;
            let InstView::Jump { dest } = f.dfg().inst(last) else {
                return None;
            };
            if body.len() > 3 || !body.iter().all(|&i| f.dfg().inst(i).can_speculate()) {
                return None;
            }
            Some((body.to_vec(), SuccessorData::new(dest.block, dest.args)))
        };
        let arms = match (arm(&yes), arm(&no)) {
            (Some(a), Some(b))
                if a.1.block == b.1.block
                    && ![parent, yes.block, no.block].contains(&a.1.block) =>
            {
                vec![a, b]
            }
            (Some(a), _) if a.1.block == no.block => vec![a, (Vec::new(), no.clone())],
            (_, Some(b)) if b.1.block == yes.block => vec![(Vec::new(), yes.clone()), b],
            _ => return false,
        };
        Some(arms)
    };
    let mut edges = vec![yes, no];
    if let Some(arms) = arms {
        for (edge, (insts, dest)) in edges.iter_mut().zip(arms) {
            if edge.block == dest.block {
                continue;
            }
            let mut map: HashMap<Value, Value> = f
                .dfg()
                .block_params(edge.block)
                .iter()
                .copied()
                .zip(edge.args.iter().copied())
                .collect();
            for inst in insts {
                let args: Vec<_> = f
                    .dfg()
                    .operands(inst)
                    .iter()
                    .map(|v| map.get(v).copied().unwrap_or(*v))
                    .collect();
                let results = f.dfg().inst_results(inst).to_vec();
                let types: Vec<_> = results.iter().map(|&v| f.dfg().value_type(v)).collect();
                let new =
                    f.edit()
                        .insert_before(branch, |w| w.copy_with_operands(inst, &args), &types);
                for (old, &new) in results.into_iter().zip(f.dfg().inst_results(new)) {
                    map.insert(old, new);
                }
            }
            let args: Vec<_> = dest
                .args
                .iter()
                .map(|v| map.get(v).copied().unwrap_or(*v))
                .collect();
            *edge = SuccessorData::new(dest.block, &args);
        }
    }
    let mut args = Vec::new();
    for (&a, &b) in edges[0].args.iter().zip(&edges[1].args) {
        if a == b {
            args.push(a);
        } else {
            let ty = f.dfg().value_type(a);
            let inst = f.edit().insert_before(
                branch,
                |w| w.ternary(veloc_mir::Opcode::Select, [condition, a, b]),
                &[ty],
            );
            args.push(f.dfg().first_result(inst).unwrap());
        }
    }
    let dest = edges[0].block;
    f.edit().replace_inst(branch, |w| {
        w.jump(Successor {
            block: dest,
            args: &args,
        })
    });
    true
}
