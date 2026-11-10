//! Convert cheap speculatable diamonds to selects.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use std::collections::HashMap;
use veloc_analyzer::AnalysisManager;
use veloc_mir::{FuncBody, Inst, InstView, Successor, SuccessorData, Value};
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
        let mut changed = 0;
        for block in blocks {
            let Some(last) = f.layout().last_inst(block) else {
                continue;
            };
            if diamond(f, last) {
                changed += 1;
            }
        }
        metrics.count("if_conversion.diamonds", changed);
        if changed == 0 {
            PassOutcome::Unchanged
        } else {
            PassOutcome::Changed
        }
    }
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
