//! Simplify branches, forwarding blocks and cheap speculatable diamonds.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use std::collections::{HashMap, HashSet};
use veloc_analyzer::AnalysisManager;
use veloc_mir::{FuncBody, Inst, InstView, Successor, SuccessorData, Value, function::EdgeRef};

pub struct CfgPass;
impl FunctionPass for CfgPass {
    fn reuse_key(&self) -> Option<core::any::TypeId> {
        Some(core::any::TypeId::of::<Self>())
    }
    fn name(&self) -> &'static str {
        "CfgPass"
    }
    fn run(&self, am: &mut AnalysisManager<'_>, _: &OptConfig, metrics: &Profile) -> PassOutcome {
        let f = am.function_mut();
        let mut changed = 0;
        loop {
            let before = changed;
            if f.edit().remove_unreachable() {
                changed += 1;
            }
            let blocks: Vec<_> = f.layout().block_order().collect();
            for block in blocks {
                if !f.layout().contains_block(block) {
                    continue;
                }
                let Some(last) = f.layout().last_inst(block) else {
                    continue;
                };
                if let InstView::Br {
                    condition,
                    then_dest,
                    else_dest,
                } = f.dfg().inst(last)
                    && let Some(def) = f.dfg().value_inst(condition)
                    && let InstView::Ternary {
                        opcode: veloc_mir::Opcode::Select,
                        args,
                    } = f.dfg().inst(def)
                    && let (Some(yes), Some(no)) = (
                        f.dfg().as_scalar_const(args[1]),
                        f.dfg().as_scalar_const(args[2]),
                    )
                    && yes.to_bits() != no.to_bits()
                {
                    let c = args[0];
                    let yes = yes.to_bits() != 0;
                    let a = SuccessorData::new(then_dest.block, then_dest.args);
                    let b = SuccessorData::new(else_dest.block, else_dest.args);
                    f.edit().replace_inst(last, |w| {
                        if yes {
                            w.br(c, a.as_view(), b.as_view())
                        } else {
                            w.br(c, b.as_view(), a.as_view())
                        }
                    });
                    changed += 1;
                }
                if let InstView::Br {
                    condition,
                    then_dest,
                    else_dest,
                } = f.dfg().inst(last)
                    && let Some(c) = f.dfg().as_scalar_const(condition)
                {
                    let edge = if c.to_bits() != 0 {
                        then_dest
                    } else {
                        else_dest
                    };
                    let edge = SuccessorData::new(edge.block, edge.args);
                    f.edit().replace_inst(last, |w| w.jump(edge.as_view()));
                    changed += 1;
                }
                let mut edges = Vec::new();
                f.dfg()
                    .inst(last)
                    .visit_successors(|e| edges.push(SuccessorData::new(e.block, e.args)));
                for (index, mut edge) in edges.into_iter().enumerate() {
                    let original = edge.block;
                    let mut seen = HashSet::from([block]);
                    while seen.insert(edge.block) {
                        if let Some(next) = thread_comparison(f, last, index, &edge) {
                            if seen.contains(&next.block) {
                                break;
                            }
                            edge = next;
                            continue;
                        }
                        let Some(inst) = f.layout().first_inst(edge.block) else {
                            break;
                        };
                        if f.layout().last_inst(edge.block) != Some(inst) {
                            break;
                        }
                        let dest = match f.dfg().inst(inst) {
                            InstView::Jump { dest } => dest,
                            InstView::Br {
                                condition,
                                then_dest,
                                else_dest,
                            } => {
                                let params = f.dfg().block_params(edge.block);
                                let mapped = params
                                    .iter()
                                    .position(|&p| p == condition)
                                    .map_or(condition, |i| edge.args[i]);
                                let Some(value) = f.dfg().as_scalar_const(mapped) else {
                                    break;
                                };
                                if value.to_bits() != 0 {
                                    then_dest
                                } else {
                                    else_dest
                                }
                            }
                            _ => break,
                        };
                        if f.dfg()
                            .block_params(edge.block)
                            .iter()
                            .any(|&p| f.dfg().uses(p).any(|site| site.inst() != inst))
                        {
                            break;
                        }
                        if seen.contains(&dest.block) {
                            break;
                        }
                        let params = f.dfg().block_params(edge.block);
                        let args = dest
                            .args
                            .iter()
                            .map(|v| {
                                params
                                    .iter()
                                    .position(|p| p == v)
                                    .map_or(*v, |i| edge.args[i])
                            })
                            .collect::<Vec<_>>();
                        edge = SuccessorData::new(dest.block, &args);
                    }
                    if edge.block != original {
                        f.edit().redirect_edge(
                            EdgeRef {
                                inst: last,
                                index: index as u32,
                            },
                            edge.block,
                            &edge.args,
                        );
                        changed += 1;
                    }
                }
                if diamond(f, last) {
                    changed += 1;
                }
                if compare_chain(f, last) {
                    changed += 1;
                }
            }
            if f.edit().remove_unreachable() {
                changed += 1;
            }
            let blocks: Vec<_> = f.layout().block_order().collect();
            for block in blocks {
                if f.layout().contains_block(block) {
                    while f.edit().merge_successor(block) {
                        changed += 1;
                    }
                }
            }
            if before == changed {
                break;
            }
        }
        metrics.count("cfg.simplified", changed);
        if changed == 0 {
            PassOutcome::Unchanged
        } else {
            PassOutcome::Changed
        }
    }
}

/// Bypass a comparison-only block when the incoming branch already proves
/// its outcome. Relational outcome sets avoid ad hoc signed/unsigned cases.
fn thread_comparison(
    f: &FuncBody,
    source: Inst,
    edge_index: usize,
    edge: &SuccessorData,
) -> Option<SuccessorData> {
    let InstView::Br { condition, .. } = f.dfg().inst(source) else {
        return None;
    };
    let source_cmp = f.dfg().value_inst(condition)?;
    let InstView::IntCompare {
        kind: source_kind,
        args: source_args,
    } = f.dfg().inst(source_cmp)
    else {
        return None;
    };
    let insts: Vec<_> = f.layout().block_insts(edge.block).collect();
    let [compare, branch] = insts.as_slice() else {
        return None;
    };
    let InstView::IntCompare { kind, args } = f.dfg().inst(*compare) else {
        return None;
    };
    let result = f.dfg().first_result(*compare)?;
    let InstView::Br {
        condition,
        then_dest,
        else_dest,
    } = f.dfg().inst(*branch)
    else {
        return None;
    };
    if condition != result || f.dfg().uses(result).any(|u| u.inst() != *branch) {
        return None;
    }
    let params = f.dfg().block_params(edge.block);
    if params.iter().any(|&v| {
        f.dfg()
            .uses(v)
            .any(|u| u.inst() != *compare && u.inst() != *branch)
    }) {
        return None;
    }
    let mapped = |v| {
        params
            .iter()
            .position(|&p| p == v)
            .map_or(v, |i| edge.args[i])
    };
    let args = [mapped(args[0]), mapped(args[1])];
    let kind = if args == [source_args[1], source_args[0]] {
        kind.swap()
    } else if args == *source_args {
        kind
    } else {
        return None;
    };
    let known = if edge_index == 1 {
        source_kind.complement()
    } else {
        source_kind
    };
    let dest = if known.implies(kind)? {
        then_dest
    } else {
        else_dest
    };
    // The comparison's result cannot become an argument on the new edge.
    if dest.args.contains(&result) {
        return None;
    }
    let args: Vec<_> = dest.args.iter().map(|&v| mapped(v)).collect();
    Some(SuccessorData::new(dest.block, &args))
}

/// Combine two equality tests with the same taken edge. A pair separated by
/// one bit is a masked equality; translating by the smaller literal extends
/// the same identity to any pair whose distance is a power of two.
fn compare_chain(f: &mut FuncBody, branch: Inst) -> bool {
    use veloc_mir::{Int, IntCC, Opcode, Type, TypeInfo};
    let equality = |branch| {
        let InstView::Br {
            condition,
            then_dest,
            else_dest,
        } = f.dfg().inst(branch)
        else {
            return None;
        };
        let def = f.dfg().value_inst(condition)?;
        let InstView::IntCompare { kind, args: [a, b] } = f.dfg().inst(def) else {
            return None;
        };
        let (hit, miss) = match kind {
            IntCC::Eq => (then_dest, else_dest),
            IntCC::Ne => (else_dest, then_dest),
            _ => return None,
        };
        let (value, literal) = if let Some(c) = f.dfg().as_scalar_const(*b) {
            (*a, c.to_bits())
        } else if let Some(c) = f.dfg().as_scalar_const(*a) {
            (*b, c.to_bits())
        } else {
            return None;
        };
        let ty = f.dfg().value_type(value);
        if ty.is_ptr() || !ty.is_integer() {
            return None;
        }
        Some((
            condition,
            def,
            value,
            literal,
            SuccessorData::new(hit.block, hit.args),
            SuccessorData::new(miss.block, miss.args),
        ))
    };
    let Some((_, _, value, a, hit, next)) = equality(branch) else {
        return false;
    };
    if !next.args.is_empty() || Some(next.block) == f.layout().inst_block(branch) {
        return false;
    }
    let body: Vec<_> = f.layout().block_insts(next.block).collect();
    let [compare, tail] = body.as_slice() else {
        return false;
    };
    let Some((condition, def, other, b, other_hit, miss)) = equality(*tail) else {
        return false;
    };
    if *compare != def
        || value != other
        || hit.as_view() != other_hit.as_view()
        || f.dfg().uses(condition).any(|site| site.inst() != *tail)
    {
        return false;
    }
    let (a, b) = (a.min(b), a.max(b));
    let xor = a ^ b;
    let difference = b - a;
    if !xor.is_power_of_two() && !difference.is_power_of_two() {
        return false;
    }
    let ty = f.dfg().value_type(value);
    let mut input = value;
    let (mask, expected) = if xor.is_power_of_two() {
        (!xor, a & !xor)
    } else {
        let base = f.edit().constant(Int::from_bits(ty, a).unwrap().into());
        let sub = f
            .edit()
            .insert_before(branch, |w| w.binary(Opcode::ISub, [value, base]), &[ty]);
        input = f.dfg().first_result(sub).unwrap();
        (!difference, 0)
    };
    let mask = f.edit().constant(Int::from_bits(ty, mask).unwrap().into());
    let expected = f
        .edit()
        .constant(Int::from_bits(ty, expected).unwrap().into());
    let masked = f
        .edit()
        .insert_before(branch, |w| w.binary(Opcode::IAnd, [input, mask]), &[ty]);
    let masked = f.dfg().first_result(masked).unwrap();
    let cmp = f.edit().insert_before(
        branch,
        |w| w.int_compare(IntCC::Eq, [masked, expected]),
        &[Type::BOOL],
    );
    let condition = f.dfg().first_result(cmp).unwrap();
    f.edit()
        .replace_inst(branch, |w| w.br(condition, hit.as_view(), miss.as_view()));
    true
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
