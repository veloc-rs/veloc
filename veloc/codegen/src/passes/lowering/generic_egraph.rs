//! Generic pre-isel e-graph combine.
//!
//! 第一版只处理“单 basic block 内、纯整数、同一 opcode 的可交换/可结合树”。
//! 这类规范化属于 generic MIR 规范化，应该发生在指令选择之前。

use crate::mir::{
    GenericOpcode, InstId, MachineFunction, MachineInst, MachineOpcode, MachineOperand, Reg,
    UseDefChain, Writable,
};
use crate::pipeline::FunctionAnalysisCtx;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use egg::{
    define_language, rewrite as rw, CostFunction, Extractor, Id, RecExpr, Rewrite, Runner, Symbol,
};
use hashbrown::{HashMap, HashSet};

#[derive(Debug, Clone)]
struct RewritePlan {
    opcode: GenericOpcode,
    result: Reg,
    canonical_leaves: Vec<Reg>,
    covered: Vec<InstId>,
}

define_language! {
    enum AssocExpr {
        Symbol(Symbol),
        "add" = Add([Id; 2]),
        "mul" = Mul([Id; 2]),
        "and" = And([Id; 2]),
        "or" = Or([Id; 2]),
        "xor" = Xor([Id; 2]),
    }
}

#[derive(Default)]
struct CanonicalCost;

impl CostFunction<AssocExpr> for CanonicalCost {
    type Cost = (usize, String);

    fn cost<C>(&mut self, enode: &AssocExpr, mut child_cost: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        match enode {
            AssocExpr::Symbol(sym) => (1, sym.to_string()),
            AssocExpr::Add([a, b]) => merge_cost("add", child_cost(*a), child_cost(*b)),
            AssocExpr::Mul([a, b]) => merge_cost("mul", child_cost(*a), child_cost(*b)),
            AssocExpr::And([a, b]) => merge_cost("and", child_cost(*a), child_cost(*b)),
            AssocExpr::Or([a, b]) => merge_cost("or", child_cost(*a), child_cost(*b)),
            AssocExpr::Xor([a, b]) => merge_cost("xor", child_cost(*a), child_cost(*b)),
        }
    }
}

#[inline]
fn merge_cost(op: &str, left: (usize, String), right: (usize, String)) -> (usize, String) {
    (
        1 + left.0 + right.0,
        format!("({op} {} {})", left.1, right.1),
    )
}

#[inline]
fn is_egraph_supported_opcode(opcode: GenericOpcode) -> bool {
    matches!(
        opcode,
        GenericOpcode::G_ADD
            | GenericOpcode::G_MUL
            | GenericOpcode::G_AND
            | GenericOpcode::G_OR
            | GenericOpcode::G_XOR
    )
}

#[inline]
fn expr_node_for_opcode(opcode: GenericOpcode, lhs: Id, rhs: Id) -> AssocExpr {
    match opcode {
        GenericOpcode::G_ADD => AssocExpr::Add([lhs, rhs]),
        GenericOpcode::G_MUL => AssocExpr::Mul([lhs, rhs]),
        GenericOpcode::G_AND => AssocExpr::And([lhs, rhs]),
        GenericOpcode::G_OR => AssocExpr::Or([lhs, rhs]),
        GenericOpcode::G_XOR => AssocExpr::Xor([lhs, rhs]),
        _ => unreachable!("unsupported opcode for pre-isel e-graph combine"),
    }
}

fn rewrites_for_opcode(opcode: GenericOpcode) -> Vec<Rewrite<AssocExpr, ()>> {
    match opcode {
        GenericOpcode::G_ADD => vec![
            rw!("add-comm"; "(add ?a ?b)" => "(add ?b ?a)"),
            rw!("add-assoc-l"; "(add ?a (add ?b ?c))" => "(add (add ?a ?b) ?c)"),
            rw!("add-assoc-r"; "(add (add ?a ?b) ?c)" => "(add ?a (add ?b ?c))"),
        ],
        GenericOpcode::G_MUL => vec![
            rw!("mul-comm"; "(mul ?a ?b)" => "(mul ?b ?a)"),
            rw!("mul-assoc-l"; "(mul ?a (mul ?b ?c))" => "(mul (mul ?a ?b) ?c)"),
            rw!("mul-assoc-r"; "(mul (mul ?a ?b) ?c)" => "(mul ?a (mul ?b ?c))"),
        ],
        GenericOpcode::G_AND => vec![
            rw!("and-comm"; "(and ?a ?b)" => "(and ?b ?a)"),
            rw!("and-assoc-l"; "(and ?a (and ?b ?c))" => "(and (and ?a ?b) ?c)"),
            rw!("and-assoc-r"; "(and (and ?a ?b) ?c)" => "(and ?a (and ?b ?c))"),
        ],
        GenericOpcode::G_OR => vec![
            rw!("or-comm"; "(or ?a ?b)" => "(or ?b ?a)"),
            rw!("or-assoc-l"; "(or ?a (or ?b ?c))" => "(or (or ?a ?b) ?c)"),
            rw!("or-assoc-r"; "(or (or ?a ?b) ?c)" => "(or ?a (or ?b ?c))"),
        ],
        GenericOpcode::G_XOR => vec![
            rw!("xor-comm"; "(xor ?a ?b)" => "(xor ?b ?a)"),
            rw!("xor-assoc-l"; "(xor ?a (xor ?b ?c))" => "(xor (xor ?a ?b) ?c)"),
            rw!("xor-assoc-r"; "(xor (xor ?a ?b) ?c)" => "(xor ?a (xor ?b ?c))"),
        ],
        _ => Vec::new(),
    }
}

#[inline]
fn symbol_for_reg(reg: Reg) -> String {
    format!("r{}", reg.0)
}

fn flatten_leaf_order(
    expr: &RecExpr<AssocExpr>,
    root: Id,
    symbols: &HashMap<String, Reg>,
    out: &mut Vec<Reg>,
) {
    match &expr[root] {
        AssocExpr::Symbol(sym) => {
            if let Some(reg) = symbols.get(&sym.to_string()) {
                out.push(*reg);
            }
        }
        AssocExpr::Add([a, b])
        | AssocExpr::Mul([a, b])
        | AssocExpr::And([a, b])
        | AssocExpr::Or([a, b])
        | AssocExpr::Xor([a, b]) => {
            flatten_leaf_order(expr, *a, symbols, out);
            flatten_leaf_order(expr, *b, symbols, out);
        }
    }
}

#[inline]
fn generic_binary_of_opcode(inst: &MachineInst, opcode: GenericOpcode) -> Option<(Reg, Reg, Reg)> {
    match inst.opcode {
        MachineOpcode::Generic(op) if op == opcode => {}
        _ => return None,
    }

    if inst.operands.len() != 3 {
        return None;
    }

    let def = match inst.operands[0] {
        MachineOperand::Def(w) => w.to_reg(),
        _ => return None,
    };
    let lhs = match inst.operands[1] {
        MachineOperand::Use(r) => r,
        _ => return None,
    };
    let rhs = match inst.operands[2] {
        MachineOperand::Use(r) => r,
        _ => return None,
    };

    Some((def, lhs, rhs))
}

fn add_tree_from_reg(
    mfunc: &MachineFunction,
    use_def: &UseDefChain,
    block_set: &HashSet<InstId>,
    parent_inst: InstId,
    opcode: GenericOpcode,
    reg: Reg,
    expr: &mut RecExpr<AssocExpr>,
    leaves: &mut Vec<Reg>,
    symbols: &mut HashMap<String, Reg>,
    covered: &mut Vec<InstId>,
) -> Id {
    if let Some(def_inst) = use_def.single_def_of(reg) {
        if block_set.contains(&def_inst) && use_def.is_single_use_by(reg, parent_inst) {
            if let Some((_, lhs, rhs)) = generic_binary_of_opcode(&mfunc.dfg[def_inst], opcode) {
                covered.push(def_inst);
                let lhs_id = add_tree_from_reg(
                    mfunc, use_def, block_set, def_inst, opcode, lhs, expr, leaves, symbols,
                    covered,
                );
                let rhs_id = add_tree_from_reg(
                    mfunc, use_def, block_set, def_inst, opcode, rhs, expr, leaves, symbols,
                    covered,
                );
                return expr.add(expr_node_for_opcode(opcode, lhs_id, rhs_id));
            }
        }
    }

    let sym = symbol_for_reg(reg);
    leaves.push(reg);
    symbols.insert(sym.clone(), reg);
    expr.add(AssocExpr::Symbol(sym.into()))
}

fn build_rewrite_plan(
    mfunc: &MachineFunction,
    use_def: &UseDefChain,
    block_set: &HashSet<InstId>,
    root: InstId,
) -> Option<RewritePlan> {
    let inst = &mfunc.dfg[root];
    let MachineOpcode::Generic(opcode) = inst.opcode else {
        return None;
    };
    if !is_egraph_supported_opcode(opcode) {
        return None;
    }

    let (result, lhs, rhs) = generic_binary_of_opcode(inst, opcode)?;

    let mut expr = RecExpr::default();
    let mut current_leaves = Vec::new();
    let mut symbols = HashMap::new();
    let mut covered = vec![root];

    let lhs_id = add_tree_from_reg(
        mfunc,
        use_def,
        block_set,
        root,
        opcode,
        lhs,
        &mut expr,
        &mut current_leaves,
        &mut symbols,
        &mut covered,
    );
    let rhs_id = add_tree_from_reg(
        mfunc,
        use_def,
        block_set,
        root,
        opcode,
        rhs,
        &mut expr,
        &mut current_leaves,
        &mut symbols,
        &mut covered,
    );
    expr.add(expr_node_for_opcode(opcode, lhs_id, rhs_id));

    if current_leaves.len() < 2 {
        return None;
    }

    let rewrites = rewrites_for_opcode(opcode);
    let runner = Runner::default()
        .with_expr(&expr)
        .with_iter_limit(8)
        .with_node_limit(5_000)
        .run(&rewrites);
    let root_id = runner.roots[0];
    let extractor = Extractor::new(&runner.egraph, CanonicalCost);
    let (_, best) = extractor.find_best(root_id);

    let best_root = Id::from(best.as_ref().len() - 1);
    let mut canonical_leaves = Vec::new();
    flatten_leaf_order(&best, best_root, &symbols, &mut canonical_leaves);

    if canonical_leaves == current_leaves {
        return None;
    }

    Some(RewritePlan {
        opcode,
        result,
        canonical_leaves,
        covered,
    })
}

fn apply_rewrite_plan(
    mfunc: &mut MachineFunction,
    new_insts: &mut Vec<InstId>,
    plan: &RewritePlan,
) -> bool {
    if plan.canonical_leaves.len() < 2 || !plan.result.is_vreg() {
        return false;
    }

    let result_ty = mfunc.vreg_data(plan.result).ty.clone();
    let mut acc = plan.canonical_leaves[0];

    for (index, rhs) in plan.canonical_leaves.iter().copied().enumerate().skip(1) {
        let is_last = index + 1 == plan.canonical_leaves.len();
        let def = if is_last {
            plan.result
        } else {
            mfunc.alloc_vreg(result_ty.clone())
        };

        let inst =
            MachineInst::build_binary(MachineOpcode::Generic(plan.opcode), Writable(def), acc, rhs);
        let inst_id = mfunc.alloc_inst(inst);
        new_insts.push(inst_id);
        acc = def;
    }

    true
}

pub(crate) fn run_generic_pre_isel_egraph_combine(
    mfunc: &mut MachineFunction,
    analyses: &mut FunctionAnalysisCtx,
) -> usize {
    let mut total_changed = 0usize;
    loop {
        let mut changed = 0usize;
        for block_idx in 0..mfunc.num_blocks() {
            let use_def = analyses.use_def(mfunc).clone();
            changed +=
                rewrite_block_assoc_commutative_trees_with_use_def(mfunc, block_idx, &use_def);
        }
        if changed == 0 {
            break;
        }
        total_changed += changed;
        analyses.apply(
            crate::pipeline::ChangeSet::INST_SEMANTICS | crate::pipeline::ChangeSet::INST_OPERANDS,
        );
    }
    total_changed
}

fn rewrite_block_assoc_commutative_trees_with_use_def(
    mfunc: &mut MachineFunction,
    block_idx: usize,
    use_def: &UseDefChain,
) -> usize {
    let inst_ids = mfunc.block_insts(block_idx).to_vec();
    let block_set: HashSet<_> = inst_ids.iter().copied().collect();

    let mut planned_roots = HashMap::<InstId, RewritePlan>::new();
    let mut covered = HashSet::<InstId>::new();

    for &inst_id in inst_ids.iter().rev() {
        if covered.contains(&inst_id) {
            continue;
        }
        let Some(plan) = build_rewrite_plan(mfunc, use_def, &block_set, inst_id) else {
            continue;
        };
        if plan.covered.iter().any(|id| covered.contains(id)) {
            continue;
        }
        for covered_id in &plan.covered {
            covered.insert(*covered_id);
        }
        planned_roots.insert(inst_id, plan);
    }

    if planned_roots.is_empty() {
        return 0;
    }

    let mut changed = 0usize;
    mfunc
        .rewrite_block(block_idx, |cursor| {
            let inst_id = cursor.current_inst_id();
            if let Some(plan) = planned_roots.get(&inst_id) {
                let mut rewritten = Vec::new();
                if apply_rewrite_plan(cursor.mfunc_mut(), &mut rewritten, plan) {
                    changed += 1;
                } else {
                    rewritten.push(inst_id);
                }
                cursor.remove_current();
                for new_id in rewritten {
                    cursor.emit_existing_before(new_id);
                }
                return Ok::<(), ()>(());
            }

            if !covered.contains(&inst_id) {
                cursor.keep_current();
            } else {
                cursor.remove_current();
            }

            Ok(())
        })
        .expect("generic egraph block rewriting should not fail");

    changed
}

#[cfg(test)]
mod tests {
    use super::run_generic_pre_isel_egraph_combine;
    use crate::mir::{
        GenericOpcode, MachineBlock, MachineFunction, MachineInst, MachineOpcode, MachineOperand,
        Writable,
    };
    use crate::pipeline::FunctionAnalysisCtx;
    use veloc_ir::{Block, Type};

    fn new_test_function() -> MachineFunction {
        let mut mfunc = MachineFunction::new("test".into());
        mfunc.blocks.push(MachineBlock::new(Block::from_u32(0)));
        mfunc
    }

    #[test]
    fn canonicalizes_single_block_add_tree() {
        let mut mfunc = new_test_function();
        let mut analyses = FunctionAnalysisCtx::default();
        let a = mfunc.alloc_vreg(Type::I64);
        let b = mfunc.alloc_vreg(Type::I64);
        let c = mfunc.alloc_vreg(Type::I64);
        let t0 = mfunc.alloc_vreg(Type::I64);
        let out = mfunc.alloc_vreg(Type::I64);

        let add0 = mfunc.alloc_inst(MachineInst::build_binary(
            MachineOpcode::Generic(GenericOpcode::G_ADD),
            Writable(t0),
            b,
            a,
        ));
        let add1 = mfunc.alloc_inst(MachineInst::build_binary(
            MachineOpcode::Generic(GenericOpcode::G_ADD),
            Writable(out),
            c,
            t0,
        ));

        mfunc.append_inst_id_to_block(0, add0);
        mfunc.append_inst_id_to_block(0, add1);

        run_generic_pre_isel_egraph_combine(&mut mfunc, &mut analyses);

        let block_insts = mfunc.block_insts(0);
        assert_eq!(block_insts.len(), 2);

        let first = &mfunc.dfg[block_insts[0]];
        let second = &mfunc.dfg[block_insts[1]];

        assert!(matches!(
            first.opcode,
            MachineOpcode::Generic(GenericOpcode::G_ADD)
        ));
        assert!(matches!(
            second.opcode,
            MachineOpcode::Generic(GenericOpcode::G_ADD)
        ));
        assert!(matches!(first.operands[1], MachineOperand::Use(reg) if reg == a));
        assert!(matches!(first.operands[2], MachineOperand::Use(reg) if reg == b));
    }
}
