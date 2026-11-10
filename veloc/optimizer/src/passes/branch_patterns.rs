//! Local branch patterns; no data-flow analysis or CFG cleanup.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use veloc_analyzer::AnalysisManager;
use veloc_mir::{FuncBody, Inst, InstView, SuccessorData};
pub struct BranchPatternsPass;
impl FunctionPass for BranchPatternsPass {
    fn reuse_key(&self) -> Option<core::any::TypeId> {
        Some(core::any::TypeId::of::<Self>())
    }
    fn name(&self) -> &'static str {
        "BranchPatternsPass"
    }
    fn run(&self, am: &mut AnalysisManager<'_>, _: &OptConfig, metrics: &Profile) -> PassOutcome {
        let f = am.function_mut();
        let blocks: Vec<_> = f.layout().block_order().collect();
        let mut changed = 0;
        for block in blocks {
            let Some(last) = f.layout().last_inst(block) else {
                continue;
            };
            if crate::rewrite::rewrite_inst(f, last) {
                changed += 1;
            }
            if compare_chain(f, last) {
                changed += 1;
            }
        }
        metrics.count("branch_patterns.simplified", changed);
        if changed == 0 {
            PassOutcome::Unchanged
        } else {
            PassOutcome::Changed
        }
    }
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
