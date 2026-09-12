//! Generated scalar evaluation and local rewrites; MIR owns only representation.
use alloc::vec::Vec;
use smallvec::SmallVec;
use veloc_mir::constant::ScalarConst;
use veloc_mir::{Function, Inst, InstView, IntCC, Opcode, Type, Value};

enum Replacement {
    Constants(Vec<ScalarConst>),
    Value(Value),
}

include!(concat!(env!("OUT_DIR"), "/evaluation.rs"));

fn match_rule(func: &Function, inst: Inst) -> Option<Replacement> {
    let data = &func.dfg().inst(inst);
    if can_fold(data.opcode()) {
        let mut args = Some(SmallVec::<[ScalarConst; 4]>::new());
        data.visit_type_operands(|v| {
            if let Some(constants) = &mut args {
                match func.dfg().as_scalar_const(v) {
                    Some(c) => constants.push(c),
                    None => args = None,
                }
            }
        });
        if let Some(args) = args {
            let types = func
                .dfg()
                .inst_results(inst)
                .iter()
                .map(|&v| func.dfg().value_type(v))
                .collect::<SmallVec<[Type; 2]>>();
            if let Some(values) = evaluate(data.opcode(), &args, &types, &properties(data)) {
                return Some(Replacement::Constants(values));
            }
        }
    }
    if let InstView::Binary { opcode, args } = data {
        return algebraic(*opcode, args, &args.map(|v| func.dfg().as_scalar_const(v)));
    }
    None
}

/// Apply a rewrite through the MIR editor. Return the neighborhood that may
/// match differently after definition, operand or use-count changes.
pub(crate) fn rewrite(func: &mut Function, inst: Inst) -> Option<SmallVec<[Inst; 8]>> {
    let replacement = match_rule(func, inst)?;
    let results = SmallVec::<[Value; 2]>::from_slice(func.dfg().inst_results(inst));
    assert!(!results.is_empty());
    let mut affected = SmallVec::<[Inst; 8]>::new();
    let mut values = SmallVec::<[Value; 4]>::from_slice(func.dfg().operands(inst));
    values.extend_from_slice(&results);
    if let Replacement::Value(v) = &replacement {
        values.push(*v);
    }
    values.sort_unstable();
    values.dedup();
    for value in values {
        if let Some(def) = func.dfg().value_inst(value) {
            affected.push(def);
        }
        affected.extend(func.dfg().uses(value).map(|site| site.inst()));
    }
    let mut edit = func.edit();
    match replacement {
        Replacement::Value(value) => {
            assert_eq!(results.len(), 1);
            assert_eq!(
                edit.function().dfg().value_type(results[0]),
                edit.function().dfg().value_type(value)
            );
            edit.replace_all_uses(results[0], value);
            edit.erase_inst(inst);
        }
        Replacement::Constants(constants) => {
            assert_eq!(results.len(), constants.len());
            let mut previous = inst;
            for (index, (value, constant)) in results.into_iter().zip(constants).enumerate() {
                assert_eq!(edit.function().dfg().value_type(value), constant.ty());
                if index == 0 {
                    edit.replace_inst(inst, |writer| writer.scalar_const(constant));
                } else {
                    let next =
                        edit.insert_after(previous, |writer| writer.scalar_const(constant), &[]);
                    edit.move_result(value, next);
                    previous = next;
                }
            }
        }
    }
    Some(affected)
}
