//! Generated scalar evaluation and local rewrites; MIR owns only representation.
use alloc::{borrow::ToOwned, vec::Vec};
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
                edit.body().dfg().value_type(results[0]),
                edit.body().dfg().value_type(value)
            );
            edit.replace_results(inst, &[value]);
        }
        Replacement::Constants(constants) => {
            assert_eq!(results.len(), constants.len());
            let mut replacements = SmallVec::<[Value; 4]>::new();
            for (value, constant) in results.into_iter().zip(constants) {
                assert_eq!(edit.body().dfg().value_type(value), constant.ty());
                let next = edit.insert_before(
                    inst,
                    |writer| writer.scalar_const(constant),
                    &[constant.ty()],
                );
                let replacement = edit.body().dfg().first_result(next).unwrap();
                let name = edit.body().dfg().value_name(value).to_owned();
                edit.set_value_name(replacement, &name);
                replacements.push(replacement);
            }
            edit.replace_results(inst, &replacements);
        }
    }
    Some(affected)
}
