//! Generated scalar evaluation and local rewrites; MIR owns only representation.
use alloc::vec::Vec;
use smallvec::SmallVec;
use veloc_analyzer::AnalysisManager;
use veloc_mir::constant::Constant;
use veloc_mir::{Block, Function, Inst, InstructionData, IntCC, Opcode, Type, Value, ValueDef};

enum Replacement {
    Constants(Vec<Constant>),
    Value(Value),
}

include!(concat!(env!("OUT_DIR"), "/evaluation.rs"));

fn match_rule(func: &Function, inst: Inst) -> Option<Replacement> {
    let data = &func.dfg.instructions[inst];
    if can_fold(data.opcode()) {
        let mut args = Some(SmallVec::<[Constant; 4]>::new());
        data.visit_type_operands(&func.dfg, |v| {
            if let Some(constants) = &mut args {
                match func.dfg.as_const(v) {
                    Some(c) => constants.push(c),
                    None => args = None,
                }
            }
        });
        if let Some(args) = args {
            let types = func
                .dfg
                .inst_results(inst)
                .iter()
                .map(|&v| func.dfg.value_type(v))
                .collect::<SmallVec<[Type; 2]>>();
            if let Some(values) = evaluate(data.opcode(), &args, &types, &properties(data)) {
                return Some(Replacement::Constants(values));
            }
        }
    }
    if let InstructionData::Binary { opcode, args } = data {
        return algebraic(*opcode, args, &args.map(|v| func.dfg.as_const(v)));
    }
    None
}

/// Apply one generated rewrite, maintaining SSA, layout, and the use-def cache.
pub(crate) fn rewrite(
    func: &mut Function,
    block: Block,
    inst: Inst,
    am: &mut AnalysisManager,
) -> bool {
    let Some(replacement) = match_rule(func, inst) else {
        return false;
    };
    let results = SmallVec::<[Value; 2]>::from_slice(func.dfg.inst_results(inst));
    assert!(!results.is_empty());
    match replacement {
        Replacement::Value(value) => {
            assert_eq!(results.len(), 1);
            assert_eq!(func.dfg.value_type(results[0]), func.dfg.value_type(value));
            let uses = am.use_def_mut(func);
            uses.detach_inst(func, inst);
            uses.replace_all_uses_with(func, results[0], value);
            func.dfg.remove_inst(inst);
        }
        Replacement::Constants(constants) => {
            assert_eq!(results.len(), constants.len());
            am.use_def_mut(func).detach_inst(func, inst);
            let mut added = SmallVec::<[Inst; 1]>::new();
            for (index, (value, constant)) in results.into_iter().zip(constants).enumerate() {
                assert_eq!(func.dfg.value_type(value), constant.ty());
                let definition = if index == 0 {
                    func.dfg.replace_inst(inst, constant.into());
                    inst
                } else {
                    let next = func.dfg.instructions.push(constant.into());
                    added.push(next);
                    next
                };
                func.dfg.inst_results[definition] = func.dfg.make_value_list(&[value]);
                func.dfg.values[value].def = ValueDef::Inst(definition);
            }
            if !added.is_empty() {
                let insts = &mut func.layout.blocks[block].insts;
                let position = insts
                    .iter()
                    .position(|&i| i == inst)
                    .expect("instruction in block");
                insts.splice(position + 1..position + 1, added);
            }
        }
    }
    func.bump_revision();
    am.update_use_def_revision(func);
    true
}
