//! Constant Folding Pass
//!
//! This pass folds constant expressions into their results.
//! For example: `iconst 1 + iconst 2` -> `iconst 3`

use crate::{FunctionPass, Metrics, OptConfig, PreservedAnalyses};
use smallvec::SmallVec;
use veloc_analyzer::AnalysisManager;
use veloc_mir::constant::Constant;
use veloc_mir::function::Function;
use veloc_mir::inst::{Inst, InstructionData};
use veloc_mir::{Type, Value};

const CONSTANT_FOLDING: &str = "constant_folding";

pub struct ConstantFoldingPass;

impl FunctionPass for ConstantFoldingPass {
    fn name(&self) -> &str {
        "ConstantFoldingPass"
    }

    fn run(
        &self,
        func: &mut Function,
        am: &mut AnalysisManager,
        config: &OptConfig,
        metrics: &mut Metrics,
    ) -> PreservedAnalyses {
        let changed =
            run_constant_folding(func, am, config.is_debug_enabled(CONSTANT_FOLDING), metrics);
        if changed {
            PreservedAnalyses::none()
        } else {
            PreservedAnalyses::all()
        }
    }
}

pub fn run_constant_folding(
    func: &mut Function,
    am: &mut AnalysisManager,
    print_folded: bool,
    metrics: &mut Metrics,
) -> bool {
    let mut changed = false;
    let mut fold_count = 0u64;

    loop {
        let mut pass_changed = false;

        // 获取指令快照进行遍历。
        let insts: Vec<_> = func
            .layout
            .block_order
            .iter()
            .flat_map(|&block| {
                func.layout.blocks[block]
                    .insts
                    .iter()
                    .map(move |&inst| (block, inst))
            })
            .collect();

        for (block, inst) in insts {
            // 检查指令是否有效（之前的 fold 可能已经将其变为 Nop）
            if matches!(func.dfg.instructions[inst], InstructionData::Nop) {
                continue;
            }

            if let Some(folded_const) = try_fold_instruction(func, inst) {
                if print_folded {
                    log::info!(
                        "[ConstantFolding] Folding instruction {} results in {:?}",
                        inst,
                        folded_const
                    );
                }

                let results = SmallVec::<[Value; 2]>::from_slice(func.dfg.inst_results(inst));
                assert_eq!(results.len(), folded_const.len());
                assert!(!results.is_empty());
                am.use_def_mut(func).detach_inst(func, inst);
                let mut added = SmallVec::<[Inst; 1]>::new();
                for (index, (value, constant)) in results.into_iter().zip(folded_const).enumerate()
                {
                    let definition = if index == 0 {
                        func.dfg.replace_inst(inst, constant.into());
                        inst
                    } else {
                        let next = func.dfg.instructions.push(constant.into());
                        added.push(next);
                        next
                    };
                    func.dfg.inst_results[definition] = func.dfg.make_value_list(&[value]);
                    func.dfg.values[value].def = veloc_mir::ValueDef::Inst(definition);
                }
                if !added.is_empty() {
                    let insts = &mut func.layout.blocks[block].insts;
                    let position = insts
                        .iter()
                        .position(|&i| i == inst)
                        .expect("instruction is in its block");
                    insts.splice(position + 1..position + 1, added);
                }
                func.bump_revision();
                am.update_use_def_revision(func);
                pass_changed = true;
                fold_count += 1;
            }
        }

        if !pass_changed {
            break;
        }
        changed = true;
    }

    if changed {
        metrics.add("constant_folding.folded_insts", fold_count);
        compact_layout(func);
    }

    changed
}

/// 从布局中移除 Nop 指令
fn compact_layout(func: &mut Function) {
    for block in &func.layout.block_order.clone() {
        func.layout.blocks[*block]
            .insts
            .retain(|&inst| !matches!(func.dfg.instructions[inst], InstructionData::Nop));
    }
}

/// 尝试折叠单个指令
fn try_fold_instruction(func: &Function, inst: Inst) -> Option<Vec<Constant>> {
    let idata = &func.dfg.instructions[inst];
    let results = func.dfg.inst_results(inst);

    idata.opcode().spec().semantics?;
    let mut args = Some(SmallVec::<[Constant; 4]>::new());
    idata.visit_type_operands(&func.dfg, |v| {
        if let Some(constants) = &mut args {
            match func.dfg.as_const(v) {
                Some(constant) => constants.push(constant),
                None => args = None,
            }
        }
    });
    let args = args?;
    let types = results
        .iter()
        .map(|&v| func.dfg.value_type(v))
        .collect::<SmallVec<[Type; 2]>>();
    Constant::evaluate(idata.opcode(), &args, &types, &idata.semantic_properties())
}
