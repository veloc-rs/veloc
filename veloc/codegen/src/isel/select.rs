//! Instruction Selector - 指令选择器
//!
//! 将通用 LIR 指令转换为目标架构特定指令。
//!
//! 这是一个通用的指令选择驱动器，实际的架构特定选择逻辑
//! 通过 TargetInstructionSelector trait 委托给具体的目标后端实现。

use crate::target::TargetInstructionSelector;
use std::vec::Vec;
use veloc_lir::{InstId, MachineFunction};

fn format_select_failure_inst(mfunc: &MachineFunction, inst_id: InstId) -> std::string::String {
    use std::format;

    let inst = &mfunc.inst(inst_id);
    let operand_types = inst
        .results()
        .iter()
        .copied()
        .chain(inst.uses())
        .filter_map(|reg| {
            if reg.is_vreg() {
                Some(format!("{:?}:{:?}", reg, mfunc.vreg_data(reg).ty))
            } else {
                Some(format!("{:?}:preg", reg))
            }
        })
        .collect::<Vec<_>>();

    if operand_types.is_empty() {
        format!("{inst:?}")
    } else {
        format!("{inst:?}; operand_types=[{}]", operand_types.join(", "))
    }
}

/// 指令选择结果
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectResult {
    /// 保持原指令不变
    Keep,
    /// 原地替换为新指令（保持 InstId）
    /// 新指令已插入原指令前
    InPlace,
    /// 用多条新指令替换（新指令已插入原指令前）
    Replace,
    /// 删除该指令
    Remove,
}

/// 指令选择上下文
pub struct SelectionContext<'a> {
    pub mfunc: &'a mut MachineFunction,
    pub inst_id: InstId,
    pub selected: &'a mut Vec<InstId>,
    pub edge_transfers: &'a mut Vec<(veloc_lir::EdgeId, veloc_lir::EdgeId)>,
}

fn apply_select_result(
    mfunc: &mut MachineFunction,
    id: InstId,
    selected: &mut Vec<InstId>,
    result: SelectResult,
    edge_transfers: &mut Vec<(veloc_lir::EdgeId, veloc_lir::EdgeId)>,
) -> Result<(), crate::error::Error> {
    let mut edit = mfunc.editor();
    if matches!(result, SelectResult::Keep | SelectResult::Remove) {
        assert!(edge_transfers.is_empty());
    }
    if result == SelectResult::InPlace && selected.len() != 1 {
        return Err(crate::error::Error::select(
            edit.inst(id).opcode(),
            "InPlace expects one selected instruction",
        ));
    }
    for &(original, replacement) in edge_transfers.iter() {
        assert!(edit.inst(id).edge_ids().any(|edge| edge == original));
        assert!(
            selected
                .iter()
                .any(|&inst| edit.inst(inst).edge_ids().any(|edge| edge == replacement))
        );
    }
    for (original, replacement) in edge_transfers.drain(..) {
        edit.transfer_edge(original, replacement);
    }
    match result {
        SelectResult::Keep => {
            assert!(selected.is_empty());
        }
        SelectResult::InPlace => {
            edit.replace_inst(id, selected.pop().unwrap());
        }
        SelectResult::Replace => {
            selected.clear();
            edit.invalidate_inst(id);
        }
        SelectResult::Remove => {
            assert!(selected.is_empty());
            edit.invalidate_inst(id);
        }
    }
    Ok(())
}

/// 指令选择器
///
/// 这是 GlobalISel 的核心组件之一，负责驱动指令选择过程。
/// 实际的选择逻辑委托给 TargetInstructionSelector 实现。
pub struct InstructionSelector<'a> {
    target: &'a dyn TargetInstructionSelector,
}

impl<'a> InstructionSelector<'a> {
    /// 创建新的指令选择器
    pub fn new(target: &'a dyn TargetInstructionSelector) -> Self {
        Self { target }
    }

    /// 对所有基本块执行指令选择
    ///
    /// 与 `select` 相同，提供更清晰的命名。
    pub fn select(&self, mfunc: &mut MachineFunction) -> Result<(), crate::error::Error> {
        // 复用的临时缓冲区，避免每条指令分配
        let mut selected: Vec<InstId> = Vec::with_capacity(4);
        let mut edge_transfers = Vec::new();
        let mut block = mfunc.blocks().next_back();
        while let Some(current_block) = block {
            let previous_block = mfunc.layout().prev_block(current_block);
            let mut inst = mfunc.layout().last_inst(current_block);
            while let Some(inst_id) = inst {
                // Selection may detach this instruction or insert replacements
                // before it. Capture the original predecessor first so selected
                // instructions are not selected a second time.
                let previous_inst = mfunc.layout().prev_inst(inst_id);
                // 如果指令在之前的融合中已被标记为无效，则跳过
                if mfunc.inst(inst_id).is_invalid() || mfunc.inst(inst_id).is_call_frame() {
                    inst = previous_inst;
                    continue;
                }
                // Consumers are selected first so their generic producers remain
                // available to graph patterns. Only unused pure values disappear;
                // matching a producer does not imply ownership of all its uses.
                let inst_ref = mfunc.inst(inst_id);
                if inst_ref.is_pure_value()
                    && inst_ref
                        .results()
                        .iter()
                        .all(|reg| mfunc.uses(*reg).next().is_none())
                {
                    mfunc.editor().invalidate_inst(inst_id);
                    inst = previous_inst;
                    continue;
                }

                // 进行指令选择。由具体的后端返回选择结果
                selected.clear();
                let result = {
                    let mut ctx = SelectionContext {
                        mfunc,
                        inst_id,
                        selected: &mut selected,
                        edge_transfers: &mut edge_transfers,
                    };
                    match self.target.select_instruction(&mut ctx) {
                        Ok(result) => result,
                        Err(crate::error::Error::Select(err)) => {
                            return Err(crate::error::Error::select(
                                err.opcode.clone(),
                                std::format!(
                                    "{}; inst_id={:?}, inst={}",
                                    err.reason,
                                    inst_id,
                                    format_select_failure_inst(ctx.mfunc, inst_id)
                                ),
                            ));
                        }
                        Err(err) => return Err(err),
                    }
                };

                apply_select_result(mfunc, inst_id, &mut selected, result, &mut edge_transfers)?;
                inst = previous_inst;
            }
            block = previous_block;
        }

        Ok(())
    }
}
