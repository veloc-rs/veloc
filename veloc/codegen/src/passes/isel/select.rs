//! Instruction Selector - 指令选择器
//!
//! 将通用 LIR 指令转换为目标架构特定指令。
//!
//! 这是一个通用的指令选择驱动器，实际的架构特定选择逻辑
//! 通过 TargetInstructionSelector trait 委托给具体的目标后端实现。

use crate::target::arch::TargetInstructionSelector;
use alloc::vec::Vec;
use veloc_lir::{BlockRewriteCursor, InstId, MachineFunction};

fn format_select_failure_inst(mfunc: &MachineFunction, inst_id: InstId) -> alloc::string::String {
    use alloc::format;

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
    /// 选择结果已写入临时缓冲区
    InPlace,
    /// 用多条新指令替换（替换结果已写入输出缓冲区）
    Replace,
    /// 删除该指令
    Remove,
}

/// 指令选择上下文
pub struct SelectionContext<'a> {
    pub mfunc: &'a mut MachineFunction,
    pub inst_id: InstId,
    pub selected: &'a mut Vec<InstId>,
}

impl crate::target::arch::LoweringContext for SelectionContext<'_> {
    fn alloc_tmp(&mut self, like: veloc_lir::Reg) -> veloc_lir::Reg {
        let data = self.mfunc.vreg_data(like).clone();
        veloc_lir::Reg::new_vreg(self.mfunc.vregs.push(data).as_u32())
    }
    fn get_type(&self, vreg: veloc_lir::VReg) -> veloc_mir::Type {
        self.mfunc.vregs[vreg].ty
    }

    fn get_vreg(&self, inst: &veloc_lir::InstRef<'_>, index: usize) -> Option<veloc_lir::VReg> {
        let mut current = 0;
        for reg in inst
            .results()
            .iter()
            .copied()
            .chain(inst.inputs().iter().copied())
            .map(Some)
        {
            if let Some(r) = reg {
                if current == index {
                    if r.is_vreg() {
                        use cranelift_entity::EntityRef;
                        return Some(veloc_lir::VReg::new(r.index() as usize));
                    } else {
                        return None;
                    }
                }
                current += 1;
            }
        }
        None
    }
}

fn apply_select_result<'a>(
    cursor: &mut BlockRewriteCursor<'a>,
    selected: &mut Vec<InstId>,
    result: SelectResult,
) -> Result<(), crate::error::Error> {
    match result {
        SelectResult::Keep => {
            debug_assert!(selected.is_empty());
            cursor.keep_current();
        }
        SelectResult::InPlace => {
            let inst = selected.pop().ok_or_else(|| {
                crate::error::Error::select(
                    cursor.current_inst().opcode().clone(),
                    alloc::string::String::from("InPlace expects one selected inst"),
                )
            })?;
            debug_assert!(selected.is_empty());
            cursor.replace_current(inst);
        }
        SelectResult::Replace => {
            cursor.remove_current();
            for inst in selected.drain(..) {
                cursor.emit(inst);
            }
        }
        SelectResult::Remove => {
            debug_assert!(selected.is_empty());
            cursor.remove_current();
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
        let num_blocks = mfunc.blocks.len();
        // 复用的临时缓冲区，避免每条指令分配
        let mut selected: Vec<InstId> = Vec::with_capacity(4);
        for i in 0..num_blocks {
            mfunc.rewrite_block(i, |cursor| {
                let inst_id = cursor.current_inst_id();
                // 如果指令在之前的融合中已被标记为无效，则跳过
                if cursor.current_inst().is_invalid() {
                    cursor.remove_current();
                    return Ok(());
                }

                // 进行指令选择。由具体的后端返回选择结果
                selected.clear();
                let result = {
                    let mut ctx = SelectionContext {
                        mfunc: cursor.mfunc_mut(),
                        inst_id,
                        selected: &mut selected,
                    };
                    match self.target.select_instruction(&mut ctx) {
                        Ok(result) => result,
                        Err(crate::error::Error::Select(err)) => {
                            return Err(crate::error::Error::select(
                                err.opcode.clone(),
                                alloc::format!(
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

                apply_select_result(cursor, &mut selected, result)
            })?;
        }

        Ok(())
    }
}
