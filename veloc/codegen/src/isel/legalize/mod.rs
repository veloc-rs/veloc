pub mod info;

pub use info::*;

use crate::mir::{GenericOpcode, MachineFunction, MachineInst, MachineOpcode, MachineOperand};
use crate::target::arch::TargetLowering;
use alloc::vec::Vec;
use veloc_ir::Type;

pub struct Legalizer<'a> {
    info: &'a LegalizerInfo,
    lowering: &'a dyn TargetLowering,
}

impl<'a> Legalizer<'a> {
    pub fn new(info: &'a LegalizerInfo, lowering: &'a dyn TargetLowering) -> Self {
        Self { info, lowering }
    }

    pub fn legalize(&self, mfunc: &mut MachineFunction) {
        let num_blocks = mfunc.blocks.len();
        for i in 0..num_blocks {
            mfunc
                .rewrite_block(i, |cursor| {
                    let inst_id = cursor.current_inst_id();
                    // 如果指令已无效，跳过
                    if cursor.current_inst().is_invalid() {
                        cursor.remove_current();
                        return Ok::<(), ()>(());
                    }

                    let (types, opcode_opt) = {
                        let inst = cursor.current_inst();
                        let types = self.get_inst_types(inst, cursor.mfunc());
                        let opcode = if let MachineOpcode::Generic(op) = &inst.opcode {
                            Some(*op)
                        } else {
                            None
                        };
                        (types, opcode)
                    };

                    if let Some(opcode) = opcode_opt {
                        let action = self.info.get_action(&opcode, &types);

                        match action {
                            LegalizeAction::Legal => {
                                cursor.keep_current();
                            }
                            LegalizeAction::Lower => {
                                // 调用架构特定的合法化逻辑
                                // 后端负责将原指令或新生成的指令推入输出序列
                                let mut output = Vec::new();
                                self.lowering
                                    .legalize_instruction(inst_id, cursor.mfunc_mut(), &mut output);
                                cursor.remove_current();
                                for new_id in output {
                                    cursor.emit_existing_before(new_id);
                                }
                            }
                            LegalizeAction::WidenScalar => {
                                self.widen_scalar(cursor.mfunc_mut(), inst_id, &opcode, types);
                                if cursor.current_inst().is_invalid() {
                                    cursor.remove_current();
                                } else {
                                    cursor.keep_current();
                                }
                            }
                            _ => {
                                // 其他 action 暂不处理，保留原指令
                                cursor.keep_current();
                            }
                        }
                    } else {
                        // 非通用指令直通
                        cursor.keep_current();
                    }

                    Ok(())
                })
                .expect("legalize block rewriting should not fail");
        }
    }

    fn get_inst_types(&self, inst: &MachineInst, mfunc: &MachineFunction) -> Vec<Type> {
        let mut types = Vec::new();
        for op in &inst.operands {
            let reg = match op {
                MachineOperand::Use(r) => Some(*r),
                MachineOperand::Def(w) => Some(w.to_reg()),
                MachineOperand::TiedDefUse(w) => Some(w.to_reg()),
                _ => None,
            };
            if let Some(r) = reg {
                if r.is_vreg() {
                    types.push(mfunc.vreg_data(r).ty.clone());
                }
            }
        }
        types
    }

    fn widen_scalar(
        &self,
        _mfunc: &mut crate::mir::MachineFunction,
        _inst_id: crate::mir::InstId,
        _opcode: &GenericOpcode,
        _types: Vec<Type>,
    ) {
        // 简化实现：示例性的逻辑
    }

    fn narrow_scalar(
        &self,
        _mfunc: &mut crate::mir::MachineFunction,
        _inst_id: crate::mir::InstId,
        _opcode: &GenericOpcode,
        _types: Vec<Type>,
    ) {
        // 简化实现
    }

    fn lower(
        &self,
        _mfunc: &mut crate::mir::MachineFunction,
        _inst_id: crate::mir::InstId,
        _opcode: &GenericOpcode,
        _types: Vec<Type>,
    ) {
        // 简化实现
    }
}
