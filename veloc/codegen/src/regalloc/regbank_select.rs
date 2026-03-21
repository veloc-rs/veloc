use crate::mir::{MachineFunction, MachineOperand};
use crate::target::arch::TargetMachine;
use alloc::vec::Vec;
use veloc_ir::Type;

/// 寄存器库 (Register Bank)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegisterBank {
    GPR,
    FPR,
}

pub trait TargetRegBankSelect: Send + Sync {
    fn suggest_bank(
        &self,
        opcode: crate::mir::GenericOpcode,
        index: usize,
        ty: Type,
    ) -> Option<RegisterBank>;
}

pub struct RegisterBankSelector;

impl RegisterBankSelector {
    pub fn new() -> Self {
        Self
    }

    pub fn select(&self, mfunc: &mut MachineFunction, tm: &dyn TargetMachine) {
        let rb_select = tm.target_regbank_select();

        let mut updates = Vec::new();
        for block in &mfunc.blocks {
            for &inst_id in &block.insts {
                let inst = &mfunc.dfg[inst_id];
                if let crate::mir::MachineOpcode::Generic(opcode) = inst.opcode {
                    for (op_idx, op) in inst.operands.iter().enumerate() {
                        let reg = match op {
                            MachineOperand::Use(r) => Some(*r),
                            MachineOperand::Def(w) => Some(w.to_reg()),
                            MachineOperand::TiedDefUse(w) => Some(w.to_reg()),
                            _ => None,
                        };
                        if let Some(r) = reg {
                            if r.is_vreg() {
                                let ty = mfunc.vreg_data(r).ty.clone();
                                if let Some(bank) = rb_select.suggest_bank(opcode, op_idx, ty) {
                                    updates.push((r, bank));
                                }
                            }
                        }
                    }
                }
            }
        }
        for (r, bank) in updates {
            mfunc.vreg_data_mut(r).bank = Some(bank);
        }

        let mut changed = true;
        while changed {
            changed = false;
            let mut local_updates = Vec::new();
            for block in &mfunc.blocks {
                for &inst_id in &block.insts {
                    let inst = &mfunc.dfg[inst_id];
                    if let crate::mir::MachineOpcode::Generic(crate::mir::GenericOpcode::G_COPY) =
                        inst.opcode
                    {
                        let dst = inst.defs().next();
                        let src = inst.uses().next();
                        if let (Some(d), Some(s)) = (dst, src) {
                            if d.is_vreg() && s.is_vreg() {
                                let src_bank = mfunc.vreg_data(s).bank;
                                let dst_bank = mfunc.vreg_data(d).bank;

                                if src_bank.is_some() && dst_bank.is_none() {
                                    local_updates.push((d, src_bank.unwrap()));
                                } else if dst_bank.is_some() && src_bank.is_none() {
                                    local_updates.push((s, dst_bank.unwrap()));
                                }
                            }
                        }
                    }
                }
            }
            for (r, bank) in local_updates {
                mfunc.vreg_data_mut(r).bank = Some(bank);
                changed = true;
            }
        }

        for i in 0..mfunc.vregs.len() {
            let vreg = crate::mir::VReg::from_u32(i as u32);
            let data = &mut mfunc.vregs[vreg];
            if data.bank.is_none() {
                if data.ty.is_float() || data.ty.is_vector() {
                    data.bank = Some(RegisterBank::FPR);
                } else {
                    data.bank = Some(RegisterBank::GPR);
                }
            }
        }
    }
}
