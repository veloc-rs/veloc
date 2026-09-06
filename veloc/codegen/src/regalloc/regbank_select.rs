use crate::target::arch::TargetMachine;
use alloc::vec::Vec;
use veloc_lir::{MachineFunction, MachineOperand};
use veloc_mir::Type;

use veloc_lir::RegisterBank;

/// 寄存器库分辨策略。
///
/// `TypeDerived` 适合 bank 主要由值类型决定的目标；
/// `GlobalHints` 适合需要借助操作数 hint 与 copy 传播的目标；
/// `Disabled` 则表示当前目标不依赖显式 bank 标注。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterBankSelectMode {
    Disabled,
    TypeDerived,
    GlobalHints,
}

pub trait TargetRegBankSelect: Send + Sync {
    fn regbank_select_mode(&self) -> RegisterBankSelectMode {
        RegisterBankSelectMode::TypeDerived
    }

    fn default_bank_for_type(&self, ty: Type) -> RegisterBank {
        if ty.is_predicate() {
            RegisterBank::PR
        } else if ty.is_vector() {
            RegisterBank::VR
        } else if ty.is_float() {
            RegisterBank::FPR
        } else {
            RegisterBank::GPR
        }
    }

    fn suggest_bank(
        &self,
        opcode: veloc_lir::GenericOpcode,
        index: usize,
        ty: Type,
    ) -> Option<RegisterBank> {
        let _ = (opcode, index, ty);
        None
    }
}

pub struct RegisterBankSelector;

impl RegisterBankSelector {
    pub fn new() -> Self {
        Self
    }

    pub fn select<S>(&self, mfunc: &mut MachineFunction<S>, tm: &dyn TargetMachine) -> bool {
        let rb_select = tm.target_regbank_select();
        match rb_select.regbank_select_mode() {
            RegisterBankSelectMode::Disabled => false,
            RegisterBankSelectMode::TypeDerived => self.assign_derived_banks(mfunc, rb_select),
            RegisterBankSelectMode::GlobalHints => self.assign_hint_banks(mfunc, rb_select),
        }
    }

    fn assign_derived_banks<S>(
        &self,
        mfunc: &mut MachineFunction<S>,
        rb_select: &dyn TargetRegBankSelect,
    ) -> bool {
        let mut changed = false;
        for i in 0..mfunc.vregs.len() {
            let vreg = veloc_lir::VReg::from_u32(i as u32);
            let data = &mut mfunc.vregs[vreg];
            if data.bank.is_none() {
                data.bank = Some(rb_select.default_bank_for_type(data.ty));
                changed = true;
            }
        }
        changed
    }

    fn assign_hint_banks<S>(
        &self,
        mfunc: &mut MachineFunction<S>,
        rb_select: &dyn TargetRegBankSelect,
    ) -> bool {
        let mut updates = Vec::new();
        for block in &mfunc.blocks {
            for &inst_id in &block.insts {
                let inst = &mfunc.dfg[inst_id];
                if let veloc_lir::MachineOpcode::Generic(opcode) = inst.opcode {
                    for (op_idx, op) in inst.operands.iter().enumerate() {
                        let reg = match op {
                            MachineOperand::Use(r) => Some(*r),
                            MachineOperand::Def(w) => Some(w.to_reg()),
                            MachineOperand::TiedDefUse(w) => Some(w.to_reg()),
                            _ => None,
                        };
                        if let Some(r) = reg {
                            if r.is_vreg() {
                                let ty = mfunc.vreg_data(r).ty;
                                if let Some(bank) = rb_select.suggest_bank(opcode, op_idx, ty) {
                                    updates.push((r, bank));
                                }
                            }
                        }
                    }
                }
            }
        }
        let mut any_change = false;
        for (r, bank) in updates {
            any_change |= assign_bank(mfunc, r, bank);
        }

        let mut changed = true;
        while changed {
            changed = false;
            let mut local_updates = Vec::new();
            for block in &mfunc.blocks {
                for &inst_id in &block.insts {
                    let inst = &mfunc.dfg[inst_id];
                    if let veloc_lir::MachineOpcode::Generic(veloc_lir::GenericOpcode::G_COPY) =
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
                let updated = assign_bank(mfunc, r, bank);
                changed |= updated;
                any_change |= updated;
            }
        }

        for i in 0..mfunc.vregs.len() {
            let vreg = veloc_lir::VReg::from_u32(i as u32);
            let data = &mut mfunc.vregs[vreg];
            if data.bank.is_none() {
                data.bank = Some(rb_select.default_bank_for_type(data.ty));
                any_change = true;
            }
        }

        any_change
    }
}

fn assign_bank<S>(mfunc: &mut MachineFunction<S>, reg: veloc_lir::Reg, bank: RegisterBank) -> bool {
    let data = mfunc.vreg_data_mut(reg);
    if data.bank == Some(bank) {
        return false;
    }

    if data.bank.is_none() {
        data.bank = Some(bank);
        return true;
    }

    // `GlobalHints` 仍是启发式传播器，不在这里覆盖已经稳定的 bank 结论。
    false
}
