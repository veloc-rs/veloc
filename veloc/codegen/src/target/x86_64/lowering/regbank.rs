use super::*;

#[derive(Debug, Clone, Copy)]
pub struct X86_64RegBankSelect;

impl X86_64RegBankSelect {
    pub fn new(_cpu: CpuDescription) -> Self {
        Self
    }
}

impl crate::regalloc::regbank_select::TargetRegBankSelect for X86_64RegBankSelect {
    fn regbank_select_mode(&self) -> crate::regalloc::regbank_select::RegisterBankSelectMode {
        crate::regalloc::regbank_select::RegisterBankSelectMode::TypeDerived
    }

    fn default_bank_for_type(&self, ty: Type) -> crate::regalloc::regbank_select::RegisterBank {
        use crate::regalloc::regbank_select::RegisterBank;

        if ty.is_float() || ty.is_vector() {
            RegisterBank::FPR
        } else {
            RegisterBank::GPR
        }
    }
}
