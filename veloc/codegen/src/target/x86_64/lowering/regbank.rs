use super::*;

#[derive(Debug, Clone, Copy)]
pub struct X86_64RegBankSelect;

impl crate::regalloc::regbank_select::TargetRegBankSelect for X86_64RegBankSelect {
    fn regbank_select_mode(&self) -> crate::regalloc::regbank_select::RegisterBankSelectMode {
        crate::regalloc::regbank_select::RegisterBankSelectMode::TypeDerived
    }

    fn default_bank_for_type(&self, ty: Type) -> veloc_lir::RegisterBank {
        use veloc_lir::RegisterBank;

        if ty.is_float() || ty.is_vector() {
            RegisterBank::FPR
        } else {
            RegisterBank::GPR
        }
    }
}
