use super::inst as generated;
use crate::passes::lowering::legalize::LegalizePolicy;

// Declared contracts are checked even if a particular target rule does not use
// every method yet.
#[allow(dead_code)]
mod host {
    include!(concat!(env!("OUT_DIR"), "/legalize_x86_64.rs"));
}

impl host::Instruction for crate::target::x86_64::inst::TargetInst {
    const POPCNT32: &'static [u64] = Self::X86Popcnt32.required_features().as_words();
    const POPCNT64: &'static [u64] = Self::X86Popcnt64.required_features().as_words();
}
impl host::Target for generated::FeatureSet {
    fn words(&self) -> &[u64] {
        self.as_words()
    }
}

pub(super) fn policy(features: &generated::FeatureSet) -> LegalizePolicy<'_> {
    LegalizePolicy {
        program: &host::PROGRAM,
        features: host::Target::words(features),
    }
}
