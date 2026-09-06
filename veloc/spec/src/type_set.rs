//! Exact sets of compact MIR types, independent of their packed bit positions.
//! A scalar code maps to a shape mask: bit 0 is scalar, bits 1..15 are fixed
//! vector lane exponents, and bits 17..31 are scalable vector lane exponents.

use std::collections::BTreeMap;

#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct TypeSet(pub BTreeMap<u8, u32>);

impl TypeSet {
    pub fn singleton(code: u8, exponent: u32, scalable: bool) -> Self {
        Self(BTreeMap::from([(
            code,
            1 << (exponent + if scalable { 16 } else { 0 }),
        )]))
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn union(&mut self, other: &Self) {
        for (&code, &shapes) in &other.0 {
            *self.0.entry(code).or_default() |= shapes;
        }
    }

    pub fn subset_of(&self, other: &Self) -> bool {
        !self.is_empty()
            && self
                .0
                .iter()
                .all(|(code, shapes)| shapes & !other.0.get(code).copied().unwrap_or(0) == 0)
    }

    pub fn intersect(&mut self, other: &Self) {
        self.0.retain(|code, shapes| {
            *shapes &= other.0.get(code).copied().unwrap_or(0);
            *shapes != 0
        });
    }

    pub fn shapes(&self) -> u32 {
        self.0.values().fold(0, |all, shapes| all | shapes)
    }

    pub fn retain_shapes(&mut self, allowed: u32) {
        self.0.retain(|_, shapes| {
            *shapes &= allowed;
            *shapes != 0
        });
    }

    /// Caller has checked that all members are non-pointer scalars.
    pub fn vectors(&self, max_exponent: u32) -> Self {
        let fixed = ((1u32 << (max_exponent + 1)) - 1) & !1;
        let shapes = fixed | (fixed << 16);
        Self(self.0.keys().map(|&code| (code, shapes)).collect())
    }
}
