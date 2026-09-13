//! Comparison predicates encoded as accepted outcomes plus integer ordering.
// Low bits describe accepted comparison outcomes; upper bits distinguish
// signed and unsigned integer ordering. Equality carries neither ordering tag.
// The outcome bit positions also form the offline IntPredicate contract.
const LESS: u8 = 1 << 0;
const EQUAL: u8 = 1 << 1;
const GREATER: u8 = 1 << 2;
const UNORDERED: u8 = 1 << 3;
const SIGNED: u8 = 1 << 4;
const UNSIGNED: u8 = 1 << 5;
const ORDERED: u8 = LESS | EQUAL | GREATER;
const OUTCOMES: u8 = ORDERED | UNORDERED;

const fn swap_outcomes(bits: u8) -> u8 {
    // Exchange less/greater only when exactly one is present; preserve all tags.
    if (bits & LESS != 0) != (bits & GREATER != 0) {
        bits ^ (LESS | GREATER)
    } else {
        bits
    }
}

#[derive(Debug, Clone, Copy, Hash)]
#[derive_const(PartialEq, Eq)]
#[repr(u8)]
pub enum IntCC {
    Eq = EQUAL,
    Ne = LESS | GREATER,
    LtS = SIGNED | LESS,
    LtU = UNSIGNED | LESS,
    GtS = SIGNED | GREATER,
    GtU = UNSIGNED | GREATER,
    LeS = SIGNED | LESS | EQUAL,
    LeU = UNSIGNED | LESS | EQUAL,
    GeS = SIGNED | GREATER | EQUAL,
    GeU = UNSIGNED | GREATER | EQUAL,
}
impl IntCC {
    const fn from_bits(bits: u8) -> Option<Self> {
        match bits {
            bits if bits == Self::Eq as u8 => Some(Self::Eq),
            bits if bits == Self::Ne as u8 => Some(Self::Ne),
            bits if bits == Self::LtS as u8 => Some(Self::LtS),
            bits if bits == Self::LtU as u8 => Some(Self::LtU),
            bits if bits == Self::GtS as u8 => Some(Self::GtS),
            bits if bits == Self::GtU as u8 => Some(Self::GtU),
            bits if bits == Self::LeS as u8 => Some(Self::LeS),
            bits if bits == Self::LeU as u8 => Some(Self::LeU),
            bits if bits == Self::GeS as u8 => Some(Self::GeS),
            bits if bits == Self::GeU as u8 => Some(Self::GeU),
            _ => None,
        }
    }
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::LtS => "lts",
            Self::LtU => "ltu",
            Self::GtS => "gts",
            Self::GtU => "gtu",
            Self::LeS => "les",
            Self::LeU => "leu",
            Self::GeS => "ges",
            Self::GeU => "geu",
        }
    }
    pub fn from_mnemonic(text: &str) -> Option<Self> {
        match text {
            "eq" => Some(Self::Eq),
            "ne" => Some(Self::Ne),
            "lts" => Some(Self::LtS),
            "ltu" => Some(Self::LtU),
            "gts" => Some(Self::GtS),
            "gtu" => Some(Self::GtU),
            "les" => Some(Self::LeS),
            "leu" => Some(Self::LeU),
            "ges" => Some(Self::GeS),
            "geu" => Some(Self::GeU),
            _ => None,
        }
    }
    pub const fn outcomes(self) -> u8 {
        self as u8 & OUTCOMES
    }
    pub const fn swap(self) -> Self {
        match Self::from_bits(swap_outcomes(self as u8)) {
            Some(cc) => cc,
            None => unreachable!(),
        }
    }
    pub const fn is_signed(self) -> bool {
        self as u8 & SIGNED != 0
    }
    pub const fn is_unsigned(self) -> bool {
        self as u8 & UNSIGNED != 0
    }
    pub const fn complement(self) -> Self {
        match Self::from_bits(self as u8 ^ ORDERED) {
            Some(cc) => cc,
            None => unreachable!(),
        }
    }
    /// Compare fixed-width bitvectors; discard any bits outside the width.
    pub const fn test(self, bits: u16, lhs: u128, rhs: u128) -> bool {
        assert!(bits > 0 && bits as u32 <= u128::BITS);
        let mask = u128::MAX >> (u128::BITS - bits as u32);
        let sign = if self.is_signed() {
            1u128 << (bits - 1)
        } else {
            0
        };
        let lhs = (lhs & mask) ^ sign;
        let rhs = (rhs & mask) ^ sign;
        let outcome = if lhs < rhs {
            LESS
        } else if lhs == rhs {
            EQUAL
        } else {
            GREATER
        };
        self.outcomes() & outcome != 0
    }
}
impl core::fmt::Display for IntCC {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.mnemonic())
    }
}

#[derive(Debug, Clone, Copy, Hash)]
#[derive_const(PartialEq, Eq)]
#[repr(u8)]
pub enum FloatCC {
    Eq = EQUAL,
    Ne = LESS | GREATER | UNORDERED,
    Lt = LESS,
    Gt = GREATER,
    Le = LESS | EQUAL,
    Ge = GREATER | EQUAL,
}
impl FloatCC {
    const fn from_bits(bits: u8) -> Option<Self> {
        match bits {
            bits if bits == Self::Eq as u8 => Some(Self::Eq),
            bits if bits == Self::Ne as u8 => Some(Self::Ne),
            bits if bits == Self::Lt as u8 => Some(Self::Lt),
            bits if bits == Self::Gt as u8 => Some(Self::Gt),
            bits if bits == Self::Le as u8 => Some(Self::Le),
            bits if bits == Self::Ge as u8 => Some(Self::Ge),
            _ => None,
        }
    }
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Lt => "lt",
            Self::Gt => "gt",
            Self::Le => "le",
            Self::Ge => "ge",
        }
    }
    pub fn from_mnemonic(text: &str) -> Option<Self> {
        match text {
            "eq" => Some(Self::Eq),
            "ne" => Some(Self::Ne),
            "lt" => Some(Self::Lt),
            "gt" => Some(Self::Gt),
            "le" => Some(Self::Le),
            "ge" => Some(Self::Ge),
            _ => None,
        }
    }
    pub const fn outcomes(self) -> u8 {
        self as u8 & OUTCOMES
    }
    pub const fn swap(self) -> Self {
        match Self::from_bits(swap_outcomes(self as u8)) {
            Some(cc) => cc,
            None => unreachable!(),
        }
    }
    /// Exact IEEE complement, if represented by this condition-code set.
    pub const fn complement(self) -> Option<Self> {
        Self::from_bits(self.outcomes() ^ OUTCOMES)
    }
    /// Valid only when both operands are known not to be NaN.
    pub const fn complement_ordered(self) -> Self {
        let wanted = (self.outcomes() & ORDERED) ^ ORDERED;
        // Prefer an exact set before treating Ne as the ordered not-equal set.
        match Self::from_bits(wanted) {
            Some(cc) => cc,
            None if wanted == LESS | GREATER => Self::Ne,
            _ => unreachable!(),
        }
    }
}
impl core::fmt::Display for FloatCC {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.mnemonic())
    }
}
// Keep the public inherent API usable without importing the DSL contract.
#[allow(non_upper_case_globals)]
const impl crate::traits::IntCCInfo for IntCC {
    const Eq: Self = Self::Eq;
    const Ne: Self = Self::Ne;
    const LtS: Self = Self::LtS;
    const LtU: Self = Self::LtU;
    const GtS: Self = Self::GtS;
    const GtU: Self = Self::GtU;
    const LeS: Self = Self::LeS;
    const LeU: Self = Self::LeU;
    const GeS: Self = Self::GeS;
    const GeU: Self = Self::GeU;
    fn outcomes(self) -> u8 {
        self.outcomes()
    }
    fn swap(self) -> Self {
        self.swap()
    }
    fn complement(self) -> Self {
        self.complement()
    }
    fn is_signed(self) -> bool {
        self.is_signed()
    }
    fn is_unsigned(self) -> bool {
        self.is_unsigned()
    }
    fn test(self, bits: u16, lhs: u128, rhs: u128) -> bool {
        self.test(bits, lhs, rhs)
    }
}
// Keep the public inherent API usable without importing the DSL contract.
#[allow(non_upper_case_globals)]
const impl crate::traits::FloatCCInfo for FloatCC {
    const Eq: Self = Self::Eq;
    const Ne: Self = Self::Ne;
    const Lt: Self = Self::Lt;
    const Gt: Self = Self::Gt;
    const Le: Self = Self::Le;
    const Ge: Self = Self::Ge;
    fn outcomes(self) -> u8 {
        self.outcomes()
    }
    fn swap(self) -> Self {
        self.swap()
    }
    fn complement(self) -> Option<Self> {
        self.complement()
    }
    fn complement_ordered(self) -> Self {
        self.complement_ordered()
    }
}
