use bitflags::bitflags;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntCC {
    Eq,
    Ne,
    LtS,
    LtU,
    GtS,
    GtU,
    LeS,
    LeU,
    GeS,
    GeU,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatCC {
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

impl IntCC {
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

    pub fn from_mnemonic(mnemonic: &str) -> Option<Self> {
        match mnemonic {
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

    pub const fn is_unsigned(self) -> bool {
        matches!(self, IntCC::LtU | IntCC::GtU | IntCC::LeU | IntCC::GeU)
    }

    pub const fn swap(self) -> Self {
        match self {
            Self::Eq => Self::Eq,
            Self::Ne => Self::Ne,
            Self::LtS => Self::GtS,
            Self::LtU => Self::GtU,
            Self::GtS => Self::LtS,
            Self::GtU => Self::LtU,
            Self::LeS => Self::GeS,
            Self::LeU => Self::GeU,
            Self::GeS => Self::LeS,
            Self::GeU => Self::LeU,
        }
    }

    pub const fn complement(self) -> Self {
        match self {
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
            Self::LtS => Self::GeS,
            Self::LtU => Self::GeU,
            Self::GtS => Self::LeS,
            Self::GtU => Self::LeU,
            Self::LeS => Self::GtS,
            Self::LeU => Self::GtU,
            Self::GeS => Self::LtS,
            Self::GeU => Self::LtU,
        }
    }
}

impl FloatCC {
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

    pub fn from_mnemonic(mnemonic: &str) -> Option<Self> {
        match mnemonic {
            "eq" => Some(Self::Eq),
            "ne" => Some(Self::Ne),
            "lt" => Some(Self::Lt),
            "gt" => Some(Self::Gt),
            "le" => Some(Self::Le),
            "ge" => Some(Self::Ge),
            _ => None,
        }
    }

    pub const fn swap(self) -> Self {
        match self {
            Self::Eq => Self::Eq,
            Self::Ne => Self::Ne,
            Self::Lt => Self::Gt,
            Self::Gt => Self::Lt,
            Self::Le => Self::Ge,
            Self::Ge => Self::Le,
        }
    }

    /// Exact logical complement under IEEE comparisons. Relational predicates
    /// cannot be complemented with this enum because unordered (NaN) is not
    /// represented separately.
    pub const fn complement(self) -> Option<Self> {
        match self {
            Self::Eq => Some(Self::Ne),
            Self::Ne => Some(Self::Eq),
            Self::Lt | Self::Gt | Self::Le | Self::Ge => None,
        }
    }

    /// Complement valid only when both operands are known not to be NaN.
    pub const fn complement_ordered(self) -> Self {
        match self {
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
            Self::Lt => Self::Ge,
            Self::Gt => Self::Le,
            Self::Le => Self::Gt,
            Self::Ge => Self::Lt,
        }
    }
}

impl core::fmt::Display for IntCC {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.mnemonic())
    }
}

impl core::fmt::Display for FloatCC {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.mnemonic())
    }
}

include!(concat!(env!("OUT_DIR"), "/opcodes.rs"));

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct MemFlags: u16 {
        const ALIGN_MASK = 0b1111;
        const VOLATILE = 1 << 4;
    }
}

impl MemFlags {
    pub const fn new() -> Self {
        Self::empty()
    }

    pub fn is_volatile(&self) -> bool {
        self.contains(Self::VOLATILE)
    }

    pub fn with_alignment(self, align: u32) -> Self {
        let log2 = align.trailing_zeros();
        assert!(
            1 << log2 == align && align != 0,
            "Alignment must be a power of 2"
        );
        let log2 = log2.min(15) as u16;
        let bits = (self.bits() & !Self::ALIGN_MASK.bits()) | log2;
        Self::from_bits_retain(bits)
    }

    pub fn alignment(&self) -> u32 {
        let log2 = self.bits() & Self::ALIGN_MASK.bits();
        1 << log2
    }
}
