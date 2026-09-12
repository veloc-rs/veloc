//! Shared instruction contracts. Representation and behavioral rules live together.
use core::fmt;

// Declaration order determines display order; bit positions remain stable.
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[repr(transparent)]
    pub struct OpTraits: u16 {
        const TERMINATOR = 1 << 0;
        const COMMUTATIVE = 1 << 1;
        const ASSOCIATIVE = 1 << 3;
        const IDEMPOTENT = 1 << 4;
        const MAY_TRAP = 1 << 2;
        const ABORT = 1 << 5;
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[repr(transparent)]
    pub struct MemoryEffects: u8 {
        const READ = 1 << 0;
        const WRITE = 1 << 1;
        const ALLOCATE = 1 << 2;
        const FREE = 1 << 3;
    }
}

impl fmt::Display for OpTraits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        bitflags::parser::to_writer(self, f)
    }
}

impl fmt::Display for MemoryEffects {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        bitflags::parser::to_writer(self, f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryEffect {
    Known(MemoryEffects),
    Unknown,
}

impl MemoryEffect {
    pub const fn is_unknown(self) -> bool {
        matches!(self, Self::Unknown)
    }

    pub const fn is_none(self) -> bool {
        matches!(self, Self::Known(effects) if effects.is_empty())
    }

    const fn may(self, effect: MemoryEffects) -> bool {
        match self {
            Self::Known(effects) => effects.intersects(effect),
            Self::Unknown => true,
        }
    }

    pub const fn may_read(self) -> bool {
        self.may(MemoryEffects::READ)
    }
    pub const fn may_write(self) -> bool {
        self.may(MemoryEffects::WRITE)
    }
    pub const fn may_allocate(self) -> bool {
        self.may(MemoryEffects::ALLOCATE)
    }
    pub const fn may_free(self) -> bool {
        self.may(MemoryEffects::FREE)
    }

    /// Unused abstract objects can be removed; new or unmodeled effects cannot.
    pub const fn can_erase(self) -> bool {
        matches!(self, Self::Known(effects)
            if MemoryEffects::READ.union(MemoryEffects::ALLOCATE).contains(effects))
    }

    pub const fn has_side_effects(self) -> bool {
        !matches!(self, Self::Known(effects) if MemoryEffects::READ.contains(effects))
    }

    /// Only memory interference; movement also requires control/trap/ordering checks.
    pub const fn conflicts_with(self, other: Self) -> bool {
        if self.is_none() || other.is_none() {
            return false;
        }
        let supported = MemoryEffects::READ
            .union(MemoryEffects::WRITE)
            .union(MemoryEffects::ALLOCATE)
            .union(MemoryEffects::FREE);
        match (self, other) {
            (Self::Known(lhs), Self::Known(rhs))
                if supported.contains(lhs) && supported.contains(rhs) =>
            {
                self.may_free()
                    || other.may_free()
                    || (self.may_allocate() && other.may_allocate())
                    || (self.may_write() && (other.may_read() || other.may_write()))
                    || (other.may_write() && self.may_read())
            }
            _ => true,
        }
    }
}

impl core::fmt::Display for MemoryEffect {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Known(effects) => effects.fmt(f),
            Self::Unknown => f.write_str("unknown"),
        }
    }
}

/// Per-access alignment and volatility. Alignment is a conservative power-of-two
/// guarantee, saturated at the largest value supported by the compact encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct MemFlags(u16);

impl MemFlags {
    pub const ALIGNMENT_LOG2_MAX: u16 = 15;
    const VOLATILE: u16 = 1 << 4;

    pub const fn empty() -> Self {
        Self(0)
    }
    pub const fn new() -> Self {
        Self::empty()
    }

    pub const fn alignment_log2(self) -> u16 {
        self.0 & Self::ALIGNMENT_LOG2_MAX
    }

    pub const fn with_alignment_log2(self, value: u16) -> Self {
        assert!(
            value <= Self::ALIGNMENT_LOG2_MAX,
            "alignment exponent out of range"
        );
        Self((self.0 & !Self::ALIGNMENT_LOG2_MAX) | value)
    }

    pub fn with_alignment(self, align: u32) -> Self {
        assert!(align.is_power_of_two(), "Alignment must be a power of 2");
        let log2 = align.trailing_zeros().min(Self::ALIGNMENT_LOG2_MAX as u32) as u16;
        self.with_alignment_log2(log2)
    }

    pub const fn alignment(&self) -> u32 {
        1 << self.alignment_log2()
    }
    pub const fn is_volatile(self) -> bool {
        self.0 & Self::VOLATILE != 0
    }
    pub const fn with_volatile(self, value: bool) -> Self {
        Self((self.0 & !Self::VOLATILE) | ((value as u16) << 4))
    }
}

const impl crate::traits::OpTraits for OpTraits {
    const TERMINATOR: Self = Self::TERMINATOR;
    const COMMUTATIVE: Self = Self::COMMUTATIVE;
    const ASSOCIATIVE: Self = Self::ASSOCIATIVE;
    const IDEMPOTENT: Self = Self::IDEMPOTENT;
    const MAY_TRAP: Self = Self::MAY_TRAP;
    const ABORT: Self = Self::ABORT;
    fn empty() -> Self {
        Self::empty()
    }
    fn union(self, other: Self) -> Self {
        self.union(other)
    }
    fn contains(self, other: Self) -> bool {
        Self::contains(&self, other)
    }
}

const impl crate::traits::MemoryEffects for MemoryEffects {
    const READ: Self = Self::READ;
    const WRITE: Self = Self::WRITE;
    const ALLOCATE: Self = Self::ALLOCATE;
    const FREE: Self = Self::FREE;
    fn empty() -> Self {
        Self::empty()
    }
    fn union(self, other: Self) -> Self {
        self.union(other)
    }
}

const impl crate::traits::MemoryEffect for MemoryEffect {
    const NONE: Self = Self::Known(MemoryEffects::empty());
    const UNKNOWN: Self = Self::Unknown;
    fn known(effects: MemoryEffects) -> Self {
        Self::Known(effects)
    }
    fn is_none(self) -> bool {
        self.is_none()
    }
}
