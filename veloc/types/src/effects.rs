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
        const ABORT = 1 << 5;
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[repr(transparent)]
    pub struct MemoryEffect: u8 {
        const READ = 1 << 0;
        const WRITE = 1 << 1;
    }
}

impl fmt::Display for OpTraits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        bitflags::parser::to_writer(self, f)
    }
}

impl Default for MemoryEffect {
    fn default() -> Self {
        Self::all()
    }
}

impl MemoryEffect {
    pub const NONE: Self = Self::empty();

    pub const fn is_none(self) -> bool {
        self.is_empty()
    }

    const fn may(self, effect: Self) -> bool {
        !Self::all().contains(self) || self.intersects(effect)
    }

    pub const fn may_read(self) -> bool {
        self.may(Self::READ)
    }
    pub const fn may_write(self) -> bool {
        self.may(Self::WRITE)
    }

    /// Memory interference only. Movement also requires lifetime, control, trap and ordering checks.
    pub const fn conflicts_with(self, other: Self) -> bool {
        !self.is_none() && !other.is_none() && (self.may_write() || other.may_write())
    }
}

impl fmt::Display for MemoryEffect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        bitflags::parser::to_writer(self, f)
    }
}

/// Module-wide alias region. Overlapping accesses must use the same region.
/// An access with no explicit region belongs to region zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Region(pub u8);

impl Region {
    pub fn is_disjoint(self, other: Self) -> bool {
        self != other
    }
}

/// Per-access alignment, volatility and alias region. Alignment is a conservative power-of-two
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
    pub const fn region(self) -> Region {
        Region((self.0 >> 8) as u8)
    }

    pub const fn with_region(self, region: Region) -> Self {
        Self((self.0 & 0xff) | ((region.0 as u16) << 8))
    }
}
