//! Intrinsic Functions - compiler built-in functions with special semantics

use core::fmt;

/// Intrinsic identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Intrinsic(pub u16);

impl Intrinsic {
    pub const fn from_u16(index: u16) -> Self {
        Intrinsic(index)
    }
    pub const fn as_u16(self) -> u16 {
        self.0
    }
    pub fn name(self) -> &'static str {
        intrinsic_name(self)
    }
    pub fn family(self) -> IntrinsicFamily {
        intrinsic_family(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntrinsicFamily {
    Math,
    Memory,
    Sync,
    Debug,
    OverflowArith,
    SatArith,
    SatConv,
    Bit,
}

pub mod ids {
    use super::Intrinsic;

    // Math (complex functions not in Opcode)
    pub const SIN_F32: Intrinsic = Intrinsic(0);
    pub const SIN_F64: Intrinsic = Intrinsic(1);
    pub const COS_F32: Intrinsic = Intrinsic(2);
    pub const COS_F64: Intrinsic = Intrinsic(3);
    pub const POW_F32: Intrinsic = Intrinsic(4);
    pub const POW_F64: Intrinsic = Intrinsic(5);
    pub const EXP_F32: Intrinsic = Intrinsic(6);
    pub const EXP_F64: Intrinsic = Intrinsic(7);
    pub const LOG_F32: Intrinsic = Intrinsic(8);
    pub const LOG_F64: Intrinsic = Intrinsic(9);
    pub const LOG2_F32: Intrinsic = Intrinsic(10);
    pub const LOG2_F64: Intrinsic = Intrinsic(11);
    pub const LOG10_F32: Intrinsic = Intrinsic(12);
    pub const LOG10_F64: Intrinsic = Intrinsic(13);

    // Memory
    pub const MEMCPY: Intrinsic = Intrinsic(14);
    pub const MEMMOVE: Intrinsic = Intrinsic(15);
    pub const MEMSET: Intrinsic = Intrinsic(16);
    pub const MEMCMP: Intrinsic = Intrinsic(17);

    // Sync
    pub const FENCE: Intrinsic = Intrinsic(18);
    pub const FENCE_ACQ: Intrinsic = Intrinsic(19);
    pub const FENCE_REL: Intrinsic = Intrinsic(20);
    pub const FENCE_SEQ: Intrinsic = Intrinsic(21);

    // Debug
    pub const ASSUME: Intrinsic = Intrinsic(22);
    pub const EXPECT: Intrinsic = Intrinsic(23);
    pub const TRAP: Intrinsic = Intrinsic(24);
}

fn intrinsic_name(i: Intrinsic) -> &'static str {
    match i.0 {
        0 => "veloc.sin.f32",
        1 => "veloc.sin.f64",
        2 => "veloc.cos.f32",
        3 => "veloc.cos.f64",
        4 => "veloc.pow.f32",
        5 => "veloc.pow.f64",
        6 => "veloc.exp.f32",
        7 => "veloc.exp.f64",
        8 => "veloc.log.f32",
        9 => "veloc.log.f64",
        10 => "veloc.log2.f32",
        11 => "veloc.log2.f64",
        12 => "veloc.log10.f32",
        13 => "veloc.log10.f64",
        14 => "veloc.memcpy",
        15 => "veloc.memmove",
        16 => "veloc.memset",
        17 => "veloc.memcmp",
        18 => "veloc.fence",
        19 => "veloc.fence.acq",
        20 => "veloc.fence.rel",
        21 => "veloc.fence.seq",
        22 => "veloc.assume",
        23 => "veloc.expect",
        24 => "veloc.trap",
        _ => "veloc.unknown",
    }
}

fn intrinsic_family(i: Intrinsic) -> IntrinsicFamily {
    use IntrinsicFamily as F;
    match i.0 {
        0..=13 => F::Math,
        14..=17 => F::Memory,
        18..=21 => F::Sync,
        22..=24 => F::Debug,
        _ => F::Math,
    }
}

impl fmt::Display for Intrinsic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}
