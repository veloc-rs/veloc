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
}

macro_rules! define_intrinsics {
    ($($constant:ident = $id:literal => $name:literal;)*) => {
        pub mod ids {
            use super::Intrinsic;

            $(pub const $constant: Intrinsic = Intrinsic($id);)*
        }

        impl Intrinsic {
            pub const ALL: &'static [Self] = &[$(ids::$constant,)*];

            pub const fn name(self) -> &'static str {
                match self {
                    $(ids::$constant => $name,)*
                    _ => "veloc.unknown",
                }
            }

            pub fn from_name(name: &str) -> Option<Self> {
                match name {
                    $($name => Some(ids::$constant),)*
                    _ => None,
                }
            }
        }
    };
}

define_intrinsics! {
    SIN_F32 = 0 => "veloc.sin.f32";
    SIN_F64 = 1 => "veloc.sin.f64";
    COS_F32 = 2 => "veloc.cos.f32";
    COS_F64 = 3 => "veloc.cos.f64";
    POW_F32 = 4 => "veloc.pow.f32";
    POW_F64 = 5 => "veloc.pow.f64";
    EXP_F32 = 6 => "veloc.exp.f32";
    EXP_F64 = 7 => "veloc.exp.f64";
    LOG_F32 = 8 => "veloc.log.f32";
    LOG_F64 = 9 => "veloc.log.f64";
    LOG2_F32 = 10 => "veloc.log2.f32";
    LOG2_F64 = 11 => "veloc.log2.f64";
    LOG10_F32 = 12 => "veloc.log10.f32";
    LOG10_F64 = 13 => "veloc.log10.f64";
    MEMCPY = 14 => "veloc.memcpy";
    MEMMOVE = 15 => "veloc.memmove";
    MEMSET = 16 => "veloc.memset";
    MEMCMP = 17 => "veloc.memcmp";
    FENCE = 18 => "veloc.fence";
    FENCE_ACQ = 19 => "veloc.fence.acq";
    FENCE_REL = 20 => "veloc.fence.rel";
    FENCE_SEQ = 21 => "veloc.fence.seq";
    ASSUME = 22 => "veloc.assume";
    EXPECT = 23 => "veloc.expect";
    TRAP = 24 => "veloc.trap";
}

impl fmt::Display for Intrinsic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}
