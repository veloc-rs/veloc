mod encode;
pub use encode::{encode, encode_branch};

include!(concat!(env!("OUT_DIR"), "/x86_64.rs"));

/// Hardware register number, independent of an allocator's physical identity.
/// Legacy encoding currently supports sixteen registers per register bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Reg(u8);
impl Reg {
    pub const fn new(number: u8) -> Option<Self> {
        if number < 16 {
            Some(Self(number))
        } else {
            None
        }
    }
    pub const fn number(self) -> u8 {
        self.0
    }
}

pub type Instruction = crate::Encoded<15>;
