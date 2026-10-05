//! Byte ranges relative to an identity supplied by the IR or analysis host.
//! Different identities provide no alias proof by themselves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MemoryLocation<B> {
    pub base: B,
    pub offset: i64,
    pub bytes: u32,
}

impl<B: Eq> MemoryLocation<B> {
    pub fn may_overlap(&self, other: &Self, pointer_bits: u32) -> bool {
        if self.base != other.base || !(1..=64).contains(&pointer_bits) {
            return true;
        }
        let space = 1u128 << pointer_bits;
        if self.bytes == 0 || other.bytes == 0 || u128::from(self.bytes.max(other.bytes)) >= space {
            return true;
        }
        let distance = other.offset.wrapping_sub(self.offset) as u64 as u128 & (space - 1);
        distance < u128::from(self.bytes)
            || ((space - distance) & (space - 1)) < u128::from(other.bytes)
    }
}
