//! Byte ranges relative to an identity supplied by the IR or analysis host.
//! Different identities provide no alias proof by themselves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MemoryLocation<B> {
    pub base: B,
    pub offset: i64,
    pub bytes: u32,
}

impl<B: Eq> MemoryLocation<B> {
    fn end(&self) -> i128 {
        i128::from(self.offset) + i128::from(self.bytes)
    }

    /// Whether the linear ranges share a base and at least one byte.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.base == other.base
            && i128::from(self.offset.max(other.offset)) < self.end().min(other.end())
    }

    /// Whether this linear range contains all bytes of the other range.
    pub fn covers(&self, other: &Self) -> bool {
        self.base == other.base && self.offset <= other.offset && other.end() <= self.end()
    }

    pub fn may_overlap(&self, other: &Self, pointer_bits: u32) -> bool {
        self.base != other.base
            || ranges_may_overlap(
                (self.offset, self.bytes),
                (other.offset, other.bytes),
                pointer_bits,
            )
    }
}

/// Compare two byte ranges relative to the same base, allowing address wraparound.
pub fn ranges_may_overlap(
    (a_offset, a_bytes): (i64, u32),
    (b_offset, b_bytes): (i64, u32),
    pointer_bits: u32,
) -> bool {
    if !(1..=64).contains(&pointer_bits) {
        return true;
    }
    let space = 1u128 << pointer_bits;
    if a_bytes == 0 || b_bytes == 0 || u128::from(a_bytes.max(b_bytes)) >= space {
        return true;
    }
    let distance = b_offset.wrapping_sub(a_offset) as u64 as u128 & (space - 1);
    distance < u128::from(a_bytes) || ((space - distance) & (space - 1)) < u128::from(b_bytes)
}
