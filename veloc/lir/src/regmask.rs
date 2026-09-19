//! Sets of physical storage roots. Targets resolve register views to roots;
//! masks describe whole-root destruction, not partial-register writes.
use crate::{PReg, Reg};

/// Shared static clobber set emitted by the target's ABI definitions.
#[derive(Debug, Clone, Copy)]
pub struct RegMask(&'static [u64]);
impl Default for RegMask {
    fn default() -> Self {
        Self::from_static(&[])
    }
}
impl PartialEq for RegMask {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}
impl Eq for RegMask {}
impl RegMask {
    pub const fn from_static(words: &'static [u64]) -> Self {
        Self(words)
    }
    pub fn contains(&self, root: PReg) -> bool {
        let index = root.index() as usize;
        self.0
            .get(index / 64)
            .is_some_and(|word| word & (1 << (index % 64)) != 0)
    }
    pub fn iter(&self) -> impl Iterator<Item = Reg> + '_ {
        self.0.iter().enumerate().flat_map(|(index, &word)| {
            let mut bits = word;
            core::iter::from_fn(move || {
                if bits == 0 {
                    return None;
                }
                let bit = bits.trailing_zeros();
                bits &= bits - 1;
                Some(Reg::new_preg((index * 64) as u32 + bit))
            })
        })
    }
}
