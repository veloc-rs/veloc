/// Borrowed feature bits. Both sets in a comparison must use the same target's
/// feature numbering; absent words represent disabled features.
#[derive(Debug, Clone, Copy)]
pub struct FeatureSetRef<'a> {
    words: &'a [u64],
}

impl<'a> FeatureSetRef<'a> {
    pub const fn new(words: &'a [u64]) -> Self {
        Self { words }
    }

    pub fn contains_all(self, required: FeatureSetRef<'_>) -> bool {
        required
            .words
            .iter()
            .enumerate()
            .all(|(index, mask)| self.words.get(index).copied().unwrap_or(0) & mask == *mask)
    }
}
