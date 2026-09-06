use bitflags::bitflags;

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
