//! Per-access facts, independent of the instruction's register representation.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryKind {
    Read,
    Write,
}

/// A single fixed-size access. Missing information on an instruction is unknown,
/// not a promise that it has no memory effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryAccess {
    pub kind: MemoryKind,
    /// Bytes touched in memory, which need not equal the result register width.
    pub bytes: u32,
    /// Guaranteed alignment of the effective address.
    pub alignment: u32,
    pub volatile: bool,
    pub may_trap: bool,
}

impl MemoryAccess {
    pub fn new(kind: MemoryKind, bytes: u32) -> Self {
        assert!(bytes != 0, "memory access must touch at least one byte");
        Self {
            kind,
            bytes,
            alignment: 1,
            volatile: false,
            may_trap: true,
        }
    }
}
