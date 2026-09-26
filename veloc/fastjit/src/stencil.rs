//! Copy prebuilt instruction fragments, then resolve typed holes in one pass.

use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchKind {
    /// Little-endian signed 32-bit integer.
    I32,
    /// Little-endian 64-bit bit pattern.
    U64,
    /// Signed displacement from the end of the patched 32-bit field.
    X86Rel32,
}

#[derive(Debug, Clone, Copy)]
pub struct Hole {
    pub offset: u16,
    pub kind: PatchKind,
}

#[derive(Debug)]
pub struct Stencil {
    pub bytes: &'static [u8],
    pub holes: &'static [Hole],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Label(usize);

#[derive(Debug, Clone, Copy)]
pub enum Patch<'a> {
    I32(i32),
    U64(u64),
    Label(Label),
    Symbol(&'a str),
}

#[derive(Debug)]
enum Target<'a> {
    Label(Label),
    Symbol(&'a str),
}

#[derive(Debug)]
struct Fixup<'a> {
    site: usize,
    kind: PatchKind,
    target: Target<'a>,
}

#[derive(Debug)]
pub struct Relocation<'a> {
    pub offset: u64,
    pub symbol: &'a str,
    pub addend: i64,
    pub flags: object::RelocationFlags,
}

#[derive(Debug)]
pub struct Code<'a> {
    pub bytes: Vec<u8>,
    pub relocations: Vec<Relocation<'a>>,
}

#[derive(Default)]
pub struct Assembler<'a> {
    bytes: Vec<u8>,
    labels: Vec<Option<usize>>,
    fixups: Vec<Fixup<'a>>,
}

impl<'a> Assembler<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn emit_raw(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    pub fn label(&mut self) -> Label {
        let id = Label(self.labels.len());
        self.labels.push(None);
        id
    }

    pub fn bind(&mut self, label: Label) -> Result<()> {
        let slot = self.labels.get_mut(label.0).ok_or(Error::UndefinedLabel)?;
        if slot.replace(self.bytes.len()).is_some() {
            return Err(Error::InvalidStencil("label defined twice"));
        }
        Ok(())
    }

    pub fn emit(&mut self, stencil: &Stencil, patches: &[Patch<'a>]) -> Result<()> {
        if stencil.holes.len() != patches.len() {
            return Err(Error::InvalidStencil("patch count does not match stencil"));
        }
        let base = self.bytes.len();
        let mut end = 0;
        for (hole, patch) in stencil.holes.iter().zip(patches) {
            let width = match hole.kind {
                PatchKind::U64 => 8,
                PatchKind::I32 | PatchKind::X86Rel32 => 4,
            };
            let offset = usize::from(hole.offset);
            if offset < end
                || offset
                    .checked_add(width)
                    .is_none_or(|n| n > stencil.bytes.len())
            {
                return Err(Error::InvalidStencil("overlapping or out-of-bounds hole"));
            }
            end = offset + width;
            if !matches!(
                (hole.kind, patch),
                (PatchKind::I32, Patch::I32(_))
                    | (PatchKind::U64, Patch::U64(_))
                    | (PatchKind::X86Rel32, Patch::Label(_) | Patch::Symbol(_))
            ) {
                return Err(Error::InvalidStencil("patch kind mismatch"));
            }
        }
        self.bytes.extend_from_slice(stencil.bytes);
        for (hole, patch) in stencil.holes.iter().zip(patches) {
            let site = base + usize::from(hole.offset);
            match patch {
                Patch::I32(value) => {
                    self.bytes[site..site + 4].copy_from_slice(&value.to_le_bytes())
                }
                Patch::U64(value) => {
                    self.bytes[site..site + 8].copy_from_slice(&value.to_le_bytes())
                }
                Patch::Label(label) => self.fixups.push(Fixup {
                    site,
                    kind: hole.kind,
                    target: Target::Label(*label),
                }),
                Patch::Symbol(symbol) => self.fixups.push(Fixup {
                    site,
                    kind: hole.kind,
                    target: Target::Symbol(symbol),
                }),
            }
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<Code<'a>> {
        let mut relocations = Vec::new();
        for fixup in self.fixups {
            debug_assert_eq!(fixup.kind, PatchKind::X86Rel32);
            match fixup.target {
                Target::Label(label) => {
                    let target = self
                        .labels
                        .get(label.0)
                        .and_then(|position| *position)
                        .ok_or(Error::UndefinedLabel)?;
                    let displacement = target as i64 - (fixup.site + 4) as i64;
                    let displacement =
                        i32::try_from(displacement).map_err(|_| Error::BranchOutOfRange)?;
                    self.bytes[fixup.site..fixup.site + 4]
                        .copy_from_slice(&displacement.to_le_bytes());
                }
                Target::Symbol(symbol) => relocations.push(Relocation {
                    offset: fixup.site as u64,
                    symbol,
                    addend: -4,
                    flags: object::RelocationFlags::Generic {
                        kind: object::RelocationKind::Relative,
                        encoding: object::RelocationEncoding::X86Branch,
                        size: 32,
                    },
                }),
            }
        }
        Ok(Code {
            bytes: self.bytes,
            relocations,
        })
    }
}
