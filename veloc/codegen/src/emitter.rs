//! Machine-code bytes, local branch fixups, and external relocations.

use alloc::{format, vec::Vec};
use hashbrown::HashMap;
use veloc_lir::SymbolId;
use veloc_mir::Block;

#[derive(Debug, Clone)]
struct BlockFixup {
    disp_offset: usize,
    next_offset: usize,
    target: Block,
}

#[derive(Debug, Clone)]
pub struct ExternalRelocation {
    pub offset: u64,
    pub symbol: SymbolId,
    pub addend: i64,
}

#[derive(Debug, Clone)]
pub struct EmittedCode {
    pub data: Vec<u8>,
    pub relocations: Vec<ExternalRelocation>,
}

/// 机器码发射缓冲区
pub struct Emitter {
    pub data: Vec<u8>,
    block_offsets: HashMap<Block, usize>,
    fixups: Vec<BlockFixup>,
    relocations: Vec<ExternalRelocation>,
}

impl Emitter {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            block_offsets: HashMap::new(),
            fixups: Vec::new(),
            relocations: Vec::new(),
        }
    }

    pub fn write_bytes(&mut self, bytes: &[u8]) {
        self.data.extend_from_slice(bytes);
    }

    #[inline]
    pub fn position(&self) -> usize {
        self.data.len()
    }

    pub fn mark_block(&mut self, block: Block) {
        self.block_offsets.insert(block, self.position());
    }

    pub fn add_block_rel32_fixup(&mut self, disp_offset: usize, next_offset: usize, target: Block) {
        self.fixups.push(BlockFixup {
            disp_offset,
            next_offset,
            target,
        });
    }

    /// Record a PC-relative displacement whose base is the end of its 4-byte field.
    pub fn add_global_rel32_fixup(&mut self, disp_offset: usize, target: SymbolId) {
        self.relocations.push(ExternalRelocation {
            offset: disp_offset as u64,
            symbol: target,
            addend: -4,
        });
    }

    pub fn apply_fixups(&mut self) -> crate::error::Result<()> {
        for &BlockFixup {
            disp_offset,
            next_offset,
            target,
        } in &self.fixups
        {
            let Some(&target_offset) = self.block_offsets.get(&target) else {
                return Err(crate::error::Error::codegen(format!(
                    "missing block offset for {:?}",
                    target
                )));
            };
            let rel = target_offset as i64 - next_offset as i64;
            let disp = i32::try_from(rel).map_err(|_| {
                crate::error::Error::codegen(format!(
                    "relative branch displacement out of range: {}",
                    rel
                ))
            })?;
            self.data[disp_offset..disp_offset + 4].copy_from_slice(&disp.to_le_bytes());
        }
        Ok(())
    }

    pub fn finish(self) -> EmittedCode {
        EmittedCode {
            data: self.data,
            relocations: self.relocations,
        }
    }
}
