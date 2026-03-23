#![no_std]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod backend;
pub mod driver;
pub mod error;
pub mod isle {
    pub use crate::target::x86_64::isle::*;
}
pub mod mir;
pub mod object;
pub mod passes;
pub mod pipeline;
pub mod regalloc;
pub mod target;
pub mod translate;

pub use crate::passes::isel;

pub use backend::Backend;
pub use driver::{CodegenOptions, CodegenPipeline, CodegenStats};
pub use target::arch::{
    CallConv, RewriteResult, SelectResult, TargetArch, TargetConfig, TargetEmitter,
    TargetFrameLowering, TargetInstructionSelector, TargetLegalizer, TargetMachine,
    TargetOperandLowering, TargetPassConfig, TargetPostIsel,
};

/// 根据目标配置创建对应的目标机器
pub fn create_target_machine(config: TargetConfig) -> Option<alloc::boxed::Box<dyn TargetMachine>> {
    use target::arch::TargetArch;
    match config.arch {
        TargetArch::X86_64 => Some(alloc::boxed::Box::new(
            target::x86_64::X86_64TargetMachine::new(config),
        )),
        _ => None,
    }
}

pub use error::{Error, Result};

pub use crate::mir::SymbolId;
pub use alloc::format;
pub use alloc::string::String;
pub use alloc::vec::Vec;
use hashbrown::HashMap;
use veloc_ir::Block;

#[derive(Debug, Clone)]
enum Fixup {
    BlockRel32 {
        disp_offset: usize,
        next_offset: usize,
        target: Block,
    },
    GlobalRel32 {
        disp_offset: usize,
        target: SymbolId,
    },
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
    fixups: Vec<Fixup>,
}

impl Emitter {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            block_offsets: HashMap::new(),
            fixups: Vec::new(),
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
        self.fixups.push(Fixup::BlockRel32 {
            disp_offset,
            next_offset,
            target,
        });
    }

    pub fn add_global_rel32_fixup(
        &mut self,
        disp_offset: usize,
        _next_offset: usize,
        target: SymbolId,
    ) {
        self.fixups.push(Fixup::GlobalRel32 {
            disp_offset,
            target,
        });
    }

    pub fn apply_fixups(&mut self) -> crate::error::Result<()> {
        for fixup in &self.fixups {
            match *fixup {
                Fixup::BlockRel32 {
                    disp_offset,
                    next_offset,
                    target,
                } => {
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
                Fixup::GlobalRel32 { .. } => {}
            }
        }
        Ok(())
    }

    pub fn finish(self) -> EmittedCode {
        let relocations = self
            .fixups
            .into_iter()
            .filter_map(|fixup| match fixup {
                Fixup::GlobalRel32 {
                    disp_offset,
                    target,
                } => Some(ExternalRelocation {
                    offset: disp_offset as u64,
                    symbol: target,
                    addend: -4,
                }),
                Fixup::BlockRel32 { .. } => None,
            })
            .collect();

        EmittedCode {
            data: self.data,
            relocations,
        }
    }
}
