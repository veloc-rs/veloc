//! Code layout and symbolic fixups. Architecture encoders only supply bytes
//! and relative-field descriptions; they never own labels or linker symbols.
use crate::{Error, Result};
use alloc::{format, vec, vec::Vec};
use hashbrown::HashMap;
use veloc_encoder::{Encoded, Fixup};
use veloc_lir::BlockId as Block;
use veloc_lir::SymbolId;

#[derive(Debug, Clone, Copy)]
pub enum Target {
    Block(Block),
    Symbol(SymbolId),
}

struct Pending {
    start: usize,
    field: Fixup,
    target: Target,
}
struct Branch {
    start: usize,
    long_len: usize,
    short: Vec<u8>,
    short_field: Fixup,
    long_field: Fixup,
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

#[derive(Default)]
pub struct Emitter {
    data: Vec<u8>,
    labels: HashMap<Block, usize>,
    fixups: Vec<Pending>,
    branches: Vec<Branch>,
}
impl Emitter {
    pub fn new() -> Self {
        Self::default()
    }
    /// Provisional position, before branch relaxation.
    pub fn position(&self) -> usize {
        self.data.len()
    }
    pub fn mark_block(&mut self, block: Block) {
        assert!(
            self.labels.insert(block, self.position()).is_none(),
            "duplicate block label"
        );
    }
    pub fn instruction<const N: usize>(
        &mut self,
        instruction: &Encoded<N>,
        target: Option<Target>,
    ) -> Result<()> {
        match (instruction.fixup, target) {
            (Some(field), Some(target)) => self.fixups.push(Pending {
                start: self.position(),
                field,
                target,
            }),
            (None, None) => {}
            _ => return Err(Error::codegen("encoding and symbolic target disagree")),
        }
        self.data.extend_from_slice(instruction.bytes());
        Ok(())
    }
    pub fn branch<const N: usize>(
        &mut self,
        target: Block,
        short: &Encoded<N>,
        long: &Encoded<N>,
    ) -> Result<()> {
        let (Some(short_field), Some(long_field)) = (short.fixup, long.fixup) else {
            return Err(Error::codegen("branch forms must have relative fixups"));
        };
        if short.bytes().len() >= long.bytes().len()
            || short_field.bytes != 1
            || long_field.bytes != 4
        {
            return Err(Error::codegen("invalid branch relaxation forms"));
        }
        self.branches.push(Branch {
            start: self.position(),
            long_len: long.bytes().len(),
            short: short.bytes().to_vec(),
            short_field,
            long_field,
            target,
        });
        self.data.extend_from_slice(long.bytes());
        Ok(())
    }
    pub fn finish(self) -> Result<EmittedCode> {
        // Start with short branches and only widen. With fixed-size nonbranch
        // fragments (no alignment fragments), distances grow monotonically.
        let mut short = vec![true; self.branches.len()];
        let positions = loop {
            let positions = Positions::new(&self.branches, &short);
            let mut changed = false;
            for (index, branch) in self.branches.iter().enumerate() {
                if !short[index] {
                    continue;
                }
                let target = *self
                    .labels
                    .get(&branch.target)
                    .ok_or_else(|| Error::codegen("missing branch label"))?;
                let displacement = positions.at(target) as i128
                    + i128::from(branch.short_field.addend)
                    - (positions.at(branch.start) + usize::from(branch.short_field.base)) as i128;
                if i8::try_from(displacement).is_err() {
                    short[index] = false;
                    changed = true;
                }
            }
            if !changed {
                break positions;
            }
        };
        let mut result = EmittedCode {
            data: Vec::with_capacity(positions.at(self.data.len())),
            relocations: Vec::new(),
        };
        let mut cursor = 0;
        for (branch, &short) in self.branches.iter().zip(&short) {
            result
                .data
                .extend_from_slice(&self.data[cursor..branch.start]);
            let start = result.data.len();
            if short {
                result.data.extend_from_slice(&branch.short);
            } else {
                result
                    .data
                    .extend_from_slice(&self.data[branch.start..branch.start + branch.long_len]);
            }
            cursor = branch.start + branch.long_len;
            let field = if short {
                branch.short_field
            } else {
                branch.long_field
            };
            let target = positions.at(self.labels[&branch.target]);
            patch(&mut result.data, start, field, target)?;
        }
        result.data.extend_from_slice(&self.data[cursor..]);
        for fixup in self.fixups {
            let start = positions.at(fixup.start);
            match fixup.target {
                Target::Block(block) => {
                    let target = *self
                        .labels
                        .get(&block)
                        .ok_or_else(|| Error::codegen("missing fixup label"))?;
                    patch(&mut result.data, start, fixup.field, positions.at(target))?;
                }
                Target::Symbol(symbol) => {
                    if fixup.field.bytes != 4 {
                        return Err(Error::codegen(
                            "external relocation requires a signed 32-bit field",
                        ));
                    }
                    let addend = fixup
                        .field
                        .addend
                        .checked_add(i64::from(fixup.field.offset) - i64::from(fixup.field.base))
                        .ok_or_else(|| Error::codegen("relocation addend overflow"))?;
                    result.relocations.push(ExternalRelocation {
                        offset: (start + usize::from(fixup.field.offset)) as u64,
                        symbol,
                        addend,
                    });
                }
            }
        }
        Ok(result)
    }
}

// Prefix savings map provisional positions to final positions in O(log B).
struct Positions {
    ends: Vec<usize>,
    savings: Vec<usize>,
}
impl Positions {
    fn new(branches: &[Branch], short: &[bool]) -> Self {
        let mut ends = Vec::with_capacity(branches.len());
        let mut savings = Vec::with_capacity(branches.len() + 1);
        savings.push(0);
        for (b, short) in branches.iter().zip(short) {
            ends.push(b.start + b.long_len);
            savings.push(
                savings.last().copied().unwrap()
                    + if *short {
                        b.long_len - b.short.len()
                    } else {
                        0
                    },
            );
        }
        Self { ends, savings }
    }
    fn at(&self, position: usize) -> usize {
        position - self.savings[self.ends.partition_point(|end| *end <= position)]
    }
}
fn patch(data: &mut [u8], start: usize, field: Fixup, target: usize) -> Result<()> {
    let value =
        target as i128 + i128::from(field.addend) - (start + usize::from(field.base)) as i128;
    let offset = start + usize::from(field.offset);
    let fail = || Error::codegen(format!("relative displacement out of range: {value}"));
    let bytes = data
        .get_mut(offset..offset + usize::from(field.bytes))
        .ok_or_else(|| Error::codegen("fixup outside instruction bytes"))?;
    match field.bytes {
        1 => bytes.copy_from_slice(&i8::try_from(value).map_err(|_| fail())?.to_le_bytes()),
        4 => bytes.copy_from_slice(&i32::try_from(value).map_err(|_| fail())?.to_le_bytes()),
        _ => return Err(Error::codegen("unsupported relative field width")),
    }
    Ok(())
}
