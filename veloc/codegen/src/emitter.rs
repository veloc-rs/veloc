//! Symbolic code fragments and final layout, shared by all targets.
//!
//! Emission records alternatives; layout chooses encodings only after block and
//! function positions are known. Alignment is part of layout, never a guessed
//! byte count computed by an architecture emitter.
use crate::{Error, Result};
use hashbrown::HashMap;
use std::{format, vec, vec::Vec};
use veloc_encoder::{Encoded, Fixup};
use veloc_lir::{BlockId as Block, SymbolId};

#[derive(Debug, Clone, Copy)]
pub enum Target {
    Block(Block),
    Symbol(SymbolId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocationKind {
    RelativeBranch32,
    Absolute64,
    /// AUIPC/ADDI pair; the low relocation refers to the AUIPC's location.
    RiscvPcRelativeAddress,
    /// A linker-resolved AUIPC/JALR pair with a fixed eight-byte footprint.
    RiscvCall,
}
#[derive(Debug, Clone)]
pub struct ExternalRelocation {
    pub kind: RelocationKind,
    pub offset: u64,
    pub symbol: SymbolId,
    pub addend: i64,
}
#[derive(Debug, Clone)]
pub struct EmittedCode {
    pub data: Vec<u8>,
    pub relocations: Vec<ExternalRelocation>,
}

/// A target patches only its encoding, using a displacement from its start.
/// Out-of-range displacements must return an error, without panicking.
pub type PatchRelative = fn(&mut [u8], i64) -> core::result::Result<(), veloc_encoder::Error>;

#[derive(Debug, Clone, Copy)]
enum Relative {
    Field(Fixup),
    Instruction(PatchRelative),
}
impl Relative {
    fn patch(self, bytes: &mut [u8], displacement: i64) -> Result<()> {
        match self {
            Self::Instruction(patch) => patch(bytes, displacement)
                .map_err(|e| Error::codegen(format!("relative encoding: {e}"))),
            Self::Field(field) => {
                let value =
                    i128::from(displacement) + i128::from(field.addend) - i128::from(field.base);
                let start = usize::from(field.offset);
                let bytes = bytes
                    .get_mut(start..start + usize::from(field.bytes))
                    .ok_or_else(|| Error::codegen("fixup outside instruction bytes"))?;
                let fail =
                    || Error::codegen(format!("relative displacement out of range: {value}"));
                match field.bytes {
                    1 => bytes
                        .copy_from_slice(&i8::try_from(value).map_err(|_| fail())?.to_le_bytes()),
                    4 => bytes
                        .copy_from_slice(&i32::try_from(value).map_err(|_| fail())?.to_le_bytes()),
                    _ => return Err(Error::codegen("unsupported relative field width")),
                }
                Ok(())
            }
        }
    }
}

/// One encoding alternative, ordered from preferred to fallback by the target.
#[derive(Debug, Clone)]
pub struct CodeForm {
    bytes: Vec<u8>,
    relative: Option<Relative>,
    relocations: Vec<ExternalRelocation>,
    alignment: usize,
    padding: Vec<u8>,
}
impl CodeForm {
    pub fn bytes(bytes: &[u8]) -> Self {
        Self {
            bytes: bytes.to_vec(),
            relative: None,
            relocations: Vec::new(),
            alignment: 1,
            padding: Vec::new(),
        }
    }
    pub fn relative(bytes: &[u8], patch: PatchRelative) -> Self {
        Self {
            relative: Some(Relative::Instruction(patch)),
            ..Self::bytes(bytes)
        }
    }
    pub fn relative_field<const N: usize>(encoding: &Encoded<N>) -> Result<Self> {
        let field = encoding
            .fixup
            .ok_or_else(|| Error::codegen("relative form needs a fixup"))?;
        Ok(Self {
            relative: Some(Relative::Field(field)),
            ..Self::bytes(encoding.bytes())
        })
    }
    pub fn relocated(bytes: &[u8], relocation: ExternalRelocation) -> Self {
        Self {
            relocations: vec![relocation],
            ..Self::bytes(bytes)
        }
    }
    pub fn aligned(mut self, alignment: usize, padding: &[u8]) -> Self {
        assert!(alignment.is_power_of_two());
        assert!(!padding.is_empty() && alignment % padding.len() == 0);
        self.alignment = alignment;
        self.padding = padding.to_vec();
        self
    }
}

#[derive(Debug, Clone)]
struct Fragment {
    target: Option<Target>,
    forms: Vec<CodeForm>,
    branch: Option<Branch>,
}

#[derive(Debug, Clone)]
enum Branch {
    Jump,
    Conditional { inverted: Vec<CodeForm> },
}

#[derive(Debug, Clone, Default)]
pub struct Emitter {
    fragments: Vec<Fragment>,
    /// Labels name fragment boundaries, independent of selected encoding sizes.
    labels: HashMap<Block, usize>,
}

impl Emitter {
    pub fn new() -> Self {
        Self::default()
    }
    pub(crate) fn symbols(&self) -> impl Iterator<Item = SymbolId> + '_ {
        self.fragments
            .iter()
            .flat_map(|fragment| fragment.forms.iter())
            .flat_map(|form| form.relocations.iter().map(|relocation| relocation.symbol))
    }
    pub fn mark_block(&mut self, block: Block) {
        self.fold_fallthrough(block);
        assert!(
            self.labels.insert(block, self.fragments.len()).is_none(),
            "duplicate block label"
        );
        // Start a new byte run so coalescing never moves an existing label.
        self.fragments.push(Fragment {
            target: None,
            forms: vec![CodeForm::bytes(&[])],
            branch: None,
        });
    }
    /// A fixed fragment whose start requires target-specific padding.
    pub fn aligned_bytes(&mut self, bytes: &[u8], alignment: usize, padding: &[u8]) {
        self.fragments.push(Fragment {
            target: None,
            forms: vec![CodeForm::bytes(bytes).aligned(alignment, padding)],
            branch: None,
        });
    }

    /// A signed 32-bit offset from this entry to a local block.
    pub fn block_offset(&mut self, block: Block) {
        let form = CodeForm {
            relative: Some(Relative::Field(Fixup {
                offset: 0,
                bytes: 4,
                base: 0,
                addend: 0,
            })),
            ..CodeForm::bytes(&[0; 4])
        };
        self.alternatives(Target::Block(block), vec![form]);
    }

    pub fn bytes(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        if let Some(fragment) = self.fragments.last_mut()
            && fragment.target.is_none()
        {
            // Target-free fragments are ordinary bytes or empty boundaries.
            fragment.forms[0].bytes.extend_from_slice(bytes);
            return;
        }
        self.fragments.push(Fragment {
            target: None,
            forms: vec![CodeForm::bytes(bytes)],
            branch: None,
        });
    }
    /// Alternatives may grow in size or relax a distance restriction. Layout
    /// promotes them monotonically, so alignment cannot cause oscillation.
    pub fn alternatives(&mut self, target: Target, forms: Vec<CodeForm>) {
        assert!(!forms.is_empty());
        self.fragments.push(Fragment {
            target: Some(target),
            forms,
            branch: None,
        });
    }
    /// Explicit branch semantics let final block layout choose fallthroughs
    /// independently of instruction encoding and register allocation.
    pub fn jump(&mut self, target: Block, forms: Vec<CodeForm>) {
        self.alternatives(Target::Block(target), forms);
        self.fragments.last_mut().unwrap().branch = Some(Branch::Jump);
    }

    pub fn conditional_branch(
        &mut self,
        target: Block,
        forms: Vec<CodeForm>,
        inverted: Vec<CodeForm>,
    ) {
        assert!(!inverted.is_empty());
        self.alternatives(Target::Block(target), forms);
        self.fragments.last_mut().unwrap().branch = Some(Branch::Conditional { inverted });
    }

    fn fold_fallthrough(&mut self, next: Block) {
        let Some((jump, prefix)) = self.fragments.split_last_mut() else {
            return;
        };
        if !matches!(jump.branch, Some(Branch::Jump)) {
            return;
        }
        if !matches!(jump.target, Some(Target::Block(block)) if block == next) {
            // branch-if C to next; jump elsewhere => branch-if !C elsewhere.
            let Some(condition) = prefix.last_mut() else {
                return;
            };
            if !matches!(condition.target, Some(Target::Block(block)) if block == next) {
                return;
            }
            let Some(Branch::Conditional { inverted }) = &mut condition.branch else {
                return;
            };
            core::mem::swap(&mut condition.forms, inverted);
            condition.target = jump.target;
        }
        // Keep fragment identities: existing labels refer to boundaries.
        *jump = Fragment {
            target: None,
            forms: vec![CodeForm::bytes(&[])],
            branch: None,
        };
    }
    pub fn instruction<const N: usize>(
        &mut self,
        instruction: &Encoded<N>,
        target: Option<Target>,
    ) -> Result<()> {
        if instruction.fixup.is_none() && target.is_none() {
            self.bytes(instruction.bytes());
            return Ok(());
        }
        let mut form = CodeForm::bytes(instruction.bytes());
        match (instruction.fixup, target) {
            (Some(field), Some(target)) => {
                form.relative = Some(Relative::Field(field));
                if let Target::Symbol(symbol) = target {
                    if field.bytes != 4 {
                        return Err(Error::codegen(
                            "external relocation requires a signed 32-bit field",
                        ));
                    }
                    form.relocations.push(ExternalRelocation {
                        kind: RelocationKind::RelativeBranch32,
                        offset: u64::from(field.offset),
                        symbol,
                        addend: field
                            .addend
                            .checked_add(i64::from(field.offset) - i64::from(field.base))
                            .ok_or_else(|| Error::codegen("relocation addend overflow"))?,
                    });
                }
            }
            _ => return Err(Error::codegen("encoding and symbolic target disagree")),
        }
        self.fragments.push(Fragment {
            target,
            forms: vec![form],
            branch: None,
        });
        Ok(())
    }
    /// Standalone functions cannot bind calls to other independently placed code.
    pub fn finish(self) -> Result<EmittedCode> {
        Ok(layout(&[(None, &self)], 1)?.remove(0).code)
    }
}

pub(crate) struct PlacedCode {
    pub offset: usize,
    pub code: EmittedCode,
}
struct Positions {
    base: usize,
    starts: Vec<usize>,
    end: usize,
}
fn align(position: usize, alignment: usize) -> Result<usize> {
    position
        .checked_add(alignment - 1)
        .map(|p| p & !(alignment - 1))
        .ok_or_else(|| Error::codegen("code layout overflow"))
}

/// Final section layout. Symbols in these units bind within this section;
/// unresolved symbols retain relocations. No byte-changing pass may follow it.
pub(crate) fn layout(
    units: &[(Option<SymbolId>, &Emitter)],
    alignment: usize,
) -> Result<Vec<PlacedCode>> {
    assert!(alignment.is_power_of_two());
    let mut choices: Vec<Vec<usize>> = units
        .iter()
        .map(|(_, e)| vec![0; e.fragments.len()])
        .collect();
    let (positions, symbols) = loop {
        let mut cursor = 0usize;
        let mut positions = Vec::with_capacity(units.len());
        let mut symbols = HashMap::new();
        for ((symbol, emitter), choices) in units.iter().zip(&choices) {
            cursor = align(cursor, alignment)?;
            let base = cursor;
            if let Some(symbol) = symbol {
                if symbols.insert(*symbol, base).is_some() {
                    return Err(Error::codegen("duplicate code symbol"));
                }
            }
            let mut starts = Vec::with_capacity(emitter.fragments.len() + 1);
            for (fragment, &choice) in emitter.fragments.iter().zip(choices) {
                let form = &fragment.forms[choice];
                cursor = align(cursor, form.alignment)?;
                starts.push(cursor);
                cursor = cursor
                    .checked_add(form.bytes.len())
                    .ok_or_else(|| Error::codegen("code layout overflow"))?;
            }
            starts.push(cursor);
            positions.push(Positions {
                base,
                starts,
                end: cursor,
            });
        }
        let mut changed = false;
        for (unit, ((_, emitter), choices)) in units.iter().zip(&mut choices).enumerate() {
            for (index, (fragment, choice)) in emitter.fragments.iter().zip(choices).enumerate() {
                let form = &fragment.forms[*choice];
                let target = resolve(fragment.target, emitter, &positions[unit], &symbols)?;
                let valid = match (form.relative, target) {
                    (Some(relative), Some(target)) => {
                        let displacement = displacement(positions[unit].starts[index], target)?;
                        relative
                            .patch(&mut form.bytes.clone(), displacement)
                            .is_ok()
                    }
                    (Some(_), None) => !form.relocations.is_empty(),
                    (None, _) => true,
                };
                if !valid {
                    if *choice + 1 == fragment.forms.len() {
                        return Err(Error::codegen("no valid encoding for symbolic target"));
                    }
                    *choice += 1;
                    changed = true;
                }
            }
        }
        if !changed {
            break (positions, symbols);
        }
    };
    units
        .iter()
        .zip(choices)
        .zip(positions)
        .map(|(((_, emitter), choices), positions)| {
            let mut code = EmittedCode {
                data: Vec::with_capacity(positions.end - positions.base),
                relocations: Vec::new(),
            };
            for (index, (fragment, choice)) in emitter.fragments.iter().zip(choices).enumerate() {
                let form = &fragment.forms[choice];
                let start = positions.starts[index];
                let padding = start - positions.base - code.data.len();
                if padding != 0 {
                    if form.padding.is_empty() || padding % form.padding.len() != 0 {
                        return Err(Error::codegen(
                            "alignment cannot be filled with target padding",
                        ));
                    }
                    for _ in 0..padding / form.padding.len() {
                        code.data.extend_from_slice(&form.padding);
                    }
                }
                let offset = code.data.len();
                code.data.extend_from_slice(&form.bytes);
                let target = resolve(fragment.target, emitter, &positions, &symbols)?;
                if let (Some(relative), Some(target)) = (form.relative, target) {
                    relative.patch(&mut code.data[offset..], displacement(start, target)?)?;
                } else {
                    for relocation in &form.relocations {
                        let mut relocation = relocation.clone();
                        relocation.offset += offset as u64;
                        code.relocations.push(relocation);
                    }
                }
            }
            Ok(PlacedCode {
                offset: positions.base,
                code,
            })
        })
        .collect()
}
fn displacement(start: usize, target: usize) -> Result<i64> {
    i64::try_from(target as i128 - start as i128)
        .map_err(|_| Error::codegen("code displacement overflow"))
}
fn resolve(
    target: Option<Target>,
    emitter: &Emitter,
    positions: &Positions,
    symbols: &HashMap<SymbolId, usize>,
) -> Result<Option<usize>> {
    match target {
        Some(Target::Block(block)) => emitter
            .labels
            .get(&block)
            .map(|&index| Some(positions.starts[index]))
            .ok_or_else(|| Error::codegen("missing block label")),
        Some(Target::Symbol(symbol)) => Ok(symbols.get(&symbol).copied()),
        None => Ok(None),
    }
}
