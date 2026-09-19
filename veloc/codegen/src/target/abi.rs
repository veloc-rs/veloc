use super::Reg;
use super::types::TargetArch;
use alloc::vec::Vec;
use smallvec::SmallVec;
use veloc_mir::Type;

/// Locations are relative to the ABI argument area, not a concrete frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiLocation {
    Reg(Reg),
    Stack { offset: u32, size: u32, align: u32 },
}

/// One directly transferred value. Split/indirect passing needs an explicit
/// conversion model, not a vector whose consumers only accept one element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbiAssignment {
    pub ty: Type,
    pub loc: AbiLocation,
}

/// Immutable signature-specific protocol, linked to its static ABI definition.
#[derive(Debug, Clone)]
pub struct AbiPlan {
    pub abi: &'static AbiDescriptor,
    pub args: Vec<AbiAssignment>,
    pub returns: Vec<AbiAssignment>,
    pub stack: StackArea,
}

pub use veloc_lir::StackArea;

/// 由 DSL 生成或手工定义的 ABI 描述
#[derive(Debug)]
pub struct AbiDescriptor {
    pub name: &'static str,
    pub arch: TargetArch,
    pub stack: StackArea,
    pub args: AbiAssignFn,
    pub returns: AbiAssignFn,
    pub preserved: &'static [Reg],
    pub clobbers: veloc_lir::RegMask,
}

/// Generated argument/return rules share this interface; neither edits LIR.
pub type AbiAssignFn = fn(Type, &mut AbiState) -> Result<AbiLocation, crate::error::Error>;

/// Occupancy is shared across all type domains. Register IDs identify root
/// storage (register views must resolve to their root before allocation).
pub struct AbiState {
    used: SmallVec<[Reg; 16]>,
    pub(crate) stack: StackArea,
}

impl AbiState {
    pub fn new(stack: StackArea) -> Self {
        assert!(stack.align.is_power_of_two(), "invalid ABI stack alignment");
        Self {
            used: SmallVec::new(),
            stack,
        }
    }

    /// A failed attempt leaves state unchanged. Shadow registers couple
    /// positional slots from otherwise disjoint register lists.
    pub fn assign(&mut self, regs: &[Reg], shadows: &[Reg]) -> Option<AbiLocation> {
        assert!(shadows.is_empty() || shadows.len() == regs.len());
        let (index, &reg) = regs.iter().enumerate().find(|(index, reg)| {
            !self.used.contains(reg)
                && (shadows.is_empty() || !self.used.contains(&shadows[*index]))
        })?;
        self.used.push(reg);
        if let Some(&shadow) = shadows.get(index) {
            self.used.push(shadow);
        }
        Some(AbiLocation::Reg(reg))
    }

    pub fn stack(&mut self, size: u32, align: u32) -> Result<AbiLocation, crate::error::Error> {
        assert!(size != 0 && align.is_power_of_two());
        let overflow = || crate::error::Error::codegen("ABI stack area exceeds supported size");
        let offset = self
            .stack
            .size
            .checked_add(align - 1)
            .ok_or_else(overflow)?
            & !(align - 1);
        let end = offset.checked_add(size).ok_or_else(overflow)?;
        self.stack.size = end;
        self.stack.align = self.stack.align.max(align);
        Ok(AbiLocation::Stack {
            offset,
            size,
            align,
        })
    }
}
