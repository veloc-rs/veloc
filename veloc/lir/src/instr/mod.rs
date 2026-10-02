//! Instruction definitions, operand contracts and borrowed instruction access.
mod constraints;
mod control;
mod fields;
mod register_access;
mod validation;

pub use constraints::*;
pub use control::*;
pub(crate) use fields::FieldPools;
pub use fields::{FieldBuild, FieldView, Fields};
pub use register_access::RegisterAccess;
pub use validation::{Result, TypeError, ValidationError};

use crate::Reg;
use cranelift_entity::entity_impl;
pub use veloc_types::{MemFlags, MemoryEffect, MemoryEffects, OpTraits};

/// 机器指令索引
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InstId(u32);
entity_impl!(InstId, "inst");

include!(concat!(env!("OUT_DIR"), "/instructions.rs"));

/// 机器指令操作码
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MachineOpcode {
    /// 无效指令（占位符，用于指令融合或删除）
    Invalid,
    /// 通用操作码（需要指令选择）
    Generic(GenericOpcode),
    /// 目标架构特定操作码（指令选择后）
    Target(u32),
}

/// Borrowed access to an instruction in its function's store.
#[derive(Clone, Copy)]
pub struct InstRef<'a> {
    pub(crate) store: &'a crate::InstStore,
    pub(crate) id: InstId,
}

impl core::fmt::Debug for InstRef<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("InstRef")
            .field("opcode", &self.opcode())
            .field("results", &self.results())
            .field("inputs", &self.inputs())
            .field("fields", &self.fields())
            .field(
                "clobbers",
                &self.clobbers().collect::<smallvec::SmallVec<[Reg; 4]>>(),
            )
            .finish()
    }
}

impl<'a> crate::InstRead<'a> for crate::InstRef<'a> {
    type Error = crate::ValidationError;
    fn opcode(self) -> Option<crate::GenericOpcode> {
        self.generic_opcode()
    }
    fn results(self) -> &'a [crate::Reg] {
        self.results()
    }
    fn inputs(self) -> &'a [crate::Reg] {
        self.inputs()
    }
    fn fields(self) -> crate::FieldView<'a> {
        self.fields()
    }
    fn error(self, message: &str) -> crate::ValidationError {
        crate::ValidationError {
            opcode: self.opcode(),
            reason: message.into(),
        }
    }
}

impl<'a> InstRef<'a> {
    /// Symbolic stack boundaries survive instruction selection and allocation.
    pub fn is_call_frame(&self) -> bool {
        matches!(
            self.opcode(),
            MachineOpcode::Generic(GenericOpcode::CallFrameSetup | GenericOpcode::CallFrameDestroy)
        )
    }
    /// Successor identities in instruction operand order, including repeated targets.
    pub fn edge_ids(self) -> impl Iterator<Item = crate::EdgeId> + 'a {
        self.store.edge_ids(self.id)
    }

    pub fn edge(self, id: crate::EdgeId) -> crate::Successor<&'a [Reg]> {
        assert!(
            self.store.edge_ids(self.id).any(|edge| edge == id),
            "edge belongs to another instruction"
        );
        self.store.edge(id)
    }
    /// Conservative, nontrapping value computation. This deliberately excludes
    /// loads, allocation, physical-register dependencies and auxiliary effects.
    pub fn is_pure_value(self) -> bool {
        let Some(opcode) = self.generic_opcode() else {
            return false;
        };
        let meta = opcode.meta();
        opcode.control() == crate::ControlFlow::Next
            && meta.memory.is_none()
            && !meta
                .traits
                .intersects(OpTraits::MAY_TRAP | OpTraits::ABORT | OpTraits::TERMINATOR)
            && self.mem_flags().is_none()
            && self.clobbers().next().is_none()
            && !self.results().is_empty()
            && self
                .results()
                .iter()
                .chain(self.inputs())
                .all(|reg| reg.is_vreg())
            && self.store.call_info(self.id).is_none()
    }

    pub fn opcode(self) -> MachineOpcode {
        self.store.opcode(self.id)
    }
    pub fn results(self) -> &'a [Reg] {
        self.store.results(self.id)
    }
    pub fn inputs(self) -> &'a [Reg] {
        self.store.inputs(self.id)
    }
    pub fn fields(self) -> crate::FieldView<'a> {
        self.store.fields(self.id)
    }
    pub fn constraints(self) -> &'a [crate::OperandConstraint] {
        self.store.constraints(self.id)
    }
    pub fn mem_flags(self) -> Option<crate::MemFlags> {
        self.fields().mem_flags()
    }

    /// Distinct register access for dependency, liveness and pressure analyses.
    /// Operand APIs retain their positions and multiplicity.
    pub fn register_access(self) -> crate::RegisterAccess<'a> {
        crate::RegisterAccess::new(self)
    }

    /// 检查指令是否有效
    pub fn is_invalid(&self) -> bool {
        matches!(self.opcode(), MachineOpcode::Invalid)
    }

    /// Instruction and ABI destruction effects have no use-def occurrences.
    pub fn clobbers(self) -> impl Iterator<Item = Reg> + 'a {
        self.store.clobbers(self.id).iter().copied().chain(
            self.store
                .call_info(self.id)
                .into_iter()
                .flat_map(|info| info.clobbers.iter()),
        )
    }

    /// Result definitions; clobbers are separate destruction effects.
    pub fn defs(self) -> impl Iterator<Item = Reg> + 'a {
        self.results().iter().copied()
    }

    /// Register inputs and edge arguments, including explicit physical inputs.
    /// Repeated operands remain separate occurrences.
    pub fn uses(self) -> impl Iterator<Item = Reg> + 'a {
        let operands = self.inputs().iter().copied();
        operands.chain(self.store.edge_args(self.id))
    }

    /// 检查是否是通用操作码（尚未指令选择）
    pub fn is_generic(&self) -> bool {
        matches!(self.opcode(), MachineOpcode::Generic(_))
    }

    /// 检查是否是目标特定操作码
    pub fn is_target(&self) -> bool {
        matches!(self.opcode(), MachineOpcode::Target(_))
    }

    /// 如果该指令是通用 LIR 指令，返回其 GenericOpcode。
    pub fn generic_opcode(&self) -> Option<GenericOpcode> {
        match self.opcode() {
            MachineOpcode::Generic(opcode) => Some(opcode),
            _ => None,
        }
    }
}
