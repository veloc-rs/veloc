//! Instruction kinds, metadata, direct writers and borrowed storage views.

use crate::dfg::DataFlowGraph;
use crate::types::Value;
use core::fmt;
use cranelift_entity::entity_impl;

mod opcode;
pub use opcode::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Inst(pub u32);
entity_impl!(Inst, "inst");

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConstantPoolId(pub u32);
entity_impl!(ConstantPoolId, "const");

mod storage;
pub use storage::{Arguments, Successor, SuccessorMut, Successors};
pub(crate) use storage::{FieldPool, StoredInst};

/// A single-use write into one DFG. Generated methods encode directly into
/// persistent fields; no owning instruction draft is materialized.
pub struct InstWriter<'a> {
    pub(crate) dfg: &'a mut DataFlowGraph,
    pub(crate) target: Option<Inst>,
}
impl InstWriter<'_> {
    /// Copy an instruction in this DFG, retaining its logical inputs but owning new pooled data.
    pub fn copy(self, inst: Inst) -> Inst {
        let values = self.dfg.inst(inst).operands_owned();
        self.copy_with_operands(inst, &values)
    }

    /// Copy properties and replace all inputs as one construction operation.
    pub fn copy_with_operands(self, inst: Inst, values: &[Value]) -> Inst {
        assert_eq!(
            self.dfg.operands(inst).len(),
            values.len(),
            "operand count mismatch"
        );
        let fields = self.dfg.instructions[inst].fields.clone();
        let fields = fields.clone_in(&mut self.dfg.fields);
        self.write(fields, values)
    }

    fn write(self, fields: InstFields, values: &[Value]) -> Inst {
        self.dfg.write_inst(self.target, fields, values)
    }
}

include!(concat!(env!("OUT_DIR"), "/instructions.rs"));

impl InstView<'_> {
    pub(crate) fn operands_owned(&self) -> Arguments {
        let mut values = Arguments::new();
        self.visit_operands(|value| values.push(value));
        values
    }

    pub fn is_terminator(&self) -> bool {
        self.opcode().spec().is_terminator()
    }

    /// Coarse behavior only; per-access volatility is checked separately.
    pub fn memory_effect(&self) -> MemoryEffect {
        self.opcode().spec().memory_effect()
    }

    pub fn has_volatile_access(&self) -> bool {
        self.memory_flags().is_some_and(|flags| flags.is_volatile())
    }

    /// Deletion, speculation and commoning have different preconditions.
    pub fn can_erase(&self) -> bool {
        let spec = self.opcode().spec();
        !spec.is_terminator()
            && !spec.may_trap()
            && !self.opcode().transfers_ownership()
            && !self.has_volatile_access()
            && self.memory_effect().can_erase()
    }

    /// Conservative, context-free speculation. Analyses may prove more.
    pub fn can_speculate(&self) -> bool {
        self.opcode().spec().is_pure()
            && !self.opcode().transfers_ownership()
            && !self.has_volatile_access()
            && self.memory_effect().is_none()
    }

    /// Context-free commoning excludes mutable reads and fresh identities.
    pub fn can_cse(&self) -> bool {
        self.can_speculate()
    }

    pub fn has_side_effects(&self) -> bool {
        let spec = self.opcode().spec();
        spec.is_terminator()
            || spec.may_trap()
            || self.has_volatile_access()
            || self.memory_effect().has_side_effects()
    }
}

impl fmt::Display for InstView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.opcode())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Block, BlockCall};

    #[test]
    fn successor_views_preserve_occurrences_and_default_order() {
        let mut dfg = DataFlowGraph::new();
        let first = BlockCall::new(Block(1), &[Value(1)]);
        let second = BlockCall::new(Block(2), &[Value(2), Value(3)]);
        let default = BlockCall::new(Block(3), &[]);
        let inst = dfg.create_inst(|writer| {
            writer.br_table(
                Value(0),
                [first.clone(), second.clone(), first, default]
                    .iter()
                    .map(BlockCall::as_view),
            )
        });
        let InstView::BrTable { table, .. } = dfg.inst(inst) else {
            unreachable!()
        };
        assert_eq!(
            table
                .iter()
                .map(|c| c.block)
                .collect::<alloc::vec::Vec<_>>(),
            [Block(1), Block(2), Block(1), Block(3)]
        );
        assert_eq!(table.iter().nth(1).unwrap().args, &[Value(2), Value(3)]);
        let (default, cases) = table.split_last().unwrap();
        assert_eq!(default.block, Block(3));
        assert_eq!(cases.len(), 3);
        assert_eq!(
            dfg.operands(inst),
            &[Value(0), Value(1), Value(2), Value(3), Value(1)]
        );
    }
}
