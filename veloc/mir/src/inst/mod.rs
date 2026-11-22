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

mod storage;
pub(crate) use storage::StoredInst;
pub use storage::{Arguments, Edge as InstEdge, Edges as InstEdges, Successor, Successors};

/// A single-use write into one DFG. Generated methods encode directly into
/// persistent fields; no owning instruction draft is materialized.
pub struct InstWriter<'a> {
    pub(crate) dfg: &'a mut DataFlowGraph,
    pub(crate) target: Option<Inst>,
}
impl InstWriter<'_> {
    /// Copy an instruction in this DFG, retaining its logical inputs and owning its fields.
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
        self.write(fields, values)
    }

    fn write(self, fields: InstFields, values: &[Value]) -> Inst {
        self.dfg.write_inst(self.target, fields, values)
    }

    /// Install an owned instruction head and its operands without type validation.
    pub fn from_fields(self, fields: InstFields, values: &[Value]) -> Inst {
        self.write(fields, values)
    }
}

include!(concat!(env!("OUT_DIR"), "/instructions.rs"));

impl<'a> InstView<'a> {
    /// Return a successor in visitation order, preserving duplicate targets.
    /// Stops at the requested edge; returns None when the index is out of bounds.
    pub fn successor(&self, mut index: usize) -> Option<Successor<'a>> {
        self.try_visit_successors(|successor| {
            if index == 0 {
                return Err(successor);
            }
            index -= 1;
            Ok(())
        })
        .err()
    }

    pub(crate) fn operands_owned(&self) -> Arguments {
        let mut values = Arguments::new();
        self.visit_operands(|value| values.push(value));
        values
    }

    pub fn is_terminator(&self) -> bool {
        self.opcode().spec().is_terminator()
    }

    /// Direct calls inherit the declared memory bound. Volatility is checked separately.
    pub fn memory_effect(&self, decls: &crate::FuncDecls) -> MemoryEffect {
        self.memory_callee()
            .and_then(|callee| decls.get(callee))
            .map_or_else(
                || self.opcode().spec().memory_effect(),
                |decl| decl.effects.memory,
            )
    }

    pub fn has_volatile_access(&self) -> bool {
        self.memory_flags().is_some_and(|flags| flags.is_volatile())
    }

    /// A direct call can rely on its callee's contract or its own site guarantee.
    pub fn may_trap(&self, decls: &crate::FuncDecls) -> bool {
        if !self.opcode().spec().may_trap() || self.is_notrap() {
            return false;
        }
        !self
            .trap_callee()
            .and_then(|callee| decls.get(callee))
            .is_some_and(|decl| decl.effects.notrap)
    }

    /// Memory guarantees alone do not make a call removable: it might never return.
    pub fn can_erase(&self, decls: &crate::FuncDecls) -> bool {
        let spec = self.opcode().spec();
        !spec.is_terminator()
            && !self.may_trap(decls)
            && !self.opcode().transfers_ownership()
            && !self.has_volatile_access()
            && spec.memory_effect().can_erase()
    }

    /// Conservative, context-free speculation. Analyses may prove more.
    pub fn can_speculate(&self) -> bool {
        self.opcode().spec().is_pure()
            && !self.opcode().transfers_ownership()
            && !self.has_volatile_access()
    }

    /// Context-free commoning excludes mutable reads and fresh identities.
    pub fn can_cse(&self) -> bool {
        self.can_speculate()
    }

    pub fn has_side_effects(&self, decls: &crate::FuncDecls) -> bool {
        let spec = self.opcode().spec();
        spec.is_terminator()
            || self.may_trap(decls)
            || self.has_volatile_access()
            || spec.memory_effect().has_side_effects()
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
    use crate::{Block, SuccessorData};

    #[test]
    fn successor_views_preserve_occurrences_and_default_order() {
        let mut dfg = DataFlowGraph::new();
        let first = SuccessorData::new(Block(1), &[Value(1)]);
        let second = SuccessorData::new(Block(2), &[Value(2), Value(3)]);
        let default = SuccessorData::new(Block(3), &[]);
        let inst = dfg.create_inst(|writer| {
            writer.br_table(
                Value(0),
                [first.clone(), second.clone(), first, default]
                    .iter()
                    .map(SuccessorData::as_view),
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
        let view = dfg.inst(inst);
        for (index, successor) in table.iter().enumerate() {
            assert_eq!(view.successor(index), Some(successor));
        }
        assert_eq!(view.successor(4), None);
        assert_eq!(view.successor(usize::MAX), None);
        let (default, cases) = table.split_last().unwrap();
        assert_eq!(default.block, Block(3));
        assert_eq!(cases.len(), 3);
        assert_eq!(
            dfg.operands(inst),
            &[Value(0), Value(1), Value(2), Value(3), Value(1)]
        );
    }

    #[test]
    fn indexed_successors_handle_single_and_absent_edges() {
        let mut dfg = DataFlowGraph::new();
        let edge = SuccessorData::new(Block(1), &[Value(0)]);
        let jump = dfg.create_inst(|writer| writer.jump(edge.as_view()));
        assert_eq!(dfg.inst(jump).successor(0), Some(edge.as_view()));
        assert_eq!(dfg.inst(jump).successor(1), None);
        let nop = dfg.create_inst(|writer| writer.nop());
        assert_eq!(dfg.inst(nop).successor(0), None);
    }
}
