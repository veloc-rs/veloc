//! Instruction kinds, metadata, direct writers and borrowed storage views.

use crate::dfg::{DataFlowGraph, OperandRange};
use crate::{Block, SuccessorData, Value};
use core::fmt;
use cranelift_entity::entity_impl;

pub use veloc_types::{FloatCC, IntCC, MemFlags, MemoryEffect, OpTraits, Region};

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Inst(pub u32);
entity_impl!(Inst, "inst");

pub type Arguments = smallvec::SmallVec<[Value; 4]>;

include!(concat!(env!("OUT_DIR"), "/opcodes.rs"));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeError {
    Arity {
        results: bool,
        expected: usize,
        got: usize,
    },
    Pattern {
        results: bool,
        index: usize,
        expected: &'static str,
        got: crate::Type,
    },
    /// Static diagnostic emitted with a failed type-only requirement.
    Constraint(&'static str),
}

/// Shared executable rules generated from the definitions.
mod type_rules {
    include!(concat!(env!("OUT_DIR"), "/type_rules.rs"));
}

impl Opcode {
    /// Moving an operation with input preconditions onto a new path requires a proof.
    pub const fn can_speculate(self) -> bool {
        self.is_value_only()
            && !self.may_trap()
            && !matches!(
                self,
                Self::IDivS
                    | Self::IDivU
                    | Self::IRemS
                    | Self::IRemU
                    | Self::FloatToIntS
                    | Self::FloatToIntU
                    | Self::InsertElement
                    | Self::ExtractElement
            )
    }

    pub const fn is_terminator(self) -> bool {
        self.traits().contains(OpTraits::TERMINATOR)
    }

    pub const fn is_commutative(self) -> bool {
        self.traits().contains(OpTraits::COMMUTATIVE)
    }
}

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
        self.opcode().is_terminator()
    }

    /// Direct calls inherit the declared memory bound. Volatility is checked separately.
    pub fn memory_effect(&self, decls: &crate::FuncDecls) -> MemoryEffect {
        self.memory_callee()
            .and_then(|callee| decls.get(callee))
            .map_or_else(|| self.opcode().memory_effect(), |decl| decl.effects.memory)
    }

    pub fn has_volatile_access(&self) -> bool {
        self.memory_flags().is_some_and(|flags| flags.is_volatile())
    }

    /// Whether this instruction may release an allocation that already exists.
    pub fn may_free(&self, decls: &crate::FuncDecls) -> bool {
        if let Self::Call { func_id, .. } | Self::TailCall { func_id, .. } = self {
            return decls.get(*func_id).is_none_or(|decl| !decl.effects.nofree);
        }
        matches!(
            self.opcode(),
            Opcode::CallIndirect
                | Opcode::CallIntrinsic
                | Opcode::CallValue
                | Opcode::TailCallValue
                | Opcode::ClosureNew
                | Opcode::ClosureShared
                | Opcode::ClosureLocal
                | Opcode::ClosureDrop
        )
    }

    /// Whether the opcode can trap under the direct callee's contract, if any.
    pub fn may_trap(&self, decls: &crate::FuncDecls) -> bool {
        self.opcode().may_trap()
            && !self
                .trap_callee()
                .and_then(|callee| decls.get(callee))
                .is_some_and(|decl| decl.effects.notrap)
    }

    /// Memory guarantees alone do not make a call removable: it might never return.
    pub fn can_erase(&self, decls: &crate::FuncDecls) -> bool {
        let opcode = self.opcode();
        !opcode.is_terminator()
            && !self.may_trap(decls)
            && !opcode.transfers_ownership()
            && !self.has_volatile_access()
            && !opcode.memory_effect().may_write()
    }

    /// Conservative, context-free speculation. Analyses may prove more.
    pub fn can_speculate(&self) -> bool {
        self.opcode().can_speculate()
            && !self.opcode().transfers_ownership()
            && !self.has_volatile_access()
    }

    /// Context-free commoning excludes mutable reads and fresh identities.
    pub fn can_cse(&self) -> bool {
        self.can_speculate()
    }

    pub fn has_side_effects(&self, decls: &crate::FuncDecls) -> bool {
        let opcode = self.opcode();
        opcode.is_terminator()
            || self.may_trap(decls)
            || self.has_volatile_access()
            || opcode.memory_effect().may_write()
    }
}

impl fmt::Display for InstView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.opcode())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct StoredInst {
    pub operands: OperandRange,
    pub fields: InstFields,
}

impl StoredInst {
    /// Install a jump over operands already retained by the DFG. The old fields
    /// are dropped normally, while the retained operand uses remain attached.
    pub(crate) fn replace_with_jump(&mut self, block: Block, operands: OperandRange) {
        self.fields = InstFields::Jump { dest: block };
        self.operands = operands;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstEdge {
    pub block: Block,
    pub len: u32,
}

impl InstEdge {
    fn store(call: Successor<'_>, values: &mut Arguments) -> Self {
        values.extend_from_slice(call.args);
        Self {
            block: call.block,
            len: call
                .args
                .len()
                .try_into()
                .expect("too many successor arguments"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InstEdges {
    entries: alloc::vec::Vec<InstEdge>,
    len: usize,
}

impl InstEdges {
    pub fn visit_edges(
        &self,
        offset: &mut u32,
        visit: &mut impl FnMut(InstEdge, core::ops::Range<u32>),
    ) {
        for &edge in &self.entries {
            let start = *offset;
            *offset += edge.len;
            visit(edge, start..*offset);
        }
    }

    pub(crate) fn edge_count(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn set_edge_target(&mut self, index: usize, block: Block) {
        self.entries[index].block = block;
    }

    pub(crate) fn set_stored_edge_arg_count(&mut self, index: usize, len: u32) {
        let edge = &mut self.entries[index];
        self.len = self.len - edge.len as usize + len as usize;
        edge.len = len;
    }

    pub fn store<'a>(
        calls: impl IntoIterator<Item = Successor<'a>>,
        values: &mut Arguments,
    ) -> Self {
        let start = values.len();
        let entries = calls
            .into_iter()
            .map(|call| InstEdge::store(call, values))
            .collect();
        Self {
            entries,
            len: values.len() - start,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Successor<'a> {
    pub block: Block,
    pub args: &'a [Value],
}

impl SuccessorData {
    pub fn as_view(&self) -> Successor<'_> {
        Successor {
            block: self.block,
            args: &self.args,
        }
    }
}

/// Borrowed successor groups reconstructed from edge metadata and SSA operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Successors<'a> {
    edges: &'a [InstEdge],
    values: &'a [Value],
}

impl<'a> Successors<'a> {
    pub(crate) fn stored(edges: &'a [InstEdge], values: &'a [Value]) -> Self {
        Self { edges, values }
    }
    pub fn len(self) -> usize {
        self.edges.len()
    }
    pub fn is_empty(self) -> bool {
        self.edges.is_empty()
    }
    pub fn iter(self) -> impl ExactSizeIterator<Item = Successor<'a>> {
        let mut values = self.values;
        self.edges.iter().map(move |edge| {
            let (args, rest) = values.split_at(edge.len as usize);
            values = rest;
            Successor {
                block: edge.block,
                args,
            }
        })
    }
    pub fn split_last(self) -> Option<(Successor<'a>, Self)> {
        let (last, rest) = self.edges.split_last()?;
        let (values, args) = self.values.split_at(self.values.len() - last.len as usize);
        Some((
            Successor {
                block: last.block,
                args,
            },
            Self::stored(rest, values),
        ))
    }
}

struct OperandReader<'a>(&'a [Value]);

impl<'a> OperandReader<'a> {
    fn take(&mut self, len: usize) -> &'a [Value] {
        let (head, tail) = self.0.split_at(len);
        self.0 = tail;
        head
    }
    fn value(&mut self) -> Value {
        self.take(1)[0]
    }
    fn edge(&mut self, edge: InstEdge) -> Successor<'a> {
        Successor {
            block: edge.block,
            args: self.take(edge.len as usize),
        }
    }
    fn edges(&mut self, edges: &'a InstEdges) -> Successors<'a> {
        Successors::stored(&edges.entries, self.take(edges.len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_storage() {
        assert_eq!(size_of::<InstFields>(), 16);
        assert_eq!(size_of::<StoredInst>(), 24);
    }

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
