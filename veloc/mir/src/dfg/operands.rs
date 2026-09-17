//! One authoritative, contiguous operand array, with parallel reverse links.
use super::DataFlowGraph;
use crate::{Inst, Value};
use alloc::vec::Vec;
use cranelift_entity::{EntityRef, SecondaryMap, packed_option::PackedOption};
use veloc_collections::{LinkId as Operand, Links};

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct OperandRange {
    pub start: u32,
    pub len: u32,
}

impl OperandRange {
    fn range(self) -> core::ops::Range<usize> {
        self.start as usize..self.start as usize + self.len as usize
    }
}

#[derive(Debug, Clone, Default)]
pub(super) struct Operands {
    values: Vec<Value>,
    links: Links<Inst>,
    heads: SecondaryMap<Value, PackedOption<Operand>>,
    // Power-of-two ranges are recycled as units. No per-use allocation.
    free: Vec<Vec<u32>>,
}

impl Operands {
    pub fn get(&self, range: OperandRange) -> &[Value] {
        &self.values[range.range()]
    }

    pub fn alloc(&mut self, owner: Inst, values: &[Value]) -> OperandRange {
        if values.is_empty() {
            return OperandRange::default();
        }
        let capacity = values
            .len()
            .checked_next_power_of_two()
            .expect("too many operands");
        let class = capacity.trailing_zeros() as usize;
        self.free
            .resize_with(self.free.len().max(class + 1), Vec::new);
        let start = self.free[class].pop().unwrap_or_else(|| {
            let start = u32::try_from(self.values.len()).expect("too many operands");
            let end = self
                .values
                .len()
                .checked_add(capacity)
                .expect("too many operands");
            assert!(end < u32::MAX as usize, "too many operands");
            self.values.resize(end, Value::default());
            self.links.resize(end, owner);
            start
        });
        let range = OperandRange {
            start,
            len: values.len().try_into().expect("too many operands"),
        };
        self.values[range.range()].copy_from_slice(values);
        for offset in 0..range.len {
            let id = Operand::from_u32(start + offset);
            self.links.set_owner(id, owner);
            self.link(id);
        }
        range
    }

    fn link(&mut self, id: Operand) {
        let value = self.values[id.index()];
        self.links.attach(id, &mut self.heads[value]);
    }

    fn unlink(&mut self, id: Operand) {
        let value = self.values[id.index()];
        self.links.detach(id, &mut self.heads[value]);
    }

    pub fn release(&mut self, range: OperandRange) {
        if range.len == 0 {
            return;
        }
        for offset in 0..range.len {
            self.unlink(Operand::from_u32(range.start + offset));
        }
        self.free[range.len.next_power_of_two().trailing_zeros() as usize].push(range.start);
    }

    fn set(&mut self, id: Operand, value: Value) {
        if self.values[id.index()] == value {
            return;
        }
        self.unlink(id);
        self.values[id.index()] = value;
        self.link(id);
    }
}

/// A borrowed operand occurrence, not a stable handle across instruction edits.
///
/// ```compile_fail
/// use veloc_mir::{InstWriter, Opcode, Value};
/// use veloc_mir::dfg::DataFlowGraph;
/// let mut dfg = DataFlowGraph::new();
/// let inst = dfg.create_inst(|writer: crate::InstWriter<'_>| writer.unary(Opcode::INeg, Value(0)));
/// let site = dfg.uses(Value(0)).next().unwrap();
/// dfg.set_operand(inst, 0, Value(1));
/// assert_eq!(site.value(), Value(0)); // A borrowed use cannot cross the edit.
/// ```
#[derive(Clone, Copy)]
pub struct Use<'a> {
    dfg: &'a DataFlowGraph,
    id: Operand,
}

impl Use<'_> {
    pub fn inst(self) -> Inst {
        self.dfg.operands.links.owner(self.id)
    }
    pub fn index(self) -> u32 {
        self.id.as_u32() - self.dfg.instructions[self.inst()].operands.start
    }
    pub fn value(self) -> Value {
        self.dfg.operands.values[self.id.index()]
    }
}

pub struct Uses<'a> {
    dfg: &'a DataFlowGraph,
    next: PackedOption<Operand>,
}

impl<'a> Iterator for Uses<'a> {
    type Item = Use<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        let id = self.next.expand()?;
        self.next = self.dfg.operands.links.next(id);
        Some(Use { dfg: self.dfg, id })
    }
}

impl DataFlowGraph {
    pub fn operands(&self, inst: Inst) -> &[Value] {
        self.operands.get(self.instructions[inst].operands)
    }

    pub fn uses(&self, value: Value) -> Uses<'_> {
        Uses {
            dfg: self,
            next: self.operands.heads[value],
        }
    }

    pub fn use_empty(&self, value: Value) -> bool {
        self.operands.heads[value].is_none()
    }

    pub fn has_one_use(&self, value: Value) -> bool {
        self.operands.heads[value]
            .expand()
            .is_some_and(|id| self.operands.links.next(id).is_none())
    }

    pub fn set_operand(&mut self, inst: Inst, index: u32, value: Value) {
        let range = self.instructions[inst].operands;
        assert!(index < range.len, "operand index out of bounds");
        self.operands
            .set(Operand::from_u32(range.start + index), value);
    }

    pub fn replace_all_uses(&mut self, old: Value, new: Value) {
        if old == new {
            return;
        }
        while let Some(id) = self.operands.heads[old].expand() {
            self.operands.set(id, new);
        }
    }
}
