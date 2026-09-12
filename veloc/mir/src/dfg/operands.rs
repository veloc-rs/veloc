//! One authoritative, contiguous operand array, with parallel reverse links.
use super::DataFlowGraph;
use crate::{Inst, Value};
use alloc::vec::Vec;
use cranelift_entity::{EntityRef, SecondaryMap, packed_option::PackedOption};

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

#[derive(Clone, Copy, PartialEq, Eq)]
struct Operand(u32);
cranelift_entity::entity_impl!(Operand, "operand");

#[derive(Debug, Clone)]
struct Link {
    owner: Inst,
    prev: PackedOption<Operand>,
    next: PackedOption<Operand>,
}

#[derive(Debug, Clone, Default)]
pub(super) struct Operands {
    values: Vec<Value>,
    links: Vec<Link>,
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
            self.links.resize(
                end,
                Link {
                    owner,
                    prev: None.into(),
                    next: None.into(),
                },
            );
            start
        });
        let range = OperandRange {
            start,
            len: values.len().try_into().expect("too many operands"),
        };
        self.values[range.range()].copy_from_slice(values);
        for offset in 0..range.len {
            let id = Operand(start + offset);
            self.links[id.index()].owner = owner;
            self.link(id);
        }
        range
    }

    fn link(&mut self, id: Operand) {
        let value = self.values[id.index()];
        let next = self.heads[value];
        self.links[id.index()].prev = None.into();
        self.links[id.index()].next = next;
        if let Some(next) = next.expand() {
            self.links[next.index()].prev = Some(id).into();
        }
        self.heads[value] = Some(id).into();
    }

    fn unlink(&mut self, id: Operand) {
        let link = &self.links[id.index()];
        let (prev, next) = (link.prev, link.next);
        if let Some(prev) = prev.expand() {
            self.links[prev.index()].next = next;
        } else {
            self.heads[self.values[id.index()]] = next;
        }
        if let Some(next) = next.expand() {
            self.links[next.index()].prev = prev;
        }
    }

    pub fn release(&mut self, range: OperandRange) {
        if range.len == 0 {
            return;
        }
        for offset in 0..range.len {
            self.unlink(Operand(range.start + offset));
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
        self.dfg.operands.links[self.id.index()].owner
    }
    pub fn index(self) -> u32 {
        self.id.0 - self.dfg.instructions[self.inst()].operands.start
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
        self.next = self.dfg.operands.links[id.index()].next;
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
            .is_some_and(|id| self.operands.links[id.index()].next.is_none())
    }

    pub fn set_operand(&mut self, inst: Inst, index: u32, value: Value) {
        let range = self.instructions[inst].operands;
        assert!(index < range.len, "operand index out of bounds");
        self.operands.set(Operand(range.start + index), value);
    }

    pub fn replace_all_uses(&mut self, old: Value, new: Value) {
        if old == new {
            return;
        }
        while let Some(id) = self.operands.heads[old].expand() {
            self.operands.set(id, new);
        }
    }

    /// Audit both directions, allocation ownership and free ranges independently.
    pub fn check_uses(&self) -> Result<(), &'static str> {
        let store = &self.operands;
        if store.values.len() != store.links.len() {
            return Err("operand columns differ in length");
        }
        let mut allocated = alloc::vec![false; store.values.len()];
        let mut linked = alloc::vec![false; store.values.len()];
        let mut expected = 0;
        for (inst, data) in self.instructions.iter() {
            let range = data.operands;
            if range.len == 0 {
                continue;
            }
            let capacity = range.len.next_power_of_two() as usize;
            let end = range.start as usize + capacity;
            if end > allocated.len() {
                return Err("operand range out of bounds");
            }
            for slot in &mut allocated[range.start as usize..end] {
                if core::mem::replace(slot, true) {
                    return Err("overlapping operand allocations");
                }
            }
            for index in range.range() {
                if store.links[index].owner != inst {
                    return Err("incorrect operand owner");
                }
                expected += 1;
            }
            let mut decoded = Vec::new();
            self.inst(inst).visit_operands(|value| decoded.push(value));
            if decoded != self.operands(inst) {
                return Err("operand view differs from storage");
            }
        }
        for (class, ranges) in store.free.iter().enumerate() {
            for &start in ranges {
                let end = start as usize + (1usize << class);
                if end > allocated.len() {
                    return Err("free operand range out of bounds");
                }
                for slot in &mut allocated[start as usize..end] {
                    if core::mem::replace(slot, true) {
                        return Err("overlapping free operand ranges");
                    }
                }
            }
        }
        if allocated.iter().any(|&v| !v) {
            return Err("lost operand allocation");
        }
        let mut count = 0;
        for (value, &head) in store.heads.iter() {
            let mut next = head;
            let mut prev = None.into();
            while let Some(id) = next.expand() {
                let Some(slot) = linked.get_mut(id.index()) else {
                    return Err("invalid use link");
                };
                if core::mem::replace(slot, true) {
                    return Err("cyclic or duplicate use link");
                }
                let link = &store.links[id.index()];
                let range = self.instructions[link.owner].operands;
                if !range.range().contains(&id.index()) {
                    return Err("use outside owner range");
                }
                if link.prev != prev || store.values[id.index()] != value {
                    return Err("incorrect use link");
                }
                prev = Some(id).into();
                next = link.next;
                count += 1;
            }
        }
        if count != expected {
            return Err("missing use link");
        }
        Ok(())
    }
}
