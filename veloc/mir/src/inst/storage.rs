//! Physical storage and borrowed successor groups. No nested SSA-value pools.
use super::{InstFields, PackedFields};
use crate::{Block, BlockCall, Value};

pub type Arguments = smallvec::SmallVec<[Value; 4]>;

#[derive(Debug, Clone)]
pub(crate) struct StoredInst {
    pub operands: crate::dfg::OperandRange,
    pub fields: PackedFields,
}

pub(crate) use crate::constant::{ScalarBits, VectorBits};

/// Only out-of-line layouts occupy slots. Replacement and erasure recycle them.
#[derive(Debug, Clone, Default)]
pub(crate) struct FieldPool {
    slots: alloc::vec::Vec<FieldSlot>,
    free: Option<u32>,
}

#[derive(Debug, Clone)]
enum FieldSlot {
    Live(InstFields),
    Free(Option<u32>),
}

impl FieldPool {
    pub fn insert(&mut self, fields: InstFields) -> u32 {
        if let Some(id) = self.free {
            let FieldSlot::Free(next) = self.slots[id as usize] else {
                unreachable!("free field slot")
            };
            self.free = next;
            self.slots[id as usize] = FieldSlot::Live(fields);
            id
        } else {
            let id = self.slots.len().try_into().expect("too many field slots");
            self.slots.push(FieldSlot::Live(fields));
            id
        }
    }

    pub fn get(&self, id: u32) -> &InstFields {
        let FieldSlot::Live(fields) = &self.slots[id as usize] else {
            unreachable!("live field slot")
        };
        fields
    }

    pub fn get_mut(&mut self, id: u32) -> &mut InstFields {
        let FieldSlot::Live(fields) = &mut self.slots[id as usize] else {
            unreachable!("live field slot")
        };
        fields
    }

    pub fn remove(&mut self, id: u32) {
        assert!(matches!(self.slots[id as usize], FieldSlot::Live(_)));
        self.slots[id as usize] = FieldSlot::Free(self.free);
        self.free = Some(id);
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Edge {
    pub block: Block,
    pub len: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct Edges {
    entries: alloc::vec::Vec<Edge>,
    len: usize,
}

impl Edges {
    pub fn store<'a>(
        calls: impl IntoIterator<Item = Successor<'a>>,
        values: &mut Arguments,
    ) -> Self {
        let start = values.len();
        let entries = calls
            .into_iter()
            .map(|call| store_edge(call, values))
            .collect();
        Self {
            entries,
            len: values.len() - start,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Float, InstDraft, Int, Opcode, Type, VectorConst};

    #[test]
    fn persistent_layout_and_pool_costs() {
        assert_eq!(size_of::<ScalarBits>(), 9);
        assert_eq!(size_of::<VectorBits>(), 11);
        assert_eq!(size_of::<PackedFields>(), 16);
        assert_eq!(size_of::<StoredInst>(), 24);
        // A cold instruction pays for this slot in addition to its 24-byte header.
        assert_eq!(size_of::<FieldSlot>(), 32);
    }

    #[test]
    fn fields_roundtrip_and_recycle_without_changing_drafts() {
        let a = BlockCall::new(Block(1), &[Value(2), Value(3)]);
        let b = BlockCall::new(Block(2), &[Value(4)]);
        let vector = VectorConst::splat((-7i32).into(), 4, true).unwrap();
        let drafts = [
            InstDraft::binary(Opcode::IAdd, [Value(0), Value(1)]),
            InstDraft::iconst(Int::from_bits(Type::I64, u64::MAX).unwrap()),
            InstDraft::fconst(Float::from_f32_bits(0x7fa12345)),
            InstDraft::fconst(Float::from_f64_bits(0x8000000000000000)),
            InstDraft::vconst(vector),
            InstDraft::vconst(VectorConst::dense(
                Type::I32X4.as_vector().unwrap(),
                super::super::ConstantPoolId(u32::MAX),
            )),
            InstDraft::call(crate::FuncId(3), &[Value(0), Value(1)]),
            InstDraft::br(Value(0), a.as_view(), b.as_view()),
            InstDraft::br_table(Value(0), [a.as_view(), b.as_view()]),
        ];
        let mut pool = FieldPool::default();
        for draft in drafts {
            let expected = format!("{:?}", draft.as_view());
            let opcode = draft.opcode();
            let InstDraft { fields, operands } = draft;
            let packed = fields.pack(&mut pool);
            assert_eq!(
                matches!(packed, PackedFields::OutOfLine(_)),
                opcode == Opcode::BrTable
            );
            assert_eq!(packed.opcode(&pool), opcode);
            let view = packed.view(&operands, &pool);
            assert_eq!(format!("{view:?}"), expected);
            assert_eq!(format!("{:?}", view.to_draft().as_view()), expected);
            let mut cloned = pool.clone();
            packed.release(&mut pool);
            // Cloning a DFG's pool must not share mutable slots with the original.
            assert_eq!(format!("{:?}", packed.view(&operands, &cloned)), expected);
            packed.release(&mut cloned);
        }
        // Only the branch table needs a slot; constants remain inline too.
        assert_eq!(pool.slots.len(), 1);
        for _ in 0..100 {
            let packed = InstDraft::br_table(Value(0), [a.as_view(), b.as_view()])
                .fields
                .pack(&mut pool);
            assert!(matches!(packed, PackedFields::OutOfLine(0)));
            packed.release(&mut pool);
        }
        assert_eq!(pool.slots.len(), 1);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Successor<'a> {
    pub block: Block,
    pub args: &'a [Value],
}

impl BlockCall {
    pub fn as_view(&self) -> Successor<'_> {
        Successor {
            block: self.block,
            args: &self.args,
        }
    }
}

/// Borrowed successor groups in the shared draft/DFG layout.
#[derive(Debug, Clone, Copy)]
pub struct Successors<'a> {
    edges: &'a [Edge],
    values: &'a [Value],
}

impl<'a> Successors<'a> {
    pub(crate) fn stored(edges: &'a [Edge], values: &'a [Value]) -> Self {
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

pub(crate) struct OperandReader<'a>(pub &'a [Value]);

impl<'a> OperandReader<'a> {
    pub fn take(&mut self, len: usize) -> &'a [Value] {
        let (head, tail) = self.0.split_at(len);
        self.0 = tail;
        head
    }
    pub fn value(&mut self) -> Value {
        self.take(1)[0]
    }
    pub fn edge(&mut self, edge: Edge) -> Successor<'a> {
        Successor {
            block: edge.block,
            args: self.take(edge.len as usize),
        }
    }
    pub fn edges(&mut self, edges: &'a Edges) -> Successors<'a> {
        Successors::stored(&edges.entries, self.take(edges.len))
    }
}

pub(crate) fn store_edge(call: Successor<'_>, values: &mut Arguments) -> Edge {
    values.extend_from_slice(call.args);
    Edge {
        block: call.block,
        len: call
            .args
            .len()
            .try_into()
            .expect("too many successor arguments"),
    }
}

/// A single successor occurrence in a draft. Resizing its arguments preserves
/// all other operand groups, including other edges to the same block.
pub struct SuccessorMut<'a> {
    edge: &'a mut Edge,
    values: &'a mut Arguments,
    offset: usize,
}

impl SuccessorMut<'_> {
    pub fn block(&self) -> Block {
        self.edge.block
    }

    pub fn args(&self) -> &[Value] {
        &self.values[self.offset..self.offset + self.edge.len as usize]
    }

    pub fn set_block(&mut self, block: Block) {
        self.edge.block = block;
    }

    pub fn set_args(&mut self, args: &[Value]) {
        let len = u32::try_from(args.len()).expect("too many successor arguments");
        let end = self.offset + self.edge.len as usize;
        self.values.drain(self.offset..end);
        self.values.insert_from_slice(self.offset, args);
        self.edge.len = len;
    }

    /// Set a construction-time argument, filling incomplete earlier positions.
    /// Explicit validation checks the completed edge's parameter contract.
    pub fn set_arg(&mut self, index: usize, value: Value) {
        if index >= self.edge.len as usize {
            let len = u32::try_from(index.checked_add(1).expect("too many successor arguments"))
                .expect("too many successor arguments");
            let end = self.offset + self.edge.len as usize;
            self.values.insert_many(
                end,
                core::iter::repeat_n(value, (len - self.edge.len) as usize),
            );
            self.edge.len = len;
        } else {
            self.values[self.offset + index] = value;
        }
    }
}

impl Edge {
    pub(super) fn edit(
        &mut self,
        values: &mut Arguments,
        offset: &mut usize,
        f: &mut impl FnMut(&mut SuccessorMut<'_>),
    ) {
        f(&mut SuccessorMut {
            edge: self,
            values,
            offset: *offset,
        });
        *offset += self.len as usize;
    }
}

impl Edges {
    pub(super) fn edit(
        &mut self,
        values: &mut Arguments,
        offset: &mut usize,
        f: &mut impl FnMut(&mut SuccessorMut<'_>),
    ) {
        let start = *offset;
        for edge in &mut self.entries {
            edge.edit(values, offset, f);
        }
        self.len = *offset - start;
    }
}
