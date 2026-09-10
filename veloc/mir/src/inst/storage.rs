//! Physical storage and borrowed successor groups. No nested SSA-value pools.
use super::InstFields;
use crate::{Block, BlockCall, Value};

pub type Arguments = smallvec::SmallVec<[Value; 4]>;

#[derive(Debug, Clone)]
pub(crate) struct StoredInst {
    pub operands: crate::dfg::OperandRange,
    pub fields: InstFields,
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
