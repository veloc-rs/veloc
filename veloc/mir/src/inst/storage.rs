//! Physical storage and borrowed successor groups. No nested SSA-value pools.
use super::InstFields;

use crate::{Block, SuccessorData, Value};

pub type Arguments = smallvec::SmallVec<[Value; 4]>;

#[derive(Debug, Clone)]
pub(crate) struct StoredInst {
    pub operands: crate::dfg::OperandRange,
    pub fields: InstFields,
}

impl StoredInst {
    /// Install a jump over operands already retained by the DFG. The old fields
    /// are dropped normally, while the retained operand uses remain attached.
    pub(crate) fn replace_with_jump(&mut self, block: Block, operands: crate::dfg::OperandRange) {
        self.fields = InstFields::Jump { dest: block };
        self.operands = operands;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Edge {
    pub block: Block,
    pub len: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Edges {
    entries: alloc::vec::Vec<Edge>,
    len: usize,
}

impl Edges {
    pub fn visit_edges(
        &self,
        offset: &mut u32,
        visit: &mut impl FnMut(Edge, core::ops::Range<u32>),
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
    #[test]
    fn compact_storage() {
        assert_eq!(size_of::<InstFields>(), 16);
        assert_eq!(size_of::<StoredInst>(), 24);
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
