//! Physical storage and borrowed successor groups. No nested SSA-value pools.
use super::{InstFields, PayloadPool};

/// Out-of-line instruction properties; shared bytes belong to the DFG.
#[derive(Debug, Clone, Default)]
pub(crate) struct FieldPool {
    pub(super) payloads: PayloadPool,
}

use crate::{Block, SuccessorData, Value};

pub type Arguments = smallvec::SmallVec<[Value; 4]>;

#[derive(Debug, Clone)]
pub(crate) struct StoredInst {
    pub operands: crate::dfg::OperandRange,
    pub fields: InstFields,
}

pub(crate) use veloc_collections::{Pool, PoolId as Id};

/// Implemented by generated out-of-line payloads, not arbitrary runtime types.
pub(crate) trait Pooled: Sized {
    fn pool(pools: &super::FieldPool) -> &Pool<Self>;
    fn pool_mut(pools: &mut super::FieldPool) -> &mut Pool<Self>;
}
impl super::FieldPool {
    pub fn push<T: Pooled>(&mut self, value: T) -> Id<T> {
        T::pool_mut(self).push(value)
    }
    pub fn get<T: Pooled>(&self, id: Id<T>) -> &T {
        T::pool(self).get(id)
    }
    #[allow(dead_code)] // Generated remapping uses this for pooled function IDs.
    pub fn get_mut<T: Pooled>(&mut self, id: Id<T>) -> &mut T {
        T::pool_mut(self).get_mut(id)
    }
    pub fn remove<T: Pooled>(&mut self, id: Id<T>) {
        T::pool_mut(self).remove(id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

    pub fn edit_edges(
        &mut self,
        offset: &mut u32,
        edit: &mut impl FnMut(&mut Edge, core::ops::Range<u32>),
    ) {
        self.len = 0;
        for edge in &mut self.entries {
            let start = *offset;
            *offset += edge.len;
            edit(edge, start..*offset);
            self.len += edge.len as usize;
        }
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
        assert_eq!(size_of::<Id<u64>>(), 4);
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
