//! Physical storage and borrowed successor groups. No nested SSA-value pools.
use super::{ConstantPoolId, InstFields, PayloadPool};
use alloc::sync::Arc;
use core::{borrow::Borrow, hash::Hash};
use cranelift_entity::{EntityRef, PrimaryMap};
use hashbrown::HashMap;

/// Owns out-of-line data; SSA references remain in the DFG operand store.
#[derive(Debug, Clone, Default)]
pub(crate) struct FieldPool {
    pub(super) payloads: PayloadPool,
    constants: InternPool<ConstantPoolId, Arc<[u8]>>,
}

/// Immutable, shared entries. IDs remain valid for the lifetime of the pool.
/// No mutation or per-instruction removal is exposed.
#[derive(Debug, Clone)]
struct InternPool<K: EntityRef, T> {
    values: PrimaryMap<K, T>,
    index: HashMap<T, K>,
}
impl<K: EntityRef, T> Default for InternPool<K, T> {
    fn default() -> Self {
        Self {
            values: PrimaryMap::new(),
            index: HashMap::new(),
        }
    }
}
impl<K: EntityRef, T: Clone + Eq + Hash> InternPool<K, T> {
    fn intern<Q: ?Sized + Eq + Hash>(&mut self, value: &Q, make: impl FnOnce() -> T) -> K
    where
        T: Borrow<Q>,
    {
        if let Some(&id) = self.index.get(value) {
            return id;
        }
        let value = make();
        let id = self.values.push(value.clone());
        self.index.insert(value, id);
        id
    }
    fn get(&self, id: K) -> Option<&T> {
        self.values.get(id)
    }
}

impl FieldPool {
    pub(crate) fn intern(&mut self, bytes: &[u8]) -> ConstantPoolId {
        self.constants.intern(bytes, || Arc::from(bytes))
    }
    pub(crate) fn constant(&self, id: ConstantPoolId) -> Option<&[u8]> {
        self.constants.get(id).map(AsRef::as_ref)
    }
}

use crate::{Block, BlockCall, Value};

pub type Arguments = smallvec::SmallVec<[Value; 4]>;

#[derive(Debug, Clone)]
pub(crate) struct StoredInst {
    pub operands: crate::dfg::OperandRange,
    pub fields: InstFields,
}

pub(crate) use crate::constant::{ScalarBits, VectorBits};

/// Private, type-indexed handles. A handle belongs to its DFG; replacement
/// releases it before reuse. Handles never escape through public IR views.
pub(crate) struct Id<T>(u32, core::marker::PhantomData<fn() -> T>);
impl<T> Copy for Id<T> {}
impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> core::fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Pool<T> {
    slots: alloc::vec::Vec<Slot<T>>,
    free: Option<u32>,
}
impl<T> Default for Pool<T> {
    fn default() -> Self {
        Self {
            slots: alloc::vec::Vec::new(),
            free: None,
        }
    }
}
#[derive(Debug, Clone)]
enum Slot<T> {
    Live(T),
    Free(Option<u32>),
}
impl<T> Pool<T> {
    pub fn push(&mut self, value: T) -> Id<T> {
        let index = if let Some(index) = self.free {
            let Slot::Free(next) = self.slots[index as usize] else {
                unreachable!("free pool slot")
            };
            self.free = next;
            self.slots[index as usize] = Slot::Live(value);
            index
        } else {
            let index = self.slots.len().try_into().expect("too many pooled fields");
            self.slots.push(Slot::Live(value));
            index
        };
        Id(index, core::marker::PhantomData)
    }
    pub fn get(&self, id: Id<T>) -> &T {
        let Slot::Live(value) = &self.slots[id.0 as usize] else {
            unreachable!("live pool slot")
        };
        value
    }
    #[allow(dead_code)] // Used when a generated pooled payload contains remappable IDs.
    pub fn get_mut(&mut self, id: Id<T>) -> &mut T {
        let Slot::Live(value) = &mut self.slots[id.0 as usize] else {
            unreachable!("live pool slot")
        };
        value
    }
    pub fn remove(&mut self, id: Id<T>) {
        assert!(matches!(self.slots[id.0 as usize], Slot::Live(_)));
        self.slots[id.0 as usize] = Slot::Free(self.free);
        self.free = Some(id.0);
    }
}

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
    #[test]
    fn compact_storage_and_typed_pool_recycling() {
        assert_eq!(size_of::<InstFields>(), 16);
        assert_eq!(size_of::<StoredInst>(), 24);
        assert_eq!(size_of::<Id<u64>>(), 4);
        let mut pool = Pool::<u64>::default();
        let id = pool.push(7);
        assert_eq!(*pool.get(id), 7);
        let cloned = pool.clone();
        pool.remove(id);
        let next = pool.push(9);
        assert_eq!(id.0, next.0);
        assert_eq!(*pool.get(next), 9);
        assert_eq!(*cloned.get(id), 7);
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

/// Borrowed successor groups reconstructed from edge metadata and SSA operands.
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

/// A single successor occurrence during an edit. Resizing its arguments preserves
/// all other operand groups, including other edges to the same block.
pub struct SuccessorMut<'a> {
    edge: &'a mut Edge,
    values: &'a mut Arguments,
    offset: usize,
}

impl SuccessorMut<'_> {
    pub(crate) fn edit_call(call: &mut BlockCall, f: &mut impl FnMut(&mut SuccessorMut<'_>)) {
        let mut edge = Edge {
            block: call.block,
            len: call.args.len().try_into().expect("too many arguments"),
        };
        f(&mut SuccessorMut {
            edge: &mut edge,
            values: &mut call.args,
            offset: 0,
        });
        call.block = edge.block;
    }
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
