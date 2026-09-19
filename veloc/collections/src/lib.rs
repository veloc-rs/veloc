//! Indexed storage primitives shared by IRs, independent of their value semantics.
#![no_std]
extern crate alloc;

mod pool;
pub use pool::{Pool, PoolId};

mod layout;
pub use layout::EntityLayout;

use alloc::vec::Vec;
use cranelift_entity::{EntityRef, packed_option::PackedOption};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct LinkId(u32);
cranelift_entity::entity_impl!(LinkId, "link");

#[derive(Debug, Clone, Copy)]
struct Link<O> {
    owner: O,
    prev: PackedOption<LinkId>,
    next: PackedOption<LinkId>,
}

/// Intrusive reverse links parallel to a caller-owned slot array.
/// Heads and referenced values stay with the caller: links never duplicate them.
#[derive(Debug, Clone)]
pub struct Links<O> {
    nodes: Vec<Link<O>>,
}

impl<O> Default for Links<O> {
    fn default() -> Self {
        Self { nodes: Vec::new() }
    }
}

impl<O: Copy> Links<O> {
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Reserve parallel slots, including unused capacity in a pooled range.
    pub fn resize(&mut self, len: usize, owner: O) {
        assert!(
            len >= self.nodes.len(),
            "links cannot be truncated while live"
        );
        self.nodes.resize(
            len,
            Link {
                owner,
                prev: None.into(),
                next: None.into(),
            },
        );
    }

    pub fn owner(&self, id: LinkId) -> O {
        self.nodes[id.index()].owner
    }
    pub fn set_owner(&mut self, id: LinkId, owner: O) {
        self.nodes[id.index()].owner = owner;
    }
    pub fn next(&self, id: LinkId) -> PackedOption<LinkId> {
        self.nodes[id.index()].next
    }
    pub fn prev(&self, id: LinkId) -> PackedOption<LinkId> {
        self.nodes[id.index()].prev
    }

    /// Insert a detached slot at the head of its referenced value's list.
    pub fn attach(&mut self, id: LinkId, head: &mut PackedOption<LinkId>) {
        self.nodes[id.index()].prev = None.into();
        self.nodes[id.index()].next = *head;
        if let Some(next) = head.expand() {
            self.nodes[next.index()].prev = Some(id).into();
        }
        *head = Some(id).into();
    }

    /// Remove a live slot in O(1). The caller supplies its current value's head.
    pub fn detach(&mut self, id: LinkId, head: &mut PackedOption<LinkId>) {
        let node = self.nodes[id.index()];
        if let Some(prev) = node.prev.expand() {
            self.nodes[prev.index()].next = node.next;
        } else {
            assert_eq!(head.expand(), Some(id), "slot is not linked to this head");
            *head = node.next;
        }
        if let Some(next) = node.next.expand() {
            self.nodes[next.index()].prev = node.prev;
        }
        self.nodes[id.index()].prev = None.into();
        self.nodes[id.index()].next = None.into();
    }
}
