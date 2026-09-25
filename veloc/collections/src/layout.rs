//! Stable IDs with independently editable block and instruction order.
//!
//! Links live in indexed storage, not separately allocated nodes. Insertion and
//! removal update only neighboring links; instruction ownership stays directly
//! queryable. Iteration borrows the layout, so mutating passes use stable anchors
//! or explicitly collect a snapshot when they need one.
use alloc::sync::Arc;
use cranelift_entity::{
    EntityRef, SecondaryMap,
    packed_option::{PackedOption, ReservedValue},
};

#[derive(Debug, Clone)]
struct BlockNode<Block: EntityRef + ReservedValue, Inst: EntityRef + ReservedValue> {
    prev: PackedOption<Block>,
    next: PackedOption<Block>,
    inserted: bool,
    first: PackedOption<Inst>,
    last: PackedOption<Inst>,
}

#[derive(Debug, Clone)]
struct InstNode<Block: EntityRef + ReservedValue, Inst: EntityRef + ReservedValue> {
    block: PackedOption<Block>,
    prev: PackedOption<Inst>,
    next: PackedOption<Inst>,
    /// Changes on every placement, even when an ID is detached and reused.
    stamp: u64,
}

#[derive(Debug)]
pub struct EntityLayout<Block: EntityRef + ReservedValue, Inst: EntityRef + ReservedValue> {
    blocks: SecondaryMap<Block, BlockNode<Block, Inst>>,
    first: PackedOption<Block>,
    last: PackedOption<Block>,
    insts: SecondaryMap<Inst, InstNode<Block, Inst>>,
    len: usize,
    identity: Arc<()>,
    stamp: u64,
}

impl<Block: EntityRef + ReservedValue, Inst: EntityRef + ReservedValue> Clone
    for EntityLayout<Block, Inst>
{
    fn clone(&self) -> Self {
        Self {
            blocks: self.blocks.clone(),
            first: self.first,
            last: self.last,
            insts: self.insts.clone(),
            len: self.len,
            stamp: self.stamp,
            // A clone can diverge independently, so it cannot share cache identity.
            identity: Arc::new(()),
        }
    }
}

impl<Block: EntityRef + ReservedValue, Inst: EntityRef + ReservedValue> Default
    for BlockNode<Block, Inst>
{
    fn default() -> Self {
        Self {
            prev: None.into(),
            next: None.into(),
            inserted: false,
            first: None.into(),
            last: None.into(),
        }
    }
}
impl<Block: EntityRef + ReservedValue, Inst: EntityRef + ReservedValue> Default
    for InstNode<Block, Inst>
{
    fn default() -> Self {
        Self {
            block: None.into(),
            prev: None.into(),
            next: None.into(),
            stamp: 0,
        }
    }
}
impl<Block: EntityRef + ReservedValue, Inst: EntityRef + ReservedValue> Default
    for EntityLayout<Block, Inst>
{
    fn default() -> Self {
        Self {
            blocks: SecondaryMap::new(),
            insts: SecondaryMap::new(),
            first: None.into(),
            last: None.into(),
            len: 0,
            identity: Arc::new(()),
            stamp: 0,
        }
    }
}
impl<Block: EntityRef + ReservedValue, Inst: EntityRef + ReservedValue> EntityLayout<Block, Inst> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_capacity(blocks: usize, insts: usize) -> Self {
        Self {
            blocks: SecondaryMap::with_capacity(blocks),
            first: None.into(),
            last: None.into(),
            insts: SecondaryMap::with_capacity(insts),
            len: 0,
            identity: Arc::new(()),
            stamp: 0,
        }
    }
    pub fn remove_insts(&mut self, insts: &[Inst]) {
        for &inst in insts {
            self.detach_inst(inst);
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn prev_block(&self, block: Block) -> Option<Block> {
        self.blocks[block].prev.expand()
    }
    /// Placed blocks in physical order, independently of CFG traversal order.
    pub fn block_order(&self) -> impl DoubleEndedIterator<Item = Block> + '_ {
        Order {
            front: self.first.expand(),
            back: self.last.expand(),
            links: |b| (self.blocks[b].prev.expand(), self.blocks[b].next.expand()),
        }
    }
    /// Instructions in execution order; yields stable IDs, not array positions.
    pub fn block_insts(&self, block: Block) -> impl DoubleEndedIterator<Item = Inst> + '_ {
        Order {
            front: self.first_inst(block),
            back: self.last_inst(block),
            links: |i| (self.prev_inst(i), self.next_inst(i)),
        }
    }
    pub fn contains_block(&self, block: Block) -> bool {
        self.blocks[block].inserted
    }
    pub fn first_inst(&self, block: Block) -> Option<Inst> {
        self.blocks[block].first.expand()
    }
    pub fn last_inst(&self, block: Block) -> Option<Inst> {
        self.blocks[block].last.expand()
    }
    pub fn next_inst(&self, inst: Inst) -> Option<Inst> {
        self.insts[inst].next.expand()
    }
    pub fn prev_inst(&self, inst: Inst) -> Option<Inst> {
        self.insts[inst].prev.expand()
    }
    pub fn inst_block(&self, inst: Inst) -> Option<Block> {
        self.insts[inst].block.expand()
    }
    pub fn append_block(&mut self, block: Block) {
        assert!(!self.contains_block(block), "block already in layout");
        self.blocks[block].prev = self.last;
        if let Some(last) = self.last.expand() {
            self.blocks[last].next = block.into();
        } else {
            self.first = block.into();
        }
        self.last = block.into();
        self.blocks[block].inserted = true;
        self.len += 1;
    }
    pub fn remove_block(&mut self, block: Block) {
        assert!(self.contains_block(block), "unknown block");
        assert!(self.first_inst(block).is_none(), "block is not empty");
        let prev = self.blocks[block].prev;
        let next = self.blocks[block].next;
        if let Some(prev) = prev.expand() {
            self.blocks[prev].next = next;
        } else {
            self.first = next;
        }
        if let Some(next) = next.expand() {
            self.blocks[next].prev = prev;
        } else {
            self.last = prev;
        }
        self.blocks[block] = BlockNode::default();
        self.len -= 1;
    }
    pub fn move_block_before(&mut self, block: Block, before: Block) {
        assert!(self.contains_block(block) && self.contains_block(before));
        if block == before {
            return;
        }
        let prev = self.blocks[block].prev;
        let next = self.blocks[block].next;
        if let Some(prev) = prev.expand() {
            self.blocks[prev].next = next;
        } else {
            self.first = next;
        }
        if let Some(next) = next.expand() {
            self.blocks[next].prev = prev;
        } else {
            self.last = prev;
        }
        let prev = self.blocks[before].prev;
        self.blocks[block].prev = prev;
        self.blocks[block].next = before.into();
        self.blocks[before].prev = block.into();
        if let Some(prev) = prev.expand() {
            self.blocks[prev].next = block.into();
        } else {
            self.first = block.into();
        }
    }
    fn link_inst(&mut self, block: Block, inst: Inst, prev: Option<Inst>, next: Option<Inst>) {
        assert!(self.contains_block(block), "unknown block");
        assert!(
            self.inst_block(inst).is_none(),
            "instruction already in layout"
        );
        self.stamp = self.stamp.checked_add(1).expect("layout version overflow");
        self.insts[inst] = InstNode {
            block: block.into(),
            prev: prev.into(),
            next: next.into(),
            stamp: self.stamp,
        };
        if let Some(prev) = prev {
            self.insts[prev].next = inst.into();
        } else {
            self.blocks[block].first = inst.into();
        }
        if let Some(next) = next {
            self.insts[next].prev = inst.into();
        } else {
            self.blocks[block].last = inst.into();
        }
    }
    pub fn append_inst(&mut self, block: Block, inst: Inst) {
        self.link_inst(block, inst, self.last_inst(block), None);
    }
    pub fn insert_after(&mut self, after: Inst, inst: Inst) {
        let block = self.inst_block(after).expect("anchor not in layout");
        self.link_inst(block, inst, Some(after), self.next_inst(after));
    }
    pub fn insert_before(&mut self, before: Inst, inst: Inst) {
        let block = self.inst_block(before).expect("anchor not in layout");
        self.link_inst(block, inst, self.prev_inst(before), Some(before));
    }
    /// Unlink placement only; definitions and uses remain in the DFG.
    pub fn detach_inst(&mut self, inst: Inst) {
        let Some(block) = self.inst_block(inst) else {
            return;
        };
        let prev = self.insts[inst].prev;
        let next = self.insts[inst].next;
        if let Some(prev) = prev.expand() {
            self.insts[prev].next = next;
        } else {
            self.blocks[block].first = next;
        }
        if let Some(next) = next.expand() {
            self.insts[next].prev = prev;
        } else {
            self.blocks[block].last = prev;
        }
        self.insts[inst] = InstNode::default();
    }
    pub fn next_block(&self, block: Block) -> Option<Block> {
        self.blocks[block].next.expand()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Rank {
    stamp: u64,
    label: u64,
}

/// Lazy, incremental order queries over one layout at a time.
///
/// Layout links are authoritative. Cached labels remain valid across deletions
/// and insertions elsewhere: neither changes the order of surviving placements.
/// A placement stamp invalidates only a moved/reinserted instruction. Queries
/// number an unranked run between cached neighbors, using gaps for future edits.
/// If the gap is exhausted, only that block is renumbered. A cache hit is O(1);
/// repair costs O(run length), or O(block length) when renumbering is necessary.
/// No worst-case constant-time update guarantee is made for adversarial edits.
///
/// The identity token survives moves of the layout, but not cloning/replacement.
/// Switching layouts clears the cache; no pointer to the layout itself is kept.
#[derive(Debug)]
pub struct InstOrder<Inst: EntityRef> {
    identity: Option<Arc<()>>,
    ranks: SecondaryMap<Inst, Rank>,
}

impl<Inst: EntityRef> Default for InstOrder<Inst> {
    fn default() -> Self {
        Self {
            identity: None,
            ranks: SecondaryMap::new(),
        }
    }
}

impl<Inst: EntityRef + ReservedValue> InstOrder<Inst> {
    /// Strict order within a block. Cross-block and detached comparisons are
    /// programming errors; use CFG dominance for cross-block availability.
    pub fn comes_before<Block: EntityRef + ReservedValue>(
        &mut self,
        layout: &EntityLayout<Block, Inst>,
        a: Inst,
        b: Inst,
    ) -> bool {
        let block = layout.inst_block(a).expect("instruction is not placed");
        assert!(
            Some(block) == layout.inst_block(b),
            "order comparison requires the same block"
        );
        if self
            .identity
            .as_ref()
            .is_none_or(|id| !Arc::ptr_eq(id, &layout.identity))
        {
            self.ranks.clear();
            self.identity = Some(layout.identity.clone());
        }
        if a == b {
            return false;
        }
        self.ensure(layout, a);
        self.ensure(layout, b);
        self.ranks[a].label < self.ranks[b].label
    }

    fn valid<Block: EntityRef + ReservedValue>(
        &self,
        layout: &EntityLayout<Block, Inst>,
        inst: Inst,
    ) -> bool {
        let stamp = layout.insts[inst].stamp;
        stamp != 0 && self.ranks[inst].stamp == stamp
    }

    fn ensure<Block: EntityRef + ReservedValue>(
        &mut self,
        layout: &EntityLayout<Block, Inst>,
        inst: Inst,
    ) {
        if self.valid(layout, inst) {
            return;
        }
        let mut first = inst;
        let mut last = inst;
        let mut count = 1u64;
        while let Some(prev) = layout.prev_inst(first) {
            if self.valid(layout, prev) {
                break;
            }
            first = prev;
            count += 1;
        }
        while let Some(next) = layout.next_inst(last) {
            if self.valid(layout, next) {
                break;
            }
            last = next;
            count += 1;
        }
        let low = layout.prev_inst(first).map_or(0, |i| self.ranks[i].label);
        let high = layout
            .next_inst(last)
            .map_or(u64::MAX, |i| self.ranks[i].label);
        let step = (high - low) / (count + 1);
        if step == 0 {
            let block = layout.inst_block(inst).expect("placed instruction");
            let count = layout.block_insts(block).count() as u64;
            let step = u64::MAX / (count + 1);
            for (index, i) in layout.block_insts(block).enumerate() {
                self.ranks[i] = Rank {
                    stamp: layout.insts[i].stamp,
                    label: step * (index as u64 + 1),
                };
            }
            return;
        }
        let mut cursor = first;
        let mut label = low;
        loop {
            label += step;
            self.ranks[cursor] = Rank {
                stamp: layout.insts[cursor].stamp,
                label,
            };
            if cursor == last {
                break;
            }
            cursor = layout.next_inst(cursor).expect("contiguous unranked run");
        }
    }
}

struct Order<T, F> {
    front: Option<T>,
    back: Option<T>,
    links: F,
}
impl<T: Copy + Eq, F: Fn(T) -> (Option<T>, Option<T>)> Iterator for Order<T, F> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        let item = self.front?;
        if self.back == Some(item) {
            self.front = None;
            self.back = None;
        } else {
            self.front = (self.links)(item).1;
        }
        Some(item)
    }
}
impl<T: Copy + Eq, F: Fn(T) -> (Option<T>, Option<T>)> DoubleEndedIterator for Order<T, F> {
    fn next_back(&mut self) -> Option<T> {
        let item = self.back?;
        if self.front == Some(item) {
            self.front = None;
            self.back = None;
        } else {
            self.back = (self.links)(item).0;
        }
        Some(item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    struct Block(u32);
    cranelift_entity::entity_impl!(Block, "block");
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    struct Inst(u32);
    cranelift_entity::entity_impl!(Inst, "inst");

    fn check(layout: &EntityLayout<Block, Inst>, order: &mut InstOrder<Inst>) {
        for block in layout.block_order() {
            let insts: Vec<_> = layout.block_insts(block).collect();
            for (i, &a) in insts.iter().enumerate() {
                for (j, &b) in insts.iter().enumerate() {
                    assert_eq!(order.comes_before(layout, a, b), i < j);
                }
            }
        }
    }

    #[test]
    fn cached_order_tracks_edits_and_layout_identity() {
        let mut layout = EntityLayout::new();
        for b in 0..2 {
            layout.append_block(Block(b));
        }
        for i in 0..16 {
            layout.append_inst(Block(i % 2), Inst(i));
        }
        let mut order = InstOrder::default();
        let mut other_cache = InstOrder::default();
        check(&layout, &mut order);
        check(&layout, &mut other_cache);

        // Repeated movement, including cross-block moves and reused IDs.
        for n in 0..160 {
            let inst = Inst(n % 16);
            let anchor = Inst((n * 7 + 3) % 16);
            if inst == anchor {
                continue;
            }
            layout.detach_inst(inst);
            if n % 2 == 0 {
                layout.insert_before(anchor, inst);
            } else {
                layout.insert_after(anchor, inst);
            }
            check(&layout, &mut order);
            // Caches need not observe each intermediate edit.
            if n % 11 == 0 {
                check(&layout, &mut other_cache);
            }
        }

        // Concentrated insertion exhausts label gaps and exercises renumbering.
        for n in 16..160 {
            layout.insert_before(Inst(0), Inst(n));
            assert!(order.comes_before(&layout, Inst(n), Inst(0)));
            if n % 17 == 0 {
                check(&layout, &mut order);
            }
        }
        check(&layout, &mut order);
        check(&layout, &mut other_cache);

        // Divergent clones can have identical placement counters and IDs.
        let mut clone = layout.clone();
        layout.detach_inst(Inst(1));
        layout.insert_before(Inst(0), Inst(1));
        clone.detach_inst(Inst(1));
        clone.insert_after(Inst(0), Inst(1));
        for current in [&layout, &clone, &layout] {
            check(current, &mut order);
        }

        let moved = layout;
        check(&moved, &mut order);
        let mut fresh = EntityLayout::new();
        fresh.append_block(Block(0));
        fresh.append_inst(Block(0), Inst(1));
        fresh.append_inst(Block(0), Inst(0));
        check(&fresh, &mut order);

        // Removing and reusing a block ID must not revive old instruction ranks.
        fresh.remove_insts(&[Inst(0), Inst(1)]);
        fresh.remove_block(Block(0));
        fresh.append_block(Block(0));
        fresh.append_inst(Block(0), Inst(0));
        fresh.append_inst(Block(0), Inst(1));
        check(&fresh, &mut order);
    }
}
