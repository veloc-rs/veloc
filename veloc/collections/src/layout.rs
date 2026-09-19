//! Stable IDs with independently editable block and instruction order.
//!
//! Links live in indexed storage, not separately allocated nodes. Insertion and
//! removal update only neighboring links; instruction ownership stays directly
//! queryable. Iteration borrows the layout, so mutating passes use stable anchors
//! or explicitly collect a snapshot when they need one.
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
}

#[derive(Debug, Clone)]
pub struct EntityLayout<Block: EntityRef + ReservedValue, Inst: EntityRef + ReservedValue> {
    blocks: SecondaryMap<Block, BlockNode<Block, Inst>>,
    first: PackedOption<Block>,
    last: PackedOption<Block>,
    insts: SecondaryMap<Inst, InstNode<Block, Inst>>,
    len: usize,
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
        }
    }
    pub fn prepend_inst(&mut self, block: Block, inst: Inst) {
        self.link_inst(block, inst, None, self.first_inst(block));
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
        self.insts[inst] = InstNode {
            block: block.into(),
            prev: prev.into(),
            next: next.into(),
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
