//! Structural editing. Type contracts and dominance remain explicit validation.
use super::Function;
use crate::{Block, Inst, InstWriter, SuccessorMut, Type, Value};
use alloc::vec::Vec;
use smallvec::SmallVec;

/// One successor occurrence, not a pair of adjacent blocks. This location is
/// invalidated when the instruction is erased or its successor order changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeRef {
    pub inst: Inst,
    pub index: u32,
}

pub struct FunctionEditor<'a> {
    func: &'a mut Function,
}

impl<'a> FunctionEditor<'a> {
    pub(super) fn new(func: &'a mut Function) -> Self {
        Self { func }
    }

    pub fn function(&self) -> &Function {
        self.func
    }

    /// Intern vector bytes without validation; the validator checks their layout.
    pub fn dense_constant(&mut self, ty: crate::VectorType, bytes: Vec<u8>) -> crate::VectorConst {
        let id = crate::inst::ConstantPoolId::insert(&mut self.func.dfg, bytes);
        crate::VectorConst::dense(ty, id)
    }

    pub fn set_value_type(&mut self, value: Value, ty: Type) {
        self.func.dfg.set_value_type(value, ty);
    }

    pub fn set_operand(&mut self, inst: Inst, index: u32, value: Value) {
        self.func.dfg.set_operand(inst, index, value);
    }

    pub fn replace_all_uses(&mut self, old: Value, new: Value) {
        self.func.dfg.replace_all_uses(old, new);
    }

    /// Edit exactly one outgoing edge, maintaining operands, uses and CFG
    /// adjacency. Type and dominance contracts remain explicit validation.
    pub fn edit_edge(&mut self, edge: EdgeRef, edit: impl FnOnce(&mut SuccessorMut<'_>)) {
        let mut edit = Some(edit);
        let mut index = 0;
        self.func.dfg.edit_successors(edge.inst, |successor| {
            if index == edge.index {
                edit.take().expect("unique successor position")(successor);
            }
            index += 1;
        });
        assert!(edit.is_none(), "successor position out of bounds");
        let block = self
            .func
            .layout
            .inst_block(edge.inst)
            .expect("edge instruction in layout");
        self.sync_edges(block);
    }

    pub(crate) fn append_existing(&mut self, block: Block, inst: Inst) {
        self.func.layout.append_inst(block, inst);
        if self.func.dfg.opcode(inst).spec().is_terminator() {
            self.sync_edges(block);
        }
    }

    pub fn append_inst(
        &mut self,
        block: Block,
        data: impl FnOnce(InstWriter<'_>) -> Inst,
        types: &[Type],
    ) -> Inst {
        let inst = self.func.dfg.create_inst(data);
        let control = self.func.dfg.opcode(inst).spec().is_terminator();
        self.func.dfg.append_results(inst, types);
        self.func.layout.append_inst(block, inst);
        if control {
            self.sync_edges(block);
        }
        inst
    }

    /// Insert a non-terminator at block entry.
    pub fn prepend_inst(
        &mut self,
        block: Block,
        data: impl FnOnce(InstWriter<'_>) -> Inst,
        types: &[Type],
    ) -> Inst {
        let inst = self.func.dfg.create_inst(data);
        assert!(
            !self.func.dfg.opcode(inst).spec().is_terminator(),
            "cannot prepend a terminator"
        );
        self.func.dfg.append_results(inst, types);
        self.func.layout.prepend_inst(block, inst);
        inst
    }

    pub fn insert_after(
        &mut self,
        after: Inst,
        data: impl FnOnce(InstWriter<'_>) -> Inst,
        types: &[Type],
    ) -> Inst {
        let block = self
            .func
            .layout
            .inst_block(after)
            .expect("anchor not in layout");
        let inst = self.func.dfg.create_inst(data);
        let control = self.func.dfg.opcode(inst).spec().is_terminator();
        self.func.dfg.append_results(inst, types);
        self.func.layout.insert_after(after, inst);
        if control {
            self.sync_edges(block);
        }
        inst
    }

    pub fn replace_inst(&mut self, inst: Inst, data: impl FnOnce(InstWriter<'_>) -> Inst) {
        let block = self
            .func
            .layout
            .inst_block(inst)
            .expect("instruction not in layout");
        let control = self.func.dfg.opcode(inst).spec().is_terminator();
        self.func.dfg.replace_inst(inst, data);
        if control || self.func.dfg.opcode(inst).spec().is_terminator() {
            self.sync_edges(block);
        }
    }

    pub fn erase_inst(&mut self, inst: Inst) {
        self.erase_insts(&[inst]);
    }

    pub fn erase_insts(&mut self, insts: &[Inst]) {
        let mut blocks: Vec<_> = insts
            .iter()
            .filter(|&&inst| self.func.dfg.opcode(inst).spec().is_terminator())
            .map(|&inst| {
                self.func
                    .layout
                    .inst_block(inst)
                    .expect("instruction not in layout")
            })
            .collect();
        blocks.sort_unstable();
        blocks.dedup();
        self.func.dfg.remove_insts(insts);
        self.func.layout.remove_insts(insts);
        for block in blocks {
            self.sync_edges(block);
        }
    }

    /// Reattach an existing result, preserving its Value identity and all uses.
    pub fn move_result(&mut self, value: Value, to: Inst) {
        self.func.dfg.move_result(value, to);
    }

    fn sync_edges(&mut self, block: Block) {
        let mut successors = SmallVec::<[Block; 2]>::new();
        if let Some(&inst) = self.func.layout.blocks[block].insts.last() {
            self.func
                .dfg
                .inst(inst)
                .visit_successors(|call| successors.push(call.block));
        }
        successors.sort_unstable();
        successors.dedup();
        if self.func.layout.blocks[block].succs == successors.as_slice() {
            return;
        }
        let old = core::mem::take(&mut self.func.layout.blocks[block].succs);
        for &succ in &old {
            if !successors.contains(&succ) {
                self.func.layout.blocks[succ].preds.retain(|&b| b != block);
            }
        }
        for &succ in &successors {
            if !old.contains(&succ) {
                self.func.layout.blocks[succ].preds.push(block);
            }
        }
        self.func.layout.blocks[block].succs = successors.into_vec();
    }
}
