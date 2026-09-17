//! Structural editing. Type contracts and dominance remain explicit validation.
use super::FuncBody;
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

pub struct FuncEditor<'a> {
    body: &'a mut FuncBody,
}

impl<'a> FuncEditor<'a> {
    pub(super) fn new(body: &'a mut FuncBody) -> Self {
        Self { body }
    }

    pub fn body(&self) -> &FuncBody {
        self.body
    }

    pub fn create_block(&mut self) -> Block {
        self.body.dfg.create_block()
    }

    pub fn append_block(&mut self, block: Block) {
        assert!(self.body.dfg.blocks.get(block).is_some(), "unknown block");
        let first = self.body.layout.block_order().next().is_none();
        self.body.layout.append_block(block);
        if first {
            self.body.entry_block = block;
        }
    }

    pub fn set_value_name(&mut self, value: Value, name: &str) {
        self.body.dfg.set_value_name(value, name);
    }

    /// Construction primitive: incoming arguments may be filled later.
    pub(crate) fn append_block_param(&mut self, block: Block, ty: Type) -> Value {
        self.body.dfg.append_block_param(block, ty)
    }

    pub(crate) fn create_inst(&mut self, build: impl FnOnce(InstWriter<'_>) -> Inst) -> Inst {
        self.body.dfg.create_inst(build)
    }

    pub(crate) fn finish_inst(&mut self, block: Block, inst: Inst, types: &[Type]) {
        self.body.dfg.append_results(inst, types);
        self.append_existing(block, inst);
    }

    pub(crate) fn finish_parsed_inst(
        &mut self,
        block: Block,
        inst: Inst,
        results: &[(Value, Type)],
    ) {
        self.body.dfg.bind_results(inst, results);
        self.append_existing(block, inst);
    }

    pub(crate) fn reserve_value(&mut self, value: Value) {
        while self.body.dfg.values.len() <= value.0 as usize {
            self.body.dfg.values.push(crate::types::ValueData {
                ty: Type::INVALID,
                def: crate::ValueDef::Param(self.body.entry_block),
            });
        }
    }

    pub(crate) fn bind_param(&mut self, block: Block, value: Value, ty: Type) {
        self.body.dfg.values[value] = crate::types::ValueData {
            ty,
            def: crate::ValueDef::Param(block),
        };
        self.body.dfg.blocks[block].params.push(value);
    }

    pub(crate) fn remap_functions(&mut self, map: &[crate::FuncId]) {
        self.body.dfg.remap_functions(map);
    }

    pub(crate) fn edit_successors(
        &mut self,
        inst: Inst,
        mut edit: impl FnMut(&mut SuccessorMut<'_>),
    ) {
        let block = self
            .body
            .layout
            .inst_block(inst)
            .expect("instruction not placed");
        self.body.dfg.edit_successors(inst, |edge| edit(edge));
        self.sync_edges(block);
    }

    /// Intern vector bytes without validation; the validator checks their layout.
    pub fn dense_constant(&mut self, ty: crate::VectorType, bytes: Vec<u8>) -> crate::VectorConst {
        let id = crate::inst::ConstantPoolId::insert(&mut self.body.dfg, bytes);
        crate::VectorConst::dense(ty, id)
    }

    pub fn set_value_type(&mut self, value: Value, ty: Type) {
        self.body.dfg.set_value_type(value, ty);
    }

    pub fn set_operand(&mut self, inst: Inst, index: u32, value: Value) {
        self.body.dfg.set_operand(inst, index, value);
    }

    pub fn replace_all_uses(&mut self, old: Value, new: Value) {
        self.body.dfg.replace_all_uses(old, new);
    }

    /// Edit exactly one outgoing edge, maintaining operands, uses and CFG
    /// adjacency. Type and dominance contracts remain explicit validation.
    pub fn redirect_edge(&mut self, edge: EdgeRef, target: Block, args: &[Value]) {
        assert!(self.body.dfg.blocks.get(target).is_some(), "unknown target");
        self.edit_edge(edge, |successor| {
            successor.set_block(target);
            successor.set_args(args);
        });
    }

    pub fn set_edge_arg(&mut self, edge: EdgeRef, index: usize, value: Value) {
        self.edit_edge(edge, |successor| {
            assert!(
                index < successor.args().len(),
                "edge argument out of bounds"
            );
            successor.set_arg(index, value);
        });
    }

    fn edit_edge(&mut self, edge: EdgeRef, edit: impl FnOnce(&mut SuccessorMut<'_>)) {
        let block = self
            .body
            .layout
            .inst_block(edge.inst)
            .expect("edge instruction not placed");
        let mut count = 0;
        self.body
            .dfg
            .inst(edge.inst)
            .visit_successors(|_| count += 1);
        assert!(
            (edge.index as usize) < count,
            "successor position out of bounds"
        );
        let mut edit = Some(edit);
        let mut index = 0;
        self.body.dfg.edit_successors(edge.inst, |successor| {
            if index == edge.index {
                edit.take().expect("unique successor position")(successor);
            }
            index += 1;
        });
        assert!(edit.is_none(), "successor position out of bounds");
        self.sync_edges(block);
    }

    pub(crate) fn append_existing(&mut self, block: Block, inst: Inst) {
        self.body.layout.append_inst(block, inst);
        self.sync_edges(block);
    }

    pub fn append_inst(
        &mut self,
        block: Block,
        data: impl FnOnce(InstWriter<'_>) -> Inst,
        types: &[Type],
    ) -> Inst {
        assert!(self.body.layout.contains_block(block), "block not placed");
        let inst = self.body.dfg.create_inst(data);
        self.body.dfg.append_results(inst, types);
        self.body.layout.append_inst(block, inst);
        self.sync_edges(block);
        inst
    }

    /// Insert a non-terminator at block entry.
    pub fn prepend_inst(
        &mut self,
        block: Block,
        data: impl FnOnce(InstWriter<'_>) -> Inst,
        types: &[Type],
    ) -> Inst {
        let inst = self.body.dfg.create_inst(data);
        assert!(
            !self.body.dfg().opcode(inst).spec().is_terminator(),
            "cannot prepend a terminator"
        );
        self.body.dfg.append_results(inst, types);
        self.body.layout.prepend_inst(block, inst);
        inst
    }

    pub fn insert_after(
        &mut self,
        after: Inst,
        data: impl FnOnce(InstWriter<'_>) -> Inst,
        types: &[Type],
    ) -> Inst {
        let block = self
            .body
            .layout()
            .inst_block(after)
            .expect("anchor not in layout");
        let inst = self.body.dfg.create_inst(data);
        self.body.dfg.append_results(inst, types);
        self.body.layout.insert_after(after, inst);
        if self.body.layout().last_inst(block) == Some(inst) {
            self.sync_edges(block);
        }
        inst
    }

    /// Insert before a stable anchor without renumbering instructions.
    pub fn insert_before(
        &mut self,
        before: Inst,
        data: impl FnOnce(InstWriter<'_>) -> Inst,
        types: &[Type],
    ) -> Inst {
        self.body
            .layout()
            .inst_block(before)
            .expect("anchor not in layout");
        let inst = self.body.dfg.create_inst(data);
        self.body.dfg.append_results(inst, types);
        self.body.layout.insert_before(before, inst);
        inst
    }

    /// Move placement only. Callers must preserve dominance and terminator rules.
    pub fn move_before(&mut self, inst: Inst, before: Inst) {
        let old = self
            .body
            .layout()
            .inst_block(inst)
            .expect("instruction not in layout");
        let new = self
            .body
            .layout()
            .inst_block(before)
            .expect("anchor not in layout");
        if inst == before {
            return;
        }
        self.body.layout.detach_inst(inst);
        self.body.layout.insert_before(before, inst);
        self.sync_edges(old);
        if new != old {
            self.sync_edges(new);
        }
    }

    pub fn move_to_end(&mut self, inst: Inst, block: Block) {
        assert!(
            self.body.layout().contains_block(block),
            "destination not in layout"
        );
        let old = self
            .body
            .layout()
            .inst_block(inst)
            .expect("instruction not in layout");
        self.body.layout.detach_inst(inst);
        self.body.layout.append_inst(block, inst);
        self.sync_edges(old);
        if block != old {
            self.sync_edges(block);
        }
    }

    /// Change physical block order; entry identity and CFG are unchanged.
    pub fn move_block_before(&mut self, block: Block, before: Block) {
        self.body.layout.move_block_before(block, before);
    }

    /// Rewrite the instruction while retaining its existing result identities.
    /// The caller must preserve the result contract, checked by validation.
    pub fn replace_inst(&mut self, inst: Inst, data: impl FnOnce(InstWriter<'_>) -> Inst) {
        let block = self
            .body
            .layout()
            .inst_block(inst)
            .expect("instruction not in layout");
        let control = self.body.dfg().opcode(inst).spec().is_terminator();
        self.body.dfg.replace_inst(inst, data);
        if control || self.body.dfg().opcode(inst).spec().is_terminator() {
            self.sync_edges(block);
        }
    }

    pub fn erase_inst(&mut self, inst: Inst) {
        self.erase_insts(&[inst]);
    }

    pub fn erase_insts(&mut self, insts: &[Inst]) {
        for &inst in insts {
            assert!(
                self.body.layout.inst_block(inst).is_some(),
                "instruction not placed"
            );
        }
        let mut blocks: Vec<_> = insts
            .iter()
            .filter(|&&inst| {
                self.body
                    .layout()
                    .inst_block(inst)
                    .is_some_and(|b| self.body.layout().last_inst(b) == Some(inst))
            })
            .map(|&inst| {
                self.body
                    .layout()
                    .inst_block(inst)
                    .expect("instruction not in layout")
            })
            .collect();
        blocks.sort_unstable();
        blocks.dedup();
        self.body.dfg.remove_insts(insts);
        self.body.layout.remove_insts(insts);
        for block in blocks {
            self.sync_edges(block);
        }
    }

    /// Replace all results and erase the old computation. Type/dominance
    /// compatibility remains the caller's responsibility until validation.
    pub fn replace_results(&mut self, inst: Inst, replacements: &[Value]) {
        let results = self.body.dfg.inst_results(inst).to_vec();
        assert_eq!(results.len(), replacements.len(), "result count mismatch");
        assert!(
            self.body.layout.inst_block(inst).is_some(),
            "instruction not placed"
        );
        for &value in replacements {
            assert!(
                self.body.dfg.values.get(value).is_some(),
                "unknown replacement"
            );
            assert!(
                !results.contains(&value),
                "replacement depends on erased result"
            );
        }
        for (old, &new) in results.into_iter().zip(replacements) {
            self.body.dfg.replace_all_uses(old, new);
        }
        self.erase_inst(inst);
    }

    fn sync_edges(&mut self, block: Block) {
        let mut successors = SmallVec::<[Block; 2]>::new();
        if let Some(inst) = self.body.layout().last_inst(block) {
            self.body
                .dfg()
                .inst(inst)
                .visit_successors(|call| successors.push(call.block));
        }
        successors.sort_unstable();
        successors.dedup();
        if self.body.cfg.blocks[block].succs == successors.as_slice() {
            return;
        }
        let old = core::mem::take(&mut self.body.cfg.blocks[block].succs);
        for &succ in &old {
            if !successors.contains(&succ) {
                self.body.cfg.blocks[succ].preds.retain(|&b| b != block);
            }
        }
        for &succ in &successors {
            if !old.contains(&succ) {
                self.body.cfg.blocks[succ].preds.push(block);
            }
        }
        self.body.cfg.blocks[block].succs = successors.into_vec();
    }
}
