//! Structural editing. Type contracts and dominance remain explicit validation.
use super::FuncBody;
use crate::{Block, Inst, InstWriter, SuccessorData, Type, Value};
use alloc::vec::Vec;
use core::borrow::Borrow;
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

/// Insertion cursor for one block.
///
/// The cursor owns the structural editor for the function body, so generated
/// instruction constructors cannot accidentally bypass CFG/use-def updates.
/// SSA construction state remains on `SsaBuilder`; this type only exposes the
/// instruction-building surface at the current insertion point.
pub struct InstCursor<'ctx, 'body> {
    pub(crate) editor: FuncEditor<'body>,
    pub(crate) decls: &'ctx crate::FuncDecls,
    pub(crate) signatures: &'ctx veloc_types::Signatures,
    pub(crate) block: Block,
    // Stable anchor preserves emission order when inserting at block start.
    before: Option<Inst>,
}

impl<'a> FuncEditor<'a> {
    pub(super) fn new(body: &'a mut FuncBody) -> Self {
        Self { body }
    }

    pub fn body(&self) -> &FuncBody {
        self.body
    }

    /// Intern a position-independent literal, reusing its canonical Value.
    pub fn constant(&mut self, value: crate::Constant) -> Value {
        self.body.dfg.constant(value)
    }

    pub fn at_end<'ctx>(
        self,
        block: Block,
        decls: &'ctx crate::FuncDecls,
        signatures: &'ctx veloc_types::Signatures,
    ) -> InstCursor<'ctx, 'a> {
        assert!(self.body.layout.contains_block(block), "block not placed");
        InstCursor {
            editor: self,
            decls,
            signatures,
            block,
            before: None,
        }
    }

    /// Insert before the original first instruction, preserving emission order.
    pub fn at_start<'ctx>(
        self,
        block: Block,
        decls: &'ctx crate::FuncDecls,
        signatures: &'ctx veloc_types::Signatures,
    ) -> InstCursor<'ctx, 'a> {
        assert!(self.body.layout.contains_block(block), "block not placed");
        let before = self.body.layout.first_inst(block);
        InstCursor {
            editor: self,
            decls,
            signatures,
            block,
            before,
        }
    }

    pub fn create_block(&mut self) -> Block {
        self.body.dfg.create_block()
    }

    pub fn append_block(&mut self, block: Block) {
        assert!(self.body.dfg.blocks.get(block).is_some(), "unknown block");
        self.body.layout.append_block(block);
        self.body.cfg.add_block(block);
    }

    pub fn set_value_name(&mut self, value: Value, name: &str) {
        self.body.dfg.set_value_name(value, name);
    }

    /// Remove unreachable code as a closed set, including cyclic definitions.
    pub fn remove_unreachable(&mut self) -> bool {
        let mut seen = hashbrown::HashSet::new();
        let mut pending = alloc::vec![self.body.entry_block];
        while let Some(block) = pending.pop() {
            if seen.insert(block) {
                pending.extend_from_slice(self.body.cfg.succs(block));
            }
        }
        let dead: Vec<_> = self
            .body
            .layout
            .block_order()
            .filter(|b| !seen.contains(b))
            .collect();
        if dead.is_empty() {
            return false;
        }
        let insts: Vec<_> = dead
            .iter()
            .flat_map(|&b| self.body.layout.block_insts(b))
            .collect();
        self.erase_insts(&insts);
        for block in dead {
            self.body.layout.remove_block(block);
        }
        self.rebuild_cfg();
        true
    }

    /// Join a jump and its sole-predecessor successor, substituting parameters.
    pub fn merge_successor(&mut self, block: Block) -> bool {
        let Some(last) = self.body.layout.last_inst(block) else {
            return false;
        };
        let crate::InstView::Jump { dest } = self.body.dfg.inst(last) else {
            return false;
        };
        let next = dest.block;
        if next == block || next == self.body.entry_block || self.body.cfg.preds(next) != [block] {
            return false;
        }
        let args = dest.args.to_vec();
        let params = self.body.dfg.block_params(next).to_vec();
        for (param, arg) in params.into_iter().zip(args) {
            self.replace_all_uses(param, arg);
        }
        // Rebuild the CFG once both layout edits are complete.
        self.body.dfg.remove_insts([last]);
        self.body.layout.detach_inst(last);
        while let Some(inst) = self.body.layout.first_inst(next) {
            self.body.layout.detach_inst(inst);
            self.body.layout.append_inst(block, inst);
        }
        self.body.layout.remove_block(next);
        self.rebuild_cfg();
        true
    }

    fn rebuild_cfg(&mut self) {
        self.body.cfg = super::ControlFlowGraph::new(self.body.layout.block_order());
        let blocks: Vec<_> = self.body.layout.block_order().collect();
        for block in blocks {
            self.sync_edges(block);
        }
    }

    /// Construction primitive: incoming arguments may be filled later.
    pub fn append_block_param(&mut self, block: Block, ty: Type) -> Value {
        self.body.dfg.append_block_param(block, ty)
    }

    /// Append a block parameter and supply its argument on every incoming edge.
    /// The callback is called separately for repeated edges from one predecessor.
    pub fn add_param(
        &mut self,
        block: Block,
        ty: Type,
        mut incoming: impl FnMut(EdgeRef) -> Value,
    ) -> Value {
        let arity = self.body.dfg.block_params(block).len();
        let mut edits = Vec::new();
        for &pred in self.body.cfg.preds(block) {
            let inst = self
                .body
                .layout
                .last_inst(pred)
                .expect("terminated predecessor");
            let mut index = 0;
            self.body.dfg.inst(inst).visit_successors(|dest| {
                let edge = EdgeRef { inst, index };
                index += 1;
                if dest.block == block {
                    assert_eq!(dest.args.len(), arity, "incoming argument count mismatch");
                    let value = incoming(edge);
                    assert_eq!(
                        self.body.dfg.value_type(value),
                        ty,
                        "argument type mismatch"
                    );
                    let mut args = dest.args.to_vec();
                    args.push(value);
                    edits.push((edge, args));
                }
            });
        }
        let param = self.body.dfg.append_block_param(block, ty);
        for (edge, args) in edits {
            self.body.dfg.redirect_edge(edge, block, &args);
        }
        param
    }

    /// Declare a fresh fixed stack object and return its function-wide pointer.
    pub fn stack_slot(&mut self, size: u32, align: u32) -> Value {
        assert!(
            size > 0 && align.is_power_of_two(),
            "invalid stack slot layout"
        );
        let value = self.body.dfg.values.push(crate::types::ValueData {
            ty: Type::PTR,
            def: crate::ValueDef::StackSlot { size, align },
        });
        self.body.stack_slots.push(value);
        value
    }

    pub(crate) fn bind_stack_slot(&mut self, value: Value, size: u32, align: u32) {
        self.body.dfg.values[value] = crate::types::ValueData {
            ty: Type::PTR,
            def: crate::ValueDef::StackSlot { size, align },
        };
        self.body.stack_slots.push(value);
    }

    pub fn remove_unused_stack_slots(&mut self) -> usize {
        let old_len = self.body.stack_slots.len();
        let dfg = &self.body.dfg;
        self.body.stack_slots.retain(|&v| !dfg.use_empty(v));
        old_len - self.body.stack_slots.len()
    }

    /// Extend the function inputs when constructing a new definition.
    /// The containing module must give this body a matching signature.
    pub fn append_function_param(&mut self, ty: Type) -> Value {
        let index = crate::ParamIndex(
            self.body
                .params
                .len()
                .try_into()
                .expect("too many parameters"),
        );
        let value = self.body.dfg.values.push(crate::types::ValueData {
            ty,
            def: crate::ValueDef::FunctionParam(index),
        });
        self.body.params.push(value);
        value
    }

    pub(crate) fn create_inst(&mut self, build: impl FnOnce(InstWriter<'_>) -> Inst) -> Inst {
        self.body.dfg.create_inst(build)
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
                def: crate::ValueDef::BlockParam(self.body.entry_block),
            });
        }
    }

    pub(crate) fn bind_param(&mut self, block: Block, value: Value, ty: Type) {
        self.body.dfg.values[value] = crate::types::ValueData {
            ty,
            def: crate::ValueDef::BlockParam(block),
        };
        self.body.dfg.blocks[block].params.push(value);
    }

    pub(crate) fn remap_functions(&mut self, map: &[crate::FuncId]) {
        self.body.dfg.remap_functions(map);
    }

    pub(crate) fn edit_successors(&mut self, inst: Inst, edit: impl FnMut(&mut SuccessorData)) {
        let block = self
            .body
            .layout
            .inst_block(inst)
            .expect("instruction not placed");
        self.body.dfg.edit_successors(inst, edit);
        self.sync_edges(block);
    }

    /// Intern vector bytes without validation; the validator checks their layout.
    pub fn dense_constant(&mut self, ty: crate::VectorType, bytes: Vec<u8>) -> Value {
        self.constant(crate::VectorConst::dense(ty, bytes).into())
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

    /// Remove parameter positions and their incoming arguments together. The
    /// caller must remove all uses of discarded values in the complete edit.
    pub fn retain_block_params(&mut self, block: Block, keep: &[bool]) {
        assert_eq!(keep.len(), self.body.dfg.block_params(block).len());
        for &pred in self.body.cfg.preds(block) {
            let inst = self
                .body
                .layout
                .last_inst(pred)
                .expect("predecessor has no terminator");
            self.body.dfg.retain_edge_args(inst, block, keep);
        }
        let mut index = 0;
        self.body.dfg.blocks[block].params.retain(|_| {
            let retain = keep[index];
            index += 1;
            retain
        });
    }

    /// Edit exactly one outgoing edge, maintaining operands, uses and CFG
    /// adjacency. Type and dominance contracts remain explicit validation.
    pub fn redirect_edge(&mut self, edge: EdgeRef, target: Block, args: &[Value]) {
        assert!(self.body.dfg.blocks.get(target).is_some(), "unknown target");
        let block = self
            .body
            .layout
            .inst_block(edge.inst)
            .expect("edge instruction not placed");
        if self.body.dfg.redirect_edge(edge, target, args) {
            self.sync_edges(block);
        }
    }

    /// Update one use in place. Argument values do not affect CFG adjacency.
    pub fn set_edge_arg(&mut self, edge: EdgeRef, index: usize, value: Value) {
        let (_, args) = self.body.dfg.edge_args(edge);
        assert!(
            index < (args.end - args.start) as usize,
            "successor argument index out of bounds"
        );
        self.body
            .dfg
            .set_operand(edge.inst, args.start + index as u32, value);
    }

    /// Rebind a conditional branch while preserving or swapping its original edges.
    pub fn rewrite_branch(&mut self, inst: Inst, condition: Value, swap_edges: bool) -> bool {
        debug_assert_eq!(self.body.dfg.value_type(condition), crate::Type::BOOL);
        let crate::InstView::Br {
            condition: old,
            then_dest,
            else_dest,
        } = self.body.dfg.inst(inst)
        else {
            panic!("conditional branch required");
        };
        if !swap_edges || then_dest == else_dest {
            if old == condition {
                return false;
            }
            self.set_operand(inst, 0, condition);
            return true;
        }
        let yes = SuccessorData::new(else_dest.block, else_dest.args);
        let no = SuccessorData::new(then_dest.block, then_dest.args);
        self.replace_inst(inst, |w| w.br(condition, yes.as_view(), no.as_view()));
        true
    }

    /// Replace a branch with a jump along the selected edge, retaining its args.
    pub fn fold_to_edge(&mut self, edge: EdgeRef) {
        let block = self
            .body
            .layout
            .inst_block(edge.inst)
            .expect("edge instruction not placed");
        self.body.dfg.fold_to_edge(edge);
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
        self.insert_at(block, None, data, types)
    }

    pub fn insert_after(
        &mut self,
        after: Inst,
        data: impl FnOnce(InstWriter<'_>) -> Inst,
        types: &[Type],
    ) -> Inst {
        let block = self
            .body
            .layout
            .inst_block(after)
            .expect("anchor not in layout");
        let before = self.body.layout.next_inst(after);
        self.insert_at(block, before, data, types)
    }

    /// Insert before a stable anchor without renumbering instructions.
    pub fn insert_before(
        &mut self,
        before: Inst,
        data: impl FnOnce(InstWriter<'_>) -> Inst,
        types: &[Type],
    ) -> Inst {
        let block = self
            .body
            .layout
            .inst_block(before)
            .expect("anchor not in layout");
        self.insert_at(block, Some(before), data, types)
    }

    fn insert_at(
        &mut self,
        block: Block,
        before: Option<Inst>,
        data: impl FnOnce(InstWriter<'_>) -> Inst,
        types: &[Type],
    ) -> Inst {
        let inst = self.create_inst(data);
        self.finish_inst(block, before, inst, types);
        inst
    }

    /// Shared by explicit insertion and constructors that infer result types.
    fn finish_inst(&mut self, block: Block, before: Option<Inst>, inst: Inst, types: &[Type]) {
        assert!(self.body.layout.contains_block(block), "block not placed");
        if let Some(anchor) = before {
            assert_eq!(
                self.body.layout.inst_block(anchor),
                Some(block),
                "anchor in another block"
            );
            assert!(
                !self.body.dfg.opcode(inst).spec().is_terminator(),
                "cannot insert a terminator before another instruction"
            );
        } else {
            assert!(
                self.body.layout.last_inst(block).is_none_or(|last| !self
                    .body
                    .dfg
                    .opcode(last)
                    .spec()
                    .is_terminator()),
                "cannot append after a terminator"
            );
        }
        self.body.dfg.append_results(inst, types);
        if let Some(anchor) = before {
            self.body.layout.insert_before(anchor, inst);
        } else {
            self.append_existing(block, inst);
        }
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
        self.erase_insts([inst]);
    }

    /// Erase a closed batch in any order. Each traversal must yield the same IDs.
    /// Borrow the collection so cloning its iterator does not copy the elements.
    pub fn erase_insts(&mut self, insts: impl IntoIterator<Item: Borrow<Inst>, IntoIter: Clone>) {
        let insts = insts.into_iter().map(|inst| *inst.borrow());
        let mut blocks = Vec::new();
        for inst in insts.clone() {
            let block = self.body.layout.inst_block(inst).expect("instruction not placed");
            if self.body.layout.last_inst(block) == Some(inst) {
                blocks.push(block);
            }
        }
        blocks.sort_unstable();
        blocks.dedup();
        self.body.dfg.remove_insts(insts.clone());
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
        self.body.cfg.set_successors(block, &successors);
    }
}

impl<'ctx, 'body> InstCursor<'ctx, 'body> {
    pub fn block(&self) -> Block {
        self.block
    }

    pub fn dfg(&self) -> &crate::dfg::DataFlowGraph {
        self.editor.body().dfg()
    }

    pub fn value_type(&self, value: Value) -> Type {
        self.dfg().value_type(value)
    }

    /// Name a value without changing SSA definitions or control flow.
    pub fn set_value_name(&mut self, value: Value, name: &str) {
        self.editor.set_value_name(value, name);
    }

    /// Insert an instruction with caller-supplied result types.
    pub fn insert(&mut self, data: impl FnOnce(InstWriter<'_>) -> Inst, types: &[Type]) -> Inst {
        self.editor.insert_at(self.block, self.before, data, types)
    }

    pub(crate) fn emit<const N: usize>(
        &mut self,
        data: impl FnOnce(InstWriter<'_>) -> Inst,
        types: [Type; N],
    ) -> [Value; N] {
        let inst = self.insert(data, &types);
        self.dfg()
            .inst_results(inst)
            .try_into()
            .expect("insert must create one result per supplied type")
    }

    pub fn dense_const(&mut self, bytes: Vec<u8>, ty: Type) -> Value {
        self.editor
            .dense_constant(ty.as_vector().expect("vector constant type"), bytes)
    }
}
