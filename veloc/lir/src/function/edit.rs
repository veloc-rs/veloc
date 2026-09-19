//! Structural editing and scoped change reporting. Whole-program validation is explicit.
use super::*;

/// Mechanical edit notifications. They do not replace type or SSA validation.
#[derive(Debug, Default)]
pub struct EditChanges {
    pub insts: Vec<InstId>,
    pub blocks: Vec<Block>,
}

/// Exclusive structural editing. Reads are available through Deref, but there
/// is deliberately no DerefMut or escape hatch to mutable function storage.
pub struct FuncEditor<'a> {
    function: &'a mut MachineFunction,
    changes: Option<&'a mut EditChanges>,
}
impl core::ops::Deref for FuncEditor<'_> {
    type Target = MachineFunction;
    fn deref(&self) -> &Self::Target {
        self.function
    }
}
impl MachineFunction {
    pub fn editor(&mut self) -> FuncEditor<'_> {
        FuncEditor {
            function: self,
            changes: None,
        }
    }
}
impl FuncEditor<'_> {
    /// Reborrow the editor, retaining the current session's notifications.
    pub fn editor(&mut self) -> FuncEditor<'_> {
        FuncEditor {
            function: self.function,
            changes: self.changes.as_deref_mut(),
        }
    }
    fn changed_inst(&mut self, inst: InstId) {
        if let Some(changes) = &mut self.changes {
            changes.insts.push(inst);
        }
    }
    pub fn create_block(&mut self) -> Block {
        let last = self.function.body.layout.block_order().next_back();
        if let Some(last) = last {
            self.changed_block(last);
        }
        let block = self.function.body.blocks.push(BlockData::default());
        self.function.body.layout.append_block(block);
        self.changed_block(block);
        block
    }
    pub fn set_entry_block(&mut self, block: Block) {
        assert!(
            self.function.body.layout.contains_block(block),
            "unknown entry block"
        );
        self.changed_block(self.function.body.entry);
        self.function.body.entry = block;
        self.changed_block(block);
    }
    pub fn append_block_param(&mut self, block: Block, param: Reg) {
        assert!(
            self.function.body.layout.contains_block(block),
            "unknown block"
        );
        self.function.body.blocks[block].params.push(param);
        self.changed_block(block);
    }
    pub fn clear_block_params(&mut self) {
        for block in self.blocks().collect::<Vec<_>>() {
            self.function.body.blocks[block].params.clear();
            self.changed_block(block);
        }
    }
    pub fn move_block_before(&mut self, block: Block, before: Block) {
        // Moving a physical block can change implicit fallthrough edges.
        for id in [
            Some(block),
            Some(before),
            self.function.body.layout.prev_block(block),
            self.function.body.layout.prev_block(before),
        ]
        .into_iter()
        .flatten()
        {
            self.changed_block(id);
        }
        self.function.body.layout.move_block_before(block, before);
    }
    /// Commit a permutation of the existing block instructions; no insertion,
    /// removal or cross-block movement is allowed here.
    pub fn reorder_block(&mut self, block: Block, insts: &[InstId]) {
        assert!(
            self.function.body.layout.contains_block(block),
            "unknown block"
        );
        assert_eq!(
            insts.len(),
            self.block_insts(block).count(),
            "reorder must preserve instruction count"
        );
        let mut seen = hashbrown::HashSet::new();
        for &inst in insts {
            let _ = self.inst(inst);
            assert!(seen.insert(inst), "duplicate instruction");
            assert!(
                self.inst_block(inst) == Some(block),
                "instruction belongs to another block"
            );
        }
        while let Some(inst) = self.function.body.layout.first_inst(block) {
            self.function.body.layout.detach_inst(inst);
        }
        for &inst in insts {
            self.function.body.layout.append_inst(block, inst);
            self.changed_inst(inst);
        }
        self.changed_block(block);
    }
    /// Replace one placed instruction with detached instructions in the supplied
    /// order. Including the root keeps its identity; otherwise it is erased.
    pub fn replace_with(&mut self, root: InstId, output: &[InstId]) {
        let block = self.inst_block(root).expect("replacement root is detached");
        if output == [root] {
            return;
        }
        let next = self.layout().next_inst(root);
        let mut seen = hashbrown::HashSet::new();
        for &inst in output {
            let _ = self.inst(inst);
            assert!(seen.insert(inst), "duplicate replacement");
            assert!(
                inst == root || self.inst_block(inst).is_none(),
                "replacement must be detached"
            );
        }
        self.detach_inst(root);
        for &inst in output {
            if let Some(next) = next {
                self.insert_before(next, inst);
            } else {
                self.append_inst(block, inst);
            }
        }
        if !seen.contains(&root) {
            self.invalidate_inst(root);
        }
    }

    pub fn append_inst(&mut self, block: Block, inst: InstId) {
        let _ = self.inst(inst);
        self.function.body.layout.append_inst(block, inst);
        self.changed_inst(inst);
        self.changed_block(block);
    }
    pub fn insert_before(&mut self, anchor: InstId, inst: InstId) {
        let _ = self.inst(inst);
        self.function.body.layout.insert_before(anchor, inst);
        self.changed_inst(inst);
        self.changed_block(self.inst_block(anchor).unwrap());
    }
    pub fn insert_after(&mut self, anchor: InstId, inst: InstId) {
        let _ = self.inst(inst);
        self.function.body.layout.insert_after(anchor, inst);
        self.changed_inst(inst);
        self.changed_block(self.inst_block(anchor).unwrap());
    }
    pub fn detach_inst(&mut self, inst: InstId) {
        if let Some(block) = self.inst_block(inst) {
            self.changed_block(block);
        }
        self.function.body.layout.detach_inst(inst);
        self.changed_inst(inst);
    }
    pub fn move_before(&mut self, inst: InstId, anchor: InstId) {
        assert!(self.inst_block(anchor).is_some(), "detached anchor");
        if inst != anchor {
            self.detach_inst(inst);
            self.insert_before(anchor, inst);
        }
    }
    /// Split placement at an instruction. The caller supplies any required
    /// explicit branch and block arguments; no target opcode is guessed here.
    pub fn split_block(&mut self, at: InstId) -> Block {
        let source = self.inst_block(at).expect("detached split anchor");
        let next = self.function.body.layout.next_block(source);
        let block = self.create_block();
        if let Some(next) = next {
            self.move_block_before(block, next);
        }
        let mut inst = Some(at);
        while let Some(id) = inst {
            inst = self.function.body.layout.next_inst(id);
            self.detach_inst(id);
            self.append_inst(block, id);
        }
        block
    }
    /// Erase a non-entry block and its instructions. Incoming edges must be
    /// redirected by the caller; full CFG/SSA validation remains explicit.
    pub fn erase_block(&mut self, block: Block) {
        assert_ne!(self.entry_block(), block, "cannot erase entry block");
        assert!(
            self.function.body.layout.contains_block(block),
            "unknown block"
        );
        if let Some(prev) = self.function.body.layout.prev_block(block) {
            self.changed_block(prev);
        }
        while let Some(inst) = self.function.body.layout.first_inst(block) {
            self.invalidate_inst(inst);
        }
        self.function.body.blocks[block].params.clear();
        self.function.body.layout.remove_block(block);
        self.changed_block(block);
    }

    pub fn writer(&mut self) -> InstWriter<'_> {
        self.function
            .body
            .store
            .writer()
            .tracking(self.changes.as_deref_mut())
    }

    fn changed_block(&mut self, block: Block) {
        if let Some(changes) = &mut self.changes {
            changes.blocks.push(block);
        }
    }

    pub fn rewriter(&mut self, id: InstId) -> InstWriter<'_> {
        self.function
            .body
            .store
            .rewriter(id)
            .tracking(self.changes.as_deref_mut())
    }

    pub fn alloc_vreg_data(&mut self, data: VRegData) -> Reg {
        Reg::new_vreg(self.function.body.vregs.push(data).as_u32())
    }

    /// 分配新的虚拟寄存器，并显式指定寄存器 bank。
    pub fn alloc_vreg_in_bank(&mut self, ty: Type, bank: RegisterBank) -> Reg {
        self.alloc_vreg_data(VRegData {
            ty,
            bank: Some(bank),
        })
    }

    /// Create a typed virtual register without prescribing a register bank.
    pub fn alloc_vreg(&mut self, ty: Type) -> Reg {
        self.alloc_vreg_data(VRegData { ty, bank: None })
    }

    /// Split register allocation from append-only instruction construction.
    pub fn instruction_parts(&mut self) -> (VRegBuilder<'_>, crate::InstBuilder<'_>) {
        (
            VRegBuilder(&mut self.function.body.vregs),
            crate::InstBuilder {
                edge_transfers: Vec::new(),
                store: &mut self.function.body.store,
                changes: self.changes.as_deref_mut(),
            },
        )
    }

    pub fn set_call_stack(
        &mut self,
        id: InstId,
        slots: smallvec::SmallVec<[StackSlot; 2]>,
        stack: crate::StackArea,
    ) {
        self.function.body.store.set_call_stack(id, slots, stack);
        self.changed_inst(id);
    }

    pub fn set_inst_effects(&mut self, id: InstId, effects: crate::RegEffects) {
        self.function.body.store.set_effects(id, effects);
        self.changed_inst(id);
    }
    pub fn set_inst_inputs(&mut self, id: InstId, inputs: &[Reg]) {
        self.function.body.store.set_inputs(id, inputs);
        self.changed_inst(id);
    }
    pub fn set_inst_input(&mut self, id: InstId, index: usize, reg: Reg) {
        self.function.body.store.set_input(id, index, reg);
        self.changed_inst(id);
    }
    pub fn set_inst_results(&mut self, id: InstId, results: &[Reg]) {
        self.function.body.store.set_results(id, results);
        self.changed_inst(id);
    }
    pub fn set_inst_result(&mut self, id: InstId, index: usize, reg: Reg) {
        self.function.body.store.set_result(id, index, reg);
        self.changed_inst(id);
    }
    pub fn replace_uses(&mut self, old: VReg, new: VReg) {
        if old == new {
            return;
        }
        if let Some(changes) = &mut self.changes {
            changes.insts.extend(
                self.function
                    .body
                    .store
                    .uses(Reg::new_vreg(old.as_u32()))
                    .map(|site| site.inst()),
            );
        }
        self.function.body.store.replace_uses(old, new)
    }

    pub fn set_inst_memory(&mut self, id: InstId, access: Option<crate::MemoryAccess>) {
        self.function.body.store.set_memory(id, access);
        self.changed_inst(id);
    }

    pub fn alloc_stack_object(&mut self, object: StackObject, size: u32, align: u32) -> StackSlot {
        self.function.stack_frame.alloc_object(object, size, align)
    }

    /// Transfer a detached source into a stable destination ID without copying.
    /// Discards the destination's old payloads and invalidates the source ID.
    /// Passing the same ID is a no-op; this does not change block layout.
    pub fn replace_inst(&mut self, inst_id: InstId, source: InstId) {
        assert!(
            inst_id == source || self.inst_block(source).is_none(),
            "replacement source must be detached"
        );
        self.function.body.store.replace(inst_id, source);
        self.changed_inst(inst_id);
        self.changed_inst(source);
    }

    /// 将指令标记为无效。
    pub fn invalidate_inst(&mut self, inst_id: InstId) {
        self.detach_inst(inst_id);
        self.function.body.store.clear(inst_id);
    }

    /// Retarget one edge without changing its identity or argument use slots.
    pub fn redirect_edge(&mut self, edge: crate::EdgeId, target: Block) {
        assert!(
            self.function.body.layout.contains_block(target),
            "unknown successor"
        );
        let owner = self.function.body.store.redirect_edge(edge, target);
        self.changed_inst(owner);
    }

    /// Replace one edge's arguments and update their use-def links.
    pub fn set_edge_args(&mut self, edge: crate::EdgeId, args: &[Reg]) {
        let owner = self.function.body.store.set_edge_args(edge, args);
        self.changed_inst(owner);
    }

    pub fn create_edge(&mut self, block: Block, args: &[Reg]) -> crate::EdgeId {
        assert!(
            self.function.body.layout.contains_block(block),
            "unknown successor"
        );
        self.function.body.store.create_edge(block, args)
    }

    pub fn clone_edge(&mut self, edge: crate::EdgeId) -> crate::EdgeId {
        self.function.body.store.clone_edge(edge)
    }

    /// Preserve an original edge's identity on its replacement at commit.
    pub fn transfer_edge(&mut self, original: crate::EdgeId, replacement: crate::EdgeId) {
        let (from, to) = self
            .function
            .body
            .store
            .transfer_edge(original, replacement);
        self.changed_inst(from);
        self.changed_inst(to);
    }

    pub fn clear_successor_args(&mut self, inst: InstId) {
        self.function.body.store.clear_successor_args(inst);
        self.changed_inst(inst);
    }

    /// Run an edit session and report changed instructions and blocks. Edits
    /// commit as they happen; errors and unwinding do not imply rollback.
    /// Nested tracking is rejected; reborrow the active editor instead.
    pub fn track<R>(&mut self, rewrite: impl FnOnce(&mut FuncEditor<'_>) -> R) -> (R, EditChanges) {
        assert!(self.changes.is_none(), "nested edit tracking");
        let mut changes = EditChanges::default();
        let result = rewrite(&mut FuncEditor {
            function: self.function,
            changes: Some(&mut changes),
        });
        changes.insts.sort_unstable();
        changes.insts.dedup();
        changes.blocks.sort_unstable();
        changes.blocks.dedup();
        (result, changes)
    }
}
