//! Structural editing and scoped change reporting. Whole-program validation is explicit.
use super::*;

/// Instruction content and placement changes for incremental worklists.
/// Block-only edits are not reported; this is not an analysis invalidation log.
#[derive(Debug, Default)]
pub struct EditChanges {
    pub insts: Vec<InstId>,
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
    pub fn alloc_call_frame(&mut self, area: crate::StackArea) -> crate::CallFrameId {
        self.function.stack_frame.alloc_call(area)
    }
    /// Build complete instructions directly at the end of a block.
    pub fn at_end(&mut self, block: Block) -> InstInserter<'_> {
        assert!(self.layout().contains_block(block), "unknown block");
        InstInserter {
            editor: self.editor(),
            block,
            before: None,
        }
    }
    pub fn at_start(&mut self, block: Block) -> InstInserter<'_> {
        assert!(self.layout().contains_block(block), "unknown block");
        let before = self.layout().first_inst(block);
        InstInserter {
            editor: self.editor(),
            block,
            before,
        }
    }
    pub fn before(&mut self, inst: InstId) -> InstInserter<'_> {
        let block = self.inst_block(inst).expect("detached insertion anchor");
        InstInserter {
            editor: self.editor(),
            block,
            before: Some(inst),
        }
    }
    pub fn after(&mut self, inst: InstId) -> InstInserter<'_> {
        let block = self.inst_block(inst).expect("detached insertion anchor");
        let before = self.layout().next_inst(inst);
        InstInserter {
            editor: self.editor(),
            block,
            before,
        }
    }
    pub fn append_param(&mut self, param: Reg) {
        self.append_block_param(self.entry_block(), param);
    }

    /// Transfer entry definitions during ABI lowering. Their bindings must be
    /// rebuilt for the remaining register inputs; stack inputs become loads.
    pub fn take_params(&mut self) -> Vec<Reg> {
        self.function.body.entry_bindings.clear();
        let entry = self.entry_block();
        core::mem::take(&mut self.function.body.blocks[entry].params)
    }

    pub fn set_entry_bindings(&mut self, bindings: Vec<EntryBinding>) {
        self.function.body.entry_bindings = bindings;
    }
    pub fn set_inst_constraints(&mut self, id: InstId, constraints: Vec<crate::OperandConstraint>) {
        self.function.body.store.set_constraints(id, constraints);
        self.changed_inst(id);
    }

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
        let block = self.function.body.blocks.push(BlockData::default());
        self.function.body.layout.append_block(block);
        block
    }
    pub fn append_block_param(&mut self, block: Block, param: Reg) {
        assert!(
            self.function.body.layout.contains_block(block),
            "unknown block"
        );
        assert!(
            param.is_vreg(),
            "block parameters must be virtual registers"
        );
        self.function.body.blocks[block].params.push(param);
    }
    pub fn clear_block_params(&mut self) {
        self.function.body.entry_bindings.clear();
        let mut next = self.blocks().next();
        while let Some(block) = next {
            next = self.layout().next_block(block);
            self.function.body.blocks[block].params.clear();
        }
    }
    pub fn move_block_before(&mut self, block: Block, before: Block) {
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
    }
    fn detach_inst(&mut self, inst: InstId) {
        self.inst_block(inst).expect("detached instruction");

        self.function.body.layout.detach_inst(inst);
        self.changed_inst(inst);
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
            self.at_end(block).move_here(id);
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
        while let Some(inst) = self.function.body.layout.first_inst(block) {
            self.invalidate_inst(inst);
        }
        self.function.body.blocks[block].params.clear();
        self.function.body.layout.remove_block(block);
    }

    /// Rebuild one instruction in place, retaining its ID and layout position.
    /// Fields and clobbers are replaced with the new instruction.
    pub fn replace(&mut self, id: InstId) -> InstWriter<'_> {
        assert_ne!(self.inst(id).opcode(), crate::MachineOpcode::Invalid);
        InstWriter {
            store: &mut self.function.body.store,
            layout: &mut self.function.body.layout,
            changes: self.changes.as_deref_mut(),
            position: Position::Replace(id),
            clobbers: smallvec::SmallVec::new(),
            constraints: Vec::new(),
        }
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

    /// Replace instruction-local destruction effects. A call's ABI mask is
    /// owned by its call contract and remains in the unified clobbers() view.
    pub fn set_inst_clobbers(&mut self, id: InstId, clobbers: &[Reg]) {
        self.function.body.store.set_clobbers(id, clobbers);
        self.changed_inst(id);
    }
    /// Set complete call inputs, including the indirect callee, and record stack
    /// arguments and frame effects. Results, signature and memory attributes remain intact.
    /// Register placement requirements are attached separately.
    pub fn set_call_abi(
        &mut self,
        id: InstId,
        inputs: &[Reg],
        frame: crate::CallFrameId,
        clobbers: crate::RegMask,
        stack_args: smallvec::SmallVec<[StackSlot; 2]>,
    ) {
        self.function
            .body
            .store
            .set_call_abi(id, inputs, frame, clobbers, stack_args);
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
    /// Allocation replaces every register occurrence together, preserving the
    /// instruction's shape. Ordinary rewrites should use replacement builders.
    pub fn assign_registers(
        &mut self,
        id: InstId,
        results: &[crate::PReg],
        inputs: &[crate::PReg],
    ) {
        assert_eq!(results.len(), self.inst(id).results().len());
        assert_eq!(inputs.len(), self.inst(id).inputs().len());
        for (index, &reg) in results.iter().enumerate() {
            self.function.body.store.set_result(id, index, reg.into());
        }
        for (index, &reg) in inputs.iter().enumerate() {
            self.function.body.store.set_input(id, index, reg.into());
        }
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

    pub fn alloc_stack_object(&mut self, object: StackObject, size: u32, align: u32) -> StackSlot {
        self.function.stack_frame.alloc_object(object, size, align)
    }

    /// Commit a target's completed stack layout and its additional save slots.
    pub fn finalize_frame(&mut self, batch: crate::StackBatch, layout: crate::FrameLayout) {
        self.function.stack_frame.append(batch);
        self.function.stack_frame.finish(layout);
    }

    /// Replace a destination's contents with an existing instruction.
    /// The source is removed from its layout; the destination retains its ID and position.
    pub fn replace_inst(&mut self, inst_id: InstId, source: InstId) {
        assert!(
            !self.inst(inst_id).is_invalid(),
            "invalid replacement destination"
        );
        assert!(
            !self.inst(source).is_invalid(),
            "invalid replacement source"
        );
        assert!(
            self.inst_block(inst_id).is_some(),
            "detached replacement destination"
        );
        assert!(
            self.inst_block(source).is_some(),
            "detached replacement source"
        );
        if inst_id == source {
            return;
        }
        self.detach_inst(source);
        self.function.body.store.replace(inst_id, source);
        self.changed_inst(inst_id);
        self.changed_inst(source);
    }

    /// Erase a live, placed instruction. Repeated deletion is a caller error.
    pub fn invalidate_inst(&mut self, inst_id: InstId) {
        assert!(
            !self.inst(inst_id).is_invalid(),
            "instruction already invalid"
        );
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

    /// Run an edit session and report changed instructions. Edits
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
        (result, changes)
    }
}

/// A stable insertion gap for constructing or moving instructions.
/// Repeated operations preserve emission order before the fixed anchor.
pub struct InstInserter<'a> {
    editor: FuncEditor<'a>,
    block: Block,
    before: Option<InstId>,
}

enum Position {
    Insert {
        block: Block,
        before: Option<InstId>,
    },
    Replace(InstId),
}

impl core::ops::Deref for InstInserter<'_> {
    type Target = MachineFunction;
    fn deref(&self) -> &Self::Target {
        &self.editor
    }
}

impl InstInserter<'_> {
    /// Move a placed instruction to this gap without rebuilding its data.
    /// Moving the anchor is invalid; an instruction already at the gap is a no-op.
    pub fn move_here(&mut self, inst: InstId) {
        let source = self.inst_block(inst).expect("detached instruction");
        assert_ne!(self.before, Some(inst), "cannot move the insertion anchor");
        if source == self.block && self.layout().next_inst(inst) == self.before {
            return;
        }
        self.editor.detach_inst(inst);
        if let Some(anchor) = self.before {
            self.editor.function.body.layout.insert_before(anchor, inst);
        } else {
            self.editor
                .function
                .body
                .layout
                .append_inst(self.block, inst);
        }
    }

    /// Configure one complete instruction before committing it at this gap.

    pub fn edge(&mut self, block: crate::BlockId, args: &[Reg]) -> crate::EdgeId {
        self.editor.create_edge(block, args)
    }
    pub fn alloc_vreg(&mut self, ty: Type) -> Reg {
        self.editor.alloc_vreg(ty)
    }
    pub fn clone_edge(&mut self, edge: crate::EdgeId) -> crate::EdgeId {
        self.editor.clone_edge(edge)
    }
    pub fn alloc_stack_object(&mut self, object: StackObject, size: u32, align: u32) -> StackSlot {
        self.editor.alloc_stack_object(object, size, align)
    }
    pub fn writer(&mut self) -> InstWriter<'_> {
        let body = &mut self.editor.function.body;
        InstWriter {
            store: &mut body.store,
            changes: self.editor.changes.as_deref_mut(),
            layout: &mut body.layout,
            position: Position::Insert {
                block: self.block,
                before: self.before,
            },
            clobbers: smallvec::SmallVec::new(),
            constraints: Vec::new(),
        }
    }

    /// Low-level adapter for target instructions; ordinary passes use InstBuild.
    pub fn write(
        &mut self,
        opcode: crate::MachineOpcode,
        results: &[Reg],
        inputs: &[Reg],
        fields: crate::Fields,
    ) -> InstId {
        self.writer().write(opcode, results, inputs, fields)
    }
}

// Reborrow the cursor for each instruction so repeated emission keeps its gap.
impl crate::FieldBuild for &mut InstInserter<'_> {
    fn call_fields(
        &mut self,
        target: Option<crate::SymbolId>,
        info: crate::CallInfo,
    ) -> crate::Fields {
        self.writer().call_fields(target, info)
    }
    fn switch_fields(&mut self, edges: &[crate::EdgeId]) -> crate::Fields {
        self.writer().switch_fields(edges)
    }
}
impl crate::InstBuild for &mut InstInserter<'_> {
    type Inst = InstId;
    fn write(
        self,
        opcode: crate::GenericOpcode,
        results: &[Reg],
        inputs: &[Reg],
        fields: crate::Fields,
    ) -> InstId {
        self.writer().write(
            crate::MachineOpcode::Generic(opcode),
            results,
            inputs,
            fields,
        )
    }
}

/// A single committed write. Generated methods encode directly from their typed
/// arguments; no owning instruction or temporary operand vector is required.
pub struct InstWriter<'a> {
    changes: Option<&'a mut crate::EditChanges>,
    store: &'a mut crate::InstStore,
    layout: &'a mut crate::Layout,
    position: Position,
    clobbers: smallvec::SmallVec<[Reg; 4]>,
    constraints: Vec<crate::OperandConstraint>,
}

impl<'a> InstWriter<'a> {
    pub fn call_fields(
        &mut self,
        target: Option<crate::SymbolId>,
        info: crate::CallInfo,
    ) -> crate::Fields {
        self.store.call_fields(target, info)
    }

    pub fn switch_fields(&mut self, edges: &[crate::EdgeId]) -> crate::Fields {
        self.store.switch_fields(edges)
    }

    /// Copy the payload and its edges into independent, unowned storage.
    pub fn copy_fields(&mut self, id: InstId) -> crate::Fields {
        self.store.copy_fields(id)
    }

    pub fn edge(&mut self, block: crate::BlockId, args: &[Reg]) -> crate::EdgeId {
        assert!(self.layout.contains_block(block), "unknown successor");
        self.store.create_edge(block, args)
    }

    /// Add destruction effects without introducing value definitions.
    pub fn with_clobbers(mut self, regs: impl IntoIterator<Item = Reg>) -> Self {
        for reg in regs {
            assert!(reg.is_preg(), "clobbers require physical registers");
            if !self.clobbers.contains(&reg) {
                self.clobbers.push(reg);
            }
        }
        self
    }

    /// Attach occurrence constraints using the new instruction's operand layout.
    pub fn with_constraints(mut self, constraints: Vec<crate::OperandConstraint>) -> Self {
        self.constraints = constraints;
        self
    }

    /// Commit an already constructed instruction payload.
    pub fn write(
        self,
        opcode: crate::MachineOpcode,
        results: &[Reg],
        inputs: &[Reg],
        fields: crate::Fields,
    ) -> InstId {
        assert_ne!(
            opcode,
            crate::MachineOpcode::Invalid,
            "cannot construct an invalid instruction"
        );
        let id = match self.position {
            Position::Replace(id) => {
                self.store
                    .write_full_at(id, opcode, results, inputs, fields, &self.clobbers);
                id
            }
            Position::Insert { block, before } => {
                let id = self
                    .store
                    .write_full(opcode, results, inputs, fields, &self.clobbers);
                if let Some(anchor) = before {
                    self.layout.insert_before(anchor, id);
                } else {
                    self.layout.append_inst(block, id);
                }
                id
            }
        };
        self.store.set_constraints(id, self.constraints);
        if let Some(changes) = self.changes {
            changes.insts.push(id);
        }
        id
    }
}

// The generated contract owns generic builders; this adapter owns storage and
// the conversion from generic to machine opcodes.
impl crate::FieldBuild for InstWriter<'_> {
    fn call_fields(
        &mut self,
        target: Option<crate::SymbolId>,
        info: crate::CallInfo,
    ) -> crate::Fields {
        self.call_fields(target, info)
    }
    fn switch_fields(&mut self, edges: &[crate::EdgeId]) -> crate::Fields {
        self.switch_fields(edges)
    }
}
impl crate::InstBuild for InstWriter<'_> {
    type Inst = InstId;
    fn write(
        self,
        opcode: crate::GenericOpcode,
        results: &[Reg],
        inputs: &[Reg],
        fields: crate::Fields,
    ) -> InstId {
        self.write(
            crate::MachineOpcode::Generic(opcode),
            results,
            inputs,
            fields,
        )
    }
}
