//! Structural editing and scoped change reporting. Whole-program validation is explicit.
use super::*;

/// Mechanical edit notifications. They do not replace type or SSA validation.
#[derive(Debug, Default)]
pub struct EditChanges {
    pub insts: Vec<InstId>,
    pub blocks: Vec<Block>,
}

struct EditSession<'a>(&'a mut MachineFunction);
impl Drop for EditSession<'_> {
    fn drop(&mut self) {
        if self.0.body.changed_blocks.take().is_some() {
            // Restore tracking state even if the callback unwinds.
            let _ = self.0.body.store.finish_tracking();
        }
    }
}

/// Exclusive structural editing. Reads are available through Deref, but there
/// is deliberately no DerefMut or escape hatch to mutable function storage.
pub struct FuncEditor<'a> {
    function: &'a mut MachineFunction,
}
impl core::ops::Deref for FuncEditor<'_> {
    type Target = MachineFunction;
    fn deref(&self) -> &Self::Target {
        self.function
    }
}
impl MachineFunction {
    pub fn editor(&mut self) -> FuncEditor<'_> {
        FuncEditor { function: self }
    }
}
impl FuncEditor<'_> {
    pub fn create_block(&mut self) -> Block {
        if let Some(last) = self.function.body.layout.block_order().next_back() {
            if let Some(changes) = &mut self.function.body.changed_blocks {
                changes.push(last);
            }
        }
        let block = self.function.body.blocks.push(BlockData::default());
        self.function.body.layout.append_block(block);
        self.changed_block(block);
        self.function.body.entry.get_or_insert(block);
        block
    }
    pub fn set_entry_block(&mut self, block: Block) {
        assert!(
            self.function.body.layout.contains_block(block),
            "unknown entry block"
        );
        if let Some(old) = self.function.body.entry {
            self.changed_block(old);
        }
        self.function.body.entry = Some(block);
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
        }
        self.changed_block(block);
    }
    /// Replace one placed instruction with detached instructions in the supplied
    /// order. Including the root keeps its identity; otherwise it is erased.
    pub fn replace_with(&mut self, root: InstId, output: &[InstId]) {
        let block = self.inst_block(root).expect("replacement root is detached");
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
        self.changed_block(block);
    }
    pub fn insert_before(&mut self, anchor: InstId, inst: InstId) {
        let _ = self.inst(inst);
        self.function.body.layout.insert_before(anchor, inst);
        self.changed_block(self.inst_block(anchor).unwrap());
    }
    pub fn insert_after(&mut self, anchor: InstId, inst: InstId) {
        let _ = self.inst(inst);
        self.function.body.layout.insert_after(anchor, inst);
        self.changed_block(self.inst_block(anchor).unwrap());
    }
    pub fn detach_inst(&mut self, inst: InstId) {
        if let Some(block) = self.inst_block(inst) {
            self.changed_block(block);
        }
        self.function.body.layout.detach_inst(inst);
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
        assert_ne!(self.entry_block(), Some(block), "cannot erase entry block");
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
        self.function.body.store.writer()
    }

    fn changed_block(&mut self, block: Block) {
        if let Some(blocks) = &mut self.function.body.changed_blocks {
            blocks.push(block);
        }
    }

    pub fn rewriter(&mut self, id: InstId) -> InstWriter<'_> {
        self.function.body.store.rewriter(id)
    }

    pub fn set_inst_fields(&mut self, id: InstId, fields: &[crate::InstField]) {
        self.function.body.store.set_fields(id, fields);
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
                store: &mut self.function.body.store,
            },
        )
    }

    pub fn set_inst_effects(&mut self, id: InstId, effects: crate::RegEffects) {
        self.function.body.store.set_effects(id, effects);
    }
    pub fn set_inst_inputs(&mut self, id: InstId, inputs: &[Reg]) {
        self.function.body.store.set_inputs(id, inputs);
    }
    pub fn set_inst_input(&mut self, id: InstId, index: usize, reg: Reg) {
        self.function.body.store.set_input(id, index, reg);
    }
    pub fn set_inst_results(&mut self, id: InstId, results: &[Reg]) {
        self.function.body.store.set_results(id, results);
    }
    pub fn set_inst_result(&mut self, id: InstId, index: usize, reg: Reg) {
        self.function.body.store.set_result(id, index, reg);
    }
    pub fn set_inst_field(&mut self, id: InstId, index: usize, field: crate::InstField) {
        self.function.body.store.set_field(id, index, field);
    }
    pub fn replace_uses(&mut self, old: VReg, new: VReg) {
        self.function.body.store.replace_uses(old, new)
    }

    pub fn set_inst_memory(&mut self, id: InstId, access: Option<crate::MemoryAccess>) {
        self.function.body.store.set_memory(id, access);
    }

    /// 分配栈槽
    pub fn alloc_stack_slot(&mut self, size: u32, align: u32) -> StackSlot {
        self.function.stack_frame.alloc_slot(size, align)
    }

    /// 分配一个具有显式基址寄存器/偏移的栈槽。
    pub fn alloc_stack_slot_with_base(
        &mut self,
        base_reg: Reg,
        offset: i32,
        size: u32,
        align: u32,
    ) -> StackSlot {
        self.function.stack_frame.slots.push(StackSlotData {
            base: StackBase::Reg(base_reg),
            size,
            align,
            offset,
        })
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
    }

    /// 将指令标记为无效。
    pub fn invalidate_inst(&mut self, inst_id: InstId) {
        self.detach_inst(inst_id);
        self.function.body.store.write_at(
            inst_id,
            crate::MachineOpcode::Invalid,
            &[],
            &[],
            &[],
            None,
        );
    }

    /// Change an explicit successor and its arguments as one edit. The index
    /// counts block operands (or jump-table entries), not raw field positions.
    pub fn redirect_edge(&mut self, inst: InstId, edge: usize, target: Block, args: &[Reg]) {
        assert!(
            self.function.body.layout.contains_block(target),
            "unknown successor"
        );
        let mut extra = self.inst_extra(inst).map(|e| e.to_owned());
        if let Some(InstExtra::BrTable(table)) = &mut extra {
            let dest = table.targets.get_mut(edge).expect("unknown edge");
            dest.block = target;
            dest.args = args.iter().copied().collect();
            self.set_inst_extra(inst, extra.unwrap());
            return;
        }
        let fields = self.inst(inst).fields();
        let targets: Vec<_> = fields
            .iter()
            .enumerate()
            .filter_map(|(index, field)| {
                matches!(field, crate::InstField::Block(_)).then_some(index)
            })
            .collect();
        let index = *targets.get(edge).expect("unknown edge");
        match &mut extra {
            Some(InstExtra::Branch(info)) => {
                assert_eq!(targets.len(), 1, "invalid branch shape");
                info.args = args.iter().copied().collect();
            }
            Some(InstExtra::BranchCond(info)) => {
                assert_eq!(targets.len(), 2, "invalid conditional branch shape");
                if edge == 0 {
                    info.then_args = args.iter().copied().collect();
                } else {
                    info.else_args = args.iter().copied().collect();
                }
            }
            None if !args.is_empty() => {
                extra = Some(match targets.len() {
                    1 => InstExtra::Branch(crate::BranchInfo {
                        args: args.iter().copied().collect(),
                    }),
                    2 => InstExtra::BranchCond(crate::BranchCondInfo {
                        then_args: if edge == 0 {
                            args.iter().copied().collect()
                        } else {
                            Default::default()
                        },
                        else_args: if edge == 1 {
                            args.iter().copied().collect()
                        } else {
                            Default::default()
                        },
                    }),
                    _ => panic!("edge arguments require a branch payload"),
                });
            }
            _ => assert!(args.is_empty(), "payload cannot carry edge arguments"),
        }
        self.set_inst_field(inst, index, crate::InstField::Block(target));
        if let Some(extra) = extra {
            self.set_inst_extra(inst, extra);
        }
    }

    /// 为指令挂载额外 payload。
    pub fn set_inst_extra(&mut self, inst_id: InstId, extra: InstExtra) {
        self.function.body.store.set_extra(inst_id, extra);
    }

    /// 清理指令的额外 payload。
    pub fn clear_inst_extra(&mut self, inst_id: InstId) {
        self.function.body.store.clear_extra(inst_id);
    }
}

impl MachineFunction {
    /// Run an edit session and report changed instructions and blocks. Edits
    /// commit as they happen; errors and unwinding do not imply rollback.
    pub fn track_edits<R>(&mut self, rewrite: impl FnOnce(&mut Self) -> R) -> (R, EditChanges) {
        self.body.store.start_tracking();
        self.body.changed_blocks = Some(Vec::new());
        let session = EditSession(self);
        let result = rewrite(session.0);
        let mut insts = session.0.body.store.finish_tracking();
        let mut blocks = session.0.body.changed_blocks.take().unwrap();
        insts.sort_unstable();
        insts.dedup();
        blocks.sort_unstable();
        blocks.dedup();
        (result, EditChanges { insts, blocks })
    }
}
