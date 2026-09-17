//! LIR 机器函数与基本块定义

use super::{CallInfo, InstExtra, InstId, InstRef, Reg, StackSlot, VReg, VRegData};
use crate::InstWriter;
use crate::RegisterBank;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;
use cranelift_entity::PrimaryMap;
use veloc_mir::{Block, Type};

/// 机器基本块
#[derive(Debug, Clone)]
pub struct MachineBlock {
    pub id: Block,
    pub params: Vec<Reg>,
    pub insts: Vec<InstId>,
}

impl MachineBlock {
    pub fn new(id: Block) -> Self {
        Self {
            id,
            params: Vec::new(),
            insts: Vec::new(),
        }
    }

    pub fn append_inst_id(&mut self, inst_id: InstId) {
        self.insts.push(inst_id);
    }
}

/// 栈槽数据
#[derive(Debug, Clone, Copy)]
pub enum StackBase {
    /// Symbolic local frame base, resolved by the target at emission.
    Frame,
    Reg(Reg),
}

impl StackBase {
    pub fn resolve(self, frame: Reg) -> Reg {
        match self {
            Self::Frame => frame,
            Self::Reg(reg) => reg,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StackSlotData {
    pub base: StackBase,
    pub size: u32,
    pub align: u32,
    pub offset: i32,
}

/// 栈帧信息
#[derive(Debug, Clone)]
pub struct StackFrame {
    /// 局部变量占用的栈空间
    pub local_size: u32,
    /// 调用其他函数所需的最大传出参数区。
    pub arg_size: u32,
    /// 被调用者保存寄存器占用的空间
    pub callee_saved_size: u32,
    /// 当前函数实际使用到、需要保存恢复的 callee-saved 物理寄存器
    pub used_callee_saved: Vec<Reg>,
    /// 对齐后的总栈大小
    pub total_size: u32,
    /// 已分配的栈槽
    pub slots: cranelift_entity::PrimaryMap<StackSlot, StackSlotData>,
}

impl StackFrame {
    /// Allocate frame-relative storage without modifying instructions.
    pub fn alloc_slot(&mut self, size: u32, align: u32) -> StackSlot {
        assert!(align.is_power_of_two(), "invalid stack alignment");
        let end = self
            .local_size
            .checked_add(size)
            .expect("stack frame overflow");
        let end = end.checked_add(align - 1).expect("stack frame overflow") & !(align - 1);
        let offset = -i32::try_from(end).expect("stack frame exceeds signed offsets");
        self.local_size = end;
        self.slots.push(StackSlotData {
            base: StackBase::Frame,
            size,
            align,
            offset,
        })
    }
}

/// 机器函数主体数据。
#[derive(Debug, Clone)]
pub struct MachineFunction {
    pub name: String,
    pub blocks: Vec<MachineBlock>,
    store: crate::store::InstStore,
    pub vregs: PrimaryMap<VReg, VRegData>,
    pub stack_frame: StackFrame,
    /// 函数参数对应的虚拟寄存器
    pub params: Vec<Reg>,
}

/// 基本块重写游标。
/// 向输出布局写入新顺序，并提供更语义化的 keep/remove/replace/insert 操作。
pub struct BlockRewriteCursor<'a> {
    mfunc: &'a mut MachineFunction,
    current: InstId,
    output: &'a mut Vec<InstId>,
    resolved_current: bool,
}

impl<'a> BlockRewriteCursor<'a> {
    fn new(mfunc: &'a mut MachineFunction, current: InstId, output: &'a mut Vec<InstId>) -> Self {
        Self {
            mfunc,
            current,
            output,
            resolved_current: false,
        }
    }

    pub fn current_inst_id(&self) -> InstId {
        self.current
    }

    pub fn current_inst(&self) -> InstRef<'_> {
        self.mfunc.inst(self.current)
    }

    pub fn current_extra(&self) -> Option<&InstExtra> {
        self.mfunc.inst_extra(self.current)
    }

    pub fn clear_current_extra(&mut self) {
        self.mfunc.clear_inst_extra(self.current);
    }

    pub fn set_current_extra(&mut self, extra: InstExtra) {
        self.mfunc.set_inst_extra(self.current, extra);
    }

    pub fn mfunc(&self) -> &MachineFunction {
        self.mfunc
    }

    pub fn mfunc_mut(&mut self) -> &mut MachineFunction {
        self.mfunc
    }

    /// Append a stored instruction to the rewritten block's output.
    /// Call before or after keep_current() to position it on either side.
    pub fn emit(&mut self, id: InstId) {
        self.output.push(id);
    }

    pub fn emit_many(&mut self, ids: impl IntoIterator<Item = InstId>) {
        self.output.extend(ids);
    }

    pub fn keep_current(&mut self) {
        debug_assert!(
            !self.resolved_current,
            "current instruction {:?} has already been resolved",
            self.current
        );
        self.output.push(self.current);
        self.resolved_current = true;
    }

    /// Remove the current ID from this layout without deleting its contents.
    /// Used by worklist-based rewrites that may reinsert the same instruction.
    pub fn detach_current(&mut self) {
        assert!(
            !self.resolved_current,
            "current instruction already resolved"
        );
        self.resolved_current = true;
    }

    pub fn remove_current(&mut self) {
        debug_assert!(
            !self.resolved_current,
            "current instruction {:?} has already been resolved",
            self.current
        );
        self.mfunc.invalidate_inst(self.current);
        self.resolved_current = true;
    }

    pub fn replace_current(&mut self, inst: InstId) {
        debug_assert!(
            !self.resolved_current,
            "current instruction {:?} has already been resolved",
            self.current
        );
        self.mfunc.replace_inst(self.current, inst);
        self.output.push(self.current);
        self.resolved_current = true;
    }

    fn finish(mut self) {
        if !self.resolved_current {
            self.output.push(self.current);
            self.resolved_current = true;
        }
    }
}

impl MachineFunction {
    pub fn new(name: String) -> Self {
        Self {
            name,
            blocks: Vec::new(),
            store: crate::InstStore::default(),
            vregs: PrimaryMap::new(),
            stack_frame: StackFrame {
                local_size: 0,
                arg_size: 0,
                callee_saved_size: 0,
                used_callee_saved: Vec::new(),
                total_size: 0,
                slots: PrimaryMap::new(),
            },
            params: Vec::new(),
        }
    }

    /// 获取指定基本块的指令 ID 列表
    pub fn block_insts(&self, block_idx: usize) -> &[InstId] {
        &self.blocks[block_idx].insts
    }

    /// 获取基本块数量
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// 获取基本块（只读）
    pub fn get_block(&self, block_idx: usize) -> &MachineBlock {
        &self.blocks[block_idx]
    }

    /// 按 LIR block id 查找块下标。
    pub fn find_block_index(&self, block: Block) -> Option<usize> {
        self.blocks.iter().position(|mblock| mblock.id == block)
    }

    /// 获取指定块的 block 参数。
    pub fn block_params(&self, block: Block) -> Option<&[Reg]> {
        self.find_block_index(block)
            .map(|block_idx| self.blocks[block_idx].params.as_slice())
    }

    /// 创建一个新的合成基本块，返回其 block id。
    pub fn create_synthetic_block(&mut self) -> Block {
        let next_id = self
            .blocks
            .iter()
            .map(|block| block.id.as_u32())
            .max()
            .map_or(0, |max_id| max_id + 1);
        let block = Block::from_u32(next_id);
        self.blocks.push(MachineBlock::new(block));
        block
    }

    /// 获取所有块的迭代器
    pub fn blocks(&self) -> impl Iterator<Item = &MachineBlock> {
        self.blocks.iter()
    }

    /// 将一个已分配指令追加到指定基本块末尾。
    pub fn append_inst_id_to_block(&mut self, block_idx: usize, inst_id: InstId) {
        self.blocks[block_idx].insts.push(inst_id);
    }

    /// 用新列表替换一个基本块的指令列表。
    fn replace_block_insts(&mut self, block_idx: usize, new_insts: Vec<InstId>) {
        self.blocks[block_idx].insts = new_insts;
    }

    pub fn writer(&mut self) -> InstWriter<'_> {
        self.store.writer()
    }

    /// Observe committed instruction edits, including RAUW of ordinary and edge
    /// operands. Observers are scoped to a rewrite, so normal construction does
    /// not allocate a change log. Return errors as values to close the scope.
    pub fn track_inst_changes<R>(
        &mut self,
        rewrite: impl FnOnce(&mut Self) -> R,
    ) -> (R, Vec<InstId>) {
        self.store.start_tracking();
        let result = rewrite(self);
        (result, self.store.finish_tracking())
    }

    pub fn rewriter(&mut self, id: InstId) -> InstWriter<'_> {
        self.store.rewriter(id)
    }

    pub fn set_inst_fields(&mut self, id: InstId, fields: &[crate::InstField]) {
        self.store.set_fields(id, fields);
    }

    pub fn rewrite_block<E, F>(&mut self, block_idx: usize, mut f: F) -> Result<(), E>
    where
        F: FnMut(&mut BlockRewriteCursor<'_>) -> Result<(), E>,
    {
        let old_insts = self.blocks[block_idx].insts.clone();
        let mut new_insts = Vec::with_capacity(old_insts.len());

        for inst_id in old_insts {
            let mut cursor = BlockRewriteCursor::new(self, inst_id, &mut new_insts);
            f(&mut cursor)?;
            cursor.finish();
        }

        self.replace_block_insts(block_idx, new_insts);
        Ok(())
    }

    fn alloc_vreg_with_bank_opt(&mut self, ty: Type, bank: Option<RegisterBank>) -> Reg {
        let vreg = self.vregs.push(VRegData { ty, bank });
        Reg::new_vreg(vreg.as_u32())
    }

    /// 分配新的虚拟寄存器，并显式指定寄存器 bank。
    pub fn alloc_vreg_in_bank(&mut self, ty: Type, bank: RegisterBank) -> Reg {
        self.alloc_vreg_with_bank_opt(ty, Some(bank))
    }

    /// Create a typed virtual register without prescribing a register bank.
    pub fn alloc_vreg(&mut self, ty: Type) -> Reg {
        self.alloc_vreg_with_bank_opt(ty, None)
    }

    /// Split immutable register facts from mutable instruction storage.
    pub fn instruction_parts(
        &mut self,
    ) -> (&mut PrimaryMap<VReg, VRegData>, &mut crate::InstStore) {
        (&mut self.vregs, &mut self.store)
    }

    pub fn inst(&self, id: InstId) -> InstRef<'_> {
        self.store.get(id)
    }

    pub fn inst_count(&self) -> usize {
        self.store.len()
    }

    pub fn set_inst_effects(&mut self, id: InstId, effects: crate::RegEffects) {
        self.store.set_effects(id, effects);
    }
    pub fn set_inst_inputs(&mut self, id: InstId, inputs: &[Reg]) {
        self.store.set_inputs(id, inputs);
    }
    pub fn set_inst_input(&mut self, id: InstId, index: usize, reg: Reg) {
        self.store.set_input(id, index, reg);
    }
    pub fn set_inst_results(&mut self, id: InstId, results: &[Reg]) {
        self.store.set_results(id, results);
    }
    pub fn set_inst_result(&mut self, id: InstId, index: usize, reg: Reg) {
        self.store.set_result(id, index, reg);
    }
    pub fn set_inst_field(&mut self, id: InstId, index: usize, field: crate::InstField) {
        self.store.set_field(id, index, field);
    }

    pub fn uses(&self, reg: Reg) -> crate::RegRefs<'_> {
        self.store.uses(reg)
    }
    pub fn defs(&self, reg: Reg) -> crate::RegRefs<'_> {
        self.store.defs(reg)
    }
    pub fn replace_uses(&mut self, old: VReg, new: VReg) {
        self.store.replace_uses(old, new)
    }
    pub fn check_refs(&self) -> Result<(), &'static str> {
        self.store.check_refs()
    }

    pub fn set_inst_memory(&mut self, id: InstId, access: Option<crate::MemoryAccess>) {
        self.store.set_memory(id, access);
    }

    /// 获取虚拟寄存器数据
    pub fn vreg_data(&self, reg: Reg) -> &VRegData {
        debug_assert!(reg.is_vreg());
        &self.vregs[VReg::from_u32(reg.index())]
    }

    /// 获取虚拟寄存器数据（可变）
    pub fn vreg_data_mut(&mut self, reg: Reg) -> &mut VRegData {
        debug_assert!(reg.is_vreg());
        &mut self.vregs[VReg::from_u32(reg.index())]
    }

    /// 分配栈槽
    pub fn alloc_stack_slot(&mut self, size: u32, align: u32) -> StackSlot {
        self.stack_frame.alloc_slot(size, align)
    }

    /// 分配一个具有显式基址寄存器/偏移的栈槽。
    pub fn alloc_stack_slot_with_base(
        &mut self,
        base_reg: Reg,
        offset: i32,
        size: u32,
        align: u32,
    ) -> StackSlot {
        self.stack_frame.slots.push(StackSlotData {
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
        self.store.replace(inst_id, source);
    }

    /// 将指令标记为无效。
    pub fn invalidate_inst(&mut self, inst_id: InstId) {
        self.store
            .write_at(inst_id, crate::MachineOpcode::Invalid, &[], &[], &[], None);
    }

    /// 为指令挂载额外 payload。
    pub fn set_inst_extra(&mut self, inst_id: InstId, extra: InstExtra) {
        self.store.set_extra(inst_id, extra);
    }

    /// 清理指令的额外 payload。
    pub fn clear_inst_extra(&mut self, inst_id: InstId) {
        self.store.clear_extra(inst_id);
    }

    /// 获取指令的额外 payload。
    pub fn inst_extra(&self, inst_id: InstId) -> Option<&InstExtra> {
        self.store.extra(inst_id)
    }

    /// 获取调用指令的签名信息。
    pub fn call_info(&self, inst_id: InstId) -> &CallInfo {
        match self.inst_extra(inst_id) {
            Some(InstExtra::Call(info)) => info,
            Some(_) => panic!(
                "instruction {:?} in `{}` does not carry call info payload",
                inst_id, self.name
            ),
            None => panic!(
                "call instruction {:?} in `{}` is missing call info payload",
                inst_id, self.name
            ),
        }
    }

    /// 生成便于调试的文本格式 LIR。
    pub fn format_for_dump(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "function {}", self.name);

        if !self.params.is_empty() {
            let params = self
                .params
                .iter()
                .map(|reg| format!("{:?}:{}", reg, self.vreg_data(*reg).ty))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(out, "  params: {}", params);
        }

        for block in &self.blocks {
            let _ = writeln!(out, "  block {:?}:", block.id);
            if !block.params.is_empty() {
                let params = block
                    .params
                    .iter()
                    .map(|reg| format!("{:?}:{}", reg, self.vreg_data(*reg).ty))
                    .collect::<Vec<_>>()
                    .join(", ");
                let _ = writeln!(out, "    params: {}", params);
            }
            for &inst_id in &block.insts {
                let inst = self.inst(inst_id);
                let _ = write!(out, "    {:?}: {:?}", inst_id, inst);
                if let Some(extra) = self.inst_extra(inst_id) {
                    let _ = write!(out, " extra={:?}", extra);
                }
                let _ = writeln!(out);
            }
        }

        out
    }
}
