//! LIR 机器函数与基本块定义

use super::{
    CallInfo, CallInst, InstExtra, InstExtraId, InstId, MachineInst, Reg, StackSlot, VReg, VRegData,
};
use crate::pipeline::stages::AllowsUnbankedVRegAlloc;
use crate::regalloc::regbank_select::RegisterBank;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};
use cranelift_entity::{PrimaryMap, SecondaryMap};
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
#[derive(Debug, Clone)]
pub struct StackSlotData {
    pub(crate) base_reg: Reg,
    #[allow(dead_code)]
    pub(crate) size: u32,
    #[allow(dead_code)]
    pub(crate) align: u32,
    pub(crate) offset: i32,
}

/// 栈帧信息
#[derive(Debug, Clone)]
pub struct StackFrame {
    /// 局部变量占用的栈空间
    pub(crate) local_size: u32,
    /// 传入参数占用的栈空间
    pub(crate) arg_size: u32,
    /// 被调用者保存寄存器占用的空间
    pub(crate) callee_saved_size: u32,
    /// 当前函数实际使用到、需要保存恢复的 callee-saved 物理寄存器
    pub(crate) used_callee_saved: Vec<Reg>,
    /// 对齐后的总栈大小
    pub(crate) total_size: u32,
    /// 已分配的栈槽
    pub(crate) slots: cranelift_entity::PrimaryMap<StackSlot, StackSlotData>,
}

/// 机器函数主体数据。
#[derive(Debug, Clone)]
pub struct MachineFunctionData {
    pub name: String,
    pub(crate) blocks: Vec<MachineBlock>,
    pub(crate) dfg: PrimaryMap<InstId, MachineInst>,
    pub(crate) inst_extra: SecondaryMap<InstId, Option<InstExtraId>>,
    pub(crate) extras: PrimaryMap<InstExtraId, InstExtra>,
    pub(crate) vregs: PrimaryMap<VReg, VRegData>,
    pub(crate) stack_frame: StackFrame,
    /// 函数参数对应的虚拟寄存器
    pub(crate) params: Vec<Reg>,
    /// 是否已指令选择
    pub(crate) is_selected: bool,
    /// 是否已寄存器分配
    pub(crate) is_regallocated: bool,
}

/// 机器函数。
///
/// `S` 是阶段标记类型，用来表达当前函数处于哪一个 codegen 阶段。
#[derive(Debug, Clone)]
pub struct MachineFunction<S> {
    data: MachineFunctionData,
    _stage: PhantomData<S>,
}

/// 基本块重写游标。
///
/// 适合在“单次扫描、单次提交”的 block 布局重写里使用。游标负责跟踪当前指令、
/// 向输出布局写入新顺序，并提供更语义化的 keep/remove/replace/insert 操作。
pub struct BlockRewriteCursor<'a, S> {
    mfunc: &'a mut MachineFunction<S>,
    current: InstId,
    output: &'a mut Vec<InstId>,
    resolved_current: bool,
}

impl<'a, S> BlockRewriteCursor<'a, S> {
    fn new(
        mfunc: &'a mut MachineFunction<S>,
        current: InstId,
        output: &'a mut Vec<InstId>,
    ) -> Self {
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

    pub fn current_inst(&self) -> &MachineInst {
        &self.mfunc.dfg[self.current]
    }

    pub fn current_inst_clone(&self) -> MachineInst {
        self.mfunc.dfg[self.current].clone()
    }

    pub fn current_extra(&self) -> Option<&InstExtra> {
        self.mfunc.inst_extra(self.current)
    }

    pub fn current_extra_cloned(&self) -> Option<InstExtra> {
        self.current_extra().cloned()
    }

    pub fn clear_current_extra(&mut self) {
        self.mfunc.clear_inst_extra(self.current);
    }

    pub fn set_current_extra(&mut self, extra: InstExtra) {
        self.mfunc.set_inst_extra(self.current, extra);
    }

    pub fn mfunc(&self) -> &MachineFunction<S> {
        self.mfunc
    }

    pub fn mfunc_mut(&mut self) -> &mut MachineFunction<S> {
        self.mfunc
    }

    pub fn emit_before(&mut self, inst: MachineInst) -> InstId {
        let inst_id = self.mfunc.alloc_inst(inst);
        self.output.push(inst_id);
        inst_id
    }

    pub fn emit_before_many<I>(&mut self, insts: I) -> Vec<InstId>
    where
        I: IntoIterator<Item = MachineInst>,
    {
        let mut ids = Vec::new();
        for inst in insts {
            ids.push(self.emit_before(inst));
        }
        ids
    }

    pub fn emit_existing_before(&mut self, inst_id: InstId) {
        self.output.push(inst_id);
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

    pub fn remove_current(&mut self) {
        debug_assert!(
            !self.resolved_current,
            "current instruction {:?} has already been resolved",
            self.current
        );
        self.resolved_current = true;
    }

    pub fn replace_current(&mut self, inst: MachineInst) {
        debug_assert!(
            !self.resolved_current,
            "current instruction {:?} has already been resolved",
            self.current
        );
        self.mfunc.replace_inst(self.current, inst);
        self.output.push(self.current);
        self.resolved_current = true;
    }

    pub fn replace_current_with_many<I>(&mut self, insts: I)
    where
        I: IntoIterator<Item = MachineInst>,
    {
        debug_assert!(
            !self.resolved_current,
            "current instruction {:?} has already been resolved",
            self.current
        );
        self.mfunc.invalidate_inst(self.current);
        for inst in insts {
            let inst_id = self.mfunc.alloc_inst(inst);
            self.output.push(inst_id);
        }
        self.resolved_current = true;
    }

    fn finish(mut self) {
        if !self.resolved_current {
            self.output.push(self.current);
            self.resolved_current = true;
        }
    }
}

impl<S> Deref for MachineFunction<S> {
    type Target = MachineFunctionData;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<S> DerefMut for MachineFunction<S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<S> MachineFunction<S> {
    pub fn new(name: String) -> Self {
        Self {
            data: MachineFunctionData {
                name,
                blocks: Vec::new(),
                dfg: PrimaryMap::new(),
                inst_extra: SecondaryMap::new(),
                extras: PrimaryMap::new(),
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
                is_selected: false,
                is_regallocated: false,
            },
            _stage: PhantomData,
        }
    }

    pub fn into_stage<T>(self) -> MachineFunction<T> {
        MachineFunction {
            data: self.data,
            _stage: PhantomData,
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

    /// 以“编辑单条指令内容”的方式修改指定指令。
    ///
    /// 该接口不会修改 block 布局，只会在回调成功后用新指令覆盖旧指令，并同步
    /// use-def 链。适合做 opcode/operand 的原地改写。
    pub fn edit_inst<E, F>(&mut self, inst_id: InstId, f: F) -> Result<(), E>
    where
        F: FnOnce(&mut MachineInst) -> Result<(), E>,
    {
        let mut inst = self.dfg[inst_id].clone();
        f(&mut inst)?;
        self.replace_inst(inst_id, inst);
        Ok(())
    }

    /// 以 `BlockRewriteCursor` 的方式重写一个 block 的布局。
    ///
    /// 适合做插入前置/后置序列、删除当前指令、用多条指令替换当前指令等布局变换。
    pub fn rewrite_block<E, F>(&mut self, block_idx: usize, mut f: F) -> Result<(), E>
    where
        F: FnMut(&mut BlockRewriteCursor<'_, S>) -> Result<(), E>,
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

    /// 分配并追加指令到指定基本块末尾。
    pub fn alloc_inst_and_append_to_block(
        &mut self,
        block_idx: usize,
        inst: MachineInst,
    ) -> InstId {
        let inst_id = self.alloc_inst(inst);
        self.append_inst_id_to_block(block_idx, inst_id);
        inst_id
    }

    fn alloc_vreg_with_bank_opt(&mut self, ty: Type, bank: Option<RegisterBank>) -> Reg {
        let vreg = self.vregs.push(VRegData {
            ty,
            bank,
            assigned_reg: None,
            stack_slot: None,
        });
        Reg::new_vreg(vreg.as_u32())
    }

    /// 分配新的虚拟寄存器，并显式指定寄存器 bank。
    pub fn alloc_vreg_in_bank(&mut self, ty: Type, bank: RegisterBank) -> Reg {
        self.alloc_vreg_with_bank_opt(ty, Some(bank))
    }

    /// 在 DFG 中分配新指令
    pub fn alloc_inst(&mut self, inst: MachineInst) -> InstId {
        let inst_id = self.dfg.push(inst);
        self.inst_extra[inst_id] = None;
        inst_id
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
    pub fn alloc_stack_slot(&mut self, base_reg: Reg, size: u32, align: u32) -> StackSlot {
        let offset = -(self.stack_frame.local_size as i32 + size as i32);
        self.stack_frame.local_size += size;
        // 对齐
        let misalign = self.stack_frame.local_size % align;
        if misalign != 0 {
            self.stack_frame.local_size += align - misalign;
        }

        self.stack_frame.slots.push(StackSlotData {
            base_reg,
            size,
            align,
            offset,
        })
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
            base_reg,
            size,
            align,
            offset,
        })
    }

    /// 替换指令。
    pub fn replace_inst(&mut self, inst_id: InstId, inst: MachineInst) {
        self.clear_inst_extra(inst_id);
        self.dfg[inst_id] = inst;
    }

    /// 将指令标记为无效。
    pub fn invalidate_inst(&mut self, inst_id: InstId) {
        self.replace_inst(inst_id, MachineInst::invalid());
    }

    /// 为指令挂载额外 payload。
    pub fn set_inst_extra(&mut self, inst_id: InstId, extra: InstExtra) {
        let extra_id = self.extras.push(extra);
        self.inst_extra[inst_id] = Some(extra_id);
    }

    /// 清理指令的额外 payload。
    pub fn clear_inst_extra(&mut self, inst_id: InstId) {
        self.inst_extra[inst_id] = None;
    }

    /// 获取指令的额外 payload。
    pub fn inst_extra(&self, inst_id: InstId) -> Option<&InstExtra> {
        self.inst_extra[inst_id].map(|extra_id| &self.extras[extra_id])
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

    /// 以 typed view 的形式解码 call/call_indirect。
    pub fn as_call(&self, inst_id: InstId) -> CallInst<'_> {
        let inst = &self.dfg[inst_id];
        CallInst {
            shape: inst.as_call_shape(),
            info: self.call_info(inst_id),
        }
    }

    /// 生成便于调试的文本格式 LIR。
    pub fn format_for_dump(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "function {} [selected={}, regalloc={}]",
            self.name, self.is_selected, self.is_regallocated
        );

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
                let inst = &self.dfg[inst_id];
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

impl<S: AllowsUnbankedVRegAlloc> MachineFunction<S> {
    /// 分配新的虚拟寄存器。
    ///
    /// 只允许在 `regbankselect` 之前的阶段调用；进入后续阶段后，必须使用
    /// `alloc_vreg_in_bank()` 显式携带 bank。
    pub fn alloc_vreg(&mut self, ty: Type) -> Reg {
        self.alloc_vreg_with_bank_opt(ty, None)
    }
}
