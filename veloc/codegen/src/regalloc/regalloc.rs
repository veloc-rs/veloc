//! Register Allocator - 寄存器分配器
//!
//! 提供目标无关的通用寄存器分配算法（如线性扫描）

use crate::mir::{
    InstExtra, MachineFunction, MachineInst, MachineOpcode, MachineOperand, Reg, StackSlot,
    Writable,
};
use crate::target::arch::{CallConv, RegClass, TargetMachine};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use veloc_ir::Type;

/// 活跃区间
#[derive(Debug, Clone)]
pub struct LiveInterval {
    pub vreg: Reg,
    pub start: u32, // 起始指令编号
    pub end: u32,   // 结束指令编号
    pub class: RegClass,
}

/// 寄存器分配器。
///
/// 这是一个通用的线性扫描寄存器分配器，通过 `TargetMachine::desc()`
/// 获取目标相关的寄存器约束和数据布局。
pub struct RegisterAllocator<'a> {
    target: &'a dyn TargetMachine,
    /// 寄存器分配映射: Reg -> Reg
    allocation: BTreeMap<Reg, Reg>,
    /// 当前活跃的物理寄存器及其释放位置
    active_regs: BTreeMap<Reg, u32>,
    /// 溢出到栈的虚拟寄存器
    spilled: BTreeMap<Reg, StackSlot>,
}

impl<'a> RegisterAllocator<'a> {
    pub fn new(target: &'a dyn TargetMachine) -> Self {
        Self {
            target,
            allocation: BTreeMap::new(),
            active_regs: BTreeMap::new(),
            spilled: BTreeMap::new(),
        }
    }

    /// 执行寄存器分配
    pub fn allocate(&mut self, mfunc: &mut MachineFunction, call_conv: veloc_ir::CallConv) {
        // 1. 计算活跃区间
        let intervals = self.compute_live_intervals(mfunc);
        let preserved_regs = CallConv::from(call_conv).preserved_regs(self.target.desc().arch);
        let call_positions = self.collect_call_positions(mfunc);

        // 2. 线性扫描分配
        for interval in intervals {
            // 释放已经过期的寄存器
            self.expire_old_intervals(interval.start);

            // 尝试分配寄存器
            if let Some(preg) = self.try_allocate(&interval, &preserved_regs, &call_positions) {
                self.allocation.insert(interval.vreg, preg);
                self.active_regs.insert(preg, interval.end);

                // 更新 MachineFunction 中的 VRegData
                let data = mfunc.vreg_data_mut(interval.vreg);
                data.assigned_reg = Some(preg);
            } else {
                // 需要溢出
                self.spill_vreg(interval.vreg, mfunc);
            }
        }

        // 3. 重写指令中的虚拟寄存器引用
        self.rewrite_instructions(mfunc);

        mfunc.is_regallocated = true;
    }

    /// 计算活跃区间
    fn compute_live_intervals(&self, mfunc: &MachineFunction) -> Vec<LiveInterval> {
        let mut intervals: BTreeMap<Reg, LiveInterval> = BTreeMap::new();
        let mut inst_idx: u32 = 0;

        for block in &mfunc.blocks {
            for &inst_id in &block.insts {
                let inst = &mfunc.dfg[inst_id];
                // 记录定义点
                for def in inst.defs() {
                    if !def.is_vreg() {
                        continue;
                    }
                    let class = self.get_vreg_class(def, mfunc);
                    intervals.entry(def).or_insert_with(|| LiveInterval {
                        vreg: def,
                        start: inst_idx,
                        end: inst_idx,
                        class,
                    });
                }

                // 记录使用点，扩展活跃区间
                for use_vreg in inst.uses() {
                    if !use_vreg.is_vreg() {
                        continue;
                    }
                    if let Some(interval) = intervals.get_mut(&use_vreg) {
                        interval.end = inst_idx;
                    } else {
                        let class = self.get_vreg_class(use_vreg, mfunc);
                        intervals.insert(
                            use_vreg,
                            LiveInterval {
                                vreg: use_vreg,
                                start: 0,
                                end: inst_idx,
                                class,
                            },
                        );
                    }
                }
                inst_idx += 1;
            }
        }

        let mut result: Vec<_> = intervals.into_values().collect();
        result.sort_by_key(|i| i.start);
        result
    }

    fn get_vreg_class(&self, vreg: Reg, mfunc: &MachineFunction) -> RegClass {
        let ty = mfunc.vreg_data(vreg).ty;
        self.target.desc().reg_class_for_type(&ty)
    }

    fn collect_call_positions(&self, mfunc: &MachineFunction) -> Vec<u32> {
        let mut positions = Vec::new();
        let mut inst_idx = 0;

        for block in &mfunc.blocks {
            for &inst_id in &block.insts {
                if matches!(mfunc.inst_extra(inst_id), Some(InstExtra::Call(_))) {
                    positions.push(inst_idx);
                }
                inst_idx += 1;
            }
        }

        positions
    }

    /// 释放过期的寄存器
    fn expire_old_intervals(&mut self, pos: u32) {
        let to_remove: Vec<_> = self
            .active_regs
            .iter()
            .filter(|(_, end)| **end < pos)
            .map(|(reg, _)| *reg)
            .collect();

        for reg in to_remove {
            self.active_regs.remove(&reg);
        }
    }

    /// 尝试为虚拟寄存器分配物理寄存器
    fn try_allocate(
        &self,
        interval: &LiveInterval,
        preserved_regs: &[Reg],
        call_positions: &[u32],
    ) -> Option<Reg> {
        let available_regs = self.target.desc().allocatable_regs_in_class(interval.class);
        let crosses_call = call_positions
            .iter()
            .any(|&pos| interval.start < pos && pos < interval.end);

        if crosses_call {
            for &preg in available_regs {
                if preserved_regs.contains(&preg) && !self.active_regs.contains_key(&preg) {
                    return Some(preg);
                }
            }
        }

        // 找一个不在活跃集合且不是保留的寄存器
        for &preg in available_regs {
            if !self.active_regs.contains_key(&preg) {
                return Some(preg);
            }
        }

        None
    }

    /// 将虚拟寄存器溢出到栈
    fn spill_vreg(&mut self, vreg: Reg, mfunc: &mut MachineFunction) {
        let ty = mfunc.vreg_data(vreg).ty;
        let size = self.target.desc().data_layout.type_size(&ty);
        let align = self.target.desc().data_layout.type_align(&ty);
        let slot = mfunc.alloc_stack_slot(size, align);
        self.spilled.insert(vreg, slot);

        let data = mfunc.vreg_data_mut(vreg);
        data.stack_slot = Some(slot);
    }

    /// 重写指令，替换虚拟寄存器
    fn rewrite_instructions(&self, mfunc: &mut MachineFunction) {
        let num_blocks = mfunc.num_blocks();
        for block_idx in 0..num_blocks {
            mfunc
                .rewrite_block_insts(block_idx, |mfunc, inst_id, output| {
                    let mut inst = mfunc.dfg[inst_id].clone();
                    let mut before = Vec::new();
                    let mut after = Vec::new();
                    let mut scratch_bindings: Vec<(Reg, Reg)> = Vec::new();
                    let mut used_gpr_scratch = 0usize;
                    let mut used_fpr_scratch = 0usize;

                    for operand in &mut inst.operands {
                        match operand {
                            MachineOperand::Use(vreg) if vreg.is_vreg() => {
                                if let Some(&preg) = self.allocation.get(vreg) {
                                    *operand = MachineOperand::Use(preg);
                                } else if let Some(&slot) = self.spilled.get(vreg) {
                                    let ty = mfunc.vreg_data(*vreg).ty;
                                    let scratch = self.ensure_scratch_reg(
                                        *vreg,
                                        ty,
                                        &mut scratch_bindings,
                                        &mut used_gpr_scratch,
                                        &mut used_fpr_scratch,
                                    );
                                    if !before.iter().any(|(loaded_slot, reg, _)| {
                                        *loaded_slot == slot && *reg == scratch
                                    }) {
                                        before.push((slot, scratch, ty));
                                    }
                                    *operand = MachineOperand::Use(scratch);
                                }
                            }
                            MachineOperand::Def(w) if w.to_reg().is_vreg() => {
                                let vreg = w.to_reg();
                                if let Some(&preg) = self.allocation.get(&vreg) {
                                    *operand = MachineOperand::Def(Writable(preg));
                                } else if let Some(&slot) = self.spilled.get(&vreg) {
                                    let ty = mfunc.vreg_data(vreg).ty;
                                    let scratch = self.ensure_scratch_reg(
                                        vreg,
                                        ty,
                                        &mut scratch_bindings,
                                        &mut used_gpr_scratch,
                                        &mut used_fpr_scratch,
                                    );
                                    *operand = MachineOperand::Def(Writable(scratch));
                                    after.push((slot, scratch, ty));
                                }
                            }
                            MachineOperand::TiedDefUse(w) if w.to_reg().is_vreg() => {
                                let vreg = w.to_reg();
                                if let Some(&preg) = self.allocation.get(&vreg) {
                                    *operand = MachineOperand::TiedDefUse(Writable(preg));
                                } else if let Some(&slot) = self.spilled.get(&vreg) {
                                    let ty = mfunc.vreg_data(vreg).ty;
                                    let scratch = self.ensure_scratch_reg(
                                        vreg,
                                        ty,
                                        &mut scratch_bindings,
                                        &mut used_gpr_scratch,
                                        &mut used_fpr_scratch,
                                    );
                                    if !before.iter().any(|(loaded_slot, reg, _)| {
                                        *loaded_slot == slot && *reg == scratch
                                    }) {
                                        before.push((slot, scratch, ty));
                                    }
                                    *operand = MachineOperand::TiedDefUse(Writable(scratch));
                                    after.push((slot, scratch, ty));
                                }
                            }
                            _ => {}
                        }
                    }

                    for (slot, scratch, ty) in before {
                        output.push(
                            mfunc.alloc_inst(self.build_stack_load(mfunc, slot, scratch, ty)),
                        );
                    }
                    mfunc.replace_inst(inst_id, inst);
                    output.push(inst_id);
                    for (slot, scratch, ty) in after {
                        output.push(
                            mfunc.alloc_inst(self.build_stack_store(mfunc, slot, scratch, ty)),
                        );
                    }

                    Ok::<(), ()>(())
                })
                .expect("register allocation rewrite should not fail");
        }
    }

    fn ensure_scratch_reg(
        &self,
        vreg: Reg,
        ty: Type,
        bindings: &mut Vec<(Reg, Reg)>,
        used_gpr_scratch: &mut usize,
        used_fpr_scratch: &mut usize,
    ) -> Reg {
        if let Some((_, scratch)) = bindings.iter().find(|(bound_vreg, _)| *bound_vreg == vreg) {
            return *scratch;
        }

        let scratch = match self.target.desc().reg_class_for_type(&ty) {
            RegClass::GPR => {
                let scratch = match *used_gpr_scratch {
                    0 => Reg::new_preg(10),
                    1 => Reg::new_preg(11),
                    _ => panic!("ran out of GPR scratch registers while expanding spills"),
                };
                *used_gpr_scratch += 1;
                scratch
            }
            RegClass::FPR => {
                let scratch = match *used_fpr_scratch {
                    0 => Reg::new_preg(30),
                    1 => Reg::new_preg(31),
                    _ => panic!("ran out of FPR scratch registers while expanding spills"),
                };
                *used_fpr_scratch += 1;
                scratch
            }
            other => panic!(
                "spill expansion does not support register class {:?}",
                other
            ),
        };

        bindings.push((vreg, scratch));
        scratch
    }

    fn build_stack_load(
        &self,
        mfunc: &MachineFunction,
        slot: StackSlot,
        dst: Reg,
        ty: Type,
    ) -> MachineInst {
        let offset = mfunc.stack_frame.slots[slot].offset as i64;
        let rbp = Reg::new_preg(5);
        let opcode = self.stack_load_opcode(ty);
        MachineInst::build_generic(
            MachineOpcode::Target(opcode),
            smallvec::smallvec![
                MachineOperand::Def(Writable(dst)),
                MachineOperand::Use(rbp),
                MachineOperand::Imm(offset),
            ],
        )
    }

    fn build_stack_store(
        &self,
        mfunc: &MachineFunction,
        slot: StackSlot,
        src: Reg,
        ty: Type,
    ) -> MachineInst {
        let offset = mfunc.stack_frame.slots[slot].offset as i64;
        let rbp = Reg::new_preg(5);
        let opcode = self.stack_store_opcode(ty);
        MachineInst::build_generic(
            MachineOpcode::Target(opcode),
            smallvec::smallvec![
                MachineOperand::Use(src),
                MachineOperand::Use(rbp),
                MachineOperand::Imm(offset),
            ],
        )
    }

    fn stack_load_opcode(&self, ty: Type) -> u32 {
        use crate::target::arch::TargetArch;
        use crate::target::x86_64::isle::TargetInst;

        match self.target.desc().arch {
            TargetArch::X86_64 => match ty {
                Type::F32 => TargetInst::X86LoadF32.as_u32(),
                Type::F64 => TargetInst::X86LoadF64.as_u32(),
                _ if ty == Type::PTR || ty.size_bytes() >= 8 => TargetInst::X86Load64.as_u32(),
                _ if ty.size_bytes() >= 4 => TargetInst::X86Load32.as_u32(),
                _ if ty.size_bytes() >= 2 => TargetInst::X86Load16U32.as_u32(),
                _ => TargetInst::X86Load8U32.as_u32(),
            },
            other => panic!(
                "spill expansion does not support target architecture {:?}",
                other
            ),
        }
    }

    fn stack_store_opcode(&self, ty: Type) -> u32 {
        use crate::target::arch::TargetArch;
        use crate::target::x86_64::isle::TargetInst;

        match self.target.desc().arch {
            TargetArch::X86_64 => match ty {
                Type::F32 => TargetInst::X86StoreF32.as_u32(),
                Type::F64 => TargetInst::X86StoreF64.as_u32(),
                _ if ty == Type::PTR || ty.size_bytes() >= 8 => TargetInst::X86Store64.as_u32(),
                _ if ty.size_bytes() >= 4 => TargetInst::X86Store32.as_u32(),
                _ if ty.size_bytes() >= 2 => TargetInst::X86Store16.as_u32(),
                _ => TargetInst::X86Store8.as_u32(),
            },
            other => panic!(
                "spill expansion does not support target architecture {:?}",
                other
            ),
        }
    }

    /// 获取分配结果
    pub fn get_allocation(&self, vreg: Reg) -> Option<Reg> {
        self.allocation.get(&vreg).copied()
    }

    /// 获取栈槽
    pub fn get_stack_slot(&self, vreg: Reg) -> Option<StackSlot> {
        self.spilled.get(&vreg).copied()
    }
}
