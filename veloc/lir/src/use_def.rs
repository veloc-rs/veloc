//! LIR 层的 Use-Def 链分析与增量更新实现

use crate::instr::{InstId, MachineInst, Reg};
use hashbrown::HashMap;
use smallvec::SmallVec;

/// LIR 层的 Use-Def 链信息。
///
/// 该结构体用于在前向指令选择、寄存器分配等各个 LIR 变换阶段，
/// 维护寄存器 (Reg) 与其定义指令 (Def) 或是使用指令 (Use) 之间的双向映射关系。
#[derive(Debug, Clone, Default)]
pub struct UseDefChain {
    /// 映射: 寄存器 -> 使用该寄存器的所有指令列表
    uses: HashMap<Reg, SmallVec<[InstId; 4]>>,
    /// 映射: 寄存器 -> 定义该寄存器的所有指令列表
    defs: HashMap<Reg, SmallVec<[InstId; 2]>>,
}

impl UseDefChain {
    /// 清空所有记录。
    pub fn clear(&mut self) {
        self.uses.clear();
        self.defs.clear();
    }

    /// 向链中增量添加一条指令及其寄存器引用关系。
    pub fn add_inst(&mut self, inst_id: InstId, inst: &MachineInst) {
        if inst.is_invalid() {
            return;
        }
        for def in inst.defs() {
            self.defs.entry(def).or_default().push(inst_id);
        }
        for use_reg in inst.uses() {
            self.uses.entry(use_reg).or_default().push(inst_id);
        }
    }

    /// 从链中移除某条指令与其寄存器的引用关系。
    pub fn remove_inst(&mut self, inst_id: InstId, inst: &MachineInst) {
        if inst.is_invalid() {
            return;
        }
        for def in inst.defs() {
            Self::remove_from_def_map(&mut self.defs, def, inst_id);
        }
        for use_reg in inst.uses() {
            Self::remove_from_use_map(&mut self.uses, use_reg, inst_id);
        }
    }

    /// 获取定义了某寄存器的所有指令列表。
    pub fn defs_of(&self, reg: Reg) -> &[InstId] {
        self.defs.get(&reg).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// 获取使用了某寄存器的所有指令列表 (用户列表)。
    pub fn users_of(&self, reg: Reg) -> &[InstId] {
        self.uses.get(&reg).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// 获取某寄存器的定义指令数量。
    #[inline]
    pub fn def_count(&self, reg: Reg) -> usize {
        self.defs_of(reg).len()
    }

    /// 获取某寄存器的使用指令数量。
    #[inline]
    pub fn use_count(&self, reg: Reg) -> usize {
        self.users_of(reg).len()
    }

    /// 如果某寄存器有且只有一个定义位置，则返回该指令 ID。
    #[inline]
    pub fn single_def_of(&self, reg: Reg) -> Option<InstId> {
        let defs = self.defs_of(reg);
        if defs.len() == 1 { Some(defs[0]) } else { None }
    }

    /// 如果某寄存器有且只有一个用户位置，则返回该指令 ID。
    #[inline]
    pub fn single_user_of(&self, reg: Reg) -> Option<InstId> {
        let users = self.users_of(reg);
        if users.len() == 1 {
            Some(users[0])
        } else {
            None
        }
    }

    /// 检查某寄存器是否仅被一条特定的指令使用。
    #[inline]
    pub fn is_single_use_by(&self, reg: Reg, inst_id: InstId) -> bool {
        self.single_user_of(reg) == Some(inst_id)
    }

    fn remove_from_def_map(
        map: &mut HashMap<Reg, SmallVec<[InstId; 2]>>,
        reg: Reg,
        inst_id: InstId,
    ) {
        if let Some(list) = map.get_mut(&reg) {
            if let Some(pos) = list.iter().position(|id| *id == inst_id) {
                list.swap_remove(pos);
            }
            if list.is_empty() {
                map.remove(&reg);
            }
        }
    }

    fn remove_from_use_map(
        map: &mut HashMap<Reg, SmallVec<[InstId; 4]>>,
        reg: Reg,
        inst_id: InstId,
    ) {
        if let Some(list) = map.get_mut(&reg) {
            if let Some(pos) = list.iter().position(|id| *id == inst_id) {
                list.swap_remove(pos);
            }
            if list.is_empty() {
                map.remove(&reg);
            }
        }
    }
}
