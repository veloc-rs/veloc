use cranelift_entity::SecondaryMap;
use veloc_ir::{Function, Inst, Value};

/// 侧表分析：Use-Def (Def-Use) 链分析。
/// 记录了每个 Value 被哪些指令使用。
#[derive(Debug, Clone, Default)]
pub struct UseDefAnalysis {
    /// 映射: Value -> 使用该 Value 的指令列表
    users: SecondaryMap<Value, Vec<Inst>>,
}

impl UseDefAnalysis {
    /// 全量计算 Use-Def 分析。
    pub fn new(func: &Function) -> Self {
        let mut users: SecondaryMap<Value, Vec<Inst>> = SecondaryMap::new();

        for block in &func.layout.block_order {
            for &inst in &func.layout.blocks[*block].insts {
                inst.visit_operands(&func.dfg, |v| {
                    users[v].push(inst);
                });
            }
        }

        Self { users }
    }

    /// 获取使用该 Value 的所有指令。
    pub fn users_of(&self, val: Value) -> &[Inst] {
        &self.users[val]
    }

    /// 增量添加用户。
    pub fn add_user(&mut self, val: Value, user: Inst) {
        self.users[val].push(user);
    }

    /// 增量移除用户。
    pub fn remove_user(&mut self, val: Value, user: Inst) {
        let list = &mut self.users[val];
        if let Some(pos) = list.iter().position(|&i| i == user) {
            list.swap_remove(pos);
        }
    }

    /// 当指令从 DFG 中移除或其操作数发生变化前，调用此方法来从侧表中同步清理旧的操作数引用。
    pub fn detach_inst(&mut self, func: &Function, inst: Inst) {
        inst.visit_operands(&func.dfg, |v| {
            self.remove_user(v, inst);
        });
    }

    /// 清空所有分析数据。
    pub fn clear(&mut self) {
        self.users.clear();
    }
}
