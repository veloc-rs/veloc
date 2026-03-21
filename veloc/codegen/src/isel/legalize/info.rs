extern crate alloc;
use crate::mir::GenericOpcode;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use veloc_ir::Type;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegalizeAction {
    Legal,         // 无需操作
    Unsupported,   // 不支持
    NarrowScalar,  // 缩小标量宽度
    WidenScalar,   // 扩宽标量宽度
    Lower,         // 使用较为简单的操作指令替代
    Libcall,       // 转换成运行时调用
    FewerElements, // 减少矢量指令宽度
    MoreElements,  // 扩宽矢量指令宽度
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LegalityKey {
    pub opcode: GenericOpcode,
    pub types: Vec<Type>, // 操作数类型，通常只有前两个（如 G_ADD i32 i32）
}

pub struct LegalizerInfo {
    rules: BTreeMap<LegalityKey, LegalizeAction>,
}

pub struct LegalizeRuleSet<'a> {
    info: &'a mut LegalizerInfo,
    opcode: GenericOpcode,
}

impl<'a> LegalizeRuleSet<'a> {
    pub fn new(info: &'a mut LegalizerInfo, opcode: GenericOpcode) -> Self {
        Self { info, opcode }
    }

    /// 标记特定类型组合为合法
    pub fn legal_for(mut self, types: Vec<Type>) -> Self {
        self.info
            .set_action(self.opcode.clone(), types, LegalizeAction::Legal);
        self
    }

    /// 标记多个类型组合为合法
    pub fn legal_for_many(mut self, types_list: Vec<Vec<Type>>) -> Self {
        for types in types_list {
            self.info
                .set_action(self.opcode.clone(), types, LegalizeAction::Legal);
        }
        self
    }

    /// 标记需要 WidenScalar 的类型
    pub fn widen_scalar_for(mut self, types: Vec<Type>) -> Self {
        self.info
            .set_action(self.opcode.clone(), types, LegalizeAction::WidenScalar);
        self
    }

    /// 标记需要 Lower 的类型
    pub fn lower_for(mut self, types: Vec<Type>) -> Self {
        self.info
            .set_action(self.opcode.clone(), types, LegalizeAction::Lower);
        self
    }

    /// 标记不支持的类型（显式标记）
    pub fn unsupported_for(mut self, types: Vec<Type>) -> Self {
        self.info
            .set_action(self.opcode.clone(), types, LegalizeAction::Unsupported);
        self
    }

    // ==================== 便捷方法 ====================

    /// 标记多个类型为合法（每种类型作为独立操作数）
    pub fn legal_for_types(mut self, types: &[Type]) -> Self {
        for ty in types {
            self.info
                .set_action(self.opcode.clone(), vec![*ty; 3], LegalizeAction::Legal);
        }
        self
    }

    /// 标记需要扩展的类型
    pub fn widen_scalar_for_type(mut self, ty: Type) -> Self {
        self.info.set_action(
            self.opcode.clone(),
            vec![ty; 3],
            LegalizeAction::WidenScalar,
        );
        self
    }

    /// 标记需要 Lower 的类型
    pub fn lower_for_type(mut self, ty: Type) -> Self {
        self.info
            .set_action(self.opcode.clone(), vec![ty; 3], LegalizeAction::Lower);
        self
    }
}

impl LegalizerInfo {
    pub fn new() -> Self {
        Self {
            rules: BTreeMap::new(),
        }
    }

    /// 获取特定指令的操作集构建器 (类似 LLVM getActionDefinitionsBuilder)
    pub fn get_action_definitions_builder(&mut self, opcode: GenericOpcode) -> LegalizeRuleSet {
        LegalizeRuleSet::new(self, opcode)
    }

    pub fn set_action(&mut self, opcode: GenericOpcode, types: Vec<Type>, action: LegalizeAction) {
        self.rules.insert(LegalityKey { opcode, types }, action);
    }

    pub fn get_action(&self, opcode: &GenericOpcode, types: &[Type]) -> LegalizeAction {
        self.rules
            .get(&LegalityKey {
                opcode: opcode.clone(),
                types: types.to_vec(),
            })
            .copied()
            .unwrap_or(LegalizeAction::Unsupported)
    }
}
