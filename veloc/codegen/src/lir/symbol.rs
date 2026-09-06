//! LIR 中的符号定义与管理

use alloc::string::{String, ToString};
use cranelift_entity::PrimaryMap;
use cranelift_entity::entity_impl;
use veloc_ir::Linkage;

/// 符号标识符
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymbolId(u32);
entity_impl!(SymbolId, "symbol");

/// 符号类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    Function,
    GlobalVariable,
}

/// 符号可见性
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Visibility {
    Default,
    Hidden,
    Protected,
}

/// 符号定义
#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub linkage: Linkage,
    pub visibility: Visibility,
}

/// 符号名存储方案
#[derive(Debug, Clone)]
pub struct SymbolTable {
    pub symbols: PrimaryMap<SymbolId, Symbol>,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self {
            symbols: PrimaryMap::new(),
        }
    }

    /// 从 IR 函数创建符号，并保留 Linkage 信息
    pub fn get_or_create_func(
        &mut self,
        func_id: veloc_ir::FuncId,
        module: &veloc_ir::Module,
    ) -> SymbolId {
        let name = module.get_function_name(func_id);
        if let Some((id, _)) = self
            .symbols
            .iter()
            .find(|(_, sym)| sym.name == name && sym.kind == SymbolKind::Function)
        {
            return id;
        }

        let func = module.get_function(func_id);
        self.symbols.push(Symbol {
            name: name.to_string(),
            kind: SymbolKind::Function,
            linkage: func.linkage,
            visibility: Visibility::Default,
        })
    }

    /// 获取符号详情
    pub fn get(&self, id: SymbolId) -> &Symbol {
        &self.symbols[id]
    }

    /// 获取符号详情（可变）
    pub fn get_mut(&mut self, id: SymbolId) -> &mut Symbol {
        &mut self.symbols[id]
    }
}
