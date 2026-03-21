//! 机器模块定义
//!
//! MachineModule 是 MIR 层级的顶级容器，包含模块级共享资源、符号表和函数集合。

use super::{MachineFunction, SymbolTable};
use alloc::vec::Vec;
use cranelift_entity::PrimaryMap;
use cranelift_entity::entity_impl;

/// 机器函数标识符
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MachineFuncId(u32);
entity_impl!(MachineFuncId, "mfunc");

/// 机器模块
pub struct MachineModule {
    /// 模块名称
    pub name: alloc::string::String,
    /// 模块级符号表（跨函数共享）
    pub symbols: SymbolTable,
    /// 包含的机器函数
    pub functions: PrimaryMap<MachineFuncId, MachineFunction>,
    /// 函数顺序（保持与 IR 一致或优化后的顺序）
    pub func_order: Vec<MachineFuncId>,
    /// 模块级全局常量/数据（可选扩展）
    pub data_sections: Vec<()>,
}

impl MachineModule {
    pub fn new(name: alloc::string::String) -> Self {
        Self {
            name,
            symbols: SymbolTable::new(),
            functions: PrimaryMap::new(),
            func_order: Vec::new(),
            data_sections: Vec::new(),
        }
    }

    /// 添加一个机器函数
    pub fn add_function(&mut self, func: MachineFunction) -> MachineFuncId {
        let id = self.functions.push(func);
        self.func_order.push(id);
        id
    }

    /// 获取符号表
    pub fn symbols(&self) -> &SymbolTable {
        &self.symbols
    }

    /// 获取符号表（可变）
    pub fn symbols_mut(&mut self) -> &mut SymbolTable {
        &mut self.symbols
    }

    /// 通过名称查找机器函数
    pub fn find_function_by_name(&self, name: &str) -> Option<MachineFuncId> {
        self.functions
            .iter()
            .find(|(_, f)| f.name == name)
            .map(|(id, _)| id)
    }
}
