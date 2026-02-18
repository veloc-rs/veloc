use super::dfg::DataFlowGraph;
use crate::types::{BlockCall, FuncId, JumpTable, StackSlot, Value, ValueList};
use crate::{FloatCC, IntCC, Intrinsic, MemFlags, Opcode, SigId};
use core::fmt;
use cranelift_entity::entity_impl;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PtrIndexImm {
    pub offset: i32,
    pub scale: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PtrIndexImmId(pub u32);
entity_impl!(PtrIndexImmId, "ptr_index_imm");

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Inst(pub u32);
entity_impl!(Inst, "inst");

impl Inst {
    pub fn visit_operands<F>(self, dfg: &DataFlowGraph, f: F)
    where
        F: FnMut(Value),
    {
        dfg.instructions[self].visit_operands(dfg, f)
    }
}

#[derive(Debug, Clone)]
pub enum InstructionData {
    /// 一元运算
    Unary { opcode: Opcode, arg: Value },
    /// 二元运算
    Binary { opcode: Opcode, args: [Value; 2] },
    /// 从内存加载
    Load {
        ptr: Value,
        offset: u32,
        flags: MemFlags,
    },
    /// 存储到内存
    Store {
        ptr: Value,
        value: Value,
        offset: u32,
        flags: MemFlags,
    },
    /// 从栈槽加载
    StackLoad { slot: StackSlot, offset: u32 },
    /// 存储到栈槽
    StackStore {
        slot: StackSlot,
        value: Value,
        offset: u32,
    },
    /// 获取栈槽地址
    StackAddr { slot: StackSlot, offset: u32 },
    /// 整数常量
    Iconst { value: i64 },
    /// 浮点常量
    Fconst { value: u64 },
    /// 布尔常量
    Bconst { value: bool },
    /// 直接函数调用
    Call { func_id: FuncId, args: ValueList },
    /// 无条件跳转
    Jump { dest: BlockCall },
    /// 条件分支
    Br {
        condition: Value,
        then_dest: BlockCall,
        else_dest: BlockCall,
    },
    /// 跳转表
    BrTable { index: Value, table: JumpTable },
    /// 函数返回
    Return { value: Option<Value> },
    /// 条件选择
    Select {
        condition: Value,
        then_val: Value,
        else_val: Value,
    },
    /// 整数比较
    IntCompare { kind: IntCC, args: [Value; 2] },
    /// 浮点比较
    FloatCompare { kind: FloatCC, args: [Value; 2] },
    /// 不可达代码
    Unreachable,
    /// 间接函数调用
    CallIndirect {
        ptr: Value,
        args: ValueList,
        sig_id: SigId,
    },
    /// 整数转指针
    IntToPtr { arg: Value },
    /// 指针转整数
    PtrToInt { arg: Value },
    /// 指针偏移
    PtrOffset { ptr: Value, offset: i32 },
    /// 带有立即数的指针索引
    PtrIndex {
        ptr: Value,
        index: Value,
        imm_id: PtrIndexImmId,
    },
    /// 内建函数调用
    CallIntrinsic {
        intrinsic: Intrinsic,
        args: ValueList,
        sig_id: SigId,
    },
    /// 从多值中提取单个值
    ExtractValue { val: Value, index: u32 },
    /// 构造多返回值
    ConstructMulti { values: ValueList },
    /// 空操作
    Nop,
}

impl InstructionData {
    pub fn visit_operands<F>(&self, dfg: &DataFlowGraph, mut f: F)
    where
        F: FnMut(Value),
    {
        match self {
            InstructionData::Unary { arg, .. } => f(*arg),
            InstructionData::Binary { args, .. } => {
                f(args[0]);
                f(args[1]);
            }
            InstructionData::Load { ptr, .. } => f(*ptr),
            InstructionData::Store { ptr, value, .. } => {
                f(*ptr);
                f(*value);
            }
            InstructionData::StackLoad { .. } => {}
            InstructionData::StackStore { value, .. } => f(*value),
            InstructionData::StackAddr { .. } => {}
            InstructionData::Iconst { .. } => {}
            InstructionData::Fconst { .. } => {}
            InstructionData::Bconst { .. } => {}
            InstructionData::Call { args, .. } => {
                for &arg in dfg.get_value_list(*args) {
                    f(arg);
                }
            }
            InstructionData::CallIndirect { ptr, args, .. } => {
                f(*ptr);
                for &arg in dfg.get_value_list(*args) {
                    f(arg);
                }
            }
            InstructionData::Jump { dest } => {
                let call_data = &dfg.block_calls[*dest];
                for &arg in dfg.get_value_list(call_data.args) {
                    f(arg);
                }
            }
            InstructionData::Br {
                condition,
                then_dest,
                else_dest,
            } => {
                f(*condition);
                let then_data = &dfg.block_calls[*then_dest];
                for &arg in dfg.get_value_list(then_data.args) {
                    f(arg);
                }
                let else_data = &dfg.block_calls[*else_dest];
                for &arg in dfg.get_value_list(else_data.args) {
                    f(arg);
                }
            }
            InstructionData::BrTable { index, table } => {
                f(*index);
                let table_data = &dfg.jump_tables[*table];
                for &dest in table_data.targets.iter() {
                    let call_data = &dfg.block_calls[dest];
                    for &arg in dfg.get_value_list(call_data.args) {
                        f(arg);
                    }
                }
            }
            InstructionData::Return { value } => {
                if let Some(v) = value {
                    f(*v);
                }
            }
            InstructionData::Select {
                condition,
                then_val,
                else_val,
                ..
            } => {
                f(*condition);
                f(*then_val);
                f(*else_val);
            }
            InstructionData::IntCompare { args, .. } => {
                f(args[0]);
                f(args[1]);
            }
            InstructionData::FloatCompare { args, .. } => {
                f(args[0]);
                f(args[1]);
            }
            InstructionData::Unreachable => {}
            InstructionData::IntToPtr { arg } => f(*arg),
            InstructionData::PtrToInt { arg, .. } => f(*arg),
            InstructionData::PtrOffset { ptr, .. } => f(*ptr),
            InstructionData::PtrIndex { ptr, index, .. } => {
                f(*ptr);
                f(*index);
            }
            InstructionData::CallIntrinsic { args, .. } => {
                for &arg in dfg.get_value_list(*args) {
                    f(arg);
                }
            }
            InstructionData::ExtractValue { val, .. } => {
                f(*val);
            }
            InstructionData::ConstructMulti { values } => {
                for &arg in dfg.get_value_list(*values) {
                    f(arg);
                }
            }
            InstructionData::Nop => {}
        }
    }

    pub fn opcode(&self) -> Opcode {
        match self {
            InstructionData::Unary { opcode, .. } => *opcode,
            InstructionData::Binary { opcode, .. } => *opcode,
            InstructionData::Load { .. } => Opcode::Load,
            InstructionData::Store { .. } => Opcode::Store,
            InstructionData::StackLoad { .. } => Opcode::StackLoad,
            InstructionData::StackStore { .. } => Opcode::StackStore,
            InstructionData::StackAddr { .. } => Opcode::StackAddr,
            InstructionData::Iconst { .. } => Opcode::Iconst,
            InstructionData::Fconst { .. } => Opcode::Fconst,
            InstructionData::Bconst { .. } => Opcode::Bconst,
            InstructionData::Call { .. } => Opcode::Call,
            InstructionData::Jump { .. } => Opcode::Jump,
            InstructionData::Br { .. } => Opcode::Br,
            InstructionData::BrTable { .. } => Opcode::BrTable,
            InstructionData::Return { .. } => Opcode::Return,
            InstructionData::Select { .. } => Opcode::Select,
            InstructionData::IntCompare { .. } => Opcode::Icmp,
            InstructionData::FloatCompare { .. } => Opcode::Fcmp,
            InstructionData::Unreachable => Opcode::Unreachable,
            InstructionData::CallIndirect { .. } => Opcode::CallIndirect,
            InstructionData::IntToPtr { .. } => Opcode::IntToPtr,
            InstructionData::PtrToInt { .. } => Opcode::PtrToInt,
            InstructionData::PtrOffset { .. } => Opcode::PtrOffset,
            InstructionData::PtrIndex { .. } => Opcode::PtrIndex,
            InstructionData::CallIntrinsic { .. } => Opcode::CallIntrinsic,
            InstructionData::ExtractValue { .. } => Opcode::ExtractValue,
            InstructionData::ConstructMulti { .. } => Opcode::ConstructMulti,
            InstructionData::Nop => Opcode::Nop,
        }
    }

    pub fn is_terminator(&self) -> bool {
        matches!(
            self.opcode(),
            Opcode::Jump | Opcode::Br | Opcode::BrTable | Opcode::Return | Opcode::Unreachable
        )
    }

    pub fn has_side_effects(&self) -> bool {
        match self.opcode() {
            Opcode::Store
            | Opcode::StackStore
            | Opcode::Call
            | Opcode::CallIndirect
            | Opcode::CallIntrinsic
            | Opcode::Return
            | Opcode::Jump
            | Opcode::Br
            | Opcode::BrTable
            | Opcode::Unreachable => true,
            _ => false,
        }
    }
}

impl fmt::Display for InstructionData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.opcode())
    }
}
