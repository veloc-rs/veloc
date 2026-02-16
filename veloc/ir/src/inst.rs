use super::dfg::DataFlowGraph;
use super::types::{BlockCall, FuncId, JumpTable, StackSlot, Type, Value, ValueList};
use crate::{FloatCC, IntCC, MemFlags, Opcode, SigId};
use core::fmt;
use cranelift_entity::entity_impl;

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
    Unary {
        opcode: Opcode,
        arg: Value,
        ty: Type,
    },
    Binary {
        opcode: Opcode,
        args: [Value; 2],
        ty: Type,
    },
    Load {
        ptr: Value,
        offset: u32,
        ty: Type,
        flags: MemFlags,
    },
    Store {
        ptr: Value,
        value: Value,
        offset: u32,
        flags: MemFlags,
    },
    StackLoad {
        slot: StackSlot,
        offset: u32,
        ty: Type,
    },
    StackStore {
        slot: StackSlot,
        value: Value,
        offset: u32,
    },
    StackAddr {
        slot: StackSlot,
        offset: u32,
    },
    Iconst {
        value: i64,
        ty: Type,
    },
    Fconst {
        value: u64,
        ty: Type,
    },
    Bconst {
        value: bool,
    },
    Call {
        func_id: FuncId,
        args: ValueList,
        ret_ty: Type,
    },
    Jump {
        dest: BlockCall,
    },
    Br {
        condition: Value,
        then_dest: BlockCall,
        else_dest: BlockCall,
    },
    BrTable {
        index: Value,
        table: JumpTable,
    },
    Return {
        value: Option<Value>,
    },
    Select {
        condition: Value,
        then_val: Value,
        else_val: Value,
        ty: Type,
    },
    IntCompare {
        kind: IntCC,
        args: [Value; 2],
        ty: Type,
    },
    FloatCompare {
        kind: FloatCC,
        args: [Value; 2],
        ty: Type,
    },
    Unreachable,
    CallIndirect {
        ptr: Value,
        args: ValueList,
        sig_id: SigId,
        ret_ty: Type,
    },
    IntToPtr {
        arg: Value,
    },
    PtrToInt {
        arg: Value,
        ty: Type,
    },
    PtrOffset {
        ptr: Value,
        offset: i32,
    },
    PtrIndex {
        ptr: Value,
        index: Value,
        scale: u32,
        offset: i32,
    },
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
            InstructionData::Nop => Opcode::Nop,
        }
    }

    pub fn is_terminator(&self) -> bool {
        matches!(
            self.opcode(),
            Opcode::Jump | Opcode::Br | Opcode::BrTable | Opcode::Return | Opcode::Unreachable
        )
    }

    pub fn result_type(&self) -> Type {
        match self {
            InstructionData::Unary { ty, .. } => *ty,
            InstructionData::Binary { ty, .. } => *ty,
            InstructionData::Load { ty, .. } => *ty,
            InstructionData::Store { .. } => Type::Void,
            InstructionData::StackLoad { ty, .. } => *ty,
            InstructionData::StackStore { .. } => Type::Void,
            InstructionData::StackAddr { .. } => Type::Ptr,
            InstructionData::Iconst { ty, .. } => *ty,
            InstructionData::Fconst { ty, .. } => *ty,
            InstructionData::Bconst { .. } => Type::Bool,
            InstructionData::Call { ret_ty, .. } => *ret_ty,
            InstructionData::Jump { .. } => Type::Void,
            InstructionData::Br { .. } => Type::Void,
            InstructionData::BrTable { .. } => Type::Void,
            InstructionData::Return { .. } => Type::Void,
            InstructionData::Select { ty, .. } => *ty,
            InstructionData::IntCompare { ty, .. } => *ty,
            InstructionData::FloatCompare { ty, .. } => *ty,
            InstructionData::Unreachable => Type::Void,
            InstructionData::CallIndirect { ret_ty, .. } => *ret_ty,
            InstructionData::IntToPtr { .. } => Type::Ptr,
            InstructionData::PtrToInt { ty, .. } => *ty,
            InstructionData::PtrOffset { .. } => Type::Ptr,
            InstructionData::PtrIndex { .. } => Type::Ptr,
            InstructionData::Nop => Type::Void,
        }
    }

    pub fn has_side_effects(&self) -> bool {
        match self.opcode() {
            Opcode::Store
            | Opcode::StackStore
            | Opcode::Call
            | Opcode::CallIndirect
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
