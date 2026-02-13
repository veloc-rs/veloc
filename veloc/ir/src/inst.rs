use super::types::{BlockCall, FuncId, JumpTable, StackSlot, Type, Value, ValueList};
use crate::SigId;
use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntCC {
    Eq,
    Ne,
    LtS,
    LtU,
    GtS,
    GtU,
    LeS,
    LeU,
    GeS,
    GeU,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatCC {
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

impl fmt::Display for IntCC {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            IntCC::Eq => "eq",
            IntCC::Ne => "ne",
            IntCC::LtS => "lts",
            IntCC::LtU => "ltu",
            IntCC::GtS => "gts",
            IntCC::GtU => "gtu",
            IntCC::LeS => "les",
            IntCC::LeU => "leu",
            IntCC::GeS => "ges",
            IntCC::GeU => "geu",
        };
        write!(f, "{}", s)
    }
}

impl fmt::Display for FloatCC {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            FloatCC::Eq => "eq",
            FloatCC::Ne => "ne",
            FloatCC::Lt => "lt",
            FloatCC::Gt => "gt",
            FloatCC::Le => "le",
            FloatCC::Ge => "ge",
        };
        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    Iconst,
    Fconst,
    Bconst,
    // Arithmetic (Conflicting)
    Iadd,
    Isub,
    Imul,
    Ineg,
    Fadd,
    Fsub,
    Fmul,
    Fneg,
    // Integer-only Arithmetic
    Idiv,
    Udiv,
    Irem,
    Urem,
    // Bitwise (Integer-only)
    And,
    Or,
    Xor,
    Shl,
    ShrS,
    ShrU,
    Rotl,
    Rotr,
    // Integer Unary
    Clz,
    Ctz,
    Popcnt,
    Eqz,
    // Float-only Arithmetic/Unary
    Fdiv,
    Min,
    Max,
    Copysign,
    Abs,
    Sqrt,
    Ceil,
    Floor,
    Trunc,
    Nearest,
    // Comparisons (Keep prefixes for clarity or consistency)
    Icmp,
    Fcmp,
    // Conversions
    ExtendS,
    ExtendU,
    Wrap,
    TruncS,
    TruncU,
    ConvertS,
    ConvertU,
    Promote,
    Demote,
    Reinterpret,
    IntToPtr,
    PtrToInt,
    // Memory
    Load,
    Store,
    StackLoad,
    StackStore,
    StackAddr,
    Gep,
    // Control Flow
    Call,
    CallIndirect,
    Jump,
    Br,
    BrTable,
    Return,
    Select,
    Unreachable,
}

impl fmt::Display for Opcode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Opcode::Iconst => "iconst",
            Opcode::Fconst => "fconst",
            Opcode::Bconst => "bconst",
            Opcode::Iadd => "iadd",
            Opcode::Isub => "isub",
            Opcode::Imul => "imul",
            Opcode::Ineg => "ineg",
            Opcode::Fadd => "fadd",
            Opcode::Fsub => "fsub",
            Opcode::Fmul => "fmul",
            Opcode::Fneg => "fneg",
            Opcode::Idiv => "idiv",
            Opcode::Udiv => "udiv",
            Opcode::Irem => "irem",
            Opcode::Urem => "urem",
            Opcode::And => "and",
            Opcode::Or => "or",
            Opcode::Xor => "xor",
            Opcode::Shl => "shl",
            Opcode::ShrS => "shrs",
            Opcode::ShrU => "shru",
            Opcode::Rotl => "rotl",
            Opcode::Rotr => "rotr",
            Opcode::Clz => "clz",
            Opcode::Ctz => "ctz",
            Opcode::Popcnt => "popcnt",
            Opcode::Eqz => "eqz",
            Opcode::Fdiv => "fdiv",
            Opcode::Min => "min",
            Opcode::Max => "max",
            Opcode::Copysign => "copysign",
            Opcode::Abs => "abs",
            Opcode::Sqrt => "sqrt",
            Opcode::Ceil => "ceil",
            Opcode::Floor => "floor",
            Opcode::Trunc => "trunc",
            Opcode::Nearest => "nearest",
            Opcode::Icmp => "icmp",
            Opcode::Fcmp => "fcmp",
            Opcode::ExtendS => "extends",
            Opcode::ExtendU => "extendu",
            Opcode::Wrap => "wrap",
            Opcode::TruncS => "truncs",
            Opcode::TruncU => "truncu",
            Opcode::ConvertS => "converts",
            Opcode::ConvertU => "convertu",
            Opcode::Promote => "promote",
            Opcode::Demote => "demote",
            Opcode::Reinterpret => "reinterpret",
            Opcode::IntToPtr => "inttoptr",
            Opcode::PtrToInt => "ptrtoint",
            Opcode::Load => "load",
            Opcode::Store => "store",
            Opcode::StackLoad => "stack_load",
            Opcode::StackStore => "stack_store",
            Opcode::StackAddr => "stack_addr",
            Opcode::Gep => "gep",
            Opcode::Call => "call",
            Opcode::CallIndirect => "call_indirect",
            Opcode::Jump => "jump",
            Opcode::Br => "br",
            Opcode::BrTable => "br_table",
            Opcode::Return => "return",
            Opcode::Select => "select",
            Opcode::Unreachable => "unreachable",
        };
        write!(f, "{}", s)
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
    },
    Store {
        ptr: Value,
        value: Value,
        offset: u32,
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
    Gep {
        ptr: Value,
        offset: Value,
    },
}

impl InstructionData {
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
            InstructionData::Gep { .. } => Opcode::Gep,
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
            InstructionData::Gep { .. } => Type::Ptr,
        }
    }
}

impl fmt::Display for InstructionData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.opcode())
    }
}
