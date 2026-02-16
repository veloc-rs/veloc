use bitflags::bitflags;
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
    DivS,
    DivU,
    RemS,
    RemU,
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
    PtrOffset,
    PtrIndex,
    // Control Flow
    Call,
    CallIndirect,
    Jump,
    Br,
    BrTable,
    Return,
    Select,
    Unreachable,
    Nop,
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
            Opcode::DivS => "idiv",
            Opcode::DivU => "udiv",
            Opcode::RemS => "irem",
            Opcode::RemU => "urem",
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
            Opcode::PtrOffset => "ptr_offset",
            Opcode::PtrIndex => "ptr_index",
            Opcode::Call => "call",
            Opcode::CallIndirect => "call_indirect",
            Opcode::Jump => "jump",
            Opcode::Br => "br",
            Opcode::BrTable => "br_table",
            Opcode::Return => "return",
            Opcode::Select => "select",
            Opcode::Unreachable => "unreachable",
            Opcode::Nop => "nop",
        };
        write!(f, "{}", s)
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct MemFlags: u16 {
        const TRUSTED = 1 << 0;
        const ALIGN_MASK = 0b1111 << 1;
    }
}

impl MemFlags {
    pub const fn new() -> Self {
        Self::empty()
    }

    pub const fn trusted() -> Self {
        Self::TRUSTED
    }

    pub fn is_trusted(&self) -> bool {
        self.contains(Self::TRUSTED)
    }

    pub fn with_alignment(self, align: u32) -> Self {
        let log2 = align.trailing_zeros();
        assert!(
            1 << log2 == align && align != 0,
            "Alignment must be a power of 2"
        );
        let log2 = log2.min(15) as u16;
        let bits = (self.bits() & !Self::ALIGN_MASK.bits()) | (log2 << 1);
        Self::from_bits_retain(bits)
    }

    pub fn alignment(&self) -> u32 {
        let log2 = (self.bits() & Self::ALIGN_MASK.bits()) >> 1;
        1 << log2
    }
}
