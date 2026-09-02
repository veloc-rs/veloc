use crate::opspec::define_opcodes;
use bitflags::bitflags;

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

impl IntCC {
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::LtS => "lts",
            Self::LtU => "ltu",
            Self::GtS => "gts",
            Self::GtU => "gtu",
            Self::LeS => "les",
            Self::LeU => "leu",
            Self::GeS => "ges",
            Self::GeU => "geu",
        }
    }

    pub fn from_mnemonic(mnemonic: &str) -> Option<Self> {
        match mnemonic {
            "eq" => Some(Self::Eq),
            "ne" => Some(Self::Ne),
            "lts" => Some(Self::LtS),
            "ltu" => Some(Self::LtU),
            "gts" => Some(Self::GtS),
            "gtu" => Some(Self::GtU),
            "les" => Some(Self::LeS),
            "leu" => Some(Self::LeU),
            "ges" => Some(Self::GeS),
            "geu" => Some(Self::GeU),
            _ => None,
        }
    }

    pub const fn is_unsigned(self) -> bool {
        matches!(self, IntCC::LtU | IntCC::GtU | IntCC::LeU | IntCC::GeU)
    }

    pub const fn swap(self) -> Self {
        match self {
            Self::Eq => Self::Eq,
            Self::Ne => Self::Ne,
            Self::LtS => Self::GtS,
            Self::LtU => Self::GtU,
            Self::GtS => Self::LtS,
            Self::GtU => Self::LtU,
            Self::LeS => Self::GeS,
            Self::LeU => Self::GeU,
            Self::GeS => Self::LeS,
            Self::GeU => Self::LeU,
        }
    }

    pub const fn complement(self) -> Self {
        match self {
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
            Self::LtS => Self::GeS,
            Self::LtU => Self::GeU,
            Self::GtS => Self::LeS,
            Self::GtU => Self::LeU,
            Self::LeS => Self::GtS,
            Self::LeU => Self::GtU,
            Self::GeS => Self::LtS,
            Self::GeU => Self::LtU,
        }
    }
}

impl FloatCC {
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Lt => "lt",
            Self::Gt => "gt",
            Self::Le => "le",
            Self::Ge => "ge",
        }
    }

    pub fn from_mnemonic(mnemonic: &str) -> Option<Self> {
        match mnemonic {
            "eq" => Some(Self::Eq),
            "ne" => Some(Self::Ne),
            "lt" => Some(Self::Lt),
            "gt" => Some(Self::Gt),
            "le" => Some(Self::Le),
            "ge" => Some(Self::Ge),
            _ => None,
        }
    }

    pub const fn swap(self) -> Self {
        match self {
            Self::Eq => Self::Eq,
            Self::Ne => Self::Ne,
            Self::Lt => Self::Gt,
            Self::Gt => Self::Lt,
            Self::Le => Self::Ge,
            Self::Ge => Self::Le,
        }
    }

    /// Exact logical complement under IEEE comparisons. Relational predicates
    /// cannot be complemented with this enum because unordered (NaN) is not
    /// represented separately.
    pub const fn complement(self) -> Option<Self> {
        match self {
            Self::Eq => Some(Self::Ne),
            Self::Ne => Some(Self::Eq),
            Self::Lt | Self::Gt | Self::Le | Self::Ge => None,
        }
    }

    /// Complement valid only when both operands are known not to be NaN.
    pub const fn complement_ordered(self) -> Self {
        match self {
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
            Self::Lt => Self::Ge,
            Self::Gt => Self::Le,
            Self::Le => Self::Gt,
            Self::Ge => Self::Lt,
        }
    }
}

impl core::fmt::Display for IntCC {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.mnemonic())
    }
}

impl core::fmt::Display for FloatCC {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.mnemonic())
    }
}

define_opcodes! {
    Iconst { mnemonic: "iconst", format: Iconst, types: INTEGER_RESULT, traits: [], memory: NONE }
    Fconst { mnemonic: "fconst", format: Fconst, types: FLOAT_RESULT, traits: [], memory: NONE }
    Bconst { mnemonic: "bconst", format: Bconst, types: BOOL_RESULT, traits: [], memory: NONE }
    Vconst { mnemonic: "vconst", format: Vconst, types: VECTOR_RESULT, traits: [], memory: NONE, constraints: [VectorConstant] }

    IAdd { mnemonic: "iadd", format: Binary, types: INTEGER_BINARY, builder: binary(iadd), traits: [COMMUTATIVE, ASSOCIATIVE], memory: NONE, identity: Zero }
    ISub { mnemonic: "isub", format: Binary, types: INTEGER_BINARY, builder: binary(isub), traits: [], memory: NONE }
    IMul { mnemonic: "imul", format: Binary, types: INTEGER_BINARY, builder: binary(imul), traits: [COMMUTATIVE, ASSOCIATIVE], memory: NONE, identity: One, absorbing: Zero }
    INeg { mnemonic: "ineg", format: Unary, types: INTEGER_UNARY, builder: unary(ineg), traits: [], memory: NONE }
    IAddSat { mnemonic: "iadd-sat", format: Binary, types: INTEGER_BINARY, builder: binary(iadd_sat), traits: [COMMUTATIVE], memory: NONE }
    ISubSat { mnemonic: "isub-sat", format: Binary, types: INTEGER_BINARY, builder: binary(isub_sat), traits: [], memory: NONE }
    IAddWithOverflow { mnemonic: "iadd-with-overflow", format: Binary, types: INTEGER_OVERFLOW, builder: binary_pair(iadd_with_overflow), traits: [COMMUTATIVE], memory: NONE }
    ISubWithOverflow { mnemonic: "isub-with-overflow", format: Binary, types: INTEGER_OVERFLOW, builder: binary_pair(isub_with_overflow), traits: [], memory: NONE }
    IMulWithOverflow { mnemonic: "imul-with-overflow", format: Binary, types: INTEGER_OVERFLOW, builder: binary_pair(imul_with_overflow), traits: [COMMUTATIVE], memory: NONE }
    IDivS { mnemonic: "idiv-s", format: Binary, types: INTEGER_BINARY, builder: binary(idiv_s), traits: [MAY_TRAP], memory: NONE }
    IDivU { mnemonic: "idiv-u", format: Binary, types: INTEGER_BINARY, builder: binary(idiv_u), traits: [MAY_TRAP], memory: NONE }
    IRemS { mnemonic: "irem-s", format: Binary, types: INTEGER_BINARY, builder: binary(irem_s), traits: [MAY_TRAP], memory: NONE }
    IRemU { mnemonic: "irem-u", format: Binary, types: INTEGER_BINARY, builder: binary(irem_u), traits: [MAY_TRAP], memory: NONE }

    FAdd { mnemonic: "fadd", format: Binary, types: FLOAT_BINARY, builder: binary(fadd), traits: [], memory: NONE }
    FSub { mnemonic: "fsub", format: Binary, types: FLOAT_BINARY, builder: binary(fsub), traits: [], memory: NONE }
    FMul { mnemonic: "fmul", format: Binary, types: FLOAT_BINARY, builder: binary(fmul), traits: [], memory: NONE }
    FNeg { mnemonic: "fneg", format: Unary, types: FLOAT_UNARY, builder: unary(fneg), traits: [], memory: NONE }
    FDiv { mnemonic: "fdiv", format: Binary, types: FLOAT_BINARY, builder: binary(fdiv), traits: [], memory: NONE }
    FMin { mnemonic: "fmin", format: Binary, types: FLOAT_BINARY, builder: binary(fmin), traits: [], memory: NONE }
    FMax { mnemonic: "fmax", format: Binary, types: FLOAT_BINARY, builder: binary(fmax), traits: [], memory: NONE }
    FCopysign { mnemonic: "fcopysign", format: Binary, types: FLOAT_BINARY, builder: binary(fcopysign), traits: [], memory: NONE }
    FAbs { mnemonic: "fabs", format: Unary, types: FLOAT_UNARY, builder: unary(fabs), traits: [], memory: NONE }
    FSqrt { mnemonic: "fsqrt", format: Unary, types: FLOAT_UNARY, builder: unary(fsqrt), traits: [], memory: NONE }
    FCeil { mnemonic: "fceil", format: Unary, types: FLOAT_UNARY, builder: unary(fceil), traits: [], memory: NONE }
    FFloor { mnemonic: "ffloor", format: Unary, types: FLOAT_UNARY, builder: unary(ffloor), traits: [], memory: NONE }
    FTrunc { mnemonic: "ftrunc", format: Unary, types: FLOAT_UNARY, builder: unary(ftrunc), traits: [], memory: NONE }
    FNearest { mnemonic: "fnearest", format: Unary, types: FLOAT_UNARY, builder: unary(fnearest), traits: [], memory: NONE }

    IAnd { mnemonic: "iand", format: Binary, types: INTEGER_OR_BOOL_BINARY, builder: binary(iand), traits: [COMMUTATIVE, ASSOCIATIVE, IDEMPOTENT], memory: NONE, identity: AllOnes, absorbing: Zero }
    IOr { mnemonic: "ior", format: Binary, types: INTEGER_OR_BOOL_BINARY, builder: binary(ior), traits: [COMMUTATIVE, ASSOCIATIVE, IDEMPOTENT], memory: NONE, identity: Zero, absorbing: AllOnes }
    IXor { mnemonic: "ixor", format: Binary, types: INTEGER_OR_BOOL_BINARY, builder: binary(ixor), traits: [COMMUTATIVE, ASSOCIATIVE], memory: NONE, identity: Zero }
    IShl { mnemonic: "ishl", format: Binary, types: INTEGER_BINARY, builder: binary(ishl), traits: [], memory: NONE }
    IShrS { mnemonic: "ishr-s", format: Binary, types: INTEGER_BINARY, builder: binary(ishr_s), traits: [], memory: NONE }
    IShrU { mnemonic: "ishr-u", format: Binary, types: INTEGER_BINARY, builder: binary(ishr_u), traits: [], memory: NONE }
    IRotl { mnemonic: "irotl", format: Binary, types: INTEGER_BINARY, builder: binary(irotl), traits: [], memory: NONE }
    IRotr { mnemonic: "irotr", format: Binary, types: INTEGER_BINARY, builder: binary(irotr), traits: [], memory: NONE }
    IClz { mnemonic: "iclz", format: Unary, types: INTEGER_UNARY, builder: unary(iclz), traits: [], memory: NONE }
    ICtz { mnemonic: "ictz", format: Unary, types: INTEGER_UNARY, builder: unary(ictz), traits: [], memory: NONE }
    IPopcnt { mnemonic: "ipopcnt", format: Unary, types: INTEGER_UNARY, builder: unary(ipopcnt), traits: [], memory: NONE }
    IEqz { mnemonic: "ieqz", format: Unary, types: INTEGER_TO_BOOL, builder: unary(ieqz), traits: [], memory: NONE }
    Icmp { mnemonic: "icmp", format: IntCompare, types: INTEGER_COMPARE, builder: int_compare(icmp), traits: [], memory: NONE, constraints: [PointerComparison] }
    Fcmp { mnemonic: "fcmp", format: FloatCompare, types: FLOAT_COMPARE, builder: float_compare(fcmp), traits: [], memory: NONE }

    ExtendS { mnemonic: "extends", format: Unary, types: EXTEND_SIGNED, builder: unary_typed(extend_s), traits: [], memory: NONE }
    ExtendU { mnemonic: "extendu", format: Unary, types: EXTEND_UNSIGNED, builder: unary_typed(extend_u), traits: [], memory: NONE }
    Wrap { mnemonic: "wrap", format: Unary, types: NARROW_INTEGER, builder: unary_typed(wrap), traits: [], memory: NONE }
    FloatToIntSatS { mnemonic: "float-to-int-sat-s", format: Unary, types: FLOAT_TO_INTEGER, builder: unary_typed(float_to_int_sat_s), traits: [], memory: NONE }
    FloatToIntSatU { mnemonic: "float-to-int-sat-u", format: Unary, types: FLOAT_TO_INTEGER, builder: unary_typed(float_to_int_sat_u), traits: [], memory: NONE }
    FloatToIntS { mnemonic: "float-to-int-s", format: Unary, types: FLOAT_TO_INTEGER, builder: unary_typed(float_to_int_s), traits: [MAY_TRAP], memory: NONE }
    FloatToIntU { mnemonic: "float-to-int-u", format: Unary, types: FLOAT_TO_INTEGER, builder: unary_typed(float_to_int_u), traits: [MAY_TRAP], memory: NONE }
    IntToFloatS { mnemonic: "int-to-float-s", format: Unary, types: INTEGER_TO_FLOAT, builder: unary_typed(int_to_float_s), traits: [], memory: NONE }
    IntToFloatU { mnemonic: "int-to-float-u", format: Unary, types: INTEGER_TO_FLOAT, builder: unary_typed(int_to_float_u), traits: [], memory: NONE }
    FloatPromote { mnemonic: "float-promote", format: Unary, types: FLOAT_PROMOTE, builder: unary(float_promote), traits: [], memory: NONE }
    FloatDemote { mnemonic: "float-demote", format: Unary, types: FLOAT_DEMOTE, builder: unary(float_demote), traits: [], memory: NONE }
    Reinterpret { mnemonic: "reinterpret", format: Unary, types: REINTERPRET, builder: unary_typed(reinterpret), traits: [], memory: NONE }
    IntToPtr { mnemonic: "inttoptr", format: IntToPtr, types: INT_TO_PTR, builder: unary_inst(int_to_ptr), traits: [], memory: NONE }
    PtrToInt { mnemonic: "ptrtoint", format: PtrToInt, types: PTR_TO_INT, builder: unary_inst_typed(ptr_to_int), traits: [], memory: NONE }

    Load { mnemonic: "load", format: Load, types: LOAD, traits: [MAY_TRAP], memory: HEAP_READ }
    Store { mnemonic: "store", format: Store, types: STORE, traits: [MAY_TRAP], memory: HEAP_WRITE }
    StackLoad { mnemonic: "stack-load", format: StackLoad, types: STACK_LOAD, traits: [], memory: STACK_READ }
    StackStore { mnemonic: "stack-store", format: StackStore, types: STACK_STORE, traits: [], memory: STACK_WRITE }
    StackAddr { mnemonic: "stack-addr", format: StackAddr, types: STACK_ADDR, traits: [], memory: NONE }
    PtrOffset { mnemonic: "ptr-offset", format: PtrOffset, types: PTR_OFFSET, traits: [], memory: NONE }
    PtrIndex { mnemonic: "ptr-index", format: PtrIndex, types: PTR_INDEX, traits: [], memory: NONE, constraints: [NonZeroScale] }

    Call { mnemonic: "call", format: Call, types: SIGNATURE, traits: [MAY_TRAP], memory: UNKNOWN }
    CallIndirect { mnemonic: "call-indirect", format: CallIndirect, types: SIGNATURE, traits: [MAY_TRAP], memory: UNKNOWN }
    CallIntrinsic { mnemonic: "call-intrinsic", format: CallIntrinsic, types: SIGNATURE, traits: [MAY_TRAP], memory: UNKNOWN }
    Jump { mnemonic: "jump", format: Jump, types: EXTERNAL_OPERANDS, traits: [TERMINATOR], memory: NONE }
    Br { mnemonic: "br", format: Br, types: EXTERNAL_OPERANDS, traits: [TERMINATOR], memory: NONE }
    BrTable { mnemonic: "br-table", format: BrTable, types: EXTERNAL_OPERANDS, traits: [TERMINATOR], memory: NONE }
    Return { mnemonic: "return", format: Return, types: EXTERNAL_OPERANDS, traits: [TERMINATOR], memory: NONE }
    Select { mnemonic: "select", format: Ternary, types: SELECT, builder: ternary(select), traits: [], memory: NONE }
    Unreachable { mnemonic: "unreachable", format: Unreachable, types: NONE, builder: nullary(unreachable), traits: [TERMINATOR, MAY_TRAP], memory: NONE }
    Nop { mnemonic: "nop", format: Nop, types: NONE, builder: nullary(nop), traits: [], memory: NONE }

    Splat { mnemonic: "splat", format: Unary, types: SPLAT, builder: unary_typed(splat), traits: [], memory: NONE }
    Shuffle { mnemonic: "shuffle", format: Shuffle, types: SHUFFLE, traits: [], memory: NONE, constraints: [ShuffleMask] }
    InsertElement { mnemonic: "insertelement", format: Ternary, types: INSERT_ELEMENT, traits: [], memory: NONE }
    ExtractElement { mnemonic: "extractelement", format: Binary, types: EXTRACT_ELEMENT, traits: [], memory: NONE }
    ReduceSum { mnemonic: "reduce-sum", format: Unary, types: REDUCTION, builder: unary(reduce_sum), traits: [], memory: NONE }
    ReduceAdd { mnemonic: "reduce-add", format: Unary, types: REDUCTION, builder: unary(reduce_add), traits: [], memory: NONE }
    ReduceMin { mnemonic: "reduce-min", format: Unary, types: REDUCTION, builder: unary(reduce_min), traits: [], memory: NONE }
    ReduceMax { mnemonic: "reduce-max", format: Unary, types: REDUCTION, builder: unary(reduce_max), traits: [], memory: NONE }
    ReduceAnd { mnemonic: "reduce-and", format: Unary, types: REDUCTION, builder: unary(reduce_and), traits: [], memory: NONE }
    ReduceOr { mnemonic: "reduce-or", format: Unary, types: REDUCTION, builder: unary(reduce_or), traits: [], memory: NONE }
    ReduceXor { mnemonic: "reduce-xor", format: Unary, types: REDUCTION, builder: unary(reduce_xor), traits: [], memory: NONE }
    LoadStride { mnemonic: "load-stride", format: VectorLoadStrided, types: VECTOR_LOAD_STRIDED, traits: [MAY_TRAP], memory: HEAP_READ }
    StoreStride { mnemonic: "store-stride", format: VectorStoreStrided, types: VECTOR_STORE_STRIDED, traits: [MAY_TRAP], memory: HEAP_WRITE }
    Gather { mnemonic: "gather", format: VectorGather, types: GATHER, traits: [MAY_TRAP], memory: HEAP_READ }
    Scatter { mnemonic: "scatter", format: VectorScatter, types: SCATTER, traits: [MAY_TRAP], memory: HEAP_WRITE }
    SetVL { mnemonic: "setvl", format: Unary, types: SET_VECTOR_LENGTH, builder: unary(setvl), traits: [], memory: NONE }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct MemFlags: u16 {
        const TRUSTED = 1 << 0;
        const ALIGN_MASK = 0b1111 << 1;
        const VOLATILE = 1 << 5;
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

    pub fn is_volatile(&self) -> bool {
        self.contains(Self::VOLATILE)
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
