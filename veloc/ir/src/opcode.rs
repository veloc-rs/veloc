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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Opcode {
    Iconst,
    Fconst,
    Bconst,
    Vconst,
    // Integer Arithmetic
    IAdd,
    ISub,
    IMul,
    INeg,
    // Saturating Arithmetic
    IAddSat,
    ISubSat,
    // Arithmetic with Overflow (returns result and overflow flag)
    IAddWithOverflow,
    ISubWithOverflow,
    IMulWithOverflow,
    // Integer-only Arithmetic
    IDivS,
    IDivU,
    IRemS,
    IRemU,
    // Float Arithmetic
    FAdd,
    FSub,
    FMul,
    FNeg,
    FDiv,
    // Float Unary
    FMin,
    FMax,
    FCopysign,
    FAbs,
    FSqrt,
    FCeil,
    FFloor,
    FTrunc,
    FNearest,
    // Bitwise (Integer-only)
    IAnd,
    IOr,
    IXor,
    IShl,
    IShrS,
    IShrU,
    IRotl,
    IRotr,
    // Integer Unary
    IClz,
    ICtz,
    IPopcnt,
    IEqz,
    // Comparisons (Keep prefixes for clarity or consistency)
    Icmp,
    Fcmp,
    // Conversions
    // Integer extension/truncation
    ExtendS,
    ExtendU,
    Wrap,
    // Float -> Int
    FloatToIntSatS, // Saturating float to signed int
    FloatToIntSatU, // Saturating float to unsigned int
    FloatToIntS,    // Float to signed int (truncate)
    FloatToIntU,    // Float to unsigned int (truncate)
    // Int -> Float
    IntToFloatS, // Signed int to float
    IntToFloatU, // Unsigned int to float
    // Float <-> Float
    FloatPromote, // F32 -> F64
    FloatDemote,  // F64 -> F32
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
    CallIntrinsic,
    Jump,
    Br,
    BrTable,
    Return,
    Select,
    Unreachable,
    Nop,

    // ======================================
    // Vector Operations
    // ======================================

    // --- 1. 纯向量操作 (Vector Only) ---
    /// 标量 -> 向量广播 (Scalar to Vector broadcast)
    /// e.g., splat 5 -> [5, 5, 5, 5]
    Splat,

    /// 向量重排/混洗 (Vector shuffle)
    /// 两个输入向量 + 常量掩码 -> 重排后的向量
    Shuffle,

    /// 插入标量到向量的指定通道
    /// args: [vector, scalar, lane_index]
    InsertElement,

    /// 从向量提取指定通道的标量
    /// args: [vector, lane_index]
    ExtractElement,

    // --- 2. 归约操作 (Horizontal/Reduction) ---
    /// 向量求和 -> 标量 (无序/浮点可能无序)
    ReduceSum,
    /// 向量求和 -> 标量 (有序/确定顺序，主要用于浮点)
    ReduceAdd,
    /// 向量最小值 -> 标量
    ReduceMin,
    /// 向量最大值 -> 标量
    ReduceMax,
    /// 向量按位与 -> 标量
    ReduceAnd,
    /// 向量按位或 -> 标量
    ReduceOr,
    /// 向量按位异或 -> 标量
    ReduceXor,

    // --- 3. 向量内存操作 ---
    /// 固定步长加载 (Strided Load)
    /// ptr + stride * i
    LoadStride,
    /// 固定步长存储 (Strided Store)
    StoreStride,
    /// 离散/聚集加载 (Gather/Indexed Load)
    /// base_ptr + indices[i]
    Gather,
    /// 离散/分散存储 (Scatter/Indexed Store)
    Scatter,

    // --- 4. 控制流与配置 ---
    /// 设置向量长度 (Set Vector Length)
    /// 类似 RISC-V V 的 vsetvli 指令
    /// 输入: 请求的向量长度 (AVL)
    /// 输出: 实际向量长度 (VL)
    SetVL,
}

impl fmt::Display for Opcode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Opcode::Iconst => "iconst",
            Opcode::Fconst => "fconst",
            Opcode::Bconst => "bconst",
            Opcode::Vconst => "vconst",
            Opcode::IAdd => "iadd",
            Opcode::ISub => "isub",
            Opcode::IMul => "imul",
            Opcode::INeg => "ineg",
            Opcode::IAddSat => "iadd_sat",
            Opcode::ISubSat => "isub_sat",
            Opcode::IAddWithOverflow => "iadd_with_overflow",
            Opcode::ISubWithOverflow => "isub_with_overflow",
            Opcode::IMulWithOverflow => "imul_with_overflow",
            Opcode::IDivS => "idiv_s",
            Opcode::IDivU => "idiv_u",
            Opcode::IRemS => "irem_s",
            Opcode::IRemU => "irem_u",
            Opcode::FAdd => "fadd",
            Opcode::FSub => "fsub",
            Opcode::FMul => "fmul",
            Opcode::FNeg => "fneg",
            Opcode::FDiv => "fdiv",
            Opcode::FMin => "fmin",
            Opcode::FMax => "fmax",
            Opcode::FCopysign => "fcopysign",
            Opcode::FAbs => "fabs",
            Opcode::FSqrt => "fsqrt",
            Opcode::FCeil => "fceil",
            Opcode::FFloor => "ffloor",
            Opcode::FTrunc => "ftrunc",
            Opcode::FNearest => "fnearest",
            Opcode::IAnd => "iand",
            Opcode::IOr => "ior",
            Opcode::IXor => "ixor",
            Opcode::IShl => "ishl",
            Opcode::IShrS => "ishr_s",
            Opcode::IShrU => "ishr_u",
            Opcode::IRotl => "irotl",
            Opcode::IRotr => "irotr",
            Opcode::IClz => "iclz",
            Opcode::ICtz => "ictz",
            Opcode::IPopcnt => "ipopcnt",
            Opcode::IEqz => "ieqz",
            Opcode::Icmp => "icmp",
            Opcode::Fcmp => "fcmp",
            Opcode::ExtendS => "extends",
            Opcode::ExtendU => "extendu",
            Opcode::Wrap => "wrap",
            Opcode::FloatToIntSatS => "float_to_int_sat_s",
            Opcode::FloatToIntSatU => "float_to_int_sat_u",
            Opcode::FloatToIntS => "float_to_int_s",
            Opcode::FloatToIntU => "float_to_int_u",
            Opcode::IntToFloatS => "int_to_float_s",
            Opcode::IntToFloatU => "int_to_float_u",
            Opcode::FloatPromote => "float_promote",
            Opcode::FloatDemote => "float_demote",
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
            Opcode::CallIntrinsic => "call_intrinsic",
            Opcode::Jump => "jump",
            Opcode::Br => "br",
            Opcode::BrTable => "br_table",
            Opcode::Return => "return",
            Opcode::Select => "select",
            Opcode::Unreachable => "unreachable",
            Opcode::Nop => "nop",
            // Vector operations
            Opcode::Splat => "splat",
            Opcode::Shuffle => "shuffle",
            Opcode::InsertElement => "insertelement",
            Opcode::ExtractElement => "extractelement",
            Opcode::ReduceSum => "reduce_sum",
            Opcode::ReduceAdd => "reduce_add",
            Opcode::ReduceMin => "reduce_min",
            Opcode::ReduceMax => "reduce_max",
            Opcode::ReduceAnd => "reduce_and",
            Opcode::ReduceOr => "reduce_or",
            Opcode::ReduceXor => "reduce_xor",
            Opcode::LoadStride => "load_stride",
            Opcode::StoreStride => "store_stride",
            Opcode::Gather => "gather",
            Opcode::Scatter => "scatter",
            Opcode::SetVL => "setvl",
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
