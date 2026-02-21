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
    pub fn is_unsigned(self) -> bool {
        matches!(self, IntCC::LtU | IntCC::GtU | IntCC::LeU | IntCC::GeU)
    }
}

impl FloatCC {}

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

impl Opcode {}

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
