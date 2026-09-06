//! Low-level IR (LIR) 指令和操作数定义

use alloc::string::String;
use cranelift_entity::entity_impl;
use smallvec::SmallVec;
use veloc_ir::semantics::BvOp;
use veloc_ir::{Block, FloatCC, IntCC, Type};

pub use crate::lir::RegisterBank;
use crate::lir::extra::CallInfo;
use crate::lir::symbol::SymbolId;

/// 机器指令索引
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InstId(u32);
entity_impl!(InstId, "inst");

/// 虚拟寄存器索引
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VReg(u32);
entity_impl!(VReg, "vreg");

/// 寄存器标识符 (虚拟或物理)
///
/// 最高位为 1 表示虚拟寄存器 (VReg)，为 0 表示物理寄存器 (PReg)
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Reg(pub u32);

impl Reg {
    const VREG_MARK: u32 = 1 << 31;

    /// 创建一个虚拟寄存器
    pub fn new_vreg(index: u32) -> Self {
        debug_assert!(index < Self::VREG_MARK);
        Self(index | Self::VREG_MARK)
    }

    /// 创建一个物理寄存器
    pub fn new_preg(index: u32) -> Self {
        debug_assert!(index < Self::VREG_MARK);
        Self(index)
    }

    /// 检查是否为虚拟寄存器
    pub fn is_vreg(&self) -> bool {
        (self.0 & Self::VREG_MARK) != 0
    }

    /// 检查是否为物理寄存器
    pub fn is_preg(&self) -> bool {
        (self.0 & Self::VREG_MARK) == 0
    }

    /// 获取原始索引 (去掉 VReg 标记)
    pub fn index(&self) -> u32 {
        self.0 & !Self::VREG_MARK
    }
}

impl core::fmt::Debug for Reg {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_vreg() {
            write!(f, "v{}", self.index())
        } else {
            write!(f, "p{}", self.index())
        }
    }
}

/// 保证只有标记为可写的寄存器才能被修改的类型级 Wrapper
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Writable<T>(pub T);

impl<T> Writable<T> {
    /// 获取只读引用
    #[inline(always)]
    pub fn to_reg(&self) -> T
    where
        T: Copy,
    {
        self.0
    }
}

impl<T: core::fmt::Debug> core::fmt::Debug for Writable<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "def({:?})", self.0)
    }
}

/// 栈槽标识符
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StackSlot(pub u32);
entity_impl!(StackSlot, "stackslot");

/// 寄存器数据
#[derive(Debug, Clone)]
pub struct VRegData {
    pub ty: Type,
    pub bank: Option<RegisterBank>, // 寄存器库，在合法化/指令选择阶段确定
    pub assigned_reg: Option<Reg>,  // 寄存器分配后填充，通常为 PReg
    pub stack_slot: Option<StackSlot>, // 如果溢出到栈
}

macro_rules! define_generic_opcodes {
    ($($opcode:ident $(=> $semantic:ident)?),* $(,)?) => {
        /// Target-independent machine operations, selected into target instructions.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[allow(non_camel_case_types)]
        pub enum GenericOpcode {
            $($opcode,)*
        }

        impl GenericOpcode {
            /// Declared scalar or per-lane bitvector operation. This does not
            /// describe register effects or prove a complete LIR lowering.
            /// Consumers must check the instruction's types and predication;
            /// this binding gives no pointer or memory semantics.
            pub const fn semantics(self) -> Option<BvOp> {
                match self {
                    $(Self::$opcode => define_generic_opcodes!(@semantic $($semantic)?),)*
                }
            }

            /// Find a generic instruction implementing this declared operation.
            /// Operations without a binding remain outside this lowering path.
            pub fn from_semantics(semantics: BvOp) -> Option<Self> {
                const BINDINGS: &[(BvOp, GenericOpcode)] = &[
                    $($( (BvOp::$semantic, GenericOpcode::$opcode), )?)*
                ];
                BINDINGS.iter().find_map(|&(semantic, opcode)| {
                    (semantic == semantics).then_some(opcode)
                })
            }
        }
    };
    (@semantic $semantic:ident) => { Some(BvOp::$semantic) };
    (@semantic) => { None };
}

define_generic_opcodes! {
    // ==================== 整数算术 ====================
    G_ADD => Add,  // 加法
    G_SUB => Sub,  // 减法
    G_MUL => Mul,  // 乘法
    G_SDIV, // 有符号除法
    G_UDIV, // 无符号除法
    G_SREM, // 有符号取余
    G_UREM, // 无符号取余
    G_NEG => Neg,  // 取负

    // ==================== 浮点算术 ====================
    G_FADD,  // 浮点加法
    G_FSUB,  // 浮点减法
    G_FMUL,  // 浮点乘法
    G_FDIV,  // 浮点除法
    G_FNEG,  // 浮点取负
    G_FABS,  // 浮点绝对值
    G_FSQRT, // 浮点开方

    // ==================== 位运算 ====================
    G_AND => And,   // 按位与
    G_OR => Or,    // 按位或
    G_XOR => Xor,   // 按位异或
    G_CTPOP, // 统计置位位数
    G_CTLZ,  // 统计前导零
    G_CTTZ,  // 统计尾随零
    G_SHL,   // 逻辑左移
    G_LSHR,  // 逻辑右移
    G_ASHR,  // 算术右移
    G_ROTL,  // 循环左移
    G_ROTR,  // 循环右移

    // ==================== 比较 ====================
    G_ICMP,            // 整数比较
    G_FCMP,            // 浮点比较
    G_IEQZ,            // 整数等于零比较 (Dst, Src)
    G_ANYEXT,          // 任意扩展
    G_ABS,             // 绝对值
    G_SMIN,            // 有符号最小值
    G_SMAX,            // 有符号最大值
    G_UMIN,            // 无符号最小值
    G_UMAX,            // 无符号最大值
    G_UADDO,           // 无符号加法溢出
    G_SADDO,           // 有符号加法溢出
    G_USUBO,           // 无符号减法溢出
    G_SSUBO,           // 有符号减法溢出
    G_UADDE,           // 带进位无符号加法
    G_SADDE,           // 带进位有符号加法
    G_USUBE,           // 带借位无符号减法
    G_SSUBE,           // 带借位有符号减法
    G_UMULO,           // 无符号乘法溢出
    G_SMULO,           // 有符号乘法溢出
    G_UMULH,           // 无符号乘法高位
    G_SMULH,           // 有符号乘法高位
    G_CTLZ_ZERO_UNDEF, // 计数前导零，零输入未定义
    G_CTTZ_ZERO_UNDEF, // 计数尾随零，零输入未定义
    G_SADDSAT,         // 有符号饱和加法
    G_UADDSAT,         // 无符号饱和加法
    G_SSUBSAT,         // 有符号饱和减法
    G_USUBSAT,         // 无符号饱和减法

    // ==================== 内存操作 ====================
    G_LOAD,        // 加载 (Dst, Ptr)
    G_STORE,       // 存储 (Src, Ptr)
    G_PTR_ADD,     // 指针加法 (Base, Offset)
    G_STACK_LOAD,  // 从栈加载
    G_STACK_STORE, // 存储到栈
    G_STACK_ADDR,  // 获取栈地址

    G_OFFSET_LOAD,   // [Dst] = Load (Base + Offset)
    G_OFFSET_STORE,  // [Base + Offset] = Store (Src)
    G_INDEXED_LOAD,  // 带写回的加载 (Dst, Base_out, Base_in, Offset)
    G_INDEXED_STORE, // 带写回的存储 (Base_out, Src, Base_in, Offset)

    // ==================== 常量 ====================
    G_CONSTANT,  // 整数常量
    G_FCONSTANT, // 浮点常量

    // ==================== 类型转换 ====================
    G_TRUNC,   // 截断
    G_ZEXT,    // 零扩展
    G_SEXT,    // 符号扩展
    G_FPTOSI,  // 浮点到有符号整数
    G_FPTOUI,  // 浮点到无符号整数
    G_SITOFP,  // 有符号整数到浮点
    G_UITOFP,  // 无符号整数到浮点
    G_FPTRUNC, // 浮点截断
    G_FPEXT,   // 浮点扩展
    G_BITCAST, // 位转换

    G_INTTOPTR, // 整数转指针
    G_PTRTOINT, // 指针转整数

    // ==================== 控制流 ====================
    G_BR,      // 无条件跳转
    G_BRCOND,  // 条件跳转
    G_BRIND,   // 间接跳转
    G_BRJT,    // 跳转表
    G_RET,     // 返回
    G_CALL,    // 直接调用
    G_CALLIND, // 间接调用
    G_ARG,     // 获取函数参数 (Index)

    // ==================== 其他 ====================
    G_SELECT,  // 选择
    G_COPY,    // 寄存器拷贝
    G_PHI,     // PHI 节点
    G_EXTRACT, // 提取向量元素
    G_INSERT,  // 插入向量元素
    G_UNMERGE, // 拆分数值
    G_MERGE,   // 合并数值

    // ==================== 平台相关（需 Lower）====================
    G_READCYCLECOUNTER, // 读取周期计数器
    G_UNREACHABLE,      // 不可达
}

/// 机器指令操作码
#[derive(Debug, Clone)]
pub enum MachineOpcode {
    /// 无效指令（占位符，用于指令融合或删除）
    Invalid,
    /// 通用操作码（需要指令选择）
    Generic(GenericOpcode),
    /// 目标架构特定操作码（指令选择后）
    Target(u32),
}

/// 条件码
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CondCode {
    Int(IntCC),
    Float(FloatCC),
}

/// 机器指令操作数
#[derive(Debug, Clone)]
pub enum MachineOperand {
    /// 纯粹的定义 (覆盖写)
    Def(Writable<Reg>),
    /// 纯粹的使用 (只读)
    Use(Reg),
    /// 读改写 (对于两地址指令如 x86 的 add a, b -> a = a + b)
    /// 寄存器分配器必须保证它前后的物理寄存器相同
    TiedDefUse(Writable<Reg>),
    /// 整数立即数
    Imm(i64),
    /// 浮点立即数
    FImm(f64),
    /// 基本块引用
    Block(Block),
    /// 栈槽
    StackSlot(StackSlot),
    /// 条件码（用于比较）
    CondCode(CondCode),
    /// 全局符号
    Global(SymbolId),
}

impl MachineOperand {
    pub fn is_def(&self) -> bool {
        matches!(self, Self::Def(_) | Self::TiedDefUse(_))
    }

    pub fn is_use(&self) -> bool {
        matches!(self, Self::Use(_) | Self::TiedDefUse(_))
    }

    pub fn as_reg(&self) -> Option<Reg> {
        match self {
            Self::Def(w) | Self::TiedDefUse(w) => Some(w.0),
            Self::Use(r) => Some(*r),
            _ => None,
        }
    }

    pub fn as_writable(&self) -> Option<Writable<Reg>> {
        match self {
            Self::Def(w) | Self::TiedDefUse(w) => Some(*w),
            _ => None,
        }
    }

    pub fn as_stack_slot(&self) -> Option<StackSlot> {
        match self {
            Self::StackSlot(slot) => Some(*slot),
            _ => None,
        }
    }
}

macro_rules! define_mir_ops {
    (
        $(
            $schema:ident => $struct_name:ident / $accessor:ident {
                opcodes: [$($opcode:path),+ $(,)?],
                builders: {
                    $($builder_spec:tt)*
                },
                len: $len_kind:ident($($len_args:expr),*),
                len_message: $len_message:literal,
                fields: {
                    $( $field:ident : $field_ty:ty = $decoder:ident($index:expr, $message:literal) ),* $(,)?
                }
            }
        )*
    ) => {
        $(
            #[derive(Debug, Clone, PartialEq)]
            pub struct $struct_name {
                $( pub $field: $field_ty, )*
            }
        )*

        /// 通用 LIR 指令的 schema 分类。
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum GenericInstSchema {
            $(
                $schema,
            )*
        }

        #[derive(Debug, Clone, PartialEq)]
        pub enum DecodedGenericInst {
            $(
                $schema($struct_name),
            )*
        }

        fn simple_generic_schema_for_opcode(opcode: GenericOpcode) -> Option<GenericInstSchema> {
            match opcode {
                $(
                    $( $opcode )|+ => Some(GenericInstSchema::$schema),
                )*
                _ => None,
            }
        }

        fn decode_simple_generic(
            inst: &MachineInst,
            schema: GenericInstSchema,
        ) -> crate::error::Result<DecodedGenericInst> {
            match schema {
                $(
                    GenericInstSchema::$schema => inst.$accessor().map(DecodedGenericInst::$schema),
                )*
            }
        }

        impl MachineInst {
            $(
                define_mir_ops!(@emit_builders $($builder_spec)*);

                pub fn $accessor(&self) -> crate::error::Result<$struct_name> {
                    self.expect_schema(GenericInstSchema::$schema)?;
                    define_mir_ops!(@check_len self, $len_kind, ($($len_args),*), $len_message);
                    Ok($struct_name {
                        $( $field: self.$decoder($index, $message)?, )*
                    })
                }
            )*
        }

        impl GenericInstSchema {
            pub fn for_opcode(opcode: GenericOpcode) -> Option<Self> {
                simple_generic_schema_for_opcode(opcode)
            }
        }
    };

    (@emit_builders) => {};

    (@emit_builders
        $builder:ident => $builder_opcode:path => (
            $( $arg:ident : $arg_ty:ty => $operand_kind:ident ),* $(,)?
        );
        $($rest:tt)*
    ) => {
        pub fn $builder($($arg: $arg_ty),*) -> Self {
            Self {
                opcode: MachineOpcode::Generic($builder_opcode),
                operands: smallvec::smallvec![
                    $( define_mir_ops!(@build_operand $operand_kind $arg) ),*
                ],
            }
        }

        define_mir_ops!(@emit_builders $($rest)*);
    };

    (@emit_builders
        $item:item
        $($rest:tt)*
    ) => {
        $item
        define_mir_ops!(@emit_builders $($rest)*);
    };

    (@check_len $self:ident, exact, ($expected:expr), $message:expr) => {
        $self.expect_len($expected, $message)?;
    };

    (@check_len $self:ident, one_of, ($first:expr, $second:expr), $message:expr) => {
        if !($self.operands.len() == $first || $self.operands.len() == $second) {
            return Err($self.decode_error($message));
        }
    };

    (@check_len $self:ident, any, (), $message:expr) => {};

    (@build_operand Def $arg:ident) => {
        MachineOperand::Def($arg)
    };
    (@build_operand Use $arg:ident) => {
        MachineOperand::Use($arg)
    };
    (@build_operand TiedDefUse $arg:ident) => {
        MachineOperand::TiedDefUse($arg)
    };
    (@build_operand Imm $arg:ident) => {
        MachineOperand::Imm($arg)
    };
    (@build_operand FImm $arg:ident) => {
        MachineOperand::FImm($arg)
    };
    (@build_operand Block $arg:ident) => {
        MachineOperand::Block($arg)
    };
    (@build_operand StackSlot $arg:ident) => {
        MachineOperand::StackSlot($arg)
    };
    (@build_operand Global $arg:ident) => {
        MachineOperand::Global($arg)
    };
    (@build_operand IntCC $arg:ident) => {
        MachineOperand::CondCode(CondCode::Int($arg))
    };
    (@build_operand FloatCC $arg:ident) => {
        MachineOperand::CondCode(CondCode::Float($arg))
    };

}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallCallee {
    Direct(SymbolId),
    Indirect(Reg),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallShape {
    pub defs: SmallVec<[Reg; 2]>,
    pub callee: CallCallee,
    pub args: SmallVec<[Reg; 4]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallInst<'a> {
    pub shape: CallShape,
    pub info: &'a CallInfo,
}

define_mir_ops! {
    UnaryReg => UnaryRegInst / as_unary_reg {
        opcodes: [
            GenericOpcode::G_ANYEXT,
            GenericOpcode::G_NEG,
            GenericOpcode::G_ABS,
            GenericOpcode::G_FNEG,
            GenericOpcode::G_FABS,
            GenericOpcode::G_FSQRT,
            GenericOpcode::G_CTPOP,
            GenericOpcode::G_CTLZ,
            GenericOpcode::G_CTTZ,
            GenericOpcode::G_CTLZ_ZERO_UNDEF,
            GenericOpcode::G_CTTZ_ZERO_UNDEF,
            GenericOpcode::G_TRUNC,
            GenericOpcode::G_ZEXT,
            GenericOpcode::G_SEXT,
            GenericOpcode::G_FPTOSI,
            GenericOpcode::G_FPTOUI,
            GenericOpcode::G_SITOFP,
            GenericOpcode::G_UITOFP,
            GenericOpcode::G_FPTRUNC,
            GenericOpcode::G_FPEXT,
            GenericOpcode::G_BITCAST,
            GenericOpcode::G_INTTOPTR,
            GenericOpcode::G_PTRTOINT,
            GenericOpcode::G_COPY,
            GenericOpcode::G_IEQZ,
        ],
        builders: {
            build_copy => GenericOpcode::G_COPY => (
                def: Writable<Reg> => Def,
                src: Reg => Use
            );
            /// 构建一元操作指令
            pub fn build_unary(opcode: MachineOpcode, def: Writable<Reg>, src: Reg) -> Self {
                Self {
                    opcode,
                    operands: smallvec::smallvec![MachineOperand::Def(def), MachineOperand::Use(src)],
                }
            }
        },
        len: exact(2),
        len_message: "unary instruction expects def/use operands",
        fields: {
            dst: Reg = expect_def_reg(0, "operand 0 must be a def"),
            src: Reg = expect_use_reg(1, "operand 1 must be a use"),
        }
    }
    BinaryReg => BinaryRegInst / as_binary_reg {
        opcodes: [
            GenericOpcode::G_ADD,
            GenericOpcode::G_SUB,
            GenericOpcode::G_MUL,
            GenericOpcode::G_SMIN,
            GenericOpcode::G_SMAX,
            GenericOpcode::G_UMIN,
            GenericOpcode::G_UMAX,
            GenericOpcode::G_SADDSAT,
            GenericOpcode::G_UADDSAT,
            GenericOpcode::G_SSUBSAT,
            GenericOpcode::G_USUBSAT,
            GenericOpcode::G_SDIV,
            GenericOpcode::G_UDIV,
            GenericOpcode::G_SREM,
            GenericOpcode::G_UREM,
            GenericOpcode::G_FADD,
            GenericOpcode::G_FSUB,
            GenericOpcode::G_FMUL,
            GenericOpcode::G_FDIV,
            GenericOpcode::G_AND,
            GenericOpcode::G_OR,
            GenericOpcode::G_XOR,
            GenericOpcode::G_SHL,
            GenericOpcode::G_LSHR,
            GenericOpcode::G_ASHR,
            GenericOpcode::G_ROTL,
            GenericOpcode::G_ROTR,
            GenericOpcode::G_PTR_ADD,
            GenericOpcode::G_UMULH,
            GenericOpcode::G_SMULH,
        ],
        builders: {
            /// 构建三地址二元操作指令
            pub fn build_binary(
                opcode: MachineOpcode,
                def: Writable<Reg>,
                src0: Reg,
                src1: Reg,
            ) -> Self {
                Self {
                    opcode,
                    operands: smallvec::smallvec![
                        MachineOperand::Def(def),
                        MachineOperand::Use(src0),
                        MachineOperand::Use(src1)
                    ],
                }
            }

            /// 构建两地址二元指令 (如 x86 的 add eax, ecx)
            pub fn build_tied_binary(
                opcode: MachineOpcode,
                def_use: Writable<Reg>,
                src: Reg,
            ) -> Self {
                Self {
                    opcode,
                    operands: smallvec::smallvec![
                        MachineOperand::TiedDefUse(def_use),
                        MachineOperand::Use(src)
                    ],
                }
            }
        },
        len: exact(3),
        len_message: "binary instruction expects def/use/use operands",
        fields: {
            dst: Reg = expect_def_reg(0, "operand 0 must be a def"),
            lhs: Reg = expect_use_reg(1, "operand 1 must be a use"),
            rhs: Reg = expect_use_reg(2, "operand 2 must be a use"),
        }
    }
    BinaryRegWithFlags => BinaryRegWithFlagsInst / as_binary_reg_with_flags {
        opcodes: [
            GenericOpcode::G_UADDO,
            GenericOpcode::G_SADDO,
            GenericOpcode::G_USUBO,
            GenericOpcode::G_SSUBO,
            GenericOpcode::G_UADDE,
            GenericOpcode::G_SADDE,
            GenericOpcode::G_USUBE,
            GenericOpcode::G_SSUBE,
            GenericOpcode::G_UMULO,
            GenericOpcode::G_SMULO,
        ],
        builders: {
            build_uaddo => GenericOpcode::G_UADDO => (dst: Writable<Reg> => Def, flag: Writable<Reg> => Def, lhs: Reg => Use, rhs: Reg => Use);
            build_saddo => GenericOpcode::G_SADDO => (dst: Writable<Reg> => Def, flag: Writable<Reg> => Def, lhs: Reg => Use, rhs: Reg => Use);
            build_usubo => GenericOpcode::G_USUBO => (dst: Writable<Reg> => Def, flag: Writable<Reg> => Def, lhs: Reg => Use, rhs: Reg => Use);
            build_ssubo => GenericOpcode::G_SSUBO => (dst: Writable<Reg> => Def, flag: Writable<Reg> => Def, lhs: Reg => Use, rhs: Reg => Use);
            build_uadde => GenericOpcode::G_UADDE => (dst: Writable<Reg> => Def, flag: Writable<Reg> => Def, lhs: Reg => Use, rhs: Reg => Use, carry_in: Reg => Use);
            build_sadde => GenericOpcode::G_SADDE => (dst: Writable<Reg> => Def, flag: Writable<Reg> => Def, lhs: Reg => Use, rhs: Reg => Use, carry_in: Reg => Use);
            build_usube => GenericOpcode::G_USUBE => (dst: Writable<Reg> => Def, flag: Writable<Reg> => Def, lhs: Reg => Use, rhs: Reg => Use, carry_in: Reg => Use);
            build_ssube => GenericOpcode::G_SSUBE => (dst: Writable<Reg> => Def, flag: Writable<Reg> => Def, lhs: Reg => Use, rhs: Reg => Use, carry_in: Reg => Use);
            build_umulo => GenericOpcode::G_UMULO => (dst: Writable<Reg> => Def, flag: Writable<Reg> => Def, lhs: Reg => Use, rhs: Reg => Use);
            build_smulo => GenericOpcode::G_SMULO => (dst: Writable<Reg> => Def, flag: Writable<Reg> => Def, lhs: Reg => Use, rhs: Reg => Use);
        },
        len: one_of(4, 5),
        len_message: "binary-with-flags instruction expects def/def/use/use or def/def/use/use/use operands",
        fields: {
            dst: Reg = expect_def_reg(0, "binary-with-flags operand 0 must be a def"),
            flag: Reg = expect_def_reg(1, "binary-with-flags operand 1 must be a def"),
            lhs: Reg = expect_use_reg(2, "binary-with-flags operand 2 must be a use"),
            rhs: Reg = expect_use_reg(3, "binary-with-flags operand 3 must be a use"),
            carry_in: Option<Reg> = expect_optional_use_reg(4, "binary-with-flags operand 4 must be a use"),
        }
    }
    Load => LoadInst / as_load {
        opcodes: [GenericOpcode::G_LOAD],
        builders: {
            build_load => GenericOpcode::G_LOAD => (
                def: Writable<Reg> => Def,
                ptr: Reg => Use
            );
        },
        len: exact(2),
        len_message: "load expects def/use operands",
        fields: {
            dst: Reg = expect_def_reg(0, "load operand 0 must be a def"),
            base: Reg = expect_use_reg(1, "load operand 1 must be a base register"),
        }
    }
    LoadOffset => LoadOffsetInst / as_load_offset {
        opcodes: [GenericOpcode::G_OFFSET_LOAD],
        builders: {
            build_load_offset => GenericOpcode::G_OFFSET_LOAD => (
                def: Writable<Reg> => Def,
                ptr: Reg => Use,
                offset: i64 => Imm
            );
        },
        len: exact(3),
        len_message: "load offset expects def/use/imm operands",
        fields: {
            dst: Reg = expect_def_reg(0, "load offset operand 0 must be a def"),
            base: Reg = expect_use_reg(1, "load offset operand 1 must be a base register"),
            offset: i64 = expect_imm(2, "load offset operand 2 must be an immediate offset"),
        }
    }
    IndexedLoad => IndexedLoadInst / as_indexed_load {
        opcodes: [GenericOpcode::G_INDEXED_LOAD],
        builders: {
            build_indexed_load => GenericOpcode::G_INDEXED_LOAD => (
                def: Writable<Reg> => Def,
                writeback_ptr: Writable<Reg> => TiedDefUse,
                ptr: Reg => Use,
                offset: i64 => Imm
            );
        },
        len: exact(4),
        len_message: "indexed load expects def/tied-def/use/imm operands",
        fields: {
            dst: Reg = expect_def_reg(0, "indexed load operand 0 must be a def"),
            wb_dst: Reg = expect_tied_def_reg(1, "indexed load operand 1 must be a tied def"),
            base: Reg = expect_use_reg(2, "indexed load operand 2 must be a base register"),
            offset: i64 = expect_imm(3, "indexed load operand 3 must be an immediate offset"),
        }
    }
    Store => StoreInst / as_store {
        opcodes: [GenericOpcode::G_STORE],
        builders: {
            build_store => GenericOpcode::G_STORE => (
                src: Reg => Use,
                ptr: Reg => Use
            );
        },
        len: exact(2),
        len_message: "store expects use/use operands",
        fields: {
            src: Reg = expect_use_reg(0, "store value register not found"),
            base: Reg = expect_use_reg(1, "store base register not found"),
        }
    }
    StackLoad => StackLoadInst / as_stack_load {
        opcodes: [GenericOpcode::G_STACK_LOAD],
        builders: {
            build_stack_load => GenericOpcode::G_STACK_LOAD => (
                def: Writable<Reg> => Def,
                slot: StackSlot => StackSlot
            );
        },
        len: exact(2),
        len_message: "stack load expects def/stackslot operands",
        fields: {
            dst: Reg = expect_def_reg(0, "stack load operand 0 must be a def"),
            slot: StackSlot = expect_stackslot(1, "stack load operand 1 must be a stack slot"),
        }
    }
    StackStore => StackStoreInst / as_stack_store {
        opcodes: [GenericOpcode::G_STACK_STORE],
        builders: {
            build_stack_store => GenericOpcode::G_STACK_STORE => (
                src: Reg => Use,
                slot: StackSlot => StackSlot
            );
        },
        len: exact(2),
        len_message: "stack store expects use/stackslot operands",
        fields: {
            src: Reg = expect_use_reg(0, "stack store operand 0 must be a value register"),
            slot: StackSlot = expect_stackslot(1, "stack store operand 1 must be a stack slot"),
        }
    }
    StoreOffset => StoreOffsetInst / as_store_offset {
        opcodes: [GenericOpcode::G_OFFSET_STORE],
        builders: {
            build_store_offset => GenericOpcode::G_OFFSET_STORE => (
                src: Reg => Use,
                ptr: Reg => Use,
                offset: i64 => Imm
            );
        },
        len: exact(3),
        len_message: "store offset expects use/use/imm operands",
        fields: {
            src: Reg = expect_use_reg(0, "store offset operand 0 must be a value register"),
            base: Reg = expect_use_reg(1, "store offset operand 1 must be a base register"),
            offset: i64 = expect_imm(2, "store offset operand 2 must be an immediate offset"),
        }
    }
    IndexedStore => IndexedStoreInst / as_indexed_store {
        opcodes: [GenericOpcode::G_INDEXED_STORE],
        builders: {
            build_indexed_store => GenericOpcode::G_INDEXED_STORE => (
                writeback_ptr: Writable<Reg> => TiedDefUse,
                src: Reg => Use,
                ptr: Reg => Use,
                offset: i64 => Imm
            );
        },
        len: exact(4),
        len_message: "indexed store expects tied-def/use/use/imm operands",
        fields: {
            wb_dst: Reg = expect_tied_def_reg(0, "indexed store operand 0 must be a tied def"),
            src: Reg = expect_use_reg(1, "indexed store operand 1 must be a value register"),
            base: Reg = expect_use_reg(2, "indexed store operand 2 must be a base register"),
            offset: i64 = expect_imm(3, "indexed store operand 3 must be an immediate offset"),
        }
    }
    Constant => ConstantInst / as_constant {
        opcodes: [GenericOpcode::G_CONSTANT],
        builders: {
            build_constant => GenericOpcode::G_CONSTANT => (
                def: Writable<Reg> => Def,
                imm: i64 => Imm
            );
        },
        len: exact(2),
        len_message: "constant expects def/imm operands",
        fields: {
            dst: Reg = expect_def_reg(0, "constant operand 0 must be a def"),
            imm: i64 = expect_imm(1, "constant operand 1 must be an immediate"),
        }
    }
    FloatConstant => FConstantInst / as_fconstant {
        opcodes: [GenericOpcode::G_FCONSTANT],
        builders: {
            build_fconstant => GenericOpcode::G_FCONSTANT => (
                def: Writable<Reg> => Def,
                fimm: f64 => FImm
            );
        },
        len: exact(2),
        len_message: "fconstant expects def/fimm operands",
        fields: {
            dst: Reg = expect_def_reg(0, "fconstant operand 0 must be a def"),
            imm: f64 = expect_fimm(1, "fconstant operand 1 must be a floating immediate"),
        }
    }
    Branch => BranchInst / as_branch {
        opcodes: [GenericOpcode::G_BR],
        builders: {
            build_br => GenericOpcode::G_BR => (
                target: Block => Block
            );
        },
        len: exact(1),
        len_message: "branch expects one block operand",
        fields: {
            target: Block = expect_block(0, "branch operand 0 must be a block"),
        }
    }
    BranchCond => BranchCondInst / as_branch_cond {
        opcodes: [GenericOpcode::G_BRCOND],
        builders: {
            build_br_cond => GenericOpcode::G_BRCOND => (
                cond: Reg => Use,
                then_blk: Block => Block,
                else_blk: Block => Block
            );
        },
        len: exact(3),
        len_message: "conditional branch expects use/block/block operands",
        fields: {
            cond: Reg = expect_use_reg(0, "branch operand 0 must be a condition register"),
            then_blk: Block = expect_block(1, "branch operand 1 must be the then block"),
            else_blk: Block = expect_block(2, "branch operand 2 must be the else block"),
        }
    }
    BranchTable => BranchTableInst / as_branch_table {
        opcodes: [GenericOpcode::G_BRJT],
        builders: {
            build_br_jt => GenericOpcode::G_BRJT => (
                index: Reg => Use
            );
        },
        len: exact(1),
        len_message: "branch table expects one index register operand",
        fields: {
            index: Reg = expect_use_reg(0, "branch table operand 0 must be an index register"),
        }
    }
    Select => SelectInst / as_select {
        opcodes: [GenericOpcode::G_SELECT],
        builders: {
            build_select => GenericOpcode::G_SELECT => (
                def: Writable<Reg> => Def,
                cond: Reg => Use,
                v1: Reg => Use,
                v2: Reg => Use
            );
        },
        len: exact(4),
        len_message: "select expects def/use/use/use operands",
        fields: {
            dst: Reg = expect_def_reg(0, "select operand 0 must be a def"),
            cond: Reg = expect_use_reg(1, "select operand 1 must be a condition"),
            v1: Reg = expect_use_reg(2, "select operand 2 must be a use"),
            v2: Reg = expect_use_reg(3, "select operand 3 must be a use"),
        }
    }
    ICmp => ICmpInst / as_icmp {
        opcodes: [GenericOpcode::G_ICMP],
        builders: {
            build_icmp => GenericOpcode::G_ICMP => (
                def: Writable<Reg> => Def,
                src0: Reg => Use,
                src1: Reg => Use,
                cc: IntCC => IntCC
            );
        },
        len: one_of(2, 4),
        len_message: "icmp expects def/use or def/use/use/condcode operands",
        fields: {
            dst: Reg = expect_def_reg(0, "icmp operand 0 must be a def"),
            lhs: Reg = expect_use_reg(1, "icmp operand 1 must be a use"),
            rhs: Option<Reg> = expect_optional_use_reg(2, "icmp operand 2 must be a use"),
            cc: Option<IntCC> = expect_optional_intcc(3, "icmp operand 3 must be an integer condition code"),
        }
    }
    FCmp => FCmpInst / as_fcmp {
        opcodes: [GenericOpcode::G_FCMP],
        builders: {
            build_fcmp => GenericOpcode::G_FCMP => (
                def: Writable<Reg> => Def,
                src0: Reg => Use,
                src1: Reg => Use,
                cc: FloatCC => FloatCC
            );
        },
        len: exact(4),
        len_message: "fcmp expects def/use/use/condcode operands",
        fields: {
            dst: Reg = expect_def_reg(0, "fcmp operand 0 must be a def"),
            lhs: Reg = expect_use_reg(1, "fcmp operand 1 must be a use"),
            rhs: Reg = expect_use_reg(2, "fcmp operand 2 must be a use"),
            cc: FloatCC = expect_floatcc(3, "fcmp operand 3 must be a float condition code"),
        }
    }
    Arg => ArgInst / as_arg {
        opcodes: [GenericOpcode::G_ARG],
        builders: {
            build_arg => GenericOpcode::G_ARG => (
                def: Writable<Reg> => Def,
                index: i64 => Imm
            );
        },
        len: exact(2),
        len_message: "arg expects def/imm operands",
        fields: {
            dst: Reg = expect_def_reg(0, "arg operand 0 must be a def"),
            index: usize = expect_nonnegative_imm_usize(1, "arg operand 1 must be a non-negative immediate index"),
        }
    }
    Return => RetInst / as_ret {
        opcodes: [GenericOpcode::G_RET],
        builders: {
            /// 构建返回指令
            pub fn build_ret(results: SmallVec<[Reg; 2]>) -> Self {
                let mut operands = SmallVec::new();
                for res in results {
                    operands.push(MachineOperand::Use(res));
                }
                Self {
                    opcode: MachineOpcode::Generic(GenericOpcode::G_RET),
                    operands,
                }
            }
        },
        len: any(),
        len_message: "return accepts zero or more use operands",
        fields: {
            values: SmallVec<[Reg; 2]> = collect_use_regs_from(0, "return operands must all be use registers"),
        }
    }
    Unreachable => UnreachableInst / as_unreachable {
        opcodes: [GenericOpcode::G_UNREACHABLE],
        builders: {
            build_unreachable => GenericOpcode::G_UNREACHABLE => ();
        },
        len: exact(0),
        len_message: "unreachable expects no operands",
        fields: {}
    }
    Call => CallShapeInst / as_call_shape_data {
        opcodes: [GenericOpcode::G_CALL, GenericOpcode::G_CALLIND],
        builders: {
            /// 构建直接调用指令。
            pub fn build_call<D, A>(defs: D, callee: SymbolId, args: A) -> Self
            where
                D: IntoIterator<Item = Writable<Reg>>,
                A: IntoIterator<Item = Reg>,
            {
                let mut operands = SmallVec::new();
                for def in defs {
                    operands.push(MachineOperand::Def(def));
                }
                operands.push(MachineOperand::Global(callee));
                for arg in args {
                    operands.push(MachineOperand::Use(arg));
                }
                Self {
                    opcode: MachineOpcode::Generic(GenericOpcode::G_CALL),
                    operands,
                }
            }

            /// 构建间接调用指令。
            pub fn build_call_indirect<D, A>(defs: D, callee: Reg, args: A) -> Self
            where
                D: IntoIterator<Item = Writable<Reg>>,
                A: IntoIterator<Item = Reg>,
            {
                let mut operands = SmallVec::new();
                for def in defs {
                    operands.push(MachineOperand::Def(def));
                }
                operands.push(MachineOperand::Use(callee));
                for arg in args {
                    operands.push(MachineOperand::Use(arg));
                }
                Self {
                    opcode: MachineOpcode::Generic(GenericOpcode::G_CALLIND),
                    operands,
                }
            }
        },
        len: any(),
        len_message: "call accepts defs + callee + args",
        fields: {
            shape: CallShape = decode_call_shape_field(0, "call operands are malformed"),
        }
    }
}

/// 机器指令
#[derive(Debug, Clone)]
pub struct MachineInst {
    pub opcode: MachineOpcode,
    pub operands: SmallVec<[MachineOperand; 4]>,
}

impl MachineInst {
    /// 构建复杂指令或变长参数指令
    pub fn build_generic(opcode: MachineOpcode, operands: SmallVec<[MachineOperand; 4]>) -> Self {
        // 验证操作数顺序：Def 必须在 Use 之前
        let mut seen_non_def = false;
        for op in &operands {
            if op.is_def() {
                if seen_non_def {
                    panic!(
                        "Invalid MachineInst: all Def operands must come before Use operands. Opcode: {:?}",
                        opcode
                    );
                }
            } else {
                seen_non_def = true;
            }
        }
        Self { opcode, operands }
    }

    /// 创建一个无效指令占位符
    pub fn invalid() -> Self {
        Self {
            opcode: MachineOpcode::Invalid,
            operands: SmallVec::new(),
        }
    }

    /// 检查指令是否有效
    pub fn is_invalid(&self) -> bool {
        matches!(self.opcode, MachineOpcode::Invalid)
    }

    /// 获取所有定义的结果寄存器
    pub fn defs(&self) -> impl Iterator<Item = Reg> + '_ {
        self.operands.iter().filter_map(|op| match op {
            MachineOperand::Def(w) | MachineOperand::TiedDefUse(w) => Some(w.to_reg()),
            _ => None,
        })
    }

    /// 获取所有使用的寄存器
    pub fn uses(&self) -> impl Iterator<Item = Reg> + '_ {
        self.operands.iter().filter_map(|op| match op {
            MachineOperand::Use(r) => Some(*r),
            MachineOperand::TiedDefUse(w) => Some(w.to_reg()),
            _ => None,
        })
    }

    /// 检查是否是通用操作码（尚未指令选择）
    pub fn is_generic(&self) -> bool {
        matches!(self.opcode, MachineOpcode::Generic(_))
    }

    /// 检查是否是目标特定操作码
    pub fn is_target(&self) -> bool {
        matches!(self.opcode, MachineOpcode::Target(_))
    }

    /// 如果该指令是通用 LIR 指令，返回其 GenericOpcode。
    pub fn generic_opcode(&self) -> Option<GenericOpcode> {
        match self.opcode {
            MachineOpcode::Generic(opcode) => Some(opcode),
            _ => None,
        }
    }

    /// 返回该通用 LIR 指令对应的 schema。
    pub fn generic_schema(&self) -> Option<GenericInstSchema> {
        self.generic_opcode()
            .and_then(GenericInstSchema::for_opcode)
    }

    /// 按 schema 解码通用 LIR 指令。
    pub fn decode_generic(&self) -> crate::error::Result<DecodedGenericInst> {
        match self.generic_schema() {
            Some(schema) => decode_simple_generic(self, schema),
            None => Err(self.decode_error("no registered schema for opcode")),
        }
    }

    pub fn as_call_shape(&self) -> CallShape {
        self.as_call_shape_data()
            .unwrap_or_else(|err| panic!("{}", err))
            .shape
    }

    fn expect_schema(&self, expected: GenericInstSchema) -> crate::error::Result<()> {
        match self.generic_schema() {
            Some(actual) if actual == expected => Ok(()),
            Some(actual) => Err(self.decode_error_owned(alloc::format!(
                "schema mismatch: expected {:?}, got {:?}",
                expected,
                actual
            ))),
            None => Err(self.decode_error_owned(alloc::format!(
                "opcode {:?} does not have a registered schema",
                self.opcode
            ))),
        }
    }

    fn expect_len(&self, expected: usize, message: &str) -> crate::error::Result<()> {
        if self.operands.len() == expected {
            Ok(())
        } else {
            Err(self.decode_error(message))
        }
    }

    fn expect_def_reg(&self, index: usize, message: &str) -> crate::error::Result<Reg> {
        match self.operands.get(index) {
            Some(MachineOperand::Def(w)) => Ok(w.to_reg()),
            _ => Err(self.decode_error(message)),
        }
    }

    fn expect_use_reg(&self, index: usize, message: &str) -> crate::error::Result<Reg> {
        match self.operands.get(index) {
            Some(MachineOperand::Use(r)) => Ok(*r),
            _ => Err(self.decode_error(message)),
        }
    }

    fn expect_optional_use_reg(
        &self,
        index: usize,
        message: &str,
    ) -> crate::error::Result<Option<Reg>> {
        match self.operands.get(index) {
            None => Ok(None),
            Some(MachineOperand::Use(reg)) => Ok(Some(*reg)),
            Some(_) => Err(self.decode_error(message)),
        }
    }

    fn expect_imm(&self, index: usize, message: &str) -> crate::error::Result<i64> {
        match self.operands.get(index) {
            Some(MachineOperand::Imm(imm)) => Ok(*imm),
            _ => Err(self.decode_error(message)),
        }
    }

    fn expect_nonnegative_imm_usize(
        &self,
        index: usize,
        message: &str,
    ) -> crate::error::Result<usize> {
        match self.operands.get(index) {
            Some(MachineOperand::Imm(imm)) if *imm >= 0 => Ok(*imm as usize),
            _ => Err(self.decode_error(message)),
        }
    }

    fn expect_fimm(&self, index: usize, message: &str) -> crate::error::Result<f64> {
        match self.operands.get(index) {
            Some(MachineOperand::FImm(imm)) => Ok(*imm),
            _ => Err(self.decode_error(message)),
        }
    }

    fn expect_tied_def_reg(&self, index: usize, message: &str) -> crate::error::Result<Reg> {
        match self.operands.get(index) {
            Some(MachineOperand::TiedDefUse(w)) => Ok(w.to_reg()),
            _ => Err(self.decode_error(message)),
        }
    }

    fn expect_stackslot(&self, index: usize, message: &str) -> crate::error::Result<StackSlot> {
        match self.operands.get(index) {
            Some(MachineOperand::StackSlot(slot)) => Ok(*slot),
            _ => Err(self.decode_error(message)),
        }
    }

    fn expect_block(&self, index: usize, message: &str) -> crate::error::Result<Block> {
        match self.operands.get(index) {
            Some(MachineOperand::Block(block)) => Ok(*block),
            _ => Err(self.decode_error(message)),
        }
    }

    fn expect_optional_intcc(
        &self,
        index: usize,
        message: &str,
    ) -> crate::error::Result<Option<IntCC>> {
        match self.operands.get(index) {
            None => Ok(None),
            Some(MachineOperand::CondCode(CondCode::Int(cc))) => Ok(Some(*cc)),
            Some(_) => Err(self.decode_error(message)),
        }
    }

    fn expect_floatcc(&self, index: usize, message: &str) -> crate::error::Result<FloatCC> {
        match self.operands.get(index) {
            Some(MachineOperand::CondCode(CondCode::Float(cc))) => Ok(*cc),
            _ => Err(self.decode_error(message)),
        }
    }

    fn collect_use_regs_from(
        &self,
        index: usize,
        message: &str,
    ) -> crate::error::Result<SmallVec<[Reg; 2]>> {
        let mut regs = SmallVec::new();
        for operand in &self.operands[index..] {
            match operand {
                MachineOperand::Use(reg) => regs.push(*reg),
                _ => return Err(self.decode_error(message)),
            }
        }
        Ok(regs)
    }

    fn decode_call_shape_field(
        &self,
        _index: usize,
        _message: &str,
    ) -> crate::error::Result<CallShape> {
        let mut defs = SmallVec::<[Reg; 2]>::new();
        let mut index = 0;
        while let Some(MachineOperand::Def(w)) = self.operands.get(index) {
            defs.push(w.to_reg());
            index += 1;
        }

        let callee = match self.generic_opcode() {
            Some(GenericOpcode::G_CALL) => match self.operands.get(index) {
                Some(MachineOperand::Global(sym)) => {
                    index += 1;
                    CallCallee::Direct(*sym)
                }
                _ => {
                    return Err(
                        self.decode_error("direct call expects a global callee after def operands")
                    );
                }
            },
            Some(GenericOpcode::G_CALLIND) => match self.operands.get(index) {
                Some(MachineOperand::Use(reg)) => {
                    index += 1;
                    CallCallee::Indirect(*reg)
                }
                _ => {
                    return Err(
                        self.decode_error("indirect call expects a callee register after defs")
                    );
                }
            },
            _ => return Err(self.decode_error("call decoder received a non-call opcode")),
        };

        let mut args = SmallVec::<[Reg; 4]>::new();
        for operand in &self.operands[index..] {
            match operand {
                MachineOperand::Use(reg) => args.push(*reg),
                _ => return Err(self.decode_error("call arguments must be use operands")),
            }
        }
        Ok(CallShape { defs, callee, args })
    }

    fn decode_error(&self, message: &str) -> crate::error::Error {
        self.decode_error_owned(message.into())
    }

    fn decode_error_owned(&self, message: String) -> crate::error::Error {
        crate::error::Error::select(self.opcode.clone(), message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_constant_uses_registered_schema() {
        let inst = MachineInst::build_constant(Writable(Reg::new_vreg(0)), 42);
        assert_eq!(inst.generic_schema(), Some(GenericInstSchema::Constant));
        assert_eq!(
            inst.as_constant().unwrap(),
            ConstantInst {
                dst: Reg::new_vreg(0),
                imm: 42,
            }
        );
    }

    #[test]
    fn decode_load_normalizes_simple() {
        let inst = MachineInst::build_load(Writable(Reg::new_vreg(0)), Reg::new_vreg(1));
        assert_eq!(
            inst.as_load().unwrap(),
            LoadInst {
                dst: Reg::new_vreg(0),
                base: Reg::new_vreg(1),
            }
        );
    }

    #[test]
    fn decode_binary_and_unary_forms() {
        let binary = MachineInst::build_binary(
            MachineOpcode::Generic(GenericOpcode::G_ADD),
            Writable(Reg::new_vreg(0)),
            Reg::new_vreg(1),
            Reg::new_vreg(2),
        );
        let unary = MachineInst::build_copy(Writable(Reg::new_vreg(3)), Reg::new_vreg(4));

        assert_eq!(
            binary.as_binary_reg().unwrap(),
            BinaryRegInst {
                dst: Reg::new_vreg(0),
                lhs: Reg::new_vreg(1),
                rhs: Reg::new_vreg(2),
            }
        );
        assert_eq!(
            unary.as_unary_reg().unwrap(),
            UnaryRegInst {
                dst: Reg::new_vreg(3),
                src: Reg::new_vreg(4),
            }
        );
    }

    #[test]
    fn decode_icmp_and_fcmp_use_typed_condition_codes() {
        let unary = MachineInst::build_unary(
            MachineOpcode::Generic(GenericOpcode::G_ICMP),
            Writable(Reg::new_vreg(0)),
            Reg::new_vreg(1),
        );
        let binary = MachineInst::build_icmp(
            Writable(Reg::new_vreg(2)),
            Reg::new_vreg(3),
            Reg::new_vreg(4),
            IntCC::Eq,
        );
        let fcmp = MachineInst::build_fcmp(
            Writable(Reg::new_vreg(5)),
            Reg::new_vreg(6),
            Reg::new_vreg(7),
            FloatCC::Lt,
        );

        assert_eq!(
            unary.as_icmp().unwrap(),
            ICmpInst {
                dst: Reg::new_vreg(0),
                lhs: Reg::new_vreg(1),
                rhs: None,
                cc: None,
            }
        );
        assert_eq!(
            binary.as_icmp().unwrap(),
            ICmpInst {
                dst: Reg::new_vreg(2),
                lhs: Reg::new_vreg(3),
                rhs: Some(Reg::new_vreg(4)),
                cc: Some(IntCC::Eq),
            }
        );
        assert_eq!(
            fcmp.as_fcmp().unwrap(),
            FCmpInst {
                dst: Reg::new_vreg(5),
                lhs: Reg::new_vreg(6),
                rhs: Reg::new_vreg(7),
                cc: FloatCC::Lt,
            }
        );
    }

    #[test]
    fn decode_select_and_branch_forms() {
        let select = MachineInst::build_select(
            Writable(Reg::new_vreg(0)),
            Reg::new_vreg(1),
            Reg::new_vreg(2),
            Reg::new_vreg(3),
        );
        let br = MachineInst::build_br(Block::from_u32(7));
        let br_cond =
            MachineInst::build_br_cond(Reg::new_vreg(4), Block::from_u32(8), Block::from_u32(9));

        assert_eq!(
            select.as_select().unwrap(),
            SelectInst {
                dst: Reg::new_vreg(0),
                cond: Reg::new_vreg(1),
                v1: Reg::new_vreg(2),
                v2: Reg::new_vreg(3),
            }
        );
        assert_eq!(
            br.as_branch().unwrap(),
            BranchInst {
                target: Block::from_u32(7),
            }
        );
        assert_eq!(
            br_cond.as_branch_cond().unwrap(),
            BranchCondInst {
                cond: Reg::new_vreg(4),
                then_blk: Block::from_u32(8),
                else_blk: Block::from_u32(9),
            }
        );
    }

    #[test]
    fn decode_call_shapes() {
        let direct = MachineInst::build_call(
            [Writable(Reg::new_vreg(0))],
            SymbolId::from_u32(3),
            [Reg::new_vreg(1), Reg::new_vreg(2)],
        );
        let indirect = MachineInst::build_call_indirect(
            [Writable(Reg::new_vreg(4))],
            Reg::new_vreg(5),
            [Reg::new_vreg(6)],
        );

        assert_eq!(
            direct.as_call_shape(),
            CallShape {
                defs: smallvec::smallvec![Reg::new_vreg(0)],
                callee: CallCallee::Direct(SymbolId::from_u32(3)),
                args: smallvec::smallvec![Reg::new_vreg(1), Reg::new_vreg(2)],
            }
        );
        assert_eq!(
            indirect.as_call_shape(),
            CallShape {
                defs: smallvec::smallvec![Reg::new_vreg(4)],
                callee: CallCallee::Indirect(Reg::new_vreg(5)),
                args: smallvec::smallvec![Reg::new_vreg(6)],
            }
        );
    }

    #[test]
    fn decode_arg_ret_fconstant_and_unreachable() {
        let arg = MachineInst::build_arg(Writable(Reg::new_vreg(0)), 3);
        let ret = MachineInst::build_ret(smallvec::smallvec![Reg::new_vreg(1), Reg::new_vreg(2)]);
        let fconst = MachineInst::build_fconstant(Writable(Reg::new_vreg(3)), 1.5);
        let unreachable = MachineInst::build_unreachable();

        assert_eq!(arg.generic_schema(), Some(GenericInstSchema::Arg));
        assert_eq!(
            arg.as_arg().unwrap(),
            ArgInst {
                dst: Reg::new_vreg(0),
                index: 3,
            }
        );

        assert_eq!(ret.generic_schema(), Some(GenericInstSchema::Return));
        assert_eq!(
            ret.as_ret().unwrap(),
            RetInst {
                values: smallvec::smallvec![Reg::new_vreg(1), Reg::new_vreg(2)],
            }
        );

        assert_eq!(
            fconst.generic_schema(),
            Some(GenericInstSchema::FloatConstant)
        );
        assert_eq!(
            fconst.as_fconstant().unwrap(),
            FConstantInst {
                dst: Reg::new_vreg(3),
                imm: 1.5,
            }
        );

        assert_eq!(
            unreachable.generic_schema(),
            Some(GenericInstSchema::Unreachable)
        );
        assert_eq!(unreachable.as_unreachable().unwrap(), UnreachableInst {});
    }
}
