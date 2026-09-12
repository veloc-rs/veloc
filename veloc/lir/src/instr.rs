//! Low-level IR (LIR) 指令和操作数定义

use alloc::string::String;
use cranelift_entity::entity_impl;
use smallvec::SmallVec;
use veloc_mir::{Block, FloatCC, IntCC, Type};
pub use veloc_types::{MemFlags, MemoryEffect, MemoryEffects, OpTraits};

use crate::symbol::SymbolId;

/// Abstract register bank; target-specific selection belongs to codegen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegisterBank {
    GPR,
    FPR,
    VR,
    PR,
    Special,
}

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
}

/// Control transfer independent of instruction encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlFlow {
    Next,
    /// A conditional transfer with a path continuing at the next instruction.
    Branch,
    /// All paths transfer to explicit successors; no fallthrough.
    Jump,
    Return,
    Call,
    Trap,
}

include!(concat!(env!("OUT_DIR"), "/instructions.rs"));

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallCallee {
    Direct(SymbolId),
    Indirect(Reg),
}

/// Borrowed, structurally checked register operands. Construction is private so
/// iteration never silently skips a malformed operand.
#[derive(Debug, Clone, Copy)]
pub struct RegList<'a>(&'a [MachineOperand]);

impl RegList<'_> {
    pub fn len(self) -> usize {
        self.0.len()
    }

    pub fn is_empty(self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(self) -> impl ExactSizeIterator<Item = Reg> + DoubleEndedIterator {
        self.0.iter().map(|operand| match operand {
            MachineOperand::Def(reg) => reg.to_reg(),
            MachineOperand::Use(reg) => *reg,
            _ => unreachable!("checked register list"),
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CallShape<'a> {
    pub defs: RegList<'a>,
    pub callee: CallCallee,
    pub args: RegList<'a>,
}

// Variable-arity calls/returns and target-independent construction helpers.
impl MachineInst {
    /// 构建一元操作指令
    pub fn build_unary(opcode: MachineOpcode, def: Writable<Reg>, src: Reg) -> Self {
        Self {
            memory: None,
            opcode,
            operands: smallvec::smallvec![MachineOperand::Def(def), MachineOperand::Use(src)],
        }
    }

    /// 构建三地址二元操作指令
    pub fn build_binary(opcode: MachineOpcode, def: Writable<Reg>, src0: Reg, src1: Reg) -> Self {
        Self {
            memory: None,
            opcode,
            operands: smallvec::smallvec![
                MachineOperand::Def(def),
                MachineOperand::Use(src0),
                MachineOperand::Use(src1)
            ],
        }
    }

    /// 构建两地址二元指令 (如 x86 的 add eax, ecx)
    pub fn build_tied_binary(opcode: MachineOpcode, def_use: Writable<Reg>, src: Reg) -> Self {
        Self {
            memory: None,
            opcode,
            operands: smallvec::smallvec![
                MachineOperand::TiedDefUse(def_use),
                MachineOperand::Use(src)
            ],
        }
    }

    /// 构建返回指令
    pub fn build_ret(results: SmallVec<[Reg; 2]>) -> Self {
        let mut operands = SmallVec::new();
        for res in results {
            operands.push(MachineOperand::Use(res));
        }
        Self {
            memory: None,
            opcode: MachineOpcode::Generic(GenericOpcode::G_RET),
            operands,
        }
    }

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
            memory: None,
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
            memory: None,
            opcode: MachineOpcode::Generic(GenericOpcode::G_CALLIND),
            operands,
        }
    }
}

/// 机器指令
#[derive(Debug, Clone)]
pub struct MachineInst {
    pub memory: Option<crate::MemoryAccess>,
    pub opcode: MachineOpcode,
    pub operands: SmallVec<[MachineOperand; 4]>,
}

impl MachineInst {
    pub fn with_memory(mut self, access: crate::MemoryAccess) -> Self {
        self.memory = Some(access);
        self
    }

    /// 构建复杂指令或变长参数指令
    pub fn build_generic(opcode: MachineOpcode, operands: SmallVec<[MachineOperand; 4]>) -> Self {
        // Explicit operand order belongs to the instruction schema. Implicit
        // ABI uses/defs can follow encoded operands and are not encoding fields.
        Self {
            opcode,
            operands,
            memory: None,
        }
    }

    /// 创建一个无效指令占位符
    pub fn invalid() -> Self {
        Self {
            memory: None,
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

    fn expect_intcc(&self, index: usize, message: &str) -> crate::error::Result<IntCC> {
        match self.operands.get(index) {
            Some(MachineOperand::CondCode(CondCode::Int(cc))) => Ok(*cc),
            _ => Err(self.decode_error(message)),
        }
    }

    fn expect_floatcc(&self, index: usize, message: &str) -> crate::error::Result<FloatCC> {
        match self.operands.get(index) {
            Some(MachineOperand::CondCode(CondCode::Float(cc))) => Ok(*cc),
            _ => Err(self.decode_error(message)),
        }
    }

    fn borrow_use_regs_from(
        &self,
        index: usize,
        message: &str,
    ) -> crate::error::Result<RegList<'_>> {
        let operands = self
            .operands
            .get(index..)
            .ok_or_else(|| self.decode_error(message))?;
        if operands
            .iter()
            .any(|op| !matches!(op, MachineOperand::Use(_)))
        {
            return Err(self.decode_error(message));
        }
        Ok(RegList(operands))
    }

    fn decode_call_shape_field(
        &self,
        _index: usize,
        _message: &str,
    ) -> crate::error::Result<CallShape<'_>> {
        let mut index = 0;
        while let Some(MachineOperand::Def(_)) = self.operands.get(index) {
            index += 1;
        }

        let defs = RegList(&self.operands[..index]);
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

        let args = self.borrow_use_regs_from(index, "call arguments must be use operands")?;
        Ok(CallShape { defs, callee, args })
    }

    fn decode_error(&self, message: &str) -> crate::DecodeError {
        self.decode_error_owned(message.into())
    }

    fn decode_error_owned(&self, message: String) -> crate::DecodeError {
        crate::DecodeError {
            opcode: self.opcode.clone(),
            reason: message,
        }
    }
}
