//! Target Architecture Abstraction
//!
//! 提供目标架构的抽象接口，支持多后端（x86_64, ARM, RISC-V 等）

mod abi;
mod callconv;
mod types;

use crate::Emitter;
pub use crate::passes::lowering::{LegalizeAction, RewriteContext};
use crate::pipeline::{FunctionPass, ModuleCodegenPass};
use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::vec::Vec;
pub use veloc_lir::{InstId, MachineFunction, Reg, VReg};
use veloc_mir::Type;

pub use abi::{
    AbiAssignment, AbiClassifierEntry, AbiClassifierFn, AbiDescriptor, AbiLocation, AbiPart,
    AbiPreservedSet, AbiRegisterPool, AbiStackBase, AbiStackDescriptor, AbiValueClass,
    CallConvPlan,
};
pub use callconv::CallConv;
pub use types::{
    RegClass, RegClassInfo, RegInfo, RegisterFile, RegisterView, RegisterWrite, SpecialRegs,
    TargetArch, TargetConfig, TargetDescription,
};

/// 基础 Lowering Context 接口 (所有后端共用)
/// Immutable target capabilities, independent of instruction operands or graph analyses.
pub trait TargetFeatures {
    type Features;
    fn supports_features(&self, required: Self::Features) -> bool;
}

pub trait LoweringContext {
    /// Create a fresh machine SSA temporary with the exemplar's type and bank.
    fn alloc_tmp(&mut self, like: Reg) -> Reg;
    /// 获取值的类型
    fn get_type(&self, val: VReg) -> Type;

    /// 获取指定的寄存器操作数
    fn get_vreg(&self, inst: &veloc_lir::InstRef<'_>, index: usize) -> Option<VReg>;
}

/// Operand rendering and symbol naming belong to the assembly host. Instruction
/// mnemonics, widths and operand order come from the target definition schema.
pub trait AssemblyWriter: core::fmt::Write {
    fn register(&mut self, reg: Reg, bits: u32) -> core::fmt::Result;
    fn immediate(&mut self, value: i64) -> core::fmt::Result;
    fn block(&mut self, block: veloc_lir::BlockId) -> core::fmt::Result;
    fn symbol(&mut self, symbol: veloc_lir::SymbolId) -> core::fmt::Result;
    fn memory(
        &mut self,
        base: Reg,
        index: Option<Reg>,
        offset: i64,
        bits: u32,
    ) -> core::fmt::Result;
    fn stack_slot(&mut self, slot: veloc_lir::StackSlot, bits: u32) -> core::fmt::Result;
}

/// Explicit validator policy; this does not tag or mutate the IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationMode {
    /// Virtual registers are permitted; fixed physical operands are still checked.
    Virtual,
    /// No virtual operands remain, and two-address ties must be satisfied.
    Allocated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpillKind {
    Load,
    Store,
}

/// Immutable target description shared by backend algorithms.
pub trait TargetInfo {
    fn desc(&self) -> &TargetDescription;
}

/// Definition-owned instruction facts and their generated consumers.
pub trait TargetInstructions {
    fn validate_instruction(
        &self,
        inst: &veloc_lir::InstRef<'_>,
        mode: ValidationMode,
    ) -> crate::Result<()>;
    fn write_assembly(
        &self,
        inst: &veloc_lir::InstRef<'_>,
        out: &mut dyn AssemblyWriter,
    ) -> core::fmt::Result;
    /// Static instruction facts shared by control-flow and scheduling queries.
    fn instruction_metadata(&self, opcode: u32) -> &'static TargetInstMetadata;

    fn control_flow(&self, inst: &veloc_lir::InstRef<'_>) -> veloc_lir::ControlFlow {
        match inst.opcode() {
            veloc_lir::MachineOpcode::Invalid => veloc_lir::ControlFlow::Next,
            veloc_lir::MachineOpcode::Generic(op) => op.control(),
            veloc_lir::MachineOpcode::Target(op) => self.instruction_metadata(op).flow,
        }
    }

    fn is_call(&self, inst: &veloc_lir::InstRef<'_>) -> bool {
        self.control_flow(inst) == veloc_lir::ControlFlow::Call
    }
}

/// CPU-specific estimates, separate from instruction semantics and pass policy.
pub trait TargetSchedule: TargetInfo + TargetInstructions {
    /// CPU latency override; None retains the definition's baseline estimate.
    /// Providing a cost does not establish that the instruction may be moved.
    fn schedule_latency(&self, _opcode: u32) -> Option<u32> {
        None
    }
}

/// Required primitives for the allocation algorithm; no late unsupported defaults.
pub trait TargetRegalloc: TargetInfo + TargetInstructions {
    /// Reserved temporaries must not overlap any allocatable register set.
    fn spill_scratch(&self, class: RegClass) -> &'static [Reg];
    fn jump_instruction(
        &self,
        writer: veloc_lir::InstWriter<'_>,
        target: veloc_lir::BlockId,
    ) -> crate::Result<InstId>;
    fn copy_instruction(
        &self,
        writer: veloc_lir::InstWriter<'_>,
        dst: Reg,
        src: Reg,
        ty: Type,
    ) -> crate::Result<InstId>;
    fn spill_instruction(
        &self,
        writer: veloc_lir::InstWriter<'_>,
        kind: SpillKind,
        reg: Reg,
        base: Reg,
        offset: i64,
        ty: Type,
    ) -> crate::Result<InstId>;
}

/// Backend composition root. Consumers accept narrower supertraits.
pub trait TargetMachine: TargetRegalloc + TargetSchedule {
    /// 获取架构配置
    fn config(&self) -> &TargetConfig;

    /// 获取 legalize 组件。
    fn legalizer(&self) -> &dyn TargetLegalizer;

    /// 获取指令选择组件。
    fn selector(&self) -> &dyn TargetInstructionSelector;

    /// 获取操作数/寄存器拷贝 lowering 组件。
    fn operand_lowering(&self) -> &dyn TargetOperandLowering;

    /// 获取 post-isel 组件。
    fn post_isel(&self) -> &dyn TargetPostIsel;

    /// 获取栈帧和序言/尾声 lowering 组件。
    fn frame_lowering(&self) -> &dyn TargetFrameLowering;

    /// 获取 target-specific pipeline 配置。
    fn pass_config(&self) -> &dyn TargetPassConfig;

    /// 获取汇编器/发射器
    fn emitter(&self) -> &dyn crate::target::arch::TargetEmitter;
}

/// A movable, nontrapping operation. It must not access memory, read flags, or
/// change control flow. The scheduler preserves the region's final flag writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduleInfo {
    pub latency: u32,
    pub writes_flags: bool,
}

/// 机器码发射器接口
pub trait TargetEmitter: Send + Sync {
    /// 开始发射一个基本块。
    fn begin_block(
        &self,
        _emitter: &mut Emitter,
        _block: veloc_lir::BlockId,
        _mfunc: &MachineFunction,
    ) -> Result<(), crate::error::Error> {
        Ok(())
    }

    /// 发射单条指令 (此时指令应该是目标特定的 Opcode)
    fn emit_instruction(
        &self,
        emitter: &mut Emitter,
        inst: &veloc_lir::InstRef<'_>,
        mfunc: &MachineFunction,
    ) -> Result<(), crate::error::Error>;

    /// 完成整个函数的发射，例如回填分支位移。
    fn finish_function(
        &self,
        _emitter: &mut Emitter,
        _mfunc: &MachineFunction,
    ) -> Result<(), crate::error::Error> {
        Ok(())
    }
}

pub use crate::isel::{SelectResult, SelectionContext};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewriteResult {
    Keep,
    InPlace,
    Replace,
    Remove,
}

/// A result and an input that must occupy the same physical register after allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TiedOperandConstraint {
    pub result: usize,
    pub use_operand: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedUseConstraint {
    pub use_operand: usize,
    pub reg: Reg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenericInstMetadata {
    pub fixed_uses: &'static [FixedUseConstraint],
}

/// pre-isel rewrite 规则表项。
///
/// 规则以一个紧凑的 typed IR 存储，运行时不需要再解析字符串。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreIselRewriteExpr {
    Var(u32),
    Imm(i64),
    Op {
        opcode: veloc_lir::GenericOpcode,
        args: &'static [PreIselRewriteExpr],
    },
}

/// pre-isel rewrite 规则表项。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreIselRewriteRuleData {
    pub name: &'static str,
    pub match_expr: PreIselRewriteExpr,
    pub replace_expr: PreIselRewriteExpr,
    pub cost: i64,
    pub priority: i64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OperandConstraintSet {
    pub fixed_uses: Cow<'static, [FixedUseConstraint]>,
}

impl OperandConstraintSet {
    pub fn is_empty(&self) -> bool {
        self.fixed_uses.is_empty()
    }
}

impl GenericInstMetadata {
    pub const EMPTY: Self = Self { fixed_uses: &[] };

    pub fn operand_constraints(&self) -> OperandConstraintSet {
        OperandConstraintSet {
            fixed_uses: self.fixed_uses.into(),
        }
    }
}

/// Target-defined allowed physical registers for one explicit register operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisterConstraint {
    pub result: bool,
    pub operand: usize,
    pub registers: &'static [Reg],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetInstMetadata {
    pub register_constraints: &'static [RegisterConstraint],
    /// Fixed access encoded by this instruction; absence is not an effect proof.
    pub memory: Option<(veloc_lir::MemoryKind, u32)>,
    pub flow: veloc_lir::ControlFlow,
    pub schedule: Option<ScheduleInfo>,
    pub tied_operands: &'static [TiedOperandConstraint],
    pub fixed_uses: &'static [FixedUseConstraint],
    pub implicit_uses: &'static [Reg],
    pub implicit_defs: &'static [Reg],
    pub clobbers: &'static [&'static str],
}

impl TargetInstMetadata {
    pub const EMPTY: Self = Self {
        register_constraints: &[],
        memory: None,
        flow: veloc_lir::ControlFlow::Next,
        schedule: None,
        tied_operands: &[],
        fixed_uses: &[],
        implicit_uses: &[],
        implicit_defs: &[],
        clobbers: &[],
    };

    pub fn operand_constraints(&self) -> OperandConstraintSet {
        OperandConstraintSet {
            fixed_uses: self.fixed_uses.into(),
        }
    }
}

pub trait TargetLegalizer: Send + Sync {
    /// Pure instruction-local query. Missing coverage is an error at the driver,
    /// never an implicit declaration of legality.
    fn legalize_action(
        &self,
        query: &crate::passes::lowering::legalize::Query<'_>,
    ) -> Result<Option<LegalizeAction>, crate::error::Error>;
}

pub trait TargetInstructionSelector: Send + Sync {
    /// 选择目标指令
    ///
    /// 返回选择结果，由指令选择驱动器统一处理。
    fn select_instruction(
        &self,
        ctx: &mut SelectionContext<'_>,
    ) -> Result<SelectResult, crate::error::Error>;
}

pub trait TargetOperandLowering: Send + Sync {
    /// 查询一条 pre-isel 指令需要满足的操作数约束。
    ///
    /// 适合处理会在 isel 后丢失语义信息的 destructive/two-address 约束。
    fn preselect_operand_constraints(
        &self,
        _inst: &veloc_lir::InstRef<'_>,
        _mfunc: &MachineFunction,
    ) -> OperandConstraintSet {
        OperandConstraintSet::default()
    }

    /// 查询一条 selected LIR 指令需要满足的操作数约束。
    ///
    /// 适合处理固定寄存器等 target instruction 级别的约束。
    fn postselect_operand_constraints(
        &self,
        _inst: &veloc_lir::InstRef<'_>,
        _mfunc: &MachineFunction,
    ) -> OperandConstraintSet {
        OperandConstraintSet::default()
    }

    /// 为 pre-isel 约束阶段构造一条目标相关的寄存器拷贝指令。
    ///
    /// 当拷贝两端任一操作数已经绑定到物理寄存器时，调用方应优先使用这条
    /// hook，而不是继续发射通用 `Copy`。这样可以保证位宽/寄存器别名等
    /// 目标相关语义在进入后续阶段前已经明确。
    fn build_preselect_reg_copy(
        &self,
        _mfunc: &mut MachineFunction,
        _dst: Reg,
        _src: Reg,
    ) -> Result<InstId, crate::error::Error> {
        panic!("target does not support pre-select register copy construction",)
    }

    /// 为 post-isel 约束阶段构造一条目标相关的寄存器拷贝指令。
    fn build_postselect_reg_copy(
        &self,
        _mfunc: &mut MachineFunction,
        _dst: Reg,
        _src: Reg,
    ) -> Result<InstId, crate::error::Error> {
        panic!("target does not support post-select register copy construction",)
    }
}

pub trait TargetPostIsel: Send + Sync {
    /// 指令融合（可选）
    ///
    /// 在指令选择后、寄存器分配前执行。
    /// 默认不做任何处理，目标后端可以按需覆写。
    fn combine_instructions(&self, _mfunc: &mut MachineFunction) {}
}

pub trait TargetFrameLowering: Send + Sync {
    /// 完成目标相关的栈帧布局。
    ///
    /// 在寄存器分配之后、插入序言/尾声之前调用，用于计算 callee-saved 保存区、
    /// 最终栈大小和 ABI 对齐等目标相关信息。
    fn finalize_stack_frame(&self, _mfunc: &mut MachineFunction, _call_conv: CallConv) {}

    /// 插入函数序言和尾声 (Prologue/Epilogue Insertion)
    /// 在寄存器分配之后调用，将序言/尾声指令插入到 LIR 中。
    fn insert_prologue_epilogue(&self, mfunc: &mut MachineFunction);
}

pub trait TargetPassConfig: Send + Sync {
    /// 在合法化之后追加 target 自定义 function passes。
    fn post_legalize_passes(&self) -> Vec<Box<dyn FunctionPass>> {
        Vec::new()
    }

    /// Target preparation before instruction selection, after generic combine.
    /// Targets needing bank assignment may install it here; it is not a
    /// prerequisite imposed by the common pipeline. Changes to virtual-register
    /// placement constraints invalidate INST_SEMANTICS analyses.
    fn pre_isel_passes(&self) -> Vec<Box<dyn FunctionPass>> {
        Vec::new()
    }

    /// 在指令选择之后追加 target 自定义 function passes。
    fn post_isel_passes(&self) -> Vec<Box<dyn FunctionPass>> {
        Vec::new()
    }

    /// 在寄存器分配之后追加 target 自定义 function passes。
    fn post_regalloc_passes(&self) -> Vec<Box<dyn FunctionPass>> {
        Vec::new()
    }

    /// 在函数发射前追加 target 自定义模块级 late codegen passes。
    fn pre_emit_module_passes(&self) -> Vec<Box<dyn ModuleCodegenPass>> {
        Vec::new()
    }

    /// 在函数发射后追加 target 自定义模块级 late codegen passes。
    fn post_emit_module_passes(&self) -> Vec<Box<dyn ModuleCodegenPass>> {
        Vec::new()
    }
}
