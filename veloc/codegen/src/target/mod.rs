//! Target Architecture Abstraction
//!
//! 提供目标架构的抽象接口，支持多后端（x86_64, ARM, RISC-V 等）

pub mod riscv64;
pub mod x86_64;

mod abi;
mod callconv;
mod features;
mod types;

use crate::Emitter;
pub use crate::passes::lowering::RewriteContext;
use crate::pipeline::{FunctionPass, ModuleCodegenPass};
use std::boxed::Box;
use std::vec::Vec;
pub use veloc_lir::{InstId, MachineFunction, Reg, VReg};
use veloc_mir::Type;

pub use abi::{AbiAssignment, AbiDescriptor, AbiLocation, AbiPlan, AbiState, StackArea};
pub use callconv::CallConv;
pub use features::FeatureSetRef;
pub use types::{
    RegClass, RegClassInfo, RegInfo, RegisterFile, RegisterView, RegisterWrite, SpecialRegs,
    TargetArch, TargetConfig, TargetDescription,
};

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
        function: &MachineFunction,
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
        slot: veloc_lir::StackSlot,
        ty: Type,
    ) -> crate::Result<InstId>;
}

/// Backend composition root. Consumers accept narrower supertraits.
pub trait TargetMachine: TargetRegalloc + TargetSchedule {
    /// 获取架构配置
    fn config(&self) -> &TargetConfig;

    /// Immutable legalization rules and target capabilities.
    fn legalizer(&self) -> crate::passes::lowering::legalize::LegalizePolicy<'_>;

    /// Immutable selection rules and explicit host extensions.
    fn selector(&self) -> crate::isel::SelectPolicy<'_>;

    /// 获取 post-isel 组件。
    fn post_isel(&self) -> &dyn TargetPostIsel;

    /// 获取栈帧和序言/尾声 lowering 组件。
    fn frame_lowering(&self) -> &dyn TargetFrameLowering;

    /// 获取 target-specific pipeline 配置。
    fn pass_config(&self) -> &dyn TargetPassConfig;

    /// 获取汇编器/发射器
    fn emitter(&self) -> &dyn crate::target::TargetEmitter;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewriteResult {
    Keep,
    InPlace,
    Replace,
    Remove,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetInstMetadata {
    pub constraints: &'static [veloc_lir::OperandConstraint],
    /// Fixed access encoded by this instruction; absence is not an effect proof.
    pub memory: Option<(veloc_lir::MemoryKind, u32)>,
    pub flow: veloc_lir::ControlFlow,
    pub schedule: Option<ScheduleInfo>,
    pub implicit_uses: &'static [Reg],
    pub implicit_defs: &'static [Reg],
    pub clobbers: &'static [&'static str],
}

impl TargetInstMetadata {
    pub const EMPTY: Self = Self {
        constraints: &[],
        memory: None,
        flow: veloc_lir::ControlFlow::Next,
        schedule: None,
        implicit_uses: &[],
        implicit_defs: &[],
        clobbers: &[],
    };
}

pub trait TargetPostIsel: Send + Sync {
    /// 指令融合（可选）
    ///
    /// 在指令选择后、寄存器分配前执行。
    /// 默认不做任何处理，目标后端可以按需覆写。
    fn combine_instructions(&self, _mfunc: &mut MachineFunction) {}
}

pub trait TargetFrameLowering: Send + Sync {
    /// Alignment guaranteed by the current prologue; larger ABI requirements
    /// must be rejected until the frame strategy supports realignment.
    fn stack_alignment(&self) -> u32;

    /// 完成目标相关的栈帧布局。
    ///
    /// 在寄存器分配之后、插入序言/尾声之前调用，用于计算 callee-saved 保存区、
    /// 最终栈大小和 ABI 对齐等目标相关信息。
    fn finalize_stack_frame(
        &self,
        mfunc: &mut MachineFunction,
        call_conv: CallConv,
    ) -> crate::Result<()>;

    /// 插入函数序言和尾声 (Prologue/Epilogue Insertion)
    /// 在寄存器分配之后调用，将序言/尾声指令插入到 LIR 中。
    fn insert_prologue_epilogue(&self, mfunc: &mut MachineFunction);
}

pub trait TargetPassConfig: Send + Sync {
    /// Target preparation before legalization; may introduce generic operations.
    fn prepare_passes(&self) -> Vec<Box<dyn FunctionPass>> {
        Vec::new()
    }

    /// Selection preparation. Must preserve ABI lowering and target legality.
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
    fn pre_emit_module_passes(
        &self,
    ) -> Vec<Box<dyn ModuleCodegenPass<crate::pipeline::CompiledModule>>> {
        Vec::new()
    }

    /// 在符号化发射后、最终布局和重定位之前运行模块级 passes。
    /// 此时仍可修改编码片段；最终布局之后不能再改变代码大小或顺序。
    fn post_emit_module_passes(
        &self,
    ) -> Vec<Box<dyn ModuleCodegenPass<crate::pipeline::EmissionModule>>> {
        Vec::new()
    }
}
