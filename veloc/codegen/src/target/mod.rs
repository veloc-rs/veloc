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
}

/// CPU-specific estimates, separate from instruction semantics and pass policy.
pub trait TargetSchedule: TargetInfo + TargetInstructions {
    /// Costs never grant permission to move an instruction.
    fn schedule_model(&self) -> &ScheduleModel;
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
    fn selector(&self) -> crate::passes::isel::SelectPolicy<'_>;

    /// 获取栈帧和序言/尾声 lowering 组件。
    fn frame_lowering(&self) -> &dyn TargetFrameLowering;

    /// 获取 target-specific pipeline 配置。
    fn pass_config(&self) -> &dyn TargetPassConfig;

    /// 获取汇编器/发射器
    fn emitter(&self) -> &dyn crate::target::TargetEmitter;
}

/// Dense scheduling class index, shared by all CPU models of one target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduleClassId(pub(crate) u16);

impl ScheduleClassId {
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Dense resource index within one CPU model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceId(pub(crate) u16);

impl ResourceId {
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// CPU cost category, independent of permission to reorder an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleClass {
    Modeled(ScheduleClassId),
    /// No machine cost until this instruction has been expanded.
    Pseudo,
}

/// A scheduling class reserves one resource pool. Classes can share a pool.
/// Occupancy is its initiation interval, not result latency; this coarse model
/// does not yet describe instructions that reserve multiple execution ports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduleCost {
    pub resource: ResourceId,
    pub latency: u32,
    pub occupancy: u32,
}

/// Execution resources belong to the CPU, independently of instruction classes.
#[derive(Debug, Clone, Copy)]
pub struct ScheduleResource {
    /// Diagnostic name; resource lookup uses ResourceId.
    pub name: &'static str,
    pub units: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct ScheduleModel {
    pub issue_width: u32,
    /// Indexed by this CPU model's ResourceId.
    pub resources: &'static [ScheduleResource],
    /// Indexed by the target's ScheduleClassId, identically across CPU models.
    pub classes: &'static [ScheduleCost],
}

impl ScheduleModel {
    /// Spec generation checks every instruction class, including optional ISA
    /// features, so feature overrides cannot introduce an unmodeled operation.
    pub fn cost(&self, class: ScheduleClassId) -> ScheduleCost {
        self.classes[class.index()]
    }
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
pub struct TargetInstMetadata {
    pub constraints: &'static [veloc_lir::OperandConstraint],
    /// Hardware state operands must name these physical units after selection.
    pub state_operands: &'static [veloc_lir::StateOperand],
    pub flow: veloc_lir::ControlFlow,
    pub schedule_class: ScheduleClass,
    /// The target guarantees local reordering is safe when register and state
    /// dependencies and memory/trap order are preserved. Unknown effects remain barriers.
    pub movable: bool,
    /// Destroyed physical storage roots, without defining result values.
    /// Per-call ABI destruction is supplied separately by CallInfo.
    pub clobbers: &'static [Reg],
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
        mfunc: &mut veloc_lir::FuncEditor<'_>,
        call_conv: CallConv,
    ) -> crate::Result<()>;

    /// 插入函数序言和尾声 (Prologue/Epilogue Insertion)
    /// 在寄存器分配之后调用，将序言/尾声指令插入到 LIR 中。
    fn insert_prologue_epilogue(&self, mfunc: &mut veloc_lir::FuncEditor<'_>);
}

/// Build target extensions for the requested optimization policy.
/// Required lowering must be included at every level, including `None`.
pub trait TargetPassConfig: Send + Sync {
    /// Target preparation before legalization; may introduce generic operations.
    fn prepare_passes(&self, _level: crate::OptLevel) -> Vec<Box<dyn FunctionPass>> {
        Vec::new()
    }

    /// Selection preparation. Must preserve ABI lowering and target legality.
    /// Targets needing bank assignment may install it here; it is not a
    /// prerequisite imposed by the common pipeline. Changes to virtual-register
    /// placement constraints invalidate INST_SEMANTICS analyses.
    fn pre_isel_passes(&self, _level: crate::OptLevel) -> Vec<Box<dyn FunctionPass>> {
        Vec::new()
    }

    /// 在指令选择之后追加 target 自定义 function passes。
    fn post_isel_passes(&self, _level: crate::OptLevel) -> Vec<Box<dyn FunctionPass>> {
        Vec::new()
    }

    /// 在寄存器分配之后追加 target 自定义 function passes。
    fn post_regalloc_passes(&self, _level: crate::OptLevel) -> Vec<Box<dyn FunctionPass>> {
        Vec::new()
    }

    /// 在函数发射前追加 target 自定义模块级 late codegen passes。
    fn pre_emit_module_passes(
        &self,
        _level: crate::OptLevel,
    ) -> Vec<Box<dyn ModuleCodegenPass<crate::pipeline::CompiledModule>>> {
        Vec::new()
    }

    /// 在符号化发射后、最终布局和重定位之前运行模块级 passes。
    /// 此时仍可修改编码片段；最终布局之后不能再改变代码大小或顺序。
    fn post_emit_module_passes(
        &self,
        _level: crate::OptLevel,
    ) -> Vec<Box<dyn ModuleCodegenPass<crate::pipeline::EmissionModule>>> {
        Vec::new()
    }
}
