//! Target Architecture Abstraction
//!
//! 提供目标架构的抽象接口，支持多后端（x86_64, ARM, RISC-V 等）

mod abi;
mod callconv;
mod types;

use crate::Emitter;
pub use crate::passes::lowering::{LegalizeAction, LegalizeResult};
use crate::pipeline::{FunctionPass, ModuleCodegenPass};
use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::vec::Vec;
pub use veloc_lir::ValueId;
use veloc_lir::stages::{
    LegalizedLir, PreIselPrepared, PrologueEpilogueInserted, RegAllocated, SelectedLir,
};
pub use veloc_lir::{InstId, MachineFunction, Reg, VReg};
use veloc_mir::{Type, TypeInfo};

pub use abi::{
    AbiAssignment, AbiClassifierEntry, AbiClassifierFn, AbiDescriptor, AbiLocation, AbiPart,
    AbiPreservedSet, AbiRegisterPool, AbiStackBase, AbiStackDescriptor, AbiValueClass,
    CallConvPlan,
};
pub use callconv::CallConv;
pub use types::{
    CpuDescription, RegClass, RegClassInfo, RegInfo, RegisterFile, RegisterView, RegisterWrite,
    SpecialRegs, TargetArch, TargetConfig, TargetDescription,
};

/// 基础 Lowering Context 接口 (所有后端共用)
pub trait LoweringContext {
    /// Create a fresh machine SSA temporary with the exemplar's type and bank.
    fn alloc_tmp(&mut self, like: Reg) -> Reg;
    /// 获取值的类型
    fn get_type(&self, val: VReg) -> Type;

    /// 谓词：检查是否为 i32
    fn is_i32(&self, val: VReg) -> bool {
        self.get_type(val).is_integer()
            && self
                .get_type(val)
                .bit_size()
                .and_then(|size| size.fixed_bits())
                == Some(32)
    }

    /// 谓词：检查是否为 i16
    fn is_i16(&self, val: VReg) -> bool {
        self.get_type(val).is_integer()
            && self
                .get_type(val)
                .bit_size()
                .and_then(|size| size.fixed_bits())
                == Some(16)
    }

    /// 谓词：检查是否为 i8
    fn is_i8(&self, val: VReg) -> bool {
        self.get_type(val).is_integer()
            && self
                .get_type(val)
                .bit_size()
                .and_then(|size| size.fixed_bits())
                == Some(8)
    }

    /// 谓词：检查是否为 i64
    fn is_i64(&self, val: VReg) -> bool {
        self.get_type(val).is_integer()
            && self
                .get_type(val)
                .bit_size()
                .and_then(|size| size.fixed_bits())
                == Some(64)
    }

    /// 谓词：检查是否为 32 位整数宽度的值
    fn is_int32like(&self, val: VReg) -> bool {
        let ty = self.get_type(val);
        ty.is_integer()
            && ty
                .bit_size()
                .and_then(|size| size.fixed_bits())
                .is_some_and(|bits| bits <= 32)
    }

    /// 谓词：检查是否为 64 位整数或指针宽度的值
    fn is_64like(&self, val: VReg) -> bool {
        let ty = self.get_type(val);
        (ty.is_integer() && ty.bit_size().and_then(|size| size.fixed_bits()) == Some(64))
            || ty.is_ptr()
    }

    /// 谓词：检查是否为 bool
    fn is_bool(&self, val: VReg) -> bool {
        self.get_type(val) == Type::BOOL
    }

    /// 谓词：检查是否为 f32
    fn is_f32(&self, val: VReg) -> bool {
        self.get_type(val) == Type::F32
    }

    /// 谓词：检查是否为 f64
    fn is_f64(&self, val: VReg) -> bool {
        self.get_type(val) == Type::F64
    }

    /// 谓词：检查是否为指针
    fn is_ptr(&self, val: VReg) -> bool {
        self.get_type(val).is_ptr()
    }

    /// 获取寄存器库
    fn get_bank(&self, val: VReg) -> Option<veloc_lir::RegisterBank>;

    /// 谓词：检查是否在 FPR (浮点寄存器库)
    fn is_fpr(&self, val: VReg) -> bool {
        matches!(self.get_bank(val), Some(veloc_lir::RegisterBank::FPR))
    }

    /// 获取指定的寄存器操作数
    fn get_vreg(&self, inst: &veloc_lir::InstRef<'_>, index: usize) -> Option<VReg>;
}

/// Target Machine: 封装特定目标架构的所有组件和策略。
/// 模仿 LLVM TargetMachine，作为从通用流程获取架构特定逻辑的统一入口。
/// Operand rendering and symbol naming belong to the assembly host. Instruction
/// mnemonics, widths and operand order come from the target definition schema.
pub trait AssemblyWriter: core::fmt::Write {
    fn register(&mut self, reg: Reg, bits: u32) -> core::fmt::Result;
    fn immediate(&mut self, value: i64) -> core::fmt::Result;
    fn block(&mut self, block: veloc_mir::Block) -> core::fmt::Result;
    fn symbol(&mut self, symbol: veloc_lir::SymbolId) -> core::fmt::Result;
    fn memory(&mut self, base: Reg, offset: i64, bits: u32) -> core::fmt::Result;
    fn stack_slot(&mut self, slot: veloc_lir::StackSlot, bits: u32) -> core::fmt::Result;
}

pub trait TargetMachine {
    fn validate_instruction(
        &self,
        inst: &veloc_lir::InstRef<'_>,
        allocated: bool,
    ) -> crate::Result<()>;
    fn write_assembly(
        &self,
        inst: &veloc_lir::InstRef<'_>,
        out: &mut dyn AssemblyWriter,
    ) -> core::fmt::Result;
    /// Static instruction facts shared by control-flow and scheduling queries.
    fn target_inst_metadata(&self, opcode: u32) -> &'static TargetInstMetadata;

    fn control_flow(&self, inst: &veloc_lir::InstRef<'_>) -> veloc_lir::ControlFlow {
        match inst.opcode() {
            veloc_lir::MachineOpcode::Invalid => veloc_lir::ControlFlow::Next,
            veloc_lir::MachineOpcode::Generic(op) => op.control(),
            veloc_lir::MachineOpcode::Target(op) => self.target_inst_metadata(op).flow,
        }
    }

    /// Unknown operations are scheduling barriers. Costs are estimates, not
    /// cycle-accurate promises for every CPU implementing an ISA.
    fn schedule_info(&self, inst: &veloc_lir::InstRef<'_>) -> Option<ScheduleInfo> {
        if inst.memory().is_some() {
            return None;
        }
        let veloc_lir::MachineOpcode::Target(op) = inst.opcode() else {
            return None;
        };
        self.target_inst_metadata(op).schedule
    }

    fn is_call(&self, inst: &veloc_lir::InstRef<'_>) -> bool {
        self.control_flow(inst) == veloc_lir::ControlFlow::Call
    }

    /// Dedicated spill temporaries must not belong to any allocatable set.
    fn spill_scratch(&self, _class: RegClass) -> &'static [Reg] {
        &[]
    }

    /// A physical register copy for allocation edits, with the value's type.
    fn jump_instruction(
        &self,
        _writer: veloc_lir::InstWriter<'_>,
        _target: veloc_mir::Block,
    ) -> crate::Result<InstId> {
        Err(crate::Error::codegen("target does not support edge jumps"))
    }

    fn copy_instruction(
        &self,
        _writer: veloc_lir::InstWriter<'_>,
        _dst: Reg,
        _src: Reg,
        _ty: Type,
    ) -> crate::Result<InstId> {
        Err(crate::Error::codegen(
            "target does not support allocation copies",
        ))
    }

    fn spill_instruction(
        &self,
        _writer: veloc_lir::InstWriter<'_>,
        _load: bool,
        _reg: Reg,
        _base: Reg,
        _offset: i64,
        _ty: Type,
    ) -> crate::error::Result<InstId> {
        Err(crate::error::Error::codegen(
            "target does not support spill expansion",
        ))
    }

    /// 获取架构配置
    fn config(&self) -> &TargetConfig;

    /// 获取当前 target instance 的完整描述。
    fn desc(&self) -> &TargetDescription;

    /// 获取 legalize 组件。
    fn target_legalizer(&self) -> &dyn TargetLegalizer;

    /// 获取指令选择组件。
    fn target_selector(&self) -> &dyn TargetInstructionSelector;

    /// 获取操作数/寄存器拷贝 lowering 组件。
    fn target_operand_lowering(&self) -> &dyn TargetOperandLowering;

    /// 获取 post-isel 组件。
    fn target_post_isel(&self) -> &dyn TargetPostIsel;

    /// 获取栈帧和序言/尾声 lowering 组件。
    fn target_frame_lowering(&self) -> &dyn TargetFrameLowering;

    /// 获取 target-specific pipeline 配置。
    fn target_pass_config(&self) -> &dyn TargetPassConfig;

    /// 获取汇编器/发射器
    fn target_emitter(&self) -> &dyn crate::target::arch::TargetEmitter;

    /// 获取寄存器库选择逻辑
    fn target_regbank_select(&self) -> &dyn crate::regalloc::regbank_select::TargetRegBankSelect;
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
        _block: &veloc_lir::MachineBlock,
        _mfunc: &MachineFunction<PrologueEpilogueInserted>,
    ) -> Result<(), crate::error::Error> {
        Ok(())
    }

    /// 发射单条指令 (此时指令应该是目标特定的 Opcode)
    fn emit_instruction(
        &self,
        emitter: &mut Emitter,
        inst: &veloc_lir::InstRef<'_>,
        mfunc: &MachineFunction<PrologueEpilogueInserted>,
    ) -> Result<(), crate::error::Error>;

    /// 完成整个函数的发射，例如回填分支位移。
    fn finish_function(
        &self,
        _emitter: &mut Emitter,
        _mfunc: &MachineFunction<PrologueEpilogueInserted>,
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
    /// 查询一条 generic LIR 指令在当前目标上的 legalize 动作。
    fn legalize_action(
        &self,
        _inst: &veloc_lir::InstRef<'_>,
        _mfunc: &MachineFunction<LegalizedLir>,
    ) -> Result<Option<LegalizeAction>, crate::error::Error> {
        Ok(None)
    }

    /// 应用 target-specific legalization。
    ///
    /// 只有当 `legalize_action()` 返回 `Some(LegalizeAction::Lower)` 时，
    /// driver 才会调用这个 hook。
    ///
    /// 返回的指令按执行顺序排列，driver 会继续合法化其中的 generic 指令。
    /// 原地改写时应返回原来的 ID；否则 driver 会使原指令失效。
    /// 可以追加新块，但不能悄悄修改已处理的其他指令：driver 不会回访它们。
    fn legalize_instruction(
        &self,
        inst_id: veloc_lir::InstId,
        mfunc: &mut veloc_lir::MachineFunction<LegalizedLir>,
    ) -> Result<LegalizeResult, crate::error::Error>;
}

pub trait TargetInstructionSelector: Send + Sync {
    /// 选择目标指令
    ///
    /// 返回选择结果，由指令选择驱动器统一处理。
    fn select_instruction(
        &self,
        ctx: &mut SelectionContext<'_, PreIselPrepared>,
    ) -> Result<SelectResult, crate::error::Error>;
}

pub trait TargetOperandLowering: Send + Sync {
    /// 查询一条 pre-isel 指令需要满足的操作数约束。
    ///
    /// 适合处理会在 isel 后丢失语义信息的 destructive/two-address 约束。
    fn preselect_operand_constraints(
        &self,
        _inst: &veloc_lir::InstRef<'_>,
        _mfunc: &MachineFunction<PreIselPrepared>,
    ) -> OperandConstraintSet {
        OperandConstraintSet::default()
    }

    /// 查询一条 selected LIR 指令需要满足的操作数约束。
    ///
    /// 适合处理固定寄存器等 target instruction 级别的约束。
    fn postselect_operand_constraints(
        &self,
        _inst: &veloc_lir::InstRef<'_>,
        _mfunc: &MachineFunction<SelectedLir>,
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
        _mfunc: &mut MachineFunction<PreIselPrepared>,
        _dst: Reg,
        _src: Reg,
    ) -> Result<InstId, crate::error::Error> {
        panic!("target does not support pre-select register copy construction",)
    }

    /// 为 post-isel 约束阶段构造一条目标相关的寄存器拷贝指令。
    fn build_postselect_reg_copy(
        &self,
        _mfunc: &mut MachineFunction<SelectedLir>,
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
    fn combine_instructions(&self, _mfunc: &mut MachineFunction<SelectedLir>) {}
}

pub trait TargetFrameLowering: Send + Sync {
    /// 完成目标相关的栈帧布局。
    ///
    /// 在寄存器分配之后、插入序言/尾声之前调用，用于计算 callee-saved 保存区、
    /// 最终栈大小和 ABI 对齐等目标相关信息。
    fn finalize_stack_frame(
        &self,
        _mfunc: &mut MachineFunction<RegAllocated>,
        _call_conv: CallConv,
    ) {
    }

    /// 插入函数序言和尾声 (Prologue/Epilogue Insertion)
    /// 在寄存器分配之后调用，将序言/尾声指令插入到 LIR 中。
    fn insert_prologue_epilogue(&self, mfunc: &mut MachineFunction<RegAllocated>);
}

pub trait TargetPassConfig: Send + Sync {
    /// 在合法化之后追加 target 自定义 function passes。
    fn post_legalize_passes(&self) -> Vec<Box<dyn FunctionPass<veloc_lir::stages::LegalizedLir>>> {
        Vec::new()
    }

    /// 在 generic combine 之后追加 target 自定义 function passes。
    fn pre_isel_passes(&self) -> Vec<Box<dyn FunctionPass<veloc_lir::stages::PreIselPrepared>>> {
        Vec::new()
    }

    /// 在指令选择之后追加 target 自定义 function passes。
    fn post_isel_passes(&self) -> Vec<Box<dyn FunctionPass<veloc_lir::stages::SelectedLir>>> {
        Vec::new()
    }

    /// 在寄存器分配之后追加 target 自定义 function passes。
    fn post_regalloc_passes(&self) -> Vec<Box<dyn FunctionPass<veloc_lir::stages::RegAllocated>>> {
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
