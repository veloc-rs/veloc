//! Target Architecture Abstraction
//!
//! 提供目标架构的抽象接口，支持多后端（x86_64, ARM, RISC-V 等）

mod abi;
mod callconv;
mod types;

pub use crate::mir::ValueId;
pub use crate::mir::{InstId, MachineFunction, MachineInst, Reg, VReg};
pub use crate::passes::lowering::{LegalizeAction, LegalizeResult};
use crate::pipeline::stages::{
    LegalizedMir, PreIselPrepared, PrologueEpilogueInserted, RegAllocated, SelectedMir,
};
use crate::pipeline::{FunctionPass, ModuleCodegenPass};
use crate::Emitter;
use alloc::boxed::Box;
use alloc::vec::Vec;
use veloc_ir::Type;

pub use abi::{
    AbiAssignment, AbiClassifierEntry, AbiClassifierFn, AbiDescriptor, AbiLocation, AbiPart,
    AbiPreservedSet, AbiRegisterPool, AbiStackBase, AbiStackDescriptor, AbiValueClass,
    CallConvPlan,
};
pub use callconv::CallConv;
pub use types::{
    CpuDescription, DataLayout, RegClass, RegClassInfo, RegInfo, RegisterFile, SpecialRegs,
    TargetArch, TargetConfig, TargetDescription,
};

/// 基础 Lowering Context 接口 (所有后端共用)
pub trait LoweringContext {
    /// 获取值的类型
    fn get_type(&self, val: VReg) -> Type;

    /// 谓词：检查是否为 i32
    fn is_i32(&self, val: VReg) -> bool {
        self.get_type(val).is_integer() && self.get_type(val).size_bytes() == 4
    }

    /// 谓词：检查是否为 i16
    fn is_i16(&self, val: VReg) -> bool {
        self.get_type(val).is_integer() && self.get_type(val).size_bytes() == 2
    }

    /// 谓词：检查是否为 i8
    fn is_i8(&self, val: VReg) -> bool {
        self.get_type(val).is_integer() && self.get_type(val).size_bytes() == 1
    }

    /// 谓词：检查是否为 i64
    fn is_i64(&self, val: VReg) -> bool {
        self.get_type(val).is_integer() && self.get_type(val).size_bytes() == 8
    }

    /// 谓词：检查是否为 32 位整数宽度的值
    fn is_int32like(&self, val: VReg) -> bool {
        let ty = self.get_type(val);
        ty.is_integer() && ty.size_bytes() <= 4
    }

    /// 谓词：检查是否为 64 位整数或指针宽度的值
    fn is_64like(&self, val: VReg) -> bool {
        let ty = self.get_type(val);
        (ty.is_integer() && ty.size_bytes() == 8) || ty.is_ptr()
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
    fn get_bank(&self, val: VReg) -> Option<crate::regalloc::regbank_select::RegisterBank>;

    /// 谓词：检查是否在 FPR (浮点寄存器库)
    fn is_fpr(&self, val: VReg) -> bool {
        matches!(
            self.get_bank(val),
            Some(crate::regalloc::regbank_select::RegisterBank::FPR)
        )
    }

    /// 获取指定的寄存器操作数
    fn get_vreg(&self, inst: &MachineInst, index: usize) -> Option<VReg>;
}

/// Target Machine: 封装特定目标架构的所有组件和策略。
/// 模仿 LLVM TargetMachine，作为从通用流程获取架构特定逻辑的统一入口。
pub trait TargetMachine {
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

/// 机器码发射器接口
pub trait TargetEmitter: Send + Sync {
    /// 开始发射一个基本块。
    fn begin_block(
        &self,
        _emitter: &mut Emitter,
        _block: &crate::mir::MachineBlock,
        _mfunc: &MachineFunction<PrologueEpilogueInserted>,
    ) -> Result<(), crate::error::Error> {
        Ok(())
    }

    /// 发射单条指令 (此时指令应该是目标特定的 Opcode)
    fn emit_instruction(
        &self,
        emitter: &mut Emitter,
        inst: &MachineInst,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TiedOperandConstraint {
    pub def_operand: usize,
    pub use_operand: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetTiedOperandMetadata {
    pub operand: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedUseConstraint {
    pub use_operand: usize,
    pub reg: Reg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenericInstMetadata {
    pub tied_operands: &'static [TiedOperandConstraint],
    pub commute_operand_pairs: &'static [(usize, usize)],
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
        opcode: crate::mir::GenericOpcode,
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
    pub tied_operands: Vec<TiedOperandConstraint>,
    pub commute_operand_pairs: Vec<(usize, usize)>,
    pub fixed_uses: Vec<FixedUseConstraint>,
}

impl OperandConstraintSet {
    pub fn is_empty(&self) -> bool {
        self.tied_operands.is_empty()
            && self.commute_operand_pairs.is_empty()
            && self.fixed_uses.is_empty()
    }
}

impl GenericInstMetadata {
    pub const EMPTY: Self = Self {
        tied_operands: &[],
        commute_operand_pairs: &[],
        fixed_uses: &[],
    };

    pub fn operand_constraints(&self) -> OperandConstraintSet {
        OperandConstraintSet {
            tied_operands: self.tied_operands.to_vec(),
            commute_operand_pairs: self.commute_operand_pairs.to_vec(),
            fixed_uses: self.fixed_uses.to_vec(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetInstMetadata {
    pub tied_operands: &'static [TargetTiedOperandMetadata],
    pub fixed_uses: &'static [FixedUseConstraint],
    pub implicit_uses: &'static [Reg],
    pub implicit_defs: &'static [Reg],
    pub clobbers: &'static [&'static str],
}

impl TargetInstMetadata {
    pub const EMPTY: Self = Self {
        tied_operands: &[],
        fixed_uses: &[],
        implicit_uses: &[],
        implicit_defs: &[],
        clobbers: &[],
    };

    pub fn operand_constraints(&self) -> OperandConstraintSet {
        OperandConstraintSet {
            tied_operands: Vec::new(),
            commute_operand_pairs: Vec::new(),
            fixed_uses: self.fixed_uses.to_vec(),
        }
    }
}

pub trait TargetLegalizer: Send + Sync {
    /// 查询一条 generic MIR 指令在当前目标上的 legalize 动作。
    fn legalize_action(
        &self,
        _inst: &MachineInst,
        _mfunc: &MachineFunction<LegalizedMir>,
    ) -> Result<Option<LegalizeAction>, crate::error::Error> {
        Ok(None)
    }

    /// 应用 target-specific legalization。
    ///
    /// 只有当 `legalize_action()` 返回 `Some(LegalizeAction::Lower)` 或其他
    /// 需要目标私有重写的动作时，driver 才会调用这个 hook。
    fn legalize_instruction(
        &self,
        inst_id: crate::mir::InstId,
        mfunc: &mut crate::mir::MachineFunction<LegalizedMir>,
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
        _inst: &MachineInst,
        _mfunc: &MachineFunction<PreIselPrepared>,
    ) -> OperandConstraintSet {
        OperandConstraintSet::default()
    }

    /// 查询一条 selected MIR 指令需要满足的操作数约束。
    ///
    /// 适合处理固定寄存器等 target instruction 级别的约束。
    fn postselect_operand_constraints(
        &self,
        _inst: &MachineInst,
        _mfunc: &MachineFunction<SelectedMir>,
    ) -> OperandConstraintSet {
        OperandConstraintSet::default()
    }

    /// 为 pre-isel 约束阶段构造一条目标相关的寄存器拷贝指令。
    ///
    /// 当拷贝两端任一操作数已经绑定到物理寄存器时，调用方应优先使用这条
    /// hook，而不是继续发射通用 `G_COPY`。这样可以保证位宽/寄存器别名等
    /// 目标相关语义在进入后续阶段前已经明确。
    fn build_preselect_reg_copy(
        &self,
        _mfunc: &MachineFunction<PreIselPrepared>,
        _dst: Reg,
        _src: Reg,
    ) -> Result<MachineInst, crate::error::Error> {
        panic!("target does not support pre-select register copy construction",)
    }

    /// 为 post-isel 约束阶段构造一条目标相关的寄存器拷贝指令。
    fn build_postselect_reg_copy(
        &self,
        _mfunc: &MachineFunction<SelectedMir>,
        _dst: Reg,
        _src: Reg,
    ) -> Result<MachineInst, crate::error::Error> {
        panic!("target does not support post-select register copy construction",)
    }
}

pub trait TargetPostIsel: Send + Sync {
    /// 指令融合（可选）
    ///
    /// 在指令选择后、寄存器分配前执行。
    /// 默认不做任何处理，目标后端可以按需覆写。
    fn combine_instructions(&self, _mfunc: &mut MachineFunction<SelectedMir>) {}
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
    /// 在寄存器分配之后调用，将序言/尾声指令插入到 MIR 中。
    fn insert_prologue_epilogue(&self, mfunc: &mut MachineFunction<RegAllocated>);
}

pub trait TargetPassConfig: Send + Sync {
    /// 在合法化之后追加 target 自定义 function passes。
    fn post_legalize_passes(
        &self,
    ) -> Vec<Box<dyn FunctionPass<crate::pipeline::stages::LegalizedMir>>> {
        Vec::new()
    }

    /// 在 generic combine 之后追加 target 自定义 function passes。
    fn pre_isel_passes(
        &self,
    ) -> Vec<Box<dyn FunctionPass<crate::pipeline::stages::PreIselPrepared>>> {
        Vec::new()
    }

    /// 在指令选择之后追加 target 自定义 function passes。
    fn post_isel_passes(&self) -> Vec<Box<dyn FunctionPass<crate::pipeline::stages::SelectedMir>>> {
        Vec::new()
    }

    /// 在寄存器分配之后追加 target 自定义 function passes。
    fn post_regalloc_passes(
        &self,
    ) -> Vec<Box<dyn FunctionPass<crate::pipeline::stages::RegAllocated>>> {
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
