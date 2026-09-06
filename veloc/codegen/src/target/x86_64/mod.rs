//! x86_64 Target Implementation
//!
//! x86_64 架构的具体实现

pub mod emitter;
pub mod isle;
pub mod lowering;

pub use emitter::X86_64CodeEmitter;
pub use lowering::{
    X86_64FrameLowering, X86_64Legalizer, X86_64OperandLowering, X86_64PassConfig, X86_64PostIsel,
    X86_64RegBankSelect, X86_64Selector,
};

use crate::regalloc::regbank_select::TargetRegBankSelect;
use crate::target::arch::{
    CpuDescription, DataLayout, RegClass, RegClassInfo, RegisterFile, SpecialRegs, TargetConfig,
    TargetDescription, TargetEmitter, TargetFrameLowering, TargetInstructionSelector,
    TargetLegalizer, TargetMachine, TargetOperandLowering, TargetPassConfig, TargetPostIsel,
};
use veloc_lir::RegisterBank;

const X86_64_GPR_ALLOCATABLE: &[veloc_lir::Reg] = &[
    isle::REG_RAX,
    isle::REG_RDX,
    isle::REG_RBX,
    isle::REG_RSI,
    isle::REG_RDI,
    isle::REG_R8,
    isle::REG_R9,
    isle::REG_R12,
    isle::REG_R13,
    isle::REG_R14,
    isle::REG_R15,
];
const X86_64_FPR_ALLOCATABLE: &[veloc_lir::Reg] = &[
    isle::REG_XMM0,
    isle::REG_XMM1,
    isle::REG_XMM2,
    isle::REG_XMM3,
    isle::REG_XMM4,
    isle::REG_XMM5,
    isle::REG_XMM6,
    isle::REG_XMM7,
    isle::REG_XMM8,
    isle::REG_XMM9,
    isle::REG_XMM10,
    isle::REG_XMM11,
    isle::REG_XMM12,
    isle::REG_XMM13,
];

const X86_64_REG_CLASSES: &[RegClassInfo] = &[
    RegClassInfo {
        kind: RegClass::GPR,
        bank: RegisterBank::GPR,
        members: isle::REGCLASS_GPR64,
        allocatable: X86_64_GPR_ALLOCATABLE,
    },
    RegClassInfo {
        kind: RegClass::FPR,
        bank: RegisterBank::FPR,
        members: isle::REGCLASS_FPR128,
        allocatable: X86_64_FPR_ALLOCATABLE,
    },
];
static X86_64_REGISTER_FILE: RegisterFile = RegisterFile {
    regs: isle::PHYS_REG_INFOS,
    reg_classes: X86_64_REG_CLASSES,
    reserved_regs: isle::RESERVED_REGS,
    special_regs: SpecialRegs {
        stack_pointer: isle::SPECIAL_REG_STACK_POINTER,
        frame_pointer: Some(isle::SPECIAL_REG_FRAME_POINTER),
    },
};
static X86_64_DATA_LAYOUT: DataLayout = DataLayout {
    pointer_size: 8,
    little_endian: true,
};
const GENERIC_CPU_NAME: &str = "generic";

/// x86_64 目标机器实现
pub struct X86_64TargetMachine {
    config: TargetConfig,
    desc: TargetDescription,
    legalizer: X86_64Legalizer,
    selector: X86_64Selector,
    operand_lowering: X86_64OperandLowering,
    post_isel: X86_64PostIsel,
    frame_lowering: X86_64FrameLowering,
    pass_config: X86_64PassConfig,
    regbank_select: X86_64RegBankSelect,
    emitter: X86_64CodeEmitter,
}

impl X86_64TargetMachine {
    pub fn new(config: TargetConfig) -> Self {
        let cpu = Self::select_cpu(&config.cpu);
        let desc = TargetDescription {
            arch: crate::target::arch::TargetArch::X86_64,
            registers: X86_64_REGISTER_FILE,
            data_layout: X86_64_DATA_LAYOUT,
            cpu,
        };

        Self {
            config,
            desc,
            legalizer: X86_64Legalizer::new(cpu),
            selector: X86_64Selector::new(cpu),
            operand_lowering: X86_64OperandLowering::new(cpu),
            post_isel: X86_64PostIsel::new(cpu),
            frame_lowering: X86_64FrameLowering::new(cpu),
            pass_config: X86_64PassConfig::new(cpu),
            regbank_select: X86_64RegBankSelect::new(cpu),
            emitter: X86_64CodeEmitter::new(),
        }
    }

    /// 根据 CPU 名称选择 ISLE 生成的 CPU 描述。
    fn select_cpu(cpu_name: &str) -> CpuDescription {
        use isle::SUPPORTED_CPUS;

        SUPPORTED_CPUS
            .iter()
            .copied()
            .find(|cpu| cpu.name == cpu_name)
            .unwrap_or_else(|| {
                SUPPORTED_CPUS
                    .iter()
                    .copied()
                    .find(|cpu: &CpuDescription| cpu.name == GENERIC_CPU_NAME)
                    .expect("generic CPU description must exist")
            })
    }
}

impl TargetMachine for X86_64TargetMachine {
    fn config(&self) -> &TargetConfig {
        &self.config
    }

    fn desc(&self) -> &TargetDescription {
        &self.desc
    }

    fn target_legalizer(&self) -> &dyn TargetLegalizer {
        &self.legalizer
    }

    fn target_selector(&self) -> &dyn TargetInstructionSelector {
        &self.selector
    }

    fn target_operand_lowering(&self) -> &dyn TargetOperandLowering {
        &self.operand_lowering
    }

    fn target_post_isel(&self) -> &dyn TargetPostIsel {
        &self.post_isel
    }

    fn target_frame_lowering(&self) -> &dyn TargetFrameLowering {
        &self.frame_lowering
    }

    fn target_pass_config(&self) -> &dyn TargetPassConfig {
        &self.pass_config
    }

    fn target_emitter(&self) -> &dyn TargetEmitter {
        &self.emitter
    }

    fn target_regbank_select(&self) -> &dyn TargetRegBankSelect {
        &self.regbank_select
    }
}
