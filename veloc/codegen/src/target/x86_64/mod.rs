//! x86_64 Target Implementation
//!
//! x86_64 架构的具体实现

pub mod emitter;
pub mod isle;
pub mod lowering;

pub use emitter::X86_64CodeEmitter;
pub use lowering::X86_64Lowering;

use crate::regalloc::regbank_select::TargetRegBankSelect;
use crate::target::arch::{
    CpuDescription, DataLayout, RegClass, RegClassInfo, RegisterFile, SpecialRegs, TargetConfig,
    TargetDescription, TargetEmitter, TargetLowering, TargetMachine,
};

const X86_64_GPR_ALLOCATABLE: &[crate::mir::Reg] = &[
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
const X86_64_FPR_ALLOCATABLE: &[crate::mir::Reg] = &[
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
        members: isle::REGCLASS_GPR64,
        allocatable: X86_64_GPR_ALLOCATABLE,
    },
    RegClassInfo {
        kind: RegClass::FPR,
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
    lowering: X86_64Lowering,
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
            lowering: X86_64Lowering::new(cpu),
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

    fn target_lowering(&self) -> &dyn TargetLowering {
        &self.lowering
    }

    fn target_emitter(&self) -> &dyn TargetEmitter {
        &self.emitter
    }

    fn target_regbank_select(&self) -> &dyn TargetRegBankSelect {
        &self.lowering
    }
}
