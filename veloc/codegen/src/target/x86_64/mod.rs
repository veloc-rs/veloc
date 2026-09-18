//! x86_64 Target Implementation
//!
//! x86_64 架构的具体实现

pub mod assembly;
pub mod emitter;
pub mod isle;
pub mod lowering;
mod machine;

pub use emitter::X86_64CodeEmitter;
pub use lowering::{
    X86_64FrameLowering, X86_64Legalizer, X86_64OperandLowering, X86_64PassConfig, X86_64PostIsel,
    X86_64Selector,
};

use crate::target::arch::{
    CpuDescription, RegClass, RegClassInfo, RegisterFile, SpecialRegs, SpillKind, TargetConfig,
    TargetDescription, TargetEmitter, TargetFrameLowering, TargetInfo, TargetInstructionSelector,
    TargetInstructions, TargetLegalizer, TargetMachine, TargetOperandLowering, TargetPassConfig,
    TargetPostIsel, TargetRegalloc, TargetSchedule, ValidationMode,
};
use veloc_lir::RegisterBank;
use veloc_types::{DataLayout, Type, TypeLayout};

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
pub const DATA_LAYOUT: DataLayout = DataLayout {
    types: &[
        (Type::BOOL, TypeLayout::fixed(1, 1)),
        (Type::I8, TypeLayout::fixed(1, 1)),
        (Type::I16, TypeLayout::fixed(2, 2)),
        (Type::I32, TypeLayout::fixed(4, 4)),
        (Type::I64, TypeLayout::fixed(8, 8)),
        (Type::F32, TypeLayout::fixed(4, 4)),
        (Type::F64, TypeLayout::fixed(8, 8)),
        (Type::PTR, TypeLayout::fixed(8, 8)),
        (Type::I8X16, TypeLayout::fixed(16, 16)),
        (Type::I16X8, TypeLayout::fixed(16, 16)),
        (Type::I32X4, TypeLayout::fixed(16, 16)),
        (Type::I64X2, TypeLayout::fixed(16, 16)),
        (Type::F32X4, TypeLayout::fixed(16, 16)),
        (Type::F64X2, TypeLayout::fixed(16, 16)),
    ],
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
    emitter: X86_64CodeEmitter,
}

impl X86_64TargetMachine {
    pub fn new(config: TargetConfig) -> Self {
        let cpu = Self::select_cpu(&config.cpu);
        let desc = TargetDescription {
            arch: crate::target::arch::TargetArch::X86_64,
            registers: X86_64_REGISTER_FILE,
            data_layout: DATA_LAYOUT,
            cpu,
        };

        Self {
            config,
            desc,
            legalizer: X86_64Legalizer { cpu },
            selector: X86_64Selector::new(cpu),
            operand_lowering: X86_64OperandLowering,
            post_isel: X86_64PostIsel,
            frame_lowering: X86_64FrameLowering,
            pass_config: X86_64PassConfig,
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

impl TargetInstructions for X86_64TargetMachine {
    fn validate_instruction(
        &self,
        inst: &veloc_lir::InstRef<'_>,
        mode: ValidationMode,
    ) -> crate::Result<()> {
        let veloc_lir::MachineOpcode::Target(op) = inst.opcode() else {
            return Err(crate::Error::codegen("expected a target instruction"));
        };
        let opcode = isle::TargetInst::from_u32(op);
        for feature in opcode.required_features() {
            if !self.desc.cpu.has_feature(feature) {
                return Err(crate::Error::codegen(alloc::format!(
                    "{opcode:?} requires target feature {feature}"
                )));
            }
        }
        opcode.validate(inst, mode)
    }
    fn write_assembly(
        &self,
        inst: &veloc_lir::InstRef<'_>,
        out: &mut dyn crate::target::arch::AssemblyWriter,
    ) -> core::fmt::Result {
        let veloc_lir::MachineOpcode::Target(op) = inst.opcode() else {
            return Err(core::fmt::Error);
        };
        isle::TargetInst::from_u32(op).write_assembly(inst, out)
    }
    fn instruction_metadata(
        &self,
        opcode: u32,
    ) -> &'static crate::target::arch::TargetInstMetadata {
        isle::target_inst_metadata(isle::TargetInst::from_u32(opcode))
    }
}

impl TargetSchedule for X86_64TargetMachine {}

impl TargetRegalloc for X86_64TargetMachine {
    fn spill_scratch(&self, class: RegClass) -> &'static [veloc_lir::Reg] {
        match class {
            RegClass::GPR => &[isle::REG_R10, isle::REG_R11],
            RegClass::FPR => &[isle::REG_XMM14, isle::REG_XMM15],
            _ => &[],
        }
    }

    fn jump_instruction(
        &self,
        writer: veloc_lir::InstWriter<'_>,
        target: veloc_lir::BlockId,
    ) -> crate::Result<veloc_lir::InstId> {
        Ok(writer.write(
            veloc_lir::MachineOpcode::Target(isle::TargetInst::X86Jmp.as_u32()),
            &[],
            &[],
            &[veloc_lir::InstField::Block(target)],
        ))
    }

    fn copy_instruction(
        &self,
        writer: veloc_lir::InstWriter<'_>,
        dst: veloc_lir::Reg,
        src: veloc_lir::Reg,
        ty: veloc_mir::Type,
    ) -> crate::Result<veloc_lir::InstId> {
        let opcode = lowering::x86_mov_opcode_for_type(ty)?;
        Ok(writer.unary(
            veloc_lir::MachineOpcode::Target(opcode.as_u32()),
            veloc_lir::Writable(dst),
            src,
        ))
    }

    fn spill_instruction(
        &self,
        writer: veloc_lir::InstWriter<'_>,
        kind: SpillKind,
        reg: veloc_lir::Reg,
        base: veloc_lir::Reg,
        offset: i64,
        ty: veloc_mir::Type,
    ) -> crate::error::Result<veloc_lir::InstId> {
        machine::spill_instruction(writer, kind, reg, base, offset, ty)
    }
}

impl TargetInfo for X86_64TargetMachine {
    fn desc(&self) -> &TargetDescription {
        &self.desc
    }
}

impl TargetMachine for X86_64TargetMachine {
    fn config(&self) -> &TargetConfig {
        &self.config
    }

    fn legalizer(&self) -> &dyn TargetLegalizer {
        &self.legalizer
    }

    fn selector(&self) -> &dyn TargetInstructionSelector {
        &self.selector
    }

    fn operand_lowering(&self) -> &dyn TargetOperandLowering {
        &self.operand_lowering
    }

    fn post_isel(&self) -> &dyn TargetPostIsel {
        &self.post_isel
    }

    fn frame_lowering(&self) -> &dyn TargetFrameLowering {
        &self.frame_lowering
    }

    fn pass_config(&self) -> &dyn TargetPassConfig {
        &self.pass_config
    }

    fn emitter(&self) -> &dyn TargetEmitter {
        &self.emitter
    }
}
