//! x86_64 Target Implementation
//!
//! x86_64 架构的具体实现

pub mod assembly;
pub mod emitter;
mod frame;
pub mod inst;
mod legalize;
pub mod lowering;
mod machine;
mod operand;
mod pass_config;
mod select;

pub use emitter::X86_64CodeEmitter;
pub use frame::X86_64FrameLowering;
pub use legalize::X86_64Legalizer;
pub use operand::X86_64OperandLowering;
pub use pass_config::{X86_64PassConfig, X86_64PostIsel};
pub use select::X86_64Selector;

use crate::target::{
    RegClass, RegClassInfo, RegisterFile, SpecialRegs, SpillKind, TargetConfig, TargetDescription,
    TargetEmitter, TargetFrameLowering, TargetInfo, TargetInstructionSelector, TargetInstructions,
    TargetLegalizer, TargetMachine, TargetOperandLowering, TargetPassConfig, TargetPostIsel,
    TargetRegalloc, TargetSchedule, ValidationMode,
};
use veloc_lir::RegisterBank;
use veloc_types::{DataLayout, Type, TypeLayout};

const X86_64_GPR_ALLOCATABLE: &[veloc_lir::Reg] = &[
    inst::REG_RAX,
    inst::REG_RDX,
    inst::REG_RBX,
    inst::REG_RSI,
    inst::REG_RDI,
    inst::REG_R8,
    inst::REG_R9,
    inst::REG_R12,
    inst::REG_R13,
    inst::REG_R14,
    inst::REG_R15,
];
const X86_64_FPR_ALLOCATABLE: &[veloc_lir::Reg] = &[
    inst::REG_XMM0,
    inst::REG_XMM1,
    inst::REG_XMM2,
    inst::REG_XMM3,
    inst::REG_XMM4,
    inst::REG_XMM5,
    inst::REG_XMM6,
    inst::REG_XMM7,
    inst::REG_XMM8,
    inst::REG_XMM9,
    inst::REG_XMM10,
    inst::REG_XMM11,
    inst::REG_XMM12,
    inst::REG_XMM13,
];

const X86_64_REG_CLASSES: &[RegClassInfo] = &[
    RegClassInfo {
        kind: RegClass::GPR,
        bank: RegisterBank::GPR,
        members: inst::REGCLASS_GPR64,
        allocatable: X86_64_GPR_ALLOCATABLE,
    },
    RegClassInfo {
        kind: RegClass::FPR,
        bank: RegisterBank::FPR,
        members: inst::REGCLASS_FPR128,
        allocatable: X86_64_FPR_ALLOCATABLE,
    },
];
static X86_64_REGISTER_FILE: RegisterFile = RegisterFile {
    regs: inst::PHYS_REG_INFOS,
    reg_classes: X86_64_REG_CLASSES,
    reserved_regs: inst::RESERVED_REGS,
    special_regs: SpecialRegs {
        stack_pointer: inst::SPECIAL_REG_STACK_POINTER,
        frame_pointer: Some(inst::SPECIAL_REG_FRAME_POINTER),
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

/// x86_64 目标机器实现
pub struct X86_64TargetMachine {
    config: TargetConfig,
    desc: TargetDescription,
    features: inst::FeatureSet,
    legalizer: X86_64Legalizer,
    selector: X86_64Selector,
    operand_lowering: X86_64OperandLowering,
    post_isel: X86_64PostIsel,
    frame_lowering: X86_64FrameLowering,
    pass_config: X86_64PassConfig,
    emitter: X86_64CodeEmitter,
}

impl X86_64TargetMachine {
    pub fn new(config: TargetConfig) -> crate::Result<Self> {
        let cpu = inst::SUPPORTED_CPUS
            .iter()
            .find(|cpu| cpu.name == config.cpu)
            .ok_or_else(|| {
                crate::Error::codegen(alloc::format!("unknown x86-64 CPU: {}", config.cpu))
            })?;
        let features = cpu
            .features
            .resolve(&config.features)
            .map_err(crate::Error::codegen)?;
        let desc = TargetDescription {
            arch: crate::target::TargetArch::X86_64,
            registers: X86_64_REGISTER_FILE,
            data_layout: DATA_LAYOUT,
        };

        Ok(Self {
            config,
            desc,
            features,
            legalizer: X86_64Legalizer { features },
            selector: X86_64Selector::new(features),
            operand_lowering: X86_64OperandLowering,
            post_isel: X86_64PostIsel,
            frame_lowering: X86_64FrameLowering,
            pass_config: X86_64PassConfig,
            emitter: X86_64CodeEmitter::new(features),
        })
    }
}

impl TargetInstructions for X86_64TargetMachine {
    fn validate_instruction(
        &self,
        function: &veloc_lir::MachineFunction,
        inst: &veloc_lir::InstRef<'_>,
        mode: ValidationMode,
    ) -> crate::Result<()> {
        let veloc_lir::MachineOpcode::Target(op) = inst.opcode() else {
            return Err(crate::Error::codegen("expected a target instruction"));
        };
        let opcode = inst::TargetInst::from_u32(op);
        for feature in opcode.required_features().iter() {
            if !self.features.contains(feature) {
                return Err(crate::Error::codegen(alloc::format!(
                    "{opcode:?} requires target feature {}",
                    feature.name()
                )));
            }
        }
        opcode.validate(function, inst, mode)
    }
    fn write_assembly(
        &self,
        inst: &veloc_lir::InstRef<'_>,
        out: &mut dyn crate::target::AssemblyWriter,
    ) -> core::fmt::Result {
        let veloc_lir::MachineOpcode::Target(op) = inst.opcode() else {
            return Err(core::fmt::Error);
        };
        inst::TargetInst::from_u32(op).write_assembly(inst, out)
    }
    fn instruction_metadata(&self, opcode: u32) -> &'static crate::target::TargetInstMetadata {
        inst::target_inst_metadata(inst::TargetInst::from_u32(opcode))
    }
}

impl TargetSchedule for X86_64TargetMachine {}

impl TargetRegalloc for X86_64TargetMachine {
    fn spill_scratch(&self, class: RegClass) -> &'static [veloc_lir::Reg] {
        match class {
            RegClass::GPR => &[inst::REG_R10, inst::REG_R11],
            RegClass::FPR => &[inst::REG_XMM14, inst::REG_XMM15],
            _ => &[],
        }
    }

    fn jump_instruction(
        &self,
        mut writer: veloc_lir::InstWriter<'_>,
        target: veloc_lir::BlockId,
    ) -> crate::Result<veloc_lir::InstId> {
        let edge = writer.edge(target, &[]);
        Ok(writer.write(
            veloc_lir::MachineOpcode::Target(inst::TargetInst::X86Jmp.as_u32()),
            &[],
            &[],
            [veloc_lir::FieldValue::Edge(edge)],
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
        Ok(opcode.write(writer, &[dst], &[src], []))
    }

    fn spill_instruction(
        &self,
        writer: veloc_lir::InstWriter<'_>,
        kind: SpillKind,
        reg: veloc_lir::Reg,
        slot: veloc_lir::StackSlot,
        ty: veloc_mir::Type,
    ) -> crate::error::Result<veloc_lir::InstId> {
        machine::spill_instruction(writer, kind, reg, slot, ty)
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
