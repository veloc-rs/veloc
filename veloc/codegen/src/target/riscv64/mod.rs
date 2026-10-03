//! RV64GC scalar backend using the LP64D calling convention.
//!
//! Integer i32 values are kept sign-extended to XLEN, as required by the ABI.
//! x5/x6 and f30/f31 are spill temporaries; x31 handles large addresses and
//! far transfers. C908 enables optional Zba/Zbb selection;
//! the generic CPU retains base-ISA fallback sequences.
pub mod emitter;
#[allow(dead_code, unused_imports)]
pub mod inst {
    include!(concat!(env!("OUT_DIR"), "/machine_riscv64.rs"));
}
mod frame;
mod load_extensions;
mod zero;

use crate::target::*;
use veloc_lir::{MachineOpcode, RegisterBank};

pub use inst::ABI_RV64LP64D as ABI;
pub use inst::DATA_LAYOUT_RV64 as DATA_LAYOUT;
fn metadata(op: u32) -> &'static TargetInstMetadata {
    inst::target_inst_metadata(inst::TargetInst::from_u32(op))
}
#[allow(dead_code)]
mod legalize {
    include!(concat!(env!("OUT_DIR"), "/legalize_riscv64.rs"));
}

pub struct Riscv64TargetMachine {
    config: TargetConfig,
    desc: TargetDescription,
    features: inst::FeatureSet,
    schedule: ScheduleModel,
    emitter: emitter::Emit,
}
impl Riscv64TargetMachine {
    pub fn new(config: TargetConfig) -> crate::Result<Self> {
        let cpu = inst::SUPPORTED_CPUS
            .iter()
            .find(|cpu| cpu.name == config.cpu)
            .ok_or_else(|| crate::Error::codegen(format!("unknown RV64 CPU: {}", config.cpu)))?;
        let features = cpu
            .features
            .resolve(&config.features)
            .map_err(crate::Error::codegen)?;
        for required in [
            inst::Feature::I,
            inst::Feature::M,
            inst::Feature::F,
            inst::Feature::D,
        ] {
            if !features.contains(required) {
                return Err(crate::Error::codegen(
                    "RV64 backend requires I/M/F/D with LP64D",
                ));
            }
        }
        Ok(Self {
            features,
            emitter: emitter::Emit {
                compressed: features.contains(inst::Feature::C),
                external_calls: config.external_calls,
            },
            config,
            schedule: cpu.schedule,
            desc: TargetDescription {
                arch: TargetArch::Riscv64,
                data_layout: DATA_LAYOUT,
                registers: RegisterFile {
                    regs: inst::PHYS_REG_INFOS,
                    reg_classes: &[
                        RegClassInfo {
                            kind: RegClass::GPR,
                            bank: RegisterBank::GPR,
                            members: inst::REGCLASS_GPR,
                            allocatable: inst::REGCLASS_GPR_ALLOCATABLE,
                        },
                        RegClassInfo {
                            kind: RegClass::FPR,
                            bank: RegisterBank::FPR,
                            members: inst::REGCLASS_FPR,
                            allocatable: inst::REGCLASS_FPR_ALLOCATABLE,
                        },
                    ],
                    reserved_regs: inst::RESERVED_REGS,
                    special_regs: SpecialRegs {
                        stack_pointer: inst::SPECIAL_REG_STACK_POINTER,
                        frame_pointer: None,
                    },
                },
            },
        })
    }
}
impl TargetInfo for Riscv64TargetMachine {
    fn desc(&self) -> &TargetDescription {
        &self.desc
    }
}
impl TargetSchedule for Riscv64TargetMachine {
    fn schedule_model(&self) -> &ScheduleModel {
        &self.schedule
    }
}
impl TargetInstructions for Riscv64TargetMachine {
    fn instruction_metadata(&self, op: u32) -> &'static TargetInstMetadata {
        metadata(op)
    }
    fn validate_instruction(
        &self,
        f: &MachineFunction,
        i: &veloc_lir::InstRef<'_>,
        mode: ValidationMode,
    ) -> crate::Result<()> {
        let MachineOpcode::Target(op) = i.opcode() else {
            return Err(crate::Error::codegen("expected target instruction"));
        };
        inst::TargetInst::from_u32(op).validate(f, i, mode, self.features)
    }
    fn write_assembly(
        &self,
        i: &veloc_lir::InstRef<'_>,
        out: &mut dyn AssemblyWriter,
    ) -> core::fmt::Result {
        let MachineOpcode::Target(op) = i.opcode() else {
            return Err(core::fmt::Error);
        };
        inst::TargetInst::from_u32(op).write_assembly(i, out)
    }
}
impl TargetRegalloc for Riscv64TargetMachine {
    fn rematerializable_constant(&self, source: veloc_lir::InstRef<'_>) -> Option<i64> {
        let MachineOpcode::Target(op) = source.opcode() else {
            return None;
        };
        if !matches!(
            inst::TargetInst::from_u32(op),
            inst::TargetInst::RvLi32 | inst::TargetInst::RvLi64
        ) {
            return None;
        }
        let veloc_lir::FieldValueRef::Imm(&imm) = source.fields().read(0) else {
            return None;
        };
        (-2048..=2047).contains(&imm).then_some(imm)
    }
    fn spill_scratch(&self, c: RegClass) -> &'static [Reg] {
        match c {
            RegClass::GPR => &[Reg(5), Reg(6)],
            RegClass::FPR => &[Reg(62), Reg(63)],
            _ => &[],
        }
    }
    fn jump_instruction(
        &self,
        mut w: veloc_lir::InstWriter<'_>,
        b: veloc_lir::BlockId,
    ) -> crate::Result<InstId> {
        let edge = w.edge(b, &[]);
        Ok(w.write(
            MachineOpcode::Target(inst::TargetInst::RvJump.as_u32()),
            &[],
            &[],
            veloc_lir::Fields::Jump([edge]),
        ))
    }
    fn copy_instruction(
        &self,
        w: veloc_lir::InstWriter<'_>,
        dst: Reg,
        src: Reg,
        ty: Type,
    ) -> crate::Result<InstId> {
        Ok(copy(w, dst, src, ty))
    }
    fn spill_instruction(
        &self,
        w: veloc_lir::InstWriter<'_>,
        kind: SpillKind,
        r: Reg,
        s: veloc_lir::StackSlot,
        ty: Type,
    ) -> crate::Result<InstId> {
        let load = kind == SpillKind::Load;
        let op = spill_opcode(load, self.desc.scalar_storage_type(r, ty)?);
        let one = [r];
        Ok(op.write(
            w,
            if load { &one } else { &[] },
            if load { &[] } else { &one },
            veloc_lir::Fields::StackMemory {
                slot: s,
                flags: veloc_lir::MemFlags::new(),
            },
        ))
    }
}
fn spill_opcode(load: bool, ty: Type) -> inst::TargetInst {
    use inst::TargetInst::*;
    match (load, ty) {
        (true, Type::BOOL | Type::I8) => RvLoad8Stack,
        (false, Type::BOOL | Type::I8) => RvStore8Stack,
        (true, Type::I16) => RvLoad16Stack,
        (false, Type::I16) => RvStore16Stack,
        (true, Type::I32) => RvLoad32Stack,
        (false, Type::I32) => RvStore32Stack,
        (true, Type::F32) => RvLoadF32Stack,
        (false, Type::F32) => RvStoreF32Stack,
        (true, Type::F64) => RvLoadF64Stack,
        (false, Type::F64) => RvStoreF64Stack,
        (true, _) => RvLoad64Stack,
        (false, _) => RvStore64Stack,
    }
}
fn copy(w: veloc_lir::InstWriter<'_>, dst: Reg, src: Reg, ty: Type) -> InstId {
    let op = if matches!(
        ty,
        Type::BOOL | Type::I8 | Type::I16 | Type::I32 | Type::F32
    ) {
        inst::TargetInst::RvMove32
    } else {
        inst::TargetInst::RvMove64
    };
    op.write(w, &[dst], &[src], veloc_lir::Fields::None)
}
struct Passes;
impl TargetPassConfig for Passes {
    fn post_isel_passes(
        &self,
        level: crate::OptLevel,
    ) -> Vec<Box<dyn crate::pipeline::FunctionPass>> {
        match level {
            crate::OptLevel::None => vec![],
            crate::OptLevel::Default => vec![
                Box::new(load_extensions::FoldLoadExtensions),
                Box::new(zero::ZeroOperands),
            ],
        }
    }
    fn prepare_passes(
        &self,
        _level: crate::OptLevel,
    ) -> Vec<Box<dyn crate::pipeline::FunctionPass>> {
        vec![Box::new(
            crate::passes::lowering::control::BranchTableLowering { max_cases: 8 },
        )]
    }
}
impl TargetMachine for Riscv64TargetMachine {
    fn resolve_abi(&self, convention: CallConv) -> crate::Result<&'static AbiDescriptor> {
        match convention {
            CallConv::Platform => Ok(&ABI),
            _ => Err(crate::Error::codegen(format!(
                "unsupported RISC-V calling convention {convention}"
            ))),
        }
    }

    fn config(&self) -> &TargetConfig {
        &self.config
    }
    fn legalizer(&self) -> crate::passes::lowering::legalize::LegalizePolicy<'_> {
        crate::passes::lowering::legalize::LegalizePolicy {
            program: &legalize::PROGRAM,
            features: crate::target::FeatureSetRef::new(self.features.as_words()),
        }
    }
    fn selector(&self) -> crate::passes::isel::SelectPolicy<'_> {
        crate::passes::isel::SelectPolicy {
            program: &inst::SELECTION_PROGRAM,
            features: crate::target::FeatureSetRef::new(self.features.as_words()),
            predicate: None,
        }
    }

    fn frame_lowering(&self) -> &dyn TargetFrameLowering {
        &frame::Frame
    }
    fn pass_config(&self) -> &dyn TargetPassConfig {
        &Passes
    }
    fn emitter(&self) -> &dyn TargetEmitter {
        &self.emitter
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object::{Object, ObjectSection};
    #[test]
    fn lp64d_assigns_banks_and_stack_independently() {
        let mut args = vec![Type::I64; 9];
        args.extend([Type::F64; 9]);
        let signature =
            veloc_mir::Signature::new(&args, [Type::I32, Type::F64], CallConv::Platform);
        let plan = ABI.plan(&signature, &args).unwrap();
        assert_eq!(plan.args[7].loc, AbiLocation::Reg(Reg(17)));
        assert_eq!(
            plan.args[8].loc,
            AbiLocation::Stack {
                offset: 0,
                size: 8,
                align: 8
            }
        );
        assert_eq!(plan.args[16].loc, AbiLocation::Reg(Reg(49)));
        assert_eq!(
            plan.args[17].loc,
            AbiLocation::Stack {
                offset: 8,
                size: 8,
                align: 8
            }
        );
        assert_eq!(plan.returns[0].loc, AbiLocation::Reg(Reg(10)));
        assert_eq!(plan.returns[1].loc, AbiLocation::Reg(Reg(42)));
    }
    #[test]
    fn objects_use_riscv_architecture_and_absolute_call_relocations() {
        use veloc_mir::{Linkage, ModuleBuilder};
        let mut mb = ModuleBuilder::new();
        let sig = mb.make_signature(vec![], vec![], veloc_mir::CallConv::Platform);
        let ext = mb.declare_function("host".into(), sig, Linkage::Import);
        let main = mb.declare_function("main".into(), sig, Linkage::Export);
        {
            let mut fb = mb.define(main);
            fb.ins().call(ext, &[]);
            fb.ins().ret(&[]);
        }
        let target = Riscv64TargetMachine::new(TargetConfig {
            arch: TargetArch::Riscv64,
            ..Default::default()
        })
        .unwrap();
        let bytes = crate::CodegenPipeline::new(&target, Default::default())
            .compile_object(&mb.build())
            .unwrap();
        let obj = object::File::parse(&*bytes).unwrap();
        assert_eq!(obj.architecture(), object::Architecture::Riscv64);
        let text = obj.section_by_name(".text").unwrap();
        let relocations = text.relocations().collect::<Vec<_>>();
        assert_eq!(relocations.len(), 1);
        assert_eq!(relocations[0].0 % 8, 0);
        assert_eq!(
            relocations[0].1.flags(),
            object::RelocationFlags::Elf {
                r_type: object::elf::R_RISCV_64
            }
        );
    }
}
