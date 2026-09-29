//! RV64GC scalar backend using the LP64D calling convention.
//!
//! Integer i32 values are kept sign-extended to XLEN, as required by the ABI.
//! x5/x6 and f30/f31 are spill temporaries; x7 and x28..x31 are reserved for
//! expansion of selected instructions. No optional bit-manipulation ISA is needed.
pub mod emitter;
#[allow(dead_code, unused_imports)]
pub mod inst {
    include!(concat!(env!("OUT_DIR"), "/machine_riscv64.rs"));
}
mod frame;

use crate::target::*;
use veloc_lir::{FieldValue, MachineOpcode, RegisterBank};
use veloc_types::{DataLayout, TypeLayout};

pub const DATA_LAYOUT: DataLayout = DataLayout {
    types: &[
        (Type::BOOL, TypeLayout::fixed(1, 1)),
        (Type::I8, TypeLayout::fixed(1, 1)),
        (Type::I16, TypeLayout::fixed(2, 2)),
        (Type::I32, TypeLayout::fixed(4, 4)),
        (Type::I64, TypeLayout::fixed(8, 8)),
        (Type::PTR, TypeLayout::fixed(8, 8)),
        (Type::F32, TypeLayout::fixed(4, 4)),
        (Type::F64, TypeLayout::fixed(8, 8)),
    ],
    pointer_size: 8,
    little_endian: true,
};
pub use inst::ABI_RV64LP64D as ABI;
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
        Ok(Self {
            config,
            features,
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
impl TargetSchedule for Riscv64TargetMachine {}
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
        inst::TargetInst::from_u32(op).validate(f, i, mode)
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
            [FieldValue::Edge(edge)],
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
        let op = spill_opcode(load, ty);
        let one = [r];
        Ok(op.write(
            w,
            if load { &one } else { &[] },
            if load { &[] } else { &one },
            [FieldValue::StackSlot(s)],
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
    op.write(w, &[dst], &[src], [])
}
struct Operands;
impl TargetOperandLowering for Operands {
    fn preselect_operand_constraints(
        &self,
        i: &veloc_lir::InstRef<'_>,
        _: &MachineFunction,
    ) -> OperandConstraintSet {
        i.generic_opcode()
            .map(|op| inst::generic_inst_metadata(op).operand_constraints())
            .unwrap_or_default()
    }
    fn postselect_operand_constraints(
        &self,
        i: &veloc_lir::InstRef<'_>,
        _: &MachineFunction,
    ) -> OperandConstraintSet {
        match i.opcode() {
            MachineOpcode::Target(op) => metadata(op).operand_constraints(),
            _ => OperandConstraintSet::default(),
        }
    }

    fn build_preselect_reg_copy(
        &self,
        mut i: veloc_lir::InstInserter<'_>,
        dst: Reg,
        src: Reg,
    ) -> crate::Result<InstId> {
        let ty = if src.is_vreg() {
            i.vreg_data(src).ty
        } else if dst.is_vreg() {
            i.vreg_data(dst).ty
        } else if src.0 >= 32 || dst.0 >= 32 {
            Type::F64
        } else {
            Type::I64
        };
        Ok(copy(i.writer(), dst, src, ty))
    }
    fn build_postselect_reg_copy(
        &self,
        i: veloc_lir::InstInserter<'_>,
        dst: Reg,
        src: Reg,
    ) -> crate::Result<InstId> {
        self.build_preselect_reg_copy(i, dst, src)
    }
}
struct Passes;
impl TargetPostIsel for Passes {}
impl TargetPassConfig for Passes {
    fn prepare_passes(&self) -> Vec<Box<dyn crate::pipeline::FunctionPass>> {
        vec![Box::new(
            crate::passes::lowering::control::BranchTableLowering,
        )]
    }
}
impl SelectionContext for inst::FeatureSet {}
impl crate::isel::SelectHooks for inst::FeatureSet {
    fn predicate(&self, id: u32, reg: Reg) -> bool {
        inst::selection_predicate(self, id, reg)
    }
}
impl TargetMachine for Riscv64TargetMachine {
    fn config(&self) -> &TargetConfig {
        &self.config
    }
    fn legalizer(&self) -> crate::passes::lowering::legalize::LegalizePolicy<'_> {
        crate::passes::lowering::legalize::LegalizePolicy {
            program: &legalize::PROGRAM,
            features: self.features.as_words(),
        }
    }
    fn selector(&self) -> crate::isel::SelectPolicy<'_> {
        crate::isel::SelectPolicy {
            program: inst::selection_program,
            features: self.features.as_words(),
            metadata,
            predicate: &self.features,
        }
    }
    fn operand_lowering(&self) -> &dyn TargetOperandLowering {
        &Operands
    }
    fn post_isel(&self) -> &dyn TargetPostIsel {
        &Passes
    }
    fn frame_lowering(&self) -> &dyn TargetFrameLowering {
        &frame::Frame
    }
    fn pass_config(&self) -> &dyn TargetPassConfig {
        &Passes
    }
    fn emitter(&self) -> &dyn TargetEmitter {
        &emitter::Emit
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
        let plan = CallConv::RiscvABI
            .plan(
                TargetArch::Riscv64,
                &DATA_LAYOUT,
                &args,
                &[Type::I32, Type::F64],
            )
            .unwrap();
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
        let sig = mb.make_signature(vec![], vec![], veloc_mir::CallConv::SystemV);
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
        let bytes = crate::CodegenPipeline::new(&target)
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

pub trait SelectionContext {}
