use super::*;
use crate::TargetConfig;
use crate::target::x86_64::{
    X86_64TargetMachine,
    inst::{REG_RAX, TargetInst},
};
use veloc_lir::{FieldValue, MachineOpcode};
use veloc_mir::Type;

#[test]
fn fills_a_dependency_gap_without_reordering_flag_consumers() {
    let target = X86_64TargetMachine::new(TargetConfig::default()).unwrap();
    let mut f = MachineFunction::new("schedule".into());
    let x = f.editor().alloc_vreg(Type::I64);
    let a = f.editor().alloc_vreg(Type::I64);
    let unused = f.editor().alloc_vreg(Type::I64);
    let imm = {
        let id = f.editor().writer().write(
            MachineOpcode::Target(TargetInst::X86Mov64Imm64.as_u32()),
            &[(unused)],
            &[],
            [FieldValue::Imm(42)],
        );
        f.editor().append_inst(veloc_lir::BlockId::from_u32(0), id);
        id
    };
    let copy = {
        let id = TargetInst::X86Mov64.write(f.editor().writer(), &[a], &[x], []);
        f.editor().append_inst(veloc_lir::BlockId::from_u32(0), id);
        id
    };
    let first = {
        let id = TargetInst::X86IMul64.write(f.editor().writer(), &[a], &[x, a], []);
        f.editor().append_inst(veloc_lir::BlockId::from_u32(0), id);
        id
    };
    let last = {
        let id = TargetInst::X86IMul64.write(f.editor().writer(), &[a], &[x, a], []);
        f.editor().append_inst(veloc_lir::BlockId::from_u32(0), id);
        id
    };
    let consume = {
        let id = f.editor().writer().write(
            MachineOpcode::Target(TargetInst::X86Sete.as_u32()),
            &[(REG_RAX)],
            &[],
            [],
        );
        f.editor().append_inst(veloc_lir::BlockId::from_u32(0), id);
        id
    };
    assert_eq!(
        schedule(&mut f, &target, &mut FunctionAnalysisCtx::default()),
        1
    );
    assert_eq!(
        f.block_insts(veloc_lir::BlockId::from_u32(0))
            .collect::<Vec<_>>(),
        &[copy, first, imm, last, consume]
    );
    // Schema constructors supply fixed implicit operands even outside Spec.
    use crate::target::x86_64::inst::REG_RDX;
    let divide = TargetInst::X86IDiv64.write(f.editor().writer(), &[], &[x], []);
    assert_eq!(f.inst(divide).inputs(), &[x]);
    assert!(f.inst(divide).results().is_empty());
    assert_eq!(f.inst(divide).implicit_uses(), &[REG_RAX, REG_RDX]);
    assert_eq!(f.inst(divide).implicit_defs(), &[REG_RAX, REG_RDX]);
    f.check_refs().unwrap();
    f.editor().replace(divide).write(
        veloc_lir::MachineOpcode::Target(TargetInst::X86Mov64 as u32),
        &[a],
        &[x],
        [],
    );
    assert!(f.inst(divide).implicit_uses().is_empty());
    assert!(f.inst(divide).implicit_defs().is_empty());
    assert_eq!(f.defs(REG_RDX).count(), 0);
    f.check_refs().unwrap();
}

#[test]
fn preserves_register_anti_dependencies_and_memory_barriers() {
    let target = X86_64TargetMachine::new(TargetConfig::default()).unwrap();
    let mut f = MachineFunction::new("dependencies".into());
    let a = f.editor().alloc_vreg(Type::I64);
    let b = f.editor().alloc_vreg(Type::I64);
    let c = f.editor().alloc_vreg(Type::I64);
    let read = {
        let id = TargetInst::X86Mov64.write(f.editor().writer(), &[b], &[a], []);
        f.editor().append_inst(veloc_lir::BlockId::from_u32(0), id);
        id
    };
    let overwrite = {
        let id = TargetInst::X86Mov64.write(f.editor().writer(), &[a], &[c], []);
        f.editor().append_inst(veloc_lir::BlockId::from_u32(0), id);
        id
    };
    let store = {
        let id = f.editor().writer().write(
            MachineOpcode::Target(TargetInst::X86Store64.as_u32()),
            &[],
            &[b, a],
            [FieldValue::Imm(0)],
        );
        f.editor().append_inst(veloc_lir::BlockId::from_u32(0), id);
        id
    };
    let after = {
        let id = TargetInst::X86Mov64.write(f.editor().writer(), &[c], &[a], []);
        f.editor().append_inst(veloc_lir::BlockId::from_u32(0), id);
        id
    };
    schedule(&mut f, &target, &mut FunctionAnalysisCtx::default());
    assert_eq!(
        f.block_insts(veloc_lir::BlockId::from_u32(0))
            .collect::<Vec<_>>(),
        &[read, overwrite, store, after]
    );
}
