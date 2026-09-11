use super::*;
use crate::TargetConfig;
use crate::target::x86_64::{
    X86_64TargetMachine,
    isle::{REG_RAX, TargetInst},
};
use veloc_lir::{
    MachineBlock, MachineInst, MachineOpcode, MachineOperand, Writable, stages::RawLir,
};
use veloc_mir::{Block, Type};

#[test]
fn fills_a_dependency_gap_without_reordering_flag_consumers() {
    let target = X86_64TargetMachine::new(TargetConfig::default());
    let mut f = MachineFunction::<RawLir>::new("schedule".into());
    f.blocks.push(MachineBlock::new(Block(0)));
    let x = f.alloc_vreg(Type::I64);
    let a = f.alloc_vreg(Type::I64);
    let unused = f.alloc_vreg(Type::I64);
    let imm = f.alloc_inst_and_append_to_block(
        0,
        MachineInst::build_generic(
            MachineOpcode::Target(TargetInst::X86Mov64Imm64.as_u32()),
            smallvec::smallvec![
                MachineOperand::Def(Writable(unused)),
                MachineOperand::Imm(42)
            ],
        ),
    );
    let copy = f.alloc_inst_and_append_to_block(
        0,
        MachineInst::build_unary(
            MachineOpcode::Target(TargetInst::X86Mov64.as_u32()),
            Writable(a),
            x,
        ),
    );
    let first = f.alloc_inst_and_append_to_block(
        0,
        MachineInst::build_tied_binary(
            MachineOpcode::Target(TargetInst::X86IMul64.as_u32()),
            Writable(a),
            x,
        ),
    );
    let last = f.alloc_inst_and_append_to_block(
        0,
        MachineInst::build_tied_binary(
            MachineOpcode::Target(TargetInst::X86IMul64.as_u32()),
            Writable(a),
            x,
        ),
    );
    let consume = f.alloc_inst_and_append_to_block(
        0,
        MachineInst::build_generic(
            MachineOpcode::Target(TargetInst::X86Sete.as_u32()),
            smallvec::smallvec![MachineOperand::Def(Writable(REG_RAX))],
        ),
    );
    assert_eq!(
        schedule(&mut f, &target, &mut FunctionAnalysisCtx::default()),
        1
    );
    assert_eq!(f.block_insts(0), &[copy, first, imm, last, consume]);
}

#[test]
fn preserves_register_anti_dependencies_and_memory_barriers() {
    let target = X86_64TargetMachine::new(TargetConfig::default());
    let mut f = MachineFunction::<RawLir>::new("dependencies".into());
    f.blocks.push(MachineBlock::new(Block(0)));
    let a = f.alloc_vreg(Type::I64);
    let b = f.alloc_vreg(Type::I64);
    let c = f.alloc_vreg(Type::I64);
    let read = f.alloc_inst_and_append_to_block(
        0,
        MachineInst::build_unary(
            MachineOpcode::Target(TargetInst::X86Mov64.as_u32()),
            Writable(b),
            a,
        ),
    );
    let overwrite = f.alloc_inst_and_append_to_block(
        0,
        MachineInst::build_unary(
            MachineOpcode::Target(TargetInst::X86Mov64.as_u32()),
            Writable(a),
            c,
        ),
    );
    let store = f.alloc_inst_and_append_to_block(
        0,
        MachineInst::build_generic(
            MachineOpcode::Target(TargetInst::X86Store64.as_u32()),
            smallvec::smallvec![
                MachineOperand::Use(b),
                MachineOperand::Use(a),
                MachineOperand::Imm(0)
            ],
        ),
    );
    let after = f.alloc_inst_and_append_to_block(
        0,
        MachineInst::build_unary(
            MachineOpcode::Target(TargetInst::X86Mov64.as_u32()),
            Writable(c),
            a,
        ),
    );
    schedule(&mut f, &target, &mut FunctionAnalysisCtx::default());
    assert_eq!(f.block_insts(0), &[read, overwrite, store, after]);
}
