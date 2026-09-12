use super::*;
use veloc_lir::{MachineBlock, stages::RawLir};
use veloc_mir::{Block, Type};

fn function() -> MachineFunction<RawLir> {
    let mut f = MachineFunction::new("reassociate".into());
    f.blocks.push(MachineBlock::new(Block(0)));
    f
}

fn add(f: &mut MachineFunction<RawLir>, op: GenericOpcode, dst: Reg, a: Reg, b: Reg) -> InstId {
    f.alloc_inst_and_append_to_block(
        0,
        MachineInst::build_binary(MachineOpcode::Generic(op), Writable(dst), a, b),
    )
}

fn eval(f: &MachineFunction<RawLir>, inputs: &[(Reg, u64)], output: Reg, mask: u64) -> u64 {
    let mut values: HashMap<_, _> = inputs.iter().copied().collect();
    for &id in f.block_insts(0) {
        let op = f.dfg[id].generic_opcode().unwrap();
        let node = binary(f, id, op).unwrap();
        let (a, b) = (values[&node.lhs], values[&node.rhs]);
        let value = match op {
            GenericOpcode::G_ADD => a.wrapping_add(b),
            GenericOpcode::G_MUL => a.wrapping_mul(b),
            GenericOpcode::G_AND => a & b,
            GenericOpcode::G_OR => a | b,
            GenericOpcode::G_XOR => a ^ b,
            _ => unreachable!(),
        };
        values.insert(node.dst, value & mask);
    }
    values[&output]
}

#[test]
fn preserves_wrapping_semantics_and_reuses_ids() {
    for ty in [Type::I8, Type::I16, Type::I32, Type::I64] {
        let mask = u64::MAX >> (64 - ty.element_bits().unwrap());
        for op in [
            GenericOpcode::G_ADD,
            GenericOpcode::G_MUL,
            GenericOpcode::G_AND,
            GenericOpcode::G_OR,
            GenericOpcode::G_XOR,
        ] {
            let mut f = function();
            let leaves: Vec<_> = (0..12).map(|_| f.alloc_vreg(ty)).collect();
            let mut acc = leaves[11];
            for &r in leaves[..11].iter().rev() {
                let dst = f.alloc_vreg(ty);
                add(&mut f, op, dst, acc, r);
                acc = dst;
            }
            let original = f.clone();
            let mut analyses = FunctionAnalysisCtx::default();
            assert_eq!(reassociate(&mut f, &mut analyses), 1);
            assert_eq!(f.dfg.len(), original.dfg.len());
            assert_eq!(f.vregs.len(), original.vregs.len());
            assert_eq!(reassociate(&mut f, &mut analyses), 0);
            for seed in 0u64..64 {
                let inputs: Vec<_> = leaves
                    .iter()
                    .enumerate()
                    .map(|(i, &r)| {
                        (
                            r,
                            seed.wrapping_mul(0x9e3779b97f4a7c15)
                                .rotate_left(i as u32)
                                .wrapping_add(u64::MAX - i as u64)
                                & mask,
                        )
                    })
                    .collect();
                assert_eq!(
                    eval(&original, &inputs, acc, mask),
                    eval(&f, &inputs, acc, mask)
                );
            }
        }
    }
}

#[test]
fn shared_subexpressions_are_not_duplicated() {
    let mut f = function();
    let a = f.alloc_vreg(Type::I32);
    let b = f.alloc_vreg(Type::I32);
    let t = f.alloc_vreg(Type::I32);
    let out = f.alloc_vreg(Type::I32);
    let shared = add(&mut f, GenericOpcode::G_ADD, t, b, a);
    let root = add(&mut f, GenericOpcode::G_ADD, out, t, t);
    reassociate(&mut f, &mut FunctionAnalysisCtx::default());
    assert_eq!(f.block_insts(0), &[shared, root]);
    assert_eq!(binary(&f, root, GenericOpcode::G_ADD).unwrap().lhs, t);
    assert_eq!(binary(&f, root, GenericOpcode::G_ADD).unwrap().rhs, t);
    assert_eq!(eval(&f, &[(a, 3), (b, 5)], out, u32::MAX as u64), 16);
}

#[test]
fn leaves_with_multiple_definitions_and_non_integer_types_are_untouched() {
    for ty in [Type::I32, Type::F32, veloc_mir::types::I32X4] {
        let mut f = function();
        let a = f.alloc_vreg(ty);
        let b = f.alloc_vreg(ty);
        let t = f.alloc_vreg(ty);
        let out = f.alloc_vreg(ty);
        add(&mut f, GenericOpcode::G_ADD, t, b, a);
        add(&mut f, GenericOpcode::G_ADD, out, t, a);
        if ty == Type::I32 {
            add(&mut f, GenericOpcode::G_ADD, t, a, b);
        }
        let before = alloc::format!("{:?}", f.dfg);
        assert_eq!(reassociate(&mut f, &mut FunctionAnalysisCtx::default()), 0);
        assert_eq!(alloc::format!("{:?}", f.dfg), before);
    }
}

#[test]
fn rebuilt_tree_stays_after_interleaved_leaf_definitions() {
    let mut f = function();
    let a = f.alloc_vreg(Type::I32);
    let b = f.alloc_vreg(Type::I32);
    let c = f.alloc_vreg(Type::I32);
    let d = f.alloc_vreg(Type::I32);
    let t = f.alloc_vreg(Type::I32);
    let u = f.alloc_vreg(Type::I32);
    let out = f.alloc_vreg(Type::I32);
    let first = add(&mut f, GenericOpcode::G_ADD, t, b, a);
    let leaf = add(&mut f, GenericOpcode::G_MUL, u, c, d);
    let root = add(&mut f, GenericOpcode::G_ADD, out, u, t);
    let original = f.clone();
    reassociate(&mut f, &mut FunctionAnalysisCtx::default());
    assert_eq!(f.block_insts(0), &[leaf, first, root]);
    let inputs = [(a, 1), (b, 2), (c, 3), (d, 4)];
    assert_eq!(eval(&f, &inputs, out, u32::MAX as u64), 15);
    assert_eq!(
        eval(&f, &inputs, out, u32::MAX as u64),
        eval(&original, &inputs, out, u32::MAX as u64)
    );
}

#[test]
fn trees_are_not_fused_across_blocks() {
    let mut f = function();
    f.blocks.push(MachineBlock::new(Block(1)));
    let a = f.alloc_vreg(Type::I32);
    let b = f.alloc_vreg(Type::I32);
    let c = f.alloc_vreg(Type::I32);
    let t = f.alloc_vreg(Type::I32);
    let out = f.alloc_vreg(Type::I32);
    let first = add(&mut f, GenericOpcode::G_ADD, t, b, a);
    let root = f.alloc_inst_and_append_to_block(
        1,
        MachineInst::build_binary(
            MachineOpcode::Generic(GenericOpcode::G_ADD),
            Writable(out),
            t,
            c,
        ),
    );
    reassociate(&mut f, &mut FunctionAnalysisCtx::default());
    assert_eq!(f.block_insts(0), &[first]);
    assert_eq!(f.block_insts(1), &[root]);
    let node = binary(&f, root, GenericOpcode::G_ADD).unwrap();
    assert!([node.lhs, node.rhs].contains(&t));
}

#[test]
fn long_trees_do_not_recurse_or_grow_storage() {
    let mut f = function();
    let leaves: Vec<_> = (0..4096).map(|_| f.alloc_vreg(Type::I64)).collect();
    let mut acc = leaves[4095];
    for &r in leaves[..4095].iter().rev() {
        let dst = f.alloc_vreg(Type::I64);
        add(&mut f, GenericOpcode::G_XOR, dst, acc, r);
        acc = dst;
    }
    let mut analyses = FunctionAnalysisCtx::default();
    assert_eq!(reassociate(&mut f, &mut analyses), 1);
    assert_eq!(f.dfg.len(), 4095);
    assert_eq!(reassociate(&mut f, &mut analyses), 0);
}
