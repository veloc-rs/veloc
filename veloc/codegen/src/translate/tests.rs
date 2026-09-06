use crate::lir::{GenericOpcode, MachineFunction, MachineInst};
use crate::pipeline::stages::RawLir;
use crate::translate::IRTranslator;
use alloc::{format, vec::Vec};
use veloc_mir::semantics::BvOp;
use veloc_mir::{InstructionData, Module, ModuleParser, Opcode, Type, ValueList};

const OPS: &[(Opcode, BvOp, GenericOpcode)] = &[
    (Opcode::IAdd, BvOp::Add, GenericOpcode::G_ADD),
    (Opcode::ISub, BvOp::Sub, GenericOpcode::G_SUB),
    (Opcode::IMul, BvOp::Mul, GenericOpcode::G_MUL),
    (Opcode::INeg, BvOp::Neg, GenericOpcode::G_NEG),
    (Opcode::IAnd, BvOp::And, GenericOpcode::G_AND),
    (Opcode::IOr, BvOp::Or, GenericOpcode::G_OR),
    (Opcode::IXor, BvOp::Xor, GenericOpcode::G_XOR),
];

fn module(opcode: Opcode, ty: Type, arity: usize) -> Module {
    let source = if arity == 1 {
        format!(
            "local function op({ty}) -> {ty}\n\
             block0(v0: {ty}):\n\
             v1 = {}.{ty} v0\n\
             return v1\n",
            opcode.spec().mnemonic
        )
    } else {
        format!(
            "local function op({ty}, {ty}) -> {ty}\n\
             block0(v0: {ty}, v1: {ty}):\n\
             v2 = {}.{ty} v0, v1\n\
             return v2\n",
            opcode.spec().mnemonic
        )
    };
    let module = ModuleParser::new().parse(&source).unwrap();
    module.validate().unwrap();
    module
}

fn translated_op(function: &MachineFunction<RawLir>) -> &MachineInst {
    let params = function.params.len();
    &function.dfg[function.blocks[0].insts[params]]
}

#[test]
fn direct_primitives_share_bindings_and_composed_negation_is_not_misidentified() {
    for &(mir, semantics, lir) in OPS {
        let program = mir.spec().semantics.unwrap();
        assert_eq!(program.arity(), semantics.arity());
        if mir == Opcode::INeg {
            assert_eq!(program.primitive(), None);
        } else {
            assert_eq!(program.primitive(), Some(semantics));
        }
        assert_eq!(lir.semantics(), Some(semantics));
        assert_eq!(GenericOpcode::from_semantics(semantics), Some(lir));
    }
    for (mir, lir) in [
        (Opcode::IDivS, GenericOpcode::G_SDIV),
        (Opcode::FAdd, GenericOpcode::G_FADD),
    ] {
        assert_eq!(mir.spec().semantics, None);
        assert_eq!(lir.semantics(), None);
    }
}

#[test]
fn semantic_lowering_preserves_widths_and_operand_order() {
    for ty in [Type::I8, Type::I16, Type::I32, Type::I64] {
        for &(core, semantics, expected) in OPS {
            let source = module(core, ty, semantics.arity());
            let translated = IRTranslator::new(&source).translate_module().unwrap();
            let function = &translated.functions[translated.func_order[0]];
            let inst = translated_op(function);
            assert_eq!(inst.generic_opcode(), Some(expected));
            assert_eq!(inst.uses().collect::<Vec<_>>(), function.params);
            assert_eq!(inst.defs().count(), 1);
            let result = inst.defs().next().unwrap();
            assert_eq!(function.vreg_data(result).ty, ty);
            for arg in inst.uses() {
                assert_eq!(function.vreg_data(arg).ty, ty);
            }
            let ret = &function.dfg[*function.blocks[0].insts.last().unwrap()];
            assert_eq!(ret.uses().collect::<Vec<_>>(), [result]);
        }
    }
}

#[test]
fn lane_bindings_preserve_existing_bool_and_vector_translation() {
    // These assertions concern translation only. The binding does not claim
    // that target legalization or instruction selection supports these types.
    for (core, ty, expected) in [
        (Opcode::IAnd, Type::BOOL, GenericOpcode::G_AND),
        (Opcode::IOr, Type::BOOL, GenericOpcode::G_OR),
        (Opcode::IXor, Type::BOOL, GenericOpcode::G_XOR),
        (Opcode::IAdd, Type::I32X4, GenericOpcode::G_ADD),
        (Opcode::INeg, Type::I32X4, GenericOpcode::G_NEG),
    ] {
        let source = module(core, ty, core.spec().semantics.unwrap().arity());
        let translated = IRTranslator::new(&source).translate_module().unwrap();
        let function = &translated.functions[translated.func_order[0]];
        let inst = translated_op(function);
        assert_eq!(inst.generic_opcode(), Some(expected));
        assert_eq!(function.vreg_data(inst.defs().next().unwrap()).ty, ty);
    }
}

#[test]
fn uncovered_operations_keep_their_existing_lowering() {
    for (core, ty, expected) in [
        (Opcode::IDivS, Type::I32, GenericOpcode::G_SDIV),
        (Opcode::FAdd, Type::F64, GenericOpcode::G_FADD),
    ] {
        let source = module(core, ty, 2);
        let translated = IRTranslator::new(&source).translate_module().unwrap();
        let function = &translated.functions[translated.func_order[0]];
        assert_eq!(translated_op(function).generic_opcode(), Some(expected));
    }
}

#[test]
fn semantic_lowering_rejects_malformed_arity_and_type_instances() {
    let source = module(Opcode::IAdd, Type::I32, 2);
    for case in 0..7 {
        let mut data = (*source).clone();
        let (_, function) = data.functions.iter_mut().next().unwrap();
        let inst = function.layout.blocks[function.entry_block.unwrap()].insts[0];
        let args = function.params().to_vec();
        let result = function.dfg.first_result(inst).unwrap();
        match case {
            0 => function.dfg.values[args[1]].ty = Type::I64,
            1 => function.dfg.values[result].ty = Type::I64,
            2 => {
                function.dfg.replace_inst(
                    inst,
                    InstructionData::Unary {
                        opcode: Opcode::IAdd,
                        arg: args[0],
                    },
                );
            }
            3 => function.dfg.inst_results[inst] = ValueList::default(),
            4 => {
                for value in [args[0], args[1], result] {
                    function.dfg.values[value].ty = Type::F32;
                }
            }
            5 => {
                function.dfg.append_results(inst, &[Type::I32, Type::I32]);
            }
            6 => function.dfg.values[args[1]].ty = Type::I32X4,
            _ => unreachable!(),
        }
        assert!(
            IRTranslator::new(&Module::new(data))
                .translate_module()
                .is_err()
        );
    }
}

#[test]
fn composed_semantic_fallback_still_validates_its_source_contract() {
    let source = module(Opcode::INeg, Type::I32, 1);
    for case in 0..4 {
        let mut data = (*source).clone();
        let (_, function) = data.functions.iter_mut().next().unwrap();
        let inst = function.layout.blocks[function.entry_block.unwrap()].insts[0];
        let arg = function.params()[0];
        let result = function.dfg.first_result(inst).unwrap();
        match case {
            0 => function.dfg.values[result].ty = Type::I64,
            1 => {
                function.dfg.values[arg].ty = Type::F32;
                function.dfg.values[result].ty = Type::F32;
            }
            2 => function.dfg.inst_results[inst] = ValueList::default(),
            3 => {
                function.dfg.replace_inst(
                    inst,
                    InstructionData::Binary {
                        opcode: Opcode::INeg,
                        args: [arg, arg],
                    },
                );
            }
            _ => unreachable!(),
        }
        assert!(
            IRTranslator::new(&Module::new(data))
                .translate_module()
                .is_err()
        );
    }
}
