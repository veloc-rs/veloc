use crate::translate::IRTranslator;
use alloc::format;
use veloc_mir::{InstructionData, Module, ModuleParser, Opcode, Type, ValueList};

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
