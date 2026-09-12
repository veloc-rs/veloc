use crate::translate::IRTranslator;
use alloc::format;
use veloc_mir::{Module, ModuleParser, Opcode, Type};

fn module(opcode: Opcode, ty: Type, arity: usize) -> Module {
    let source = if arity == 1 {
        format!(
            "local function op({ty}) -> {ty}\n\
             block0(v0: {ty}):\n\
             v1: {ty} = {} v0\n\
             return v1\n",
            opcode.spec().mnemonic
        )
    } else {
        format!(
            "local function op({ty}, {ty}) -> {ty}\n\
             block0(v0: {ty}, v1: {ty}):\n\
             v2: {ty} = {} v0, v1\n\
             return v2\n",
            opcode.spec().mnemonic
        )
    };
    let module = ModuleParser::new().parse(&source).unwrap();
    module.validate().unwrap();
    module
}

#[test]
fn pointer_access_size_comes_from_the_target_data_layout() {
    let module = ModuleParser::new()
        .parse(
            r#"
local function load_pointer(ptr) -> ptr
block0(v0: ptr):
  v1: ptr = load.align4 v0, offset=0
  return v1
"#,
        )
        .unwrap();
    module.validate().unwrap();
    for pointer_size in [4, 8] {
        let lir = IRTranslator::new(
            &module,
            crate::target::arch::DataLayout {
                pointer_size,
                little_endian: true,
            },
        )
        .translate_module()
        .unwrap();
        let accesses: alloc::vec::Vec<_> = lir
            .functions
            .iter()
            .flat_map(|(_, f)| {
                f.blocks
                    .iter()
                    .flat_map(move |b| b.insts.iter().filter_map(move |id| f.dfg[*id].memory))
            })
            .collect();
        assert_eq!(accesses.len(), 1);
        assert_eq!(accesses[0].bytes, u32::from(pointer_size));
        assert_eq!(accesses[0].alignment, 4);
    }
}

#[test]
fn semantic_lowering_rejects_malformed_arity_and_type_instances() {
    let source = module(Opcode::IAdd, Type::I32, 2);
    for case in 0..7 {
        let mut data = (*source).clone();
        let (_, function) = data.functions.iter_mut().next().unwrap();
        let inst = function.layout().blocks()[function.entry_block.unwrap()].insts[0];
        let args = function.params().to_vec();
        let result = function.dfg().first_result(inst).unwrap();
        match case {
            0 => function.edit().set_value_type(args[1], Type::I64),
            1 => function.edit().set_value_type(result, Type::I64),
            2 => {
                function
                    .edit()
                    .replace_inst(inst, |writer: veloc_mir::InstWriter<'_>| {
                        writer.unary(Opcode::IAdd, args[0])
                    });
            }
            3 => {
                let replacement = |writer: veloc_mir::InstWriter<'_>| writer.copy(inst);
                function.edit().insert_after(inst, replacement, &[]);
            }
            4 => {
                for value in [args[0], args[1], result] {
                    function.edit().set_value_type(value, Type::F32);
                }
            }
            5 => {
                let replacement = |writer: veloc_mir::InstWriter<'_>| writer.copy(inst);
                function
                    .edit()
                    .insert_after(inst, replacement, &[Type::I32, Type::I32]);
            }
            6 => function
                .edit()
                .set_value_type(args[1], veloc_mir::Type::I32X4),
            _ => unreachable!(),
        }
        assert!(
            IRTranslator::new(
                &Module::new(data),
                crate::target::arch::DataLayout {
                    pointer_size: 8,
                    little_endian: true
                }
            )
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
        let inst = function.layout().blocks()[function.entry_block.unwrap()].insts[0];
        let arg = function.params()[0];
        let result = function.dfg().first_result(inst).unwrap();
        match case {
            0 => function.edit().set_value_type(result, Type::I64),
            1 => {
                function.edit().set_value_type(arg, Type::F32);
                function.edit().set_value_type(result, Type::F32);
            }
            2 => {
                function.edit().insert_after(
                    inst,
                    |writer: veloc_mir::InstWriter<'_>| writer.unary(Opcode::INeg, arg),
                    &[],
                );
            }
            3 => {
                function
                    .edit()
                    .replace_inst(inst, |writer: veloc_mir::InstWriter<'_>| {
                        writer.binary(Opcode::INeg, [arg, arg])
                    });
            }
            _ => unreachable!(),
        }
        assert!(
            IRTranslator::new(
                &Module::new(data),
                crate::target::arch::DataLayout {
                    pointer_size: 8,
                    little_endian: true
                }
            )
            .translate_module()
            .is_err()
        );
    }
}

#[test]
fn unsupported_tail_calls_and_callable_types_return_errors() {
    for source in [
        "local function forward(i32) -> i32\nblock0(v0: i32):\n  tail-call forward(v0) : (i32) -> i32\n",
        "local function apply(local<(i32) -> i32>, i32) -> i32\nblock0(v0: local<(i32) -> i32>, v1: i32):\n  tail-call-value v0(v1)\n",
    ] {
        let module = ModuleParser::new().parse(source).unwrap();
        module.validate().unwrap();
        let error = IRTranslator::new(
            &module,
            crate::target::arch::DataLayout {
                pointer_size: 8,
                little_endian: true,
            },
        )
        .translate_module()
        .err()
        .expect("unsupported control transfer");
        assert!(format!("{error}").contains("tail-call lowering"), "{error}");
    }
}
