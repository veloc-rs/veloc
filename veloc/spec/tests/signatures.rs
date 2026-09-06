mod common;
use common::compile_mir;

const PAIR: &str = r#"
format Pair {
    fields: [op(Opcode), args(values(2))],
    opcode: dynamic(op)
}
op PairAdd<T: Integer>(left: T, right: T) -> (result: T) {
    mnemonic: "pair-add",
    storage: Pair { args: [left, right] },
    memory: NONE
}
"#;

const LOAD: &str = r#"
format Load {
    fields: [ptr(Value), offset(u32), flags(MemFlags)],
    opcode: fixed(Load)
}
op Load(ptr: PTR, @offset: u32, @flags: MemFlags) -> (result: Any) {
    mnemonic: "load",
    storage: Load { ptr: ptr, offset: offset, flags: flags },
    text: Text { args: [ptr], named: [default(offset, 0)], flags: flags },
    traits: [MAY_TRAP], memory: HEAP_READ
}
"#;

fn artifacts(output: veloc_opgen::Generated) -> [String; 9] {
    [
        output.encoding,
        output.builtins,
        output.scalars,
        output.formats,
        output.types,
        output.opcodes,
        output.instructions,
        output.text_parser,
        output.text_printer,
    ]
}

#[test]
fn result_names_do_not_change_generated_artifacts() {
    for (named, alternatives) in [
        ("(result: T)", vec!["T", "(T)", "(answer: T)"]),
        (
            "(result: T, overflow: BOOL)",
            vec!["(T, BOOL)", "(T, overflow: BOOL)", "(result: T, BOOL)"],
        ),
        (
            "(result: shape(T, Integer))",
            vec!["shape(T, Integer)", "(shape(T, Integer))"],
        ),
    ] {
        let reference = artifacts(compile_mir(&PAIR.replace("(result: T)", named)).unwrap());
        for results in alternatives {
            let source = PAIR.replace("(result: T)", results);
            assert_eq!(
                artifacts(compile_mir(&source).unwrap()),
                reference,
                "{results}"
            );
        }
    }
    let reference = artifacts(compile_mir(LOAD).unwrap());
    assert_eq!(
        artifacts(compile_mir(&LOAD.replace("(result: Any)", "Any")).unwrap()),
        reference
    );
}

#[test]
fn named_results_keep_their_position_among_anonymous_results() {
    let source = PAIR
        .replace("(result: T)", "(T, wider: I64)")
        .replace("memory: NONE", "where: [wider(left, wider)], memory: NONE");
    let output = compile_mir(&source).unwrap();
    assert!(
        output
            .types
            .contains("results[1] must have more bits per lane than operands[0]")
    );
    let both_named = source.replace("(T, wider: I64)", "(result: T, wider: I64)");
    assert_eq!(
        artifacts(output),
        artifacts(compile_mir(&both_named).unwrap())
    );
}

#[test]
fn result_only_type_variables_and_nested_type_patterns_still_bind() {
    let source = r#"
        format Empty { fields: [opcode(Opcode)], opcode: dynamic(opcode) }
        op Pair<T: Integer>() -> T {
            mnemonic: "pair", storage: Empty {}, memory: NONE
        }
    "#;
    let output = compile_mir(source).unwrap();
    assert!(output.types.contains("C::Integer.accepts(results[0])"));
    // Multiple explicit results are valid signatures, but not supported by the
    // current field-builder projection. Check their binding at the model layer.
    common::parse(&source.replace("-> T", "-> (T, T)")).unwrap();
    let vector = PAIR
        .replace("T: Integer", "T: Vector")
        .replace("(result: T)", "element(T)");
    assert_eq!(
        artifacts(compile_mir(&vector).unwrap()),
        artifacts(
            compile_mir(&vector.replace("-> element(T)", "-> (result: element(T))")).unwrap()
        )
    );
}
