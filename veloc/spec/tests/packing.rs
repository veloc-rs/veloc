mod common;
use common::parse;

const INDEXED: &str = r#"
record PtrIndexImm {
    storage: PtrIndexImmId, fields: [offset(i32, 0), scale(u32, 1)]
}
format Indexed {
    fields: [opcode(Opcode), ptr(Value), index(Value), imm_id(PtrIndexImmId)],
    opcode: dynamic(opcode)
}
op Indexed(ptr: PTR, index: I32, @imm: PtrIndexImm) -> (result: PTR) {
    mnemonic: "indexed", storage: Indexed { ptr: ptr, index: index, imm_id: pool(imm) },
    memory: NONE
}
"#;

const TABLE: &str = r#"
format Table {
    fields: [opcode(Opcode), index(Value), table(JumpTable)], opcode: dynamic(opcode)
}
op Table(index: I32, cases: successors, default: successor) -> () {
    mnemonic: "table", storage: Table { index: index, table: table(cases, default) },
    traits: [TERMINATOR], memory: NONE
}
"#;

const CALL: &str = r#"
format Invoke {
    fields: [opcode(Opcode), callee(FuncId), args(ValueList)], opcode: dynamic(opcode)
}
op Invoke(@callee: FuncId, args: values) -> signature {
    mnemonic: "invoke", storage: Invoke { callee: callee, args: args },
    signature: function(callee), memory: UNKNOWN
}
"#;

fn rejected(source: &str, message: &str) {
    let error = parse(source).err().expect("definition should be rejected");
    assert!(error.message.contains(message), "{error}");
}

#[test]
fn pool_bindings_are_typed_and_consume_the_complete_record() {
    parse(INDEXED).unwrap();
    rejected(&INDEXED.replace("pool(imm)", "pool()"), "pool(parameter)");
    rejected(
        &INDEXED.replace("@imm: PtrIndexImm", "@imm: Bytes"),
        "incompatible",
    );
    rejected(
        &INDEXED.replace(
            "@imm: PtrIndexImm",
            "@imm: PtrIndexImm, @unused: PtrIndexImm",
        ),
        "no storage mapping",
    );
    let duplicated = INDEXED
        .replace(
            "imm_id(PtrIndexImmId)",
            "imm_id(PtrIndexImmId), other(PtrIndexImmId)",
        )
        .replace("imm_id: pool(imm)", "imm_id: pool(imm), other: pool(imm)");
    rejected(&duplicated, "stored more than once");
    for ty in [
        "PtrIndexImmId",
        "ConstantPoolId",
        "VectorExtId",
        "VectorMemExtId",
    ] {
        rejected(
            &INDEXED.replace("@imm: PtrIndexImm", &format!("@imm: {ty}")),
            "unknown property type",
        );
    }
}

#[test]
fn pooled_value_lists_and_inline_arrays_keep_distinct_bindings() {
    let source = r#"
        format Triple { fields: [opcode(Opcode), args(list(3))], opcode: dynamic(opcode) }
        op Triple<T: Integer>(a: T, b: T, c: T) -> (result: T) {
            mnemonic: "triple", storage: Triple { args: [a, b, c] }, memory: NONE
        }
    "#;
    parse(source).unwrap();
    rejected(
        &source.replace("[a, b, c]", "[a, b]"),
        "requires 3 arguments",
    );
    rejected(
        &source.replace("[a, b, c]", "[a, a, c]"),
        "stored more than once",
    );
}

#[test]
fn tables_require_distinct_case_and_default_roles() {
    parse(TABLE).unwrap();
    rejected(
        &TABLE.replace("table(cases, default)", "table(cases)"),
        "table(cases, default)",
    );
    rejected(
        &TABLE.replace("table(cases, default)", "table(cases, cases)"),
        "stored more than once",
    );
    rejected(
        &TABLE.replace("default: successor", "default: successors"),
        "incompatible",
    );
    rejected(
        &TABLE.replace("table(cases, default)", "cases"),
        "incompatible",
    );
}

#[test]
fn signature_sources_are_explicit_and_match_their_parameter_types() {
    parse(CALL).unwrap();
    rejected(
        &CALL.replace("signature: function(callee),", ""),
        "explicit signature source",
    );
    rejected(
        &CALL.replace("function(callee)", "callee"),
        "SigId property",
    );
    rejected(
        &CALL.replace("function(callee)", "function(missing)"),
        "FuncId property",
    );
    rejected(
        &CALL.replace("-> signature", "-> ()"),
        "requires signature results",
    );
    let explicit = CALL
        .replace("FuncId", "SigId")
        .replace("function(callee)", "callee");
    parse(&explicit).unwrap();
}
