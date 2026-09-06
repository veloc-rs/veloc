use veloc_opgen::compile_mir;

const BINARY: &str = r#"
format Pair {
    fields: [op(Opcode), args(values(2))],
    opcode: dynamic(op), text: values(2)
}
op Add<T: Integer>(lhs: T, rhs: T) -> (result: T) {
    mnemonic: "pair-add", storage: Pair { args: [lhs, rhs] },
    traits: [], memory: NONE
}
"#;

fn rejected(source: &str, expected: &str) {
    let err = compile_mir(source).err().expect("schema must be rejected");
    assert!(err.message.contains(expected), "{err}");
}

#[test]
fn builders_are_automatic_and_use_the_declared_opcode_field() {
    let output = compile_mir(BINARY).unwrap();
    assert!(output.opcodes.contains(
        "pub fn pair_add(&mut self, lhs: crate::Value, rhs: crate::Value) -> crate::Value"
    ));
    assert!(
        output
            .opcodes
            .contains("InstructionData::Pair { op: crate::Opcode::Add, args: [lhs, rhs] }")
    );
    assert!(!output.opcodes.contains("InstructionData::from_values"));
}

#[test]
fn normalizing_mnemonics_cannot_create_duplicate_methods() {
    let source = format!(
        "{BINARY}\n\
        op Other<T: Integer>(lhs: T, rhs: T) -> (result: T) {{
            mnemonic: \"pair_add\", storage: Pair {{ args: [lhs, rhs] }}, traits: [], memory: NONE
        }}"
    );
    rejected(&source, "same method name");
}

#[test]
fn caller_selected_result_type_is_the_last_parameter() {
    let output = compile_mir(&BINARY.replace("result: T", "result: Integer")).unwrap();
    assert!(output.opcodes.contains(
        "pub fn pair_add(&mut self, lhs: crate::Value, rhs: crate::Value, ty: crate::Type) -> crate::Value"
    ));
    assert!(output.opcodes.contains(
        "self.push_with_type(crate::InstructionData::Pair { op: crate::Opcode::Add, args: [lhs, rhs] }, ty)"
    ));
}

#[test]
fn already_bound_result_variables_produce_an_inferred_pair() {
    let output = compile_mir(&BINARY.replace("result: T", "result: T, overflow: BOOL")).unwrap();
    assert!(output.opcodes.contains(
        "pub fn pair_add(&mut self, lhs: crate::Value, rhs: crate::Value) -> (crate::Value, crate::Value)"
    ));
    assert!(output.opcodes.contains("self.result_pair(inst)"));
    assert!(!output.opcodes.contains("ty: crate::Type"));
}

#[test]
fn arrays_use_logical_names_instead_of_arity_based_labels() {
    let source = BINARY
        .replace("values(2)", "values(3)")
        .replace("lhs: T, rhs: T", "base: T, index: T, increment: T")
        .replace("args: [lhs, rhs]", "args: [base, index, increment]");
    let output = compile_mir(&source).unwrap();
    assert!(output.opcodes.contains(
        "pub fn pair_add(&mut self, base: crate::Value, index: crate::Value, increment: crate::Value) -> crate::Value"
    ));
    assert!(output.opcodes.contains("args: [base, index, increment]"));
}

#[test]
fn logical_names_are_independent_of_physical_field_names() {
    let source = BINARY
        .replace("args(values(2))", "right(Value), left(Value)")
        .replace("args: [lhs, rhs]", "right: lhs, left: rhs");
    let output = compile_mir(&source).unwrap();
    assert!(output.opcodes.contains(
        "pub fn pair_add(&mut self, lhs: crate::Value, rhs: crate::Value) -> crate::Value"
    ));
    assert!(
        output
            .opcodes
            .contains("InstructionData::Pair { op: crate::Opcode::Add, right: lhs, left: rhs }")
    );
}

#[test]
fn fixed_formats_generate_property_builders_without_codec_special_cases() {
    let source = r#"
        format Iconst { fields: [value(u64)], opcode: fixed(Iconst), text: IntegerConstant }
        format Bconst { fields: [value(bool)], opcode: fixed(Bconst), text: BoolConstant }
        op Iconst(@value: u64) -> (result: ScalarInteger) {
            mnemonic: "iconst", storage: Iconst { value: value }, traits: [], memory: NONE
        }
        op Bconst(@value: bool) -> (result: BOOL) {
            mnemonic: "bconst", storage: Bconst { value: value }, traits: [], memory: NONE
        }
    "#;
    let output = compile_mir(source).unwrap();
    assert!(
        output
            .opcodes
            .contains("pub fn iconst(&mut self, value: u64, ty: crate::Type) -> crate::Value")
    );
    assert!(
        output
            .opcodes
            .contains("self.push_with_type(crate::InstructionData::Iconst { value }, ty)")
    );
    assert!(
        output
            .opcodes
            .contains("pub fn bconst(&mut self, value: bool) -> crate::Value")
    );
    assert!(
        output
            .opcodes
            .contains("self.push(crate::InstructionData::Bconst { value })")
    );
}

#[test]
fn property_order_follows_the_logical_signature_not_storage() {
    let source = r#"
        format Load {
            fields: [ptr(Value), offset(u32), flags(MemFlags)], opcode: fixed(Load), text: Load
        }
        format Store {
            fields: [ptr(Value), value(Value), offset(u32), flags(MemFlags)], opcode: fixed(Store), text: Store
        }
        op Load(@flags: MemFlags, address: PTR, @displacement: u32) -> (result: Any) {
            mnemonic: "load", storage: Load { ptr: address, offset: displacement, flags: flags },
            traits: [MAY_TRAP], memory: HEAP_READ
        }
        op Store(ptr: PTR, @flags: MemFlags, value: Any, @offset: u32) -> () {
            mnemonic: "store", storage: Store { ptr: ptr, value: value, offset: offset, flags: flags },
            traits: [MAY_TRAP], memory: HEAP_WRITE
        }
    "#;
    let output = compile_mir(source).unwrap();
    assert!(output.opcodes.contains(
        "pub fn load(&mut self, flags: crate::MemFlags, address: crate::Value, displacement: u32, ty: crate::Type) -> crate::Value"
    ));
    assert!(
        output
            .opcodes
            .contains("InstructionData::Load { ptr: address, offset: displacement, flags }")
    );
    assert!(output.opcodes.contains(
        "pub fn store(&mut self, ptr: crate::Value, flags: crate::MemFlags, value: crate::Value, offset: u32)"
    ));
}

#[test]
fn impossible_type_variable_constraints_are_definition_errors() {
    for (class, pattern) in [
        ("ScalarInteger", "element(T)"),
        ("Vector", "vector(T)"),
        ("ScalarInteger", "shape(T, Vector)"),
    ] {
        let source = BINARY
            .replace("T: Integer", &format!("T: {class}"))
            .replace("result: T", &format!("result: {pattern}"));
        rejected(&source, "impossible");
    }
}

#[test]
fn cfg_pool_and_signature_operations_keep_contextual_builders() {
    let source = [
        include_str!("../../ir/defs/formats.ops"),
        include_str!("../../ir/defs/mir.ops"),
    ]
    .join("\n");
    let output = compile_mir(&source).unwrap();
    for method in [
        "jump",
        "br",
        "br_table",
        "return",
        "call",
        "call_indirect",
        "call_intrinsic",
        "vconst",
        "ptr_index",
        "shuffle",
        "load_stride",
        "store_stride",
        "gather",
        "scatter",
    ] {
        assert!(
            !output.opcodes.contains(&format!("pub fn {method}(")),
            "unexpected bare builder for {method}"
        );
    }
    assert!(output.opcodes.contains("pub fn nop(&mut self)"));
    assert!(output.opcodes.contains("pub fn unreachable(&mut self)"));
}

#[test]
fn contextual_selection_does_not_depend_on_the_method_name() {
    let source = r#"
        format Jump { fields: [dest(BlockCall)], opcode: fixed(Jump), text: Jump }
        op Jump(dest: successor) -> () {
            mnemonic: "connect-edge", storage: Jump { dest: dest }, traits: [TERMINATOR], memory: NONE
        }
    "#;
    let output = compile_mir(source).unwrap();
    assert!(!output.opcodes.contains("pub fn connect_edge("));
    assert!(!output.opcodes.contains("self.push("));
}
