mod common;

use common::{BUILTINS, compile_mir};

fn rejected(source: &str, message: &str) {
    let error = std::panic::catch_unwind(|| compile_mir(source))
        .expect("invalid predicates must not panic")
        .err()
        .expect("invalid predicate was accepted");
    assert!(error.message.contains(message), "{error}");
}

#[test]
fn predicates_generate_const_methods_from_exact_sets() {
    let output = compile_mir("predicate is_wide = Scalar & (I32 | I64);").unwrap();
    let method = output
        .scalars
        .split("pub const fn is_wide(self)")
        .nth(1)
        .unwrap();
    assert!(method.contains("match self.element_code() {\n3 | 4 => 0x00000001,"));
    assert!(method.contains("self.0 & !USED_MASK"));
    assert!(method.contains("self.lanes_log2()"));
    assert!(!method.contains("TypeClass"));
    assert!(!method.contains("self.is_scalar()"));
    assert!(!method.contains("self.is_valid()"));

    let equivalent = compile_mir("predicate is_wide = I64 | I32;").unwrap();
    assert_eq!(output.scalars, equivalent.scalars);
}

#[test]
fn predicate_sets_resolve_after_classes_and_preserve_exact_shapes() {
    let source = "predicate is_chosen = Chosen;\nclass Chosen { members: [I32X4 | SV4] }\ntype SV4 = vector(I32, scalable(4));";
    let output = compile_mir(source).unwrap();
    let method = output
        .scalars
        .split("pub const fn is_chosen(self)")
        .nth(1)
        .unwrap();
    assert!(method.contains("3 => 0x00040004,"));
    let source = source.replace("is_chosen = Chosen", "is_chosen = I32X4 | SV4");
    assert_eq!(output.scalars, compile_mir(&source).unwrap().scalars);
}

#[test]
fn invalid_sets_names_and_method_collisions_are_diagnosed() {
    for (source, error) in [
        ("predicate is_empty = Integer & Float;", "must not be empty"),
        ("predicate is_missing = Missing;", "unknown type or class"),
        ("predicate is_bad = vectors(PTR);", "non-pointer scalar"),
        ("predicate is_scalar = Scalar;", "duplicate predicate"),
        (
            "predicate is_valid = Any;",
            "conflicts with a built-in Type method",
        ),
        (
            "predicate is_scalable = Vector;",
            "conflicts with a built-in Type method",
        ),
        ("predicate scalar_type = Any;", "start with is_"),
        ("predicate I32 = Any;", "start with is_"),
        ("predicate is_Float = Float;", "snake_case"),
        ("predicate is_ = Float;", "start with is_"),
        ("predicate is_other = is_integer;", "unknown type or class"),
        ("predicate is_other = Any", "expected `;`"),
        ("predicate is_other { set: Any }", "expected `=`"),
    ] {
        rejected(source, error);
    }
    // The old generic Type method is gone; VectorType's method is a separate namespace.
    assert!(compile_mir("predicate is_fixed = I32X4;").is_ok());
}

#[test]
fn public_predicate_meanings_come_from_defs_not_rust_name_allowlists() {
    let changed = BUILTINS.replace(
        "predicate is_integer = Integer;",
        "predicate is_integer = I32 | I64;",
    );
    let output = veloc_opgen::compile_mir(&changed).unwrap();
    let method = output
        .scalars
        .rsplit("pub const fn is_integer(self)")
        .next()
        .unwrap()
        .split("/// Membership")
        .next()
        .unwrap();
    assert!(method.contains("3 | 4 => 0x00000001,"));
    assert!(!method.contains("self.element_type().is_integer()"));
}
