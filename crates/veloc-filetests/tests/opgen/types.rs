//! Type domains, semantic applicability, and predicate generation.
use super::common::{self, compile};

const ADD: &str = r#"
struct Pair { args: values(2) }
op Add<T: Bits>(lhs: T, rhs: T) -> (result: T) {
    meta: OpInfo {},
    mnemonic: "add", storage: Pair { args: [lhs, rhs] }, semantics: bv.add(lhs, rhs)
}
typeset Bits = ScalarInteger;
"#;

#[test]
fn effect_sets_use_the_declared_vocabulary() {
    assert!(compile(&ADD.replace("OpInfo {}", "OpInfo { memory: MemoryEffect::NONE }")).is_ok());
    for value in [
        "MemoryEffect::known(MemoryEffects::READ)",
        "MemoryEffect::known(MemoryEffects::WRITE)",
        "MemoryEffect::known(MemoryEffects::ALLOCATE)",
        "MemoryEffect::known(MemoryEffects::FREE)",
        "MemoryEffect::UNKNOWN",
    ] {
        common::const_rejected(
            &ADD.replace("OpInfo {}", &format!("OpInfo {{ memory: {value} }}")),
            "executable semantics require no memory effects",
        );
    }
}

#[test]
fn analysis_contracts_are_not_hardcoded_rust_paths() {
    let source = common::source(ADD).replace("veloc_types::OpTraits", "crate::Unrelated");
    let code = veloc_opgen::compile(&source).unwrap();
    assert!(
        code.opcodes
            .contains("<crate::Unrelated as veloc_types::traits::OpTraits>")
    );
}

const PAIR: &str = r#"
struct Pair { args: values(2) }
op Add<T: DOMAIN>(lhs: T, rhs: T) -> T {
    meta: OpInfo {},
    mnemonic: "add", storage: Pair { args: [lhs, rhs] }, semantics: bv.add(lhs, rhs)
}
"#;

fn pair(domain: &str) -> String {
    PAIR.replace("DOMAIN", domain)
}

#[test]
fn declaration_diagnostics() {
    // builtin references are explicit not hidden mir defaults
    {
        common::raw_rejected(ADD, "unknown type or typeset `ScalarInteger`");
        common::raw_rejected(
            "struct Test {} op Test() -> (result: Type::I32) { mnemonic: \"test\", storage: Test {} }",
            "unknown type constant or undeclared Type",
        );
    }

    // malformed constructors alias cycles and removed syntax are rejected
    {
        for (source, message) in [
            ("type A = Missing;", "unknown type"),
            ("type A = Type::I128;", "unknown type"),
            ("type A = B; type B = A;", "cyclic type definition"),
            ("type A = vector(A, 4);", "cyclic type definition"),
            ("type A = int();", "scalar type expects a bit width"),
            ("type A = float(32, 64);", "scalar type expects a bit width"),
            ("type A = int(0);", "scalar width must be"),
            ("type A = float(129);", "scalar width must be"),
            ("type A = bool(1);", "unknown type constructor"),
            ("type A = ptr(64);", "unknown type constructor"),
            (
                "type A = matrix(Type::F32, 4, 4);",
                "unknown type constructor",
            ),
            (
                "type A = vector(Type::I32);",
                "expects an element type and lane count",
            ),
            (
                "type A = vector(Type::I32, 4, true);",
                "expects an element type and lane count",
            ),
            (
                "type A = vector(vector(Type::I32, 4), 4);",
                "must be a scalar type",
            ),
            ("type A = vector(Type::PTR, 4);", "invalid vector"),
            ("type A = vector(Type::I32, scalable());", "vector shape"),
            ("type A = vector(Type::I32, scalable(3));", "vector lanes"),
            (
                "type A = vector(Type::I32, scalable(2147483648));",
                "supported type domain",
            ),
            (
                "type A = Type::I32 | Type::I64;",
                "expected a type name or type constructor",
            ),
            ("type A = Integer;", "unknown type"),
            ("type lower = Type::I32;", "uppercase"),
            ("type INVALID = Type::I32;", "not INVALID"),
            ("type A = Type::I32; type A = Type::I64;", "duplicate type"),
            ("type A = Type::I32", "expected `;`"),
            (
                "scalar A { code: 9, kind: integer, bits: 32 }",
                "unknown definition kind",
            ),
            (
                "vector A { element: Type::I32, lanes: 4 }",
                "unknown definition kind",
            ),
        ] {
            common::raw_rejected(&common::source(source), message);
        }
    }

    // sets reject cycles unknowns duplicates and shadowing
    {
        for (defs, error) in [
            ("typeset A = B; typeset B = A;", "cyclic type set"),
            ("typeset A = A;", "cyclic type set"),
            ("typeset A = Absent;", "unknown type or typeset"),
            ("typeset A = Type::I8 & Type::I16;", "must not be empty"),
            ("typeset A = scalar_integer;", "unknown type or typeset"),
            ("typeset values = Scalar;", "shadows a signature keyword"),
            (
                "type V = Type::I32X4; typeset V = Scalar;",
                "shadows an exact type",
            ),
            ("typeset Scalar = scalar_integer;", "duplicate typeset"),
            ("typeset A = Scalar, typo: 1;", "expected `;`"),
        ] {
            common::raw_rejected(&common::source(defs), error);
        }
    }

    // vector constants reject unrepresentable or invalid types
    {
        for (source, error) in [
            ("type V = vector(Type::PTR, 4);", "invalid vector"),
            ("type V = vector(Type::I32X4, 4);", "must be a scalar type"),
            ("type V = vector(Missing, 4);", "unknown type"),
            ("type V = vector(Type::I32, 0);", "vector lanes"),
            ("type V = vector(Type::I32, 1);", "vector lanes"),
            ("type V = vector(Type::I32, 3);", "vector lanes"),
            ("type V = vector(Type::I32, 65536);", "vector lanes"),
            ("type V = vector(Type::I32, maybe);", "vector shape must be"),
            (
                "type V = Type::I32; type V = vector(Type::I32, 4);",
                "duplicate type",
            ),
            ("type INVALID = vector(Type::I32, 4);", "not INVALID"),
            (
                "type V = Type::I32X4; type V = vector(Type::I32, 4);",
                "duplicate type",
            ),
        ] {
            common::rejected(source, error);
        }
    }
}

#[test]
fn type_domains_and_semantics() {
    // set unions drive both generated contracts and semantic checks
    {
        let output = compile(ADD).unwrap();
        assert!(output.type_rules.contains("C::Bits"));
        assert!(output.opcodes.contains("veloc_types::Scalar::Int(32)"));
        let mixed = ADD.replace("= ScalarInteger;", "= ScalarInteger | ScalarFloat;");
        common::raw_rejected(
            &common::source(&mixed),
            "floating-point execution semantics are not modeled",
        );
        let vectors = ADD.replace("= ScalarInteger;", "= Integer & Vector;");
        assert!(compile(&vectors).is_ok());
    }

    // exact sets drive codegen and bitvector semantics
    {
        let source = r#"
            typeset Wide = Type::I32 | Type::I64;
            struct Pair { args: values(2) }
            op Add<T: Wide>(lhs: T, rhs: T) -> T {
        meta: OpInfo {},
                mnemonic: "add", storage: Pair { args: [lhs, rhs] }, semantics: bv.add(lhs, rhs)
            }
        "#;
        compile(source).unwrap();

        common::rejected(
            &source.replace("Type::I32 | Type::I64", "Type::I32 | Type::F64"),
            "floating-point execution semantics are not modeled",
        );
        assert!(compile(&source.replace("Type::I32 | Type::I64", "Type::I32X4")).is_ok());
    }

    // vector families require scalar sets and preserve definition checks
    {
        for (source, error) in [
            ("typeset Bad = vectors(Type::PTR);", "non-pointer scalar"),
            ("typeset Bad = vectors(Type::I32X4);", "non-pointer scalar"),
            ("typeset Bad = vectors(Any);", "non-pointer scalar"),
            (
                "typeset Bad = vectors(vectors(Type::I32));",
                "non-pointer scalar",
            ),
            ("typeset Bad = vectors(Missing);", "unknown type or typeset"),
            (
                "typeset Bad = vectors();",
                "expected a type, typeset or vectors(set)",
            ),
            (
                "typeset Bad = vectors(Type::I32, Type::I64);",
                "expected a type, typeset or vectors(set)",
            ),
            ("typeset A = vectors(B); typeset B = A;", "cyclic type set"),
            (
                "type V = Type::I32X4; typeset V = Type::I32;",
                "shadows an exact type",
            ),
            ("typeset Bad = vector_integer;", "unknown type or typeset"),
        ] {
            common::rejected(source, error);
        }
        assert!(compile("typeset Mixed = Type::I32 | ScalarInteger;").is_ok());
    }

    // floating text uses domains instead of set name allowlists
    {
        let source = r#"
            typeset Floating = ScalarFloat;
            struct Literal { value: Float }
            op Literal(value: Float) -> (result: Floating) {
        meta: OpInfo { memory: MemoryEffect::NONE },
                mnemonic: "literal", storage: Literal { value: value },
                }
        "#;
        assert!(compile(source).is_ok());
        common::raw_rejected(
            &common::source(&source.replace("= ScalarFloat;", "= vectors(ScalarFloat);")),
            "scalar float",
        );
    }
}

#[test]
fn shape_constraints() {
    // set domains check derived shapes without canonical set names
    {
        let source = r#"
            struct Unary { arg: Value }
            typeset Lanes = ScalarInteger;
            op Element<T: Lanes>(arg: T) -> (result: element(T)) {
        meta: OpInfo { memory: MemoryEffect::NONE },
                mnemonic: "element", storage: Unary { arg: arg }, }
        "#;
        common::raw_rejected(&common::source(source), "impossible element constraint");
        assert!(compile(&source.replace("= ScalarInteger;", "= vectors(ScalarInteger);")).is_ok());
    }

    // exact shapes detect impossible relations at definition time
    {
        let source = r#"
            typeset V4 = Type::I32X4;
            typeset V2 = Type::I64X2;
            struct Unary { arg: Value }
            op Convert<T: V4, U: V2>(arg: T) -> U {
        verify { require(U.same_shape(T), "input and result must have the same shape"); }
        meta: OpInfo { memory: MemoryEffect::NONE },
                mnemonic: "convert", storage: Unary { arg: arg }, }
        "#;
        common::const_rejected(source, "no admissible type signature");
        assert!(compile(&source.replace("Type::I64X2", "Type::F32X4")).is_ok());
        let scalar = source.replace("Type::I64X2", "Type::I64");
        common::const_rejected(&scalar, "no admissible type signature");
    }
}

#[test]
fn parser_depth_limits() {
    // deeply nested type construction is bounded by the parser
    {
        let source = format!(
            "type A = {}Type::I32{};",
            "vector(".repeat(70),
            ", 4)".repeat(70)
        );
        common::raw_rejected(&common::source(&source), "nesting exceeds 64");
    }

    // expression nesting is bounded but flat unions are not recursive
    {
        let deep = format!("{}Type::I32{}", "(".repeat(70), ")".repeat(70));
        common::rejected(&pair(&deep), "nesting exceeds 64");
        let flat = vec!["Type::I32"; 1000].join(" | ");
        assert!(compile(&pair(&flat)).is_ok());
    }
}
