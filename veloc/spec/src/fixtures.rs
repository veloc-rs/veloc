//! Synthetic definition language fixtures. No runtime crate or production defs.
use crate::{Definitions, Error};

const SOURCE: &str = r#"
type I8 = int(8);
type I16 = int(16);
type I32 = int(32);
type I64 = int(64);
type F32 = float(32);
type F64 = float(64);
type BOOL = bool;
type PTR = ptr;
type I32X4 = vector(I32, 4);
type I64X2 = vector(I64, 2);
type Type = rust("test::Type");
type Value = rust("test::Value") { field: operand, }
type Float = rust("test::Float") { fn ty(self) -> Type { value: type(self) } }
type Int = rust("test::Int") { fn ty(self) -> Type { value: type(self) } }
type VectorConst = rust("test::VectorConst") { fn ty(self) -> Type { value: type(self) } }

// Rust owns the representation; declarations are checked by generated traits.
type OpTraits = rust("test::OpTraits") {
    analysis: traits,
    const TERMINATOR: Self;
    const COMMUTATIVE: Self;
    const ASSOCIATIVE: Self;
    const IDEMPOTENT: Self;
    const MAY_TRAP: Self;
    const ABORT: Self;
    const fn empty() -> Self;
    const fn union(self, other: Self) -> Self;
    const fn contains(self, other: Self) -> bool;
}
type MemoryEffects = rust("test::MemoryEffects") {
    const READ: Self;
    const WRITE: Self;
    const ALLOCATE: Self;
    const FREE: Self;
    const fn empty() -> Self;
    const fn union(self, other: Self) -> Self;
}
type MemoryEffect = rust("test::MemoryEffect") {
    analysis: memory(MemoryEffect::NONE),
    const NONE: Self;
    const UNKNOWN: Self;
    const fn known(effects: MemoryEffects) -> Self;
    const fn is_none(self) -> bool;
}
type MemFlags = rust("test::MemFlags");

struct OpInfo { traits: OpTraits, memory: MemoryEffect }
typeset ScalarInteger = Type::I8 | Type::I16 | Type::I32 | Type::I64;
typeset ScalarFloat = Type::F32 | Type::F64;
typeset Scalar = ScalarInteger | ScalarFloat | Type::BOOL;
typeset Integer = ScalarInteger | vectors(ScalarInteger);
typeset Float = ScalarFloat | vectors(ScalarFloat);
typeset Vector = vectors(Scalar);
typeset Any = Scalar | Type::PTR | Vector;
"#;

pub fn parse(source: &str) -> Result<Definitions, Error> {
    crate::parse(&format!("{SOURCE}\n{source}"))
}
pub fn types() -> crate::types::Types {
    parse("").unwrap().types
}
pub fn types_policy(name: &str) -> crate::model::records::Policy {
    parse("").unwrap().data.rust.policy(name)
}
pub fn set(expression: &str) -> crate::types::TypeSet {
    parse(&format!("typeset TestSet = {expression};"))
        .unwrap()
        .types
        .sets
        .remove("TestSet")
        .unwrap()
}
