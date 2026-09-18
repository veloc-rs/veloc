import "type_sets.spec";

type Value = rust("crate::Value") {
    field = operand;
    // Fundamental SSA type query; callers use the declared method.
    fn ty(self) -> Type { value = type(self); }
}

// Operand groups retain only lengths and control-flow metadata in instruction fields.
type ValueList = rust("crate::inst::Arguments") {
    field = list(Value);
}
type BlockCall = rust("crate::inst::Successor") {
    field = edge(Value);
}
type JumpTable = rust("crate::inst::Successors") {
    field = list(BlockCall);
}

// Rust-owned data types used by operation properties and host interfaces.
type Int = rust("crate::Int") {
    fn ty(self) -> Type { value = type(self); }
}
type Float = rust("crate::Float") {
    fn ty(self) -> Type { value = type(self); }
}
type VectorConst = rust("crate::VectorConst") {
    trait = rust("crate::type_methods::VectorConstInfo");
    fn ty(self) -> Type { value = type(self); }
    const fn is_dense(self) -> bool;
    const fn encoded_size(self) -> optional(u32);
}
type FuncId = rust("crate::FuncId");
type Intrinsic = rust("crate::Intrinsic");
type ConstantPoolId = rust("crate::inst::ConstantPoolId");
type SymbolId = rust("crate::SymbolId");

// Shared effect interfaces are imported from the Rust owner.

struct OpInfo {
    traits: OpTraits,
    memory: MemoryEffect,
}

// Reusable checked arithmetic, not a special verifier query.
fn is_power_of_two(value: u32) -> bool {
    value = value != 0 && (value & (value - 1)) == 0;
}

// Pairwise type checking is an IR policy, not a language primitive.
// all rejects unequal lengths before evaluating the predicate.
fn matches_types(values: sequence(Value), types: sequence(Type)) -> bool {
    value = all(values, types, |value, ty| value.ty() == ty);
}
