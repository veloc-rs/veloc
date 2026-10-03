import "../../defs/prelude.spec";

type GlobalId = rust("crate::GlobalId");

// Operand groups retain only lengths and control-flow metadata in instruction fields.
type ValueList = rust("crate::inst::Arguments") {
    field = list(Value);
}
type Successor = rust("crate::inst::Successor") {
    field = edge(Value);
}
type JumpTable = rust("crate::inst::Successors") {
    field = list(Successor);
}

// Contexts are ordinary Rust-bound types. Borrowed results are tied to &self.
type Signature = rust("veloc_types::Signature") {
    trait = rust("crate::type_methods::SignatureInfo");
    fn params(&self) -> sequence(Type);
    fn returns(&self) -> sequence(Type);
    fn types(&self) -> sequence(Type);
    fn is_variadic(&self) -> bool;
    fn same_call_conv(&self, other: &Signature) -> bool;
}

type VerifyContext = rust("crate::host::VerifyContext") {
    trait = rust("crate::type_methods::VerifyContextInfo");
    fn vector_constant(&self, value: Value) -> optional(&VectorConst);
    fn function_signature(&self, func: FuncId) -> optional(&Signature);
    fn has_global(&self, global: GlobalId) -> bool;
    fn current_signature(&self) -> optional(&Signature);
    fn signature(&self, sig: SigId) -> optional(&Signature);
}
