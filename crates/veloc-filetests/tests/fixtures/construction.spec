type I32 = int(32);
type Type = rust("crate::Ty");
type Query = rust("crate::Q") {
    fn opcode(&self) -> RewriteOpcode;
    fn arity(&self, result: bool) -> usize;
    fn value_type(&self, result: bool, index: u32) -> Type;
}
typeset Word = Type::I32;

fn host_twice<T: Word>(x: T) -> T = rust("crate::twice");
// Rust must also check the signature of this unreferenced binding.
fn unused<T: Word>(x: T) -> T = rust("crate::twice");
fn compose<T: Word>(x: T) -> T {
    let doubled = host_twice<T>(x);
    host_twice<T>(lir::Add<T>(doubled, doubled))
}
select(inst: lir::Ctpop<Type::I32>) {
    replace(inst, build(compose<Type::I32>(inst.src)));
}
select(inst: lir::Ctlz<Type::I32>) {
    let x = build(lir::Add<Type::I32>(inst.src, inst.src));
    let y = build(host_twice<Type::I32>(x));
    replace(inst, x);
}

type RewriteValue = rust("crate::Value");
type RewriteOpcode = rust("crate::Opcode");
type RewriteField = rust("crate::Field");
type RewriteContext = rust("crate::Eval") {
    fn emit(&mut self, opcode: RewriteOpcode, ty: Type,
        inputs: sequence(RewriteValue), fields: sequence(RewriteField),
        result: optional(RewriteValue)) -> RewriteValue;
}

rewrite_interface ValueRules {
    contract = RewriteContext;
    emit = emit;
}
