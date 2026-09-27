type I32 = int(32);
type Type = rust("crate::Ty");
typeset Word = Type::I32;

fn twice<T: Word>(x: T) -> T {
    lir::Add<T>(x, x)
}
fn compose<T: Word>(x: T) -> T {
    let doubled = twice<T>(x);
    twice<T>(lir::Add<T>(doubled, doubled))
}
select(inst: lir::Ctpop<Type::I32>) {
    replace(inst, build(compose<Type::I32>(inst.src)));
}
select(inst: lir::Ctlz<Type::I32>) {
    let x = build(lir::Add<Type::I32>(inst.src, inst.src));
    let y = build(twice<Type::I32>(x));
    replace(inst, x);
}
