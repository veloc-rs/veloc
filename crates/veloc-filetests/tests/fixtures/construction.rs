#![allow(dead_code, unused_variables)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Ty {
    I32,
}
#[derive(Clone, Copy)]
pub enum Opcode {
    Ctpop,
    Ctlz,
    Add,
}
pub struct Q;
impl generated::Query for Q {
    fn signature(&self, _: &[&[Ty]], _: &[&[Ty]]) -> bool {
        true
    }
}
pub type Field = ();
pub struct Action(fn(&mut Eval));
pub fn rewrite(_: &str, f: fn(&mut Eval)) -> Action {
    Action(f)
}
pub fn replace_values(ctx: &mut Eval, body: impl FnOnce(&mut Eval, &[u32], &[Ty], u32) -> u32) {
    let input = ctx.input;
    ctx.result = body(ctx, &[input], &[Ty::I32, Ty::I32], 0);
}
pub struct Eval {
    input: u32,
    result: u32,
    emissions: usize,
    host_calls: usize,
}
pub type Value = u32;
impl generated::RewriteContext for Eval {
    fn emit(
        &mut self,
        _: Opcode,
        _: Ty,
        inputs: &[u32],
        fields: &[Field],
        result: Option<u32>,
    ) -> u32 {
        assert!(fields.is_empty());
        self.emissions += 1;
        let value = inputs[0].wrapping_add(inputs[1]);
        if result.is_some() {
            self.result = value;
        }
        value
    }
}
fn twice<C: generated::RewriteContext>(ctx: &mut C, ty: Ty, x: Value) -> Value {
    ctx.emit(Opcode::Add, ty, &[x, x], &[], None)
}
fn main() {
    for input in [0, 1, 99, u32::MAX] {
        let mut eval = Eval {
            input,
            result: 0,
            emissions: 0,
            host_calls: 0,
        };
        let plan = generated::decide(Opcode::Ctpop, &Q).unwrap();
        assert_eq!(eval.emissions, 0);
        (plan.0)(&mut eval);
        assert_eq!(eval.result, input.wrapping_mul(8));
        assert_eq!(eval.emissions, 3);
        eval.emissions = 0;
        let plan = generated::decide(Opcode::Ctlz, &Q).unwrap();
        (plan.0)(&mut eval);
        assert_eq!(eval.result, input.wrapping_mul(2));
        assert_eq!(eval.emissions, 2);
    }
}
