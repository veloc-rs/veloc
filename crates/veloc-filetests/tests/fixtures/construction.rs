#![allow(dead_code, unused_variables)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Ty {
    I32,
}
impl generated::Type for Ty {
    const I32: Self = Self::I32;
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
pub struct Action(fn(&mut Eval));
pub fn values(_: &str, f: fn(&mut Eval)) -> Action {
    Action(f)
}
pub struct Eval {
    input: u32,
    result: u32,
    emissions: usize,
    host_calls: usize,
}
impl generated::ValueBuild for Eval {
    type Value = u32;
    fn emit(&mut self, _: Opcode, _: Ty, inputs: &[u32]) -> u32 {
        self.emissions += 1;
        inputs[0].wrapping_add(inputs[1])
    }
    fn emit_integer(&mut self, _: Opcode, _: Ty, value: i64) -> u32 {
        self.emissions += 1;
        value as u32
    }
}
impl generated::ValueRewrite for Eval {
    fn input(&self, _: usize) -> u32 {
        self.input
    }
    fn value_type(&self, _: bool, _: usize) -> Ty {
        Ty::I32
    }
    fn emit_at(&mut self, op: Opcode, ty: Ty, inputs: &[u32], result: Option<usize>) -> u32 {
        let value = generated::ValueBuild::emit(self, op, ty, inputs);
        if result.is_some() {
            self.result = value;
        }
        value
    }
    fn emit_integer_at(&mut self, op: Opcode, ty: Ty, n: i64, result: Option<usize>) -> u32 {
        let value = generated::ValueBuild::emit_integer(self, op, ty, n);
        if result.is_some() {
            self.result = value;
        }
        value
    }
    fn bind(&mut self, _: usize, value: u32) {
        self.result = value;
    }
}
// The host implementation sees construction capabilities, not a root or a DFG.
fn twice<C: generated::ValueBuild>(ctx: &mut C, ty: Ty, x: C::Value) -> C::Value {
    ctx.emit(Opcode::Add, ty, &[x, x])
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
