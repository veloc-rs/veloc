#![allow(dead_code, unused_variables)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    I32,
}
#[derive(Clone, Copy)]
pub enum Opcode {
    Ctpop = CTPOP_CODE,
    Ctlz = CTLZ_CODE,
    Add = ADD_CODE,
}
impl Opcode {
    fn from_code(code: usize) -> Option<Self> {
        match code {
            n if n == Self::Ctpop as usize => Some(Self::Ctpop),
            n if n == Self::Ctlz as usize => Some(Self::Ctlz),
            n if n == Self::Add as usize => Some(Self::Add),
            _ => None,
        }
    }
}
pub struct Q(Opcode);
impl Q {
    fn opcode(&self) -> Opcode {
        self.0
    }
    fn arity(&self, _: bool) -> usize {
        1
    }
    fn value_type(&self, _: bool, _: u32) -> Ty {
        Ty::I32
    }
}
pub type Field = ();
pub struct Action(&'static vm::Program, usize, usize);
pub fn replace_values(ctx: &mut Eval, body: impl FnOnce(&mut Eval, &[u32], &[Ty], u32) -> u32) {
    let input = ctx.input;
    ctx.result = body(ctx, &[input], &[Ty::I32, Ty::I32], 0);
}
pub struct Eval {
    input: u32,
    result: u32,
    emissions: usize,
}
pub type Value = u32;
impl Eval {
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
fn main() {
    for input in [0, 1, 99, u32::MAX] {
        let mut eval = Eval {
            input,
            result: 0,
            emissions: 0,
        };
        let program = &generated::DECIDE;
        let entry = program.entries[Opcode::Ctpop as usize].unwrap();
        let plan = vm::select(program, entry, &Q(Opcode::Ctpop)).unwrap();
        assert_eq!(eval.emissions, 0);
        vm::apply(plan, &mut eval);
        assert_eq!(eval.result, input.wrapping_mul(8));
        assert_eq!(eval.emissions, 3);
        eval.emissions = 0;
        let entry = program.entries[Opcode::Ctlz as usize].unwrap();
        let plan = vm::select(program, entry, &Q(Opcode::Ctlz)).unwrap();
        vm::apply(plan, &mut eval);
        assert_eq!(eval.result, input.wrapping_mul(2));
        assert_eq!(eval.emissions, 2);
    }
}

mod vm {
    use super::*;
    pub use veloc_bytecode::OperandRef;
    use veloc_bytecode::{Reader, codec::WordCodec, signature::TypePattern};

    #[derive(Clone, Copy, Debug)]
    pub struct TypeCodec;
    impl TypeCodec {
        pub const fn encode(ty: Ty) -> usize {
            match ty {
                Ty::I32 => 0,
            }
        }
    }
    impl WordCodec for TypeCodec {
        type Value = Ty;
        fn decode(word: usize) -> Ty {
            assert_eq!(word, 0);
            Ty::I32
        }
        fn encode(ty: Ty) -> usize {
            Self::encode(ty)
        }
    }
    type Op<'a> = veloc_bytecode::rewrite::Instruction<'a, TypeCodec>;
    pub enum FieldSource {
        Constant(Field),
        Root(usize),
    }
    impl FieldSource {
        fn read(&self) -> Field {
            match self {
                Self::Constant(field) => *field,
                Self::Root(_) => unreachable!("fixture has no runtime attributes"),
            }
        }
    }
    pub enum TypeSource {
        Exact(Ty),
        Value(OperandRef),
    }
    #[derive(Clone, Copy)]
    pub enum Action {
        Legal,
        Recipe {
            name: &'static str,
            entry: usize,
            slots: usize,
        },
    }
    pub struct Program {
        pub entries: &'static [Option<usize>],
        pub code: &'static [u8],
        pub features: &'static [&'static [u64]],
        pub actions: &'static [Action],
        pub types: &'static [TypeSource],
        pub fields: &'static [FieldSource],
    }
    pub fn select(program: &'static Program, entry: usize, query: &Q) -> Option<super::Action> {
        let mut reader = Reader {
            bytes: program.code,
            pc: entry,
        };
        let action = loop {
            match Op::read(&mut reader) {
                Op::Reject {} => return None,
                Op::Jump { target } => reader.pc = target,
                Op::CheckSignature {
                    results,
                    inputs,
                    failure,
                } => {
                    let matches = |result,
                                   patterns: veloc_bytecode::signature::TypePatterns<
                        '_,
                        TypeCodec,
                    >| {
                        let mut patterns = patterns.iter();
                        (0..query.arity(result)).all(|i| {
                            matches!(patterns.next(), Some(TypePattern::Exact(ty))
                                if ty == query.value_type(result, i as u32))
                        }) && patterns.next().is_none()
                    };
                    if !matches(true, results) || !matches(false, inputs) {
                        reader.pc = failure;
                    }
                }
                Op::Accept { action } => break action,
                _ => panic!("construction instruction in read-only query"),
            }
        };
        let Action::Recipe { entry, slots, .. } = program.actions[action] else {
            panic!("expected recipe")
        };
        Some(super::Action(program, entry, slots))
    }
    pub fn apply(super::Action(program, entry, slots): super::Action, ctx: &mut Eval) {
        replace_values(ctx, |ctx, inputs, types, destination| {
            let mut values = vec![0; slots];
            values[..inputs.len()].copy_from_slice(inputs);
            let ty = |id: usize| match program.types[id] {
                TypeSource::Exact(ty) => ty,
                TypeSource::Value(operand) => *operand.get(&types[1..], &types[..1]).unwrap(),
            };
            let mut reader = Reader {
                bytes: program.code,
                pc: entry,
            };
            loop {
                match Op::read(&mut reader) {
                    Op::Emit {
                        opcode,
                        ty: t,
                        inputs,
                        fields,
                        dst,
                        reuse,
                    } => {
                        let inputs: Vec<_> = inputs.iter().map(|i| values[i]).collect();
                        let fields: Vec<_> =
                            fields.iter().map(|i| program.fields[i].read()).collect();
                        values[dst] = ctx.emit(
                            Opcode::from_code(opcode).expect("generated opcode"),
                            ty(t),
                            &inputs,
                            &fields,
                            (reuse != 0).then_some(destination),
                        );
                    }
                    Op::Return { value } => return values[value],
                    _ => panic!("invalid recipe"),
                }
            }
        });
    }
}
