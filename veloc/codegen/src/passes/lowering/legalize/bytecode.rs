//! Runtime binding of the shared rewrite schema to semantic LIR types.
use veloc_bytecode::codec::WordCodec;
use veloc_lir::Type;

#[derive(Clone, Copy, Debug)]
pub struct TypeCodec;

impl TypeCodec {
    /// Also used by generated constant expressions to fill inline type operands.
    pub const fn encode(ty: Type) -> usize {
        ty.to_raw() as usize
    }
}

impl WordCodec for TypeCodec {
    type Value = Type;

    fn decode(word: usize) -> Type {
        Type::from_raw(u16::try_from(word).expect("type code overflow"))
            .expect("invalid bytecode type")
    }

    fn encode(ty: Type) -> usize {
        Self::encode(ty)
    }
}

pub type TypePatterns<'a> = veloc_bytecode::signature::TypePatterns<'a, TypeCodec>;
pub type Instruction<'a> = veloc_bytecode::rewrite::Instruction<'a, TypeCodec>;
