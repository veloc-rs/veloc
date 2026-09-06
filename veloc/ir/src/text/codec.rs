//! Text grammar implementations selected by the MIR storage schema.
//!
//! The codec vocabulary names the parser/printer algorithms. Format selection
//! and memory-flag support are generated from the same field definitions as
//! [`InstructionData`](crate::InstructionData).

use crate::{Opcode, opspec::OpFormat};

/// Operand grammar used by an opcode in the canonical text format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextCodec {
    /// A comma-separated, fixed-arity list of SSA values.
    Values {
        arity: u8,
    },
    Nullary,
    IntegerConstant,
    FloatConstant,
    BoolConstant,
    VectorConstant,
    Load,
    Store,
    StackLoad,
    StackStore,
    StackAddr,
    PtrOffset,
    PtrIndex,
    DirectCall,
    IndirectCall,
    IntrinsicCall,
    Jump,
    Branch,
    BranchTable,
    Return,
    IntCompare,
    FloatCompare,
    VectorLoadStrided,
    VectorStoreStrided,
    VectorGather,
    VectorScatter,
    Shuffle,
}

include!(concat!(env!("OUT_DIR"), "/codecs.rs"));

impl TextCodec {
    pub const fn for_opcode(opcode: Opcode) -> Self {
        Self::for_format(opcode.spec().format)
    }
}

#[cfg(test)]
mod tests {
    use super::TextCodec;
    use crate::Opcode;

    #[test]
    fn every_opcode_has_a_text_codec() {
        for &opcode in Opcode::ALL {
            let _ = TextCodec::for_opcode(opcode);
        }
    }
}
