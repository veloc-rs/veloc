//! Queries and value-construction recipes, with a host-supplied type codec.
pub use crate::OperandRef;
use crate::codec::{Operand, RawWord, Word, WordCodec};
use crate::signature::PatternsCodec;

crate::bytecode! {
    pub enum Instruction<T: WordCodec>, Opcode {
        Reject {},
        Jump { target: u32 },
        CheckSignature {
            results: (codec PatternsCodec<T>),
            inputs: (codec PatternsCodec<T>),
            failure: u32
        },
        CheckType { value: (codec Operand), ty: (codec Word<T>), failure: u32 },
        CheckSignedRange { field: uleb, bits: uleb, expected: uleb, failure: u32 },
        CheckFeatures { set: uleb, failure: u32 },
        Accept { action: uleb },
        // opcode is the definition-order IR opcode, not a program-local index.
        Emit { opcode: uleb, ty: uleb, inputs: [uleb], fields: [uleb], dst: uleb, reuse: uleb },
        Return { value: uleb },
        // Pairs of (operand index, value slot) and (field index, field source).
        Update { inputs: [uleb], fields: [uleb] },
    }
}

pub type RawInstruction<'a> = Instruction<'a, RawWord>;
