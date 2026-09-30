//! Wire format for ordered rule queries and value-construction recipes.
//! Execution and edit tracking belong to the consuming runtime.
//! Type and feature sets index program tables. Operand references have a shared
//! codec; runtimes and generators use explicit input/result identities.

pub use crate::OperandRef;

/// A signature visits results before inputs. Each Bind introduces the next
/// local slot; Same refers to a slot already bound in this signature match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TypePattern {
    Set(usize),
    Bind(usize),
    Same(usize),
}

impl TypePattern {
    pub fn encode(self) -> usize {
        let (index, tag) = match self {
            Self::Set(set) => (set, 0),
            Self::Bind(set) => (set, 1),
            Self::Same(slot) => (slot, 2),
        };
        index.checked_mul(4).expect("type pattern index overflow") | tag
    }

    pub fn decode(value: usize) -> Self {
        match value & 3 {
            0 => Self::Set(value >> 2),
            1 => Self::Bind(value >> 2),
            2 => Self::Same(value >> 2),
            _ => panic!("invalid type pattern tag"),
        }
    }
}

crate::bytecode! {
    pub enum Instruction, Opcode {
        Reject {},
        Jump { target: u32 },
        CheckSignature { results: [uleb], inputs: [uleb], failure: u32 },
        CheckType { value: uleb, set: uleb, failure: u32 },
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
