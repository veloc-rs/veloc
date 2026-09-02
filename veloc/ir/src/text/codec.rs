//! Declarative bridge between semantic opcode formats and the textual IR.
//!
//! The core [`OpSpec`](crate::OpSpec) deliberately does not know about a parser
//! or a printer.  This closed mapping is the text dialect's view of those
//! formats.  Keeping it exhaustive means adding a new [`OpFormat`] cannot
//! silently fall through to an "unsupported instruction" path.

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

impl TextCodec {
    /// Select the textual codec from the physical format in `OpSpec`.
    ///
    /// This match intentionally has no wildcard arm.  A new physical format
    /// therefore requires an explicit text-format decision at compile time.
    pub const fn for_format(format: OpFormat) -> Self {
        match format {
            OpFormat::Unary | OpFormat::IntToPtr | OpFormat::PtrToInt => Self::Values { arity: 1 },
            OpFormat::Binary => Self::Values { arity: 2 },
            OpFormat::Ternary => Self::Values { arity: 3 },
            OpFormat::Iconst => Self::IntegerConstant,
            OpFormat::Fconst => Self::FloatConstant,
            OpFormat::Bconst => Self::BoolConstant,
            OpFormat::Vconst => Self::VectorConstant,
            OpFormat::Load => Self::Load,
            OpFormat::Store => Self::Store,
            OpFormat::StackLoad => Self::StackLoad,
            OpFormat::StackStore => Self::StackStore,
            OpFormat::StackAddr => Self::StackAddr,
            OpFormat::PtrOffset => Self::PtrOffset,
            OpFormat::PtrIndex => Self::PtrIndex,
            OpFormat::Call => Self::DirectCall,
            OpFormat::CallIndirect => Self::IndirectCall,
            OpFormat::CallIntrinsic => Self::IntrinsicCall,
            OpFormat::Jump => Self::Jump,
            OpFormat::Br => Self::Branch,
            OpFormat::BrTable => Self::BranchTable,
            OpFormat::Return => Self::Return,
            OpFormat::IntCompare => Self::IntCompare,
            OpFormat::FloatCompare => Self::FloatCompare,
            OpFormat::VectorLoadStrided => Self::VectorLoadStrided,
            OpFormat::VectorStoreStrided => Self::VectorStoreStrided,
            OpFormat::VectorGather => Self::VectorGather,
            OpFormat::VectorScatter => Self::VectorScatter,
            OpFormat::Shuffle => Self::Shuffle,
            OpFormat::Unreachable | OpFormat::Nop => Self::Nullary,
        }
    }

    pub const fn for_opcode(opcode: Opcode) -> Self {
        Self::for_format(opcode.spec().format)
    }

    pub const fn accepts_memory_flags(self) -> bool {
        matches!(
            self,
            Self::Load
                | Self::Store
                | Self::VectorLoadStrided
                | Self::VectorStoreStrided
                | Self::VectorGather
                | Self::VectorScatter
        )
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
