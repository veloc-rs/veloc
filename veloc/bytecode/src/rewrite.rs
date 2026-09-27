//! Wire format for ordered rule queries and value-construction recipes.
//! Execution and edit tracking belong to the consuming runtime.
//! Query value references encode `(index << 1) | result`: inputs have low bit
//! zero, results have low bit one. Type and feature sets index program tables.
crate::bytecode! {
    pub enum Instruction, Opcode {
        Reject {},
        Jump { target: u32 },
        CheckSignature { results: [uleb], inputs: [uleb], failure: u32 },
        CheckSameType { values: [uleb], failure: u32 },
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
