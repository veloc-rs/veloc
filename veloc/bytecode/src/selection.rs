//! Shared instruction-selection wire format. Compact indices and fixed jump offsets.
crate::bytecode! {
    pub enum Instruction, Opcode {
        Reject {},
        Jump { target: u32 },
        ReadReg { dst: uleb, node: uleb, field: uleb },
        GetDef { dst: uleb, value: uleb, failure: u32 },
        CheckOpcode { node: uleb, opcode: uleb, failure: u32 },
        CheckType { value: uleb, set: uleb, failure: u32 },
        CheckInt { node: uleb, field: uleb, constant: uleb, failure: u32 },
        CheckFeatures { set: uleb, failure: u32 },
        CallPredicate { value: uleb, id: uleb, failure: u32 },
        CheckFoldable { definition: uleb, consumer: uleb, failure: u32 },
        Accept {},
        MakeTemp { dst: uleb, ty: uleb },
        ReadResult { dst: uleb, index: uleb },
        ConstReg { dst: uleb, reg: uleb },
        ReadField { dst: uleb, node: uleb, field: uleb },
        ConstImm { dst: uleb, imm: uleb },
        BuildInst { target: uleb, results: [uleb], inputs: [uleb], fields: [uleb] },
        Finish {},
    }
}
