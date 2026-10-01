//! Shared instruction-selection wire format. Compact indices and fixed jump offsets.
use crate::codec::Operand;

crate::bytecode! {
    pub enum Instruction, Opcode {
        Reject {},
        Jump { target: u32 },
        ReadReg { dst: uleb, node: uleb, operand: (codec Operand) },
        GetDef { dst: uleb, value: uleb, failure: u32 },
        CheckOpcode { node: uleb, opcode: u32, failure: u32 },
        CheckType { value: uleb, set: uleb, failure: u32 },
        CheckInt { node: uleb, index: uleb, constant: i64, failure: u32 },
        CheckIntRange { node: uleb, index: uleb, bits: uleb, signed: uleb, failure: u32 },
        CheckFeatures { set: uleb, failure: u32 },
        CallPredicate { value: uleb, id: uleb, failure: u32 },
        CheckFoldable { definition: uleb, consumer: uleb, failure: u32 },
        Accept {},
        MakeTemp { dst: uleb, ty: uleb },
        ReadResult { dst: uleb, index: uleb },
        ConstReg { dst: uleb, reg: uleb },
        ReadField { dst: uleb, node: uleb, index: uleb },
        ConstImm { dst: uleb, imm: i64 },
        BuildInst { target: uleb, results: [uleb], inputs: [uleb], fields: [uleb] },
        Finish {},
    }
}
