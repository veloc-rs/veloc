//! Shared wire format for equality-saturation rules. No MIR dependencies.
//!
//! Slot zero holds the root throughout a group. `OpenScan` identifies a scan
//! independently of byte offsets and references a static table of bindings. `ScanNext`
//! yields one binding satisfying its input constraint, or exits on exhaustion.
//! Queries end with `Return`. `Capture` carries a rule ID and the inline slots
//! required by its checked construction plan: `values` are class bindings and
//! `nodes` retain concrete instruction witnesses for attribute matching.
//! The host saves those bindings and
//! applies the shared plan after enumeration; this VM performs no IR mutation.
//! Incremental inputs select query entries compiled for all relevant rules;
//! only Capture names a rule. Checks branch on `otherwise`, while iterators
//! leave their loop on `exhausted`.
//! Both constant predicates require a known constant, including `Ne`.

crate::bytecode! {
    pub enum Instruction, Opcode {
        CheckTypeIn { value: u32, types: u32, otherwise: u32 },
        CheckSameType { lhs: u32, rhs: u32, otherwise: u32 },
        OpenScan { scan: u32, cursor: u32, source: u32, opcode: u32, bindings: u32 },
        ScanNext { cursor: u32, exhausted: u32 },
        CheckEqual { lhs: u32, rhs: u32, otherwise: u32 },
        CheckIsConstant { value: u32, otherwise: u32 },
        CheckConstantEq { value: u32, constant: u32, otherwise: u32 },
        CheckConstantNe { value: u32, constant: u32, otherwise: u32 },
        CheckProperties { lhs: u32, rhs: u32, predicate: u32, otherwise: u32 },
        Capture { rule: u32, values: [u32], nodes: [u32] },
        Jump { target: u32 },
        Return {},
    }
}
