//! Shared wire format for equality-saturation rules. No MIR dependencies.
//!
//! Slot zero holds the root throughout a group. `OpenScan` identifies a scan
//! independently of byte offsets and references a static table of bindings. `ScanNext`
//! yields one binding satisfying its input constraint, or exits on exhaustion.
//! Query and rewrite entries both end with `Return`. The host batches queries
//! against a stable graph, then restores captures into slots starting at one
//! before calling each rewrite entry. Capture records a rule ID, so zero-input
//! matches remain distinguishable. Phase scheduling is not part of the VM.
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
        CheckConstantEq { value: u32, constant: u32, otherwise: u32 },
        CheckConstantNe { value: u32, constant: u32, otherwise: u32 },
        Capture { rule: u32 },
        Jump { target: u32 },
        Constant { dst: u32, constant: u32 },
        Build { dst: u32, opcode: u32, args: [u32] },
        Union { value: u32 },
        SetConstant { constant: u32 },
        Return {},
    }
}
