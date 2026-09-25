//! Shared wire format for equality-saturation rules. No MIR dependencies.
//!
//! Slot zero holds the root throughout a rule. `Next` stores `plans`
//! consecutive column-to-slot maps in `bindings`, and enumerates each map
//! for each relation row. `Apply` ends the read-only search; `NextMatch`
//! restores captured values into slots starting at one.
//! Both constant predicates require a known constant, including `Ne`.

crate::bytecode! {
    pub enum Instruction, Opcode {
        CheckTypeIn { value: u32, types: u32, failure: u32 },
        CheckSameType { lhs: u32, rhs: u32, failure: u32 },
        StopIfConstant {},
        Begin { name: u32, captures: u32, len: u32 },
        Open { cursor: u32, source: u32, opcode: u32 },
        Next { cursor: u32, plans: u32, failure: u32, bindings: [u32] },
        CheckEqual { lhs: u32, rhs: u32, failure: u32 },
        CheckConstantEq { value: u32, constant: u32, failure: u32 },
        CheckConstantNe { value: u32, constant: u32, failure: u32 },
        Capture {},
        Apply {},
        Jump { target: u32 },
        Constant { dst: u32, constant: u32, failure: u32 },
        Build { dst: u32, opcode: u32, args: [u32], failure: u32 },
        Union { value: u32 },
        SetConstant { constant: u32 },
        NextMatch { failure: u32 },
        Return {},
    }
}
