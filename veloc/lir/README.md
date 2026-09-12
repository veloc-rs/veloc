# Low-level IR

`veloc-lir` owns the machine-facing representation: instructions and operand
schemas, registers and register banks, functions and blocks, stack-frame data,
symbols, use-def chains and stage markers. It supports `no_std` with `alloc` and
has no dependency on codegen or any target backend.

Codegen owns MIR-to-LIR translation, legalization, register-bank selection,
instruction selection, register allocation, ABI handling and machine-code
emission. Pipeline scheduling and transitions stay there; stage markers live
here because they parameterize `MachineFunction` and its allocation APIs.
Markers do not independently prove that a pass established its postconditions.

Consumers import `veloc_lir::{MachineFunction, MachineInst, ...}` and
`veloc_lir::stages::RawLir`, or use the top-level `veloc::lir` facade. There is no
compatibility module at `veloc_codegen::lir`.

The public function data is mutable IR storage for transformation passes.
Instruction-extra pools remain private behind their accessors. Schema decoding
returns `DecodeError`; codegen wraps it without making LIR depend on backend
error types. Symbol interning takes a name and linkage, not a MIR module.

`defs/module.ops` imports the shared `veloc/defs/prelude.ops` and the local
`generic.ops`. Both MIR and LIR use `veloc_opgen::Source::load(...).compile()`,
one logical operation model and one type/semantic checker. Import resolution,
dependency tracking and physical-file diagnostics are shared build infrastructure.

`storage Operands` selects operand-array projection; MIR uses packed storage.
Machine `struct` declarations describe register roles and properties. Logical `op`
records declare signatures, effects, control behavior and optional semantics.
The generated Rust contains opcodes, views, builders, decoding checks and direct
type validation. Construction does not run validation.

```text
storage Operands { prefix: "G_" }
struct BinaryReg { dst: Def, lhs: Use, rhs: Use }
op G_ADD<T: Integer>(lhs: T, rhs: T) -> (dst: T) {
    meta: OpInfo {},
    storage: BinaryReg { dst, lhs, rhs },
    semantics: bv.add(lhs, rhs)
}
```

Fixed pure register recipes use logical input/result order. The reviewed primitive
set is retained; this is not a proof of target legalization or encoding. Codegen
joins MIR and LIR primitive bindings to generate direct lowering dispatch.
Offline tools can include generated semantic artifacts; runtime LIR has no
semantics dependency.

Instruction access uses `inst.generic_view()?` and pattern matching:

```rust,ignore
match inst.generic_view()? {
    InstView::BinaryReg(binary) => {
        // The restricted opcode distinguishes ADD, SUB, etc. sharing this shape.
        use_binary(binary.opcode, binary.dst, binary.lhs, binary.rhs);
    }
    InstView::Return(ret) => {
        for reg in ret.values.iter() {
            use_return(reg);
        }
    }
    _ => {}
}
```

Views are generated from structs; there are no accessor-name overrides or runtime
schema dispatch. The decoder checks opcode, operand count and operand kinds,
not semantic type contracts. Fixed fields are copied; variable register lists
borrow the operands through `RegList`, without allocation. Typed payload structs
allow passing an already decoded instruction to helpers. Generic views reject
target/invalid opcodes; target-specific decoding remains the backend's concern.

Fixed builders are named from the opcode (`G_BRCOND` -> `build_brcond`). Variadic
call/return construction and decoding, and architecture-independent binary/tied
construction helpers, remain ordinary Rust. Explicit mappings bind every layout field to logical inputs or named results.
Field shorthand expands to `field: field`; there is no name/order-based layout
inference. Builders take results in signature order, then inputs in signature
order, and encode them in physical layout order. A `tied(input, result)` field
uses a single writable register for both, so it contributes only the result
argument to the physical builder. Optional uses require `some(input)` or `none`;
only trailing fields may be absent. Carry instructions therefore select their
exact arity explicitly. Variadic returns map `values` directly; call storage uses
`shape: call(callee, args)` with signature-driven results. `G_ICMP` always
requires two inputs and an explicit condition; zero comparison is `G_IEQZ`.

Logical type contracts now use the same compiler as MIR and can be checked with
`GenericOpcode::validate_types`; target legality remains a separate decision.
Structured constraints/text projections are not yet supported by operand-array
emission and are rejected rather than silently ignored. Shared `Type`,
`Signature`, linkage, condition codes and some entity identifiers still come
from `veloc-mir`.

Control descriptors are shared by generic and selected instructions:
`Next` and `Call` continue, `Branch` has explicit targets plus fallthrough,
`Jump` has only explicit targets, and `Return`/`Trap` have no successors.
Target definitions supply these facts; LIR does not depend on target decoding.
In particular, a two-successor generic branch is a `Jump`, whereas a selected
conditional jump followed by an unconditional jump is `Branch` then `Jump`.

`MachineInst::memory` carries a single fixed-size `MemoryAccess`: direction,
bytes touched, guaranteed effective-address alignment, volatility and trap
behavior. It describes the access, not the result register width. Missing data
means unknown, never memory-free. Ordinary cloning retains the descriptor;
rewrites changing the access must explicitly provide a corresponding descriptor.
This is not yet a model for atomic accesses, multiple accesses or scalable sizes.

```sh
cargo test -p veloc-lir
cargo check -p veloc-lir --no-default-features
```
