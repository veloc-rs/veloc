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

## Machine SSA migration

The target architecture is a value-based machine SSA with separate machine
constraints and execution effects. Physical locations must not replace SSA
value identities during allocation. Register roles, fixed locations, reuse
constraints and access timing belong to instruction contracts, not value types.

The first implemented boundary is allocation versus materialization. Codegen's
`RegisterAllocator::allocate` consumes a `PostIselOptimized` function into an
`Allocation` without changing its input instructions, layout or frame. Spill
instructions are created as detached IDs in the same store. The result owns that
exact input, exposes read-only per-instruction operand locations (`PReg`) and
before/after spill sequences, and separately owns the planned frame. Calling
`materialize` consumes the plan into physical IR without cloning the function
or rerunning target decisions. Plans cannot be applied to a different or edited
input. The current algorithm still allocates whole ranges; the result is indexed
by operand occurrence so future range splitting does not require a new result
interface. It does not yet model direct stack operands or early-clobber slots.

Two-address instructions now have separate Def and Use operands. ISLE expands
a tied encoding declaration into an output, an appended input and a static
location constraint. Selection explicitly supplies both values. Allocation
resolves the constraint with physical input/output copies, including scratch
handling when the destination overlaps another input. Indexed memory operations
likewise have independent output and base operands; there is no ReadWrite role.

Virtual registers remain in SSA through selection and scheduling. Block
parameters and branch arguments survive until allocation; edge moves are
planned over physical locations and materialized in dedicated edge blocks.
Parallel copies handle register/stack moves and cycles, sharing cycle-save
slots of the same layout. An original entry block with backedges gets a separate
ABI entry predecessor, so function inputs and loop parameters have distinct
definitions.

ISLE rules declare fresh local values with `(temp $bit $dst)`, inheriting an
exemplar's type and bank. ABI calls define physical result registers; only the
following copies define the virtual results. Physical register uses/clobbers
are intentionally not SSA.

`CodegenOptions::verify` runs independent checks at pipeline boundaries:
unique virtual definitions, dominance/use order, reference-index consistency,
and edge argument count/types. It defaults on in debug builds and can be
disabled without changing construction APIs. Register allocation still uses
whole ranges and does not yet support general multiple-output reuse or
early-clobber constraints; these are allocator limitations, not SSA exceptions.

## Current representation

Consumers import `veloc_lir::{MachineFunction, InstId, InstRef, ...}` and
`veloc_lir::stages::RawLir`, or use the top-level `veloc::lir` facade. There is no
compatibility module at `veloc_codegen::lir`.

Each function owns an `InstStore`. Instructions have stable IDs; operand ranges
and cold payloads live in recyclable pools. `InstRef` is a borrowed handle, not
an owning instruction or copy-on-write wrapper. Generic and target opcodes use
the same storage and no independent instruction drafts exist.

`function.writer().add(dst, lhs, rhs)` creates a detached instruction and returns
its ID; adding that ID to a block is a separate layout operation.
`function.rewriter(id).add(...)` replaces contents while preserving the ID.
Replacement resets old memory and extra payloads; operand-only edits preserve
them. Sequences used by selection, ABI lowering and allocation contain IDs,
not copies of instructions. The block cursor's `emit`, `keep_current`,
`detach_current` and `remove_current` distinguish layout edits from deletion.
`replace_current(source)` transfers a detached source into the current ID and
invalidates the source ID without copying its operands or payloads.

Register occurrences are maintained by the store, not a cached analysis.
`uses(reg)` and `defs(reg)` return borrowed occurrence iterators, including
branch arguments. Writers, replacements and controlled operand/extra setters
update the links automatically; mutable operand slices are not exposed.
Detached live instructions remain indexed until explicitly removed.
MIR and LIR share the reverse-link primitive in `veloc-collections`, while each
owns its values, allocation policy and instruction semantics.

`replace_uses(old, new)` rewrites virtual-register reads, including edge
arguments without changing definitions. Dominance and type compatibility remain
the caller's responsibility. `check_refs()` independently
audits links and pooled ranges against canonical instruction contents; it is
not an SSA verifier. Physical-register occurrences likewise do not imply a
unique reaching definition.

Generated selection computes a matched rule's operand buffers before committing
any instructions, then writes the entire sequence to the store. Ordinary failed
matches do not allocate instructions. This is not a speculative transaction or
a general rollback facility. Schema decoding
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

Fixed builders are named from the opcode (`G_BRCOND` -> `writer.brcond`). Variadic
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

`InstRef::memory()` carries a single fixed-size `MemoryAccess`: direction,
bytes touched, guaranteed effective-address alignment, volatility and trap
behavior. It describes the access, not the result register width. Missing data
means unknown, never memory-free. Cloning the function retains the descriptor;
rewrites changing the access must explicitly provide a corresponding descriptor.
This is not yet a model for atomic accesses, multiple accesses or scalable sizes.

```sh
cargo test -p veloc-lir
cargo check -p veloc-lir --no-default-features
```
