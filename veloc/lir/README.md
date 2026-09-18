# Low-level IR

`veloc-lir` owns the machine-facing representation: instructions and operand
schemas, registers and register banks, functions and blocks, stack-frame data,
symbols and use-def chains. It supports `no_std` with `alloc` and
has no dependency on codegen or any target backend.

Codegen owns MIR-to-LIR translation, legalization,
instruction selection, register allocation, ABI handling and machine-code
emission. A single mutable `MachineFunction` is shared by these passes; it does
not carry phase type parameters or mutable selected/allocated flags. Codegen
orders passes explicitly and optionally verifies the required invariants at
boundaries. Analysis validity is tracked separately through pass change sets.

Register-bank selection is not a mandatory stage. A target may install its own
pre-selection pass through `TargetPassConfig::pre_isel_passes`. The x86 backend
selects instructions directly from typed values without preassigning banks.
Optional explicit bank constraints remain available for target temporaries;
otherwise allocation uses the target's type-based class defaults together with
the selected instructions' operand constraints. This is not a global bank-cost
optimizer and does not yet synthesize transfers for incompatible operand classes.

## Machine SSA migration

The target architecture is a value-based machine SSA with separate machine
constraints and execution effects. Physical locations must not replace SSA
value identities during allocation. Register roles, fixed locations, reuse
constraints and access timing belong to instruction contracts, not value types.

The first implemented boundary is allocation versus materialization. Codegen's
`RegisterAllocator::allocate` consumes a selected SSA function into an
`Allocation` without changing its input instructions, layout or frame. Spill
instructions are created as detached IDs in the same store. The result owns that
exact input, exposes read-only per-instruction operand locations (`PReg`) and
before/after spill sequences, and separately owns the planned frame. Calling
`materialize` consumes the plan into physical IR without cloning the function
or rerunning target decisions. Plans cannot be applied to a different or edited
input. The current algorithm still allocates whole ranges; the result is indexed
by operand occurrence so future range splitting does not require a new result
interface. It does not yet model direct stack operands or early-clobber slots.

Two-address instructions have separate result and input lists. ISLE expands
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
and edge argument count/types. Selected-code checks additionally reject generic
instructions; allocated-code checks reject virtual registers and block parameters
and validate physical operand constraints. It defaults on in debug builds and can be
disabled without changing construction APIs. Register allocation still uses
whole ranges and does not yet support general multiple-output reuse or
early-clobber constraints; these are allocator limitations, not SSA exceptions.

## Current representation

Consumers import `veloc_lir::{MachineFunction, InstId, InstRef, ...}`,
or use the top-level `veloc::lir` facade. There is no
compatibility module at `veloc_codegen::lir`.

Each function owns an `InstStore`. `results()` borrows a compact `Reg` slice in logical signature order;
`inputs()` borrows a compact register slice; `fields()` borrows non-register
attributes. All three use separate recyclable typed pools.
Generated builders and encoders address these slices directly: there is no
mixed operand enum or per-instruction order map. Use-def locations index inputs.
Register allocation visits only results and inputs, leaving attributes untouched.
`RegEffects` separately describes
implicit physical register reads/writes, including call clobbers; dependency
queries include these effects without treating them as SSA results.

Instructions have stable IDs; operand ranges
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

`defs/module.spec` imports the shared `veloc/defs/prelude.spec` and the local
`generic.spec`. Both MIR and LIR use `veloc_spec::Source::load(...).compile()`,
one logical operation model and one type/semantic checker. Import resolution,
dependency tracking and physical-file diagnostics are shared build infrastructure.

`storage Operands` selects operand-array projection; MIR uses packed storage.
Machine `struct` declarations describe register and property types; storage bindings infer each register field's input/result role. Logical `op`
records declare signatures, effects, control behavior and optional semantics.
The generated Rust contains opcodes, views, builders, decoding checks and direct
type validation. Construction does not run validation.

```text
type Reg = rust("crate::Reg");
enum InstField { variants: [Imm(i64)] }
storage Operands {
    opcode: GenericOpcode, view: InstView,
    reader: InstRead, writer: InstBuild,
    register: Reg, attributes: InstField,
}
struct BinaryReg { dst: Reg, lhs: Reg, rhs: Reg }
op Add<T: Integer>(lhs: T, rhs: T) -> (dst: T) {
    meta: OpInfo { memory: MemoryEffect::NONE },
    storage: BinaryReg { dst, lhs, rhs },
    semantics: bv.add(lhs, rhs)
}
```

Fixed pure register recipes use logical input/result order. The reviewed primitive
set is retained; this is not a proof of target legalization or encoding. Codegen
joins MIR and LIR primitive bindings to generate direct lowering dispatch.
Offline tools can include generated semantic artifacts; runtime LIR has no
semantics dependency.

The generated `InstRead<'a>` and `InstBuild` traits define the host boundary.
`InstRef` implements opcode/slice access and error construction; `InstWriter`
implements writes and converts generic opcodes to `MachineOpcode`.
The builder's associated `Def` type is `Writable<Reg>` in LIR; generated code
does not know that wrapper, the concrete writer, or the instruction ID type.
The trait default methods are statically dispatched; there is no `dyn` adapter.
Import `InstRead` for `view()/validate()` and `InstBuild` for builders.

Instruction access uses `inst.view()` and pattern matching:

```rust,ignore
use veloc_lir::InstRead;
match inst.view() {
    InstView::BinaryReg(binary) => {
        // The restricted opcode distinguishes ADD, SUB, etc. sharing this shape.
        use_binary(binary.opcode, binary.dst, binary.lhs, binary.rhs);
    }
    InstView::Return(ret) => {
        for &reg in ret.values {
            use_return(reg);
        }
    }
    _ => {}
}
```

Views are generated from structs; there are no accessor-name overrides or runtime
schema dispatch. Accessors assume the opcode's structural invariants and read
directly; wrong field variants or missing required fields panic using safe Rust.
They do not call a validator, even in debug builds. Fixed fields are copied;
variable register lists borrow `&[Reg]` without allocation.
The separately generated `validate()` checks counts, attribute kinds
and structural index constraints and returns `ValidationError`. The existing
`CodegenOptions::verify` switch opts into this check at pass boundaries.
Generated `GenericOpcode::validate_types()` remains an independent type contract;
SSA/dominance checks use function-level algorithms. A generic view of a target
or invalid opcode is an internal error, not a fallible decode operation.

All builders are generated from the opcode name (`Brcond` -> `writer.brcond`,
`Callind` -> `writer.callind`), including calls and returns. Only generic
unary/binary helpers for target instructions remain ordinary Rust.
Mappings bind every layout field to a logical input or named result.
Builders take results first, then inputs in signature order. Results are encoded
in signature order; input and attribute slots follow the layout's field order.

The storage declaration identifies the register type and attribute enum.
Enum payload types determine attribute codecs; there is no built-in registry
of immediate, condition-code or symbol types. `optional(Reg)` fields require
`some(input)` or `none`; absent fields need not form a suffix.
`sequence(Reg)` borrows a register slice. Each storage domain permits one
trailing sequence, which may follow fixed fields. Attributes currently support
single and optional fields, not sequence pools.
Calls are ordinary structs: direct calls have a symbol attribute, indirect
calls have a register callee followed by argument registers.
`results: results()` binds the result slice. Two-address constraints do not
merge SSA inputs and outputs.
`Icmp` requires two inputs and an explicit condition; zero comparison is
`Ieqz`.

Logical type contracts now use the same compiler as MIR and can be checked with
`GenericOpcode::validate_types`; target legality remains a separate decision.
Logical parameter accesses, queries, property/operation constraints and ownership
visitors share the MIR compiler. Checks accept the contexts explicitly declared
in defs. A reader's `value_type` method is required only when generated
expressions need a value-to-type lookup; the current LIR definitions do not.
Property contracts are inlined at their instruction uses, not implemented as
inherent methods on potentially foreign Rust types. Queries become reader methods.
View declarations share MIR's field/lifetime emitter, with named records instead
of inline enum fields. Physical reads remain storage-specific; builders and text
parsers use the same prepared-call emitter as MIR. Explicit text projections
generate parser/printer artifacts using the shared atom protocol (the LIR crate
does not yet include a complete textual frontend). Signature sources generate
only the required reader lookups and share MIR's contract emission. Shared `Type`,
`Signature`, linkage, condition codes and some entity identifiers still come
from `veloc-mir`.

The optional `control: ControlFlow::Next` storage setting names an ordinary
enum and its default variant. Opcode `flow` values are checked against that
enum. Effects and traits are declared separately in metadata, not inferred by
the storage generator from control variant names.
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
