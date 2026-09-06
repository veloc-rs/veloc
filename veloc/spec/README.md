# Operation definitions

`veloc-opgen` is the build-time definition compiler. It is independent of runtime
IR containers; `veloc-ir/build.rs` currently uses its MIR emitter. HIR is reserved
for a future structured representation. The machine-facing IR is LIR, in
`veloc-codegen::lir`; bytecode is a separate execution format.

The MIR definitions live in `veloc/ir/defs/`:

- `formats.ops`: compact storage layouts and textual codecs.
- `mir.ops`: logical operation signatures, storage mappings, effects, constraints
  and semantic expressions.

```text
format Binary {
    fields: [opcode(Opcode), args(values(2))],
    opcode: dynamic(opcode),
    text: values(2)
}

op IAdd<T: Integer>(lhs: T, rhs: T) -> (result: T) {
    mnemonic: "iadd",
    storage: Binary { args: [lhs, rhs] },
    semantics: bv.add(lhs, rhs)
}

op ExtendU<T: IntegerOrBool>(arg: T) -> (result: shape(T, Integer)) {
    mnemonic: "extendu",
    storage: Unary { arg: arg },
    where: [wider(arg, result)],
    memory: NONE
}

op Load(ptr: PTR, @offset: u32, @flags: MemFlags) -> (result: Any) {
    mnemonic: "load",
    storage: Load { ptr: ptr, offset: offset, flags: flags },
    traits: [MAY_TRAP], memory: HEAP_READ
}
```

SSA operands and results have names and types in the operation signature.
Generic variables such as `T` are scoped to that operation; their first direct
occurrence binds the type. Derived types use `element(T)`, `vector(T)` or
`shape(T, Integer)`. Relations refer to operand/result names, not numeric slots.
There are no separate `types` records or references to named type schemes.

An `@` parameter is a property, not an SSA use: `@offset: u32` and
`@flags: MemFlags` are stored values, while `ptr: PTR` is an SSA value with a
pointer type. Variable-length SSA groups use `args: values`; one successor uses
`dest: successor`, and a successor group uses `table: successors`. Empty input
and result lists are `()`; signature-selected results use `-> signature`.
For example, an indirect call can declare both a statically checked `ptr: PTR`
and a variable-length `args: values` group.

The storage mapping explicitly connects logical parameters to physical fields.
Every non-opcode field is mapped; arrays and fixed-length pooled lists use
`[lhs, rhs]`, and the dynamic opcode field is supplied by the compiler. The
current MIR adapter still requires SSA traversal order to agree with its
existing physical layouts. Naming the mapping does not yet make arbitrary
physical reordering safe for consumers that directly destructure those layouts.

The definition compiler checks references, field coverage, type variables,
arities, constraints, semantic compatibility and generated method names before
emitting Rust. Definitions may refer to later layouts. Diagnostics include source
line and column; the build script maps combined input locations back to the source
file.

## Generated consumers

The same definitions generate `Opcode`, `OpFormat`, `InstructionData`, type
contracts, opcode extraction, operand traversal/replacement, memory flag access,
text codec selection and ordinary builders. Named variables are assigned slots
by the definition compiler; runtime inference and validation share the same
binding machinery and are not limited to four variables.

Ordinary builders are generated automatically, without a `builder` field or an
auto/custom/off switch. Their method name is the mnemonic with `-` replaced by
`_`: `iadd-sat` becomes `iadd_sat`. Definitions whose mnemonics normalize to the
same method name are rejected. The removed `builder` field is an error; there is
no legacy configuration path or alias for an old method name.

Parameters take their names and order from the logical operation signature;
physical array field names no longer invent argument names. The storage mapping
constructs the compact instruction. A caller-selected result `ty` is always
last. For example, the standard definitions produce
`iconst(value: u64, ty: Type)` and
`load(ptr: Value, offset: u32, flags: MemFlags, ty: Type)`. Zero-result operations
return nothing, inferred single results return `Value`, and supported inferred
two-result operations return a pair. Construction comes directly from the field
mapping, independently of the text codec.

Context is needed when construction allocates pooled values or extensions,
maintains CFG edges, or resolves call signatures and external result types.
Those operations retain contextual helper algorithms; the generator recognizes
the need from field and result types, not a per-opcode name list. In particular,
a branch is never automatically implemented as a bare instruction push that
forgets predecessor bookkeeping. Context helpers can provide higher-level inputs
such as slices and blocks while the storage schema retains compact IDs.

Text codecs are shared grammar implementations. Their format selection, regular
value construction and traversal are generated. Specialized syntax for calls,
memory, constants and vectors still has explicit parser/printer code; this
compiler does not claim to synthesize arbitrary grammars or context algorithms.

## Semantics and lowering

`semantics: bv.add(lhs, rhs)` describes the result using the named logical
operands and the modular primitives in `veloc-semantics`. Expressions can compose
primitives: integer negation is `bv.sub(bv.zero(), arg)`, rather than a second
handwritten implementation of negation. Unary/binary constant folding executes
these programs. MIR-to-LIR arithmetic translation directly maps recognized
primitive applications; composed negation retains the existing `G_NEG` lowering.
Programs can also be instantiated at a concrete width for SMT export. Each describes a scalar
operation or a per-lane operation, not an entire vector, memory or machine-state
model. Absent expressions mean **unmodeled**,
not a claim that an operation is pure or verified.

Pure semantic expressions imply `memory: NONE`; unmodeled operations must
declare their memory effect explicitly. Empty `traits` lists may be omitted.

The current semantic backend covers modular arithmetic/bitwise expressions.
Direct primitive applications inherit shared trusted algebraic facts, so an add
definition does not repeat its commutativity, associativity or identity. Explicit
claims are checked against the semantic contract; an operation cannot bind to
subtraction while claiming commutativity. Composed expressions are not assumed
to inherit the outer primitive's algebraic laws without justification.
Floating point, traps, memory, ABI state and general representation conversions
are not modeled yet. LIR target descriptions and contextual lowering still use
their own algorithms; migrating those descriptions into the shared definition
model is a subsequent step. The `split_add` semantics example demonstrates a
fixed-width representation check; it is not an enabled wide-integer backend pass.

```sh
cargo test -p veloc-opgen -p veloc-ir -p veloc-semantics
cargo run -q -p veloc-semantics --example split_add | z3 -in
```

Generated Rust lives in Cargo's `OUT_DIR`, not in the source tree. Building the
compiler does not require a solver or a model service.
