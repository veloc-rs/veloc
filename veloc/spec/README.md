# Operation definitions

`veloc-opgen` is the build-time definition compiler. It is independent of runtime
IR containers; `veloc-mir/build.rs` currently uses its MIR emitter. HIR is reserved
for a future structured representation. The machine-facing IR is LIR, in
`veloc-codegen::lir`; bytecode is a separate execution format.

The MIR definitions live in `veloc/mir/defs/`:

- `formats.ops`: compact storage layouts, structured property records and
  alternate-layout projections.
- `mir.ops`: logical operation signatures, storage mappings, effects, constraints
  semantic expressions and bidirectional text projections.

```text
format Binary {
    fields: [opcode(Opcode), args(values(2))],
    opcode: dynamic(opcode)
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
    text: Text { args: [ptr], named: [default(offset, 0)], flags: flags },
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
`dest: successor`, and a successor group uses `cases: successors`. Empty input
and result lists are `()`; signature-selected results use `-> signature`.
For example, an indirect call declares a statically checked `ptr: PTR`, a
variable-length `args: values` group and `signature: sig_id`. Direct calls use
`signature: function(func_id)` to identify the callee's signature. The source of
dynamic result types is explicit, not inferred from the opcode's name.

The storage mapping explicitly connects logical parameters to physical fields.
Every non-opcode field is mapped; arrays and fixed-length pooled lists use
`[lhs, rhs]`, and the dynamic opcode field is supplied by the compiler. Logical
properties use their actual data, not physical pool IDs: `@bytes: Bytes` maps to
`pool(bytes)`; `@imm: PtrIndexImm` and `@mem: VectorMemOptions` map to their
corresponding pools. A branch table maps its named `cases` and `default`
successors with `table(cases, default)`, making the default-last representation
explicit. Construction and projection use the same checked mapping. The
current MIR adapter still requires SSA traversal order to agree with its
existing physical layouts. Naming the mapping does not yet make arbitrary
physical reordering safe for consumers that directly destructure those layouts.

The three structured property records are emitted from their field definitions.
Their names and field types are checked against the existing typed DFG pool
adapters: adding a field also requires updating those adapters, including SSA
visitation and replacement for value fields. Unsupported changes fail during
definition checking instead of silently losing operand uses.

The definition compiler checks references, field coverage, type variables,
arities, constraints, semantic compatibility and generated method names before
emitting Rust. Definitions may refer to later layouts. Diagnostics include source
line and column; the build script maps combined input locations back to the source
file.

## Generated consumers

The same definitions generate `Opcode`, `OpFormat`, `InstructionData`, type
contracts, opcode extraction, operand traversal/replacement, memory flag access,
operation-specific parsing/printing and ordinary builders. Named variables are assigned slots
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
mapping, independently of the text projection.

Context is needed when construction allocates pooled values or extensions,
maintains CFG edges, or resolves call signatures and external result types.
Those operations retain contextual helper algorithms; the generator recognizes
the need from field and result types, not a per-opcode name list. In particular,
a branch is never automatically implemented as a bare instruction push that
forgets predecessor bookkeeping. Context helpers can provide higher-level inputs
such as slices and blocks while the storage schema retains compact IDs.

## Bidirectional text projections

Without a `text` field, an operation prints its logical parameters in signature
order, separated by commas; a zero-parameter operation has no operand text.
An explicit projection changes notation without changing the builder API or
storage layout:

```text
op Store(ptr: PTR, value: Any, @offset: u32, @flags: MemFlags) -> () {
    mnemonic: "store",
    storage: Store { ptr: ptr, value: value, offset: offset, flags: flags },
    text: Text {
        args: [value, ptr],
        named: [default(offset, 0)],
        flags: flags
    },
    traits: [MAY_TRAP], memory: HEAP_WRITE
}
```

The same projection generates both directions. `args` are positional atoms;
`named` fields use `name=value`. An ordinary atom derives its reader/writer from
the logical type. `integer(value)` preserves integer bit patterns with signed
text; `float(value)` preserves raw hexadecimal floating-point bits, including
NaN payloads; `bytes(data)` is hexadecimal byte text. `space(kind, lhs)` composes
the comparison spelling `eq v0`. Invocation syntax is composed with
`invoke(func_id, args)` or `invoke(ptr, args, sig_id)`. Variadic values and
successors retain their generic comma-list and bracketed-list syntax.

`default(offset, 0)` accepts an omitted value and omits it when printing zero.
`optional(mem.mask)` is an optional SSA value in a structured property record.
`flags: flags` or `flags: mem.flags` reads and writes mnemonic suffix flags.
Nested paths refer to checked record fields; they are not unchecked Rust
expressions. Each operation's projection defines its accepted named fields, so
strided memory accepts `stride`, while gather/scatter accept `index` and `scale`.
Unsupported fields are rejected rather than parsed and silently discarded.

Text projections must account for logical data, apart from declared record
defaults. Unknown or duplicate references, incompatible atom types and invalid
optional/default annotations are definition errors. Ordinary formats do not
also select a named text codec: that would duplicate the operation's grammar.
An alternate storage layout instead declares how its shared extension data is
projected alongside the operation, preserving mask/EVL predication.

The runtime retains generic lexical, symbol-resolution, pool and CFG algorithms:
function/module syntax, type spelling, token splitting, forward SSA references,
signature lookup and final validation are not opcode definitions. The generated
code composes these primitives; there is no per-opcode parser/printer switch to
keep synchronized manually. This is a finite text schema, not an arbitrary
parser-generator language.

### Generated/runtime contracts

Text atoms implement the internal `AtomCodec` trait, pairing `parse` with `print`.
The emitter selects one codec type for both directions instead of maintaining
separate reader/writer function-name mappings. Codec identity describes notation:
`IntegerBits` and `FloatBits` both store `u64`, while `Decimal<u64>` is unsigned
decimal text. Associated `Owned` and `View` types let parsing produce a vector
and printing borrow a slice. Contextual codecs reuse the scanner and symbol
resolution algorithms; this does not require a trait for every syntax helper.

Interned property handles implement `dfg::PoolKey`, with associated insertion
input and borrowed view types. Generated `pool(...)` mappings call only `insert`
and `get`; concrete DFG pool names and byte-storage wrappers stay in the runtime
implementation. Existing convenience methods delegate to the same pools. Value
lists and jump-table construction retain their shared, non-interning algorithms.
Both contracts use static dispatch and do not add a codec registry or trait-object
dispatch. Rust checks the implementations and generated calls; round-trip tests
remain necessary to check that the two directions agree semantically.

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
cargo test -p veloc-opgen -p veloc-mir -p veloc-semantics
cargo run -q -p veloc-semantics --example split_add | z3 -in
```

Generated Rust lives in Cargo's `OUT_DIR`, not in the source tree. Building the
compiler does not require a solver or a model service.
