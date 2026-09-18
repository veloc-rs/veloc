# Operation definitions

`veloc-spec` is the build-time definition compiler. It is independent of runtime
IR containers and `veloc-types` (including test dependencies); `veloc-mir/build.rs` uses its MIR emitter. HIR is reserved
for a future structured representation. The machine-facing IR is LIR, in
`veloc-lir`; bytecode is a separate execution format.

Rust methods are declared on their types. A `trait: rust("owner::traits::TypeInfo")`
binding selects the owning trait; `Source::interfaces(namespace)` emits its
signature, never a forwarding implementation. The owning crate uses Spec in
its build script and writes the Rust implementation itself. Free functions may
still bind a Rust path directly; methods use their declared trait.

Spec never loads a runtime type catalog or calls a foreign Rust function.
Logical scalar/vector domains are declared in ops. Associated constants and
`const fn` calls remain typed expressions in generated Rust, where rustc checks
their signatures, evaluates static metadata and selects admissible semantic
specializations. There is no `bindings.rs`, query registry or foreign-value schema.
The generic semantic expression library remains a normal dependency.

Integration tests live in `crates/veloc-filetests/tests/opgen/`, grouped into
types, operation contracts, imports, and generated-code execution. Standalone
Rust host and formatting checks share this suite; private invariant tests stay
beside the implementation.
File tests distinguish definition errors from Rust type errors and const-evaluation
failures; moving a check to rustc does not remove its negative test.

Shared vocabulary lives in `veloc/defs/`: `type_sets.spec` contains only reusable
type domains and imports the Rust owner's `veloc/types/defs/types.spec`.
`types.spec` adds IR-specific value, successor, property and metadata declarations;
`prelude.spec` imports it. Legalization imports only `type_sets.spec`, retaining
policy-specific domains such as `Narrow` and `Word` in codegen. MIR owns its packed
`formats.spec` and logical `mir.spec`; LIR owns `generic.spec` with operand-array
formats and logical operations. Each consumer has a `defs/module.spec` entry:

```text
import "../../defs/prelude.spec";
import "formats.spec";
import "mir.spec";
```

`Source::load(path)` resolves relative imports against the importing file,
deduplicates canonical files, rejects cycles and requires imports before
declarations. Its `dependencies()` includes requested and canonical paths for
Cargo rebuild tracking; `parse()` and `compile()` report physical file locations.
Files are parsed independently in one pass, including their import preambles.
The loader resolves imports from that AST and retains the original source for
diagnostics; it does not mask imports or parse the file again. Syntax cannot
cross import boundaries.

The lexer produces tokens on demand with one-token lookahead. Declaration syntax
keeps type bindings, sets, functions, and constants distinct; type members remain
nested under their owner. Name resolution and model checking consume these
declarations directly; there is no flattened record representation between the
parser and the checked model. Associated constants retain their declared type
rather than being represented as zero-argument functions.

Each file sees its own declarations and its transitive imports, not unrelated
files loaded by an entry module. For example, `mir.spec` imports `formats.spec`,
which imports the shared prelude. Shared Rust bindings live together in `types.spec`.
Rust data types such as `Type` and `Float` need an explicit declaration or import.
IR type sets and Rust data types occupy distinct namespaces: importing the
`Float` set does not import the Rust `Float` property type. Primitive syntax types
such as `u32` and `bool` remain built in. Imports are file-wide (no selective
imports or aliases yet), and duplicate declarations in the combined unit are
still rejected.
`Source::load` is the definition entry point; its `parse`, `plan` and `compile`
methods preserve import visibility and original-file diagnostics.

There is one checked `Definitions` model. `storage Operands` selects machine
operand-array emission; packed storage emits MIR views and pools. Storage
projections differ, but signatures, type expressions and semantic checking are
shared. Unsupported projection capabilities fail while preparing an output plan,
before any Rust artifact is emitted.

The public stages remain available for inspecting and reusing an IR plan:

```rust,ignore
let source = veloc_spec::Source::load("defs/module.spec")?;
let plan = source.plan()?;      // resolve and check IR output projections
let generated = plan.generate(); // infallible emission; reusable
```

`Source::compile()` composes planning and generation of all supported IR
artifacts. Production consumers use `Source::generate` with an explicit list. `parse` checks the definition model without
emitting storage code; it is not a promise that every output supports every
contract. Property validators are checked there, independently of output choice.

An output plan owns the checked definitions and immutable structured projections,
not pre-rendered Rust strings. It resolves operation-to-format indices, builder
result inference and naming, alternate-layout constraints, text schemas and
dispatch, and supported scalar evaluator instances/property bindings. Each
alternate is prepared once and shared by text and validation emission; its
parser receives the actual opcode separately from the shared layout.

Generation consumes those plans without source text or fallible definition
checking. Filesystem writes, rustfmt failures and integration with the consuming
Rust crate remain separate artifact-boundary concerns. No general Rust AST,
plugin framework or additional runtime descriptor is introduced.

## Definition compiler organization

- `syntax/lexer.rs` produces one lookahead token at a time, borrowing names and
  numbers and decoding strings once. Compound operators are single tokens.
- `syntax/parser.rs` uses recursive descent for declarations, signatures,
  lists, objects and type sets, and precedence climbing for pure expressions.
  A named parsing context distinguishes declaration values, types and pure expressions;
  nested syntax is bounded, while flat type sets remain n-ary.
- `syntax/mod.rs` holds the untyped syntax tree and source offsets.
- `source.rs` resolves file dependencies and maps diagnostics back to files.
- `model/` owns operations, structs, metadata, effects and ownership contracts.
  `model/expr.rs` is the shared typed expression checker, helper expander,
  constant evaluator and Rust emitter. Verification and interface modules
  only adapt it to their respective consumers.
- `types/` groups type declarations, exact sets, encoding, resolution and
  generated type rules. Small set operations live with the type model.
- `generate/` prepares output plans and assembles Rust artifacts, constructors and constant evaluation;
  formatting and direct lowering helpers live in its entry module.
- `storage/`, `semantic/` and `text/` own their specialized representations
  and projections. These remain separate because their grammars and consumers
  differ, not because each helper needs a module.

Parsing establishes syntax, not validity of operation contracts. Type resolution,
storage compatibility and semantic checks still precede generation. The text
template parser stays separate because quoted templates have a different grammar.

## Runtime organization

MIR groups related runtime code under one module boundary:

- `inst/mod.rs` exports instruction handles, writers, views and opcode metadata.
  `inst/opcode.rs` owns generated opcode/type contracts and their support;
  `inst/storage.rs` owns physical instruction storage.
- `function/mod.rs` owns functions, with `edit.rs` for structural editing and
  `layout.rs` for block order, instruction placement and CFG edges.
- `veloc-types` owns types, signatures and calling conventions. MIR\'s `types`\n  module re-exports them alongside MIR entity handles and generated constants.
- `dfg` remains independent; `builder` and `validator` serve both functions and
  modules and remain top-level modules.

Use `veloc_mir::inst::{OpSpec, OpFormat, TypeError, MemoryEffect}` for instruction
metadata and `veloc_mir::function::{Layout, BlockData}` for layout types. Common
types such as `Opcode`, `Function`, `Signature` and `CallConv` remain re-exported
from the crate root.

Generated Rust artifacts follow their consumers, not the input file boundaries:

- `types.rs`: compact type encoding, constants and type helpers.
- `opcodes.rs`: opcode metadata, formats, type sets, packed encodings.
- `instructions.rs`: storage, writers, views, accessors and result type inference.
- `builders.rs`: operation-specific `InstBuilder` methods.
- `type_rules.rs`: type validation dispatch and shared signature checks.
- `validation.rs`: operation constraints checked in function context.
- `text_parser.rs` and `text_printer.rs`: their respective text codecs.

Construction and validation remain separate. Optimizer evaluation, offline
semantics and backend lowering retain separate artifacts and consumers.

Value validity requirements belong to the operation's explicit `verify` block.
For example, `Vconst` checks its dense byte storage there using `VerifyContext`;
parameter types do not implicitly attach additional validation contracts.

### Rust type bindings

Use the same `type` declaration for IR value types and Rust-owned data types;
the right-hand side selects the meaning:

```text
type I32 = int(32);
type I32X4 = vector(I32, 4);
type SigId = rust("crate::SigId");
type Token = rust("crate::tokens::Token");
```

A Rust binding declares a nominal, opaque data type. It emits no struct, enum
or alias. Property fields, enum payloads, host signatures and query results
resolve the declared Rust path; generated builder and text-codec signatures
use the same mapping. Different defs names remain distinct even when their
Rust paths happen to be identical.

Bindings do not introduce SSA value types, arithmetic, accessible fields,
literal constructors or text codecs. IR result types and type sets still use
structural definitions such as `int` and `vector`. External values flow through
typed fields and declared host methods; their Rust implementations supply any
required traits. In particular, a new binding alone does not enable a new
text projection.

Paths must be qualified by `crate` or an external crate name, with identifier
segments only: no references, generic arguments, relative `self/super` paths
or embedded Rust code. Ordinary imports load these declarations; primitives
remain built in, and defs-owned structs/enums register themselves.
MIR operand-list and successor storage roles remain structural, not opaque
Rust type bindings.

### Type method interfaces

A Rust-bound type can declare its methods alongside its representation:

```text
type Type = rust("crate::Type") {
    const fn element_bits(self) -> optional(u32);
    const fn wider_than(self, other: Type) -> bool {
        value: self.element_bits()? > other.element_bits()?
    }
}
```

A declaration without a body binds the same-named method in the generated trait.
A defs body is checked and inlined into the shared expression tree. Free functions
can use `= rust("crate::path::function");`; methods use the type's trait binding.
There is no separate `impl` or `intrinsic` declaration mechanism.
`Value`, `Int`, `Float` and `VectorConst` declare `ty()` in their type blocks.
Their defs bodies use the fundamental `type(self)` query: SSA values need a DFG
lookup, whereas typed constants carry their own type. User-facing verification
and projection expressions call `value.ty()`; the result-pattern
`-> type(property)` and semantic sort reference `type(operand)` are separate
grammatical uses and remain unchanged.

Every method must be declared, for every Rust-bound type, not just `Type`.
Calls are resolved by the nominal receiver type, not a global method-name whitelist.
Rust-bound method declarations generate per-type traits in `type_methods`, unless
an explicit trait path selects another owner. The Rust owner implements these
traits explicitly; opgen does not generate forwarding implementations. Rust
checks every declared signature, including unused methods. Defs-body methods
expand normally and need no external implementation; data-only types need no
empty trait.

`const fn` methods generate a `pub const trait`; the owner supplies a `const impl`
and enables nightly `const_trait_impl`. If a type also has runtime-only methods,
they remain in the ordinary trait and the constant methods use a separate trait
with a `Const` suffix. This avoids requiring runtime methods to be const. Free
functions also accept `const fn ... = rust("...");`.

Metadata supports pure expression syntax, for example
`count: Type::I64.element_bits()? + 1`. Defs-owned literals and expressions can be
folded by opgen, but foreign constant calls remain typed expressions until rustc
evaluates the generated static initializer. No query registration is needed.
Runtime-only calls are rejected in metadata and in `const fn` bodies. `?` and
checked arithmetic become const-compatible matches with a diagnostic panic on
failure, so absent values and overflow fail compilation instead of falling back
to runtime computation. Generated const trait calls require `const_trait_impl`
in the consuming crate; const equality on Rust types may additionally require
const comparison support from that type and the toolchain.
Free functions and methods share argument checking and normalize their expanded
body to the declared return type. Field access and calls use structural postfix
syntax: `object.field`, `object.method(args)`, and `Interface::method(args)`.
Associated constants use `Type::BOOL` or `MemoryEffect::NONE`; `.` never resolves
an associated member.
Parentheses and whitespace do not change name resolution.
Bound functions are trusted pure, deterministic, read-only operations. Contexts
use the same Rust-bound type declarations and generated traits as other data.
Declaring methods does not execute Rust code inside the definition compiler.

The no_std `veloc-types` crate owns the shared `Type`, checked scalar/vector
views, physical encoding and size queries. MIR re-exports these types. Its build
script uses opgen to generate interface declarations without needing a type
catalog. The logical definitions are shared with IR generation; opgen itself does
not link the runtime type crate.

Signature stores are context-local and append-only. Each signature stores its
parameters and returns in one buffer; the interning index stores only IDs.
Borrowed interning hits do not allocate a signature. Callable types retain a
direct signature ID and ownership tag. The interpreter imports each module's
signature graph once into a program-wide canonical store; runtime comparisons
use remapped IDs instead of walking nested signatures. Raw IDs from different
stores must never be compared without remapping or structural comparison.

Semantic specialization enumerates logical element kinds and shape domains,
then emits Rust const predicates over those candidates. Unsupported semantic
signatures are rejected only if the Rust predicate admits them. Const assertions
also reject empty admissible domains. The resulting boolean tables guard generated
evaluator arms; constant folding does not instantiate a semantic expression graph.
Type-only semantic predicates must be const-capable. Runtime-only predicates
remain valid for ordinary validation, but cannot filter an offline semantic domain.

Definition-owned arithmetic and boolean expressions can still be folded directly.
Foreign Rust calls are never interpreted by the generator. Static metadata, type
constraints and semantic candidate checks reuse the typed expression emitter.
Semantic instances are prepared during validation and reused by code generation;
independent type variables are enumerated separately and predicates filter their
combinations. Per-lane recipes reject any admitted shape-changing combination;
only pointer instances are expanded for both target widths. Runtime type-set
tables include only sets referenced by operation signatures.

### Rust types and borrowed contexts

Contexts use the ordinary Rust-bound `type` declaration. There is no separate
`extern interface` syntax, host method table, or host trait output:

```text
type Signature = rust("veloc_types::Signature") {
    trait: rust("crate::type_methods::SignatureInfo"),
    fn params(&self) -> sequence(Type);
    fn returns(&self) -> sequence(Type);
    fn types(&self) -> sequence(Type);
}
type VerifyContext = rust("crate::host::VerifyContext") {
    trait: rust("crate::type_methods::VerifyContextInfo"),
    fn function_signature(&self, func: FuncId) -> optional(&Signature);
    fn signature(&self, sig: SigId) -> optional(&Signature);
}
fn parameter_count(ctx: &VerifyContext, func: FuncId) -> i128 {
    value: len(ctx.function_signature(func)?.params())
}
```

`self` passes a value; `&self` borrows the receiver. `&T` is a reference to
the concrete Rust type, not `&dyn Trait` or `&impl Trait`. Generated traits
check method signatures, and generated calls use static dispatch. Method syntax
automatically borrows a value for a declared `&self` receiver; helpers take
references explicitly. Borrowed results from a method follow Rust's receiver
lifetime elision, so the result cannot outlive that receiver.

`verify(ctx: VerifyContext)` and `query name(ctx: SomeContext) -> Result { ... }` bind a
read-only reference supplied by the caller. Generated validators take concrete
context parameters; they never construct a context or invoke a conversion.
The MIR validation entry creates one `VerifyContext` for the current function and
module and passes it through its instruction checks. It provides access to both
constant storage and signatures.

Each named query generates an `Inst` method whose parameters include its
concrete context reference when needed. For example:
`inst.stamp_info(dfg, &tokens)`. Context-free queries omit that
parameter. No generic query-dispatch trait is needed. One query's opcode
implementations must agree on the context type; context-free arms may share that
entry. No context provider trait, conversion registry, or generic adapter is needed.

MIR-specific context declarations live in `veloc/mir/defs/types.spec`, not the
shared prelude. All types and helpers still require ordinary file-local imports.

`optional(T)` lowers to `Option<T>`; `sequence(T)` lowers to a borrowed slice.
Query-result records must remain owned; consume borrowed views and sequences
inside the query or a helper. `?` propagates absence using the current validation
diagnostic, or `None` in an instruction query. Helpers expand as expressions;
their Rust implementation calls are emitted, never executed by the generator.

Read-only determinism is a trusted implementation contract: `&self` alone cannot
prove the absence of interior mutation. Only declared const methods may appear
in static contracts. Rust evaluates those calls when compiling generated code.

### Validation and ownership contracts

`verify` blocks compile to direct Rust checks at the explicit validation phase.
They do not run in builders and are not interpreted at runtime. The same
expression language applies to operations and alternate storage layouts:

```text
verify(ctx: VerifyContext) {
    let sig = ctx.function_signature(function)?;
    require(matches(args, sig.params()), "argument types differ");
    require(sig.returns() == ctx.current_signature()?.returns(), "answer types differ");
    require(all(options.evl, |v| v.ty() == I32), "EVL must be i32");
}
```

`ctx.function_signature` resolves a `FuncId` directly to a borrowed signature;
`ty.signature()` extracts a callable's `SigId` without consulting context, and
`ctx.signature(id)` resolves it. Signature views expose their own parameter,
return and combined type slices. `let` evaluates once in source order and can
be reused by subsequent checks; names cannot shadow existing bindings. A failing
`?` in a binding reports `cannot evaluate binding '<name>'` (with backticks in
the actual diagnostic). A failing `?` inside `require` uses that check's message.
`results()` exposes the instruction's result types. `matches` compares SSA value types with a type sequence without allocating.
`prefix(sequence, count)` and `suffix(sequence, count)` use checked slicing.
`all` supports both sequences and optional values (absence satisfies the predicate).
Invalid handles or slices produce the constraint diagnostic, not a panic.
Layout constraints describe auxiliary operands such as masks independently of
the underlying opcode. A signature-selected call's argument/result checks and a
`table(cases, default)` mapping's required default are derived automatically.

`move operand: Type` or `move args: sequence(Value)` marks a non-edge parameter as
transferring ownership. Unmarked inputs cannot consume owned values. Successor arguments
transfer on their own mutually exclusive edges; non-edge inputs execute once
before the branch. `ABORT` marks an abnormal exit that need not transfer remaining
owned values and requires `TERMINATOR`. The generated operand visitor feeds one
shared CFG dataflow analysis: it contains no opcode whitelist. Control lowering
interfaces describe execution, not an implicit ownership contract.
Their required `MAY_TRAP`/`TERMINATOR` traits are derived from the control
primitive; operations need not repeat them. Memory effects remain explicit.

## Comparison predicates

Condition codes are ordinary Rust-bound types declared in `veloc/types/defs/types.spec`.
Rust owns `IntCC` and `FloatCC` in veloc-types; the definition language has no
special comparison declaration. Associated constants and methods use the same
generated trait contracts as other Rust-bound types.

Predicates encode accepted less/equal/greater/unordered outcomes. Rust const
methods derive swap and complement from these sets. Float `complement()`
returns None when the exact IEEE complement is absent; `complement_ordered()`
requires callers to establish that neither operand is NaN.

Generated constant evaluators call `IntCC::test` at execution time. Offline
verification converts the same condition code's signedness and outcomes into
an IntPredicate. The generator does not execute Rust condition-code methods.

## Packed encodings

`encoding` declares sequential bit fields for defs-owned packed data. For example:

```text
encoding Options {
    storage: u16,
    fields: [count(4), enabled(1)]
}
```

Ordinary encodings generate a private integer representation, zero initialization
(`empty`/`Default`), and const field accessors. One-bit fields generate
`is_enabled()` and `with_enabled(bool)`; wider fields generate
`count()`, `with_count(value)` and `COUNT_MAX`.
Setters preserve neighboring fields and assert that values fit instead of
silently truncating them. Storage supports `u8` through `u128`. Widths, duplicate
fields and generated method conflicts are checked before code generation.
Unlike flag sets, packed structs do not expose `union` or `contains`.

Shared contracts `MemFlags`, `OpTraits`, `MemoryEffects` and `MemoryEffect` are
defined in `veloc-types` and explicitly bound in `types.spec`. Their representation,
display and behavioral rules live together in Rust. Ops declare the constants and
methods they use, and generated traits enforce this interface. Opgen never reads
`bitflags::Flags::FLAGS` or interprets a Rust enum. Each operation declares its
traits and memory behavior through ordinary typed expressions.

`MemFlags::with_alignment` validates power-of-two alignment, converts it to log2
and conservatively clamps it to the compact field limit. Neither this layout nor
the shared `Type` encoding is configured in defs.

## Shared type representation

`Type` is an eight-byte tagged value. Scalars and vectors use a compact u16
payload; callables use a disjoint ownership tag and a context-local signature
ID. Scalar codes, masks, shifts, construction and decoding live together in Rust.
Invalid or reserved raw encodings are rejected.

Defs declare logical scalar domains, aliases and exact type sets. Scalar and
common vector constants are associated constants (`Type::I32`, `Type::I32X4`);
defs refer to them as `Type::I32` and `Type::I32X4` after importing `Type`.
Custom aliases generate module constants. Standard classification and
representation remain in Rust. Ordinary `encoding` declarations describe defs-owned packed
data, not `Type` or the shared memory contracts.

Callable operations use ordinary signatures, `move` parameters, `verify`
predicates and explicit metadata. Their definitions declare `MAY_TRAP`, and tail
calls also declare `TERMINATOR`; there is no separate `control` contract or
generated callable classification. Backends lower instruction views directly.
The interpreter's callable lowering returns executable `ControlSite` data and
records live roots at those operations and ordinary calls. Root recording is
a runtime requirement, not a consequence of `MAY_TRAP`. Native lowering retains
callable type guards and explicitly rejects unsupported tail-call instructions.
Global signature/ownership dataflow remains a shared validator algorithm. The
`Callable` type pattern and `apply(callee, args)` text projection require no
opcode-specific parser switch. `signature: callable(callee)` selects the signature
from the SSA value's type, derives argument/result validation and infers the
results of an ordinary `call-value`. Tail calls instead require their answer to
match the enclosing function. See [the MIR callable contract](../mir/docs/callables.md)
for ownership, dropping and interpreter behavior.

Signature sources exist only in the generator. Result inference directly reads
the declared function, signature ID or callable type; it does not construct call
metadata or validate arguments. `Opcode::has_signature()` classifies operations
with such a declaration without unpacking their operands. Invalid source handles
still report inference errors; complete type validation remains a separate stage.

## Types and type sets

```text
import "../../defs/types.spec";

type SV4 = vector(Type::I32, scalable(4));
type WORD = Type::I32;
type WORDS = vector(WORD, 4);
typeset WideInteger = Type::I32 | Type::I64;
typeset WideVectors = vectors(WideInteger);
typeset ChosenShapes = Type::I32X4 | SV4;
```

`type NAME = expression;` declares one concrete IR type alias. The supported
constructor is `vector(element, lanes)`; `scalable(lanes)` denotes a scalable
shape. Alias names must be uppercase Rust constant names. Aliases support
forward references; cycles, unknown members, wrong arity and vector-of-vector
types are rejected. `vector(...)` constructs one type; `vectors(set)` constructs
a type set.

Logical scalar domains use `int(bits)`, `float(bits)`, `bool` and `ptr`.
Their declarations live beside the Rust interface in `veloc/types/defs/types.spec`.
Rust owns codes and physical representations. Generated constant assertions
require those representations to agree with the declared scalar domains.
Vector aliases emit calls to Rust's checked constructor. None of these methods
run inside opgen. New primitive kinds still require semantic and target support.

Construct vectors through a checked scalar view:
`Type::I32.as_scalar()?.vector(4, false)?.as_type()`. Obtain a vector's scalar
view with `vector.element_type()`. Existing vectors and INVALID cannot become
scalar views; vector construction rejects pointers and invalid lane counts.
The scalar name lookup is
`Type::from_name`; the MIR text parser additionally handles vector-shape syntax.

`Type::as_scalar` and `Type::as_vector` return checked `ScalarType` / `VectorType`
views, rejecting INVALID and mismatched shapes. These transparent wrappers share
the same encoding; their fields are private. Vector-only `shape()` is available
only on `VectorType`, whose `element_type()` returns a `ScalarType`. Both views
convert back losslessly via `as_type()` or `Into<Type>`. No trait object or second
type hierarchy is involved. Generic `Type::lane_count()` treats valid scalars as
one lane and rejects INVALID. Vector construction is available only on
`ScalarType`, not on the generic `Type`.

`to_raw` / `from_raw` encode the full type in a checked 16-bit representation.
`ScalarType::code` / `Type::from_scalar_code` are the separate, layout-independent
8-bit scalar encoding boundary: a vector or INVALID cannot become a scalar view. This lets
the interpreter retain its packed 16-bit conversion type pairs without retaining
a second semantic type. Stack type slots hold the complete 16-bit encoding.
The current interpreter rejects non-scalar values before bytecode emission.

`typeset Name = expression;` declares an exact type set, using scalar/vector type constants, other
sets, or `vectors(S)`, which includes every legal fixed and scalable vector
shape over the non-pointer scalar set S. Named vector constants are conveniences,
not an exhaustive enumeration of legal vectors. Numeric lane counts are fixed;
`scalable(lanes)` explicitly selects a scalable shape.
Forward references work; cycles, unknown names and empty sets are errors.
Passing pointers or vector types to `vectors()` is an error, not silent filtering.

Sets preserve both scalar identity and vector shape. `{I32, I64}` does not include
I8/I16 or any vectors; `{I32X4}` does not include I32X8 or scalable I32 vectors.
The build-time model maps logical scalar kinds to shape bitsets; generated runtime checks
use integer masks and matches, not heap-allocated sets. These exact sets also drive
definition-time shape constraints, bitvector semantic compatibility and floating
text checks. There is no separate seven-domain vocabulary or name allowlist.

Type-set expressions also work directly in operation signatures; a named set
is just a reusable alias, not a required declaration for every combination:

```text
op IAnd<T: Integer | BOOL | vectors(BOOL)>(lhs: Value<T>, rhs: Value<T>) -> Value<T> { meta: OpInfo {}, ... }
op Gather<T: Integer & Vector, U: Vector>(ptr: Value<PTR>, index: Value<T>) -> Value<U> {
    verify { require(U.same_shape(T), "index and result must have the same shape"); }
    meta: OpInfo {}, ...
}
op Convert<T: I32 | I64, U: F32 | F64>(arg: Value<T>) -> Value<U> { meta: OpInfo {}, ... }
```

`|` means union and `&` means intersection; `&` binds more tightly. Parentheses
group expressions, for example `(I32 | F32) & Scalar`. Both operators also work
inside `vectors(...)` and `typeset` declarations. Repeated operands are idempotent:
`I32 | I32` is the same set as `I32`. Empty intermediate sets are allowed,
but an empty final set or signature constraint is an error. Unknown names and
invalid vector inputs are checked even in branches whose intersection is empty.

`T: I32 | I64` selects one concrete type for `T`; all occurrences of `T` must match.
By contrast, `lhs: I32 | I64, rhs: I32 | I64` allows the operands to independently
select their types. Set expressions contain concrete types and set aliases, not
type variables. Independent generics and ordinary `verify` predicates express
relations such as `U.same_shape(T)`. `element(T)` and `vector(T)` remain
structural type patterns. There is no separate `shape(T, set)` pattern.

Generation evaluates and interns equal sets, including anonymous expressions.
Named aliases and inline constraints share the same compact runtime membership
checks; runtime code neither evaluates expressions nor constructs sets. Builder
inference, semantic checks and text codecs inspect resolved sets, not set names.

## Traits and effects

Shared flags are Rust `bitflags` types. Their interfaces are ordinary declarations:

```text
type MemoryEffects = rust("veloc_types::MemoryEffects") {
    trait: rust("veloc_types::traits::MemoryEffects"),
    const READ: Self;
    const WRITE: Self;
    const fn empty() -> Self;
    const fn union(self, other: Self) -> Self;
}
```

The Rust owner explicitly implements the generated const trait. Ops use
`MemoryEffects::READ.union(MemoryEffects::WRITE)`. Associated constants and
static methods are checked like instance methods; unused declarations still
belong to the Rust trait contract. Repeating a flag in a union is legal and has
the semantics of Rust's implementation.

The explicit `analysis: traits` and `analysis: memory(MemoryEffect::NONE)`
annotations select the operation-analysis interfaces. They are not value schemas:
opgen neither knows the bit positions nor recognizes `Known` / `Unknown`
constructors. Primitive algebra supplies derived logical facts; generated Rust
checks that user-declared flags and memory behavior do not contradict them.
The generator does not guess an analysis role from a Rust path or type name.

### Typed operation metadata

Structs are shared by instruction properties and build-time metadata. Enums
and flags are ordinary field types, not special operation keywords:

```text
import "../../defs/types.spec";
struct Example {}

op Example() -> () {
    meta: OpInfo { traits: OpTraits::MAY_TRAP, memory: MemoryEffect::known(MemoryEffects::READ.union(MemoryEffects::WRITE)) },
    mnemonic: "example", storage: Example {}
}
```

The generic value checker resolves each field against its declared type. It
checks defs-owned enum variants and their payloads, declared associated members,
nested structs, optional values (`none` / `some(value)`), integer ranges, and
required fields. A struct body consists directly of `name: Type` fields, in
declaration order. Every field must be supplied when constructing a struct,
including optional fields (use `none` explicitly). Structs have no defaults
and do not generate Rust `Default` implementations. Nested struct and enum
values are checked recursively; inline recursive types are rejected.
There is no separate metadata schema or runtime attribute dictionary.

Each compilation unit uses one metadata struct type, selected by its operations'
`meta: StructName { ... }` values. That struct may contain arbitrary declared
fields; it need not be named `OpInfo`. MIR stores it inline in each static
`OpSpec`, and `Opcode::meta()` borrows that value. LIR emits its own typed
`GenericOpcode::meta()` table. Neither stores metadata on each instruction.

The operation-contract adapter selects fields with the explicitly bound traits and
memory interface types, regardless of field names; more than one field of either type
is ambiguous and rejected. Other fields are just typed data. Semantic laws and
LIR flow contracts populate inferred facts before final struct checking. Pure
interface expressions may supply constant metadata fields explicitly. If a memory
field exists, an unmodeled operation must supply it. A struct without a memory
contract is conservatively unknown.
Missing operation traits mean no additional declared traits, not inferred purity.

The former top-level `traits` / `memory` entries and named `effect`
declarations are rejected. `MemoryEffect::NONE` means no memory effects; `MemoryEffect::UNKNOWN`
is a separate enum variant, not a flag or every known member combined.
The `MemoryEffect` enum and its trusted methods remain in Rust; they implement
memory interference, deletion and possible behaviors.
Per-access volatility belongs to `MemFlags`; MIR atomic ordering is not
currently modeled. Addresses and widths belong to access contracts and alias
analysis.

This is a vocabulary, not an arbitrary executable extension language. Generic
type inference, memory-conflict algorithms, primitive bitvector meanings and
reviewed algebraic laws remain Rust. Declaring a new trait does not invent an
optimization or prove a law. Trusted laws use `BvConst` directly, rather than
copying their constant names into a second definition whitelist.

## Operation signatures

```text
struct Binary {
    args: values(2),
}

op IAdd<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    meta: OpInfo {},
    mnemonic: "iadd",
    storage: Binary { args: [lhs, rhs] },
    semantics: bv.add(lhs, rhs)
}

op ExtendU<T: Integer | BOOL | vectors(BOOL), U: Integer>(arg: Value<T>) -> Value<U> {
    meta: OpInfo { memory: MemoryEffect::NONE },
    mnemonic: "extendu",
    storage: Unary { arg: arg },
    verify {
        require(U.same_shape(T), "input and result must have the same shape");
        require(U.wider_than(T), "result must have more bits per lane than arg");
    }
    }

op Load(ptr: Value<PTR>, offset: u32, flags: MemFlags) -> (result: Value<Any>) {
    meta: OpInfo { traits: OpTraits::MAY_TRAP, memory: field(memory_access, effects) },
    mnemonic: "load",
    storage: Load { ptr: ptr, offset: offset, flags: flags },
    text: "{.flags} {ptr}, offset={offset}",
    query memory_access -> MemoryAccess {
        ptr, offset: i64(offset), ty: result, stored: none, flags,
        effects: MemoryEffect::known(MemoryEffects::READ),
    }
    }
```

Type requirements use the same `verify { require(predicate, diagnostic); }`
expressions as structural checks; the old `where` relation list is not supported.
Named results and generic variables directly denote types: `result.wider_than(T)`.
Operand names denote SSA values, so their types use the declared `arg.ty()` method.
Give a result a name when referencing it in a verifier or instruction query.
Result/type bindings resolve to signature slots at build
time; no runtime name lookup or generic environment is stored. `type.element_bits()?` is a logical per-lane width, while
`type.bit_size()?` preserves the whole-value fixed/scalable distinction. Undefined
width queries (e.g. target-dependent pointers) fail the requirement; boolean
short-circuiting can guard such queries.

After expanding helper functions, constraints independent of SSA identities,
properties and host queries are emitted into `Opcode::validate_types`. They are
not repeated in the function's structural validator and never run during
instruction construction. The build-time semantic instantiator evaluates these
same expressions on admitted scalar/vector types before preparing lane recipes.
It preserves checked arithmetic and short-circuiting, and does not promote a
vector-only valid recipe into a scalar constant evaluator. Constraints requiring
properties or DFG/host queries remain in the structural validator; alternate
layout constraints remain local to that layout.

### Named instruction queries

Query results are ordinary structs. Declaring a struct alone does not generate
an instruction method or make that struct an instruction-storage layout.

```text
struct MemoryAccess {
    ptr: Value(Type::PTR),
    offset: i64,
    ty: Type,
    stored: optional(Value),
    flags: MemFlags,
    effects: MemoryEffect,
}

op Load(ptr: Value<Type::PTR>, offset: u32, flags: MemFlags) -> (result: Value<Any>) {
    meta: OpInfo { traits: OpTraits::MAY_TRAP, memory: field(memory_access, effects) },
    mnemonic: "load",
    storage: Load { ptr, offset, flags },
    query memory_access -> MemoryAccess {
        ptr,
        offset: i64(offset),
        ty: result,
        stored: none,
        flags,
        effects: MemoryEffect::known(MemoryEffects::READ),
    }
}
```

The query name determines the generated method: `inst.memory_access(dfg)`.
The return type is independent of that name; different queries can return the
same struct. Every implementation of a particular query must agree on its
result type and any explicit context type. A query may occur only once per
operation. Context-free implementations can participate in a contextual query.

The body uses ordinary checked struct construction with same-name field
shorthand. All fields are required and helper calls are allowed in expressions.
There is no separate interface field model, data emitter or recursive-type
checker. The old `interface` declaration and `implements` field are rejected.
`Value(Type::PTR)` refines an SSA reference at definition time but is represented
as `Value` in Rust. Ordinary struct fields may also contain other structs,
enums, optional values and fixed arrays; inline cycles are rejected.

For an external dependency, declare a concrete Rust context explicitly:

```text
query stamp_info(ctx: Tokens) -> StampInfo {
    stamp: ctx.stamp(number),
    doubled: ctx.stamp(number).twice(),
}
```

This generates `inst.stamp_info(dfg, &tokens)`, without context construction or
conversion. Queries dispatch by opcode and read fields and result types only
when their expressions need them. Callers cannot supply an unrelated result
list. Unsupported instructions or unavailable results return `None`; queries
do not revalidate MIR, add persistent fields, or allocate a new instruction.

Metadata can reference a query by its explicit name, for example
`field(memory_access, effects)`. The selected field must be compile-time
constant. Memory behavior remains a trusted declaration, not a generator
special case or a proof about executable semantics.

The query emitter currently targets packed MIR. Operand-array output rejects
query declarations during planning. Bounds and provenance analyses remain
ordinary Rust code consuming the generated query result.

Ownership transfer is attached to the logical parameter:

```text
op Call(func_id: FuncId, move args: sequence(Value)) -> signature {
    meta: OpInfo { traits: OpTraits::MAY_TRAP, memory: MemoryEffect::UNKNOWN },
    mnemonic: "call",
    storage: Call { func_id: func_id, args: args },
    signature: function(func_id),
    text: "{func_id}({args}) : {function(func_id)}",
     }
```

`move` consumes owned values; ordinary scalar, pointer and reusable callable
values remain duplicable. Properties cannot move. Successor arguments transfer
on the selected edge, without a separate annotation. There is no `moves` list.

SSA operands explicitly use `Value<T>` in the operation signature. Result names are
optional: a single result is `-> Value<T>`, multiple results are `-> (Value<T>, Value<Type::BOOL>)`, and
zero results are `-> ()`. Parenthesized results may be named when a constraint
needs to reference them, as in `-> (result: Value<U>)`. Otherwise the
MIR definitions omit result names, including overflow operations.
Names do not affect the
generated representation; anonymous results have no implicit names or aliases.
Generic variables such as `T` are scoped to that operation; their first direct
occurrence binds the type, including a result-only generic such as `U`.
Derived types use `element(T)` or `vector(T)`. Other type relationships use
ordinary declared methods in `verify`; generics are bound from actual operand
and result types, not solved by those predicates. Construction does not validate
these relationships. Relations refer to generic or operand/result names, not numeric slots.
There are no separate `types` structs or references to named type schemes.

Parameter roles are declared independently of storage: `ptr: Value<Type::PTR>`
is an SSA input, whereas `offset: u32` and `flags: MemFlags` are data properties.
Storage mappings are checked against these roles; they cannot change them.
`Value<T>` describes a logical SSA reference, not a runtime wrapper allocation.
Both packed MIR storage and operand-array LIR storage use the same signature checker.
The constructor must be imported and declared with `field: operand`; another
such declaration, for example `Ref`, can be used as `Ref<T>` with the same rules.
Variable-length SSA groups use `args: sequence(Value)`; one successor uses
`dest: successor`, and a successor group uses `cases: successors`. Empty input
and result lists are `()`; signature-selected results use `-> signature`.
The sequence element must be an imported Rust-bound type declared with
`field: operand`; its spelling is not special. `move args: sequence(Value)`
transfers each element. Sequence storage remains the existing compact operand
list, not a newly allocated Rust collection. The old bare `values` parameter
keyword is not accepted. Storage declarations such as `values(2)` describe
fixed-size physical fields and are separate from operation parameter types.
For example, an indirect call declares a statically checked `ptr: Value<Type::PTR>`, a
variable-length `args: sequence(Value)` group and `signature: sig_id`. Direct calls use
`signature: function(func_id)` to identify the callee's signature. The source of
dynamic result types is explicit, not inferred from the opcode's name.

Rust-bound types declare reference structure separately from payload placement:

```text
type Value = rust("crate::Value") {
    field: operand,
    fn ty(self) -> Type { value: type(self) }
}
type ValueList = rust("crate::inst::Arguments") { field: list(Value), }
type BlockCall = rust("crate::inst::Successor") { field: edge(Value), }
type JumpTable = rust("crate::inst::Successors") { field: list(BlockCall), }
type Payload = rust("crate::Payload") {
    storage: pooled,
}
```

An absent `field` means plain data. `operand` is an SSA leaf;
`list(T)` composes a sequence and `edge(T)` associates a block target with
a sequence of SSA parameters. The element type must be declared and imported.
Aliases resolve structurally, with cycles rejected. Rust-bound views currently
support operand leaves, operand lists, edges, and edge lists; unsupported
nested reference containers are diagnosed rather than losing uses. Structural
records support direct and optional SSA leaves.
Placement defaults to `auto`: compact codecs may inline metadata if the whole
layout fits. `pooled` forces the containing payload out of line, without
moving its SSA references out of the operand store.
The generator uses checked structure, not the Rust type's spelling, to split and
traverse SSA fields. Fixed `values(N)` groups are inherently SSA operands.

Each out-of-line instruction layout has a generated payload type and its own
`Pool<T>`. A generated trait selects that pool for the common `push<T>`,
`get<T>`, and `remove<T>` interface. A payload groups the layout's non-SSA
fields to avoid a separate allocation/lookup for every field; the pools do not
store a maximum-sized enum containing every layout. Small layouts stay inline.
Typed IDs are private to the owning DFG and invalidated on replacement or
erasure; freed slots are reused. They are not public, generation-checked handles.
SSA uses remain in the DFG's operand arena and are never managed through these pools.
`FieldPool` also owns an immutable `InternPool` for constant bytes. Interned
entries are shared by content and live until the DFG is dropped; erasing an
instruction never releases them. Small scalar and splat constants stay inline.

A single `struct Name { field: Type, ... }` declaration describes both plain
structured data and instruction storage. `storage: Name { ... }` selects its
instruction use; there is no separate `format` declaration. The generator
derives opcode discrimination from the operations using the struct: a single
operation has a fixed opcode; a shared struct has a generated opcode parameter.
Neither form adds a field to the ordinary Rust struct when it is also used as
a nested property or metadata.

The mapping connects logical parameters to generated construction/view fields.
Every declared field is mapped; fixed groups use `[lhs, rhs]`. The generated
builder supplies any dynamic opcode. Variadic groups use `ValueList` in the
schema, but become ordinary slices in views, never a mutable Value-list pool.
Only byte properties use `pool(bytes)`; structs such as `PtrIndexImm` and
`VectorMemOptions` bind directly. Branch tables use `table(cases, default)`,
with the default destination last.

Both storage strategies use explicit `storage: Layout { ... }` mappings.
Object fields support same-name shorthand: `Unary { arg }` means
`Unary { arg: arg }`. Bare layout names are rejected. Declarations such as
`struct` still require field types; shorthand is only for object values.

Operand-array storage also requires `opcode`, `view`, `reader` and `writer`
names. The latter two name generated traits, not concrete host types.
Their default methods use only their declared access/write contracts.
Hosts supply an error type, instruction ID type and output-register wrapper
through associated types. No `InstRef`, `InstWriter` or `MachineOpcode` name
is assumed by the operand generator.

In operand-array storage, the storage declaration explicitly names the register
type and the attribute enum. Enum payload types select codecs; there is no
hardcoded type-name registry. Register fields bind named results or input
parameters. Layout names, field names and field order need not match signatures.
`optional(T)` uses `some(input)` or `none`; omitted fields need not be a suffix.
`sequence(Reg)` binds a register slice, and `results()` binds the result slice.
Each domain supports one trailing sequence after any fixed prefix; attribute
sequences are not yet supported. Calls use these same rules, without a call-shape
adapter. Every input and fixed result must be mapped exactly once.
Builders, direct views and the optional structural validator share the checked
projection. Neither construction nor views automatically run validation.
Both storage backends normalize logical input reads into `model/access.rs`.
Arrays, optional fields, pools and branch-table projections are resolved once.
Constraint, query, ownership, text-printing and result-type consumers use these
logical paths rather than reinterpreting storage bindings. Query generation
shares one loop with host-specific dispatch/read setup.

Property contracts and operation constraints use the same emitter on both IRs,
including explicit contexts, local bindings and fallible operations. Properties
are checked inline; no inherent validator is added to a foreign Rust type.
Array readers supply value-to-type lookup only when their expressions need it.
Packed constraint validation receives its read-only DFG explicitly from the
Rust validation entry point; generated reads do not access Function fields.
Borrowed view declarations share a storage-independent plan and emitter: fields,
lifetime propagation and opcode subsets are described once. Layout adapters
select inline variants or named records and supply physical field types/reads.
Construction uses a shared prepared argument list, not another expression AST. Layout adapters supply
argument order, attribute conversions, pooling and fixed/tail slices; builders
and parsers compile the same plan. Packed hosts create SSA results; array hosts
accept result registers. These allocation policies are intentionally different.

Text schemas, atom parsing, named-field checks, printing and record assembly use
one compiler. Array operations with an explicit `text` projection now produce
`text_parser`/`text_printer` artifacts; the host supplies Cursor/AtomCodec,
OperandParser::write and InstPrinter::fmt_head. Result registers are supplied
by the enclosing parser and checked before indexing. The current LIR crate does
not yet expose a complete textual frontend; these artifacts are exercised by
the generated-code execution tests, not silently substituted into MIR's parser.

Signature contracts share resolution/check emission. An array reader declares
only the used signature lookup methods, returning parameter/result type slices;
its generated validator compares counts and value types. No generator-side
module lookup or concrete signature container is required. Physical decoding and
the enclosing function/parser infrastructure remain host responsibilities.

Packed MIR retains its existing SSA operand-order invariant and its `pool`,
`table` and fixed-array adapters; unifying syntax does not change physical
storage or introduce optional runtime wrappers.

Record names and fields belong to definitions. Generated storage extracts Value
and optional(Value) members as auxiliary operands and stores only non-SSA fields
and presence bits. Adding a struct operand needs no DFG visitor or mutation
adapter. Built-in format field contracts remain checked because hand-written
lowering consumers destructure their generated, typed views.

The definition compiler checks references, field coverage, type variables,
arities, constraints, semantic compatibility and generated method names before
emitting Rust. Definitions may refer to later structs.
A separate `layout Name { ... }` configures storage-specific projections without
redeclaring fields: MIR predicated alternatives select canonical formats;
LIR derives view names from structs and does not accept layout overrides.
Its opcode-dispatched `view()` returns a `InstView`, with borrowed
register lists for variable operands. Shared formats carry a generated restricted
opcode enum. LIR operand counts follow explicit storage mappings, rather than a
second explicit list of lengths. Diagnostics include source
line and column; the build script maps combined input locations back to the source
file.

## Generated consumers

The same definitions generate `Opcode`, `OpFormat`, `InstWriter`, type
contracts, opcode extraction, operand traversal/replacement, memory flag access,
operation-specific parsing/printing and ordinary builders.

### Compiled type contracts

Type signatures drive two independent paths: result construction and validation.
Generated builders compute result types directly from logical arguments and pass
fixed-size arrays to `InstBuilder::emit`, which inserts and retrieves exactly
that many results. Generated methods return a `Value` or a tuple; inferred result
counts are not limited to two. Zero-result instructions call `insert` with an
empty type slice instead of invoking dynamic inference. All paths use the same
`insert` primitive and none validates the type contract. The public insertion API
also accepts caller-supplied result types for generic transformations or
deliberately incomplete IR.

`InstView::result_types` is the dynamic construction entry point used by
contextual builders and generic construction clients. Its generated opcode
branches return the final types directly, using operand types, explicit types or the referenced
signature. There is no runtime result-strategy enum. Missing explicit types,
unknown signatures and operands whose types cannot determine the result are
construction errors, not full contract validation.

`Opcode::validate_types` dispatches to shared generated checks. Full module
validation is an explicit pipeline/caller decision; builders and the parser do
not invoke it implicitly. The parser checks syntax and symbol resolution, but
takes every result type directly from its SSA declaration. Result counts and
operand/result type consistency belong to the validator, not the parser.

Selection uses signature structure, not opcode names. Equal operand/result
patterns and type sets share structural handlers, using equality of the checked
definition model rather than serialized runtime descriptors. Each opcode then
runs its type-only requirements, even when its signature is shared with another
opcode. Exact pattern slots are retained so sharing preserves error diagnostics.

The definition compiler resolves type variables to concrete operand/result
positions. Generated handlers check arity before indexing and retain the
declaration's diagnostic order, without allocating bindings or interpreting
patterns or constraint records. The number of variables is not limited to four.
Result construction knows statically whether results are fixed, require explicit
types, or come from a function signature. It performs no type-set, operand
equality or verification-expression checks. Those checks run only during validation,
including for result types computed by generated builders.

Type schemes, patterns and constraint expressions exist only in the definition compiler.
MIR contains no runtime `TypeScheme`, `TypeList`, `TypePattern`, relation table,
or per-opcode type descriptor. Generated checks return `TypeError` with static
diagnostic strings and relevant operand/result positions. Interned `TypeClass`
membership checks remain executable helpers, not a second type-rule interpreter.

Structural `verify` blocks contain typed, pure expressions, compiled directly into
Rust checks. For example:

```text
verify {
    require(imm.scale != 0, "scale must be non-zero");
    len(mask) == lhs.ty().lanes()?;
    all(mask, |i| i < 2 * lhs.ty().lanes()?);
}
```

Expressions reference logical parameters and struct fields, not physical pool
IDs or layout names. The existing storage projection resolves those references.
The language provides Boolean logic (`!`, `&&`, `||`), comparisons, checked
integer arithmetic (`+`, `-`, `*`) and bitwise operations (`&`, `|`), comparison-enum literals such as `IntCC::Eq`,
and lexical `all(sequence, |element| predicate)` over finite sequences and optional values.
Multiple inputs use `all(lhs, rhs, |a, b| predicate)`: input expressions evaluate once,
left-to-right; unequal lengths return false without running the predicate. Equal-length
inputs are traversed in order and stop at the first false predicate. Empty inputs
satisfy the predicate vacuously; an optional value has length zero or one.
This emits a loop without temporary collections, including in Rust const contexts.
Domain policies are ordinary helpers, for example:

```text
fn matches_types(values: sequence(Value), types: sequence(Type)) -> bool {
    value: all(values, types, |value, ty| value.ty() == ty),
}
```

It has no arbitrary Rust callbacks, user recursion or unbounded loops.
Verification arithmetic uses checked signed 128-bit integers, not wrapping
instruction values; property integers are widened without truncation.
Typed helper bodies and arguments use their declared integer types instead.
Constant arithmetic overflow is a definition error; dynamic overflow fails
validation (or returns `None` from an instruction query). Neither path wraps.

Queries use named results, `value.ty()`, `len(sequence)`,
`type.lanes()?`, `type.min_size_bytes()?`, `type.is_ptr()`, `type.is_scalar()`,
`type.is_vector()` and `type.is_fixed()` (a fixed-width vector).
Lane counts and byte sizes are minima for scalable types. A target-dependent
byte size is an evaluation error. Result names must refer to declared fixed
results. Associated constants are checked against their type declarations; struct
fields against the struct definitions. Unsupported property kinds are rejected instead of guessed or silently coerced.

`require(predicate, "diagnostic")` supplies an optional diagnostic; a bare
predicate uses its expression text. Errors include the instruction and opcode.
Short-circuit operators do not evaluate the skipped branch, including pool
lookups; `all` stops at its first false element and is true for an empty list.
Nested binders are lexically scoped and emitted with hygienic identifiers.

The checker produces a typed predicate tree, folds constants and uses signature
type sets to fold type queries when all possible types give the same answer.
Statically true checks disappear; an always-false top-level constraint is a
definition error. Even dead branches must be well typed. Type validation runs
first, allowing generated checks to use operand/result positions directly,
including for alternative physical layouts. Actual property checks remain
dynamic. Constraint diagnostics are emitted directly into the generated checks;
there is no runtime rule interpreter or predefined constraint-name registry.

The same pure `fn` can be used by verification, interface projections and
compile-time metadata. For example, the alignment check is a definition, not a
built-in verifier keyword:

```text
fn is_power_of_two(value: u32) -> bool {
    value: value != 0 && (value & (value - 1)) == 0,
}
op Alloca(size: u32, align: u32) -> Value<PTR> {
    // ... storage, metadata and text ...
    verify {
        require(size > 0, "alloca size must be positive");
        require(is_power_of_two(align), "invalid alignment");
    }
}
```

Helpers are checked once and expanded at build time, with fresh names for
lexical binders. They cannot recurse. Runtime value/type queries still need a
DFG; module queries are only available to instruction verification. Pure helpers
may call them; generated Rust trait calls enforce their transitive host requirements.
An ordinary instruction query cannot gain module access by hiding it in a helper.
Static metadata must reduce to a constant; it cannot read runtime operands.
The old `constraints: [...]` syntax is rejected; there is no compatibility path.
These declarations do not add construction-time checks to builders.

These predicates specify IR legality, not instruction execution or traps.
For example, a division instruction with a zero divisor can be valid IR with
defined trapping behavior; that belongs in executable semantics, not here.

The synthetic benchmark separates contract checks, module construction,
validation and constant folding:

```sh
cargo run --release -q -p veloc-optimizer --example type_schemes -- all 3
cargo test -p veloc-mir -p veloc-spec -p veloc-filetests
```

It contains 200 repetitions of add/sub/mul/extend/wrap, plus constants and return
(1004 instructions). Folding excludes cloning, final validation and destruction
from its timer. This contract-heavy workload is not representative of overall
Wasm or compiler throughput; gains in construction or validation do not imply
equal gains in optimization or execution.

Migration snapshot: release profile on an AMD Ryzen 9 9950X virtualized host,
CPU affinity 2, three alternating runs against the saved pre-migration generic
executable. Each value is the median of three nine-sample medians:

| Work per iteration | Generic | Compiled contracts |
| --- | ---: | ---: |
| Five type validations | 78.0 ns | 28.3 ns |
| Three result inferences | 69.9 ns | 13.2 ns |
| Build and drop module | 44.03 μs | 36.57 μs |
| Validate module | 27.64 μs | 13.83 μs |
| Fold 1000 instructions | 474.8 μs | 454.9 μs |

These are local observations, not statistical guarantees. The linked example's
`.text` grows from 469013 to 514293 bytes (9.7%). Full generation also has a
tradeoff against the earlier partial-specialization prototype: a comparison run
of that prototype measured 9.5 ns for inference and 30.15 μs for construction,
both faster than the fully generated version on this workload. Eliminating the
interpreter does not guarantee that every path gets faster; generated code size
and the Rust compiler's inlining decisions still matter. Cold build time and
representative application throughput have not been measured.

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

Pool-backed and fixed-length-list operations use the same generated builders:
`vconst(bytes: Vec<u8>, ty: Type)`, `ptr_index(ptr, index, imm: PtrIndexImm)` and
`gather(ptr, index, mem: VectorMemOptions, ty: Type)`. Packing interns byte properties
through `ConstantPoolId::insert`; struct properties are stored inline. Contextual helpers
remain for variadic groups, CFG destinations and signature-selected results.
They provide higher-level slices and blocks while installed operands occupy one
flat range. Generated builders compute result types without validating the type
contract; contextual builders resolve only the information needed for results.
Full validation remains an explicit phase.

## Bidirectional text projections

Without a `text` field, an operation prints its logical parameters in signature
order, separated by commas; a zero-parameter operation has no operand text.
An explicit projection changes notation without changing the builder API or
storage layout:

```text
op Store(ptr: Value<PTR>, value: Value<Any>, offset: u32, flags: MemFlags) -> () {
    meta: OpInfo { traits: OpTraits::MAY_TRAP, memory: MemoryEffect::known(MemoryEffects::WRITE) },
    mnemonic: "store",
    storage: Store { ptr: ptr, value: value, offset: offset, flags: flags },
    text: "{.flags} {value}, {ptr}, offset={offset}",
     }
```

The template generates both the parser and canonical printer at build time;
there is no runtime template interpreter. Without an explicit template, logical
parameters retain their declaration order.

- `{field}` selects a logical parameter; `{struct.field}` selects a struct leaf.
- `{kind} {lhs}, {rhs}` spells a comparison such as `eq v0, v1`.
- `offset={offset}` is required even when the offset is zero.
- `[, mask={mem.mask}]` is optional only because the field is an optional SSA
  value. Ordinary numeric fields do not have text defaults or optional groups.
- A leading `{mem.scale=1}` fixes an unprinted integer field for this text
  projection. Its literal must fit the field type. Parsing supplies that value;
  printing rejects a different value rather than silently dropping it.
- `{.flags}` or `{.mem.flags}`, at the start of a template, binds mnemonic
  suffix flags. Empty flags produce no suffix.
- `{data:bytes}` selects hexadecimal byte text; `{value:integer}` selects
  signed bit-pattern text for a u64 property. Most atoms infer their codec from
  the logical property type, including Int, Float and VectorConst.

For example, strided loads use:

```text
text: "{.mem.flags} {ptr}, stride={stride}, offset={mem.offset}[, mask={mem.mask}][, evl={mem.evl}]"
```

Calls use `{func_id}({args}) : {function(func_id)}` for direct functions,
`{ptr}({args}) : {sig_id}` for explicit signatures, and `{callee}({args})`
for callable values. The function signature projection reuses the referenced
function's signature, without adding a property to instruction storage.
The parser checks that the textual declaration agrees with the function symbol;
the validator checks argument and result types.

The grammar is intentionally bounded: positional operands precede named fields,
and optional groups contain a single named optional SSA value. Named fields may
be parsed in any order, but print in template order. Variadic values and
successors retain their comma-list and bracketed-list syntax.

Templates account for every logical field through an input, an optional field,
a flags suffix or an explicit fixed binding. Unknown or duplicate references,
incompatible codecs, missing
fields and invalid optional groups are definition errors. Unsupported input
fields are rejected, never silently discarded. Alternate storage layouts
declare their own extension templates, preserving mask/EVL predication.

MIR text uses an on-demand token cursor with source spans and recursive-descent
parsers for declarations, types, signatures and successor lists. Physical
newlines delimit statements; punctuation does not require surrounding whitespace.
The source is consumed once, without declaration prepasses, saved body ranges or
instruction reparsing. Lookahead caches tokens until consumption, including the
top-level named-field lookahead for alternate layouts. Line/column positions are
tracked while lexing rather than recomputed by scanning a source prefix.
One outer statement loop checks the line boundary for every declaration and
instruction; generated operand parsers do not repeat that check. Parse errors
carry a structured source location separately from their message. Adding operand
context preserves that location, and unresolved symbol errors retain the
original reference position.

Function references reserve parser-local slots; explicit call signatures establish
the referenced function's signature before its declaration. Later declarations
must agree, and unresolved names are errors, not implicit imports. Function names
are first parsed as temporary text references; registration happens only after
the argument list and complete signature have been read. Every created Function
therefore has a valid signature ID. Symbol structs do not duplicate the signature
or represent a signature-unknown state. Standalone FuncId atoms likewise carry a
signature, for example `foo : (i32) -> i32`; invoke syntax places that signature
after the arguments. Finalizing function IDs
preserves source declaration order and numeric references without rereading text;
the field schema generates the function-reference remapping. Identity mappings
skip this IR traversal. Blocks reserve storage on first use but join layout order
only at their definitions. SSA definitions fill the same Value slots reserved by
references, without RAUW. Numbered spellings and their name hints share a symbol;
if an unnumbered spelling already occupies the preferred slot, the numbered
symbol receives a distinct slot rather than accidentally aliasing that value.

Every SSA result declares its type using the same `name: Type` syntax as block
parameters. Single results use `sum: i32 = iadd lhs, rhs`; multiple results use
`(sum: i32, overflow: bool) = iadd-with-overflow lhs, rhs`. Zero-result
instructions have no assignment. Type suffixes on opcodes and untyped result
definitions are not accepted; suffixes describe memory flags only.

Parsing result declarations claims their Value IDs and rejects duplicate names
before reading operands. After ordinary instruction insertion, the DFG binds
those reserved IDs to the instruction and stores its result list. This preserves
forward uses without a separate instruction-creation path or temporary result IDs.

The parser creates exactly the declared results without inferring their types
from operands or signatures. Explicit types remain in the IR even when wrong,
so the validator can diagnose type or result-count mismatches. Forward references
reserve value IDs and definitions fill their types; there is no result-type
propagation, pending-instruction queue or inference retry. Contextual Rust builders
retain their generated type inference independently of the text syntax.

Generated instruction parsers consume the cursor directly in projection order.
Named fields use typed local slots and a generated key match, with explicit
duplicate/unknown-field checks. Alternate layouts use top-level named-field
lookahead. There are no operand substring lists or runtime grammar descriptors.
Symbol resolution, pool access and CFG construction remain shared Rust algorithms.
Full type/IR contracts are checked only by an explicit validator call, not by parsing. This is a finite text schema, not an arbitrary
parser-generator language.

### Generated/runtime contracts

Text atoms implement the internal `AtomCodec` trait, pairing `parse` with `print`.
The emitter selects one codec type for both directions instead of maintaining
separate reader/writer function-name mappings. Codec identity describes notation:
`IntegerBits` and `FloatBits` both store `u64`, while `Decimal<u64>` is unsigned
decimal text. Associated `Owned` and `View<'a>` types let parsing produce a vector
and printing borrow a slice. Contextual codecs reuse the token cursor and symbol
resolution algorithms; this does not require a trait for every syntax helper.

Only immutable byte constants are interned. Generated `pool(...)` mappings call
`ConstantPoolId::insert` and `get` directly; there is no generic pool trait or
parallel set of DFG getters/interners. Byte constants are stored as shared `Arc<[u8]>` buffers: the
pool and deduplication index share one payload, while reads borrow `[u8]`.
Operand groups and successor arguments use the unified operand storage instead.
The atom codecs use static dispatch without a registry or trait objects. Rust checks the implementations and generated calls; round-trip tests
remain necessary to check that the two directions agree semantically.

## Semantics and lowering

`semantics: bv.add(lhs, rhs)` describes the result using the named logical
operands and the modular primitives in `veloc-semantics`. Expressions can compose
primitives: integer negation is `bv.sub(bv.zero(), arg)`, rather than a second
handwritten implementation of negation. Constants optionally specify their type,
e.g. `bv.zero(type(arg))`; without it they use the first input type, or the first
result type for an input-free operation. MIR-to-LIR arithmetic translation maps recognized
primitive applications; composed negation retains the existing `Neg` lowering.
The definition compiler binds concrete input/result sorts and emits specialized
scalar evaluators. The optional offline `Program` API binds signatures to the
graph used for reference execution and SMT export. Each recipe describes a scalar
operation or a per-lane operation, not an entire vector, memory or machine-state
model. Absent expressions mean **unmodeled**,
not a claim that an operation is pure or verified.

Executable semantic expressions infer `MemoryEffect::NONE`. A complete `access` contract
also supplies its memory summary; other unmodeled operations must declare their
memory effect in `meta` explicitly. The metadata traits field may be omitted.

The semantic backend supports modular arithmetic, bitwise operations, comparison,
selection, shifts, division, bit counts, signed/zero extension, truncation and
multiple typed results:

```text
semantics: bv.cmp(kind, lhs, rhs)
semantics: bv.sext(arg, result(0))
semantics: [bv.add(lhs, rhs), bv.ult(bv.add(lhs, rhs), lhs)]
```

The last example returns a sum and UNSIGNED carry; MIR's `IAddWithOverflow`
instead explicitly defines SIGNED overflow using sign-bit arithmetic.
`type(operand)` and `result(index)` refer to signature sorts, not runtime values.
`kind` is an `IntCC` property whose signedness/outcomes come from the Rust condition-code implementation,
not a runtime SSA input. Property extraction follows the storage mapping.
Bool is a distinct sort: bitwise and/or/xor support it, arithmetic does not.
Zero extension explicitly converts Bool to a zero-or-one bitvector.

Operation-level traps are explicit, typed guards over the same graph:

```text
semantics: bv.sdiv(lhs, rhs),
traps: [
    DivisionByZero(bv.eq(rhs, bv.zero())),
    IntegerOverflow(bv.and(bv.eq(lhs, bv.smin()), bv.eq(rhs, bv.ones())))
]
```

The first true guard wins; `MAY_TRAP` is inferred from these guards. Declaring
`MAY_TRAP` with executable semantics but no guards is rejected. Guards model
observable failures, not preconditions that remove inputs from SMT checks.
Constant folding replaces an operation only when no guard fires; a known trap
stays as an instruction. Lowering must separately preserve these outcomes;
adding the contract does not generate backend trap checks automatically.

Raw `bv.sdiv/udiv/srem/urem` use total SMT-LIB semantics. Signed remainder of
MIN and -1 is zero, not an overflow trap. Raw `bv.shl/lshr/ashr` do not mask
counts: MIR explicitly uses `bv.urem(rhs, bv.width())`. `bv.width()` and
`bv.smin()` are constants in the selected bitvector sort, not runtime queries.
Rotations compose shifts and bitwise OR; `bv.clz/ctz/popcnt` provide bit counts.

The definition compiler checks the recipe against all admitted scalar element
types, evaluating type-only requirements and requiring shared lane shapes. It does not
claim to model reductions, scalar broadcasts, predication, or whole vectors.
The same signature enumeration drives `evaluation.rs`: only legal scalar types
representable by MIR `Constant` are emitted, including legal conversion pairs.
Widths, step references, masks and output layouts become Rust literals and local
variables. Comparison properties remain parameters, avoiding a predicate/width
Cartesian expansion. The optimizer calls this generated code; it
does not instantiate or interpret a graph, and has no graph fallback. Exact
signature dispatch also rejects unsupported calls without a second full type
contract check. Multiple results become constant definitions while preserving
SSA value IDs and use-def information.

MIR owns representation and validation, not evaluation or rewrite rules. Neither
MIR nor the optimizer has a normal dependency on `veloc-semantics`; generators and
offline tools depend on it at build/test time. `OpSpec` contains no primitive
identifier, identity/absorbing constants, or semantic recipe.

`veloc_optimizer::rewrite::evaluate` executes specialized Rust arithmetic, with
no runtime primitive dispatch. `SimplifyPass` applies constant evaluation and
generated identity, absorbing-element and idempotence rules through a common
replacement path that maintains SSA, layout and use-def information. O1 runs
simplification before dead-code elimination. Removing a use of a trapping
instruction does not authorize deleting that instruction.

### Exact operands and structural editing

SSA operands have one authoritative representation. Every instruction owns a
contiguous range in a function-wide Value array. A parallel link array stores
only owner, previous and next occurrence; per-value heads select these same
slots. Ordinary operand traversal reads a compact slice without loading links.
Single-occurrence replacement unlinks, writes one Value and relinks in constant
time. Analyses iterate `dfg.operands(inst)` directly. Instruction views inspect
operands, successors, formats and memory flags without requiring a DFG; type
resolution and constant-pool access still require their context. RAUW visits occurrences, not distinct users. Iteration order is unspecified.

Operand ranges use power-of-two size sets and are recycled as units. Range
growth/replacement rebuilds links for that instruction, never the whole function.
There is no separate UseIndex, per-instruction UseId list, storage-path locator,
mirrored operand Value, or generation table.

Definitions generate named layout constructors on `InstWriter`, borrowed
`InstView` variants, and private SSA-free `InstFields`. A draft owns
one flat operand buffer; an installed instruction holds an arena range instead.
Both use the same fields, group metadata and view projection. There is no owned
instruction-shape enum or per-variant draft-to-storage conversion.

Fixed and variadic operands, successor arguments and struct inputs (mask/EVL)
are flattened by the generated constructors. Installation moves fields, copies
the flat operands into the arena and links their uses. `dfg.draft(inst)` snapshots
fields and operands without decoding/reconstructing instruction variants.
Draft operand edits and successor argument growth preserve this layout; type
contracts remain an explicit validation step. Successor constructors borrow
argument slices and branch tables accept iterators, avoiding nested temporary
argument copies.

Successor metadata contains targets and argument
lengths, not another Value list. Borrowed successor views preserve duplicate
edges and the final default edge. Small record properties are stored inline; only
immutable byte constants are interned. No mutable SSA-bearing pool is shared.

`Use<'a>` is a borrowed view with `inst()`, `index()` and `value()`, not a
persistent entity ID. Editing uses `set_operand(inst, index, value)`. Replacing
an instruction invalidates its previous positions. Borrowing prevents live
views from crossing mutations; manually saved numeric positions have no identity
guarantee and must be looked up again after structural replacement.

`Function::dfg()` and `layout()` expose read-only storage. `Function::edit()`
updates instructions, result transfers, layout and affected CFG edges. Builder,
parser, simplification and dead-code removal use this path. Standalone DFG
editing maintains operand references but does not own block layout. Type
contracts remain explicit validation, not implicit builder checks.

Storage definitions also generate successor-occurrence editing. `EdgeRef`
identifies an instruction and a successor position, not a source/destination
pair. `SuccessorMut` can redirect or resize one edge without changing parallel
edges. CFG adjacency is derived from the final instruction and unchanged targets
retain their predecessor relations. Sealing is SSA-builder-local state, not a
property of finished blocks. Validation checks structure before type projections,
then CFG consistency and entry-reachable dominance. The callable migration
contract and its implementation status are documented in
[callables.md](../mir/docs/callables.md).

Simplification uses a deduplicated worklist of affected definitions and users
instead of repeatedly scanning the whole function. Multi-result constant
replacement updates uses through the function editor. Dead-code removal erases
closed sets together so dead internal references do not obstruct deletion.

`AnalysisManager` borrows one function exclusively. Reading analyses is cached;
requesting mutable function access clears derived analyses. It cannot switch to a
different function. MIR has no revision counter or use-def synchronization
protocol. Fixed-point tests compare complete function state across another pass. LIR retains its separate non-SSA register analysis.

This establishes the ordinary mutable MIR path. It does not add an e-graph, an AI
search driver, solver invocation, general rollback or fine-grained incremental
liveness. Those mechanisms can use the edit boundary without becoming IR storage
requirements. Runtime and memory improvements require measurement.

Codegen joins checked direct MIR primitive applications with the reviewed LIR
semantics in `lir/defs/generic.spec` at build time. Both definition modules use
the shared `Source` API and checked operation model; operand-array storage
emission is separate from MIR's packed SSA projection. The same definitions supply opcode/schema mappings,
builders, decoders, control behavior and build-only primitive bindings. The result is a direct
`Opcode -> Option<GenericOpcode>` match, not an `OpSpec` semantic lookup. Composed,
reordered, property-dependent, trapping and multi-result recipes do not qualify;
contextual lowering remains explicit. The bindings are contracts, not proofs of
target legalization or code generation.

A separate `semantics.rs` artifact contains `SPECS`, a static slice of
`veloc_semantics::SemanticSpec<veloc_mir::Opcode>`, and the offline `IntCC` predicate
conversion. Only offline examples/tests include it; MIR exports neither a table
nor a macro. Tools reuse each opcode's `OpSpec` for type/effect information,
without duplicating those contracts or requiring a feature switch. Offline checks
do not imply an automatic rule-proof pipeline or runtime solver calls.

```rust
mod offline {
    use veloc_mir::{Opcode, IntCC};
    include!(concat!(env!("OUT_DIR"), "/semantics.rs"));
}
let add = offline::SPECS.iter().find(|s| s.opcode == veloc_mir::Opcode::IAdd).unwrap();
let contract = add.opcode.spec();
let recipe = add.program;
```

To compare generated evaluation with the former per-fold graph path:

```sh
cargo run --release -p veloc-optimizer --example fold_bench
```

This is a microbenchmark of i64 constant evaluation, including result allocation,
not an end-to-end compiler or interpreter benchmark. The verification test suite
also compares generated evaluation against graph execution over every supported
scalar signature, boundary values, and deterministic random samples. These are
differential tests, not universal proofs of evaluator correctness.

Direct primitive applications inherit shared trusted algebraic facts, so an add
definition does not repeat its commutativity, associativity or identity. Explicit
claims are checked against the semantic contract; an operation cannot bind to
subtraction while claiming commutativity. Composed expressions are not assumed
to inherit the outer primitive's algebraic laws without justification.
Floating point, memory, pointer provenance, ABI state and general representation conversions
are not modeled yet. LIR target descriptions and contextual lowering still use
their own algorithms; migrating those descriptions into the shared definition
model is a subsequent step. The `split_add` semantics example demonstrates a
fixed-width representation check; it is not an enabled wide-integer backend pass.

```sh
cargo test -p veloc-spec -p veloc-mir -p veloc-optimizer -p veloc-semantics -p veloc-filetests
cargo run -q -p veloc-semantics --example split_add | z3 -in
cargo run -q -p veloc-optimizer --example semantic_check -- overflow | z3 -in
cargo run -q -p veloc-optimizer --example semantic_check -- overflow --broken | z3 -in
```

Generated Rust lives in Cargo's `OUT_DIR`, not in the source tree. MIR, optimizer
and codegen build scripts format their emitted Rust files in one rustfmt batch
per crate, using the workspace's `rustfmt.toml`. This includes offline recipes
and ISLE output; Markdown is left untouched. Formatting does not follow module
declarations into other files. The pinned toolchain includes rustfmt, and the
`RUSTFMT` environment variable can override its executable. Missing rustfmt or
invalid generated syntax fails the build rather than silently skipping formatting.
Building the compiler does not require a solver or a model service.

## Artifact selection

The `veloc-spec` binary belongs to this crate and requires the optional `cli`
feature. Build-script dependencies do not enable it or pull in clap.
It uses the same `Source::generate(&[Emit], Options)`
entry point as every production build script. Parsing, import resolution and
checking are shared; only selected artifacts are emitted. There is no runtime
rule interpreter or build-time call into arbitrary Rust code.

```sh
cargo run -p veloc-spec --features cli -- veloc/mir/defs/module.spec \
  --emit opcodes,builders --out-dir /tmp/mir-generated

cargo run -p veloc-spec --features cli -- veloc/codegen/defs/x86_64/module.spec \
  --emit target --arch x86_64 --context crate::target::x86_64::lowering::X86LoweringContext \
  --definitions veloc/codegen/defs/x86_64/instructions.spec -o /tmp/machine.rs
```

`--emit` accepts a comma-separated list or repeated options. `-o -` writes
one artifact to stdout; `--out-dir` writes standard artifact filenames.
Invalid configuration or source errors produce no output files. Output
selection does not disable definition checks. Unsupported storage/artifact
combinations are errors, not empty compatibility outputs.

- IR: `types`, `type-rules`, `opcodes`, `instructions`, `builders`,
  `validator`, `checks`, `evaluation`, `semantics`, `text-parser`, `text-printer`.
- Rust bindings: `data-types`, `interfaces` (with `--namespace`).
- Machine descriptions: `target` generates the complete integration unit;
  `selector`, `encoder`, `assembly` generate fragments for that unit's host
  scope, not standalone Rust crates.
- Transformations: `rules` and `decisions`, with explicit IR and Rust bindings.

```sh
cargo run -p veloc-spec --features cli -- veloc/codegen/defs/x86_64/legalize.spec \
  --emit decisions --definitions veloc/lir/defs/module.spec \
  --source-dialect lir --source-opcode veloc_lir::GenericOpcode \
  --field veloc_lir::InstField \
  --function decide --result Action \
  --value-interface ValueRules \
  --value-adapter crate::passes::lowering::RewriteContext::replace_values \
  --rewrite crate::passes::lowering::LegalizeAction::rewrite \
  --legal-action crate::passes::lowering::LegalizeAction::Legal -o /tmp/legalize.rs
```

Cross-IR `rules` takes `--source-definitions`, `--definitions` (destination),
`--source-dialect`, `--target-dialect`, `--source-opcode`,
`--target-opcode`, `--function`, and `--context`.
Primitive identity-rule inference is opt-in via `--infer-primitives`.
See [rule contracts and composable construction](rules.md).

The Wasm conformance runner is a separate package named `veloc-wasm-spec`.
