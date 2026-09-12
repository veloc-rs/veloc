# Operation definitions

`veloc-opgen` is the build-time definition compiler. It is independent of runtime
IR containers; `veloc-mir/build.rs` currently uses its MIR emitter. HIR is reserved
for a future structured representation. The machine-facing IR is LIR, in
`veloc-lir`; bytecode is a separate execution format.

Shared vocabulary lives in `veloc/defs/`: `types.ops`, `builtins.ops` and
`comparisons.ops`, imported by `prelude.ops`. MIR owns its packed
`formats.ops` and logical `mir.ops`; LIR owns `generic.ops` with operand-array
formats and logical operations. Each consumer has a `defs/module.ops` entry:

```text
import "../../defs/prelude.ops";
import "formats.ops";
import "mir.ops";
```

`Source::load(path)` resolves relative imports against the importing file,
deduplicates canonical files, rejects cycles and requires imports before
declarations. Its `dependencies()` includes requested and canonical paths for
Cargo rebuild tracking; `parse()` and `compile()` report physical file locations.
Files are parsed independently in one pass, including their import preambles.
The loader resolves imports from that AST and retains the original source for
diagnostics; it does not mask imports or parse the file again. Syntax cannot
cross import boundaries.
String-based `parse/compile` remain available for self-contained definitions;
they do not resolve imports or inject an implicit vocabulary.

There is one checked `Definitions` model. `storage Operands` selects machine
operand-array emission; packed storage emits MIR views and pools. Storage
projections differ, but signatures, type expressions and semantic checking are
shared. Unsupported projection capabilities fail while preparing an output plan,
before any Rust artifact is emitted.

The public stages are explicit:

```rust,ignore
let source = veloc_opgen::Source::load("defs/module.ops")?;
let plan = source.plan()?;      // resolve and check the selected output
let generated = plan.generate(); // infallible emission; reusable
```

String input has the same `plan(text)?` entry. Existing `compile` entry points
compose planning and generation. `parse` checks the definition model without
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
  lists, objects and type sets, and precedence climbing for constraints.
  A named parsing context distinguishes values, result types and constraints;
  nested syntax is bounded, while flat type sets remain n-ary.
- `syntax/mod.rs` holds the untyped syntax tree and source offsets.
- `source.rs` resolves file dependencies and maps diagnostics back to files.
- `model/` owns operations, structs, metadata, effects and ownership contracts.
  Each contract keeps its checker and related projection together.
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

- `inst/mod.rs` exports instruction handles, drafts, views and opcode metadata.
  `inst/opcode.rs` owns generated opcode/type contracts and their support;
  `inst/storage.rs` owns physical instruction storage.
- `function/mod.rs` owns functions, with `edit.rs` for structural editing and
  `layout.rs` for block order, instruction placement and CFG edges.
- `types/signature.rs` owns both signatures and calling conventions.
- `dfg` remains independent; `builder` and `validator` serve both functions and
  modules and remain top-level modules.

Use `veloc_mir::inst::{OpSpec, OpFormat, TypeError, MemoryEffect}` for instruction
metadata and `veloc_mir::function::{Layout, BlockData}` for layout types. Common
types such as `Opcode`, `Function`, `Signature` and `CallConv` remain re-exported
from the crate root.

Generated Rust artifacts follow their consumers, not the input file boundaries:

- `types.rs`: compact type encoding, constants and type helpers.
- `opcodes.rs`: opcode metadata, formats, type classes, flags and comparisons.
- `instructions.rs`: storage, drafts, views, accessors and result type inference.
- `builders.rs`: operation-specific `InstBuilder` methods.
- `type_rules.rs`: type validation dispatch and shared signature checks.
- `validation.rs`: function-level property constraints.
- `text_parser.rs` and `text_printer.rs`: their respective text codecs.

Construction and validation remain separate. Optimizer evaluation, offline
semantics and backend lowering retain separate artifacts and consumers.

### Validation and ownership contracts

`constraints` compile to direct Rust checks at the explicit validation phase.
They do not run in builders and are not interpreted at runtime. The same
expression language applies to operations and alternate storage layouts:

```text
constraints: [
    require(matches(args, params(signature(function))), "argument types differ"),
    require(returns(signature(function)) == returns(current_signature()), "answer types differ"),
    require(all(options.evl, |v| type(v) == I32), "EVL must be i32")
]
```

`signature` accepts a `FuncId`, `SigId`, or callable `Type`. `params` and
`returns` borrow type sequences; `results()` exposes the instruction's result
types. `matches` compares SSA value types with a type sequence without allocating.
`prefix(sequence, count)` and `suffix(sequence, count)` use checked slicing.
`all` supports both sequences and optional values (absence satisfies the predicate).
Invalid handles or slices produce the constraint diagnostic, not a panic.
Layout constraints describe auxiliary operands such as masks independently of
the underlying opcode. A signature-selected call's argument/result checks and a
`table(cases, default)` mapping's required default are derived automatically.

`move operand: Type` or `move args: values` marks a non-edge parameter as
transferring ownership. Unmarked inputs cannot consume owned values. Successor arguments
transfer on their own mutually exclusive edges; non-edge inputs execute once
before the branch. `ABORT` marks an abnormal exit that need not transfer remaining
owned values and requires `TERMINATOR`. The generated operand visitor feeds one
shared CFG dataflow analysis: it contains no opcode whitelist. Control lowering
interfaces describe execution, not an implicit ownership contract.
Their required `MAY_TRAP`/`TERMINATOR` traits are derived from the control
primitive; operations need not repeat them. Memory effects remain explicit.

## Comparison predicates

```text
comparison Conditions {
    domain: float,
    predicates: [
        Eq([equal]), Ne([less, greater, unordered]),
        Lt([less]), Gt([greater]),
        Le([less, equal]), Ge([greater, equal]),
    ],
}
```

Predicates accept subsets of `less`, `equal`, `greater`, and, for floats,
`unordered` (either operand is NaN). Integer ordering predicates specify
`signed` or `unsigned`, e.g. `LtS(signed, [less])`; equality predicates omit
signedness because their meaning does not depend on it. Variant names generate
lowercase mnemonics, parsing and Display, but do not determine semantics.

The compiler derives swap by exchanging less/greater and complement by taking
the outcome-set complement. Swap must be representable, as must integer
complement. Float `complement()` returns `None` if the exact IEEE complement
is absent. Float `complement_ordered()` excludes unordered outcomes and requires
a representable complement; callers must establish that neither operand is NaN.
If multiple predicates are equivalent in this restricted domain, generation
prefers the exact outcome set, then the first declared equivalent. Duplicate
full-domain meanings, mnemonic collisions and invalid outcomes are errors.
All transforms run at build time and emit direct const Rust matches. Comparison
evaluation and target lowering remain separate Rust/ISLE consumers; this finite
model does not specify floating-point exceptions or memory behavior.

## Packed encodings

`encoding` declares sequential bit fields, shared by compact types and memory
attributes. For example, `builtins.ops` defines:

```text
encoding MemFlags {
    storage: u16,
    fields: [alignment_log2(4), volatile(1)]
}
```

Ordinary encodings generate a private integer representation, zero initialization
(`empty`/`Default`), and const field accessors. One-bit fields generate
`is_volatile()` and `with_volatile(bool)`; wider fields generate
`alignment_log2()`, `with_alignment_log2(value)` and `ALIGNMENT_LOG2_MAX`.
Setters preserve neighboring fields and assert that values fit instead of
silently truncating them. Storage supports `u8` through `u128`. Widths, duplicate
fields and generated method conflicts are checked before code generation.
Unlike flag sets, packed structs do not expose `union` or `contains`.

Rust implements semantic wrappers such as `MemFlags::with_alignment`: validating
power-of-two alignment, converting to log2 and conservatively clamping to the
declared field limit. `Type` retains its specialized Rust projection and scalar
code table while sharing the physical layout parser and checks.

## Type encoding

```text
encoding Type {
    storage: u16,
    fields: [scalar(4), lanes_log2(4), scalable(1)],
    codes: [I8(1), I16(2), I32(3), I64(4), F32(5), F64(6), BOOL(7), PTR(8)]
}
```

Fields are packed in declaration order, from low to high bits. Unused high bits
of this compact payload are reserved. `Type` itself is a tagged u64: structural
callable types use a disjoint tag plus a module-interned signature, and do not
have a u16 raw encoding. This generates the storage declaration, layout documentation,
masks, shifts, used-bit mask and lane-exponent limit. Scalar code validation uses
the same field width; it does not independently assume codes fit in four bits.
Type construction and decoding consume these generated values, including a
nonzero scalar offset and narrower lane-count fields. Named vectors are checked
against this layout before generation and use the checked runtime constructor.
Field accessors (`element_code`, `lanes_log2`, `is_scalable`, `element_type`) and
raw encoding access are generated with the layout. Raw decoding uses those
accessors rather than repeating the unpacking expressions. Vector construction,
legality checks and target-independent size calculations remain shared Rust code.

The current MIR adapter requires `u16` storage, scalar codes fitting `u8`, lane
counts fitting `u16`, and a one-bit scalable flag. Unknown, missing, duplicate,
zero-width and overflowing fields are definition errors. Changing the layout
changes raw encodings; the checked-in layout preserves the existing representation.
Vector legality and type semantics remain Rust algorithms, separate from this
MIR-specific physical representation. HIR and LIR need not share this layout.

Callable operations additionally declare a typed `control` interface:
`owned(function, captures, cleanup)`, `local(function, captures)`,
`shared(function, captures)`, `tail_call(function, args)`,
`call(callee, args)`, `tail_call_value(callee, args)` or `drop(callee)`.
These are trusted semantic primitives, like the bitvector
primitives, not arbitrary Rust snippets. The generator checks their operand,
result, storage and effect contracts and emits `Opcode::has_control()` for
classification. Backends lower instruction views directly; there is no runtime
control wrapper or automatic implementation of new operations.
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

The `codes` table assigns each primitive scalar its stable backend code separately
from its type expression. Codes must be nonzero, unique and fit the scalar field.
Each entry must reference a declared scalar under its canonical MIR adapter name;
unknown types, vectors, duplicate entries and missing primitive codes are errors.
Aliases reuse the canonical type's encoding instead of acquiring a second code.

## Types and type sets

```text
type I32 = int(32);
type BOOL = bool();
type PTR = ptr();
type I32X4 = vector(I32, 4);
type SV4 = vector(I32, scalable(4));
type WORD = I32;
type WORDS = vector(WORD, 4);
class WideInteger { members: [I32, I64] }
class WideVectors { members: [vectors(WideInteger)] }
class ChosenShapes { members: [I32X4, SV4] }
```

`type Name = expression;` is the only type declaration syntax. Constructors are
interpreted by the semantic checker, not special cases in the grammar. Supported
constructors are `int(bits)`, `float(bits)`, `bool()`, `ptr()` and
`vector(element, lanes)`; `scalable(lanes)` is a vector shape expression. Type
names must be uppercase Rust constant names. Expressions support aliases, forward
references and nested construction, such as `vector(int(32), 4)`. Cyclic aliases,
unknown constructors, wrong arity and vector-of-vector types are rejected.
`vector(...)` constructs one type; `vectors(set)` constructs a type set.

Type expressions generate compact-code decoding, text names, element widths
and exact `Type` constants. IR storage and interpreter metadata use a single
`Type` representation. Codes remain stable; this
change does not widen the 16-bit `Type` representation. The MIR codec adapter
currently supports integer lanes of 8/16/32/64 bits, float lanes of 32/64 bits,
`Bool` (one logical bit, one storage byte), and target-sized `Ptr`. It checks
canonical names against kind/width so changing a declaration cannot silently
contradict typed codec/target adapters. New primitive kinds or widths still need
those adapters to be extended.

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

Classes are exact type sets: members can be scalar/vector type constants, other
classes, or `vectors(S)`, which includes every legal fixed and scalable vector
shape over the non-pointer scalar set S. Named vector constants are conveniences,
not an exhaustive enumeration of legal vectors. Numeric lane counts are fixed;
`scalable(lanes)` explicitly selects a scalable shape.
Forward references work; cycles, unknown names and empty classes are errors.
Passing pointers or vector types to `vectors()` is an error, not silent filtering.

Sets preserve both scalar identity and vector shape. `{I32, I64}` does not include
I8/I16 or any vectors; `{I32X4}` does not include I32X8 or scalable I32 vectors.
The build-time model maps scalar codes to shape bitsets; generated runtime checks
use integer masks and matches, not heap-allocated sets. These exact sets also drive
definition-time shape constraints, bitvector semantic compatibility and floating
text checks. There is no separate seven-domain vocabulary or name allowlist.

Type-set expressions also work directly in operation signatures; a named class
is just a reusable alias, not a required declaration for every combination:

```text
op IAnd<T: Integer | BOOL | vectors(BOOL)>(lhs: T, rhs: T) -> T { meta: OpInfo {}, ... }
op Gather<T: Integer & Vector>(ptr: PTR, index: T) -> shape(T, Vector) { meta: OpInfo {}, ... }
op Convert<T: I32 | I64>(arg: T) -> shape(T, F32 | F64) { meta: OpInfo {}, ... }
```

`|` means union and `&` means intersection; `&` binds more tightly. Parentheses
group expressions, for example `(I32 | F32) & Scalar`. Both operators also work
inside `vectors(...)` and class member lists. Empty intermediate sets are allowed,
but an empty final class or signature constraint is an error. Unknown names and
invalid vector inputs are checked even in branches whose intersection is empty.

`T: I32 | I64` selects one concrete type for `T`; all occurrences of `T` must match.
By contrast, `lhs: I32 | I64, rhs: I32 | I64` allows the operands to independently
select their types. Set expressions contain concrete types and class aliases, not
type variables; dependent constraints still use `element(T)`, `vector(T)` and
`shape(T, set)`.

Generation evaluates and interns equal sets, including anonymous expressions.
Named aliases and inline constraints share the same compact runtime membership
checks; runtime code neither evaluates expressions nor constructs sets. Builder
inference, semantic checks and text codecs inspect resolved sets, not class names.

## Type predicates

```text
predicate is_integer = Integer;
predicate is_scalar = Scalar | PTR;
predicate is_predicate = vectors(BOOL);
predicate is_wide = (I32 | I64) & Scalar;
```

Predicates generate public `const fn` methods on `Type`. They use the same exact
set-expression compiler and membership projection as operation constraints, but
emit direct checks instead of calling another predicate or a runtime set object.
All invalid encodings return false, including reserved bits and illegal shapes.
In particular, `Type::is_scalar()` includes pointers; the `Scalar` class does not.

Predicate names must be snake_case starting with `is_`. `is_valid`, `is_scalable`
and `is_fixed` are reserved validity/physical-shape APIs, not set aliases.
Predicates may forward-reference classes and exact types, but are not themselves
type-set names. Empty sets, unknown references and duplicate names are definition
errors. Defining a predicate does not change type construction or layout legality.

## Traits and effects

```text
flags OpTraits {
    storage: u16,
    members: [TERMINATOR(0), COMMUTATIVE(1)],
    separator: ", "
}
flags MemoryEffects {
    storage: u8,
    members: [READ(0), WRITE(1), ALLOCATE(2), FREE(3)],
    separator: ", "
}
```

`flags` declares a named set with unsigned storage (`u8` through `u128`), explicit
member bit positions and a display separator. It generates the private-storage
struct, member constants, `NONE`, `ALL`, `empty`, `is_empty`, `union`, `contains`,
`intersects` and `Display`. Display follows declaration order, lowercases member
names and changes underscores to hyphens; the empty set prints `none`.
Bit positions preserve representation independently of declaration/display order.
Set names are not restricted to `OpTraits`. The old standalone `trait`
and `region` declarations are no longer supported. Memory effects do not use
resource flag sets; `MemoryEffects` is a set of behaviors, not resources.

Flag sets and `encoding Type` use the same checked storage and bit-layout code
for widths, masks, occupied bits and overlap detection. Flags are one-bit fields
at explicit positions; Type packs multi-bit fields in declaration order. Their
meaning and generated APIs remain separate: Type is not a set, and its raw APIs
still require `u16`. Its scalar-code and vector-shape constraints remain in the
Type adapter.

### Typed operation metadata

Structs are shared by instruction properties and build-time metadata. Enums
and flags are ordinary field types, not special operation keywords:

```text
enum MemoryEffect { variants: [Known(MemoryEffects), Unknown] }
struct OpInfo { traits: OpTraits, memory: MemoryEffect }

op Example() -> () {
    meta: OpInfo { traits: [MAY_TRAP], memory: Known([READ, WRITE]) },
    mnemonic: "example", storage: Example {}
}
```

The generic value checker resolves each field against its declared type. It
checks enum variants and their payloads, flags membership and duplicates,
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

The operation-contract adapter interprets fields *typed* `OpTraits` and
`MemoryEffect`, regardless of field names; more than one field of either type
is ambiguous and rejected. Other fields are just typed data. Semantic laws,
control contracts populate inferred facts before final struct checking. Pure
interface expressions may supply constant metadata fields explicitly. If a memory
field exists, an unmodeled operation must supply it. A struct without a memory
contract is conservatively unknown.
Missing operation traits mean no additional declared traits, not inferred purity.

The former top-level `traits` / `memory` entries and named `effect`
declarations are rejected. `Known([])` means no memory effects; `Unknown`
is a separate enum variant, not a flag or every known member combined.
The `MemoryEffect` enum is generated like any other enum. Trusted Rust queries
still implement memory interference, deletion and possible behaviors.
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

op IAdd<T: Integer>(lhs: T, rhs: T) -> T {
    meta: OpInfo {},
    mnemonic: "iadd",
    storage: Binary { args: [lhs, rhs] },
    semantics: bv.add(lhs, rhs)
}

op ExtendU<T: Integer | BOOL | vectors(BOOL)>(arg: T) -> (result: shape(T, Integer)) {
    meta: OpInfo { memory: Known([]) },
    mnemonic: "extendu",
    storage: Unary { arg: arg },
    where: [wider(arg, result)],
    }

op Load(ptr: PTR, @offset: u32, @flags: MemFlags) -> Any {
    meta: OpInfo { traits: [MAY_TRAP], memory: field(MemoryAccess, effects) },
    mnemonic: "load",
    storage: Load { ptr: ptr, offset: offset, flags: flags },
    text: "{.flags} {ptr}, offset={offset}",
    implements: [ReadAccess(ptr, i64(offset), result_type(0), flags)],
    }
```

### Typed query interfaces

An `interface` declares query data, not instruction storage. Pure `fn` definitions
assemble that data from explicitly bound logical parameters. These are ordinary
library names: the generator has no `read/write` parser, pointer-field convention,
implicit flags lookup, or special memory-access result rule.

```text
interface MemoryAccess {
    ptr: Value(PTR),
    offset: i64,
    ty: Type,
    stored: optional(Value),
    flags: MemFlags,
    effects: MemoryEffect,
}

fn ReadAccess(ptr: Value(PTR), offset: i64, ty: Type, flags: MemFlags) -> MemoryAccess {
    value: MemoryAccess {
        ptr: ptr, offset: offset, ty: ty, stored: none, flags: flags,
        effects: Known([READ]),
    },
}
```

`Value(PTR)` is an SSA-reference refinement, checked against the operation
signature at definition time and erased to `Value` in Rust. `Value` accepts any
SSA type; `Type` denotes an IR type. Interface fields can use declared structs,
enums and flags, `optional(T)`, and `array(T, N)` (up to 255 elements). Arrays
generate inline Rust arrays, not `Vec`. All fields are required. The data
declaration keyword is `struct`; the former `record` spelling is rejected.

`implements: [ReadAccess(...)]` expands a checked helper. Direct interface
constructors are also supported. Each operation may implement multiple different
interfaces, but may not implement the same one twice. A helper may call another
helper, including a forward declaration. Every body is checked even when unused;
recursive functions and recursive inline interface types are rejected. Function
parameters are immutable, nongeneric values, with one declared return type.

Projection expressions support parameter references, typed struct/enum
constructors, flags, option constructors, fixed arrays, `field(value, name)`,
`type(value)`, `result_type(index)` and explicit lossless integer conversions.
Result indices are checked against fixed operation signatures; there is no
single-result restriction. Narrowing conversions are rejected. This is a
restricted pure expression language, not arbitrary Rust or a runtime interpreter.

Metadata can explicitly refer to a bound interface, for example
`memory: field(MemoryAccess, effects)`. The selected expression must reduce to
constant declaration data. Operand- or result-dependent metadata is rejected.
The READ/WRITE mapping lives in the helper definitions, not in the generator.
Missing interfaces never imply purity. These declarations are trusted contracts,
not proofs that the declared effects agree with executable semantics.

Generation emits an interface struct and an `InstructionQuery` implementation
with direct opcode dispatch. Consumers call
`view.query::<MemoryAccess>(dfg, results)`; unsupported opcodes or unavailable
result data return `None`. The queries do not revalidate MIR types and add no
fields to stored instructions. MIR's `memory::Access` re-exports the generated
query data, while bounds/provenance analysis remains ordinary Rust.

This initial interface emitter targets packed MIR. Operand-array output rejects
operation interface bindings at plan time. Control/ownership contracts, the
constraint expression checker, and storage's existing flags classification have
not yet been migrated to this mechanism. Multiple/masked accesses retain
conservative effects until their runtime consumers support appropriate queries.

Ownership transfer is attached to the logical parameter:

```text
op Call(@func_id: FuncId, move args: values) -> signature {
    meta: OpInfo { traits: [MAY_TRAP], memory: Unknown },
    mnemonic: "call",
    storage: Call { func_id: func_id, args: args },
    signature: function(func_id),
    text: "{func_id}({args}) : {function(func_id)}",
     }
```

`move` consumes owned values; ordinary scalar, pointer and reusable callable
values remain duplicable. Properties cannot move. Successor arguments transfer
on the selected edge, without a separate annotation. There is no `moves` list.

SSA operands have names and types in the operation signature. Result names are
optional: a single result is `-> T`, multiple results are `-> (T, BOOL)`, and
zero results are `-> ()`. Parenthesized results may be named when a constraint
needs to reference them, as in `-> (result: shape(T, Integer))`. Otherwise the
MIR definitions omit result names, including overflow operations (`-> (T, BOOL)`).
Names do not affect the
generated representation; anonymous results have no implicit names or aliases.
Generic variables such as `T` are scoped to that operation; their first direct
occurrence binds the type. Derived types use `element(T)`, `vector(T)` or
`shape(T, Integer)`. Relations refer to operand/result names, not numeric slots.
There are no separate `types` structs or references to named type schemes.

An `@` parameter is a property, not an SSA use: `@offset: u32` and
`@flags: MemFlags` are stored values, while `ptr: PTR` is an SSA value with a
pointer type. Variable-length SSA groups use `args: values`; one successor uses
`dest: successor`, and a successor group uses `cases: successors`. Empty input
and result lists are `()`; signature-selected results use `-> signature`.
For example, an indirect call declares a statically checked `ptr: PTR`, a
variable-length `args: values` group and `signature: sig_id`. Direct calls use
`signature: function(func_id)` to identify the callee's signature. The source of
dynamic result types is explicit, not inferred from the opcode's name.

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
LIR may override view/accessor names. LIR operand counts follow required and
trailing optional roles, rather than a second explicit list of lengths. Diagnostics include source
line and column; the build script maps combined input locations back to the source
file.

## Generated consumers

The same definitions generate `Opcode`, `OpFormat`, `InstDraft`, type
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

`InstDraft::result_types` is the dynamic construction entry point used by
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
patterns, type sets and relations share handlers, using structural equality of
the checked definition model rather than serialized runtime descriptors.
Exact pattern slots are retained so sharing preserves error diagnostics.

The definition compiler resolves type variables to concrete operand/result
positions. Generated handlers check arity before indexing and retain the
declaration's diagnostic order, without allocating bindings or interpreting
patterns and relation slots. The number of variables is not limited to four.
Result construction knows statically whether results are fixed, require explicit
types, or come from a function signature. It performs no type-class, operand
equality or width-relation checks. Those checks run only during validation,
including for result types computed by generated builders.

Type schemes, patterns and relations exist only in the definition compiler.
MIR contains no runtime `TypeScheme`, `TypeList`, `TypePattern`, relation table,
or per-opcode type descriptor. Generated checks return `TypeError` with static
diagnostic strings and relevant operand/result positions. Interned `TypeClass`
membership checks remain executable helpers, not a second type-rule interpreter.

Structural `constraints` are typed, pure expressions, compiled directly into
Rust checks. For example:

```text
constraints: [
    require(imm.scale != 0, "scale must be non-zero"),
    len(mask) == lanes(type(lhs)),
    all(mask, |i| i < 2 * lanes(type(lhs)))
]
```

Expressions reference logical parameters and struct fields, not physical pool
IDs or layout names. The existing storage projection resolves those references.
The language provides Boolean logic (`!`, `&&`, `||`), comparisons, checked
integer arithmetic (`+`, `-`, `*`), comparison-enum literals such as `IntCC.Eq`,
and lexical `all(sequence, |element| predicate)` over finite byte/value lists.
It has no arbitrary Rust callbacks, user recursion or unbounded loops.
Numbers are checked signed 128-bit integers, not wrapping instruction values;
property integers are widened without truncation. Constant overflow is a
definition error; dynamic overflow is a validation error.

Queries are `type(value)`, `result_type(constant_index)`, `len(sequence)`,
`lanes(type)`, `min_bytes(type)`, `is_ptr(type)`, `is_scalar(type)`,
`is_vector(type)` and `is_fixed(type)` (a fixed-width vector).
Lane counts and byte sizes are minima for scalable types. A target-dependent
byte size is an evaluation error. Result indices must refer to declared fixed
results. Enum literals are checked against the comparison definitions; struct
fields against the struct definitions. Unsupported property kinds and optional
fields are rejected instead of guessed or silently coerced.

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

These predicates specify IR legality, not instruction execution or traps.
For example, a division instruction with a zero divisor can be valid IR with
defined trapping behavior; that belongs in executable semantics, not here.

The synthetic benchmark separates contract checks, module construction,
validation and constant folding:

```sh
cargo run --release -q -p veloc-optimizer --example type_schemes -- all 3
cargo test -p veloc-mir -p veloc-opgen
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
op Store(ptr: PTR, value: Any, @offset: u32, @flags: MemFlags) -> () {
    meta: OpInfo { traits: [MAY_TRAP], memory: Known([WRITE]) },
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
primitive applications; composed negation retains the existing `G_NEG` lowering.
The definition compiler binds concrete input/result sorts and emits specialized
scalar evaluators. The optional offline `Program` API binds signatures to the
graph used for reference execution and SMT export. Each recipe describes a scalar
operation or a per-lane operation, not an entire vector, memory or machine-state
model. Absent expressions mean **unmodeled**,
not a claim that an operation is pure or verified.

Executable semantic expressions infer `MemoryEffect::Known([])`. A complete `access` contract
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
`kind` is an `IntCC` property whose signedness/outcomes come from comparisons.ops,
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
types, respecting width relations and requiring shared lane shapes. It does not
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

Operand ranges use power-of-two size classes and are recycled as units. Range
growth/replacement rebuilds links for that instruction, never the whole function.
There is no separate UseIndex, per-instruction UseId list, storage-path locator,
mirrored operand Value, or generation table.

Definitions generate named layout constructors on `InstDraft`, borrowed
`InstructionView` variants, and private SSA-free `InstFields`. A draft owns
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
edges and the final default edge. Record properties are stored inline; only
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
instead of repeatedly scanning the whole function. It preserves Value IDs during
multi-result constant replacement. Dead-code removal erases closed sets together
so dead internal references do not obstruct deletion. `check_uses` independently
reconstructs occurrences for structural tests; it is not run on every edit.

`AnalysisManager` borrows one function exclusively. Reading analyses is cached;
requesting mutable function access clears derived analyses. It cannot switch to a
different function. MIR has no revision counter or use-def synchronization
protocol. Fixed-point tests compare complete function state across another pass. LIR retains its separate non-SSA register analysis.

This establishes the ordinary mutable MIR path. It does not add an e-graph, an AI
search driver, solver invocation, general rollback or fine-grained incremental
liveness. Those mechanisms can use the edit boundary without becoming IR storage
requirements. Runtime and memory improvements require measurement.

Codegen joins checked direct MIR primitive applications with the reviewed LIR
semantics in `lir/defs/generic.ops` at build time. Both definition modules use
the shared `parse/compile` API and checked operation model; operand-array storage
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
cargo test -p veloc-opgen -p veloc-mir -p veloc-optimizer -p veloc-semantics
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
