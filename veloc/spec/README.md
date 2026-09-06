# Operation definitions

`veloc-opgen` is the build-time definition compiler. It is independent of runtime
IR containers; `veloc-mir/build.rs` currently uses its MIR emitter. HIR is reserved
for a future structured representation. The machine-facing IR is LIR, in
`veloc-lir`; bytecode is a separate execution format.

The MIR definitions live in `veloc/mir/defs/`:

- `types.ops`: compact type encoding, scalar definitions, named vectors and type sets.
- `builtins.ops`: trait bits, memory-region bits and named memory effects.
- `comparisons.ops`: integer and floating-point condition-code semantics.
- `formats.ops`: compact storage layouts, structured property records and
  alternate-layout projections.
- `mir.ops`: logical operation signatures, storage mappings, effects, constraints
  semantic expressions and bidirectional text projections.

Callers supply these files as one definition unit; the compiler does not inject
an implicit MIR vocabulary. `build.rs` tracks each file and maps diagnostics back
to its original location.

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

## Type encoding

```text
encoding Type {
    storage: u16,
    fields: [scalar(4), lanes_log2(4), scalable(1)],
    codes: [I8(1), I16(2), I32(3), I64(4), F32(5), F64(6), BOOL(7), PTR(8)]
}
```

Fields are packed in declaration order, from low to high bits. Unused high bits
are reserved. This generates the `Type` storage declaration, layout documentation,
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
op IAnd<T: Integer | BOOL | vectors(BOOL)>(lhs: T, rhs: T) -> T { ... }
op Gather<T: Integer & Vector>(ptr: PTR, index: T) -> shape(T, Vector) { ... }
op Convert<T: I32 | I64>(arg: T) -> shape(T, F32 | F64) { ... }
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
trait COMMUTATIVE { bit: 1 }
region GLOBAL { bit: 2 }
effect GLOBAL_READ { reads: [GLOBAL], writes: [] }
```

Trait/region declarations generate constants and display metadata. Explicit bit
positions preserve the representation independently of declaration/display order.
`MemoryRegions::NONE` and `ALL` are derived; effects refer to region sets, with
`[ALL]` denoting all declared regions. The reserved effect `NONE` must be empty;
`UNKNOWN` must read and write every region. Purity checks inspect the sets, not
the effect's name. Duplicate bits and unknown references fail before emission.

This is a vocabulary, not an arbitrary executable extension language. Generic
type inference, memory-conflict algorithms, primitive bitvector meanings and
reviewed algebraic laws remain Rust. Declaring a new trait does not invent an
optimization or prove a law. Trusted laws use `BvConst` directly, rather than
copying their constant names into a second definition whitelist.

## Operation signatures

```text
format Binary {
    fields: [opcode(Opcode), args(values(2))],
    opcode: dynamic(opcode)
}

op IAdd<T: Integer>(lhs: T, rhs: T) -> T {
    mnemonic: "iadd",
    storage: Binary { args: [lhs, rhs] },
    semantics: bv.add(lhs, rhs)
}

op ExtendU<T: Integer | BOOL | vectors(BOOL)>(arg: T) -> (result: shape(T, Integer)) {
    mnemonic: "extendu",
    storage: Unary { arg: arg },
    where: [wider(arg, result)],
    memory: NONE
}

op Load(ptr: PTR, @offset: u32, @flags: MemFlags) -> Any {
    mnemonic: "load",
    storage: Load { ptr: ptr, offset: offset, flags: flags },
    text: Text { args: [ptr], named: [default(offset, 0)], flags: flags },
    traits: [MAY_TRAP], memory: HEAP_READ
}
```

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
operation-specific parsing/printing and ordinary builders.

### Compiled type contracts

Type signatures drive two independent paths: result construction and validation.
Generated builders compute result types directly from logical arguments and pass
fixed-size arrays to `InstBuilder::insert`. Neither operation validates the type
contract. The insertion API also accepts caller-supplied result types for generic
transformations or deliberately incomplete IR.

`InstructionData::result_types` is the dynamic construction entry point used by
the text parser and contextual builders. Its generated opcode branches return
the final types directly, using operand types, explicit types or the referenced
signature. There is no runtime result-strategy enum. Missing explicit types,
unknown signatures and operands whose types cannot determine the result are
construction errors, not full contract validation.

`Opcode::validate_types` dispatches to shared generated checks. Full module
validation is an explicit pipeline/caller decision; builders and the parser do
not invoke it implicitly. The parser still checks syntax, symbol resolution and
conflicting result annotations, but does not maintain a deferred type-validator
queue. Forward references that are not needed to determine results can be
constructed and subsequently validated normally.

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

Expressions reference logical parameters and record fields, not physical pool
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
results. Enum literals are checked against the comparison definitions; record
fields against the record definitions. Unsupported property kinds and optional
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
`gather(ptr, index, mem: VectorMemOptions, ty: Type)`. Packing inserts properties
through `PoolKey` before pushing the completed instruction. Contextual helpers
remain for variadic groups, CFG destinations and signature-selected results.
They provide higher-level slices and blocks while storage retains compact IDs.
All insertion paths check the format and resolve/check the type scheme once;
inferred results are not redundantly revalidated before insertion.

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
and `get`; hand-written consumers use that same API rather than parallel DFG
getters/interners. Byte constants are stored as shared `Arc<[u8]>` buffers: the
pool and deduplication index share one payload, while reads borrow `[u8]`. Value
lists and jump-table construction retain their shared, non-interning algorithms.
Both contracts use static dispatch and do not add a codec registry or trait-object
dispatch. Rust checks the implementations and generated calls; round-trip tests
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

Executable semantic expressions imply `memory: NONE`; unmodeled operations must
declare their memory effect explicitly. Empty `traits` lists may be omitted.

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

Codegen joins checked direct MIR primitive applications with the reviewed LIR
bindings in `lir/defs/generic.rs` at build time. The same declaration supplies
the generic opcode enum and build-only bindings. The result is a direct
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
