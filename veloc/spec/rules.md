# Spec rules

Spec compiles typed rules at build time. It uses OpSpec's lexer/parser,
diagnostics, logical operation signatures and exact type sets; it does not
maintain another instruction catalog or depend on a runtime IR.

## Value rules

Register checked definition units under explicit dialect names with
`rules::Dialects`, then compile a rule file with `rules::Program::compile`.
The rule syntax uses the same declarations and expressions as .spec files:

```text
rule negate {
    match = mir.INeg(x);
    emit = lir.Neg(x);
}
```

Nested target calls construct instructions in dependency order and allocate
typed temporaries. A result list supports multiple results; a value reference
forwards a source operand. One target call may itself return multiple results.
The checker rejects unknown operations, wrong arities, unbound values,
overlapping unconditional rules and type relationships not justified by the
source contract.

Source type variables are universal: matching an integer operation does not
permit silently restricting it to i32 or to scalar integers. Target types must
accept the entire source domain. Source IR must already satisfy its verifier;
target verifier conditions not covered by the structural type contract are
rejected rather than assumed.

The Rust backend generates direct opcode dispatch, static operand arrays,
temporary allocation and final-result binding through an explicit host
interface. It does not interpret rules at runtime. The host supplies enum
paths, values, types and construction. Consumers can invoke OpSpec-generated
builders so physical storage ordering is not duplicated. Native MIR-to-LIR
translation currently uses handwritten Rust, not this generic rule compiler.

Exact, pure primitive matches can be inferred from OpSpec semantics. These
matches go through the same checker and Rust emitter as explicit rules.
Explicit rules take precedence; ambiguous inferred mappings are errors.
Type checking is **not semantic equivalence verification**. Explicit rules
remain reviewed transformations, including trap and floating-point contracts;
there is no SMT invocation or new proof claim here.

## Expression equivalences

Anonymous `rule` groups describe semantic equalities. One checker produces a
`CheckedRule` containing typed pattern slots, guards and a replacement recipe.
Allocation-free instruction reductions and e-class queries are projections of
this same model. The checker identifies direct operand/constant reductions;
all other rules are handled by equality search.

```text
rule(root: mir::IAdd) {
    where T: ScalarInteger {
        case root<T>(mir::ISub(x, y), y) => x;
        case root<T>(x, mir::IXor(x, -1)) => -1;
        case root<T>(mir::IMul(x, y), mir::IMul(x, z))
            => mir::IMul(x, mir::IAdd(y, z));
    }
}
```

Group cases by root opcode; the group declares only the operation, not its
result type. `root<T>(...)` matches the declared root with result type `T`;
the replacement must preserve that type. The pattern name must match the
group's parameter name. Fixed result types use the same case syntax, such as
`root<Type::BOOL>(...)`.

Type variables and their constraints are declared only in `where` blocks,
whether shared by all cases or just a subset. `rule<T: ...>` and
`case<T: ...>` declarations are rejected:

```text
rule(root: mir::Select) {
    where T: ScalarInteger {
        case root<T>(condition, 1, 0) => mir::ExtendU<T>(condition);
        case root<T>(condition, mir::Wrap<T>(x), mir::Wrap<T>(y))
            if type_of(x) == type_of(y)
            => mir::Wrap<T>(mir::Select<type_of(x)>(condition, x, y));
    }
    case root<Type::BOOL>(condition, true, false) => condition;
}
```

`where` blocks may nest and declare multiple parameters separated by commas,
such as `where T: Integer, W: Integer { ... }`. Inner declarations cannot
shadow outer parameters or the root name. Parameters do not escape into
following sibling cases. Parsing flattens blocks into case declarations in
source order; they add no runtime matching state.

Simplify and SCCP inspect only the current operation, operand identities and
constant facts for expression reductions. They do not follow operand definitions
to match nested expressions. The e-class matcher enumerates alternative
expressions, tries commutative operand orders and compares repeated variables by
class identity. Matching is read-only; captured matches produce checked plans.
Simplify also applies the directed instruction rewrites described below.

Attributes use their logical positions in the operation declaration:

```text
rule(root: mir::Icmp) {
    case root<Type::BOOL>(IntCC::Eq, x, x) => true;
    where T: ScalarInteger {
        case root<Type::BOOL>(kind, mir::ISub<T>(x, y), 0)
            if kind == IntCC::Eq || kind == IntCC::Ne
            => mir::Icmp<Type::BOOL>(kind, x, y);
    }
}
```

The checker distinguishes value and attribute bindings using the operation
signature. Attribute literals are checked against the declared type. Attribute
reading and hashing use MIR's generated `InstFields`, borrowed through
`DataFlowGraph::inst_fields()`. MIR and e-graph nodes share this owned instruction
head; SSA operands remain in their respective stores. The head also determines
the opcode. Candidate construction uses generated field constructors and writes
the same head through `InstWriter::from_fields()`. E-class captures retain
concrete node witnesses for attributes; an equivalence-class representative
cannot substitute for that witness.
Construction checks type and property constraints before allocating anything.
Properties containing hidden SSA operands and context-dependent contracts are
not expression attributes.

Guards retain a typed predicate tree. Capture dependencies and conservative query
filters are derived from it; Rust expressions are emitted only at the end.
`unsigned(value)` and `signed(value)` query known integer constants;
`shift_amount(value)` uses modulo-width shift semantics; `low_mask(bits)` creates
a mask for 1–64 low bits. Missing facts and checked-arithmetic failures reject a
plan. A replacement result type may use `type_of(binding)` and is validated when
the plan is built. Matched subexpressions reused by a recipe retain their values.

Equivalence and local-fold outputs share one parsing and checking step per build.

## Directed instruction rewrites

`Emit::Rewrites` consumes rule groups through `Options::rewrites`. The caller
supplies the operation definitions, dialect and Rust namespace of the IR.
The current host uses packed instruction views and `FuncBody::edit`; operands
and constructors are resolved through the same logical storage mapping as
expression attributes. No opcode-specific transformations live in the emitter.

```text
rule(root: mir::Load) {
    case root(mir::PtrOffset(ptr, inner), outer, flags)
        => mir::Load(ptr, checked_cast(i64(inner) + i64(outer), u32)?, flags);
}
```

These rules replace one instruction in place and retain its SSA results.
Patterns may inspect nested pure, single-result definitions without deleting
them. Replacement operands can use captured values, attributes, field access,
typed helper functions and `constant_bits(value)` (the raw scalar constant bits).
Unavailable facts or a failed `?` reject the case before any IR mutation.
Repeated bindings require equality; cases are tried in source order per opcode.

Root operations can have zero or multiple fixed results. Their existing result
types are checked against the replacement's signature and property constraints.
Opcode changes require pure operations without ownership transfers. Effectful
roots keep their opcode and position; rule authors must preserve their memory,
trap and ownership semantics. Variable operands, embedded SSA attributes and
context-dependent instruction contracts require a separate host adapter.

Matching and construction planning are read-only and use no runtime rule
interpreter. A successful plan is materialized once. Identical replacements are
ignored; the caller owns worklist updates. Directed rules must converge when
reapplied; they are not automatically sent to equality saturation. In contrast,
expression equivalences retain alternatives for cost-based extraction.

### Shared expression evaluation

`evaluate_expression` accepts an opcode, operand identities, result types,
properties and an operand-fact query. `evaluate_inst` adapts an existing MIR
instruction to the same interface. Identities are distinct from constant facts:
two different values both described as `Varying` are not equal operands.

The constant lattice is `Unknown`, `Constant` or `Varying`. `Unknown` means an
iterative analysis has not supplied information yet, not an IR undef value.
Evaluation returns either an operand position or a result fact. Spec identities
and concrete evaluation run first; the shared evaluator then propagates facts,
including joining Select arms for a varying condition. Unsupported or effectful
operations produce `Varying` rather than waiting on unknown inputs.

SCCP supplies its current facts and publishes updates through its worklist.
`reduce` and `reduce_inst` are folding adapters for Simplify, e-graph construction,
rewrite plans and block specialization: their callers supply established
constants, and other operands are treated as `Varying`. They expose only operand
or constant replacements, never pending analysis facts. Evaluation does not edit
IR, activate CFG edges or schedule analysis work.

### Application and extraction

Direct reductions return an existing operand or constant without constructing
replacement instructions. They need no structural search or profitability walk.
Rules that inspect nested operations, use guards or construct instructions run
in the e-graph, including equal-cost alternatives:

```text
rule(root: mir::ISub) {
    where T: ScalarInteger {
        case root<T>(x, c) if is_const(c) => mir::IAdd<T>(x, mir::INeg<T>(c));
    }
}
```

Construction folds proposed operations and removes dead planned steps before
allocating candidates against the graph's node budget. Extraction chooses which
alternatives to materialize based on cost, sharing and legal placement. A rule
does not force an equal-cost alternative to replace the original expression.
Search limits bound rule exploration; the compiler does not prove semantic
equivalence.

The pipeline performs cheap reductions around other passes and runs structural
expression search in ExpressionPass. Moving an expression rule into equality
search does not guarantee that its result is available to earlier memory passes
or that later-created expressions will be searched again.

### Types and predicates

Untyped nested calls inherit the result type implied by their operand position.
Casts and other independent types can use explicit result types:

```text
rule(root: mir::Wrap) {
    where T: Integer, W: Integer {
        case root<T>(mir::ExtendU<W>(x)) if type_of(x) == T => x;
        case root<T>(mir::ExtendU<W>(x)) if bits(type_of(x)) < bits(T) => mir::ExtendU<T>(x);
    }
}
```

Type parameters bind at matched nodes. Each pattern slot carries its own domain
and type source, including scalar/vector shapes. Query checks and incremental
triggers use the slot's domain, rather than assuming it shares the root type.
Construction validates each operation's full type contract before allocating IR,
including cast width and shape constraints. Operations must be pure, non-trapping,
value-only and single-result. Missing target layout rejects a layout-dependent
case.

Guards support `is_const(value)`, `type_of(value)`, `bits(type)`, `pointer_bits()`,
typed comparisons, conjunction and disjunction. Value/literal equality and
inequality both require a known scalar constant. Pattern literals are masked to
the matched integer width. Variables and type parameters are local to each case;
replacements cannot refer to unmatched values or unbound types.

### Generated execution

`Emit::LocalFolds` emits e-graph guard checks, plan constructors and
allocation-free direct operand/constant reductions. It emits no SSA subgraph
matcher. Primitive identity, absorbing and idempotence laws go through the same checker. Scalar
constant results and trap checks still come from generated operation semantics.

`Emit::Equivalences` emits incremental query bytecode. It retains shared scan
prefixes, commutative bindings and reverse-path triggers. Constant conditions on
conjunctive guard paths filter triggers, including through repeated-variable
aliases; disjunctions remain conservative. A class becoming constant can enable
previously rejected matches.

The query compiler moves type, constant, equality and attribute checks ahead
of dependent scans as soon as their inputs are available. When several changed
positions request overlapping queries, the runtime compares their static scan
counts with a full root query and chooses the smaller plan. This is a work
estimate, not a cardinality model; the same search limits apply to both paths.
Cached relations enumerate each distinct pair of canonical operands and
attributes once. Concrete SSA definitions remain available to extraction.

`Capture` encodes the slots needed by the replacement, guards and result types.
Successful matches are deduplicated and applied after querying the stable graph.
Queries check generated conditions without constructing a replacement plan.
Application canonicalizes captures, checks conditions again and constructs the
replacement once. Construction plans retain each operation's result type;
there is no separate construction bytecode assuming all operations have the
root type. The plan folds constants and validates types before the graph
materializes it. E-class
representatives are used only for search candidates; executable placement and
dominance remain the extractor's responsibility.

Flat reductions also run during graph rebuilding and before candidate allocation,
independently of search fuel. A complete reduction marks its source `Folded`;
actual operand witnesses form a sparse replacement mapping for extraction. A constant-valued
class alone cannot discharge a potentially trapping operation. Ordinary DCE
subsequently removes unused dependencies.

Equality search imports supported operations into compact search storage. Imported
operations and new candidates share one representation: MIR instruction fields
and ranges in a shared input/result buffer. Search IDs are dense and independent of
MIR IDs. Referenced parameters and unsupported results are opaque typed leaves;
calls, branches and other unsupported operations stay in MIR. Types belong to search
values, and a class with a known constant is rooted at its interned literal.

The original function stays intact throughout search. Separate occurrence mappings
retain its values and instructions for dominance checks and reuse; matching and
congruence read only search storage. Extraction reads anchor operands through the
import mapping and resolves proven folds on demand. No candidate changes MIR
storage or use-lists. The rewrite `View` exposes search facts to Spec-generated
guards and construction recipes. Only extraction materializes candidates in MIR;
Simplify consumes direct reductions without a rewrite plan.

Extraction returns an owned plan containing search storage, proven folds,
instruction placements, operand rewrites and folded instructions to erase. Applying
the plan materializes selected operations and constants into MIR; dropping it leaves
the function unchanged. Search storage moves into the plan without copying operations,
and the function borrow ends before application. No temporary MIR suffix needs
truncation or cleanup.

An accepted extraction may update an existing pure operation's operands while
preserving its result identities. Its original definition must dominate the planned
use, and all selected inputs must be available at the original definition. Each
definition is updated at most once; opcode, properties and result types stay intact
so other planned occurrences can still use it as a template. Otherwise extraction
inserts a new occurrence at its planned use.

Memory rewrites and analysis-dependent range transforms remain in their owning
passes until their required effects and facts have explicit rule contracts.

## Target descriptions

Target memory metadata can name a base input and a signed byte-offset attribute:

```text
memory = { kind: Read, bytes: 8, address: { base: base, offset: offset } };
```

The generator checks these names against the operation declaration and emits
operand accessors. The scheduler compares byte ranges only when both bases
refer to the same register version. Unknown addresses retain conservative
dependencies. Indexed forms may omit `address` until their addressing mode has
an explicit model. This metadata does not authorize motion of trapping or
volatile operations.

`Source::generate` with `Emit::Target` consumes OpSpec instruction contracts and the target-selection
rules. Register definitions, scheduling metadata, assembly and encoding
expressions live in `.spec`; ABI and selection rules use the same `.spec` frontend.
Encoding expressions use the shared typed expression compiler and explicitly
declared Rust host interfaces. Byte encoding lives in `veloc-encoder`, not in
a second macro language.

The current generic core deliberately handles fixed-arity value rules.
Properties, successors, variadic signatures, structural/shape-dependent
types, guarded alternatives and multi-node source matching require additional
checked adapters. They are diagnosed rather than silently discarded.
Memory/control/call lowering and the single-root target selector
remain specialized consumers; target selection has not yet been migrated to
the new value-rule core. Sharing the contracts does not imply sharing graph
mutation, CFG construction or instruction-encoding algorithms.

Run `CARGO_INCREMENTAL=0 cargo test -p veloc-filetests --test rules` for rule-to-Rust execution
tests and the existing target-description tests.

The unified CLI selects consumers with `--emit rules`, `--emit decisions`,
or `--emit target`. See [the driver reference](README.md#artifact-selection).
Build scripts supply explicit Rust bindings through `Options`; both paths call
`Source::generate`. Definitions and imported rule modules are loaded only once.
Selection has one explicit, typed root and anonymous entry declarations.
`choose` tries cases in declaration order; the first successful case wins.
The input dialect and its OpSpec definitions are supplied by the caller through
`Target::input` (CLI: `--source-definitions` and `--source-dialect`).
Root opcodes and logical fields are checked against operation signatures.
Storage mappings are applied only when generating their concrete reads. Hidden
storage fields are not exposed to rules; a used logical field without an
invertible direct projection is diagnosed rather than guessed.

## Selection operations

Selection cases use ordinary statement syntax and explicit operations:

```text
typeset SmallInt = Type::I8 | Type::I16 | Type::I32;
select(n: lir::Copy<Type::I32>) {
    choose {
        case {
            replace(n, build(X86Mov32(n.src)));
        }
    }
}
```

Import the shared type declarations before using these patterns. Type domains
use the same named-set and union resolver as legalization signatures. Bytecode
guards query the virtual-register table directly; they do not imply a register bank
or generate one host predicate per type. Physical registers do not match these
typed virtual-value patterns. General predicate extractors remain available.
The operations are:

- `root.field`: read a source field, optionally named with `let value = root.field;`.
- `require(type_is<T>(value))`: constrain a source value's logical type.
- `require(matches(root.field, literal))`: match an integer or condition code.
- `require(fits_signed(node.field, bits))` and `fits_unsigned`: check an i64
  attribute against an immediate width in 1..=64. Definition fields are supported;
  unsigned checks reject negative attributes. These lower to shared bytecode
  range checks, without target-specific callbacks.
- `temp(Type::I32)`: declare a fresh register with an explicit concrete type.
  The argument currently resolves to one declared type constant (including template
  substitution). Its bank is determined by target operand constraints, never copied
  from another value. Dynamic host type expressions are not yet supported.
- `build(TargetInst(...))`: describe a target instruction.
- `replace(root, build(...))` or `replace(root, [instructions...])`: commit builds in order.

Bindings are candidate-local. Field reads are interned; requirements precede construction;
`replace` is mandatory and terminal. The compiler rejects duplicate names,
unbound construction inputs and unused/repeated/reordered build handles.
Build handles denote instructions, not their SSA results; explicit result
registers still appear in multi-instruction constructors.

Target `Value<...>` signatures constrain the representations of virtual operands.
The rule compiler checks domains known from type guards and typed temporaries
against these signatures. Unknown domains are not assumed valid: the explicit
target validator checks concrete virtual values using the function's register
table. Physical operands are checked against register classes and, after
allocation, ties; they do not retain a logical pointer or floating-point type.
Builders do not invoke this validation.

Generic memory operations keep pointer-typed addresses. Selected address
components instead follow the target representation contract (for example,
64-bit integers or pointers on x86-64). These sets describe neither provenance
nor the validity of an access. Likewise, a GPR value domain alone is not a proof
of a low-bit read, extension, or defined upper bits; those are instruction
semantics, not register-class or type-membership facts.

Candidates directly store their root, field checks, temporary declarations and
ordered deferred builds. There is no legacy node-bind/covers wrapper or synthetic
sequence constructor. Identical pure checks within a root's candidates are shared
by the Rust emitter; candidate-local mutation stays behind matching.

This is not yet a general typed matcher SSA IR: reusable statement functions and arbitrary result-value composition remain
future work. Definition-level templates use the existing common Spec expansion.

Operation type arguments follow the generic declaration order and may use a
finite type set. Omitting arguments leaves them unconstrained. Explicit arguments
are checked against the OpSpec bounds and generate one guard on each generic's
defining value. Equal-type checks on other values of that same generic are
combined. This relies on valid input LIR; instruction validation remains a
separate pipeline responsibility. Derived shapes are not treated as equal types.

## Definition matching

A candidate can follow a virtual SSA value to its defining instruction:

```text
select(n: lir::Load) {
    choose {
        case {
            let addr = def<lir::Add<Type::I64>>(n.base);
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86Load64Index(addr.lhs, addr.rhs, n.offset)));
        }
        // Other cases handle roots without a matching producer.
    }
}
```

Lookup/opcode failure tries the next case. Definition bindings may reference fields
of earlier definition bindings. Their opcodes and fields use the input OpSpec
contracts, just like the root. Attribute fields cannot be `def` inputs.

`def` is a read-only lookup of a virtual value's unique defining instruction;
it does not imply permission to fuse or erase it. Before committing a graph
rewrite, a separate conservative safety check currently requires pure,
nontrapping, single-result generic instructions with virtual operands and no
extra effects. Thus the current selector does not fold loads, calls or trapping
computations. The memory access stays at the original root;
its access metadata is transferred by the target selector.

Selection visits consumers before producers within each block. A producer with
other uses remains; an unused pure producer is erased when visited. Block order
is reversed too, but is not a global reverse-topological scheduling algorithm:
an already-selected producer simply fails the generic match. No unconditional
"matched means erased" rule is used.

Current x86 rules fold zero-offset StackAddr accesses (scalar loads/stores) and
64-bit Add/PtrAdd addresses into 64-bit integer/pointer indexed accesses.
Scaled indexing and memory-operation folding are not implemented.

## Legalization decisions

`rules::decisions` binds one checked OpSpec unit to an explicit dialect name
(`DecisionRust::dialect`). Node parameters use that namespace; their generics,
operand names and type relationships come from the operation declarations.
No second handwritten operand signature or implicit host variable is needed.

Legalization and instruction selection use the same anonymous `select` syntax.
A single candidate needs no `choose` wrapper:

```text
select<T: Narrow>(inst: lir::Add<T>) {
    replace(inst, build(lir::Trunc<T>(
        lir::Add<Type::I32>(
            lir::Zext<Type::I32>(inst.lhs),
            lir::Zext<Type::I32>(inst.rhs),
        ),
    )));
}
select(inst: lir::Ctpop<Type::I32>, target: &Target) {
    choose {
        case {
            require(target.supports(Instruction::POPCNT32));
            legal(inst);
        }
        case { popcount32(inst); }
    }
}
```

Node type arguments specialize the declared operation generics. Construction
expressions specify result types and infer input types from their arguments.
Alternatives such as `lir::Add<T> | lir::Sub<T>` must have identical named
value signatures. Candidates are tried in source order; the first matching
candidate returns a plan. All `require` conditions precede construction.
`let x = build(...)` names a constructed value without duplicating it; the
terminal operation is `legal(inst)`, `replace(inst, value)`,
`libcall(inst, "symbol")`, or a named rewrite call.
Candidate diagnostics identify source locations, not artificial rule names.

Shared `rewrite` declarations are callable templates, not matching rules.
Loading a template never enables a fallback implicitly. Both implementation
forms use ordinary calls such as `trailing_zeros(inst)`, checked against the
matched instruction and type domains:

```text
rewrite trailing_zeros<T: Word>(inst: lir::Cttz<T>) {
    replace {
        let low = lir::And<T>(inst.src, lir::Sub<T>(lir::Constant<T>(0), inst.src));
        lir::Ctpop<T>(lir::Sub<T>(low, lir::Constant<T>(1)));
    }
}
rewrite load_displacement<T: Scalar>(inst: lir::Load<T>)
    = rust("crate::target::x86_64::legalize::displacement");
```

Value construction is independently composable. A function receives values, not
a root instruction, and returns a value without performing replacement:

```text
fn low_bit<T: Word>(x: T) -> T {
    lir::And<T>(x, lir::Sub<T>(lir::Constant<T>(0), x))
}
fn low_mask<T: Word>(x: T) -> T {
    lir::Sub<T>(low_bit<T>(x), lir::Constant<T>(1))
}
rewrite trailing_zeros<T: Word>(inst: lir::Cttz<T>) {
    replace = lir::Ctpop<T>(low_mask<T>(inst.src));
}
```

Functions support explicit generic arguments, typed parameters/results, local
bindings, forward references and nested calls. The compiler checks all bodies,
including unused functions, and rejects recursive calls, mismatched types and
references to a caller's locals/root. Arguments are constructed once; each call
has its own local scope. Calls are inlined into a single checked construction
plan at generation time. The plan becomes a bytecode recipe; named DSL functions
are not looked up or interpreted at runtime.
Only the enclosing rewrite binds the root result. This currently covers
single-result pure values, not control-flow/effect tokens.
The syntax parser and OpSpec type checker are shared, not reimplemented for
each target.

A case can also update the matched instruction directly:

```text
replace(inst, build(inst {
    base: lir::PtrAdd<Type::PTR>(inst.base, lir::Constant<Type::I64>(inst.offset)),
    offset: 0,
}));
```

The update preserves the opcode, result identities, untouched fields, memory
metadata and physical register effects. It supports zero-result roots such as
stores. Field names and input types are checked against the root's declaration;
integer construction can read an i64 attribute at runtime. Updates currently
require a fixed-signature, non-control root without additional verifier
constraints, and modify only scalar inputs or i64 attributes. They do not
describe arbitrary effectful instruction sequences or split memory accesses.

Construction helper functions require DSL bodies and are inlined when generating
bytecode. There is no Rust helper table, helper-call opcode or builder trait.
The VM invokes the concrete rewrite context's `emit` method directly.
`DecisionRust::value` supplies the Rust value type; the generator does not depend
on the runtime IR crate. Instruction attributes use the checked OpSpec storage
codecs.

`replace` compiles checked, fixed-arity pure value expressions with local
bindings and integer attributes. Rewrite bodies compile to bytecode; Rust
rewrite callbacks are not supported.
`legal(inst)` is a built-in decision. `DecisionRust::runtime` names the
runtime module supplying the program tables and adapters. Selection only runs
read-only predicates and returns a plan; the worklist applies it under edit
tracking and revisits the affected operations.

`libcall` is an explicit terminal action for converting a value operation to a
runtime function. For example, a target with a matching runtime implementation
could select:

```text
select(inst: lir::Fdiv<Type::F64>) {
    libcall(inst, "runtime_f64_div");
}
```

The symbol is a nonempty string literal. The root must have a fixed value
signature, at least one result, concrete input/result types, and no attributes,
memory or control effects requiring an adapter. All inputs become arguments in
their original order, and all results retain their SSA identities. The root's
types define the runtime signature; the named function must implement that
signature and the operation's semantics. A case cannot combine `libcall` with
`let ... = build(...)`; value conversions must be expressed in separate rewrites.

The action compiles to an `Action::Libcall` table entry selected by the same
read-only decision bytecode. The codegen service interns the function symbol and
uses the target's System V ABI plan to construct a fully lowered call under edit
tracking. Argument transfers, result copies and call-frame instructions return
to the legalization worklist. Ordinary calls must already be ABI-lowered on
entry to legalization; an unresolved call is an error, not a request to run this
service. No target libcall fallback is enabled by declaring this action alone;
target rules explicitly choose each symbol, and the runtime/linker must provide it.

The compiler shares its priority-preserving decision graph with instruction
selection, including common-prefix sharing. Both backends share the assembler,
fixed-width branch relocation and constant-pool interning. Execution remains
separate: legalization recipes construct generic values; selection recipes
construct machine instructions and transfer edge information.

The legalization bytecode contains `CheckSignature`, `CheckType`,
`CheckSignedRange` and `CheckFeatures` for queries, with
`Jump / Accept / Reject` for control flow
and `Emit / Return / Update` for recipes. `choose`, `case`, and `let` are
structural syntax, not one-to-one VM opcodes. Checked native operations lower
to `Emit`; DSL helper functions inline into the same recipe. Identical recipes
share code. Generated byte arrays include decoded instructions and PC offsets.
`Emit` encodes the IR opcode directly in ULEB form. Definition order supplies
both explicit enum discriminants and the recipe's numeric codes; the generated
`from_code` decoder rejects invalid values without unsafe casts. There is no
program-local opcode remapping table. These numbers are build-local, not a
stable serialized ABI across compiler versions.

`CheckSignature` contains inline result and input patterns. Each pattern starts
with a tagged u32 word: an exact type code, a set/binding with an inline type-code
list, or a reference to a previous binding. Results are visited before inputs; bindings
are local to one signature match. For example, `Add<T: Word>` binds `T` at its
result and checks both inputs against it. Arity, type domains and generic type
equality are checked together. Concrete singleton domains need no variable
binding. There are no signature or type-set tables. Extra exact type predicates
use `CheckType` with an inline type code. The generated Rust evaluates the host
`TypeCodec::encode` in constant expressions and writes the codes directly into
the byte array. Fixed-width operands keep branch offsets independent of those
expressions; the spec compiler does not assign or duplicate host type IDs.

Field semantics and wire formats are declared separately. All bytecode dialects
use the same `bytecode!` macro, which accepts optional codec type parameters.
The rewrite schema is bound as `Instruction<RawWord>` in the compiler and
`Instruction<TypeCodec>` in the VM through ordinary Rust type aliases.
A `(codec C)` field uses `FieldCodec` for reading, writing and sizing;
`WordCodec` supplies the interpretation shared by scalar `Word<C>` fields and
`List<C>` fields. Lists expose borrowed `Values<C>` iterators. These primitives
live in `veloc_bytecode::codec`; inline type patterns and their codec live in
`veloc_bytecode::signature`, available to every dialect.

For example, a schema can declare `ty: (codec Word<T>)` and
`types: (codec List<T>)` under `Instruction<T: WordCodec>`. Both decode through
the supplied codec, so adding typed fields needs no dialect-specific macro or
separate decoder. Existing plain `u32`, `uleb` and list fields use the same macro.
The runtime decodes rewrite `CheckType` into an `OperandRef` and a `Type`.
Signature patterns expose exact types and borrowed type-set iterators through
the same codec. Matching compares semantic types, with no integer conversion or
temporary type array in the VM. These are compile-time bindings; the bytecode
crate does not depend on the host type crate.

Operand positions use the shared `OperandRef::Input` / `OperandRef::Result`
representation, including recipe type sources. Only the wire codec packs the
input/result tag into an integer. The matcher borrows a checked instruction type
view: all generic value operands read `VRegData.ty`, including ABI arguments
and results. ABI register locations are operand constraints rather than untyped
value operands. Physical operands in generic value instructions are rejected.
Recipes snapshot this same view before mutating the function.

Native query methods explicitly bind to VM operations, for example:

```text
type Value = rust("veloc_lir::Reg") {
    fn ty(&self) -> Type = vm("type");
}
fn supports(&self, features: sequence(u64)) -> bool = vm("features");
```

These declarations are checked but do not generate Rust forwarding methods.
Signature/type/range/feature checks execute directly in the VM. Conditions
without a supported bytecode lowering are generation errors. There is no
arbitrary Rust predicate callback. Future analysis queries should expose typed,
semantically defined VM operations; the underlying analysis remains implemented
and cached by the runtime.

Legalization generates a public static program containing an opcode-indexed
entry table alongside shared bytecode and data tables. The configured output
name is uppercased for the static symbol. The target policy holds this program
directly; the VM looks up the entry without a dispatch callback. Selection entry
functions remain separate. Policies also supply feature words. Matching stays read-only until
acceptance. The legalization matcher returns a borrowed static action entry.
One engine step checks the budget and ABI boundary, applies that entry under
edit tracking, and reports changes. The worklist only requeues affected
instructions, including a surviving root. ABI lowering uses the same edit
tracking and convergence budget but remains a separate service.
Verification stops at matching and never runs a rewrite.

Selection builds accepted instructions and the driver commits replacement and
edge transfers. Neither engine requires a separately allocated rewrite plan.
Same-typed legalization results either reuse their identity or replace uses;
one-to-many type conversion requires a coordinated mapping of uses and CFG
edges and is not implemented by the scalar recipe VM.

`require` supports native conjunctions and direct native tests, compiled to
bytecode. External query receivers must be explicitly passed in the parameter
list and their methods must declare supported VM bindings. Named instruction
fields use the same checked storage projections as instruction views.
The built-in predicate `fits_signed(inst.offset, 32)` reads the checked offset
projection and compiles directly to a signed-range bytecode check. It requires
an i64 instruction attribute and a literal bit width in 1..=64; it does not
require a type binding or Rust callback. Negation is supported. Value methods such as `inst.src.ty()` bind on the declared
register type. Rules do not declare a query adapter; matching receives a read-only function and instruction ID. The runtime
checks the input boundary and handles physical Copy operand types without a
separate query object. External target capabilities remain explicit
host parameters. A VM binding is not an SMT model; offline proofs still require
semantics for the query and conservative handling of unknown analysis facts.

Machine instructions declare `requires = ["POPCNT"]` in OpSpec. Feature names
are checked against the target catalog. Generated requirements are shared by
legality predicates, selector candidate filters (including nested emissions),
and explicit final-instruction validation. Target CPU descriptions, not
build-host CPUID, determine compilation. POPCNT is the first end-to-end
consumer; other target instructions still need their requirements annotated.

This is not yet a general graph-rewrite/proof engine: 1:N value conversion,
analysis invalidation contracts, feature-expression coverage proofs and
offline SMT rule certification are not implemented by this decision compiler.
