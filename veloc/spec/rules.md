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

`Emit::Equivalences` compiles opcode groups into the e-graph's query and apply
bytecode. The root instruction is explicit; each case describes its operands:

```text
rule<T: ScalarInteger>(root: mir::ISub<T>) {
    case (x, x) => 0;
    case (x, 0) => x;
    case (mir::IAdd(x, y), x) => y;
    case (x, mir::ISub(x, y)) => y;
}
```

Pattern variables bind independently in each case. Repeated names require the
same equivalence class; replacements and guards may only reference variables
bound by that case. The root parameter names the instruction, not an operand
variable. Operand arity and the common scalar type domain are checked against
OpSpec. Guards use `case (x, y) if y == 0 => x;`; the current guard language
supports equality or inequality between a bound value and an integer literal.

The compiler separates two execution contracts from the checked pattern shape:

- A root-only pattern containing variables and constants, whose replacement is
  an operand or literal, becomes a local fold. Repeated variables and constant
  guards use current equality/constant facts without enumerating child nodes.
- Nested patterns and replacements that construct operations become query/apply
  bytecode. Every matching case contributes an equality, subject to the search
  budget; these cases are not ordered alternatives.

Local folds run during graph import, before allocating a constructed expression,
and when rebuilding an expression whose operands changed classes. Primitive
identity, absorbing and idempotence laws use the same checked rule model.
The generated semantic evaluator supplies concrete constant results, including
multi-result operations and trap checks. Neither path creates new operations.
Local reductions therefore finish independently of exploratory rule fuel.

Local patterns are checked against each operand's declared type, not a single
type shared by every input. For example, `Select<T: Any>` accepts the cases
`(true, x, y) => x`, `(false, x, y) => y`, and `(condition, x, x) => x`:
the condition is boolean while the branches and result share `T`. Only literal
patterns and guards query constant contents, so unknown pointer, float and vector
branches can be returned directly. No complete semantic recipe is required for
an authored, non-trapping pure operation; its equality remains author-reviewed.
Heterogeneous patterns currently belong to local folding; the query/action VM
still requires same-type expressions and a domain of named exact types.

A folded expression leaves matching and memo indexes. No historical key is
retained to prevent reconstruction: construction runs the same local folds.
MIR definitions remain stable during searching. Once search ends, original
operand witnesses commit established folds to executable uses before extraction,
even when extraction has no budget. Ranking and pricing read the simplified MIR;
candidate templates remain immutable until emission finishes. Matching indexes
are no longer queried after this boundary.
Equivalence-class representatives are not executable replacement values.

A complete replacement marks its source operation `Folded`; this state is the
single authority for removal, not a separate deletion queue. After extraction,
placed `Folded` operations are collected into a temporary batch, candidate
references are released, and the sources are erased in the same commit. A constant-valued equivalence class alone
does not grant that permission: potentially trapping operations still require
successful evaluation with their actual inputs. Ordinary DCE subsequently cleans
up unused dependencies; it does not need equality-graph facts or exceptions.

Rules must be written against these local normal forms: removing an identity
can remove a syntactic match for another rule. This is an intentional search
policy, not a proof that arbitrary rule reachability is preserved. Type checking
does not prove an authored equality. General equalities still retain alternatives
for cost-based extraction.

Multiple groups and ordinary template expansions may contribute rules for the
same root opcode. The compiler merges them and shares matching prefixes before
emitting bytecode; source grouping does not define separate runtime searches.

## Target descriptions

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
