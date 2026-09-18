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
Selection currently supports one root; unsupported multi-root or non-root
`covers` declarations are diagnosed. Static `cost` orders candidates; ties
retain declaration order.

## Legalization decisions

`rules::decisions` binds one checked OpSpec unit to an explicit dialect name
(`DecisionRust::dialect`). Node parameters use that namespace; their generics,
operand names and type relationships come from the operation declarations.
No second handwritten operand signature or implicit host variable is needed.

```text
rule widen_add<T: Narrow>(inst: lir::Add<T>) {
    replace = lir::Trunc<T>(
        lir::Add<Type::I32>(
            lir::Zext<Type::I32>(inst.lhs),
            lir::Zext<Type::I32>(inst.rhs),
        ),
    );
}
rule ctpop32(inst: lir::Ctpop<Type::I32>, target: &Target) {
    action = match target.supports(Instruction::POPCNT32) {
        true => legal,
        _ => expand(popcount32, inst),
    };
}
```

Node type arguments specialize the declared operation generics. Construction
expressions specify result types and infer input types from their arguments.
Alternatives such as `lir::Add<T> | lir::Sub<T>` must have identical named
value signatures. Rules are tried in source order; the first matching rule
returns a plan. Related native candidates and their fallback can instead live
in one action-level `match`, removing their dependence on inter-rule ordering.
Matches evaluate the scrutinee once, test declared constants/literals and guards
in order, and require a final unguarded `_` fallback. Nested matches are supported;
binding/destructuring patterns and exhaustiveness inference are not yet supported.

Shared `rewrite` declarations are callable templates, not matching rules.
Loading a template never enables a fallback implicitly. Both implementation
forms use `expand(name, inst)`, which checks the instruction and type domains:

```text
rewrite trailing_zeros<T: Word>(inst: lir::Cttz<T>) {
    replace {
        let low = lir::And<T>(inst.src, lir::Sub<T>(lir::Constant<T>(0), inst.src));
        lir::Ctpop<T>(lir::Sub<T>(low, lir::Constant<T>(1)));
    }
}
rewrite load_displacement<T: Scalar>(inst: lir::OffsetLoad<T>)
    = rust("crate::target::x86_64::lowering::legalize::displacement");
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
plan at generation time, with no runtime DSL interpreter or function lookup.
Only the enclosing rewrite binds the root result. This currently covers
single-result pure values, not control-flow/effect tokens. Those still use
explicit whole-root host rewrites.
The syntax parser and OpSpec type checker are shared, not reimplemented for
each target.

A construction fragment may have an explicit Rust implementation:

```text
fn helper<T: Word>(x: T) -> T = rust("crate::helper");
```

Its Rust signature is generic over the generated builder contract:

```rust,ignore
fn helper<C: ValueRewrite>(ctx: &mut C, ty: Type, x: Reg) -> Reg
```

Type arguments precede value arguments, in declaration order. Generated wrappers
check even unused bindings. The host contract is an ordinary Rust-bound type
declaration in `legalize.spec`; `rewrite_interface` names its construction method.
Inputs, types and an optional destination value are explicit parameters of the
construction function; it returns the replacement value. The outer adapter reads
the matched instruction, binds the returned value and erases the root. The
generated helper is bounded by that contract and cannot access arbitrary graph
editing APIs. Instruction attributes use the checked OpSpec storage codecs.

`replace` compiles checked, fixed-arity pure value expressions with local
bindings and integer attributes. Rust bindings support memory and other complex
graph rewrites; Rust checks their callback signatures even when unused.
`legal` is a built-in decision. `DecisionRust` supplies the runtime bindings for
legal decisions, generated value rewrites, and Rust callbacks. Decision selection
does not mutate the function; the worklist applies the selected plan and revisits
the resulting operations.
`when` can combine declared predicates with boolean operators. A rule can
only call a host explicitly passed in its parameter list. Instruction-local
queries and immutable target capabilities are separate contexts; graph
analysis is not implicitly available. Rust predicates are opaque to offline
proof: declaring an interface does not provide an SMT model.

Machine instructions declare `requires = ["POPCNT"]` in OpSpec. Feature names
are checked against the target catalog. Generated requirements are shared by
legality predicates, selector candidate filters (including nested emissions),
and explicit final-instruction validation. Target CPU descriptions, not
build-host CPUID, determine compilation. POPCNT is the first end-to-end
consumer; other target instructions still need their requirements annotated.

This is not yet a general graph-rewrite/proof engine: 1:N value conversion,
analysis invalidation contracts, feature-expression coverage proofs and
offline SMT rule certification are not implemented by this decision compiler.
