# ISLE

ISLE compiles typed rules at build time. It uses OpSpec's lexer/parser,
diagnostics, logical operation signatures and exact type sets; it does not
maintain another instruction catalog or depend on a runtime IR.

## Value rules

Register checked definition units under explicit dialect names with
`rules::Dialects`, then compile a rule file with `rules::Program::compile`.
The rule syntax uses the same declarations and expressions as .ops files:

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

`target::compile` consumes OpSpec instruction contracts and the target-selection
rules. Register definitions, scheduling metadata, assembly and encoding
expressions live in `.ops`; ABI and selection rules currently remain in `.isle`.
Encoding expressions use the shared typed expression compiler and explicitly
declared Rust host interfaces. Byte encoding lives in `veloc-encoder`, not in
an ISLE macro language.

The current generic core deliberately handles fixed-arity value rules.
Properties, successors, variadic signatures, structural/shape-dependent
types, guarded alternatives and multi-node source matching require additional
checked adapters. They are diagnosed rather than silently discarded.
Memory/control/call lowering and the existing graph-covering target selector
remain specialized consumers; target selection has not yet been migrated to
the new value-rule core. Sharing the contracts does not imply sharing graph
mutation, CFG construction or instruction-encoding algorithms.

Run `CARGO_INCREMENTAL=0 cargo test -p veloc-filetests --test rules` for rule-to-Rust execution
tests and the existing target-description tests.

The CLI exposes the two consumers explicitly:

```text
cargo run -p veloc --bin veloc-isle -- target INPUT DEFINITIONS OUTPUT ARCH
cargo run -p veloc --bin veloc-isle -- rules INPUT OUTPUT SOURCE_NAME SOURCE_OPS TARGET_NAME TARGET_OPS
```

The rules command emits `lower` and its `Context` trait, referring to the host's
`SourceOpcode` and `TargetOpcode` aliases. Build scripts can use `rules::Rust`
to bind different enum paths and function names without changing rule files.

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
rule ctpop<T: Word>(inst: lir::Ctpop<T>, recipes: &Recipes, target: &Target) {
    action = match T {
        Type::I32 if target.supports(Instruction::POPCNT32) => recipes.legal(),
        Type::I64 if target.supports(Instruction::POPCNT64) => recipes.legal(),
        _ => recipes.bit_count(),
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

`replace` compiles a checked, fixed-arity pure value expression. `action`
calls a declared Rust interface to obtain a plan for memory, control, variadic
or other complex rewrites; planning does not mutate the function.
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
