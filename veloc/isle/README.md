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
    match: mir.INeg(x),
    emit: lir.Neg(x),
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
paths, values, types and construction; MIR-to-LIR construction invokes
OpSpec-generated builders so physical storage ordering is not duplicated.

Exact, pure primitive matches can be inferred from OpSpec semantics. These
matches go through the same checker and Rust emitter as explicit rules.
Explicit rules take precedence; ambiguous inferred mappings are errors.
Type checking is **not semantic equivalence verification**. Explicit rules
remain reviewed transformations, including trap and floating-point contracts;
there is no SMT invocation or new proof claim here.

## Target descriptions

`target` contains the existing machine-description language and its compiler:
registers, ABI, encodings, scheduling metadata and target selection. Its
entry point is `target::compile`, not the generic rule compiler.

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
cargo run -p veloc --bin veloc-isle -- target INPUT OUTPUT ARCH
cargo run -p veloc --bin veloc-isle -- rules INPUT OUTPUT SOURCE_NAME SOURCE_OPS TARGET_NAME TARGET_OPS
```

The rules command emits `lower` and its `Context` trait, referring to the host's
`SourceOpcode` and `TargetOpcode` aliases. Build scripts can use `rules::Rust`
to bind different enum paths and function names without changing rule files.
