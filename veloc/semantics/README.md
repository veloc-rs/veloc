# Executable semantics

`veloc-semantics` is independent of program IRs, instruction encodings, and solver
libraries. Its initial model contains pure, fixed-width bitvectors (1–128 bits)
and booleans, plus explicit operation-level traps.

The primitive/value core supplies the build-time vocabulary and reference
evaluation. `Program`, `Expr`, `Function`, and SMT export support generators,
offline checks, and tests without a feature switch. The optimizer's generated
constant evaluator executes specialized Rust arithmetic; it neither builds graphs
nor calls this crate. MIR and the optimizer have no normal dependency on this
crate. Offline recipes live in a separate generated `semantics.rs` artifact that
tools/tests include directly. `SemanticSpec<O>` is owned by this crate and generic
over the IR's operation identifier, without a dependency on MIR. Each entry
contains an opcode and a program; shared type/effect information is accessed
through `entry.opcode.spec()`.

- `BvOp` is the shared vocabulary for modular arithmetic and bitwise operations.
  `eval` normalizes inputs and results to the declared width.
- `BvOp::algebra` supplies reviewed primitive facts valid at every supported
  width. These include two-sided identity/absorbing constants and supported
  algebraic traits. The definition compiler rejects conflicting declarations;
  these facts are trusted definitions, not automatically synthesized proofs.
- `Program` is a borrowed/static recipe for a typed, multiple-result graph.
  `TypeRef` binds constants and conversions to input/result sorts; comparison
  properties are separate from runtime inputs. `instantiate(inputs, results,
  properties)` builds every step through the checked `Expr` constructors, including
  unused steps, then checks the output sorts. There is no separate Program evaluator.
  `primitive` recognizes only direct applications to all inputs in order; it does
  not infer equivalences. The same recipe representation accepts owned operand
  lists in the definition compiler and static slices in generated offline metadata.
- `Expr` constructors enforce operation arities and sorts. Cloning an expression
  shares its immutable nodes.
- `Function` checks every parameter reference against an explicit positional
  signature. `with_outputs` and `eval_all` support multiple, possibly shared or
  repeated outputs; `new` and `eval` are single-result conveniences. Execution
  and query generation use the same signature and graph.
- `with_traps` attaches ordered Bool guards: the first true guard selects a
  `Trap`. `execute` returns `Outcome::Values` or `Outcome::Trap`; malformed calls
  remain errors. Value-only `eval` helpers report a trap as `Error::Trapped`.
- `equivalence_query` produces SMT-LIB2 asking whether two functions can disagree.
  It requires identical input signatures and corresponding output sorts, and
  compares trap outcomes and, only on normal returns, whether ANY result can
  differ. Trap guards are not assumptions excluding invalid inputs. A representation
  conversion must explicitly reconstruct a comparable result, e.g. by concatenating
  two 32-bit result halves into a 64-bit value.

## Try a lowering check

```sh
cargo test -p veloc-semantics --all-targets
cargo run -q -p veloc-semantics --example split_add | z3 -in
cargo run -q -p veloc-semantics --example split_add -- --broken | z3 -in
cargo run -q -p veloc-semantics --example composed_neg | z3 -in
```

The example lowers a 64-bit modular addition into 32-bit additions, a carry
comparison, extension, and concatenation. A solver should report `unsat` for the
correct implementation and `sat` when `--broken` omits the carry. Z3 is optional
and is not a build or test dependency.

`composed_neg` compares `sub(Zero, x)` with primitive negation at 64 bits. It
should produce `unsat`; the composed program remains structurally distinct from
a primitive. Typed recipes also support comparisons returning Bool, signed/zero
extension, truncation, selection and multiple results. `IntPredicate` interprets
signedness and accepted outcomes; MIR generates these properties from its
comparison definitions. Bitwise and/or/xor accept Bool as well as bitvectors;
arithmetic requires bitvectors. Explicit conversion bridges Bool and bitvectors.

The optimizer's offline example checks MIR definitions against independent expressions:

```sh
cargo run -q -p veloc-optimizer --example semantic_check -- overflow | z3 -in
cargo run -q -p veloc-optimizer --example semantic_check -- overflow --broken | z3 -in
cargo run -q -p veloc-optimizer --example semantic_check -- comparison | z3 -in
cargo run -q -p veloc-optimizer --example semantic_check -- sext | z3 -in
cargo run -q -p veloc-optimizer --example semantic_check -- division | z3 -in
cargo run -q -p veloc-optimizer --example semantic_check -- division --broken | z3 -in
cargo run -q -p veloc-optimizer --example semantic_check -- shift | z3 -in
cargo run -q -p veloc-semantics --example check_primitives | z3 -in
```

Correct checks should return `unsat`; `--broken` changes only the overflow flag
and should return `sat`, exercising verification of the second result. `zext`
and `trunc` modes check the other conversion definitions. These examples use
concrete widths, not width-independent proofs. `IAddWithOverflow` is signed
overflow plus a modular sum, not unsigned carry.

`division` compares 8-bit signed division and its guards against widened arithmetic;
`--broken` omits the overflow guard and must return `sat`. `shift` compares
modulo-width counts with a bitmask at 32 bits. `check_primitives` compares concrete
evaluator edge cases with SMT execution, including 128-bit arithmetic and bit
counts; it checks these examples, not all inputs.

Primitive shifts and division follow total SMT-LIB bitvector semantics. In
particular, shift counts are not masked and division by zero produces a bitvector.
Operation recipes explicitly impose masked shifts and language-level traps.
Clz/Ctz of zero return the bit width; their portable SMT encodings and Popcnt use
bit extraction rather than solver-specific operators.

`unsat` establishes equality in this model at the specified widths. `sat` means
a counterexample exists; `unknown` proves nothing. The evaluator, SMT encoder,
and solver are trusted components here: no proof certificate is checked, and
there is no claim of end-to-end compiler verification. The model does not yet
cover memory, undefined behavior, floating point, vectors, loops, or ABI
state. Concrete-width checks are not width-independent theorems.
