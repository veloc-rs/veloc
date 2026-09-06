# Executable semantics

`veloc-semantics` is independent of program IRs, instruction encodings, and solver
libraries. Its initial model contains pure, fixed-width bitvectors (1–128 bits)
and booleans.

- `BvOp` is the shared vocabulary for modular arithmetic and bitwise operations.
  `eval` normalizes inputs and results to the declared width.
- `BvOp::algebra` supplies reviewed primitive facts valid at every supported
  width. These include two-sided identity/absorbing constants and supported
  algebraic traits. The definition compiler rejects conflicting declarations;
  these facts are trusted definitions, not automatically synthesized proofs.
- `Program` stores width-parameterized compositions as static, single-result
  bitvector DAGs. It checks input references, earlier-step references, arities,
  output and index capacity before evaluation. `instantiate` binds a width to
  obtain the same `Function` used by the SMT encoder. `primitive` recognizes only
  direct applications to all inputs in order; it does not infer equivalences.
- `Expr` constructors enforce operation arities and sorts. Cloning an expression
  shares its immutable nodes.
- `Function` checks every parameter reference against an explicit positional
  signature. Execution and query generation use that same signature and graph.
- `equivalence_query` produces SMT-LIB2 asking whether two functions can disagree.
  It requires identical input signatures and output sorts. A representation
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
a primitive. Its parameters are all bitvectors of one width, with one result:
this new static representation does not add effects, floating point or traps.

`unsat` establishes equality in this model at the specified widths. `sat` means
a counterexample exists; `unknown` proves nothing. The evaluator, SMT encoder,
and solver are trusted components here: no proof certificate is checked, and
there is no claim of end-to-end compiler verification. The model does not yet
cover memory, traps, undefined behavior, floating point, vectors, loops, or ABI
state. Concrete-width checks are not width-independent theorems.
