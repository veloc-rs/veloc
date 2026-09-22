# Codegen diagnostics

Dump LIR after selected passes, optionally restricting the function name:

```sh
cargo run --release -p veloc-wasm --bin veloc-wasm -- \
  crates/veloc-wasm/tests/wasm/coremark.wasm --strategy jit --compile-only \
  --dump-after translated,legalize,selected,final \
  --dump-function func_10 2> /tmp/coremark.lir
```

`--dump-after '*'` prints every function pass, plus the translated and final
checkpoints. Dumps go to stderr and do not require verification to be enabled.
Omit `--dump-function` to include all functions. Pass names appear in dump headers
and in `--print-stats` output.

Use `--print-stats --compile-only` to inspect aggregate pass timings, instruction
counts (translated, legalized, selected, final) and emitted code/data bytes.
Timings exclude IR printing. They are not total compilation time: frontend work,
translation, object packaging and loading must also be measured separately.

## E-graph experiments

Equality exploration lives in the MIR optimizer's `ExpressionPass`, not codegen.
Enable frontend O1 to run it; at O0 it does not run. `--fast-egraph` selects a
smaller resource budget for the same optimizer. Build once, then
alternate runs of `target/release/veloc-wasm` with and without that option. Do not
run the variants concurrently; check CoreMark's CRC validation as well as scores.

The MIR optimizer imports a whole-function expression graph while retaining the
MIR CFG. Supported pure scalar computations float in the graph, including
multi-user producers. Block parameters and results of unsupported or effectful
instructions are opaque leaves. Supported trapping operations are fixed
occurrences: their inputs participate in constant propagation, but they remain
leaves for code placement and cannot participate in algebraic rewriting. Rules come from
`veloc/optimizer/defs/equivalences.spec` and share constant evaluation with direct
evaluation. Saturation is bounded by additional nodes beyond the original graph,
function-wide matching fuel and rounds; fast mode does not cut shared expressions
into independent cones. Trapping instances can fold only when successful constant
evaluation proves that they do not trap. Pure and fixed computations use the same
dependency-driven evaluation queue, not separate scanning passes.
SSA forward references allocate empty e-classes rather than placeholder nodes.
Operation identity excludes MIR instruction IDs; multi-result projections use
interned operation IDs. Extraction returns node IDs without copying expressions.
Congruence repair, constant propagation and extraction cost
updates follow parent dependencies. Pattern matching backtracks through reusable
bindings rather than allocating a Cartesian product of environments.

Extraction starts with tree costs, then performs budgeted local improvements using
the cost of the shared DAG. A separate placement phase reuses dominating values,
keeps unchanged instructions, and inserts new computations at original instruction
anchors. Fixed instructions retain their control-flow position and order. Dead
pure computations are removed by liveness, not by deleting a collected region.
This is conservative placement, not global code motion or loop-invariant hoisting;
sharing in the expression graph need not imply one executable instance across
incomparable branches. Real placement cost is not yet modeled by extraction.

The default cost model counts instructions; callers can supply a target cost
model. Extraction is not globally optimal, and the local search budget is
proportional to graph size. Smaller IR does not guarantee faster machine code. Compare
compilation time, emitted size and repeated runtime measurements separately.

### Historical LIR CoreMark snapshot (2026-09-19)

These measurements describe the removed LIR implementation, not the current MIR
optimizer. The MIR pipeline has not yet been benchmarked.

Local release build, default frontend optimization, same binary for both modes:

| Measurement | E-graph off | E-graph on |
| --- | ---: | ---: |
| Final instructions | 75,207 | 74,891 |
| Emitted code/data bytes | 356,952 | 355,424 |
| Compile-only process median, ms | 70.784 | 72.791 |
| E-graph pass median, ms | 0.009 | 2.127 |
| CoreMark iterations/sec, two alternating runs | 15,048.91; 14,507.47 | 14,194.46; 14,702.64 |

Compilation medians use ten samples per mode after two warmups. Runtime runs
were sequential in off/on/on/off order and all passed CRC validation. These
measurements show smaller output and additional compile time, **not a demonstrated
runtime speedup**; the two-run runtime sample is too small for a stable estimate.
The next optimization target is extraction cost (target instructions, sharing and
register pressure), rather than adding unconstrained saturation rules.
