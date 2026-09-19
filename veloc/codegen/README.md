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

The MIR optimizer handles bounded, same-type single-consumer SSA cones crossing
instruction and block boundaries. Cones are reconstructed at their root, where
all leaf values are available in valid SSA; multi-user producers remain boundary
inputs. Effectful instructions stay outside the graph. Rules come from
`veloc/optimizer/defs/equivalences.spec` and share constant evaluation with direct
evaluation. Saturation is bounded by cone size, node count, function-wide matching
fuel and rounds. Consumer-first partitioning avoids rebuilding overlapping cones.
Extraction commits only a smaller instruction DAG. The removed LIR pass's
dominating value-numbering sweep has not been migrated to MIR yet.

This is an instruction-count heuristic, not a target cost model or globally
optimal DAG extractor. Smaller IR does not guarantee faster machine code. Compare
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
