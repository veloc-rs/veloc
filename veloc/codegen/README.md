# Codegen diagnostics

Dump LIR after selected passes, optionally restricting the function name:

```sh
cargo run --release -p veloc-wasm --bin veloc-wasm -- emit \
  crates/veloc-wasm/tests/wasm/coremark.wasm --strategy jit --emit object -o /tmp/coremark.o \
  --dump-after translated,legalize,selected,final \
  --dump-function func_10 2> /tmp/coremark.lir
```

`--dump-after '*'` prints every function pass, plus the translated and final
checkpoints. Dumps go to stderr and do not require verification to be enabled.
Omit `--dump-function` to include all functions. Pass names appear in dump headers
and in `--print-stats` output.

Use `emit --emit object -o module.o --print-stats` to inspect aggregate pass timings, instruction
counts (translated, legalized, selected, final) and emitted code/data bytes.
Timings exclude IR printing. They are not total compilation time: frontend work,
translation, object packaging and loading must also be measured separately.

## Final code layout

The pipeline keeps emission symbolic until all module passes have finished:

```text
selection -> allocation -> frame finalization -> pre_emit module passes
          -> symbolic emission -> post_emit module passes
          -> section layout / relaxation -> object serialization
```

`Emitter` records bytes, block labels and encoding alternatives (`CodeForm`).
Targets supply the bytes, relative-field patchers, unresolved-symbol relocations
and any alignment requirement. The shared layout engine owns addresses and
encoding selection for both x86 and RISC-V. Labels refer to fragment boundaries;
they never need repair when an instruction changes size.

Layout starts with preferred forms, computes section-wide addresses, and promotes
forms whose displacements cannot be encoded. It repeats until every selected form
is valid. Choices only move towards fallbacks, so changing alignment cannot cause
oscillation; no iteration budget is needed. Alignment can make an earlier promotion
unnecessary later, so this guarantees valid layout, not globally minimum size.
Padding is recomputed from final positions and is specific to each encoding form.

The object writer lays out all defined functions together with its section
alignment. References to definitions in that section bind directly; unresolved
symbols keep relocations. Object serialization must preserve those positions.
Byte-changing module passes therefore run before layout, not after it.

These are mandatory emission steps, not instruction-selection optimizations.
Neither `pre_isel` nor `post_isel` has the final addresses needed to choose forms.

## Pass execution and analysis ownership

Function passes implement `run(&mut FunctionSession) -> Result<()>` and declare
their input/output `FunctionStage`. The shared runner checks the input stage even
when IR verification is disabled; with verification enabled it checks the output
invariants after every pass. The context assigns execution positions, including
repeated occurrences. Target extension passes use the same runner and contracts.
`FunctionPipeline` owns the entire function compilation sequence, its target
extension sequences, and per-run analysis caches. The driver invokes it once per
function and then handles module passes and emission. The register allocator is
an ownership-consuming transition inside this pipeline, outside the ordinary
function-pass interface; the pipeline invalidates analyses at that boundary.

`FunctionSession` owns access to the current function and its analysis cache.
Queries borrow the session, and edits exclusively borrow it, so a borrowed
analysis cannot survive into a mutation. An owned analysis snapshot or plan may
be retained deliberately, but it is not a query for the updated function.
Extension passes can intern function symbols through the edit guard, without
obtaining mutable access to the shared symbol table.

- `edit()` exposes the existing `FuncEditor`, with read-only access to the
  function. Passes cannot obtain `&mut MachineFunction`; generated builders,
  replacement and frame finalization all use the editor's mutation APIs.
- Opening a general edit conservatively invalidates all function analyses before
  granting write access, including when no changes follow. The exclusive borrow
  prevents queries until the edit ends. Early returns and unwinding require no
  deferred invalidation or pass-authored change report.
- Constrained operations such as `erase_block` and `reorder_block` retain their
  specific invalidation categories. The legalizer's `EditChanges` only collects
  instructions for its incremental worklist; it is not an analysis journal.
- Full verification remains optional at pass boundaries. Target operand, type
  and feature checks are generated from spec and used by that verifier.
- Scheduling first computes an owned plan from a read-only function and current
  liveness, then applies its orders through the session.

CFG and liveness track their direct IR inputs. Dominators, post-dominators, loop
information and register pressure validate prerequisite analysis revisions rather
than duplicating their prerequisites' change masks. Instruction ordering is an
input to CFG/liveness because unrestricted reorder operations can move control
flow or change use-before-definition relationships.

`CompiledModule` contains machine functions only. Emission consumes it and
produces `EmissionModule`, which contains symbolic code only. Module pipelines
are parameterized by that representation; pre-emit and post-emit passes cannot
be interchanged. Function passes have no mutable module-analysis context.

Instrumentation times execution separately from output verification and IR
formatting. A profile `Observation` retains the original scope identity after its
timer ends, so success/failure remarks and IR artifacts remain associated with
that execution. Errors abort the current compilation and may leave partial edits;
neither pass execution nor editing implies rollback. Detailed artifacts remain
opt-in and include the available function state on failure.

Execution order is explicit. This infrastructure does not infer optimization
ordering from analysis dependencies, automatically repeat passes, or schedule
parallel mutations. Incremental analysis updaters can be added when measurements
justify them; cache validity does not depend on their availability.

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
`veloc/optimizer/defs/equal.spec` and share constant evaluation with direct
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

## Driver entry points

Construct `CodegenPipeline::new(target, options)` and call `compile_object`.
Use `with_profile` to attach shared compilation profiling. Benchmarks of this
entry point include object serialization and report object bytes.

`CodegenOptions::opt_level` selects the pipeline at construction time:
`OptLevel::None` retains required lowering, selection, allocation, and frame
handling; `OptLevel::Default` additionally installs post-selection combining and
scheduling. Target pass factories receive the same level and must retain required
transformations at every level. Passes execute unconditionally once registered;
they do not receive the driver's options or decide whether they are enabled.
Verification and dumps remain runner policy, independent of optimization level.

`dump_after` selects pass names (`*` selects all), with an optional `dump_function`
filter. When `dump_after` is empty, the driver translates `VELOC_DUMP_LIR` into
these options once at construction. Function passes and the translated,
regalloc, and final boundaries use the same dump formatter.

Translation returns machine IR together with an explicit `MachineFuncId` to MIR
`FuncId` mapping; compilation does not depend on the two modules' iteration order.
