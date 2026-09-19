# Table-driven selector

## Implemented boundary

The target selector uses one bytecode VM for matching and construction.
Each root opcode selects a static program. Ordinary guards, definition lookup,
temporary creation and target construction are bytecode operations, not a
per-rule Rust callback or constructor branch. Field descriptors come from the
checked operand-storage projections for each input opcode, including absent
optional fields. The VM directly reads inputs, results and attribute slots;
there is no field-ID callback or InstView dispatch. Sequence fields cannot be
used as scalar accesses and are diagnosed by the selection compiler.

Type queries and temporary allocation use VRegBuilder directly. Feature guards
compare word slices; target descriptors reference the existing opcode metadata
and construct through the common writer, including implicit registers. Only
explicit custom predicates call a Rust callback. The former Host trait and its
LoweringContext/TargetFeatures forwarding layers have been removed.

The earlier `[test, yes, no]` callback table has been removed. The measurements
below refer to that earlier experiment, **not the current bytecode VM**.

This is a scalability/design choice, not a claim that interpretation always
runs faster. The measurements below archive the earlier three-way experiment.

The matching IR distinguishes definition lookups, typed field guards, CPU
features, and fold-safety checks. Definition paths are canonicalized independent
of the rule author's binding names. Guard identity is structural, not equality
of generated Rust strings. The generator resolves paths to typed view accesses;
runtime execution does not parse names. Different target feature requirements
remain separate guards. Rules and failure fallback retain their original order.

Contiguous common prefixes are shared. Equal nodes are interned only when their
test and both continuations agree. We deliberately do not expand arbitrary
Boolean decision diagrams: the first experiment with that approach grew some
23-rule groups to over 220 nodes. The prefix algorithm bounds graph growth by
the input test occurrences plus terminal nodes; it does not reorder predicates
or speculate field accesses before the required definition/opcode guard.

Matching never mutates the input function. `Accept` enters a construction recipe;
no reject/fallback is allowed after this boundary. All source operands are read
before `BuildInst` writes detached target instructions. The existing selection
driver commits replacements, use-def changes and edge identity transfers.
This is not a general rollback transaction or an arbitrary value-rewrite engine.

Definition-slot initialization is checked over the decision graph at generation
time. Folding remains conservative: only pure single-result computations may
be duplicated; other users keep the old definition. Memory folding is not
enabled. Host predicate purity is a contract, not a proved property.

## Encoding and opcode boundary

Opcodes occupy one byte; indexes use ULEB128; branch destinations are fixed-width
little-endian u32 offsets from the program start. The format is private build
output, not a stable external file format. Types, integer constants, physical
registers and target opcodes are held in typed constant tables. Schema names and
field names are resolved by the generator.

- Matching: ReadReg, GetDef, CheckOpcode, CheckType, CheckInt, CheckFeatures,
  CallPredicate, CheckFoldable, Jump, Reject.
- Construction: Accept, MakeTemp, ReadResult, ConstReg, ReadField, ConstImm,
  BuildInst, Finish.

BuildInst carries three slot lists: results, inputs and payload fields. Target
opcode identity is data, not a separate VM opcode. Root opcode dispatch remains
a generated Rust match; internal multiway SwitchOpcode/SwitchType operations
are not implemented yet. The VM supports byte-offset disassembly for debugging.

Numeric opcode assignments, operand counts and branch encoding live together
in the runtime declaration. Generated code uses symbolic opcode names and
compile-time encoding assertions, so generator/runtime drift fails compilation.

## Relationship to veloc-interpreter

The program interpreter uses aligned eight-byte CodeWords, typed per-opcode
decoders and tail-call dispatch with program stacks and memory. This selector
uses compact variable-width bytecode, read-only matching and detached IR
construction. Neither executor nor value representation is shared.

The single-source opcode declaration approach is reused, but the existing
interpreter macro includes its storage layouts and handler ABI. Extracting that
macro, its readers or fixups currently would couple unrelated representations.
No dependency on veloc-interpreter, common VM crate, or generic dispatch layer
has been introduced.

## Reproduce
## Reproduce

```sh
cargo run -p veloc-codegen --release --example selector_bench

cargo test -p veloc-spec -p veloc-lir -p veloc-codegen
cargo check --workspace --all-targets --all-features
```

For reliable timing, pin the executable to the same available CPU. Do not time
Cargo or concurrent builds as part of selector execution. Each process uses one warmup and nine samples; selector samples contain 64 functions
and pipeline samples contain 128 compilations. Cloning, source parsing, and
output hashing are outside the respective timed regions.

The benchmark now measures only the table implementation. Reproducing the
historical three-way comparison requires the earlier experimental source;
the removed backends are not available through CLI or Cargo switches.

## Measurements (2026-09-19)

Environment: Linux x86-64 VM, reported AMD Ryzen 9 9950X, 24 virtual CPUs;
`rustc 1.98.0-nightly (bc2112ed5 2026-06-18)`. Cargo release defaults from this
workspace, debug information enabled, no added LTO or native-CPU flags. Execution
pinned to CPU 2. Seven rounds rotating the three executable orders; entries
below are medians of the seven process medians. This is a small synthetic
comparison, not a large multi-target rule-library or CoreMark benchmark.

Time per function/module, microseconds (lower is better):

| Workload | Legacy | Rust graph | Table graph |
|---|---:|---:|---:|
| Selector: integer32 | 29.377 | 28.709 | 28.170 |
| Selector: integer64 | 27.938 | 28.017 | 28.420 |
| Selector: float32 compare/select | 160.674 | 164.772 | 166.383 |
| Selector: float64 compare/select | 153.562 | 155.761 | 161.459 |
| Selector: stack-address loads | 20.094 | 18.740 | 19.426 |
| Selector: indexed-address loads | 22.975 | 23.138 | 23.590 |
| Full pipeline: sum, no optimization | 18.529 | 18.608 | 18.663 |
| Full pipeline: sum, optimization | 22.645 | 22.649 | 22.502 |

Small differences should not be treated as significant. For example, the
float32 ranges across processes were 156.242–172.787 us (legacy),
157.763–173.238 us (Rust), and 161.612–174.637 us (table). Stack-address ranges
were 19.718–20.511, 18.458–19.005, and 18.964–19.761 us respectively.

All eight output fingerprints matched across all three modes and all rounds.
Selector fingerprints cover dumped selected LIR; pipeline fingerprints cover
the emitted machine-code bytes. Existing native execution tests also run for
the different modes, covering calls, feature-dependent selection, floating
comparisons, memory, branches, and edge arguments. Fingerprints and tests are
not a proof for arbitrary programs.

Size measurements from the same executables:

| Metric (bytes) | Legacy | Rust graph | Table graph |
|---|---:|---:|---:|
| Selector symbols including its closures and table executor | 163,019 | 157,864 | 155,240 |
| Whole executable `.text` | 1,446,563 | 1,441,411 | 1,439,411 |
| Whole executable `.rodata` | 94,744 | 97,464 | 101,616 |
| GNU `size` text+data+bss (excluding debug sections) | 1,918,926 | 1,918,930 | 1,931,206 |
| Formatted complete generated target Rust | 1,363,348 | 1,488,485 | 1,472,331 |

Symbol sizes were summed from `nm -S -C`, deduplicating addresses. Source size
includes all target metadata/encoding code, not just selection. Table data,
callback shims and relocation-related data offset the reduction in selector
machine code. The complete binary is **not smaller** in table mode, and neither
new mode reduces generated Rust source size in this first implementation.

Warm-dependency rebuild observations (`/usr/bin/time`, rebuilding codegen,
regenerating artifacts and relinking the example): legacy 10.21 s, Rust 10.60 s,
table 10.02 s. Peak reported RSS: 996,696 / 1,007,196 / 984,180 KiB. These are
single observations from a consecutive run without competing builds, not a
statistically supported build-time improvement; initial dependency builds were
excluded. Earlier observations were 10.17 / 10.66 / 10.57 s, illustrating the
variation.

For that experiment's 189 candidates, 386 guard occurrences became 172 unique guard
definitions across opcode groups. The combined control-flow graphs contain 596
nodes including accept/reject nodes. This counts generated structures, **not**
the number of dynamically executed tests or eliminated runtime checks.

## Decision and next steps

Use table dispatch as the sole matching implementation, retaining the measured
tradeoffs above. The current VM also interprets construction and ordinary matching operations;
it has not yet been benchmarked against those historical results.

Next improvements should be measured separately:

1. Measure the bytecode VM on real application modules and additional architectures. Synthetic
   duplicate rules are not a substitute for a representative large rule set.
2. Share repeated construction recipes and consider multiway bytecode dispatch.
3. Integrate deferred replacement plans independently; avoid attributing their
   allocation/use-def savings to the matcher backend.
