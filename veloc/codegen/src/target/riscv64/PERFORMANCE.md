# K230 CoreMark comparison, 2026-10-03

The subsequent optimization round is recorded under
[Follow-up toward 4000](#follow-up-toward-4000). The first comparison below is
retained as the starting point for that round.

## Setup

- K230, one Linux-visible T-Head C908 hart, Linux 6.6.36, glibc.
- ISA reported by Linux: `rv64imafdcv_zicbom_zicboz_zicntr_zicsr_zifencei_zihpm_zba_zbb_zbs_svpbmt`.
- Veloc: release build, Rust `1.98.0-nightly (bc2112ed5 2026-06-18)`,
  `riscv64gc-unknown-linux-gnu`, Clang/LLD with the board sysroot.
- Source base: `4f78038`; the initial executable included the pending CLI
  redesign. The final executable additionally includes the changes below.
- Reference: official [Wasmtime 49.0.2 RISC-V release](https://github.com/bytecodealliance/wasmtime/releases/tag/v49.0.2),
  commit `3c8a3e79a`, default compiler settings and host features.
- Both runtimes use the same Wasm file, run serially on CPU 0. No compilation,
  transfers or other benchmark jobs run on the board during measurement.
- Five pairs alternate ordering: Veloc/Wasmtime, Wasmtime/Veloc, and so on.
  Results measure the benchmark's execution loop, not startup or compilation.
  CPU frequency and temperature were not controlled separately.

## Benchmark clock

The input is the [wasm3 CoreMark artifact](https://github.com/wasm3/wasm3/blob/6d93778f6eda84b67db3d26c48d193243f4b67f2/test/wasi/coremark/coremark.wasm)
at commit `6d93778f6eda84b67db3d26c48d193243f4b67f2` (33,990 bytes).
It requests WASI clock ID 2 and ignores the error result. On this Wasmtime
release that call returns errno 8 without a timestamp, preventing calibration
from completing. Veloc's existing WASI implementation accepts this clock ID.

For **both** runtimes, change only that clock argument to ID 1 (monotonic).
The CoreMark computation is unchanged. Reproduce the exact input with:

```sh
curl -fL https://raw.githubusercontent.com/wasm3/wasm3/6d93778f6eda84b67db3d26c48d193243f4b67f2/test/wasi/coremark/coremark.wasm -o coremark-baseline.wasm
python3 - <<'PY'
from pathlib import Path
import hashlib
p = bytearray(Path('coremark-baseline.wasm').read_bytes())
assert hashlib.sha256(p).hexdigest() == 'e71e358234e39803a1d27961439d924a69c836dd81c8670bfff7dbb82c097bbe'
assert p[0x20c8:0x20cc] == bytes.fromhex('41 02 42 00')
p[0x20c9] = 1
assert hashlib.sha256(p).hexdigest() == '733a87880ff345b2f4348f94df484a0ddb3f91c6692518cbd7c13d106217887b'
Path('coremark-monotonic.wasm').write_bytes(p)
PY
```

Commands on the board:

```sh
taskset -c 0 ./veloc-final run coremark-monotonic.wasm --strategy jit --cpu c908 -O1
taskset -c 0 ./wasmtime run coremark-monotonic.wasm
```

Veloc uses the default `--memory-checks auto`, which selects native guard
pages on this platform. Wasmtime uses its default memory safety settings.
Neither command disables Wasm bounds protection. The benchmark calibrates its
own iteration count; command-line arguments do not fix it in this artifact.

## Measurements

Scores are iterations/second; higher is better.

| Pair | Veloc | Wasmtime 49.0.2 |
| --- | ---: | ---: |
| 1 | 2894.384810 | 2904.260668 |
| 2 | 2897.155471 | 2904.905107 |
| 3 | 2901.614551 | 2907.267376 |
| 4 | 2890.316990 | 2918.112935 |
| 5 | 2893.763558 | 2915.634338 |
| **Median** | **2894.384810** | **2907.267376** |
| Minimum–maximum | 2890.316990–2901.614551 | 2904.260668–2918.112935 |

Veloc reaches **99.56%** of the Wasmtime median, a **0.44%** gap. This is
comparable performance on this workload; Wasmtime remains slightly faster in
every measured pair. Five pairs do not establish equivalence on other programs.

All ten runs calibrated to 40,000 iterations and reported `Correct operation
validated`. Their measured times were 13.707489–13.839312 seconds. All had
seed/list/matrix/state/final CRCs `e9f5 / e714 / 1fd7 / 8e3a / 25b5`.

The initial Veloc executable scored **1666.584196** iterations/second in one
baseline run (20,000 iterations, 12.000594 seconds). It used explicit software
bounds checks. The final median is **73.67% higher** than that sample. This is a
single baseline sample, not a baseline median.

## Retained changes and generality

1. **Native guard-page traps.** Reuse the existing reserved linear-memory
   mapping. On Linux/glibc RV64 and x86-64, native guest calls enter a C
   `sigsetjmp` boundary; the signal path checks both the guest PC and memory
   reservation before reporting an out-of-bounds trap. Rust-owned call storage
   stays outside this boundary. Unsupported platforms retain software checks.
   On RV64, multi-byte stores first probe the last byte: an unaligned store can
   otherwise modify its accessible prefix before faulting. Wasm alignment
   annotations are treated as hints, not guaranteed alignment.
2. **SSA-edge register affinities.** Block arguments and parameters prefer the
   same physical register when interference and fixed constraints permit it.
   The shared allocator retains all legality checks.
3. **Fallthrough branches.** The shared symbolic emitter removes jumps to the
   next block and reverses a conditional branch when that avoids a trailing
   jump. Targets provide normal/inverted encodings; final layout still handles
   branch relaxation and relocations. Both RV64 and x86-64 use this mechanism.
4. **Canonical boolean conversion.** Wasm comparisons extend a boolean to an
   integer directly, avoiding `select(cond, 1, 0)`. The interpreter handles
   boolean zero extension as well, preserving the common IR contract.
5. **ISA selection rules.** Declarative RISC-V rules fold zero-extended indices
   into Zba `add.uw`, use Zbb narrow extensions, absorb truncations into stores,
   select zero comparisons directly, and turn integer selects with a zero arm
   into a two-instruction mask. ISA requirements gate Zba/Zbb patterns; none
   checks a benchmark function name. Separate mask instructions expose their
   dependencies to scheduling and allocation.
6. **Redundant moves.** Drop 64-bit physical self-moves in the RISC-V encoder.
   Preserve 32-bit moves that may normalize the upper bits.

No CoreMark-specific kernel, constant, function or special execution path was
introduced. These changes do not require a new e-graph search policy or new
CPU scheduling constants. A loop-weighted allocation-cost experiment had little
benefit and was discarded. An early guarded-store experiment was also discarded
after detecting partial writes on an out-of-bounds unaligned store; the final
binary includes the last-byte probe.

## Correctness checks and limits

- `cargo check --workspace`: passed.
- K230 release tests: `hardware_memory` 4 passed, `riscv64_selection` 2 passed,
  `jit` 2 passed (the separately measured full CoreMark test remains ignored
  in that test binary).
- Guard checks cover out-of-bounds loads/stores, preservation of memory after
  trapping stores, unused trapping loads, large offsets, repeated calls,
  growth, imported memory, start functions, software mode and unrelated signals.
- An external WAT probe passed 96 select cases: generic/C908, O0/O1, i32/i64
  signed extrema, both zero arms and conditions 0, -1 and 2. No new repository
  test files were added.
- Native host `veloc-codegen --lib --release`: 27 passed, 2 failed. The same
  two failures reproduce on a clean archive of base commit `4f78038`:
  `tied_allocation_preserves_inputs_with_collisions_and_spills` (flags-def
  expectation) and `compile_module_to_object_keeps_defined_and_imported_symbols`
  (unused import symbol expectation). They are not represented as passing.
- Runtime verification here is on RV64. The shared x86-64 changes were compiled
  but were not executed on an x86-64 host. This is not full Wasm conformance
  testing, a native C CoreMark certification, or evidence about other workloads.

## Artifacts

The local experiment archive is `target/k230-wasmtime-20261003`; the board archive
is `/root/veloc-riscv/wasmtime-20261003`. These contain binaries, raw logs,
disassemblies, compiler output and the final serial run script.

| Artifact | SHA-256 |
| --- | --- |
| Original Wasm | `e71e358234e39803a1d27961439d924a69c836dd81c8670bfff7dbb82c097bbe` |
| Monotonic-clock Wasm | `733a87880ff345b2f4348f94df484a0ddb3f91c6692518cbd7c13d106217887b` |
| Initial Veloc | `7d1a4fccfa3958dc084f500be122968c406a62a5b87e18d67a014c7cac76bc65` |
| Final Veloc | `52699bee867d75fdfdbab81aec7a6130a7b588d62feb08fe7cede9ad2f297ac8` |
| Wasmtime | `57bbf9e4ab200ce04a9464e000ec174031e1703ac8949f28900a9f8ff2d2407c` |

## Follow-up toward 4000

The follow-up uses the same board, clock-corrected input, memory protection,
compiler settings and Wasmtime executable described above. No input-specific
function names, CRC constants or benchmark detection enter the compiler.

### Changes

- **MIR value analysis:** propagate identical incoming values through block
  parameter cycles and remove dead parameter cycles together with their edge
  arguments. DCE alternates instruction deletion and parameter pruning until
  neither exposes more work.
- **Address congruence and read validity:** include pointer scale and offset in
  e-graph congruence keys. A successful load can prove a later unused read of
  the same byte range redundant within the block; unknown lifetime changes and
  volatile accesses discard the proofs. Known-valid runtime metadata loads use
  `notrap`, permitting ordinary DCE. Guest accesses retain bounds protection.
- **Integer bit analysis:** propagate demanded bits backward and known zeros
  and ones forward. Unmodeled operations and CFG edges conservatively observe
  all input bits. Constant rewrites precede fact construction; subsequent
  replacements preserve complete values so downstream facts remain valid.
- **Allocation and frames:** release x7/x8/x28/x29/x30 for allocation, express
  rotate temporaries in Spec, prefer caller-saved homes and unary input reuse,
  estimate spill cost from loop depth and uses, and save `ra` only when needed.
- **Control flow:** lower branch tables to unsigned comparison trees, collapse
  identical target ranges, and place loop traces after allocation has inserted
  edge-copy blocks. The shared emitter chooses fallthroughs and branch forms.
- **RV64 selection:** use explicit arithmetic select recipes and the reusable
  `same_value` matcher guard, simplify boolean chains, recognize narrow casts
  expressed as shifts/masks, and remove redundant zero extensions of narrow
  loads. A post-selection pass fuses sole sign-extension users into LB/LH/LW
  at the original load position, preserving access width and flags.
- **Copy encoding:** preserve the complete integer register representation.
  Only standalone self-copies are omitted; embedded copies retain the sizes
  assumed by fixed-offset expansion sequences. Float32 self-copies retain
  their NaN-box normalization behavior.

Several isolated candidates were within measurement noise. In particular,
weighted allocation, bit analysis and unary affinities are not claimed to have
individually produced a measured CoreMark speedup. The final comparison measures
the combined pipeline. An early parameter-forwarding candidate regressed before
the register resource changes and was not used as the final configuration.

### Repeated measurements

Five serial pairs alternate Veloc/Wasmtime and Wasmtime/Veloc. Final executable:
`f8dc7f7c631b8984e4fae4667540c1d53fad634459e70fdab65410c6cefca2e7`.

| Pair | Veloc | Wasmtime 49.0.2 |
| --- | ---: | ---: |
| 1 | 3489.204700 | 2907.485386 |
| 2 | 3483.447563 | 2900.687764 |
| 3 | 3483.719553 | 2908.968959 |
| 4 | 3487.142241 | 2907.167760 |
| 5 | 3486.279234 | 2910.526442 |
| **Median** | **3486.279234** | **2907.485386** |
| Minimum–maximum | 3483.447563–3489.204700 | 2900.687764–2910.526442 |

The Veloc median improves **20.45%** over the previous
2894.384810 median and is **19.91%** above this round's Wasmtime
median. **4000 has not been reached**: the remaining gap is 513.72
points, requiring another 14.74% increase from this result.

All ten runs report valid CoreMark CRCs. Veloc calibrated 60,000 iterations;
Wasmtime calibrated 40,000. The compute input and auto-calibration algorithm are
the same; iteration-dependent final CRCs differ as expected. Frequency and
temperature remain uncontrolled, so smaller per-candidate differences are not
interpreted as established improvements.

### Validation

- Workspace check passed. Optimizer library tests: 9 passed; analyzer: 2;
  Spec compiler: 10. Encoder library currently has no unit tests.
- Final K230 binaries: hardware memory 4 passed, instruction selection 2,
  JIT/ABI 2; the separately measured full CoreMark test stays ignored.
- External WAT checks cover 304 rotation/select/shift/table cases, 1024
  deterministic random integer expressions, and 26 signed-load cases per
  configuration. All run with generic/C908 and O0/O1: **5416 case executions**.
  The random probe caught an initial demanded-bits bug; that candidate was
  rejected and the corrected implementation reran the complete probes.
- No repository tests were added. Runtime testing remains on RV64, not x86-64.
- The two previously recorded codegen test failures remain. MIR library tests
  also expose the existing `const nope` parser error-column expectation:
  29 pass, 1 fails. The same failure reproduces from clean base `4f78038`.

Local artifacts: `target/k230-4000-20261003`. Board artifacts:
`/root/veloc-riscv/coremark-4000`. `final-runs.log` records test output, executable
and input hashes, alternating measurement order and every CoreMark result;
`measure.sh` contains the board commands. The uninstrumented executable is used
for every reported timing; the earlier SIGPROF sample was diagnostic only.

The remaining avenues include loop transformations, rematerialization/live-range
splitting, and memory-dependency precision. Their benefit still needs measurement;
these results do not establish that the remaining optimization space is exhausted.
