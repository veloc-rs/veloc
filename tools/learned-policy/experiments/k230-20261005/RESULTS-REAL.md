# K230 real-program neural-policy results — 2026-10-05

## Outcome

Adding real-program training data improves the **20 verified native C programs by
1.20% in geometric mean**, relative to both existing heuristics and the previous
neural model. Improvements occur in programs used for training. The two validation
programs and seven previously unmeasured test programs produce unchanged machine
code: **this experiment does not demonstrate a speedup on unseen programs**.

The new model remains opt-in, alongside the original model:

```sh
target/release/veloc-c input.i -O1 --cpu c908 \
  --policy tools/learned-policy/models/c908-real.vlp -o input.o
```

| Program / partition | Speed change versus heuristics |
| --- | ---: |
| BEEBS RC4 (`nettle-arcfour`, training) | **+14.19%** |
| BEEBS FDCT (training) | **+7.67%** |
| Embench EDN (training) | **+2.39%** |
| BEEBS integer JPEG DCT (training) | +0.89% |
| All 11 verified training programs, geometric mean | +2.20% |
| 2 verified validation programs | 0%; identical executables |
| 7 verified test programs | 0%; identical executables |
| All 20 verified programs, geometric mean | **+1.20%** |

Huffbench (+0.008%) and Statemate (+0.029%) are within measurement noise. Other
programs produce byte-identical executables. The previous model also produces
baseline-identical executables across this entire verified real-program subset.

The 95% bootstrap interval for the aggregate gain on this **fixed collection of
programs** is +1.18% to +1.23%. It resamples paired measurement rounds within each
program, with 10,000 resamples. It is not a confidence interval for performance on
future programs. Per-program intervals and raw samples are in
the local, Git-ignored `../../artifacts/results/c908-real.json` report.

## Coverage and controls

Sources, revisions, partition membership and reproduction commands are documented
in [REAL_PROGRAMS.md](REAL_PROGRAMS.md). These are adapted Embench/BEEBS compiler
experiments, not official suite scores. Both compilers receive identical
preprocessed input, including the documented header/attribute adaptations.

| Coverage | Embench | BEEBS | Total |
| --- | ---: | ---: | ---: |
| Attempted programs | 19 | 31 | 50 |
| Successfully compiled | 8 | 16 | 24 |
| Successfully verified and measured | 8 | 12 | 20 |
| Upstream verifier reports unsupported (`-1`) | 0 | 4 | 4 |
| Compilation failures | 11 | 15 | 26 |

The four unverified programs are FIR, NS, Qsort and Select. They are excluded from
training and speed aggregates. Compilation failures are retained in the result
data, including unsupported C constructs and invalid entry-block parameters in
`aha-mont64`, `crc32` and `nettle-cast128`. No source algorithms were rewritten to
force those programs into the measured set.

Final execution used K230 CPU 0, one process at a time, an unrecorded warmup and
15 measured rounds. Repetition counts were calibrated toward 250 ms and kept equal
between variants. Every measured process passed its upstream verifier; return
checksums agreed across variants. Byte-identical executables share one execution
per round, preventing placement/timing noise from becoming a false optimization
gain. An earlier measurement before this deduplication is retained in the board
experiment directories; it was not used to change the frozen model.

The LLVM control is Homebrew Clang 22.1.6, `-O3 -ffp-contract=off`, with the same
RV64GC/Zba/Zbb ISA and Linux LP64D ABI, without LTO. This installation does not
offer a C908 CPU model, so its CPU scheduling model is generic. Across these 20
programs, tuned Veloc's execution time remains **1.63× LLVM's** in geometric mean.
The results do not establish parity with LLVM.

## Compilation cost and cross-host validation

Native K230 compilation used two warmups and nine alternating measured rounds per
program. Each translation unit starts a compiler process; model loading and object
writing are included, preprocessing/linking/tracing/IR verification are excluded.
Support units are compiled separately for each program.

| Compilation statistic | Heuristics | New model | Increase |
| --- | ---: | ---: | ---: |
| Sum of the 20 per-program medians | 3.4704 s | 3.5265 s | **1.62%** |
| Geometric mean of per-program ratios | — | — | **2.50%** |

All repeated native builds were stable. Nineteen programs produced object files
identical to their host cross-compilation outputs. **Ludcmp differed in both
baseline and model builds**: the native compiler introduces an additional loop
recurrence; the difference is already visible in optimized MIR. This is an
existing cross-host reproducibility issue, independent of whether the model is
enabled. The exact origin was not changed in this experiment.

The two native Ludcmp variants were separately linked with the common adapter and
executed on K230 at 1, 31 and 10,000 repetitions; all passed the upstream verifier.
Those two native executables are byte-identical to each other. Runtime gains above
refer exclusively to the fixed host-built executables. Native compilation cost is
reported separately, with the object mismatch preserved in `host_differences` and
the validation runs in `native_validation` in the result data.

## Training and regression checks

- Added **422** single-context interventions on real programs, deduplicated to
  **212** changed executables measured on K230. All passed validation.
- Combined these labels with the previous synthetic/CoreMark tuning data.
  The new networks retain the `16 → 32 → 16 → 3/5` architectures. Inlining uses
  asymmetric regression; scheduling uses classification.
- Program names, paths and benchmark IDs remain outside model features.
  Compiler legality checks, fallback domains and per-caller advice limits remain
  unchanged. This iteration changes training data/weights and experiment tooling.
- Compared four candidate combinations using development programs, prior
  synthetic regression checks and CoreMark. The model was frozen at
  `2026-10-04T19:35:09Z` before new test-program performance measurements. No
  model weights were changed after inspecting those test results.
- The final model has 375/137 training/validation inlining contexts and 468/133
  scheduling contexts. Seeds, epochs, source-data hashes and the exact training
  command are in the local `../../artifacts/models/c908-real.training.json` file.
- Rechecked all 108 synthetic kernels with both prior seeds. All checksums passed.
  Geometric mean changes versus heuristics were +1.66% training, +1.72% validation
  and +0.23% previously excluded families. These synthetic families have been
  repeatedly inspected during development and are not a blind evaluation.
- All five CoreMark objects **and its executable** are byte-identical to those of
  the original deployed model. It therefore preserves the previously validated
  +2.48% CoreMark result; there is no additional CoreMark code improvement.
- JSON and packed models produce identical objects for all 24 built real programs.
  Host benchmark builds used IR verification. Python syntax/format checks,
  `cargo build --release -p veloc-c` and `git diff --check` passed.

## Artifacts

- Packed/JSON weights and provenance are preserved locally under
  `../../artifacts/models/`; raw measurements are in
  `../../artifacts/results/c908-real.json`. Generated files are ignored by Git.
- Model SHA-256:
  `898e758e670901bd68536c0776db60c908bf8f6d9d1912e1db939ba496def12e`.
- Host compiler SHA-256:
  `a068a5626dd8927d0c72e7ee3ad52ebd937701f6ef09a971801be65c4bfd27b0`.
- Native compiler SHA-256:
  `c0187c4cd51d69428dc57ef25f2dbd51d51779c7b5bd67b37dfefeeb0257468a`.
- Unchanged CoreMark executable SHA-256:
  `5037d24380aba61cb7ab00171db9a57862ba4963c7e5895c1f5172ae3a476f79`.
- Full local experiments: `target/embench-policy-20261005/` and
  `target/beebs-policy-20261005/`; matching board directories are under
  `/root/veloc-riscv/`. They retain sources' hashes, baseline/probe/candidate
  binaries, failed build logs, individual timings and rejected candidates.

More data improved several measured opportunities, while unseen-program gains
remain unproven. This model should remain opt-in. Broader applicability requires
more verified program coverage and training that better captures interactions
between successive decisions; merely loosening fallback checks is not supported
by these measurements.
