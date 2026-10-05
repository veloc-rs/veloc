# K230 neural-policy experiment — 2026-10-05

The subsequent [real-program experiment](RESULTS-REAL.md) adds Embench/BEEBS and
a separately named model. The results below describe the original model.

## Outcome

The trained policy improves **native C CoreMark by 2.48%**, with **1.54% additional
compilation time** on K230. The working acceptance criteria were at least 3%
runtime improvement and at most 2% compilation overhead. Compilation meets the
criterion; runtime does not. The model remains **opt-in**, with no claim of a
general speedup across programs.

| Final measurement | Existing heuristics | Neural policy | Change |
| --- | ---: | ---: | ---: |
| CoreMark median, iterations/second | 6213.914893 | 6367.843435 | +2.48% |
| Compile five preprocessed units, median | 811.35 ms | 823.82 ms | +1.54% |

```sh
target/release/veloc-c input.i -O1 --cpu c908 \
  --policy tools/learned-policy/models/c908.vlp -o input.o
```

These are native C results, not Wasm measurements or a new LLVM speed comparison.

## What was measured

- K230 Linux CPU 0, `thead,c908`, RV64GC with Zba/Zbb, LP64D.
- Unmodified EEMBC CoreMark revision
  `1f483d5b8316753a742cbf5590caf5bd0a4e4777`, five benchmark translation units,
  identical shared OS adapter, `-O1 --cpu c908`, no LTO.
- Seven alternating measured runs of 100,000 iterations after warmup; all scored
  runs exceed ten seconds and pass CoreMark validation. Arguments:
  `0 0 0x66 100000 7 1 2000`.
- Native K230 compilation: three warmups and 21 alternating measured runs of all
  five `.i` files. Includes process startup and model loading; excludes
  preprocessing/linking. Tracing and IR verification are disabled while timing.
- The final compiler produces objects byte-identical to those used for the long
  runtime measurement. This was checked on the host and on K230; binary/JSON
  model encodings also produce identical objects.

The first JSON deployment cost 3.35% extra compilation time. A direct MessagePack
representation avoids decimal conversion and tagged-enum intermediate buffers;
the final 1.54% figure includes the model-domain checks and neural inference.

## Model and training

Two separately trained MLPs: `16 → 32 → 16 → 3` for inlining and
`16 → 32 → 16 → 5` for scheduling, with ReLU hidden layers. Inlining uses
asymmetric regression; scheduling uses class-weighted classification. Weights
come from PyTorch training on measured K230 interventions, not hand-written
coefficients. The compiler runs scalar Rust inference without Python/PyTorch.

Training data comes from 108 generated C kernels and explicitly identified
CoreMark tuning measurements. There are 3,738 attempted single-context
interventions, deduplicated to 1,478 changed executables. Baseline/fixed strategies
were also measured. Results for all kernel controls were checked against an
independent LLVM build at two seeds; learned variants were checked against those
validated baselines. No checksum or CoreMark CRC mismatch occurred.

Only structural features enter the networks. Models use measured feature ranges
and at most 128 representative vectors to reject unfamiliar contexts. These
vectors contain no action labels. Inlining allows one learned deviation per
caller per pass, then continues ordinary heuristic inlining. This matches the
isolated-intervention training setup and avoids treating individual gains as
additive. Removing repeated advice reduced two observed complex-kernel
regressions from 21%/15% to approximately 0.6%/0.5%, while preserving CoreMark code.

CoreMark **was used for tuning**. Its result must not be presented as unseen-program
generalization. The partitioned kernel measurements for the final policy are:

| Program partition | Cases | Geometric mean speed change | Worst case |
| --- | ---: | ---: | ---: |
| Training families/variants | 70 | +1.72% | −2.84% |
| Validation variants | 14 | +0.53% | −1.22% |
| Excluded test families | 24 | −0.01% (essentially unchanged) | −2.91% |

Test families were excluded from fitting and calibration, but checked repeatedly
during development. These are synthetic regression checks, not a blind study of
unseen applications. They do not establish a universal performance improvement.

Short CoreMark ablations attributed most improvement to the inlining head:
baseline 6213.05, inline-only 6331.54, schedule-only 6248.46, combined 6361.01.
These short samples are screening data; use the long-run result above for the
reported CoreMark score. Always expanding inlining or applying one fixed
scheduling strategy did not reproduce the combined gain.

## Artifacts and verification

- Model JSON, packed weights and training provenance are preserved locally as
  `../../artifacts/models/c908.{json,vlp,training.json}` (ignored by Git).
- Binary model SHA-256:
  `0a138a675419580ea408995fe2062bc55aa1ecff4af381d08da87ed08d1c28fc`.
- Final RISC-V compiler SHA-256:
  `c0187c4cd51d69428dc57ef25f2dbd51d51779c7b5bd67b37dfefeeb0257468a`.
- Raw local artifacts: `target/ml-policy-20261005/` and
  `target/ml-policy-complex-20261005/`; corresponding board directories are under
  `/root/veloc-riscv/`.
- `final-measurement.json`: seven long runtime pairs and the initial JSON-loading
  compilation experiment. `compile-final.json`: the final binary-model compiler
  measurements and per-unit object hashes.
- Final kernel measurements: simple corpus `combined/model-evaluation.json`,
  complex corpus `scoped/model-evaluation.json`. Earlier attempts are retained;
  the initial model regressed CoreMark and was rejected.
- Rerunning the documented training command reproduced both final neural heads
  exactly. `source-final.tar.gz` records the final source checkout used here.
- Workspace compilation and optimizer test-target compilation passed. Generated
  variants used IR verification; malformed target, feature-schema and layer
  contracts were rejected. No new unit-test suite was added.

The remaining limitation is model profitability, especially interacting
decisions and transfer to real applications. Larger gains require richer
measured decision opportunities and training on complete optimization trajectories;
more inference alone does not supply them.
