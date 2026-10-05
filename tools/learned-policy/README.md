# Learned optimization policies

`veloc-policy` runs offline-trained networks without Python or a training service
inside the compiler. Models are opt-in with `veloc-c --policy MODEL.vlp`.
The same policy can be passed to the optimizer/code generator APIs; the Wasm
module pipeline forwards `CodegenOptions.policy` to both.

See [the design notes](docs/design.md) for the research, architecture and remaining limitations.

## Directory layout

```text
scripts/       Reusable build, measurement, training and evaluation tools
examples/      Versioned experiment configurations
docs/          Design and research notes
experiments/   Experiment sources, protocols and reviewed result summaries
artifacts/     Local generated data and models (ignored by Git)
```

Keep raw samples, traces, compiled programs, training reports and model exports
under `artifacts/` or the repository's ignored `target/`. Keep reviewed findings,
source scripts and reproducible configurations in version control. Do not ignore
all JSON files: experiment configurations are source inputs too.

## Contracts and hardware context

Passes declare their own `DecisionSchema`. Action zero retains the existing
heuristic. Inlining owns a caller scope; scheduling owns a function scope.
Each model head declares its optional `max_deviations` within that scope. The
same limit applies during probes and deployment. Legality remains in the passes.

`--policy-trace decisions.jsonl` writes a header containing all registered
contracts and the selected target's numeric context, followed by observations.
Names and paths are not model features. Register capacities, issue width and
resource information come from the target description, without a CPU-name switch.
The descriptor is intentionally limited; it is not a complete microarchitecture.

Inlining schema 2 includes specialization fractions and callee/caller growth
ratios. Scheduling schema 3 includes dependency density, exposed parallelism,
issue/resource bounds relative to the critical path, and source-order pressure
relative to register-class capacity. Absolute size features remain available for
size-sensitive costs. Old feature contracts must be recollected, not relabeled.

Format 2 stores a `decisions` map, `context_features` in input order, and optional
`requires` label restrictions. Every head contains a contract and an advisor.
Networks store normalization, a measured support domain, and dense layers with
explicit `linear`, `relu` or `tanh` activations. Depth, width, input count and action
count come from the file/contract. Unsupported contexts use the existing heuristic.
The loader checks contracts, dimensions, finite values and a 64 MiB file limit.

## Build and measure a dataset

Start from [examples/experiment.json](examples/experiment.json). Its paths and
compiler/linker arguments are experiment inputs: adapt them to your sources,
toolchain and target. Programs must provide a reproducible result validator.
Preprocess sources and build shared benchmark adapters before collecting data.

Commands are argument arrays, never shell strings. `{source}`, `{output}`,
`{trace}` and `{policy}` are supplied for compilation; `{objects}` expands into
linker arguments. The compile template must emit the requested policy trace and
consume the supplied policy. Ordinary variables are declared in the manifest.

```sh
python3 tools/learned-policy/scripts/build_probes.py experiment.json \
  --output target/policy-data --limit 64
```

The builder compiles a baseline and alternatives derived from trace contracts.
It preserves whole-program groups and samples contexts before looking at timings.
An intervention matches a feature context in one translation unit; identical
contexts may occur more than once. It does not claim to identify one dynamic call.

Copy the resulting dataset directory plus `measure_probes.py` and `data.py` to
the execution host. Run there, with affinity only if requested:

```sh
python3 measure_probes.py policy-data/manifest.json \
  --evaluator runtime --output runtime.json --affinity 0 --rounds 9
```

The generic runner takes command arguments, metric extraction, checksums and
validators from the evaluator configuration. It alternates paired runs, excludes
warmups, shares timings for identical artifacts/inputs, saves raw samples and
binary hashes, and records errors without turning failed actions into zero gain.
`--selection batch.json` evaluates a selected list of context IDs. `--splits test`
is explicit; training never consumes test results.

An evaluator has a name, `fidelity` (`measured` or `estimated`), `metric`, direction
(`min` or `max`), a command and result extraction. An estimated evaluator can invoke
an external static analyzer on `{artifact}` without executing the program. For
example, a calibrated analyzer can return `estimated_cycles=...` and use regex
extraction. Its version/configuration should be included as evaluator metadata.
Short/long hardware runs can be separate evaluator names with different commands
and budgets. Estimates are never automatically interpreted as measured runtime.

## Training and acquisition

```sh
python3 -m venv target/policy-venv
target/policy-venv/bin/pip install -r tools/learned-policy/requirements.txt
target/policy-venv/bin/python tools/learned-policy/scripts/train.py runtime.json \
  --metric elapsed_seconds --hidden 64 32 16 --output target/policy-model/model.json
target/policy-venv/bin/python tools/learned-policy/scripts/pack.py \
  target/policy-model/model.json target/policy-model/model.vlp
```

Input width and action count come from the dataset. Training balances program
families, then program/input/target cases within each family. Normalization uses
the same weights. Regression combines log-speedup error with benefit-weighted
action ranking; the objective includes the mean loss of the worst-performing
fraction of families. Set `group_risk_weight=0` and `ranking_weight=0` for a
balanced regression ablation. These choices need empirical comparison; they do
not guarantee improvement on arbitrary new distributions.

Use four disjoint family partitions:

| Partition | Permitted use |
| --- | --- |
| `train` | Weights, normalization and support points; optional estimated pretraining. |
| `validation` | Checkpoints and seeds. |
| `calibration` | Support radius and whether a candidate head merits evaluation. |
| `test` | Frozen whole-program evaluation only; never training or acquisition. |

Family, program identity and exact source-bundle digests cannot cross partitions,
including across reports/targets. `group` must identify an algorithm/program
family across suites: related DCT implementations do not become independent
because one comes from a different benchmark suite. Digest checks do not detect
arbitrary code clones. Dataset construction still needs a reviewed family map.

The default training gate requires two training and two validation families.
Calibration requires five families, benefit in at least two, at least 5% weighted
changed-binary coverage, and no family mean regression exceeding 1%. The learned
head must beat the heuristic and fixed actions satisfying the same criteria.
These configurable thresholds are experimental criteria, not statistical
guarantees. Reports include ungated predictions, coverage and each family's
reward so fallback cannot be confused with useful generalization. An empty
policy is a legitimate outcome. A nonempty output is only a **candidate** until
the complete frozen policy has been evaluated.

To pretrain on estimates, add their measurement report and explicitly name the
surrogate metric:

```sh
target/policy-venv/bin/python tools/learned-policy/scripts/train.py estimates.json runtime.json \
  --estimated-metric estimated_cycles --metric elapsed_seconds --output target/policy-model/model.json
```

Measured training, validation and calibration are required to export a head. The
trainer never optimizes a deployed head solely against a simulator's scores.
`--config training.json` accepts `defaults`, per-name `decisions` overrides and
optional `requires` label restrictions. Available defaults are declared in
`train.py`; examples include `hidden`, `activation`, `objective`, `seeds`,
`epochs`, `pretrain_epochs`, `support_points` and `radii`. No CPU name or
benchmark-specific option is part of the trainer.

### Compare network capacity

`architectures` supplies alternative hidden-layer widths; `hidden` supplies a
single architecture when no grid is configured. Use the same data, seeds and
regularization to compare capacity:

```sh
target/policy-venv/bin/python tools/learned-policy/scripts/train.py runtime.json \
  --metric elapsed_seconds --config tools/learned-policy/examples/capacity.json \
  --output target/policy-capacity/model.json
```

The example ranges from `32 → 16` to `512 → 256 → 128 → 64`. Each trial records
parameter count, dense multiply-accumulates per inference, training/validation
curves, and ungated decision quality. Architecture and seed selection use only
validation loss; calibration is run once for the selected architecture.
`max_macs` optionally limits the candidate grid using an explicit inference
budget. It is a work estimate, not a measured latency or a CPU-specific limit.

`dropout` is training-only; exported layers use evaluation behavior without
dropout or sampling. Training reports also record how many contexts contain
positive/negative alternatives and how many input features vary. A larger model
cannot learn profitable choices from a dataset containing only zero/negative
intervention rewards. More layers also cannot recover dependency information
absent from aggregate inputs. Capacity and representation are separate experiments.

The `.training.json` report records dataset hashes, source metrics, target
contexts, program groups and calibration. `.ensemble.json` retains independently
trained members for offline acquisition:

```sh
target/policy-venv/bin/python tools/learned-policy/scripts/select_probes.py \
  target/policy-data/manifest.json --ensemble target/policy-model/model.ensemble.json \
  --observed runtime.json --count 32 --output target/policy-data/batch.json
```

Without `--ensemble`, selection uses feature diversity and exploration for a
first batch. Selection considers training programs only, favors families with
fewer measured contexts, and measures diversity relative to previously observed
contexts as well as the current batch. Member disagreement is
an acquisition heuristic, not a calibrated error bar. This process reduces the
number of contexts submitted for measurement; its benefit relative to random
sampling still requires evaluation at equal budgets.

## Frozen whole-program evaluation

Freeze the model and evaluation criteria before measuring fresh program families.
Do not reuse a previously inspected/tuned benchmark as a blind test by changing
its split label. Include multiple inputs where relevant; numeric source/target
features alone cannot reveal runtime input distributions.

```sh
python3 tools/learned-policy/scripts/build_probes.py held-out-experiment.json \
  --policy target/policy-model/model.json --output target/policy-evaluation
# Copy artifacts and runner to the execution host, then run there:
python3 measure_probes.py policy-evaluation/manifest.json \
  --splits test --evaluator runtime --rounds 15 --output held-out.json \
  --environment physical-machine-and-configuration
# Back on the training host:
target/policy-venv/bin/python tools/learned-policy/scripts/report_generalization.py \
  held-out.json --training target/policy-model/model.training.json \
  --output target/policy-model/generalization.json
```

`--policy` builds all translation units with the complete model, including
interactions between passes. The measurement records the frozen model hash.
Use the exact JSON output associated with the training report for this evaluation;
packing afterwards changes the file hash, although it preserves the weights.
The report rejects probe-only evidence, a different model, reused development
families/programs/source bundles, and duplicate observations. It reports whole
program speedups, failures, changed binaries, worst program regression, and a
family-cluster bootstrap interval. The interval is conditional on the sampled
families and is not a guarantee for all programs or all timing conditions.

The report evaluates runtime generalization across program families. Compile
latency and code size require separate measurements; the instrumented build
commands are not compiler-latency benchmarks. Cross-input and physical-hardware
transfer also require separately preregistered experiments. Selecting `generic`
and `c908` on one board does not supply two physical CPUs. Keep those claims
separate, and retain held-out families when adapting a model to new hardware.

## Historical measurements

The [capacity comparison](experiments/capacity-20261005/README.md) expanded the
development dataset to 18 programs and compared four network sizes. Increasing
scheduling parameters from 1,637 to 189,189 reduced validation loss, but none of
the selected heads passed calibration. It establishes no generalization speedup.

The [generalization workflow check](experiments/generalization-20261005/README.md) used
nine previously inspected development programs: 50 contexts and 184 actions
passed K230 result validation. The new training/calibration protocol exported no
heads. Nine whole-program comparisons consequently retained identical baseline
binaries. Host/board JFDCTINT objects and feature traces matched. This validates
the implementation path; it does not demonstrate useful generalization.

The format-2 integration check exercised three
programs, 28 contexts and 108 alternative actions on K230, plus packed-network
inference on host and board. The small training batch accepted no deployed heads;
this is interface/correctness evidence, not a new speedup or generalization result.

The [format-1 results](experiments/k230-20261005/RESULTS.md),
[real-program results](experiments/k230-20261005/RESULTS-REAL.md) and
[protocol](experiments/k230-20261005/REAL_PROGRAMS.md) describe the previous K230
experiment. Its specific scripts are archived under `experiments/k230-20261005/`.
Historical model files and raw JSON reports are local data under
`artifacts/models/` and `artifacts/results/`; they are not part of a fresh checkout.
There is no format-1 compatibility path in the new
compiler. Retrain with format-2 traces before deployment; the old scores do not
establish improvements or cross-CPU generalization for this design.
