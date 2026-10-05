# Learned profitability models: design and evidence

## Research informing this design

Reviewed on 2026-10-05. The recommendations below are engineering choices for
Veloc, not claims that the cited results have been reproduced on K230.

| Source | Relevant finding | Consequence for Veloc |
| --- | --- | --- |
| [LLVM MLGO interfaces](https://llvm.org/docs/MLGO.html) | Separates pass features/decisions, model runners, tracing and external training infrastructure. | Keep legality in passes and use one decision contract for experimental controls and deployed inference. |
| [ProGraML, ICML 2021](https://proceedings.mlr.press/v139/cummins21a.html) | Graph representations expose program semantics and improve learned data-flow reasoning and downstream tasks. | Learn from structural relationships; investigate a shared graph teacher rather than program-name or opcode-count memorization. A GNN alone is not proof of performance transfer. |
| [Group-shift robustness, ICLR 2020](https://arxiv.org/abs/1911.08731) | Worst-group generalization depends on regularization; minimizing worst training loss alone is insufficient. | Balance families, retain regularization/early stopping, and measure poor-family outcomes on separate data. Our tail-risk loss is an engineering choice, not a reproduction of Group DRO. |
| [DomainBed, ICLR 2021](https://arxiv.org/abs/2007.01434) | Model selection changes domain-generalization conclusions; carefully implemented ERM is a strong baseline. | Separate checkpoint selection from calibration and final evaluation; compare robust/ranking objectives with balanced regression using identical data. |
| [CITROEN, IPDPS 2025](https://ieeexplore.ieee.org/document/11078537/) | Uses compilation statistics with Bayesian optimization, performance predictions and uncertainty to reduce phase-ordering search. | Expose compiler observations independently of the policy; prioritize informative measurements. |
| [TCL, April 2026 preprint](https://arxiv.org/abs/2604.12891) | Combines representative/diverse sampling, uncertainty and continual distillation for tensor-program optimization across hardware. | Use explicit hardware features and retain diverse measured programs when adapting to another target. Its reported sampling reduction is not a budget guarantee for a general-purpose compiler. |
| [CATBench, revised April 2025](https://arxiv.org/abs/2406.17811) | Exposes evaluation fidelity, multiple objectives and surrogate versus hardware tasks through a common interface. | Keep evaluator identity, objective, fidelity and measurement protocol in every dataset. Short and long runs should be distinguishable. |
| [GRANITE, IISWC 2022](https://charithmendis.com/assets/pdf/22-iiswc-granite.pdf) | Uses instruction dependency graphs and shared representations with microarchitecture-specific decoders to predict basic-block throughput. | A graph-based offline teacher is a reasonable future experiment. Shared representations do not establish zero-shot accuracy on unseen CPUs. |
| [COMET, MLSys 2024](https://proceedings.mlsys.org/paper_files/paper/2024/hash/eb261df4322a8bd0a73093c4d8a0d02d-Abstract-Conference.html) | Examines explanations and errors of neural and simulation-based basic-block cost models. | Compare a learned predictor against a credible analytical baseline; a neural model is not automatically the more accurate choice. |
| [LLVM pipeline study, June 2026 preprint](https://arxiv.org/abs/2606.31238) | On its PolyBench experiments, pass effects are non-monotone and IR instruction count is an unreliable runtime predictor. | Keep whole-program validation and do not label a reduced instruction count as a runtime improvement. |

The TCL sampling equations use dispersion of predicted values across different
programs. That dispersion is not automatically epistemic uncertainty about one
prediction. Our acquisition tool instead uses disagreement between independently
seeded models, plus diversity and random exploration. This is still an uncalibrated
heuristic; it must be compared with uniform sampling at the same measurement cost.

## Data collection should not execute every candidate

The intended loop is:

```
legal candidate transformations
    -> cheap static observations / estimates
    -> offline teacher predicts benefit and disagreement
    -> diverse, budgeted acquisition
    -> short paired hardware runs
    -> longer runs for close or promising decisions
    -> measured validation of the complete deployed policy
```

A static evaluator can operate entirely on compiled artifacts. A hardware evaluator
runs the program. They share the command/result interface, but their labels remain
different. Estimator version, objective and fidelity are part of the report.

A representative baseline execution can supply block/loop frequencies once, then
many candidates can be ranked using frequency-weighted local estimates. This
amortizes profiling rather than rerunning every candidate. Reusing frequencies
requires a valid mapping after CFG changes/inlining; a stale profile must not be
treated as exact. Local cycle estimates also do not capture all interactions
between regions, so final policy validation remains a whole-program measurement.

Start static scheduling estimates from dependency critical paths, resource pressure,
issue width, final spill/reload counts and final instruction count/size. For loops,
include recurrence bounds and execution frequencies where available. These are
features or imperfect estimates, not ground-truth rewards. There is no justified
universal conversion from these numbers to whole-program elapsed time.

Even [llvm-mca](https://llvm.org/docs/CommandGuide/llvm-mca.html) does not predict
cache hits/misses; its load/store unit does not model the cache hierarchy. A generic
CPU descriptor containing register capacity and execution resources has the same
limitation. Branch prediction, data locality, calling patterns and instruction
cache effects need measurements, profile information or an explicitly calibrated
model. CoreMark's score must ultimately be measured on the board.

## Current implementation

- Pass-owned `DecisionSchema`: stable decision name, version, ordered features,
  ordered actions and scope. Changing meaning or order requires a schema version
  change. New decisions require no inference-library enum or training branch.
- `Context`: numeric properties derived from the selected target description.
  Architecture/CPU/tuning names are provenance labels, optionally restricted by
  deployment configuration; they are not neural input IDs or hardcoded dispatch.
- Immutable model weights plus reusable, pass-local sessions. Session deviation
  limits are identical for learned models, fixed controls and probes.
- Feed-forward networks with model-defined dimensions/depth and explicit
  activations. Packed deployment contains only the selected networks; offline
  ensembles are not loaded by the compiler.
- Capacity grids compare widths/depths with the same data and regularization.
  Validation selects architecture/seed; calibration sees only that selection.
  Reports preserve learning curves, decision regret, parameter/MAC counts and
  training-label coverage. Dropout is removed by evaluation-mode export.
- Generic build, evaluation and training manifests. No benchmark command, CPU
  affinity, target triple, action count or feature width is inferred from a name.
- Scale-relative pass features expose growth, specialization, parallelism and
  resource pressure. Sizes remain available, but no program or CPU ID is a neural
  input. Source-order peak pressure reuses the scheduler's allocation-class model.
- Optional estimated-label pretraining followed by measured-label fine-tuning.
  Family-balanced normalization/support uses training data; checkpoints use
  validation families; abstention uses separate calibration families.
- Family/case balancing and a mean-plus-tail family loss reduce domination by
  large programs. Regression also learns relative action ranking weighted by
  measured benefit. Regularization and early stopping remain necessary.
- Calibration reports ungated behavior, per-family benefit/regression, changed
  binaries and cheaper fixed-action baselines. Heads need benefit in multiple
  families and meaningful coverage. This gate is an experiment safeguard, not a
  statistical guarantee or proof of whole-program benefit.
- Offline ensemble acquisition combines predicted benefit/disagreement, feature
  diversity and exploration. It favors underrepresented families and accounts for
  already measured contexts. Only training programs are acquisition targets.
- Frozen whole-program builds and a separate generalization report bind model
  hashes, reject development overlap and probe-only evidence, and report a
  family-cluster interval plus worst-program regressions. No weights are changed
  by this reporting tool.

This is not yet a GNN, an ONNX runtime, a calibrated uncertainty model, a dynamic
CPU simulator or a full multi-fidelity Bayesian optimizer. The explicit contracts
and evaluator protocol allow those experiments without embedding benchmark logic
in passes. A larger graph/sequence teacher should first justify its additional
data and inference cost; a small distilled student remains suitable for ordinary
compilation if it preserves decision quality.

## Evaluation requirements

1. Split whole program families, including inputs and translation units, into
   training, validation, calibration and final test partitions before collecting
   labels. Enforce the same split across reports and target datasets. Exact source
   hashes are a leakage check, not a substitute for grouping related algorithms.
2. Compare uniform acquisition with model-directed acquisition under equal
   hardware-time budgets. Count compilation, evaluation and training separately.
3. Compare measured-only training against estimated pretraining plus measured
   fine-tuning. Keep the same measured samples and independent final test programs.
4. Evaluate the complete policy, not the sum of isolated intervention rewards.
   Feature-matched probes may affect multiple identical sites; report that scope.
5. For hardware transfer, hold out an entire physical target, calibrate using an
   explicitly reported budget, then measure unseen program families. A configurable
   runtime is not evidence that a model trained on one CPU generalizes to others.
6. Freeze weights before final runtime, code-size and compilation-latency runs.
   Report regressions and confidence intervals, including cases where the model
   abstains. Preserve checksums, binary hashes, failed measurements and provenance.

The existing K230 results belong to the format-1 experiment. They do not validate
the new feature contract, acquisition procedure or model-training protocol.

## What would demonstrate useful generalization?

The historical format-1 result improved the training subset but produced no gain
on its seven held-out real programs. That is evidence to improve the learning
problem, not evidence of successful transfer. Previously inspected programs stay
development data in subsequent experiments, even after changing schemas/splits.

The intended long-term predictor scores the **effect of a legal candidate given
program structure, resource limits and available profiles**. A shared graph
encoder can represent dependency/control/data-flow relationships and an action
encoder can describe a candidate's changes. Hardware descriptors condition costs.
A larger offline teacher can supply representations or candidate rankings to a
small student, but both must be judged on fresh families. This teacher/student
design is not implemented by adding aggregate ratios to the current MLP.

Do not force an invariance that is physically false: doubling working-set size
may cross a cache boundary, and rearranging dependent instructions is not a
semantics-preserving augmentation. Collect real size/input variations and keep
them in their family's partition. Synthetic programs can cover structural gaps
but cannot stand in for independent real-program evaluation.

Next measurements should compare the same-data heuristic, best fixed strategy,
balanced regression, and the ranking/tail-risk policy. Report family gains,
regressions, intervention coverage, whole-program runtime, compilation latency
and failures. A model that abstains everywhere has not demonstrated useful
generalization. Data from one physical K230 also cannot establish unseen-CPU
transfer, regardless of how configurable the feature contract is.
