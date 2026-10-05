# Network capacity comparison — 2026-10-05

## Scope

This is a development experiment, not a blind generalization result. All programs
had been inspected in earlier work. Existing train/validation/calibration roles
were retained. Additional training programs were FDCT, RC4, LU decomposition,
Huffbench, Statemate, UD, Nsichneu and XGBoost; CNT added a calibration family.
Related DCT implementations share a family, as do the LU implementations.

All variants ran serially on K230 CPU 0, with one warmup and five measured rounds.
The new measurements cover 188 contexts and 720 alternative actions. Including
the earlier development dataset: **18 programs, 238 contexts, 904 actions, zero
program-verification failures**. These counts include baseline-identical actions.

Training uses the [capacity configuration](../../examples/capacity.json): four
architectures, three seeds each, 800 epochs, dropout 0.1 and weight decay 0.02.
Dataset, loss and regularization are shared across capacities. Architecture and
seed selection use validation loss; only the selected architecture reaches
calibration. Neither calibration nor test labels select the network size.

## Scheduling results

The original small dataset supplied 11 training contexts and no positive rewards
above the noise threshold. Additional measurements increased this to 173 contexts
from eight families, including 12 contexts with positive alternatives. There are
140 distinct input vectors; 20 of the 31 input features vary. This is still a
small dataset, and all hardware labels come from one physical board.

| Hidden layers | Parameters | Dense MACs | Best validation loss | Validation probe gain |
| --- | ---: | ---: | ---: | ---: |
| 32, 16 | 1,637 | 1,584 | 2.084e-5 | 0% |
| 128, 64, 32 | 14,597 | 14,368 | 1.839e-5 | 0% |
| 256, 128, 64 | 49,669 | 49,216 | 1.791e-5 | 0% |
| 512, 256, 128, 64 | 189,189 | 188,224 | 1.634e-5 | 0% |

Loss selects the largest network. Its ungated calibration probe reward is zero;
none of the six calibration families benefits. Raw-domain coverage is 93.3%, so
rejecting out-of-range inputs alone does not explain the absence of gains.
The scheduling head is not exported. Lower prediction loss did not produce
better optimization decisions in this experiment.

## Inlining results

Inlining has 12 training contexts from six families, with three positive contexts.
There are 11 distinct input vectors. Validation selects `34 → 256 → 128 → 64 → 3`
(50,307 parameters). Two calibration families have positive ungated probe rewards,
but MD5 regresses about 3.21%; the family-balanced ungated aggregate is −0.164%.
No calibration context passes all training-range bounds. Removing those bounds
would therefore still not establish an aggregate benefit. This head is also not
exported.

## Interpretation and artifacts

There is no deployed speedup from this run. Wider/deeper MLPs remain supported and
measurable, but this evidence does not justify using a larger network by default.
It also does not prove that capacity is useless on a richer dataset. Future work
should compare structural representations and candidate-effect inputs, broaden
training families, and measure complete frozen policies on genuinely fresh data.
Dense MAC counts are work estimates; no larger-model inference latency claim was
measured on K230.

Local generated files are in ignored `target/policy-capacity-20261005/`, with
matching board files under `/root/veloc-riscv/policy-capacity-20261005/`:

- `runtime.json` SHA-256:
  `921f29080aa5f13c1cd52b6df4cec6f98d72326c8a22027a532d5110ca22562f`.
- `calibration-runtime.json` SHA-256:
  `6c0835e828975c5cb0f474a797d449d842bf344309c1b770d9b4e3c50e7b42d5`.
- `model.training.json` SHA-256:
  `03eaf7310ac401983e0d7278fe6fe58b94b5511c21c2cecba88c0e9f162475d9`.
- `small-data.training.json` records the same capacity grid on the original
  11-context scheduling training set, before adding data.
- `experiment.json`, `calibration-experiment.json` and `capacity.json` record
  build/run commands, inputs, partitions and the training configuration.

To repeat the training on the local datasets from the repository root:

```sh
target/policy-venv/bin/python tools/learned-policy/scripts/train.py \
  target/policy-generalization-20261005/runtime.json \
  target/policy-capacity-20261005/runtime.json \
  target/policy-capacity-20261005/calibration-runtime.json \
  --metric elapsed_seconds --config tools/learned-policy/examples/capacity.json \
  --output target/policy-capacity-repeat/model.json
```
