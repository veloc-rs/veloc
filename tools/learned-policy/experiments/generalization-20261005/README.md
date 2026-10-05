# Generalization workflow check — 2026-10-05

This was a development/integration run. All programs had been inspected in earlier
experiments; none is a fresh blind test. No generalization gain was established.

| Measurement | Result |
| --- | --- |
| Physical target | K230, serial execution on CPU 0 |
| Programs / intervention contexts / alternative actions | 9 / 50 / 184 |
| Hardware rounds / warmups | 5 / 1 |
| Program validation failures | 0 |
| Scheduling training / validation / calibration contexts | 11 / 9 / 22 |
| Scheduling network | 31 → 32 → 16 → 5 |
| Calibration families benefiting from ungated predictions | 0 of 5 |
| Exported model heads | 0 |
| Whole-policy comparisons | 9; all baseline-identical and valid |

Inlining lacked sufficient independent training families. Scheduling failed the
family benefit/coverage criteria and the fixed-strategy comparison. The exported
empty model deliberately retains the compiler's heuristics.

Host cross-compilation and native K230 compilation produced identical JFDCTINT
objects and feature traces. Object SHA-256:
`336a334d3ad86cd782b2337bcd80f4bf6a89cd405f5469c40a4674b81cc28896`.
Inlining used schema 2 with 23 features; scheduling used schema 3 with 20 features.

Local raw reports are preserved in the Git-ignored
`../../artifacts/results/generalization-20261005.json` and repository
`target/policy-generalization-20261005/` directory. Source/protocol details and
model/dataset hashes are recorded there. They are not bundled into a checkout.

Use [the current workflow](../../README.md) for new experiments. Recollecting
these same benchmarks does not make them unseen programs.
