# Historical K230 experiment

These are the benchmark-specific build/measurement scripts used for the original
format-1 policies and results. Their CPU, paths and benchmark conventions describe
that experiment; they are not the generic training API.

They require the original compiler snapshot and original working-directory layout.
The new compiler intentionally does not accept format-1 models. The weights and
reports in `../../artifacts/models/` and `../../artifacts/results/` are preserved
locally, ignored by Git. Reviewed findings remain in [RESULTS.md](RESULTS.md) and
[RESULTS-REAL.md](RESULTS-REAL.md). Do not use these scripts with a format-2 compiler or attribute
their recorded gains to newly trained models.

Use the tools in [../../scripts/](../../scripts/) with explicit experiment
manifests for new work. The commands recorded in the historical reports refer
to the original snapshot's paths and interfaces.
