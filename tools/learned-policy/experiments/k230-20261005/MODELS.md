# Format-1 research artifacts

These weights and their training reports are retained unchanged locally under
the Git-ignored `../../artifacts/models/`, with their original provenance and
K230 measurements. They require the compiler used in those experiments; the new
format-2 runtime intentionally does not load them.

Generate format-2 traces and retrain using the current manifest workflow. A
retrained model needs its own whole-program and target validation. Changing the
serialization version alone does not supply the new hardware features or prove
that an old model generalizes.
