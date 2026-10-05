# Real-program policy evaluation on K230

This is the historical format-1 workflow. Its scripts now live in
this directory and require the original compiler snapshot and directory layout.
Commands below record that historical layout. For the current generic format-2
workflow, see [the tool README](../../README.md).

This extends the synthetic kernels and CoreMark with pinned programs from
[Embench](https://github.com/embench/embench-iot) and
[BEEBS](https://github.com/mageec/beebs). Sources are fetched into `target/`, with
their original licenses; benchmark source files are not vendored or rewritten.

## Experimental contract

- Embench revision: `09c2ed8c3b7008c95d08b038de4a3f6dc103ed70` (19 attempted programs).
- BEEBS revision: `049ded9f3aeb5591f553879d3a0376b8614e9422` (31 attempted programs).
- Whole-program training, validation and test partitions are fixed in
  `embench.py` before performance measurement. Related Embench/BEEBS duplicates
  are not added as independent validation programs.
- Builds that fail and programs without a successful upstream verifier are
  reported explicitly and excluded from speed aggregates and training.
- Both compilers consume identical `.i` files, RV64GC/Zba/Zbb, LP64D. LLVM uses
  `-O3 -ffp-contract=off`; Veloc uses `-O1 --cpu c908`. No LTO. Both link the same
  separately compiled Linux timing adapter and libc.
- A minimal LP64 header facade replaces libc header extensions unsupported by
  the C frontend. GNU attributes (including `noinline`) are removed for **both**
  compilers. `__SIZEOF_INT128__` is undefined for both so upstream portable
  fallbacks are selected. These are adapted compiler experiments, **not official
  Embench or BEEBS scores**.
- The BEEBS adapter reinitializes each iteration, following BEEBS `support/main.c`.
  Embench uses its own repeated benchmark body and cache warmup. Verification is
  outside the timed loop. Every measured process must pass the upstream verifier;
  integer return checksums must also agree across compiler/policy variants.
- Target execution is serial on CPU 0. Each program's repetition count is
  calibrated, then held identical across variants. Variant order changes between
  rounds, with an unrecorded warmup round.
- Byte-identical executables are grouped by SHA-256 and run once per round;
  aliases share their samples. This prevents filesystem placement and timing
  noise from appearing as an optimization gain for unchanged machine code.
- Probe experiments change one structural context in one translation unit.
  Contexts are sampled by a stable hash, capped before observing timings.
  Identical executables are measured only once. Failed or missing measurements
  invalidate the entire training context, rather than supplying a zero reward.
- Test programs supply no probe labels or model-selection timings. Freeze a model
  before running their performance evaluation.
- Native compilation measurements include process startup and model loading,
  exclude preprocessing/linking/tracing/IR verification, and check each object
  against the locally built object hash. A native object that differs is recorded
  in `host_differences`; link and validate that variant separately before accepting
  its compilation measurement. Hash instability across repeated native builds is
  a hard failure.

## Reproduction

The builder checks that source checkouts are clean at the pinned revision.
Paths for Clang, the cross sysroot and Veloc can be overridden with CLI options.

```sh
python3 tools/learned-policy/embench.py target/embench-policy --suite embench
python3 tools/learned-policy/embench.py target/beebs-policy --suite beebs
```

Copy each `target.tar.gz` to a separate directory on K230, extract it, then run
the initial training/validation measurements (test programs are excluded by
default):

```sh
python3 measure_embench.py . --rounds 9
```

Copy `timings.json` back to its local experiment directory, build intervention
executables, transfer the new archive, and measure probes on K230:

```sh
# Local; use --suite beebs for the second directory.
python3 tools/learned-policy/embench.py target/embench-policy \
  --probes --probe-limit 32

# K230, serially, after extracting the new archive.
python3 measure_embench.py . --probes --rounds 3
```

Copy `probes/times.json` back. The existing `train.py` accepts these directories
alongside the synthetic data roots. Keep a synthetic root first so its trace
header supplies the feature schema. Pack candidates with `pack.py` and build
their executables with:

```sh
python3 tools/learned-policy/embench.py target/embench-policy \
  --evaluate --mode tuned --model path/to/candidate.vlp
```

Select using training/validation programs only. After freezing the model and
transferring its executables, measure all partitions and then compilation:

```sh
# K230
python3 measure_embench.py . --splits train validation test \
  --modes baseline learned tuned llvm --rounds 15 --seconds 0.25 --report final.json
python3 measure_compile.py . --compiler /path/to/veloc-c \
  --policy /path/to/candidate.vlp --rounds 9
```

`manifest.json` records input, compiler, model and object hashes; measurement
reports retain individual samples and validation failures. Existing `times.json`
probe results are resumed, so use a fresh experiment directory when changing
compiler or input sources.
