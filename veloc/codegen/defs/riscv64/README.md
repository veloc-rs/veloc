# RV64 instruction selection

## Rule layout

`module.spec` is the single source for all target artifacts. Each `lower/*.spec`
file owns its instruction definitions, encoding, assembly and selection rules.
`common.spec` holds shared types, host interfaces and instruction templates.

`lower.spec` imports these families in priority order. Optional extension patterns
precede base-instruction fallbacks:

| File | Responsibility |
| --- | --- |
| `lower/i.spec` | Integer operations, immediates, integer memory access, control flow and ABI transport |
| `lower/m.spec` | Integer multiply, divide and remainder |
| `lower/f.spec` | Single-precision arithmetic, memory and conversions |
| `lower/d.spec` | Double-precision arithmetic, memory and conversions involving f64 |
| `lower/zba.spec` | Shift/add address generation and zero extension from i32 |
| `lower/zbb.spec` | Rotates and byte/halfword extensions |

The backend uses LP64D and currently requires I/M/F/D. `generic` enables this
baseline; `c908` additionally enables Zba/Zbb. Disabling a required baseline
extension is rejected at target construction. Other extensions advertised by
K230, such as V and Zbs, do not yet have selection rules here.

Target instruction `requires` declarations automatically become matcher feature
checks. Optional rules therefore fall back to the ordinary I sequences when
their extension is disabled. There is no CPU-name check inside the selector.

Immediate rules follow `Constant` definitions and use shared `fits_signed` /
`fits_unsigned` guards. An immediate that does not fit retains the register
form. Shifts use five bits for i32 and six for i64; arithmetic-shift encoding
sets the high immediate bits in Spec. RV64 i32 results remain sign-extended to
XLEN, including `addiw` and word shifts.

Comparison/branch rules fold i32/i64/pointer comparisons into BEQ/BNE/BLT/BGE/
BLTU/BGEU. Signed and unsigned i32 comparisons both rely on that sign-extension
invariant. Branch emission inverts the condition around the existing long-range
jump. A matched comparison with other users remains available to those users.

## Running on K230

Build the Linux RISC-V executable locally with a configured cross linker and
sysroot, then copy it and the Wasm input to the board. On the board:

```sh
./veloc-wasm coremark.wasm --strategy jit --opt-level 1 --cpu generic --print-stats
./veloc-wasm coremark.wasm --strategy jit --opt-level 1 --cpu c908 --print-stats
./veloc-wasm coremark.wasm --strategy jit --opt-level 1 --cpu c908 --cpu-features=-Zbb
./veloc-wasm coremark.wasm --strategy jit --opt-level 1 --cpu c908 --cpu-features=-Zba
```

Run performance samples serially after uploads and correctness tests finish.
File transfer to the single Linux hart measurably interferes with this workload.

## K230 measurement, 2026-09-30

Board: C908, Linux 6.6.36, one online hart. `/proc/cpuinfo` reports
`rv64imafdcv_zicbom_zicboz_zicntr_zicsr_zifencei_zihpm_zba_zbb_zbs_svpbmt`.
CPU frequency was not fixed or independently measured.

Input: the existing compatible `coremark-baseline.wasm`, 33,990 bytes,
SHA-256 `e71e358234e39803a1d27961439d924a69c836dd81c8670bfff7dbb82c097bbe`.
It executes 11,000 iterations per run at Wasm optimization level O1.
The other board input, `coremark.wasm`, contains unsupported `ReturnCall`
operators and was not used. The baseline executable was built from the working
tree immediately before this selection change, including preceding uncommitted
work.

Three serial runs per configuration, interleaved baseline / generic / C908:

| Configuration | Runs (iterations/s) | Median | Change | Selected LIR instructions | Emitted code bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | 628.56, 629.54, 630.61 | 629.54 | +0.00% | 29,325 | 549,936 |
| generic | 721.03, 725.03, 722.55 | 722.55 | +14.77% | 25,230 | 525,780 |
| c908 | 732.94, 733.28, 733.41 | 733.28 | +16.48% | 25,230 | 522,008 |

Instruction and code-size counts cover the complete compiled module, including support functions. All nine runs passed CRC validation.

Raw logs and executable snapshots are in `target/k230-isel-tuning/` locally and
`/root/veloc-riscv/isel-tuning/` on the board. Baseline executable SHA-256:
`5438920ee0e7cdb51b4278bc4f50ddae2acb460acd5fa9ff44566ad77c5475ea`;
final executable SHA-256:
`144c9ceae537601740d529f6655a4c5ca10aca2665ab76706e9e68d55806e37e`.


The initial generic sample overlapped a test-binary upload and is excluded.
Exploratory runs with immediate selection but comparison fusion disabled scored
664.89 (generic) and 667.44 (C908) iterations/s. Full rules with only Zba enabled
scored 728.29; with only Zbb enabled, 732.22. These are single-sample ablations,
not repeated performance estimates.

Correctness checks:

- CoreMark validates seed `e9f5`, list `e714`, matrix `1fd7`, state `8e3a`,
  and final `33ff` CRCs.
- `riscv64_selection` compares JIT results against the interpreter at O0/O1,
  with generic, C908, and C908 disabling Zba, Zbb, or both. It covers immediate
  boundaries, shift counts, sign extension, comparisons, rotates and shared uses.
- Existing JIT ABI and host-import regressions pass on the board.
- `cargo test -p veloc-spec -p veloc-bytecode` checks rule parsing, generated
  feature guards, and both production target descriptions.

This measures one Wasm workload on this board. It does not establish performance
for floating-point workloads or other RISC-V CPUs.
