# Native backend: executable baseline and next iterations

The current milestone is an executable, tested x86-64/System V path, not a claim
of better generated code than LLVM. The long-term objective is a generic,
declarative compiler infrastructure; correctness and measured tradeoffs come
before adding more search heuristics.

## Run it

From the repository root:

```sh
native_dir=$(mktemp -d)
CARGO_INCREMENTAL=0 cargo run -p veloc-codegen --example native -- \
  veloc/codegen/examples/sum.mir "$native_dir/sum.o"
cc veloc/codegen/examples/sum.c "$native_dir/sum.o" -o "$native_dir/sum"
"$native_dir/sum"
```

The result is `sum(100) = 5050`. `--no-opt` disables scheduling; `VELOC_DUMP_LIR=sum`
prints intermediate stages. This driver validates MIR before compilation and
emits an ELF object; it does not implement a JIT or a linker.

## Contracts

1. MIR translation produces typed generic LIR. Local stack slots have a symbolic
   `StackBase::Frame`, not an early dependency on a particular physical register.
2. Block arguments become edge copies, including cycle breaking and split edges.
3. Legalization checks expansions; ABI lowering places argument copies before
   calls and result copies after them, and reserves the maximum outgoing area.
4. ISLE selects target instructions. Calls explicitly expose ABI register uses
   and caller-saved clobbers, even though these operands have no encoding fields.
5. Operand constraints are materialized before scheduling. A move defines its
   destination; it must not pretend to read the previous destination value.
6. Scheduling preserves register dependencies and effect boundaries. Allocation
   uses the same cached CFG liveness, then inserts target-provided spill code.
7. Frame finalization saves modified callee-saved registers and restores them at
   returns. Emission rejects generic instructions or unresolved virtual registers.

## Scheduling driven by target definitions

An instruction or template may declare `(schedule 3)`: it promises a movable,
nontrapping, non-memory, non-control operation with estimated result latency 3.
Missing declarations are barriers. A concrete instruction can override an
inherited latency. The definition compiler rejects invalid/duplicate latencies,
opaque implicit dependencies, and scheduled branch/stack operands. Descriptions
remain trusted semantic contracts: the generator cannot infer safety from x86
encoding bytes alone.

`TargetInstMetadata` contains the generated schedule information. The scheduler
has no x86 opcode switch. Integer flag clobbers come from existing declarations;
flag consumers remain barriers. Within a region the final flag writer stays last
among flag writers, preserving the flags observed by following instructions.

Each region has at most 256 instructions, bounding compile-time work. A register
dependency DAG preserves RAW, WAR and WAW ordering, including physical and tied
operands. List scheduling weighs estimated readiness, critical-path height and
live virtual-register pressure separately for each register class. Memory, calls,
variable shifts and floating-point arithmetic currently remain barriers. These
are latency estimates, not a CPU port/throughput model.

## Simple allocator, explicit limitations

The replacement allocator is a global linear scan. Its intervals include CFG
live-in/out values, so loop-carried values remain live through blocks without
textual uses. CFG construction reads generated generic/target control descriptors:
conditional branches contribute targets and continue scanning, unconditional
transfers stop scanning, and returns/traps have no successors. Layout fallthrough
exists only when the block's executable instruction sequence allows it.
Operand, layout, semantic and selection changes invalidate dependent analyses.

Target instructions/templates declare `(flow Next|Branch|Jump|Return|Call|Trap)`;
omission means `Next`. Template inheritance and explicit overrides are supported.
Non-`Next` instructions cannot carry a movable `schedule` contract. Call
classification for allocation uses the same descriptor, not an opcode whitelist.

Conditional selection emits both true and false transfers; it does not assume
the false target is next in layout. Executable tests cover redirected edges and
loops. Extension encoding tests compare all 16×16 register pairs for six forms
against the system assembler, including REX byte-register distinctions.

Instruction input reads precede output writes. Separate positions allow a dying
input and a new output to share a register. Physical-register reservations and
call clobbers constrain allocation; failure to find a preserved register never
falls back to an unsafe caller-saved register. Under pressure the allocator can
evict a further-ending interval, spilling its entire lifetime. Spill temporaries
and load/store instructions are target hooks, not hardcoded in the algorithm.

There is no live-range splitting, rematerialization, stack-slot reuse or general
early-clobber support yet. Physical live ranges are conservatively enveloped per
block. Too few dedicated temporaries cause an error, not register corruption.
The current x86 frame implementation supports the tested scalar System V path;
it is not a complete SIMD or Windows ABI implementation. Typed continuations and
other currently rejected MIR operations still require dedicated lowering.

## Verification and measurements

```sh
CARGO_INCREMENTAL=0 cargo test -p veloc-codegen --test native
CARGO_INCREMENTAL=0 cargo test --workspace
CARGO_INCREMENTAL=0 cargo check -p veloc-codegen --no-default-features
CARGO_INCREMENTAL=0 cargo test -p veloc-codegen --release --test native \
  backend_benchmark -- --ignored --nocapture
```

Native tests compile MIR, emit ELF, link a C harness and execute it with scheduling
on and off. They cover loops, branches, stack loads/stores/addresses, direct and
indirect calls, eight-argument ABI permutations, integer spills across a caller
that clobbers registers, and floating-point values live across calls. Child
execution is time-limited. Scheduler tests also exercise anti-dependencies,
memory barriers, flag consumers and useful critical-path reordering.

A local release run compiling the sample `sum` function 200 times measured
2.75 ms without scheduling and 4.05 ms with scheduling; both emitted 124 bytes.
This includes the entire MIR-to-machine-code pipeline but excludes parsing and
Rust compilation. The simple loop does not establish a runtime benefit from
scheduling. These numbers are a baseline, not an LLVM comparison.

## Next architectural work

Generic LIR storage now comes from `lir/defs/generic.ops`, compiled by the shared
OpSpec frontend. Opcode-specific arity checks and generic control-flow facts are
generated. Target control descriptors are generated from ISLE and feed the same
CFG algorithm; there is no target fallthrough guess. Logical signatures/type rules
are shared with MIR, but machine schemas do not yet replace target legality or
precise memory/effect descriptions.

- Extend authoritative control descriptors to identify successor operand positions
  and precise memory effects. Separate encoding operands from register allocation
  constraints, including fixed, reused-input
  and early-clobber operands. This should remove duplicated handwritten lowering
  logic, not add an adapter for every existing representation.
- Preserve SSA longer; add segmented live ranges, copy coalescing,
  rematerialization and costed splitting only after an allocation checker and
  larger differential corpus are available.
- Extend target scheduling descriptions with resources/throughput and precise
  memory effects before attempting cross-memory or cross-block scheduling.
- Compare compile time, code size and runtime against LLVM on a pinned CPU and
  representative C/Wasm workloads. Keep regressions and unsupported cases visible.
- Continue offline semantic rule validation/synthesis. MIR OpSpec owns semantics;
  ISA definitions own machine constraints and costs. Neither should duplicate
  the other's facts in a large hand-maintained table.

## References informing the design

[LLVM's MachineScheduler interface](https://llvm.org/docs/doxygen/MachineScheduler_8h_source.html)
separates regions, dependency construction and selection strategy, and maintains
register pressure during scheduling. This implementation adopts a smaller
bounded version of that separation, not LLVM's complete scheduling machinery.

[The regalloc2 author's design overview](https://cfallin.org/blog/2022/06/09/cranelift-regalloc2/)
explains explicit operand constraints, precise live ranges and costed splitting.
It informs the next LIR constraint design; the current allocator is deliberately
a simpler whole-interval baseline, not an implementation of regalloc2.

[Optimal and Heuristic Min-Reg Scheduling Algorithms for GPU Programs](https://arxiv.org/abs/2303.06855)
studies minimum-register scheduling and heuristic gaps. It motivates measuring
register pressure alongside latency, not assuming longest-path priority alone
produces the best schedule. Its GPU results do not imply x86 performance gains.
