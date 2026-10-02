# Lowering

The native pipeline keeps three responsibilities separate:

- MIR translation preserves the operation's semantics in generic LIR. Dense MIR
  value IDs index a `PrimaryMap<Value, Reg>`; lookup does not need hashing.
- Legalization expands unsupported generic instructions. Replacements are checked
  again before instruction selection, including in-place replacements and appended
  blocks. Rules return replacements in execution order. They must not silently
  mutate already-processed instructions elsewhere in the function.
- Instruction selection chooses target instructions using the existing ISLE rules;
  ABI and register constraints remain separate concerns.

ABI lowering precedes legalization. Register parameters, call arguments, call
results and return operands remain typed SSA values. ABI lowering records their
physical placement requirements and materializes stack accesses. Function
parameters have parallel incoming-location metadata; call/return constraints
refer to input/result occurrences. Selection preserves this contract without
replanning it. A libcall uses the same ABI planning path.

Generic computations never borrow a type from a physical endpoint. Legalization
reads `VRegData.ty` for every value, including ABI boundaries. A constrained
boundary can be accepted as legal, but a value rewrite must perform any required
ABI conversion before establishing its placement contract. Scalar `Copy` is an
identity rewrite in the spec and is eliminated by replacing SSA uses.

MIR-to-LIR lowering is handwritten Rust in `src/translate.rs`. Arithmetic maps
directly to OpSpec-generated LIR builders; comparisons, memory and control flow
use the same typed instruction interfaces. There is no build-time MIR lowering
rule inference or generated host adapter. OpSpec owns instruction representation
and validation, while the translator owns the semantic mapping.

Translation requires valid MIR; validation is a separate, optional stage, not a
partial per-opcode check inside lowering. Each instruction lowering appends its
complete LIR sequence in order. Static entry-block allocations reserve stack
slots when encountered; value/register identities are still allocated up front
to support forward references and loop edges.

Address calculations use the target pointer width (32 or 64 bits). `PtrIndex`
zero-extends narrower unsigned index bit patterns or truncates wider ones before
wrapping scale/offset arithmetic. Signed narrow indices must be explicitly
sign-extended in MIR; the immediate offset is signed.

## Machine SSA and allocation

Virtual values have one definition through legalization, selection and scheduling.
Non-entry block parameters remain SSA definitions; branches carry their edge
arguments even when a generic conditional or jump table expands to several
target branches. Register entry arguments remain function parameters until
allocation; stack arguments are defined by loads. Translation provides a fresh
ABI predecessor when the original entry has backedges.

Multi-instruction selection rules declare intermediates explicitly:

```text
(temp $bit $dst)
(emit (seq (X86Cmp64 $x $y) (X86Sete $bit) (X86Movzx8to32 $dst $bit)))
```

A temporary inherits its exemplar's type and register bank, but has a fresh
identity. It is allocated only after the rule's match conditions succeed.

Allocation consumes unchanged SSA instructions and produces operand locations,
spill/copy insertions and per-edge physical move plans. Materialization creates
edge blocks, resolves branch targets, removes edge arguments and block parameters,
and finally produces physical non-SSA instructions. Parallel-copy cycles use
recyclable stack temporaries; stack-to-stack moves use reserved scratch registers.
There is no pre-selection phi destruction or virtual-register parallel-copy pass.

## Operand placement

Machine spec metadata and per-instruction ABI constraints use one representation:
`OperandConstraint { operand: OperandRef, placement: Placement }`. Placements
restrict an occurrence to a register set, a fixed physical register, or reuse of
an input's location by a result. The current machine schemas read inputs before
writing results; destructive instructions state their reuse relation explicitly.
There are no pre/post-selection passes that replace constrained values with
physical registers for ordinary data, and no inference of generic constraints
from select rules. Non-renamable state has a separate lowering boundary below.

Global linear scan chooses a preferred home for each value. Fixed operand
occurrences reserve short read/write points, with the occupying SSA value
recorded so it can use that register itself. Call clobbers reserve the write
point; a fixed result supplies the definition for its return register. Different
uses do not intersect their requirements into one global register restriction.
Compatible homes are preferences; per-occurrence constraints are mandatory.

The operand planner selects local locations, preserves borrowed live registers,
and produces simultaneous input/output transfers. Multiple uses of one value
may occupy different ABI registers; reuse groups keep distinct SSA input and
result identities. Entry, instruction and CFG-edge transfers share the same
parallel-move resolver, with distinct before/after/edge execution boundaries.
Cycles save the widest required representation. Scratch borrowing preserves the
actual live type, and target transfer hooks select moves/spills using both the
width and physical register bank.

The optional machine SSA verifier checks definitions, dominance, edge contracts
and constraint references. Allocation additionally checks location requirements;
materialization removes function/block parameters and produces physical code.
The allocator still uses whole-range homes and spills. Live-range splitting,
block-frequency weighting and rematerialization are separate future work.

## Non-renamable state

Flags retain SSA identities through selection and scheduling. Target builders
record `VRegKind::State` when defining a result, using the
spec's state operand signature. Both ordinary fixed operands and state operands
use `Placement::Fixed`; the value category determines whether ordinary allocation
or state lowering handles preservation and recovery.
The category identifies a hardware unit; it
does not assert that the unit contains this value for the value's whole lifetime.
The selected-IR verifier checks every occurrence against its operand contract.
Ordinary register allocation rejects unresolved symbolic state.

`StateContents` is the shared transfer model for scheduling and state lowering.
Inputs read the old contents; all writes and clobbers invalidate overwritten
units, then symbolic results install their identities. Effects remain precise
per state bit, including instructions that write several bits together.

Scheduling retains its ordinary physical-dependency schedule as a fallback. An
additional candidate orders symbolic state lifetimes dynamically, allowing a
whole definition/use interval to move across another writer. It must consume
the right versions without recovery and preserve observable exit contents.
Explicit physical reads keep their original dependencies. The candidate is
selected only when its estimated pressure and completion cost are no worse.
Live-in states or conflicts that this local policy cannot handle fall back to
the ordinary schedule; they do not restrict what the IR can represent.

Required state lowering runs after optional scheduling. Its CFG worklist meets
available identities across predecessors, distinguishing an unvisited edge from
an unknown hardware value so preserved values can flow around loops. Safe
producer recomputation handles unavailable states before conversion to physical
operands. Block parameters and edge transfers still require explicit lowering;
this change does not introduce arbitrary flags Phi or ordinary flags spills.
Target-specific condition materialization and costed recovery alternatives are
future extensions; current recovery retains the safe-rematerialization contract.

## Memory representation

Translation receives an explicit target `DataLayout`. Pointer loads/stores use
its pointer size; other accesses require a fixed-size representation rather than
treating a scalable type's minimum size as an exact access width.

`DataLayout` lives in `veloc-types`; backends supply complete per-type layouts.
`layout_of` returns a `TypeLayout` containing the storage size and ABI alignment,
or `None` for an unknown layout. `alloc_size` includes tail padding and rejects
scalable or overflowing allocations. Spill slots and parallel-copy temporaries
use allocation size; memory accesses use storage size. Neither infers alignment
from size or treats a scalable minimum as a fixed size.

There are no target or language presets in `veloc-types`. The x86-64 backend,
interpreter and Wasm frontend own their respective memory representation tables.
Target-independent constant encoding is separate from these layouts. The memory
optimization pass requires an explicit `OptConfig::data_layout`; without one it
leaves memory operations unchanged. `PassManager::with_layout` supplies it.

MIR load/store alignment and volatility survive in the required `MemFlags`
field of generic and target memory instructions. Offsets and stack slots are
ordinary fields too. ABI, spill and frame lowering supply flags explicitly.
There is no instruction-indexed memory side table.

x86 legalization checks offset loads/stores against the signed disp32 range.
Larger displacements become an i64 constant plus a pointer addition, followed by
the original access with offset zero. The access keeps its ID and memory
fields; address calculation has no memory effects. Thus unsigned MIR offsets
above `i32::MAX` never silently turn into negative displacements. In-range
offsets retain the compact addressing form. Generic i8/i16 and pointer accesses
use their existing target load/store rules. Physical transfers preserve their
required widths. Narrow arithmetic legalization is still separate work.

`ptr-offset` emits a pointer-valued `PtrAdd` directly, without constructing
an integer-typed address and copying it back to a pointer.

ISA definitions declare `memory = { kind: Read, bytes: 8 };` and a required
`flags: MemFlags` parameter. Generation checks that exactly one such field is
present on every concrete memory instruction and absent from other instructions.
Direction and access width come from the instruction definition; generic
load/store width comes from its value type and the target data layout.

Selection rules pass `n.flags` to the target instruction directly. Address
temporaries carry no access attributes, and selection no longer scans its output
to attach descriptors afterward. Legalization and register allocation retain
attributes through ordinary field copying. Replacement and deletion release
them with the instruction's other fields.

Scheduling reads `InstRef::mem_flags()` directly, excludes volatile accesses and
preserves source order between other accesses; these fields do not establish
alias independence or authorize speculative execution. Atomic ordering, address
spaces, scalable accesses and general memory alias analysis remain future work.

## Integer reassociation

The pre-selection combine has a closed-form operation: flatten a single-use tree
of wrapping integer add/multiply/and/or/xor, sort its leaves by numeric register ID,
and rebuild a left-associated chain. An e-graph and string-valued extraction cost
are unnecessary for this normal form. The implementation uses an explicit stack,
reuses instruction IDs and virtual registers, and commits each changed block once.

Only same-type scalar integer operations without extra metadata are eligible.
Shared subexpressions remain leaves, fusion stays within one block, and
multi-definition registers are excluded. Floating-point reassociation is not
permitted. This is canonicalization, not a target latency optimization: the
resulting chain is not claimed to minimize critical path or register pressure.

Legalization limits the number of expansion actions per input instruction to
1024 so cyclic replacement rules fail with a diagnostic. This is an operational
guard, not a termination proof; target rules must also terminate when adding
blocks. Scalar widening remains explicitly unsupported.

## Design references

- [LLVM GlobalISel Legalizer](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/GlobalISel/Legalizer.cpp)
  revisits created/changed generic instructions through worklists and observers.
  Our smaller replacement contract uses an ordered worklist without adopting
  LLVM's general mutation-observer machinery.
- [Cranelift ISLE integration](https://github.com/bytecodealliance/wasmtime/blob/main/cranelift/docs/isle-integration.md)
  separates pure SSA lowering rules from machine register constraints. We retain
  the existing rule-based selector instead of combining it with legalization.
- [Efficiently Synthesizing Lowest Cost Rewrite Rules for Instruction Selection
  (2024)](https://arxiv.org/abs/2405.06127) explores generating a minimum-cost rule
  library from ISA descriptions. It motivates future offline rule generation,
  rather than adding a solver to this compilation path.
- [ACT (2026 revision)](https://arxiv.org/html/2510.09932v2) combines semantic
  rewrites, ISA-specific selection and cost-guided candidate exploration for
  tensor accelerators. Our inference is that equality saturation is useful when
  there are meaningful target alternatives and a cost model; it is not warranted
  merely to sort an integer expression. Its accelerator results are not evidence
  of scalar native-code speedups here.

No rule synthesis, SMT verification, or learned cost model is implemented by
this change. Those would need explicit semantic preconditions and target costs.

## Measurement

The standalone reassociation microbenchmark has been removed. Historically,
for 20 clones of a reversed 12-input integer-add tree, one local before/after run
measured 859.24 ms with the old e-graph combine and 48.58 microseconds with the
direct canonicalizer. Timing includes cloning and use-def construction. This
intentionally stresses associative/commutative search; it is not a whole-compiler
benchmark or a claim about generated-program performance.

Legalization regression coverage includes multi-step legalization, in-place
replacement, added blocks and cyclic rules. Dedicated reassociation unit tests
have been removed; the implementation remains exercised by the codegen pipeline.
Instruction storage separates result registers, input registers, attributes and register clobbers. Physical reads are explicit inputs; clobbers have no use-def occurrences. Generated builders and encoders address each storage domain directly; no logical operand list or per-instruction order map is stored. Tied constraints identify a result and a dense input index. Allocation plans contain only result/input register locations, and materialization leaves attributes untouched. OpSpec register fields use the declared Reg type and derive their role from signature-to-storage bindings.
