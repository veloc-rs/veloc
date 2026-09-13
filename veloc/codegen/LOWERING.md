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

## Machine SSA and allocation

Virtual values have one definition through legalization, selection and scheduling.
Non-entry block parameters remain SSA definitions; branches carry their edge
arguments even when a generic conditional or jump table expands to several
target branches. Function entry arguments use G_ARG/ABI definitions, or a fresh
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

The optional machine SSA verifier runs at pipeline boundaries, not in builders.
It checks virtual definitions/dominance and edge contracts; physical ABI
registers and clobbers deliberately remain outside the SSA invariant.

## Memory representation

Translation receives an explicit target `DataLayout`. Pointer loads/stores use
its pointer size; other accesses require a fixed-size representation rather than
treating a scalable type's minimum size as an exact access width.

MIR load/store alignment and volatility survive as `InstRef::memory()`.
Stack accesses also receive a descriptor, with conservative alignment and MIR's
nontrapping stack-access contract. Source offsets remain instruction operands.

x86 legalization checks offset loads/stores against the signed disp32 range.
Larger displacements become an i64 constant plus a pointer addition, followed by
the original access with offset zero. The access keeps its ID and full memory
descriptor; address calculation has no memory effects. Thus unsigned MIR offsets
above `i32::MAX` never silently turn into negative displacements. In-range
offsets retain the compact addressing form. Generic i8/i16 and pointer accesses
use their existing target load/store rules; narrow copies remain available for
ABI materialization. Narrow arithmetic legalization is still separate work.

`ptr-offset` emits a pointer-valued `G_PTR_ADD` directly, without constructing
an integer-typed address and copying it back to a pointer.

ISA templates declare `(memory Read 8)` or `(memory Write 4)`. The generated
metadata states the encoded access width independently of result register width.
The current x86 selector requires one matching target access for a source access
and transfers its descriptor only to that instruction, not to address-calculation
instructions. Changing the direction, size or number of accesses fails explicitly;
future split/fused-access lowering needs its own checked mapping.

Allocation preserves descriptors when rewriting registers. Final emission
rechecks direction/width and alignment validity. Unknown memory effects and
annotated accesses remain scheduling barriers: these descriptors alone do not
establish alias independence or authorize speculative execution.
ABI/spill/prologue accesses do not yet all have per-instance descriptors, so
absence must not be interpreted as purity. Atomic ordering, address spaces,
scalable accesses and general memory alias analysis remain future work.

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

Run the isolated release microbenchmark with:

```sh
CARGO_INCREMENTAL=0 cargo test -p veloc-codegen --release reassociate_benchmark -- --ignored --nocapture
```

For 20 clones of a reversed 12-input integer-add tree, one local before/after run
measured 859.24 ms with the old e-graph combine and 48.58 microseconds with the
direct canonicalizer. Timing includes cloning and use-def construction. This
intentionally stresses associative/commutative search; it is not a whole-compiler
benchmark or a claim about generated-program performance.

Regression coverage includes wrapping arithmetic at four widths, shared uses,
multi-definition registers, excluded types, 4096-input trees, idempotence,
multi-step legalization, in-place replacement, added blocks and cyclic rules.
