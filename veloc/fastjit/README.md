# veloc-fastjit

Fast, direct MIR-to-machine-code baseline compiler. The WebAssembly frontend
selects it with `--strategy fast-jit`; the existing optimizing code generator
remains a separate tier.

## Design direction

The core should follow TPDE's approach: a small analysis of SSA values and
block order, followed by one code-generation pass that decides value placement,
selects instructions, allocates registers, and emits bytes together. It should
read MIR directly, without constructing a second LIR. Copy-and-patch stencils
are an emission technique for fixed instruction fragments, not a requirement
that every instruction live in its own template. Variable addressing modes,
register assignments, and ISA-specific encodings can use generated snippet
encoders. The common layer owns SSA use/liveness information and relocation
records; each ISA owns register classes, ABI, instruction selection, encoding,
and branch fixups. Object packaging is independent of the ISA.

The current implementation is a first x86-64/Linux slice. It emits from MIR,
uses typed stencil holes and local labels, and can compile and execute the
CoreMark WebAssembly module through the existing JIT loader. It still gives
every SSA value a stack slot and uses hand-written x86 encodings. This is
deliberately a correctness baseline, **not** the final allocator or encoder.
Other ISAs are not implemented yet. The `Target` boundary currently shares
object packaging, but the x86-64 MIR walk is still target-specific; extracting
its SSA analysis and value-placement logic is necessary before adding AArch64
without duplicating that work.

The next performance step is SSA liveness/use analysis and a bounded register
allocator in the emission pass, including rematerialization of constants and
block-edge copies. After that, generate per-ISA snippets from declarative
descriptions and compare compile time, code size, and CoreMark throughput
against the optimizing tier. Stack maps, unwind information, traps, and wider
MIR/Wasm coverage need explicit work before this becomes a general JIT.

Research: [TPDE paper](https://arxiv.org/abs/2505.22610),
[TPDE framework documentation](https://docs.tpde.org/tpde-main.html), and
[Copy-and-Patch Compilation](https://compilers.stanford.edu/software/copy-and-patch/).
