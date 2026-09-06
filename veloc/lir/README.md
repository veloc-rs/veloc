# Low-level IR

`veloc-lir` owns the machine-facing representation: instructions and operand
schemas, registers and register banks, functions and blocks, stack-frame data,
symbols, use-def chains and stage markers. It supports `no_std` with `alloc` and
has no dependency on codegen or any target backend.

Codegen owns MIR-to-LIR translation, legalization, register-bank selection,
instruction selection, register allocation, ABI handling and machine-code
emission. Pipeline scheduling and transitions stay there; stage markers live
here because they parameterize `MachineFunction` and its allocation APIs.
Markers do not independently prove that a pass established its postconditions.

Consumers import `veloc_lir::{MachineFunction, MachineInst, ...}` and
`veloc_lir::stages::RawLir`, or use the top-level `veloc::lir` facade. There is no
compatibility module at `veloc_codegen::lir`.

The public function data is mutable IR storage for transformation passes.
Instruction-extra pools remain private behind their accessors. Schema decoding
returns `DecodeError`; codegen wraps it without making LIR depend on backend
error types. Symbol interning takes a name and linkage, not a MIR module.

`defs/generic.rs` supplies the runtime opcode enum and the reviewed primitive
bindings read by codegen's build script. The direct MIR-to-LIR dispatch is still
generated in codegen. Shared `Type`, `Signature`, linkage, condition codes and
some entity identifiers currently come from `veloc-mir`; extracting this crate
does not yet make these foundational types independent of MIR.

```sh
cargo test -p veloc-lir
cargo check -p veloc-lir --no-default-features
```
