# veloc-encoder

Standalone, `no_std` machine-code encoding. Runtime code has no dependency on
MIR, LIR, ISLE, allocation, stack frames, symbols, or code layout.

`defs/x86_64.spec` owns the descriptor structs and enums. The build script uses
OpSpec's shared data-type generator; Rust implements only encoding algorithms
and the checked hardware-register number type.

The initial implementation covers the legacy integer/SSE forms used by the
x86-64 backend, including REX, ModRM/SIB, base/index addressing, disp8/disp32,
RIP-relative addresses and short/near branches. VEX/EVEX are not implemented.

Encoding returns an inline `Encoded<15>` with an optional signed PC-relative
field. The field records its byte offset, width, PC base and addend; the caller
owns the destination. In particular, the PC base is the instruction end, not
necessarily the displacement field end.

Codegen converts physical registers and stack slots, binds symbolic targets,
and chooses branch forms during layout. It consumes the same descriptor
definitions through typed `encoding` expressions on instruction declarations.
`Emission` constructors are an explicitly declared, generated Rust trait.

Future architectures belong in separate modules with their own descriptor
schemas and algorithms. Only genuinely shared output/error contracts belong
at the crate root.
