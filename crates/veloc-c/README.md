# Veloc C

`veloc-c` parses C, builds MIR, runs the Veloc optimizer, and emits a RISC-V
ELF object using the Veloc code generator. Clang supplies preprocessing and the
system linker driver; it does not compile the benchmark functions in the Veloc
build.

## Compile

```sh
cargo build --release -p veloc-c
target/release/veloc-c example.c -o example.o \
  --cpu c908 --cpp /opt/homebrew/opt/llvm/bin/clang \
  --sysroot target/riscv64-sysroot -O1
/opt/homebrew/opt/llvm/bin/clang --target=riscv64-linux-gnu \
  --sysroot=target/riscv64-sysroot -march=rv64gc_zba_zbb -mabi=lp64d \
  -fuse-ld=lld -no-pie example.o -o example
```

The sysroot needs the board's Linux headers, LP64D loader, libc, libm, linker
scripts, GCC runtime and startup objects. Native non-PIE linking needs `crt1.o`,
`crti.o`, `crtn.o`, `crtbegin.o` and `crtend.o`; PIE startup objects alone are
insufficient.

- `-O0` / `-O1`: select the optimization pipeline.
- `--cpu generic` / `--cpu c908`: RV64GC / RV64GC with Zba/Zbb.
- `--emit ast|mir|obj`: inspect parsing, optimized MIR, or emit an object.
- `-I DIR`, `-D NAME=VALUE`: preprocessing options.
- `--preprocessed`: consume preprocessed C directly; `.i` files imply this.

## Language support

The target is Linux RV64 LP64D: 32-bit `int`, 64-bit `long` and pointers,
unsigned plain `char`, IEEE f32/f64. Supported features include scalar
arithmetic, casts, pointers, arrays, structures/unions, enums, typedefs,
aggregate initializers, global data and symbol relocations, structured control
flow, `switch`, and direct/indirect calls, including calls to variadic functions.
Address-taken and volatile objects use memory; ordinary scalar locals use SSA.
Narrow integer parameters and returns keep their C object types internally and
are explicitly widened at ABI boundaries, including indirect calls.

This is a C subset, not a complete C11 implementation. Limitations include
variadic function definitions, aggregate assignment
and argument passing, bitfields, designated initializers, block-local static
storage, `goto`, GNU extensions, and `long double`. System headers with
unsupported extensions can require a compatible declaration header.
Typedef/tag scope handling and some constant-expression forms remain limited.

The native pipeline includes full and early-return-path inlining, stack-cell
promotion, loop transformations, bit-demand and function-result analysis, and exact GF(2) affine function
recognition. Affine recognition proves the transformation from MIR; it does
not recognize benchmark names or substitute handwritten CRC routines. It can
synthesize immutable lookup tables, so it is enabled by the native object
pipeline that supports module data.

Bit analysis propagates through block arguments and proven direct-call results.
Memory narrowing requires nonvolatile, nontrapping accesses and checks for
intervening writes; it does not assume pointers of different C types cannot
alias. The native backend also uses the RISC-V zero register and recreates
cheap spilled constants instead of loading them from the stack.

## CoreMark on K230

```sh
python3 crates/veloc-c/benchmarks/coremark/run.py \
  --sysroot target/riscv64-sysroot \
  --clang /opt/homebrew/opt/llvm/bin/clang \
  --host root@192.168.2.19
```

The runner fetches a pinned EEMBC revision, checks its tracked sources are
unchanged, compiles all five benchmark translation units with each compiler,
and links both against the **same OS adapter object**. LLVM uses `-O3` with the
same ISA/ABI; neither build uses LTO. It uploads the executables, checks their
SHA-256 hashes, and runs them serially on CPU 0 with alternating order.

Every reported sample must pass CoreMark validation and last at least ten
seconds. Defaults are 100,000 iterations and five samples per compiler. Logs,
compiler/source identities, build commands, binary hashes, medians and the
Veloc/LLVM ratio are saved under `target/native-coremark/`. Short development
runs screen candidates but are not valid CoreMark results.

`--source`, `--veloc`, `--out`, `--remote-dir`, `--iterations`, and `--runs`
override the checkout, compiler binary, artifact directory, board directory,
iteration count, and sample count. The runner neither commits nor pushes Git
changes.

Measurements and validation details are recorded in
[the K230 comparison](benchmarks/coremark/RESULTS.md).
