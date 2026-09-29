# RV64 backend

The baseline is RV64GC with the Linux LP64D ABI. `generic` and `c908` currently
select the same instruction set. SIMD is not implemented.

- `defs/riscv64`: registers, ABI, legalization, selection and instruction expansions.
- `veloc-encoder`: generated encoding types and RV instruction field packing.
- `emitter.rs`: allocated register/stack adapters, labels and symbol relocations.
- `frame.rs`: fixed stack-pointer-relative frames, outgoing arguments and saved registers.

32-bit integers use the ABI's sign-extended XLEN representation. Integer and
floating-point argument registers are allocated independently. Absolute 64-bit
call literals avoid imposing a 2 GiB distance limit between JIT code and host
functions. Linux `riscv_flush_icache` publishes relocated code before execution.

## Build on the Mac, run on the board

From the repository root, install the pinned toolchain's target:

```sh
rustup target add riscv64gc-unknown-linux-gnu
```

With a GNU Linux cross toolchain, Cargo can use it directly:

```sh
CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER=riscv64-linux-gnu-gcc \
  cargo build -p veloc-wasm --bin veloc-wasm --features wat \
    --target riscv64gc-unknown-linux-gnu --release
```

On this Mac, the executable named `riscv64-linux-gnu-gcc` is a Clang wrapper.
Use the installed Homebrew LLVM and LLD directly instead. No project-specific
linker script is necessary. Prepare a Debian sysroot once (this board uses GCC 14):

```sh
mkdir -p target/riscv64-sysroot
ssh root@192.168.2.19 'tar -chf - -C / \
  lib/ld-linux-riscv64-lp64d.so.1 \
  lib/riscv64-linux-gnu/libc.so.6 \
  lib/riscv64-linux-gnu/libc.so \
  lib/riscv64-linux-gnu/libc_nonshared.a \
  lib/riscv64-linux-gnu/libm.so \
  lib/riscv64-linux-gnu/libm.so.6 \
  lib/riscv64-linux-gnu/libgcc_s.so.1 \
  lib/riscv64-linux-gnu/libpthread.a \
  lib/riscv64-linux-gnu/libdl.a \
  lib/riscv64-linux-gnu/librt.a \
  lib/riscv64-linux-gnu/libutil.a \
  usr/lib/riscv64-linux-gnu/Scrt1.o \
  usr/lib/riscv64-linux-gnu/crti.o \
  usr/lib/riscv64-linux-gnu/crtn.o \
  usr/lib/riscv64-linux-gnu/libc_nonshared.a \
  usr/lib/gcc/riscv64-linux-gnu/14/crtbeginS.o \
  usr/lib/gcc/riscv64-linux-gnu/14/crtendS.o \
  usr/lib/gcc/riscv64-linux-gnu/14/libgcc.a \
  usr/lib/gcc/riscv64-linux-gnu/14/libgcc_s.so' |
  tar -xf - -C target/riscv64-sysroot

CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER=/opt/homebrew/opt/llvm/bin/clang \
CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C link-arg=--target=riscv64-linux-gnu -C link-arg=--sysroot=$PWD/target/riscv64-sysroot -C link-arg=--gcc-toolchain=$PWD/target/riscv64-sysroot/usr -C link-arg=-fuse-ld=/opt/homebrew/opt/lld/bin/ld.lld -C link-arg=-Wl,--strip-debug" \
  cargo build -p veloc-wasm --bin veloc-wasm --features wat \
    --target riscv64gc-unknown-linux-gnu --release
```

The `--strip-debug` option reduces the executable transfer size.
No Rust installation or compilation is needed on the board.

## CoreMark input

Use the WASI CoreMark binary at wasm3 commit
`6d93778f6eda84b67db3d26c48d193243f4b67f2`. Newer builds can contain Wasm tail calls,
which the current Wasm frontend does not support.

```sh
mkdir -p target/coremark
curl -fL https://raw.githubusercontent.com/wasm3/wasm3/6d93778f6eda84b67db3d26c48d193243f4b67f2/test/wasi/coremark/coremark.wasm \
  -o target/coremark/coremark-baseline.wasm
ssh root@192.168.2.19 'mkdir -p /root/veloc-riscv'
scp target/riscv64gc-unknown-linux-gnu/release/veloc-wasm \
  target/coremark/coremark-baseline.wasm root@192.168.2.19:/root/veloc-riscv/
ssh root@192.168.2.19 \
  'cd /root/veloc-riscv && ./veloc-wasm coremark-baseline.wasm --strategy jit -O 0'
```

The workspace enables debug assertions specifically for `elf_loader` in release:
version 0.17.0 places default section-group initialization inside
`debug_assert_eq!`. Remove that workaround when upgrading to a version that
initializes the groups unconditionally.

## Board run

On the K230/C908 board at `192.168.2.19`, the JIT at `-O 0` completed 11,000
iterations at 596.265 iterations/second with the Clang/LLD-linked executable.
A separate `-O 1` run completed at 638.649 iterations/second. Both runs reported
`Correct operation validated`, with seed/list/matrix/state CRCs
`e9f5 / e714 / 1fd7 / 8e3a`. This is one run of the pinned Wasm input, not a
native C CoreMark score.
