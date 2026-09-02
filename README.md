# Veloc

[![Rust 2024](https://img.shields.io/badge/Rust-2024-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

English | [简体中文](README.zh-CN.md)

Veloc is an experimental compiler infrastructure and WebAssembly runtime written in Rust. It combines a typed SSA IR, reusable analyses and optimizations, a compact register-bytecode interpreter, and an x86-64 native code generator in one workspace.

> Veloc is under active development. Public APIs and supported WebAssembly features may change without notice.

## Quick start

Veloc uses the nightly toolchain pinned in [`rust-toolchain.toml`](rust-toolchain.toml). With [Rustup](https://rustup.rs/) installed, Cargo selects and installs it automatically.

```bash
git clone https://github.com/veloc-rs/veloc.git
cd veloc
cargo build --workspace --release
```

Run the bundled CoreMark WebAssembly module with the interpreter:

```bash
cargo run --release -p veloc-wasm -- \
  crates/veloc-wasm/tests/wasm/coremark.wasm \
  --strategy interpreter
```

Use the native x86-64 JIT instead:

```bash
cargo run --release -p veloc-wasm -- path/to/module.wasm --strategy jit
```

The CLI accepts `.wasm` and `.wat` files, invokes `_start` by default, and supports `interpreter`, `jit`, and `auto` execution strategies.

## Inspect generated code

```bash
# Print Veloc IR without executing the module
cargo run -p veloc-wasm -- path/to/module.wat --dump-ir

# Write Veloc IR to a file
cargo run -p veloc-wasm -- path/to/module.wasm \
  --output-ir module.veloc-ir

# Print interpreter bytecode
cargo run -p veloc-wasm -- path/to/module.wasm \
  --strategy interpreter --dump-bytecode

# Print optimizer statistics and write a Chrome trace
cargo run -p veloc-wasm -- path/to/module.wasm \
  -O 1 --print-stats --trace-file optimizer-trace.json
```

Run `cargo run -p veloc-wasm -- --help` for all CLI options.

## How it works

```text
 WebAssembly                     Experimental C frontend
     │                                    │
     └──────────────┬─────────────────────┘
                    ▼
             Veloc typed SSA IR
                    │
          ┌─────────┴──────────┐
          │ analyses           │ optimizations
          │ use-def/liveness   │ constant folding/DCE
          └─────────┬──────────┘
                    ▼
          ┌─────────┴──────────────┐
          ▼                        ▼
 register-bytecode interpreter    MIR and x86-64 backend
          │                        │
          ▼                        ▼
      execution              ELF object / JIT
```

WebAssembly and C source are translated into the same Veloc IR. From there, the runtime can compile the IR into compact bytecode or lower it through MIR to native x86-64 code.

## Repository layout

| Crate | Purpose |
| --- | --- |
| `veloc` | Top-level facade for the IR, interpreter, and code generator. |
| `veloc-ir` | Typed SSA IR, builders, data-flow graph, text format, and validator. |
| `veloc-analyzer` | Use-def and liveness analyses. |
| `veloc-optimizer` | Pass management, metrics, constant folding, and dead-code elimination. |
| `veloc-interpreter` | IR-to-bytecode compiler and register-bytecode runtime. |
| `veloc-codegen` | Target-independent MIR pipeline and x86-64 backend. |
| `veloc-isle` | Rule compiler used by target lowering and instruction selection. |
| `veloc-wasm` | WebAssembly translator, runtime, CLI, linker, JIT, and WASI support. |
| `veloc-c` | Experimental C parser and IR frontend. |
| `veloc-spec` | WebAssembly specification test runner. |

## Development

Build and test the complete workspace:

```bash
cargo build --workspace
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets
```

Run focused interpreter and WebAssembly tests:

```bash
cargo test -p veloc-interpreter
cargo test -p veloc-wasm
```

The specification runner accepts either a `.wast` file or a directory from the upstream WebAssembly specification tests:

```bash
cargo run --release -p veloc-spec -- \
  /path/to/wasm-spec/test/core \
  --strategy interp
```

Replace `interp` with `jit` to exercise the native backend.

## Documentation

- [Veloc IR instruction reference](veloc/ir/docs/en/instructions.md)
- [Veloc IR 指令参考（中文）](veloc/ir/docs/zh/instructions.md)

## Project status

Veloc is suitable for compiler and runtime experimentation, but is not yet a stable production toolchain. Current priorities include broader WebAssembly coverage, more x86-64 instructions and ABI support, additional optimization passes and targets, completion of the C frontend, and API stabilization.

Issues and pull requests are welcome.

## License

Veloc is available under the [MIT License](LICENSE).
