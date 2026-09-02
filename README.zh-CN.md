# Veloc

[![Rust 2024](https://img.shields.io/badge/Rust-2024-orange.svg)](https://www.rust-lang.org/)
[![许可证：MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[English](README.md) | 简体中文

Veloc 是一个使用 Rust 编写的实验性编译器基础设施和 WebAssembly 运行时。它在同一个 workspace 中提供类型化 SSA IR、可复用的分析与优化、紧凑的寄存器字节码解释器，以及 x86-64 原生代码生成器。

> Veloc 正在积极开发中，公共 API 和支持的 WebAssembly 特性可能随时变化。

## 快速开始

Veloc 使用 [`rust-toolchain.toml`](rust-toolchain.toml) 中锁定的 nightly 工具链。安装 [Rustup](https://rustup.rs/) 后，Cargo 会自动选择并安装该工具链。

```bash
git clone https://github.com/veloc-rs/veloc.git
cd veloc
cargo build --workspace --release
```

使用解释器运行仓库内置的 CoreMark WebAssembly 模块：

```bash
cargo run --release -p veloc-wasm -- \
  crates/veloc-wasm/tests/wasm/coremark.wasm \
  --strategy interpreter
```

也可以使用 x86-64 原生 JIT：

```bash
cargo run --release -p veloc-wasm -- path/to/module.wasm --strategy jit
```

CLI 接受 `.wasm` 和 `.wat` 文件，默认调用 `_start`，并提供 `interpreter`、`jit` 和 `auto` 三种执行策略。

## 查看生成结果

```bash
# 打印 Veloc IR，不执行模块
cargo run -p veloc-wasm -- path/to/module.wat --dump-ir

# 将 Veloc IR 写入文件
cargo run -p veloc-wasm -- path/to/module.wasm \
  --output-ir module.veloc-ir

# 打印解释器字节码
cargo run -p veloc-wasm -- path/to/module.wasm \
  --strategy interpreter --dump-bytecode

# 打印优化统计信息并生成 Chrome Trace
cargo run -p veloc-wasm -- path/to/module.wasm \
  -O 1 --print-stats --trace-file optimizer-trace.json
```

运行 `cargo run -p veloc-wasm -- --help` 可以查看完整的 CLI 参数。

## 工作原理

```text
 WebAssembly                       实验性 C 前端
     │                                   │
     └──────────────┬────────────────────┘
                    ▼
              Veloc 类型化 SSA IR
                    │
          ┌─────────┴──────────┐
          │ 分析               │ 优化
          │ use-def/活跃变量    │ 常量折叠/DCE
          └─────────┬──────────┘
                    ▼
          ┌─────────┴──────────────┐
          ▼                        ▼
     寄存器字节码解释器           MIR 与 x86-64 后端
          │                        │
          ▼                        ▼
         执行                 ELF 对象 / JIT
```

WebAssembly 和 C 源码都会转换为同一种 Veloc IR。运行时可以将 IR 编译为紧凑字节码，也可以经过 MIR lowering 生成 x86-64 原生代码。

## 仓库结构

| Crate | 职责 |
| --- | --- |
| `veloc` | IR、解释器和代码生成器的顶层门面。 |
| `veloc-ir` | 类型化 SSA IR、构建器、数据流图、文本格式和验证器。 |
| `veloc-analyzer` | Use-def 与活跃变量分析。 |
| `veloc-optimizer` | Pass 管理、指标统计、常量折叠和死代码消除。 |
| `veloc-interpreter` | IR 到字节码的编译器及寄存器字节码运行时。 |
| `veloc-codegen` | 与目标无关的 MIR 流水线及 x86-64 后端。 |
| `veloc-isle` | 用于目标 lowering 和指令选择的规则编译器。 |
| `veloc-wasm` | WebAssembly 翻译器、运行时、CLI、链接器、JIT 和 WASI 支持。 |
| `veloc-c` | 实验性 C 解析器和 IR 前端。 |
| `veloc-spec` | WebAssembly 规范测试运行器。 |

## 开发与测试

构建并检查完整 workspace：

```bash
cargo build --workspace
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets
```

只运行解释器和 WebAssembly 测试：

```bash
cargo test -p veloc-interpreter
cargo test -p veloc-wasm
```

规范测试运行器可以接收一个 `.wast` 文件，也可以接收上游 WebAssembly 规范测试中的目录：

```bash
cargo run --release -p veloc-spec -- \
  /path/to/wasm-spec/test/core \
  --strategy interp
```

将 `interp` 替换为 `jit`，即可使用同一套测试验证原生后端。

## 文档

- [Veloc IR 指令参考](veloc/ir/docs/zh/instructions.md)
- [Veloc IR instruction reference (English)](veloc/ir/docs/en/instructions.md)

## 项目状态

Veloc 已经可以用于编译器和运行时实验，但尚不是稳定的生产级工具链。目前的开发重点包括扩大 WebAssembly 覆盖范围、完善 x86-64 指令与 ABI 支持、增加优化 pass 和目标架构、完成 C 前端，以及稳定公共 API。

欢迎提交 Issue 和 Pull Request。

## 许可证

Veloc 使用 [MIT 许可证](LICENSE)。
