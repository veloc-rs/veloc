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
cargo run --release -p veloc-wasm --bin veloc-wasm -- run \
  crates/veloc-wasm/tests/wasm/coremark.wasm \
  --strategy interpreter
```

也可以使用 x86-64 原生 JIT：

```bash
cargo run --release -p veloc-wasm --bin veloc-wasm -- run crates/veloc-wasm/tests/wasm/coremark.wasm --strategy jit
```

在 x86-64 Linux 上运行完整的 JIT CoreMark 回归测试（检查验证信息和 CRC）：

```bash
CARGO_INCREMENTAL=0 cargo test --release -p veloc-wasm --test jit \
  coremark_validates_under_jit -- --ignored --nocapture
```

`run` 默认使用解释器调用 `_start`。执行策略包括 `interpreter`、`jit`、
`fast-jit`（Linux x86-64）和 `auto`（当前选择 JIT）。
读取 WAT 文件需要使用 `--features wat` 构建。

默认的 `--memory-checks auto` 在 Linux x86-64 / RV64 glibc 原生执行时使用保护页，
其他环境使用软件边界检查。`--memory-checks software` 可强制软件检查。
Linux x86-64 glibc 上的解释器也可显式启用保护页：

```bash
cargo run --release -p veloc-wasm --bin veloc-wasm -- run \
  path/to/module.wasm --strategy interpreter --memory-checks guarded
```

保护页模式会安装进程级 SIGSEGV/SIGBUS 处理器。原生代码通过 C 陷阱边界返回，
解释器保护页要求 Rust 构建支持 unwind。RV64 在写入前探测最后一个字节，
避免非对齐越界写入先修改部分内存。交叉编译对象时也可显式选择 guarded，
但加载该对象的运行时必须支持对应的陷阱处理。

## 查看生成结果

```bash
# 打印 Veloc IR，不执行模块
cargo run -p veloc-wasm --features wat --bin veloc-wasm -- emit path/to/module.wat --emit mir

# 将 Veloc IR 写入文件
cargo run -p veloc-wasm --bin veloc-wasm -- emit path/to/module.wasm --emit mir -o module.veloc-mir

# 打印解释器字节码
cargo run -p veloc-wasm --bin veloc-wasm -- emit path/to/module.wasm --emit bytecode

# 打印优化统计信息并生成 Chrome Trace
cargo run -p veloc-wasm --bin veloc-wasm -- run path/to/module.wasm \
  -O 1 --print-stats --trace-file optimizer-trace.json
```

运行 `cargo run -p veloc-wasm --bin veloc-wasm -- --help` 可以查看完整的 CLI 参数。

CLI 只保留 `run`、`emit`、`inspect` 三个命令；不再支持直接传入文件的旧入口、
`--output-ir` 或 `--compile-only`。`-O0`、`-O1` 同时选择 MIR 与机器码优化流程，
其他等级会报错。所有 `--dump-*` 都输出到 stderr，并继续当前命令。
`emit` 只生成指定产物，不链接导入、不实例化模块，也不执行模块的 start 函数。

```sh
# 带类型参数调用导出函数
veloc-wasm run add.wasm --invoke add --arg i32:1 --arg i32:2
# 设置 WASI 环境变量与命令行参数
veloc-wasm run app.wasm --env MODE=fast -- input.txt
# 从后端声明查询 CPU 与指令集特性
veloc-wasm inspect cpus --target riscv64
veloc-wasm inspect features --target riscv64
# 本机交叉编译 ELF 目标文件，不加载执行
veloc-wasm emit module.wasm --emit object --target riscv64 --cpu c908 -o module.o
```

文本产物默认写到 stdout；`-o -` 显式选择 stdout。object 输出必须指定 `-o`。
目标文件依赖 Veloc 的运行时 ABI，不是独立可执行程序。
库接口以 `Config.codegen.opt_level` 作为统一优化等级，
`Engine::new` 和 `Engine::with_config` 均返回 `Result`。

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
     寄存器字节码解释器           LIR 与 x86-64 后端
          │                        │
          ▼                        ▼
         执行                 ELF 对象 / JIT
```

WebAssembly 和 C 源码都会转换为同一种 Veloc 中层 IR（MIR）。运行时可以将 MIR 编译为紧凑字节码，也可以经过低层 IR（LIR）生成 x86-64 原生代码。

## 仓库结构

| Crate | 职责 |
| --- | --- |
| `veloc` | IR、解释器和代码生成器的顶层门面。 |
| `veloc-mir` | 类型化 SSA IR、构建器、数据流图、文本格式和验证器。 |
| `veloc-lir` | 面向机器的 IR、操作数格式、寄存器标识和阶段标记。 |
| `veloc-analyzer` | Use-def 与活跃变量分析。 |
| `veloc-optimizer` | Pass 管理、指标统计、常量折叠和死代码消除。 |
| `veloc-interpreter` | IR 到字节码的编译器及寄存器字节码运行时。 |
| `veloc-codegen` | 与目标无关的 LIR 流水线及 x86-64 后端。 |
| `veloc-spec` | 复用 OpSpec 契约的跨 IR 类型化值规则编译器，以及目标描述和指令选择。 |
| `veloc-wasm` | WebAssembly 翻译器、运行时、CLI、链接器、JIT 和 WASI 支持。 |
| `veloc-c` | 实验性 C 解析器和 IR 前端。 |
| `veloc-wasm-spec` | WebAssembly 规范测试运行器。 |

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
cargo run --release -p veloc-wasm-spec -- \
  /path/to/wasm-spec/test/core \
  --strategy interp
```

将 `interp` 替换为 `jit`，即可使用同一套测试验证原生后端。

也可以使用仓库固定版本的 testsuite 子模块，例如运行整数测试：

```bash
git submodule update --init crates/veloc-wasm/tests/testsuite
CARGO_INCREMENTAL=0 cargo run -p veloc-wasm-spec -- \
  crates/veloc-wasm/tests/testsuite/i32.wast --strategy jit --opt-level 1 --verbose
```

运行器会单独报告跳过的栈耗尽断言。CoreMark 通过不代表完整支持 WebAssembly：完整浮点套件仍需要补充 min/max、饱和转换等 lowering。

## 文档

- [Veloc IR 指令参考](veloc/mir/docs/zh/instructions.md)
- [Veloc IR instruction reference (English)](veloc/mir/docs/en/instructions.md)

## 项目状态

Veloc 已经可以用于编译器和运行时实验，但尚不是稳定的生产级工具链。目前的开发重点包括扩大 WebAssembly 覆盖范围、完善 x86-64 指令与 ABI 支持、增加优化 pass 和目标架构、完成 C 前端，以及稳定公共 API。

欢迎提交 Issue 和 Pull Request。

## 许可证

Veloc 使用 [MIT 许可证](LICENSE)。
