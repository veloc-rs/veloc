# K230 CoreMark：2026-10-04

## 当前结果：第二阶段，目标超过 LLVM 5%

增加通用的状态机分支贯穿，并调整已有块参数清理的位置。沿用下文测量口径，
原生 C CoreMark 的五轮正式结果已超过本阶段的 +5% 目标。

| 指标，中位数 | Veloc | LLVM 22.1.6 | 比较 |
| --- | ---: | ---: | --- |
| 原生 C CoreMark，K230 | **6211.811** | **5250.315** | **Veloc 高 18.31%** |
| 五个 C 单元串行编译，含预处理和进程启动 | 184.720 ms | 201.584 ms | Veloc 时间少 8.37% |
| 同一份预处理输入到对象文件，含进程启动 | 69.174 ms | 201.328 ms | Veloc 时间少 65.64% |

Veloc 使用 `-O1 --cpu c908`，LLVM 使用 `-O3`、通用 RISC-V 调优；两者指令集、
ABI、sysroot 和 OS adapter 相同，无 LTO/PGO。这里比较原生 C，不能外推为
Wasm 或其他工作负载的结果。历史 +20% 目标仍未达到，用户本阶段目标为 +5%。

### 实现与成本

- `ThreadingPass` 识别由常量与块参数组成的状态网络，按已知入参复制小块，
  通过共享求值器折叠计算和分派。按块与常量绑定缓存版本，限制代码增长。
- SSA 依赖通过块参数显式传递；先移除不可达块，再进行复制。后续参数清理
  消除临时传递，恢复相同值身份后再进行 CFG 简化。
- 把已有 `SimplifyParamsPass` 移到分支贯穿之后，没有增加 pass 重跑次数。
- 状态机单元 profile：178 个小块版本、45 处分支折叠，pass 约 0.83 ms。
  三个热点分派表消失。最终 `.text` 为 11,030 字节，上一阶段为 10,566，
  增长 4.39%；矩阵、链表、工具单元的对象文件逐字节相同。

[设计与边界](../../../../docs/optimization-research-2026-10-04.md#第二阶段常量状态驱动的分支贯穿)。
实现位于共享 MIR 流程中，依据 IR 数据流识别，不依赖函数名或固定运行输入。

### 正式运行样本

每个编译器五轮，交替顺序，每轮 100,000 次迭代、超过 10 秒。
上传后核对 SHA-256，计时期间无文件传输或其他板端基准。
十次运行全部输出 `Correct operation validated`，所有固定 CRC 正确，最终
CRC 均为 `0xd340`。

| 轮次 | 顺序 | Veloc 分数 | 秒数 | LLVM 分数 | 秒数 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Veloc → LLVM | 6207.642497 | 16.109175 | 5242.137949 | 19.076186 |
| 2 | LLVM → Veloc | 6211.810711 | 16.098366 | 5262.060097 | 19.003964 |
| 3 | Veloc → LLVM | 6219.047107 | 16.079634 | 5250.315042 | 19.046476 |
| 4 | LLVM → Veloc | 6207.552851 | 16.109408 | 5257.100579 | 19.021892 |
| 5 | Veloc → LLVM | 6216.982522 | 16.084974 | 5248.913527 | 19.051562 |

完整 C 编译样本，单位 ms；一次预热、七次交替串行测量，期间无并行构建：

```text
Veloc: 186.412, 183.094, 183.868, 181.905, 206.601, 184.738, 184.720
LLVM:  203.176, 198.905, 220.958, 200.559, 199.942, 201.584, 201.822
```

### 复现与验证

仍基于下文记录的未提交工作树。保留了第一阶段记录，以区分新增收益与先前结果。

```text
Veloc compiler SHA-256:
16818ed397ea9ee73eb55021b645f6b484914bbc1c550535860f4cf35bf5cfef
Veloc ELF SHA-256:
cfb38a15ff5fcb02a26c152d6f59a6f6c9308f1065c9121c420d251093547cb4
LLVM ELF SHA-256:
33119578bf6742a18dcb5613aa0fd66389ddb6168db303d5ec549a1573c32e0e
```

- `target/optimizer-5pct-20261004/validated/`：正式五轮、七次 C 编译、完整命令和 report.json。
- `target/optimizer-5pct-20261004/preprocessed/`：同一预处理输入的编译比较。
- `target/optimizer-5pct-20261004/verify-formal/`：五个正式单元显式开启 `--verify-ir`
  均通过，产物与计时关闭验证时逐字节相同。
- `target/optimizer-5pct-20261004/threading/` 和 `threading-params/`：顺序选择的短测，
  初版约 6040 分，提前参数清理后约 6198 分；短测不作为正式分数。

```sh
cargo build --release -p veloc-c
python3 crates/veloc-c/benchmarks/coremark/run.py \
  --out target/optimizer-5pct-20261004/validated --compile-runs 7 --runs 5
python3 crates/veloc-c/benchmarks/coremark/run.py \
  --out target/optimizer-5pct-20261004/preprocessed --preprocessed --compile-only
```

`cargo check --workspace` 与已暂存/未暂存的 `git diff --check` 均通过。
未新增测试文件。workspace 仍有下文记录的已有 unused feature 警告。

## 第一阶段记录：SCCP 与编译开销优化

本轮完成 SCCP、支配作用域谓词传播、无修改 pass 结果复用，以及 e-graph
查询和仿射分析的开销优化。[研究与设计记录](../../../../docs/optimization-research-2026-10-04.md)。

| 指标，中位数 | Veloc | LLVM 22.1.6 | 比较 |
| --- | ---: | ---: | --- |
| 原生 C CoreMark，K230 | 5029.271 | 5253.128 | Veloc 低 4.26% |
| 五个 C 单元串行编译，含预处理和进程启动 | 179.375 ms | 196.530 ms | Veloc 时间少 8.73% |
| 同一份预处理输入到对象文件，含进程启动 | 63.710 ms | 196.141 ms | Veloc 时间少 67.52% |

**运行性能超过 LLVM 20% 的目标尚未达到。** 按本次 LLVM 中位数，目标约为
6303.754 分，还需要在当前 Veloc 得分上提高约 25.3%。
本轮显著收益在编译速度，没有观测到 CoreMark 运行性能提升。

## 口径

- 主机：本地 macOS arm64；交叉编译在本机，运行在 `root@192.168.2.19`。
- 设备：K230，C908，Linux 6.6.36，一个在线 Linux hart，固定到 CPU 0。
- EEMBC 原始代码：`1f483d5b8316753a742cbf5590caf5bd0a4e4777`，五个算法/主程序
  翻译单元的已跟踪源码未修改。此记录比较原生 C，不是 Wasm。
- 双方使用 RV64GC + Zba/Zbb、LP64D、相同 sysroot 和相同的 Clang 编译 OS adapter
  对象。Veloc 使用 `-O1 --cpu c908`；Clang 22.1.6 使用 `-O3`、通用 CPU 调优。
  安装的 Clang 没有 C908 CPU 调优模型。双方均无 LTO、PGO、fast-math。
- 运行参数：`0 0 0x66 100000 7 1 2000`，每个编译器五轮，交替运行顺序。
  上传完毕并核对 SHA-256 后才开始运行，测量期间不传输文件或运行其他基准。
- 每轮超过十秒，均输出 `Correct operation validated`。另外核对固定的 seed、
  list、matrix、state CRC，以及跨编译器/跨轮次一致的最终 CRC `0xd340`。
- 编译：一次预热，七次交替串行测量，未同时运行 Cargo 构建。只计五个翻译单元，
  不计共享 adapter 和链接。预处理输入实验使用两者相同的 `.i` 文件，单独报告。
- 编译计时关闭 profiling 和 IR 验证。另对最终版本的五个单元显式运行
  `--verify-ir`，验证前端 MIR、优化后 MIR 和每个机器 pass；均通过。

这些是该工作负载、设备和配置上的样本，不代表跨程序性能结论。

## 原始运行样本

| 轮次 | 顺序 | Veloc 分数 | 秒数 | LLVM 分数 | 秒数 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Veloc → LLVM | 5029.270976 | 19.883598 | 5245.831211 | 19.062756 |
| 2 | LLVM → Veloc | 5025.938662 | 19.896781 | 5241.515370 | 19.078452 |
| 3 | Veloc → LLVM | 5030.334352 | 19.879394 | 5255.784971 | 19.026654 |
| 4 | LLVM → Veloc | 5024.872861 | 19.901001 | 5264.245787 | 18.996074 |
| 5 | Veloc → LLVM | 5031.418740 | 19.875110 | 5253.128348 | 19.036276 |

原始 C 编译样本，单位 ms：

```text
Veloc: 179.824, 177.960, 176.633, 180.675, 178.142, 181.601, 179.375
LLVM:  199.202, 196.349, 196.530, 195.790, 205.323, 196.431, 197.922
```

## 消融和诊断

- 旧优化流程在同样关闭 IR 验证时，七次完整编译中位数为 **290.307 ms**，
  同组 LLVM 为 197.043 ms。新版本约缩短 **38.2%**；该比较没有把关闭验证本身
  当作优化收益。主要收益来自 e-graph 关系行去重及仿射符号值共享。
- 初始版本两轮运行约 5057 分。当前版本约 5029 分，没有改善；这约 0.6% 的
  差异尚未通过同轮交替的旧/新版本实验分离代码差异与设备波动。
- 移除内联前额外 SCCP 后，五个对象和最终 ELF 逐字节相同，默认流程移除了它。
- 在 CFG 后额外跑一次 Simplify 没有改善短测结果，未保留。
- RISC-V 分支树阈值从 8 改为 4 的短测约 4744 分，已恢复为 8。
- 采样诊断可执行文件包含 SIGPROF 插桩，独立于正式跑分文件。
  Veloc 样本约 35% 位于矩阵、35% 位于状态机、28% 位于链表。

下一阶段的重点是带别名证明的循环存储提升、状态机分支贯穿及内联成本建模。
本轮没有为追分加入按函数名识别、替换算法、固定 benchmark 输入等特殊处理。

## 产物身份与复现

基于提交 `1b7e6b1b821700552606fa88afd40dacffdcc861` 的未提交工作树。
HEAD 本身不足以重现改动，因此另外保存编译器和 ELF 哈希。

```text
Veloc compiler SHA-256:
8010d3c3586c756e5d8469561005f7e637af1ddd4a5fd0c6a1acc8203ed00581
Veloc ELF SHA-256:
1d6c5babe9d356530f9152a79e81556f7adfa29940e6ac19c4470f4271e6aa96
LLVM ELF SHA-256:
33119578bf6742a18dcb5613aa0fd66389ddb6168db303d5ec549a1573c32e0e
```

本机完整命令、原始日志与 report.json：

- `target/optimizer-20261004/validated/`：最终 C 编译和五轮正式运行。
- `target/optimizer-20261004/preprocessed/`：同一预处理输入的编译比较。
- `target/optimizer-20261004/baseline-no-verifier/`：旧流程关闭验证的编译对照。

```sh
cargo build --release -p veloc-c
python3 crates/veloc-c/benchmarks/coremark/run.py \
  --out target/native-coremark --compile-runs 7 --runs 5
python3 crates/veloc-c/benchmarks/coremark/run.py \
  --out target/native-coremark-preprocessed --preprocessed --compile-only
```

`cargo check --workspace` 与 `git diff --check` 均通过。未新增测试文件。
工作区检查仍有已有的 `veloc-test-mir` 未使用 `const_cmp` feature 警告。
