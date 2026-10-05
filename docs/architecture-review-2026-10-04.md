# 代码结构与精简审查

日期：2026-10-04。对象：当前工作树，包含尚未提交的改动。

## 范围与结论

初次审查完成了仓库结构、调用引用和重复实现的扫描，并重点阅读了 C/Wasm 前端、MIR/LIR、分析与 pass 管理、规则生成、表达式优化、指令调度、解释器和运行时的主要路径。这不是逐行穷尽审计。下文各问题的位置与描述保留审查时的快照，实施状态单独记录如下。性能收益均未实测。

最值得处理的是三个问题：分析缓存的生命周期与编辑接口不一致；规则信息存在多个维护入口；内存事实在多个 pass 中重复推导。直接删除无用代码可以先做，但不能替代这些结构性调整。

## 实施状态

| 问题 | 已实施内容 |
| --- | --- |
| 1、10：闲置接口与重复判断 | 删除 lexer 预读缓存、未使用字段和方法；parser 共用类型起始判断；删除无产生路径的 codegen 错误、解释器闲置枚举入口；移除 C 模块的整体 dead_code 豁免并清理暴露的闲置函数。 |
| 2：解释器能力错误 | 编译返回 Result，错误携带模块、函数、指令或值与类型信息；所有函数编译成功后再发布函数引用和模块。 |
| 3：MIR 分析生命周期 | 连续函数 pass 按函数共用 AnalysisManager；模块 pass 划分阶段；用 PassOutcome 表达是否修改，编辑入口负责失效；删除跳过模块 pass 的旧入口和无消费者的 TypeId 保留集合。 |
| 4：LIR 编辑失效 | 获取 editor 和只读访问不再清缓存；可变访问在修改前保守失效；替换输入、结果的受约束接口只报告操作数与寄存器变化。 |
| 5：guard 模型 | 引入经过类型检查的 Predicate 树，统一派生代码、捕获依赖和安全的查询过滤；区分布尔真值与部分求值所需的事实。 |
| 6：rewrite Plan | 构建器与完成后的 Plan 分开，完成计划必有结果；求值使用泛型操作数身份，不再伪造 MIR Value；收缩为实际使用的 Context。 |
| 7：共享内存事实 | MIR 共用基址、偏移与访问区间查询，LoadCSE、循环内存优化和可读性分析消费相同地址事实；共享模地址空间区间相交判断；LIR 从 Spec 地址描述和寄存器版本构建内存依赖。 |
| 8：规则重复编译 | 同一次生成只解析、检查一次规则，同时生成本地折叠与 e-graph 查询。 |
| 9：规则迁移 | 从 schema 生成固定属性的读取、检查、构建与哈希表示；查询保存具体指令见证；比较、布尔扩展、移位掩码、范围检查、截断选择、零偏移规则迁到共享 Spec，删除对应手写分支。 |
| 11：C 目标模型 | 显式 CTargetModel 持有 DataLayout、long 宽度和 char 符号约定；布局、size_t、ptrdiff_t、指针转换通过目标上下文计算。 |
| 12：Wasm 执行配置 | Engine 构造时解析 Auto、后端和内存检查模式；执行入口在翻译前检查宿主能力，交叉输出仍独立。 |
| 13：FastJIT 编译条件 | 专用后端模块与专用 stencil 发射入口按相同的 x86_64 Linux 条件编译。 |

有意保留的边界：

- MIR 的无限制可变访问仍使全部分析失效；LIR 尚未给每一种编辑提供精确摘要。
- 内存查询证明同基址区间不重叠，以及不同入口栈对象内的完整访问不重叠；未知地址保持保守。没有加入完整 MemorySSA 或统一逃逸摘要。
- 属性规则支持固定值签名、数据属性捕获及类型化字面量；包含隐藏 SSA 引用、动态签名、上下文约束的属性没有纳入纯表达式替换。Load/Store 原地改写仍独立。
- C 目前只提供 RV64 Linux 模型，未宣称支持其他 C ABI。

验证：本机 `cargo check --workspace` 已通过；未新增或运行测试，未进行 K230 基准。上述调整的运行时正确性和性能仍需专门验证。

## 一、优先处理的缺陷与接口问题

### 1. C lexer 中未使用的预读接口存在不终止路径

位置：`crates/veloc-c/src/lexer.rs:196`、`:214`。

`peek_nth` 循环调用 `next_token`，而 `next_token` 优先从同一个 `peeked` 缓存弹出 token。第一次缓存达到一个元素后，后续循环只是弹出再压回，长度不再增加。因此从正常初始状态调用 `peek_nth(n >= 1)` 无法结束。这是静态阅读可以确定的问题，目前未找到仓库内调用。

建议删除未使用的 `peek_token`、`peek_nth` 和 `peeked`，同时删除未读取的 `source` 字段。Parser 当前使用克隆 lexer 的另一条预读路径，不依赖这组接口。

`parser.rs:1367` 的 `is_type_specifier_start_at` 为了判断一个 token，额外构造 Parser 并克隆整个 typedef 集合。可把判断抽成接收 `TokenKind` 与借用 typedef 集合的函数；如果以后需要统一 token 缓存，再引入单一 TokenCursor。

### 2. 解释器把“合法但不支持的 MIR”当作 panic

位置：`veloc/interpreter/src/bytecode/compile.rs:1162`、`:1448`；`runtime/program.rs:401`。

`compile_function` 直接返回 `CompiledFunction`，对不支持的值类型使用 `assert!`；部分指令分派仍有 `todo!`。MIR 验证通过不意味着解释器支持该指令或类型，模块构建入口因此仍可能因输入能力不匹配而 panic。

建议编译接口返回 `Result<CompiledFunction, CompileError>`，将后端能力检查与 MIR 合法性验证分开。错误携带函数、指令和类型信息；真正的内部不变量仍可断言。重复的 opcode/type 支持关系可由声明式定义生成，但不必将解释器执行机制与机器码后端合并。

## 二、应当重新设计的基础接口

### 3. MIR pass 管理没有形成一致的分析缓存生命周期

位置：`veloc/optimizer/src/manager.rs:111`、`:153`；`veloc/analyzer/src/manager.rs:38`；`veloc/optimizer/src/pass.rs:9`。

当前事实：

- `run_on_module` 每运行一个函数 pass 都重新创建 AnalysisManager，缓存无法跨相邻函数 pass 复用。
- `run_on_function` 保留同一个 manager，却静默跳过模块 pass；仓库内未找到这个入口的调用。
- `function_mut()` 会立即清空分析缓存，之后返回 PreservedAnalyses 无法恢复已清空的缓存。
- `PreservedAnalyses::changed()` 使用 `!preserve_all` 判断是否修改；“修改了 IR”和“哪些分析仍有效”实际是两件事。
- 未找到 `preserve<A>()` 的调用，当前细粒度 TypeId 集合机制没有实际消费者。

建议：连续函数 pass 组成 FunctionPipeline，每个函数在同一个 session 中执行；模块 pass 是明确的阶段边界。把 `changed` 与分析失效信息分开，逐步由受约束的编辑操作产生变更摘要。删除会静默忽略模块 pass 的闲置入口，或以不能包含模块 pass 的明确类型替代。

落地时需要确认连续函数 pass 不依赖其他函数刚被修改的状态。不要仅交换循环顺序而忽略该约束。MIR 与 LIR 可以共享设计原则，无需强行共享一套泛型 pass manager。

### 4. LIR 已有细粒度失效机制，但通用编辑入口仍全部失效

位置：`veloc/codegen/src/pipeline/session.rs:67`；`veloc/codegen/src/analysis.rs`。

`edit()` 在授予编辑权限时直接应用 WHOLE_FUNCTION，即使最终没有发生修改。只有 `reorder_block` 等受约束接口能精确声明变化。

建议先给高频编辑操作增加明确的影响范围，再让 editor 累积变更摘要。必须保留提前失效或等价的安全保证，涵盖提前返回和 unwinding，不能为了缓存命中率开放未经追踪的可变访问。

这属于编译时间与接口一致性改进，不能据此声称生成代码会更快。

### 5. 规则 guard 应保留结构化中间表示

位置：`veloc/spec/src/rules/expression.rs:57`、`:620`、`:647`、`:671`。

CheckedRule 同时保存 Rust 字符串 guard、guard_slots 和 filters，三套信息分别遍历语法树获得。增加一个 guard 操作时，容易漏掉依赖收集或查询过滤逻辑。

建议先形成经过类型检查的 Predicate 表达式，再从它推导：生成代码、引用的绑定、可安全前推的过滤条件。Rust 字符串应在输出阶段生成。查询执行器和 SSA 匹配器继续保留各自的遍历策略。

### 6. rewrite Plan 不应伪造 MIR Value 作为临时身份

位置：`veloc/optimizer/src/rewrite.rs:82`、`:105`。

`Plan::build` 为调用 evaluate::reduce，将计划输入映射成 `Value::new(...)`。这些编号只表示局部计划中的身份，并不指向 MIR value。当前用途可以工作，但类型含义不准确，未来扩展 evaluate 时容易误用。

建议让求值接口接收可比较的操作数身份及常量查询，直接支持真实 Value 和计划 Input。共享常量语义即可，无需共享实体编号。

另一个可读性问题是 `Plan.result: Option<Input>` 与 `finish` 调用顺序：可将构建过程与完成后的 Plan 分开，使完成后的 result 必然存在。保留实际可删除节点的收益计算，以及构建计划后的无用步骤清理；二者关系到共享表达式的正确成本判断。

当前 Context 只有 FunctionContext 一个实现，是可以收缩的抽象候选；它的优先级低于上述两个问题，不必仅为减少 trait 数量而改。

### 7. 内存事实适合共享，内存 pass 不宜整体合并

位置：`veloc/mir/src/memory.rs:19`；`veloc/optimizer/src/passes/{memory,load_cse,loop_memory,memory_validity}.rs`；`veloc/codegen/src/passes/schedule/graph.rs:37`。

目前分别存在栈对象与偏移解析、逃逸判断、精确地址比较、区间可读性、循环访问与屏障扫描。调度器只能让普通不陷阱读之间自由重排，对写和可能陷阱的访问采用保守内存依赖。

建议建立共享的地址描述与查询：对象或未知基址、字节偏移、访问大小、逃逸信息、别名判断；为每种事实定义明确的失效条件。降低到 LIR 后保留足够的地址来源信息，再用于调度的内存依赖判断。

“内存仍可读”与“内容没有变化”不能混为一个结论。别名分析也不能单独证明陷阱、volatile 或其他可观察副作用可以重排。先统一这些基础事实，再决定是否需要 MemorySSA；不要一步合并 Simplify、Memory、Promote 和 Expression。

## 三、明确可以减少重复工作的地方

### 8. 同一次生成重复解析、检查相同规则

位置：`veloc/spec/src/emit.rs:201`；`veloc/spec/src/rules/equivalence.rs:29`；`veloc/optimizer/build.rs:25`。

构建同时请求 Equivalences 和 LocalFolds，emit 循环分别解析定义并调用 expression::compile。其他输出族已经有类似 IR/target 的局部缓存。

建议在本次生成上下文中只解析、检查一次，两个 emitter 消费同一份 CheckedRule。无需全局缓存或持久缓存系统。这项改动范围小，适合优先处理。

### 9. Simplify 仍有可删除分支与可声明化规则

位置：`veloc/optimizer/src/passes/simplify.rs:127`、`:170`、`:236`。

`fold_casts_and_ranges` 已处理布尔扩展后的零比较，因此后续 `ExtendU(bool) != 0` 分支重复；同一分支中的 `ExtendS(bool)` 不符合现有 ExtendS 的 Integer 类型约束。对合法 MIR，该分支可删除。

进一步迁移地址、比较条件和范围规则，需要共享规则模型支持指令属性，如 IntCC、offset、scale，而不只是 Value 操作数。应从指令 schema 派生属性读写能力，避免为每条规则新增 Rust 特例。

Load/Store 地址改写还涉及副作用及陷阱属性，宜先保留显式的原地改写入口。不能因为匹配形状相似就直接使用纯表达式的替换流程。

### 10. 可删除接口候选需区分内部死代码与公开 API

仓库引用检查得到的候选：

| 位置 | 现状 | 建议 |
| --- | --- | --- |
| `crates/veloc-c/src/lexer.rs` 的预读接口与 source 字段 | 无调用/读取，且预读存在缺陷 | 优先删除 |
| 同文件的 `is_type_specifier`、`is_storage_class`、`is_eof` | 仓库内无调用 | 与 parser 的分类入口一起收敛 |
| `veloc/codegen/src/error.rs:24` 的 TranslatedFunctionNotFound 与构造函数 | 仅自身构造与格式化引用 | 删除不再产生的错误状态 |
| `veloc/optimizer/src/manager.rs:153` 的 run_on_function | 无调用且跳过模块 pass | 随 FunctionPipeline 重构删除或替换 |
| `veloc/interpreter/src/runtime/program.rs:118` 的 compiled_funcs | 仓库内无调用 | 若无对外枚举需求，可删除 |

“仓库内无调用”不证明公开库接口没有外部用户。对外枚举和 builder 便利接口应根据项目 API 范围判断，不能机械清理所有零引用 pub 函数。

C 前端的整模块 `allow(dead_code)` 掩盖了部分闲置代码。清理后缩小豁免范围；生成代码、按目标启用的代码需要另行判断。

## 四、扩展前值得整理的边界

### 11. C 类型布局需要显式的目标上下文

位置：`crates/veloc-c/src/types.rs:43`、`:186`；`crates/veloc-c/src/main.rs:51`。

指针大小、size_t 和 long 等采用当前目标的固定约定。CLI 明确只支持 riscv64 Linux LP64D，因此这不是当前已支持目标的错误。

在扩展目标之前，应引入明确的 CTargetModel，由前端使用它决定语言类型的大小、对齐和默认 signedness，再向目标 ABI 请求传参分类。DataLayout 描述存储布局，不应独自承担 C 语言类型规则。

### 12. Wasm 执行策略适合一次解析为完整计划

位置：`crates/veloc-wasm/src/engine.rs`；`crates/veloc-wasm/src/module/mod.rs:328`、`:464`。

Auto 在 prepare 内转成 Jit；内存检查模式由 engine 另行决定；宿主架构与目标的执行匹配到 load 时才检查。一次运行的决定分散在多个阶段。

建议为执行路径提前解析出明确的计划，包括后端、目标、内存检查方式和宿主能力。交叉编译输出与本机加载执行保持不同要求。若 Auto 长期只是 Jit 的别名，可删掉该选项；若要保留，先明确选择或回退策略。

### 13. FastJIT 的目标条件应与模块编译条件一致

位置：`veloc/fastjit/src/lib.rs:6`、`:54`。

编译入口只在 x86_64 Linux 使用后端，但 image/x86_64 模块始终编译。在其他宿主上出现的无调用代码不能据此认定整个后端无用。

若继续保持仅支持该宿主的语义，可按相同 cfg 启用后端模块及专用依赖，保留通用 stencil 接口。若要支持交叉输出，应显式扩展目标选择，不能删除后端实现来消除警告。

## 不建议为了减少行数删除的结构

- e-graph 的等价类、增量依赖与重建职责：应根据规则和测量优化，不能仅因为包含索引就视为重复 IR。
- 调度图的依赖边：SSA 定义使用链不足以表示物理寄存器、内存及其他资源依赖。
- 对共享节点的收益计算、支配检查及陷阱约束：它们决定替换是否安全、是否真的减少工作。
- 保守的分析深度或工作量上限：需要按用途判断，不能一概当作无用预算。
- 优化流水线中重复的清理 pass：前一轮转换可能制造新机会，应先标明阶段后置条件再删。
- `.spec` 或生成器使用的接口：例如 LIR 的 `with_constraints`、目标发射及栈帧接口，普通 Rust 调用搜索可能漏报。

## 建议实施顺序

1. 小范围清理：lexer 闲置缓存和接口、冗余 Simplify 分支、无产生路径的错误类型；消除规则重复编译。
2. 基础设施：统一 MIR FunctionPipeline 的分析生命周期，拆开 changed 与 invalidation；逐步改善 LIR 编辑精度。
3. 规则模型：结构化 Predicate、属性匹配、去掉伪造 Value；完成的 Plan 显式持有结果。
4. 运行时健壮性：解释器返回能力错误，Wasm 提前解析执行计划。
5. 优化扩展：共享内存事实并贯穿 lowering，再测量别名信息对调度及 K230 的实际收益；扩展 C 目标前引入 CTargetModel。

前三步主要降低维护成本与编译期重复工作。是否提升 CoreMark 得分，需要单独测量；本次审查没有性能结论。
