//! ISLE AST (Abstract Syntax Tree) Definitions
//!
//! DSL v2 将规则拆分为 combine/select/peephole 三层，并显式表达
//! 覆盖、约束、节点绑定和 pseudo 指令。

/// ISLE 工作空间/模块
#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub defs: Vec<Def>,
}

/// ISLE 文件顶层定义
#[derive(Debug, Clone, PartialEq)]
pub enum Def {
    /// 类型定义: (type Name (primitive Type))
    Type(TypeDef),
    /// 选择规则: (select-rule ...)
    SelectRule(SelectRuleDef),
    /// 归一化/重写规则: (rewrite-rule ...)
    RewriteRule(RewriteRuleDef),
    /// 组合规则: (combine-rule ...)
    CombineRule(CombineRuleDef),
    /// 目标后期 peephole 规则: (peephole-rule ...)
    PeepholeRule(PeepholeRuleDef),
    /// Resolved register constant from the OpSpec target schema.
    Reg(RegDef),
    /// Resolved RegisterClass constant from the OpSpec target schema.
    RegClass(RegClassDef),
    /// 提取器定义: (def-extractor (Name args...) body)
    Extractor(ExtractorDef),
    /// CPU 特性定义: (def-feature Name "doc")
    Feature(FeatureDef),
    /// CPU 模型定义: (def-cpu "name" (features ...) (limitations ...))
    Cpu(CpuDef),
    /// ABI 描述: (def-abi Name ...)
    Abi(AbiDef),
    /// 外部方法声明: (decl Name (params...))
    Decl(DeclDef),
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeclDef {
    pub name: String,
    pub params: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureDef {
    pub name: String,
    pub doc: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CpuDef {
    pub name: String,
    pub features: Vec<String>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbiDef {
    pub name: String,
    pub arch: String,
    pub stack: AbiStackDef,
    pub args: Vec<AbiClassRegsDef>,
    pub returns: Vec<AbiClassRegsDef>,
    pub preserved: Vec<AbiPreservedSetDef>,
    pub classifier: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AbiStackDef {
    pub align: Option<u32>,
    pub incoming_base: Option<(String, i32)>,
    pub outgoing_slot: Option<(u32, u32)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbiClassRegsDef {
    pub class: String,
    pub regs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbiPreservedSetDef {
    pub bank: String,
    pub regs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExtractorDef {
    pub name: String,
    pub args: Vec<String>,
    pub body: Pattern,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeDef {
    pub name: String,
    pub kind: TypeKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeKind {
    /// 原始类型
    Primitive(String),
    /// 寄存器类
    RegisterClass(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchKind {
    Single,
    Pair,
    Sequence,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct RuleAttrs {
    pub root: Option<String>,
    pub predicates: Vec<PredicateExpr>,
    pub covers: Vec<String>,
    pub cost: Option<i64>,
    pub priority: Option<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectRuleDef {
    pub attrs: RuleAttrs,
    /// Fresh rule-local registers, each inheriting a bound value's type and bank.
    pub temps: Vec<(String, String)>,
    pub patterns: Vec<Pattern>,
    pub emit: Constructor,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RewriteRuleDef {
    pub attrs: RuleAttrs,
    pub patterns: Vec<Pattern>,
    pub replace: Constructor,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CombineRuleDef {
    pub attrs: RuleAttrs,
    pub match_kind: MatchKind,
    pub patterns: Vec<Pattern>,
    pub replace: Constructor,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PeepholeRuleDef {
    pub attrs: RuleAttrs,
    pub patterns: Vec<Pattern>,
    pub replace: Constructor,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RegDef {
    pub name: String,
    pub size: u32,
    pub alias: Option<RegisterAlias>,
    pub id: u32,
    pub hw_enc: u32,
    pub reserved: bool,
    pub roles: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RegisterAlias {
    pub base: String,
    pub offset: u32,
    pub write: RegisterWrite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterWrite {
    Preserve,
    ZeroExtend,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RegClassDef {
    pub name: String,
    pub regs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OperandConstraint {
    /// 普通使用: (use $name)
    Use(String),
    /// 固定寄存器使用: (use (fixed REG $name))
    FixedUse { reg: String, src: String },
    /// 普通定义: (def $name)
    Def(String),
    /// 立即数使用: (imm $name)
    Imm(String),
    /// 基本块目标: (block $name)
    Block(String),
    /// 全局符号目标: (global $name)
    Global(String),
    /// 栈槽目标: (stackslot $name)
    StackSlot(String),
}

/// 规则谓词中的参数
#[derive(Debug, Clone, PartialEq)]
pub enum PredicateArg {
    Variable(String),
    Node(String),
    Ident(String),
    Int(i64),
}

/// 规则约束谓词
#[derive(Debug, Clone, PartialEq)]
pub struct PredicateExpr {
    pub name: String,
    pub args: Vec<PredicateArg>,
}

/// 操作码模式参数
#[derive(Debug, Clone, PartialEq)]
pub enum PatternArg {
    /// 位置参数
    Positional(Pattern),
    /// 命名字段参数: (field pattern)
    Named { name: String, pattern: Box<Pattern> },
}

/// 模式匹配表达式
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    /// Schema 模式: (schema SchemaName OPCODE args...)
    Schema {
        schema: String,
        opcode: String,
        args: Vec<PatternArg>,
    },
    /// 操作码模式: (OPCODE args...)
    Opcode {
        opcode: String,
        ty: Option<String>,
        args: Vec<PatternArg>,
    },
    /// 变量绑定: $name
    Variable(String),
    /// 整数常量
    IntConst(i64),
    /// 条件码
    CondCode(CondCode),
    /// 栈槽
    StackSlot(Box<Pattern>),
    /// 目标块
    Block(String),
    /// 与模式: (and p1 p2 ...)
    And(Vec<Pattern>),
    /// 节点绑定: pattern @node
    NodeBind { inner: Box<Pattern>, node: String },
}

impl Pattern {
    pub fn strip_node_binds(&self) -> &Pattern {
        match self {
            Pattern::NodeBind { inner, .. } => inner.strip_node_binds(),
            _ => self,
        }
    }
}

/// 条件码
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CondCode {
    // 整数比较
    E,
    NE,
    L,
    LE,
    G,
    GE,
    B,
    BE,
    A,
    AE,
}

/// 构造函数表达式
#[derive(Debug, Clone, PartialEq)]
pub enum Constructor {
    /// 目标指令 / generic 构造器
    Inst {
        opcode: String,
        args: Vec<Constructor>,
    },
    /// 变量引用
    Variable(String),
    /// 立即数
    Imm(i64),
    /// 物理寄存器: (reg Name)
    Reg(String),
}
