//! Checked machine-description data projected from the shared Spec syntax.

#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub defs: Vec<Def>,
}

/// Machine description declarations
#[derive(Debug, Clone, PartialEq)]
pub enum Def {
    /// Checked single-root selection candidate.
    SelectRule(SelectRuleDef),
    /// Resolved register constant from the OpSpec target schema.
    Reg(RegDef),
    /// Resolved RegisterClass constant from the OpSpec target schema.
    RegClass(RegClassDef),
    /// Named composition of declared predicates.
    Extractor(ExtractorDef),
    /// Target CPU capability and its dependencies.
    Feature(FeatureDef),
    /// Named target CPU model.
    Cpu(CpuDef),
    /// Target calling convention metadata.
    Abi(AbiDef),
    /// Explicit predicate signature implemented by the selection host.
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
    pub requires: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CpuDef {
    pub name: String,
    pub features: Vec<String>,
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
pub struct SelectRuleDef {
    pub opcode: String,
    pub type_args: Vec<Vec<String>>,
    pub schema: String,
    pub definitions: Vec<DefMatch>,
    /// Named root fields and their pure matching constraints.
    pub fields: Vec<PatternArg>,
    /// Fresh registers allocated only after matching succeeds.
    pub temps: Vec<(String, String)>,
    /// Deferred instruction construction, committed in order.
    pub builds: Vec<Constructor>,
}

/// One fallible use-def lookup, ordered after the definitions it depends on.
#[derive(Debug, Clone, PartialEq)]
pub struct DefMatch {
    pub name: String,
    pub input: String,
    pub opcode: String,
    pub type_args: Vec<Vec<String>>,
    pub schema: String,
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
    /// A value binding constrained by a resolved logical type domain.
    Typed { name: String, types: Vec<String> },
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
