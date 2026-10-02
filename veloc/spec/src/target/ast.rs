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
    /// Target-wide scheduling category referenced by instructions and CPU costs.
    ScheduleClass(ScheduleClassDef),
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
    pub schedule: CpuSchedule,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScheduleClassDef {
    pub name: String,
    pub doc: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CpuSchedule {
    pub issue_width: u32,
    pub resources: Vec<ScheduleResource>,
    pub classes: Vec<ScheduleCost>,
}

impl Default for CpuSchedule {
    fn default() -> Self {
        Self {
            issue_width: 1,
            resources: Vec::new(),
            classes: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScheduleResource {
    pub name: String,
    pub units: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScheduleCost {
    pub class: String,
    pub resource: String,
    pub latency: u32,
    pub occupancy: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbiDef {
    pub name: String,
    pub arch: String,
    pub stack: AbiStackDef,
    pub args: Vec<AbiRuleDef>,
    pub returns: Vec<AbiRuleDef>,
    pub preserved: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AbiStackDef {
    pub align: Option<u32>,
    pub reserved: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbiRuleDef {
    pub types: Vec<String>,
    pub action: AbiActionDef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbiActionDef {
    Reg {
        regs: Vec<String>,
        shadows: Vec<String>,
    },
    Stack {
        size: u32,
        align: u32,
    },
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
    /// Exact operand category whose values reside in this non-renamable root.
    pub state_type: Option<String>,
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

/// Operand domains are kept separate from the storage type of an attribute.
#[derive(Debug, Clone, PartialEq)]
pub enum OperandConstraint {
    Use(String),
    Def(String),
    Attribute(String, AttributeKind),
}

impl OperandConstraint {
    pub fn name(&self) -> &str {
        match self {
            Self::Use(name) | Self::Def(name) | Self::Attribute(name, _) => name,
        }
    }
}

/// Machine attribute types shared by contract checking and code generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeKind {
    Imm,
    MemFlags,
    Block,
    Global,
    StackSlot,
    Call,
}

pub struct AttributeType {
    pub spec_type: &'static str,
    pub rust_type: &'static str,
    pub field_variant: &'static str,
}

impl AttributeKind {
    pub fn from_spec(ty: &str) -> Option<Self> {
        [
            Self::Imm,
            Self::MemFlags,
            Self::Block,
            Self::Global,
            Self::StackSlot,
            Self::Call,
        ]
        .into_iter()
        .find(|kind| kind.description().spec_type == ty)
    }

    /// The Spec type, constructor parameter and stored field describe one type.
    pub fn description(self) -> AttributeType {
        let (spec_type, rust_type, field_variant) = match self {
            Self::Imm => ("i64", "i64", "Imm"),
            Self::MemFlags => ("MemFlags", "veloc_lir::MemFlags", "MemFlags"),
            Self::Block => ("Successor", "veloc_lir::EdgeId", "Edge"),
            Self::Global => ("Global", "veloc_lir::SymbolId", "Global"),
            Self::StackSlot => ("StackSlot", "veloc_lir::StackSlot", "StackSlot"),
            Self::Call => ("CallInfo", "veloc_lir::CallInfo", "Call"),
        };
        AttributeType {
            spec_type,
            rust_type,
            field_variant,
        }
    }
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
    /// An integer attribute representable in the given immediate width.
    IntRange { bits: u8, signed: bool },
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
