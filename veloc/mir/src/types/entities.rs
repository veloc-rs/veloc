//! MIR entity handles and their associated data.

use super::Type;
use cranelift_entity::{EntityList, ListPool, entity_impl};

/// Value 列表的内存池
pub type ValueListPool = ListPool<Value>;
/// Value 列表（使用 cranelift-entity 的紧凑表示）
pub type ValueList = EntityList<Value>;

/// A reference to a Value.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Value(pub u32);
entity_impl!(Value, "v");

/// Data about a value: its type and definition.
#[derive(Debug, Clone)]
pub struct ValueData {
    pub ty: Type,
    pub def: ValueDef,
}

/// Definition of a value.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ValueDef {
    /// Value is defined by an instruction.
    Inst(crate::Inst),
    /// An input supplied by the caller, available throughout the function.
    FunctionParam(ParamIndex),
    /// A value supplied by incoming control-flow edges.
    BlockParam(Block),
    /// An immutable literal, available independently of control flow.
    Const(ConstId),
}

/// Position of a function parameter in its signature.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ParamIndex(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstId(pub(crate) u32);
entity_impl!(ConstId, "const");

/// A reference to a basic block.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Block(pub u32);
entity_impl!(Block, "block");

/// Owned successor data for construction and snapshots. Installed arguments live
/// in the instruction's operand array: borrow them through `Successor` and edit
/// them through `EdgeRef` rather than copying this container.
#[derive(Debug, Clone)]
pub struct SuccessorData {
    pub block: Block,
    pub args: smallvec::SmallVec<[Value; 4]>,
}

impl SuccessorData {
    pub fn new(block: Block, args: &[Value]) -> Self {
        Self {
            block,
            args: smallvec::SmallVec::from_slice(args),
        }
    }

    pub fn set_args(&mut self, args: &[Value]) {
        self.args.clear();
        self.args.extend_from_slice(args);
    }

    /// Set a construction-time argument, filling incomplete earlier positions.
    /// Explicit validation checks the completed edge's parameter contract.
    pub(crate) fn set_arg(&mut self, index: usize, value: Value) {
        if index >= self.args.len() {
            let len = index.checked_add(1).expect("too many successor arguments");
            self.args.resize(len, value);
        } else {
            self.args[index] = value;
        }
    }
}

/// A reference to a module identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModuleId(pub u32);
entity_impl!(ModuleId, "module");

/// A reference to a function identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FuncId(pub u32);
entity_impl!(FuncId, "func");

/// A module data symbol.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GlobalId(pub u32);
entity_impl!(GlobalId, "global");

/// A reference to a variable (SSA variable used in function building).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Variable(pub u32);
entity_impl!(Variable, "var");
