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
    /// Value is a block parameter.
    Param(Block),
}

/// A reference to a basic block.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Block(pub u32);
entity_impl!(Block, "block");

/// A reference to a block call (branch destination with arguments).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockCall(pub u32);
entity_impl!(BlockCall, "bc");

/// Data for a block call: the target block and arguments.
#[derive(Debug, Clone, Copy)]
pub struct BlockCallData {
    pub block: Block,
    pub args: ValueList,
}

/// A reference to a jump table.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JumpTable(pub u32);
entity_impl!(JumpTable, "jt");

/// Data for a jump table: list of block calls.
#[derive(Debug, Clone)]
pub struct JumpTableData {
    pub targets: alloc::vec::Vec<BlockCall>,
}

/// A reference to a stack slot.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StackSlot(pub u32);
entity_impl!(StackSlot, "ss");

/// A reference to a module identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModuleId(pub u32);
entity_impl!(ModuleId, "module");

/// A reference to a function identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FuncId(pub u32);
entity_impl!(FuncId, "func");

/// A reference to a signature identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SigId(pub u32);
entity_impl!(SigId, "sig");

/// A reference to a variable (SSA variable used in function building).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Variable(pub u32);
entity_impl!(Variable, "var");
