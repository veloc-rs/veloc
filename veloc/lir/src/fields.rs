//! Instruction-owned payloads and borrowed field views. Generated builders
//! construct payloads directly; dynamic rules use transient positional values.
use crate::FieldValueRef;
use crate::{CallInfo, EdgeId, FieldValue, StackSlot, SymbolId};
use alloc::vec::Vec;
use veloc_collections::{Pool, PoolId};
use veloc_types::{FloatCC, IntCC, MemFlags};

impl FieldValueRef<'_> {
    fn to_owned(self) -> FieldValue {
        match self {
            Self::Imm(v) => FieldValue::Imm(*v),
            Self::FImm(v) => FieldValue::FImm(*v),
            Self::Edge(v) => FieldValue::Edge(*v),
            Self::StackSlot(v) => FieldValue::StackSlot(*v),
            Self::CallFrame(v) => FieldValue::CallFrame(*v),
            Self::IntCC(v) => FieldValue::IntCC(*v),
            Self::FloatCC(v) => FieldValue::FloatCC(*v),
            Self::Global(v) => FieldValue::Global(*v),
            Self::Call(v) => FieldValue::Call(v.clone()),
            Self::MemFlags(v) => FieldValue::MemFlags(*v),
        }
    }
}

/// One complete payload per instruction. Handles are private implementation
/// details: copying a handle does not create another owner.
#[derive(Debug, Default)]
pub enum Fields {
    #[default]
    None,
    Imm(i64),
    FImm(f64),
    StackSlot(StackSlot),
    CallFrame(crate::CallFrameId),
    IntCC(IntCC),
    FloatCC(FloatCC),
    Symbol(SymbolId),
    Jump([EdgeId; 1]),
    Branch([EdgeId; 2]),
    Switch(SwitchFields),
    Call(CallFields),
    Memory {
        flags: MemFlags,
        offset: i64,
    },
    StackMemory {
        flags: MemFlags,
        slot: StackSlot,
    },
}
const _: () = assert!(core::mem::size_of::<Fields>() == 16);

/// Opaque, non-cloneable ownership of a call payload.
#[derive(Debug)]
pub struct CallFields(PoolId<CallData>);
/// Opaque, non-cloneable ownership of a successor list.
#[derive(Debug)]
pub struct SwitchFields(PoolId<Vec<EdgeId>>);

impl Fields {
    /// Pooled handles may only be copied when their pools are also cloned.
    pub(crate) fn clone_handle(&self) -> Self {
        match self {
            Self::None => Self::None,
            Self::Imm(v) => Self::Imm(*v),
            Self::FImm(v) => Self::FImm(*v),
            Self::StackSlot(v) => Self::StackSlot(*v),
            Self::CallFrame(v) => Self::CallFrame(*v),
            Self::IntCC(v) => Self::IntCC(*v),
            Self::FloatCC(v) => Self::FloatCC(*v),
            Self::Symbol(v) => Self::Symbol(*v),
            Self::Jump(v) => Self::Jump(*v),
            Self::Branch(v) => Self::Branch(*v),
            Self::Switch(id) => Self::Switch(SwitchFields(id.0)),
            Self::Call(id) => Self::Call(CallFields(id.0)),
            Self::Memory { offset, flags } => Self::Memory {
                offset: *offset,
                flags: *flags,
            },
            Self::StackMemory { slot, flags } => Self::StackMemory {
                slot: *slot,
                flags: *flags,
            },
        }
    }
}

#[derive(Debug, Clone)]
struct CallData {
    target: Option<SymbolId>,
    info: CallInfo,
}

/// Allocation hooks used by generated payload constructors.
pub trait FieldBuild {
    fn call_fields(&mut self, target: Option<SymbolId>, info: CallInfo) -> Fields;
    fn switch_fields(&mut self, edges: &[EdgeId]) -> Fields;
}

/// Only large payloads need pools. Scalars and ordinary branches allocate none.
#[derive(Debug, Clone, Default)]
pub(crate) struct FieldPools {
    calls: Pool<CallData>,
    switches: Pool<Vec<EdgeId>>,
}
impl FieldPools {
    pub(crate) fn call_info_mut(&mut self, fields: &Fields) -> &mut CallInfo {
        let Fields::Call(id) = fields else {
            panic!("expected call fields")
        };
        &mut self.calls.get_mut(id.0).info
    }
    pub(crate) fn switch(&mut self, edges: &[EdgeId]) -> Fields {
        match edges {
            [a] => Fields::Jump([*a]),
            [a, b] => Fields::Branch([*a, *b]),
            _ => Fields::Switch(SwitchFields(self.switches.push(edges.to_vec()))),
        }
    }
    pub(crate) fn copy(&mut self, fields: &Fields) -> Fields {
        match fields {
            Fields::Call(id) => {
                let data = self.calls.get(id.0).clone();
                Fields::Call(CallFields(self.calls.push(data)))
            }
            Fields::Switch(id) => {
                let edges = self.switches.get(id.0).clone();
                Fields::Switch(SwitchFields(self.switches.push(edges)))
            }
            _ => fields.clone_handle(),
        }
    }
    pub(crate) fn call(&mut self, target: Option<SymbolId>, info: CallInfo) -> Fields {
        Fields::Call(CallFields(self.calls.push(CallData { target, info })))
    }
    pub(crate) fn view<'a>(&'a self, fields: &'a Fields) -> FieldView<'a> {
        FieldView {
            fields,
            pools: self,
        }
    }
    pub(crate) fn successors_mut<'a>(&'a mut self, fields: &'a mut Fields) -> &'a mut [EdgeId] {
        match fields {
            Fields::Jump(edges) => edges,
            Fields::Branch(edges) => edges,
            Fields::Switch(id) => self.switches.get_mut(id.0),
            _ => &mut [],
        }
    }
    /// Consume the payload ownership. The caller must replace the instruction's
    /// field before releasing its old payload.
    pub(crate) fn remove(&mut self, fields: Fields) {
        match fields {
            Fields::Call(id) => self.calls.remove(id.0),
            Fields::Switch(id) => self.switches.remove(id.0),
            _ => {}
        }
    }
}

/// Transient borrowed adapter for generated views and the selection VM.
/// This is not another stored record.
#[derive(Clone, Copy)]
pub struct FieldView<'a> {
    fields: &'a Fields,
    pools: &'a FieldPools,
}
impl core::fmt::Debug for FieldView<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list()
            .entries((0..self.len()).map(|i| self.read(i)))
            .finish()
    }
}
impl<'a> FieldView<'a> {
    pub fn len(self) -> usize {
        match self.fields {
            Fields::None => 0,
            Fields::Branch(_) | Fields::Memory { .. } | Fields::StackMemory { .. } => 2,
            Fields::Switch(id) => self.pools.switches.get(id.0).len(),
            Fields::Call(id) => 1 + usize::from(self.pools.calls.get(id.0).target.is_some()),
            _ => 1,
        }
    }
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
    pub fn mem_flags(self) -> Option<MemFlags> {
        match self.fields {
            Fields::Memory { flags, .. } | Fields::StackMemory { flags, .. } => Some(*flags),
            _ => None,
        }
    }
    pub fn successors(self) -> &'a [EdgeId] {
        match self.fields {
            Fields::Jump(edges) => edges,
            Fields::Branch(edges) => edges,
            Fields::Switch(id) => self.pools.switches.get(id.0),
            _ => &[],
        }
    }
    pub fn call_info(self) -> Option<&'a CallInfo> {
        match self.fields {
            Fields::Call(id) => Some(&self.pools.calls.get(id.0).info),
            _ => None,
        }
    }
    pub fn read(self, index: usize) -> FieldValueRef<'a> {
        use FieldValueRef as V;
        assert!(index < self.len(), "field index out of bounds");
        match self.fields {
            Fields::Imm(v) => V::Imm(v),
            Fields::FImm(v) => V::FImm(v),
            Fields::StackSlot(v) => V::StackSlot(v),
            Fields::Memory { flags, offset } => match index {
                0 => V::Imm(offset),
                _ => V::MemFlags(flags),
            },
            Fields::StackMemory { flags, slot } => match index {
                0 => V::StackSlot(slot),
                _ => V::MemFlags(flags),
            },
            Fields::CallFrame(v) => V::CallFrame(v),
            Fields::IntCC(v) => V::IntCC(v),
            Fields::FloatCC(v) => V::FloatCC(v),
            Fields::Symbol(v) => V::Global(v),
            Fields::Jump(_) | Fields::Branch(_) | Fields::Switch(_) => {
                V::Edge(&self.successors()[index])
            }
            Fields::Call(id) => {
                let data = self.pools.calls.get(id.0);
                match (&data.target, index) {
                    (Some(target), 0) => V::Global(target),
                    _ => V::Call(&data.info),
                }
            }
            Fields::None => unreachable!(),
        }
    }
    /// Owned values are needed only across mutation boundaries.
    pub fn at(self, index: usize) -> FieldValue {
        self.read(index).to_owned()
    }
}
