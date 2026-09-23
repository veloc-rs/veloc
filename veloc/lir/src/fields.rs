//! Common instruction fields. Positional values exist only at construction and
//! selection boundaries; calls and successors have explicit storage and access.
use crate::FieldValueRef;
use crate::{CallInfo, EdgeId, FieldValue, StackSlot, SymbolId};
use alloc::vec::Vec;
use veloc_collections::{Pool, PoolId};
use veloc_types::{FloatCC, IntCC};

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
        }
    }
}

/// One complete payload per instruction. Handles are private implementation
/// details: copying a handle does not create another owner.
#[derive(Debug, Clone, Default)]
pub(crate) enum Fields {
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
    Switch(PoolId<Vec<EdgeId>>),
    Call(PoolId<CallData>),
}
const _: () = assert!(core::mem::size_of::<Fields>() == 16);

#[derive(Debug, Clone)]
pub(crate) struct CallData {
    target: Option<SymbolId>,
    info: CallInfo,
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
        &mut self.calls.get_mut(*id).info
    }
    /// Boundary adapter. Allocate a fresh owned payload, never adopt or repair
    /// an existing call handle. Unsupported shapes fail before changing an inst.
    pub(crate) fn pack(&mut self, values: impl IntoIterator<Item = FieldValue>) -> Fields {
        use FieldValue as V;
        let mut values = values.into_iter().fuse();
        let Some(first) = values.next() else {
            return Fields::None;
        };
        let second = values.next();
        // Recognize the complete shape before allocating any pooled payload.
        match (first, second) {
            (V::Imm(v), None) => Fields::Imm(v),
            (V::FImm(v), None) => Fields::FImm(v),
            (V::StackSlot(v), None) => Fields::StackSlot(v),
            (V::CallFrame(v), None) => Fields::CallFrame(v),
            (V::IntCC(v), None) => Fields::IntCC(v),
            (V::FloatCC(v), None) => Fields::FloatCC(v),
            (V::Global(v), None) => Fields::Symbol(v),
            (V::Edge(a), None) => Fields::Jump([a]),
            (V::Edge(a), Some(V::Edge(b))) => {
                let Some(third) = values.next() else {
                    return Fields::Branch([a, b]);
                };
                let edges = [V::Edge(a), V::Edge(b), third]
                    .into_iter()
                    .chain(values)
                    .map(|value| match value {
                        V::Edge(edge) => edge,
                        _ => panic!("unsupported instruction field shape"),
                    })
                    .collect();
                return Fields::Switch(self.switches.push(edges));
            }
            (V::Call(info), None) => return self.call(None, info),
            (V::Global(target), Some(V::Call(info))) => {
                assert!(
                    values.next().is_none(),
                    "unsupported instruction field shape"
                );
                return self.call(Some(target), info);
            }
            _ => panic!("unsupported instruction field shape"),
        }
    }
    fn call(&mut self, target: Option<SymbolId>, info: CallInfo) -> Fields {
        Fields::Call(self.calls.push(CallData { target, info }))
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
            Fields::Switch(id) => self.switches.get_mut(*id),
            _ => &mut [],
        }
    }
    /// Consume the payload ownership. The caller must replace the instruction's
    /// field before releasing its old payload.
    pub(crate) fn remove(&mut self, fields: Fields) {
        match fields {
            Fields::Call(id) => self.calls.remove(id),
            Fields::Switch(id) => self.switches.remove(id),
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
            Fields::Branch(_) => 2,
            Fields::Switch(id) => self.pools.switches.get(*id).len(),
            Fields::Call(id) => 1 + usize::from(self.pools.calls.get(*id).target.is_some()),
            _ => 1,
        }
    }
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
    pub fn successors(self) -> &'a [EdgeId] {
        match self.fields {
            Fields::Jump(edges) => edges,
            Fields::Branch(edges) => edges,
            Fields::Switch(id) => self.pools.switches.get(*id),
            _ => &[],
        }
    }
    pub fn call_info(self) -> Option<&'a CallInfo> {
        match self.fields {
            Fields::Call(id) => Some(&self.pools.calls.get(*id).info),
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
            Fields::CallFrame(v) => V::CallFrame(v),
            Fields::IntCC(v) => V::IntCC(v),
            Fields::FloatCC(v) => V::FloatCC(v),
            Fields::Symbol(v) => V::Global(v),
            Fields::Jump(_) | Fields::Branch(_) | Fields::Switch(_) => {
                V::Edge(&self.successors()[index])
            }
            Fields::Call(id) => {
                let data = self.pools.calls.get(*id);
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
