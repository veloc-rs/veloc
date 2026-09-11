//! Memory semantics shared by analyses and lowerings. An Access is a query
//! result, not a second authoritative copy of instruction operands.
use crate::{Function, Inst, MemFlags, Type, Value};

#[derive(Debug, Clone, Copy)]
pub struct Access {
    pub ptr: Value,
    pub offset: i64,
    pub ty: Type,
    /// A write stores this SSA value; None denotes a read.
    pub stored: Option<Value>,
    pub flags: MemFlags,
}

impl Access {
    /// Pointer width is deliberately supplied by the target, not assumed to
    /// equal the host pointer width. Scalable/opaque representations are unknown.
    pub fn bytes(self, pointer_bytes: Option<u32>) -> Option<u32> {
        if self.ty.is_ptr() {
            pointer_bytes
        } else {
            self.ty.fixed_size_bytes()
        }
    }
}

impl Function {
    pub fn memory_access(&self, inst: Inst) -> Option<Access> {
        self.dfg().inst(inst).memory_access(
            self.dfg(),
            self.dfg()
                .first_result(inst)
                .map(|v| self.dfg().value_type(v)),
        )
    }
}
