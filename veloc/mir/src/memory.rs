//! Memory semantics shared by analyses and lowerings. An Access is a query
//! result, not a second authoritative copy of instruction operands.
use crate::{Function, Inst, Value};

pub use crate::inst::MemoryAccess as Access;

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
    /// Resolve a bounded chain of constant byte offsets to a once-per-invocation
    /// allocation. Unknown provenance and dynamic allocations stay unknown.
    pub fn stack_address(&self, mut ptr: Value) -> Option<(Inst, i64)> {
        let mut offset = 0i64;
        for _ in 0..64 {
            let inst = self.dfg().value_inst(ptr)?;
            match self.dfg().inst(inst) {
                crate::InstView::Alloca { .. }
                    if self.layout().inst_block(inst) == self.entry_block =>
                {
                    return Some((inst, offset));
                }
                crate::InstView::PtrOffset {
                    ptr: base,
                    offset: delta,
                } => {
                    ptr = base;
                    offset = offset.checked_add(i64::from(delta))?;
                }
                _ => return None,
            }
        }
        None
    }

    /// A complete, aligned access inside a known live entry object.
    /// This is a bounds proof, not a claim that the bytes are initialized.
    pub fn stack_access(&self, access: Access, pointer_bytes: Option<u32>) -> Option<(Inst, u32)> {
        let (object, offset) = self.stack_address(access.ptr)?;
        let offset = u32::try_from(offset.checked_add(access.offset)?).ok()?;
        let crate::InstView::Alloca { size, align } = self.dfg().inst(object) else {
            unreachable!("stack address ends at an allocation")
        };
        let bytes = access.bytes(pointer_bytes)?;
        (bytes != 0
            && offset.checked_add(bytes)? <= size
            && align >= access.flags.alignment()
            && offset % access.flags.alignment() == 0)
            .then_some((object, offset))
    }

    pub fn memory_access(&self, inst: Inst) -> Option<Access> {
        self.dfg()
            .inst(inst)
            .query(self.dfg(), self.dfg().inst_results(inst))
    }
}
