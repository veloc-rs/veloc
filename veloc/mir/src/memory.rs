//! Memory semantics shared by analyses and lowerings. An Access is a query
//! result, not a second authoritative copy of instruction operands.
use crate::{FuncBody, Inst, Value};
use veloc_types::DataLayout;

pub use crate::inst::MemoryAccess as Access;

impl Access {
    /// Access width follows the supplied representation, never the host layout.
    /// Scalable or unlisted representations have no known fixed access width.
    pub fn bytes(self, layout: &DataLayout) -> Option<u32> {
        layout.layout_of(self.ty)?.store_size.fixed_bytes()
    }
}

impl FuncBody {
    /// Resolve a bounded chain of constant byte offsets to a once-per-invocation
    /// allocation. Unknown provenance and dynamic allocations stay unknown.
    pub fn stack_address(&self, mut ptr: Value) -> Option<(Inst, i64)> {
        let mut offset = 0i64;
        for _ in 0..64 {
            let inst = self.dfg().value_inst(ptr)?;
            match self.dfg().inst(inst) {
                crate::InstView::Alloca { .. }
                    if self.layout().inst_block(inst) == Some(self.entry_block()) =>
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
    pub fn stack_access(&self, access: Access, layout: &DataLayout) -> Option<(Inst, u32)> {
        let (object, offset) = self.stack_address(access.ptr)?;
        let offset = u32::try_from(offset.checked_add(access.offset)?).ok()?;
        let crate::InstView::Alloca { size, align } = self.dfg().inst(object) else {
            unreachable!("stack address ends at an allocation")
        };
        let bytes = access.bytes(layout)?;
        (bytes != 0
            && offset.checked_add(bytes)? <= size
            && align >= access.flags.alignment()
            && offset % access.flags.alignment() == 0)
            .then_some((object, offset))
    }
}
