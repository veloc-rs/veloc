//! Shared address normalization, access ranges and stack bounds proofs.
//! Instruction operands and access properties remain in InstView and MemFlags.
use crate::{FuncBody, Inst, InstView, MemFlags, Value};
use veloc_types::DataLayout;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Address {
    pub base: Value,
    pub offset: i64,
}
pub type Location = veloc_types::MemoryLocation<Value>;

impl FuncBody {
    /// Fixed byte range of an ordinary load or store in the target layout.
    /// Unknown sizes or addresses return None, not a proof of no memory effect.
    pub fn memory_location(&self, inst: Inst, layout: &DataLayout) -> Option<Location> {
        let dfg = self.dfg();
        let (ptr, offset, value) = match dfg.inst(inst) {
            InstView::Load { ptr, offset, .. } => (ptr, offset, dfg.first_result(inst)?),
            InstView::Store {
                ptr, offset, value, ..
            } => (ptr, offset, value),
            _ => return None,
        };
        let address = self.address(ptr, i64::from(offset))?;
        Some(Location {
            base: address.base,
            offset: address.offset,
            bytes: layout
                .layout_of(dfg.value_type(value))?
                .store_size
                .fixed_bytes()?,
        })
    }
    /// Normalize constant byte offsets without claiming allocation provenance.
    pub fn address(&self, mut ptr: Value, mut offset: i64) -> Option<Address> {
        for _ in 0..64 {
            match self.dfg().value_inst(ptr).map(|i| self.dfg().inst(i)) {
                Some(crate::InstView::PtrOffset {
                    ptr: base,
                    offset: delta,
                }) => {
                    ptr = base;
                    offset = offset.checked_add(i64::from(delta))?;
                }
                _ => return Some(Address { base: ptr, offset }),
            }
        }
        None
    }

    pub fn may_alias(&self, a: Location, b: Location, layout: &DataLayout) -> bool {
        if a.base == b.base {
            return a.may_overlap(&b, u32::from(layout.pointer_size) * 8);
        }
        let object = |location: Location| {
            let (inst, base_offset) = self.stack_address(location.base)?;
            let crate::InstView::Alloca { size, .. } = self.dfg().inst(inst) else {
                unreachable!()
            };
            let start = u32::try_from(base_offset.checked_add(location.offset)?).ok()?;
            (location.bytes != 0 && start.checked_add(location.bytes)? <= size).then_some(inst)
        };
        !matches!((object(a), object(b)), (Some(a), Some(b)) if a != b)
    }
    /// Resolve a bounded chain of constant byte offsets to a once-per-invocation
    /// allocation. Unknown provenance and dynamic allocations stay unknown.
    pub fn stack_address(&self, ptr: Value) -> Option<(Inst, i64)> {
        let address = self.address(ptr, 0)?;
        let inst = self.dfg().value_inst(address.base)?;
        (matches!(self.dfg().inst(inst), crate::InstView::Alloca { .. })
            && self.layout().inst_block(inst) == Some(self.entry_block()))
        .then_some((inst, address.offset))
    }

    /// A complete, aligned access inside a known live entry object.
    /// This is a bounds proof, not a claim that the bytes are initialized.
    pub fn stack_access(&self, location: Location, flags: MemFlags) -> Option<(Inst, u32)> {
        let (object, offset) = self.stack_address(location.base)?;
        let offset = u32::try_from(offset.checked_add(location.offset)?).ok()?;
        let crate::InstView::Alloca { size, align } = self.dfg().inst(object) else {
            unreachable!("stack address ends at an allocation")
        };
        (location.bytes != 0
            && offset.checked_add(location.bytes)? <= size
            && align >= flags.alignment()
            && offset % flags.alignment() == 0)
            .then_some((object, offset))
    }
}
