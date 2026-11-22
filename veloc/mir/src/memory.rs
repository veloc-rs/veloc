//! Shared address normalization, access ranges and stack bounds proofs.
//! Instruction operands and access properties remain in InstView and MemFlags.
use crate::{FuncBody, Inst, InstView, MemFlags, Type, Value};
use veloc_types::DataLayout;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Address {
    pub base: Value,
    pub offset: i64,
}
pub type Location = veloc_types::MemoryLocation<Value>;

struct StackRange {
    object: Value,
    offset: u32,
    object_align: u32,
}

impl FuncBody {
    /// Fixed byte range of an ordinary load or store in the target layout.
    /// Unknown sizes or addresses return None, not a proof of no memory effect.
    pub fn memory_location(&self, inst: Inst, layout: &DataLayout) -> Option<Location> {
        let dfg = self.dfg();
        let (ptr, offset, value) = match dfg.inst(inst) {
            InstView::Load { ptr, offset, .. } => (
                ptr,
                offset,
                dfg.first_result(inst).expect("load has one result"),
            ),
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
    /// Combine constant byte offsets from a pointer-typed value into a base and offset.
    /// Returns `None` if accumulating an offset overflows `i64`.
    pub fn address(&self, mut ptr: Value, mut offset: i64) -> Option<Address> {
        let dfg = self.dfg();
        debug_assert_eq!(dfg.value_type(ptr), Type::PTR);
        while let Some(InstView::PtrOffset {
            ptr: base,
            offset: delta,
        }) = dfg.value_inst_view(ptr)
        {
            offset = offset.checked_add(i64::from(delta))?;
            ptr = base;
        }
        Some(Address { base: ptr, offset })
    }

    pub fn may_alias(&self, a: Location, b: Location, layout: &DataLayout) -> bool {
        if a.base == b.base {
            return a.may_overlap(&b, u32::from(layout.pointer_size) * 8);
        }
        match (self.stack_range(a), self.stack_range(b)) {
            (Some(a), Some(b)) => a.object == b.object,
            _ => true,
        }
    }

    /// A complete, aligned access inside a known stack slot.
    /// This is a bounds proof, not a claim that the bytes are initialized.
    pub fn stack_access(&self, location: Location, flags: MemFlags) -> Option<(Value, u32)> {
        let range = self.stack_range(location)?;
        let align = flags.alignment();
        (range.object_align >= align && range.offset % align == 0)
            .then_some((range.object, range.offset))
    }

    /// Resolve a nonempty byte range wholly inside a stack slot.
    fn stack_range(&self, location: Location) -> Option<StackRange> {
        let address = self.address(location.base, 0)?;
        let object = address.base;
        let (size, align) = self.dfg().stack_slot_layout(object)?;
        let offset = u32::try_from(address.offset.checked_add(location.offset)?).ok()?;
        (location.bytes != 0 && offset.checked_add(location.bytes)? <= size).then_some(StackRange {
            object,
            offset,
            object_align: align,
        })
    }
}
