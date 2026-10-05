//! Memory semantics shared by analyses and lowerings. An Access is a query
//! result, not a second authoritative copy of instruction operands.
use crate::{FuncBody, Inst, Value};
use veloc_types::DataLayout;

pub use crate::inst::MemoryAccess as Access;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Address {
    pub base: Value,
    pub offset: i64,
}
pub type Location = veloc_types::MemoryLocation<Value>;

impl Access {
    pub fn canonical(mut self, function: &FuncBody) -> Option<Self> {
        let address = function.address(self.ptr, self.offset)?;
        self.ptr = address.base;
        self.offset = address.offset;
        Some(self)
    }
    pub fn location(self, function: &FuncBody, layout: &DataLayout) -> Option<Location> {
        let address = function.address(self.ptr, self.offset)?;
        Some(Location {
            base: address.base,
            offset: address.offset,
            bytes: self.bytes(layout)?,
        })
    }
    /// Access width follows the supplied representation, never the host layout.
    /// Scalable or unlisted representations have no known fixed access width.
    pub fn bytes(self, layout: &DataLayout) -> Option<u32> {
        layout.layout_of(self.ty)?.store_size.fixed_bytes()
    }
}

impl FuncBody {
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
