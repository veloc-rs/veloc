//! Address normalization and fixed-width memory accesses.
use crate::{FuncBody, Inst, InstView, MemFlags, Type, Value};
use veloc_types::{DataLayout, ranges_may_overlap};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Address {
    pub base: Value,
    pub offset: i64,
}

/// A normalized, fixed-width load or store. Its base may be any pointer value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Access {
    pub base: Value,
    pub offset: i64,
    pub bytes: u32,
    pub flags: MemFlags,
}

impl Access {
    /// Exclusive end relative to the base, including offsets near the i64 limit.
    pub fn end(self) -> i128 {
        i128::from(self.offset) + i128::from(self.bytes)
    }

    /// Whether the linear byte ranges share a base and at least one byte.
    /// For possible aliasing across bases or address wraparound, use [`FuncBody::may_alias`].
    pub fn overlaps(self, other: Self) -> bool {
        self.base == other.base
            && i128::from(self.offset.max(other.offset)) < self.end().min(other.end())
    }

    /// Whether this linear byte range contains the other's range at the same base.
    pub fn covers(self, other: Self) -> bool {
        self.base == other.base && self.offset <= other.offset && self.end() >= other.end()
    }
}

impl FuncBody {
    /// Describe an ordinary load or store with a known fixed storage width.
    pub fn access(&self, inst: Inst, layout: &DataLayout) -> Option<Access> {
        let dfg = self.dfg();
        let (ptr, offset, value, flags) = match dfg.inst(inst) {
            InstView::Load { ptr, offset, flags } => (
                ptr,
                offset,
                dfg.first_result(inst).expect("load has one result"),
                flags,
            ),
            InstView::Store {
                ptr,
                offset,
                value,
                flags,
            } => (ptr, offset, value, flags),
            _ => return None,
        };
        let address = self.address(ptr, i64::from(offset))?;
        // Pointer offsets wrap to the target address width.
        let bits = u32::from(layout.pointer_size) * 8;
        let offset = match bits {
            1..64 => address.offset & ((1u64 << bits) - 1) as i64,
            64 => address.offset,
            _ => return None,
        };
        Some(Access {
            base: address.base,
            offset,
            flags,
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

    pub fn may_alias(&self, a: Access, b: Access, layout: &DataLayout) -> bool {
        if a.flags.region().is_disjoint(b.flags.region()) {
            return false;
        }
        if a.base == b.base {
            return ranges_may_overlap(
                (a.offset, a.bytes),
                (b.offset, b.bytes),
                u32::from(layout.pointer_size) * 8,
            );
        }
        // Valid accesses through distinct stack slots cannot overlap.
        self.dfg().stack_slot_layout(a.base).is_none()
            || self.dfg().stack_slot_layout(b.base).is_none()
    }
}
