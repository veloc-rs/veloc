//! Addressing modes and byte ranges for explicit memory accesses.
use crate::{FuncBody, Inst, InstView, MemFlags, Type, Value, inst::VectorMemOptions};
use veloc_types::{DataLayout, MemoryLocation, Region};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Address {
    pub base: Value,
    pub offset: i64,
}

/// The exact contiguous bytes of an access, when statically known.
pub type ByteRange = MemoryLocation<Value>;

/// The bytes an access may touch. An unknown range can alias anything in its region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Location {
    pub region: Region,
    pub range: Option<ByteRange>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Addressing {
    Contiguous,
    Strided {
        stride: Value,
        mask: Option<Value>,
        evl: Option<Value>,
    },
    Indexed {
        index: Value,
        scale: u8,
        mask: Option<Value>,
        evl: Option<Value>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Access {
    pub address: Address,
    /// Type of the whole loaded or stored value.
    pub ty: Type,
    pub flags: MemFlags,
    pub mode: Addressing,
}

impl Access {
    pub fn location(&self, layout: &DataLayout) -> Location {
        Location {
            region: self.flags.region(),
            range: self.range(layout),
        }
    }

    /// A fixed byte range is available for contiguous accesses of known size.
    pub fn range(&self, layout: &DataLayout) -> Option<ByteRange> {
        if self.mode != Addressing::Contiguous {
            return None;
        }
        // Pointer offsets wrap to the target address width.
        let offset = match u32::from(layout.pointer_size) * 8 {
            bits @ 1..64 => self.address.offset & ((1u64 << bits) - 1) as i64,
            64 => self.address.offset,
            _ => return None,
        };
        Some(ByteRange {
            base: self.address.base,
            offset,
            bytes: layout.layout_of(self.ty)?.store_size.fixed_bytes()?,
        })
    }
}

impl FuncBody {
    /// Describe an explicit load or store, including strided and indexed accesses.
    pub fn access(&self, inst: Inst) -> Option<Access> {
        let dfg = self.dfg();
        let result = || dfg.first_result(inst).expect("load has one result");
        let access = |ptr, value, offset, flags, mode| Access {
            address: self
                .address(ptr, offset)
                .unwrap_or(Address { base: ptr, offset }),
            ty: dfg.value_type(value),
            flags,
            mode,
        };
        let strided = |ptr, value, stride, ext: VectorMemOptions| {
            let mode = Addressing::Strided {
                stride,
                mask: ext.mask,
                evl: ext.evl,
            };
            access(ptr, value, i64::from(ext.offset), ext.flags, mode)
        };
        let indexed = |ptr, value, index, ext: VectorMemOptions| {
            let mode = Addressing::Indexed {
                index,
                scale: ext.scale,
                mask: ext.mask,
                evl: ext.evl,
            };
            access(ptr, value, i64::from(ext.offset), ext.flags, mode)
        };
        Some(match dfg.inst(inst) {
            InstView::Load { ptr, offset, flags } => access(
                ptr,
                result(),
                i64::from(offset),
                flags,
                Addressing::Contiguous,
            ),
            InstView::Store {
                ptr,
                value,
                offset,
                flags,
            } => access(ptr, value, i64::from(offset), flags, Addressing::Contiguous),
            InstView::VectorLoadStrided { ptr, stride, ext } => strided(ptr, result(), stride, ext),
            InstView::VectorStoreStrided {
                args: &[ptr, stride, value],
                ext,
            } => strided(ptr, value, stride, ext),
            InstView::VectorGather { ptr, index, ext } => indexed(ptr, result(), index, ext),
            InstView::VectorScatter {
                args: &[ptr, index, value],
                ext,
            } => indexed(ptr, value, index, ext),
            _ => return None,
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
        if a.region.is_disjoint(b.region) {
            return false;
        }
        let (Some(a), Some(b)) = (a.range, b.range) else {
            return true;
        };
        if a.base == b.base {
            return a.may_overlap(&b, u32::from(layout.pointer_size) * 8);
        }
        // Valid accesses through distinct stack slots cannot overlap.
        self.dfg().stack_slot_layout(a.base).is_none()
            || self.dfg().stack_slot_layout(b.base).is_none()
    }
}
