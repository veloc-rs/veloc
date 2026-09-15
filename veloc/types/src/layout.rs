//! Explicit target storage layouts, independent of IR and machine instructions.
use crate::{Type, TypeSize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeLayout {
    pub store_size: TypeSize,
    pub align: u32,
}

impl TypeLayout {
    pub const fn fixed(bytes: u32, align: u32) -> Self {
        assert!(bytes != 0 && align.is_power_of_two());
        Self {
            store_size: TypeSize::Fixed(bytes),
            align,
        }
    }

    /// Fixed allocation size, including tail padding. Scalable allocation
    /// requires a runtime expression and is deliberately not approximated.
    pub fn alloc_size(self) -> Option<u32> {
        if !self.align.is_power_of_two() {
            return None;
        }
        let bytes = self.store_size.fixed_bytes()?;
        bytes
            .checked_add(self.align - 1)
            .map(|n| n & !(self.align - 1))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataLayout {
    pub pointer_size: u8,
    pub little_endian: bool,
    /// Complete layouts. No implicit scalar/vector representation fallback.
    pub types: &'static [(Type, TypeLayout)],
}

impl DataLayout {
    pub fn layout_of(&self, ty: Type) -> Option<TypeLayout> {
        let layout = self.types.iter().find(|(candidate, _)| *candidate == ty)?.1;
        if !layout.align.is_power_of_two() {
            return None;
        }
        if ty == Type::PTR && layout.store_size.fixed_bytes() != Some(u32::from(self.pointer_size))
        {
            return None;
        }
        Some(layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn storage_layout_is_explicit_and_allocation_includes_padding() {
        const LAYOUT: DataLayout = DataLayout {
            pointer_size: 4,
            little_endian: true,
            types: &[(Type::I32, TypeLayout::fixed(3, 4))],
        };
        let layout = LAYOUT;
        let value = layout.layout_of(Type::I32).unwrap();
        assert_eq!(value.store_size.fixed_bytes(), Some(3));
        assert_eq!(value.alloc_size(), Some(4));
        assert_eq!(layout.layout_of(Type::F32), None);
        let scalable = TypeLayout {
            store_size: TypeSize::Scalable { min_bytes: 16 },
            align: 16,
        };
        assert_eq!(scalable.alloc_size(), None);
        assert_eq!(TypeLayout::fixed(u32::MAX, 8).alloc_size(), None);
    }
}
