//! Checked target storage layouts, shared by ABI validation and Rust generation.
use crate::target::ast::{DataLayoutDef, Def, Module};
use crate::types::{Primitive, TypeKey, Types};
use std::collections::BTreeMap;
use std::fmt::Write;

pub(super) struct Plan {
    layouts: BTreeMap<String, Layout>,
}

pub(super) struct Layout {
    definition: DataLayoutDef,
    slots: BTreeMap<TypeKey, (u32, u32)>,
}

fn type_key(types: &Types, name: &str) -> Result<TypeKey, String> {
    let domain = &types.exact[name.rsplit("::").next().unwrap()];
    if !domain.is_singleton() {
        return Err(format!("layout requires a concrete type: {name}"));
    }
    let (&element, &shapes) = domain.0.first_key_value().unwrap();
    Ok((element, shapes.trailing_zeros()))
}

fn check_slot(size: u32, align: u32) -> Result<(), String> {
    if size == 0 || !align.is_power_of_two() {
        return Err("storage size must be nonzero and alignment must be a power of two".into());
    }
    // Allocation includes tail padding, which must remain representable.
    size.checked_add(align - 1)
        .ok_or("aligned storage size exceeds u32")?;
    Ok(())
}

impl Plan {
    pub fn prepare(module: &Module, types: &Types) -> Result<Self, String> {
        let mut layouts = BTreeMap::new();
        for def in &module.defs {
            let Def::DataLayout(def) = def else { continue };
            let error = |reason: String| format!("data layout {}: {reason}", def.name);
            check_slot(def.pointer_size, def.pointer_align).map_err(&error)?;
            if u8::try_from(def.pointer_size).is_err() {
                return Err(error("pointer size exceeds u8".into()));
            }
            let mut slots =
                BTreeMap::from([((Primitive::Ptr, 0), (def.pointer_size, def.pointer_align))]);
            for entry in &def.types {
                check_slot(entry.size, entry.align).map_err(&error)?;
                let key @ (element, shape) = type_key(types, &entry.ty).map_err(&error)?;
                if shape >= 16 {
                    return Err(error(format!(
                        "fixed storage cannot describe scalable type {}",
                        entry.ty
                    )));
                }
                if slots.insert(key, (entry.size, entry.align)).is_some() {
                    return Err(error(format!(
                        "duplicate storage definition for {}",
                        entry.ty
                    )));
                }
                let bits = element
                    .element_bits()
                    .ok_or_else(|| error("pointer storage belongs in the pointer field".into()))?;
                let bytes = (bits * (1 << shape)).div_ceil(8);
                if entry.size < bytes {
                    return Err(error(format!(
                        "storage for {} is smaller than its logical bit width",
                        entry.ty
                    )));
                }
            }
            if layouts
                .insert(
                    def.name.clone(),
                    Layout {
                        definition: def.clone(),
                        slots,
                    },
                )
                .is_some()
            {
                return Err(error("duplicate layout name".into()));
            }
        }
        Ok(Self { layouts })
    }

    pub fn get(&self, name: &str) -> Result<&Layout, String> {
        self.layouts
            .get(name)
            .ok_or_else(|| format!("unknown data layout {name}"))
    }

    pub fn generate(&self, output: &mut String) {
        for (name, layout) in &self.layouts {
            let def = &layout.definition;
            let symbol = symbol(name);
            writeln!(
                output,
                "pub const {symbol}: veloc_types::DataLayout = veloc_types::DataLayout {{"
            )
            .unwrap();
            writeln!(
                output,
                "pointer_size: {}, little_endian: {}, types: &[",
                def.pointer_size, def.little_endian
            )
            .unwrap();
            writeln!(
                output,
                "(veloc_mir::Type::PTR, veloc_types::TypeLayout::fixed({}, {})),",
                def.pointer_size, def.pointer_align
            )
            .unwrap();
            for entry in &def.types {
                writeln!(
                    output,
                    "({}, veloc_types::TypeLayout::fixed({}, {})),",
                    entry.ty, entry.size, entry.align
                )
                .unwrap();
            }
            output.push_str("] };\n");
        }
    }
}

impl Layout {
    pub fn slot(&self, types: &Types, name: &str) -> Result<(u32, u32), String> {
        self.slots
            .get(&type_key(types, name)?)
            .copied()
            .ok_or_else(|| {
                format!(
                    "data layout {} has no storage for {name}",
                    self.definition.name
                )
            })
    }

    pub fn bitcast_width(&self, types: &Types, name: &str) -> Result<u32, String> {
        match type_key(types, name)? {
            (Primitive::Ptr, 0) => Ok(self.definition.pointer_size * 8),
            (Primitive::Int(bits) | Primitive::Float(bits), shape @ 0..16) => {
                Ok(bits * (1 << shape))
            }
            _ => Err(format!(
                "ABI bitcast requires a scalar number, pointer, or fixed numeric vector: {name}"
            )),
        }
    }
}

pub(super) fn symbol(name: &str) -> String {
    super::generate::sanitize_ident(&format!("DATA_LAYOUT_{name}")).to_ascii_uppercase()
}
