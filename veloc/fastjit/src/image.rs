//! Target-neutral function packaging. Each ISA supplies code and relocation
//! records; this layer only assigns symbols and writes the object container.

use crate::stencil::Code;
use crate::{Error, Result};
use object::write::{Object, Relocation, StandardSection, Symbol, SymbolId, SymbolSection};
use object::{Architecture, BinaryFormat, Endianness, SymbolFlags, SymbolKind, SymbolScope};
use std::collections::HashMap;
use veloc_mir::{FuncBody, Linkage, Module, Signature};

pub(crate) trait Target {
    const ARCH: Architecture;
    const ENDIAN: Endianness;

    fn compile<'a>(
        module: &'a Module,
        body: &'a FuncBody,
        signature: &Signature,
    ) -> Result<Code<'a>>;
}

fn symbol(
    object: &mut Object<'static>,
    names: &mut HashMap<String, SymbolId>,
    name: &str,
) -> SymbolId {
    if let Some(&id) = names.get(name) {
        return id;
    }
    let id = object.add_symbol(Symbol {
        name: name.as_bytes().to_vec(),
        value: 0,
        size: 0,
        kind: SymbolKind::Text,
        scope: SymbolScope::Linkage,
        weak: false,
        section: SymbolSection::Undefined,
        flags: SymbolFlags::None,
    });
    names.insert(name.to_owned(), id);
    id
}

pub(crate) fn compile<T: Target>(module: &Module) -> Result<Vec<u8>> {
    let mut object = Object::new(BinaryFormat::Elf, T::ARCH, T::ENDIAN);
    let text = object.section_id(StandardSection::Text);
    let mut names = HashMap::new();
    for (id, func) in module.functions() {
        let sym = symbol(&mut object, &mut names, &func.decl.name);
        let Some(body) = func.body else { continue };
        let sig = &module.signatures()[func.decl.signature];
        let code = T::compile(module, body, sig).map_err(|error| match error {
            Error::Unsupported(reason) => {
                Error::Unsupported(format!("{} ({id:?}): {reason}", func.decl.name))
            }
            other => other,
        })?;
        let base = object.add_symbol_data(sym, text, &code.bytes, 16);
        object.symbol_mut(sym).scope = match func.decl.linkage {
            Linkage::Local => SymbolScope::Compilation,
            Linkage::Import | Linkage::Export => SymbolScope::Linkage,
        };
        for reloc in code.relocations {
            let target = symbol(&mut object, &mut names, reloc.symbol);
            object.add_relocation(
                text,
                Relocation {
                    offset: base + reloc.offset,
                    symbol: target,
                    addend: reloc.addend,
                    flags: reloc.flags,
                },
            )?;
        }
    }
    Ok(object.write()?)
}
