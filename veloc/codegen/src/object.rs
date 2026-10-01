//! Object file writer.
//!
//! 将后端生成的裸机器码封装为可重定位的 object 文件。

use crate::error::{Error, Result};
use crate::target::{TargetArch, TargetMachine};
use hashbrown::HashMap;
use object::write::{Object, Relocation, StandardSection, Symbol, SymbolId, SymbolSection};
use object::{
    Architecture, BinaryFormat, Endianness, RelocationEncoding, RelocationFlags, RelocationKind,
    SymbolFlags, SymbolKind, SymbolScope,
};
use std::format;
use std::string::String;
use veloc_mir::{FunctionRef, Linkage};

const TEXT_ALIGN: u64 = 16;

struct FunctionCode<'a> {
    symbol: SymbolId,
    label: veloc_lir::SymbolId,
    name: String,
    emission: &'a crate::Emitter,
}

pub(crate) struct ObjectFileBuilder<'a> {
    object: Object<'static>,
    text_section: object::write::SectionId,
    symbols: HashMap<String, SymbolId>,
    relocation_symbols: HashMap<veloc_lir::SymbolId, (String, SymbolId)>,
    functions: std::vec::Vec<FunctionCode<'a>>,
}

impl<'a> ObjectFileBuilder<'a> {
    pub(crate) fn new(target: &dyn TargetMachine) -> Result<Self> {
        let (format, architecture, endian) = object_format_for_target(target.desc().arch)?;
        let mut object = Object::new(format, architecture, endian);
        if target.desc().arch == TargetArch::Riscv64 {
            object.flags = object::FileFlags::Elf {
                os_abi: 0,
                abi_version: 0,
                e_flags: object::elf::EF_RISCV_FLOAT_ABI_DOUBLE,
            };
        }
        let text_section = object.section_id(StandardSection::Text);

        Ok(Self {
            object,
            text_section,
            symbols: HashMap::new(),
            relocation_symbols: HashMap::new(),
            functions: std::vec::Vec::new(),
        })
    }

    pub(crate) fn add_defined_function(
        &mut self,
        func: &FunctionRef,
        label: veloc_lir::SymbolId,
        emission: &'a crate::Emitter,
        symbols: &veloc_lir::SymbolTable,
    ) {
        let symbol = self.ensure_function_symbol(func);
        for id in emission.symbols() {
            let name = &symbols.get(id).name;
            let symbol = self.ensure_text_symbol_name(name);
            self.relocation_symbols.insert(id, (name.clone(), symbol));
        }
        self.functions.push(FunctionCode {
            symbol,
            label,
            name: func.decl.name.clone(),
            emission,
        });
    }

    pub(crate) fn finish(mut self, profile: &veloc_profile::Profile) -> Result<std::vec::Vec<u8>> {
        // Run after module post-emission passes, immediately before writing the
        // section. Every function and block participates in the same layout.
        let units = self
            .functions
            .iter()
            .map(|f| (Some(f.label), f.emission))
            .collect::<std::vec::Vec<_>>();
        let placed = profile.measure("layout", 0, || {
            crate::emitter::layout(&units, TEXT_ALIGN as usize)
        })?;
        for (function, placed) in self.functions.iter().zip(placed) {
            let emitted = placed.code;
            let base_offset = self.object.add_symbol_data(
                function.symbol,
                self.text_section,
                &emitted.data,
                TEXT_ALIGN,
            );
            if base_offset != placed.offset as u64 {
                return Err(Error::codegen(
                    "object section placement disagrees with code layout",
                ));
            }
            profile.record_lazy(veloc_profile::Metric::bytes("code"), || {
                emitted.data.len() as u64
            });
            for relocation in emitted.relocations {
                let (sym_name, target_symbol) = &self.relocation_symbols[&relocation.symbol];
                self.object
                    .add_relocation(
                        self.text_section,
                        Relocation {
                            offset: base_offset + relocation.offset,
                            symbol: *target_symbol,
                            addend: relocation.addend,
                            flags: RelocationFlags::Generic {
                                kind: match relocation.kind {
                                    crate::RelocationKind::RelativeBranch32 => {
                                        RelocationKind::Relative
                                    }
                                    crate::RelocationKind::Absolute64 => RelocationKind::Absolute,
                                },
                                encoding: match relocation.kind {
                                    crate::RelocationKind::RelativeBranch32 => {
                                        RelocationEncoding::X86Branch
                                    }
                                    crate::RelocationKind::Absolute64 => {
                                        RelocationEncoding::Generic
                                    }
                                },
                                size: match relocation.kind {
                                    crate::RelocationKind::RelativeBranch32 => 32,
                                    crate::RelocationKind::Absolute64 => 64,
                                },
                            },
                        },
                    )
                    .map_err(|err| {
                        Error::object_file_relocation_error(
                            function.name.clone(),
                            sym_name.clone(),
                            format!("{err}"),
                        )
                    })?;
            }
        }
        self.object
            .write()
            .map_err(|err| Error::object_file_write_error(format!("{err}")))
    }

    fn ensure_function_symbol(&mut self, func: &FunctionRef) -> SymbolId {
        if let Some(&symbol_id) = self.symbols.get(&func.decl.name) {
            let symbol = self.object.symbol_mut(symbol_id);
            symbol.scope = symbol_scope(func.decl.linkage);
            symbol.kind = SymbolKind::Text;
            symbol.flags = SymbolFlags::None;
            return symbol_id;
        }

        let symbol_id = self.object.add_symbol(Symbol {
            name: func.decl.name.as_bytes().to_vec(),
            value: 0,
            size: 0,
            kind: SymbolKind::Text,
            scope: symbol_scope(func.decl.linkage),
            weak: false,
            section: SymbolSection::Undefined,
            flags: SymbolFlags::None,
        });
        self.symbols.insert(func.decl.name.clone(), symbol_id);
        symbol_id
    }

    fn ensure_text_symbol_name(&mut self, name: &str) -> SymbolId {
        if let Some(&symbol_id) = self.symbols.get(name) {
            return symbol_id;
        }

        let symbol_id = self.object.add_symbol(Symbol {
            name: name.as_bytes().to_vec(),
            value: 0,
            size: 0,
            kind: SymbolKind::Text,
            scope: SymbolScope::Dynamic,
            weak: false,
            section: SymbolSection::Undefined,
            flags: SymbolFlags::None,
        });
        self.symbols.insert(name.into(), symbol_id);
        symbol_id
    }
}

fn object_format_for_target(arch: TargetArch) -> Result<(BinaryFormat, Architecture, Endianness)> {
    let endian = match arch.is_little_endian() {
        true => Endianness::Little,
        false => Endianness::Big,
    };

    let architecture = match arch {
        TargetArch::X86_64 => Architecture::X86_64,
        TargetArch::AArch64 => Architecture::Aarch64,
        TargetArch::Riscv64 => Architecture::Riscv64,
        TargetArch::Wasm32 | TargetArch::Wasm64 => {
            return Err(Error::unsupported_object_format(arch));
        }
    };

    Ok((BinaryFormat::Elf, architecture, endian))
}

fn symbol_scope(linkage: Linkage) -> SymbolScope {
    match linkage {
        Linkage::Local => SymbolScope::Compilation,
        Linkage::Import | Linkage::Export => SymbolScope::Dynamic,
    }
}

#[cfg(test)]
mod tests {
    use crate::{CodegenPipeline, TargetArch, TargetConfig, create_target_machine};
    use object::read::{Object as _, ObjectSection as _, ObjectSymbol as _};
    use object::{BinaryFormat, RelocationEncoding, RelocationKind, RelocationTarget, SymbolScope};
    use std::string::ToString;
    use std::vec;
    use std::vec::Vec;
    use veloc_mir::{CallConv, Linkage, ModuleBuilder};

    fn parse_symbol<'a>(
        file: &'a object::File<'a>,
        name: &str,
    ) -> Option<object::read::Symbol<'a, 'a>> {
        file.symbols()
            .find(|symbol| symbol.name().ok() == Some(name))
    }

    #[test]
    fn compile_single_function_to_object() {
        let mut mb = ModuleBuilder::new();
        let sig = mb.make_signature(vec![], vec![], CallConv::SystemV);
        let func_id = mb.declare_function("main".into(), sig, Linkage::Export);
        {
            let mut fb = mb.define(func_id);
            fb.ins().ret(&[]);
        }
        let module = mb.build();
        let target = create_target_machine(TargetConfig::default()).unwrap();

        let bytes = CodegenPipeline::new(&*target, Default::default())
            .compile_object(&module)
            .unwrap();
        let object = object::File::parse(&*bytes).unwrap();
        let symbol = parse_symbol(&object, "main").unwrap();

        assert_eq!(object.format(), BinaryFormat::Elf);
        assert_eq!(symbol.scope(), SymbolScope::Dynamic);
        assert!(!symbol.is_undefined());
        assert!(symbol.size() > 0);
    }

    #[test]
    fn compile_module_to_object_keeps_defined_and_imported_symbols() {
        let mut mb = ModuleBuilder::new();
        let sig = mb.make_signature(vec![], vec![], CallConv::SystemV);
        mb.declare_function("ext_func".into(), sig, Linkage::Import);
        let main_id = mb.declare_function("main".into(), sig, Linkage::Export);
        let local_id = mb.declare_function("helper".into(), sig, Linkage::Local);
        {
            let mut fb = mb.define(main_id);
            fb.ins().ret(&[]);
        }
        {
            let mut fb = mb.define(local_id);
            fb.ins().ret(&[]);
        }
        let module = mb.build();
        let target = create_target_machine(TargetConfig::default()).unwrap();

        let bytes = CodegenPipeline::new(&*target, Default::default())
            .compile_object(&module)
            .unwrap();
        let object = object::File::parse(&*bytes).unwrap();

        let main = parse_symbol(&object, "main").unwrap();
        let helper = parse_symbol(&object, "helper").unwrap();
        let ext = parse_symbol(&object, "ext_func").unwrap();

        assert!(!main.is_undefined());
        assert_eq!(main.scope(), SymbolScope::Dynamic);
        assert!(!helper.is_undefined());
        assert_eq!(helper.scope(), SymbolScope::Compilation);
        assert!(ext.is_undefined());
    }

    #[test]
    fn object_builder_rejects_wasm_targets() {
        let err = super::object_format_for_target(TargetArch::Wasm32).unwrap_err();
        assert!(err.to_string().contains("not supported"));
    }

    #[test]
    fn compile_module_to_object_emits_call_relocation() {
        let mut mb = ModuleBuilder::new();
        let sig = mb.make_signature(vec![], vec![], CallConv::SystemV);
        let ext_id = mb.declare_function("ext_func".into(), sig, Linkage::Import);
        let main_id = mb.declare_function("main".into(), sig, Linkage::Export);
        {
            let mut fb = mb.define(main_id);
            fb.ins().call(ext_id, &[]);
            fb.ins().ret(&[]);
        }
        let module = mb.build();
        let target = create_target_machine(TargetConfig::default()).unwrap();

        let bytes = CodegenPipeline::new(&*target, Default::default())
            .compile_object(&module)
            .unwrap();
        let object = object::File::parse(&*bytes).unwrap();
        let text = object.section_by_name(".text").unwrap();
        let relocations: Vec<(u64, object::Relocation)> = text.relocations().collect();

        assert_eq!(relocations.len(), 1);
        let relocation = &relocations[0].1;
        assert!(matches!(
            relocation.kind(),
            RelocationKind::Relative | RelocationKind::PltRelative
        ));
        assert!(matches!(
            relocation.encoding(),
            RelocationEncoding::X86Branch | RelocationEncoding::Generic
        ));
        assert_eq!(relocation.size(), 32);
        assert_eq!(relocation.addend(), -4);

        let RelocationTarget::Symbol(symbol_index) = relocation.target() else {
            panic!("expected relocation to target a symbol");
        };
        let symbol = object.symbol_by_index(symbol_index).unwrap();
        assert_eq!(symbol.name().unwrap(), "ext_func");
    }
}
