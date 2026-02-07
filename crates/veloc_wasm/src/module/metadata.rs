use crate::module::types::{
    GlobalInit, ModuleMetadata, WasmData, WasmElement, WasmFunction, WasmGlobal, WasmImport,
    WasmMemory, WasmSignature, WasmTable,
};
use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;
use hashbrown::HashMap;
use wasmparser::{
    CompositeInnerType, DataKind, ElementItems, ElementKind, ExternalKind, Operator, Parser,
    Payload,
};

impl ModuleMetadata {
    pub fn collect(wasm_bin: &[u8]) -> crate::error::Result<Self> {
        let mut exports = HashMap::new();
        let mut functions = Vec::<WasmFunction>::new();
        let mut signatures = Vec::<WasmSignature>::new();
        let ir_sig_ids = Vec::<veloc::ir::SigId>::new();
        let mut tables = Vec::<WasmTable>::new();
        let mut memories = Vec::<WasmMemory>::new();
        let mut elements = Vec::<WasmElement>::new();
        let mut data_list = Vec::<WasmData>::new();
        let mut imports = Vec::<WasmImport>::new();
        let mut globals = Vec::<WasmGlobal>::new();
        let mut num_imported_funcs = 0;
        let mut num_imported_tables = 0;
        let mut num_imported_memories = 0;
        let mut num_imported_globals = 0;

        for payload in Parser::new(0).parse_all(wasm_bin) {
            let payload = payload?;
            match payload {
                Payload::ImportSection(reader) => {
                    for import in reader.into_imports() {
                        let import = import?;
                        let kind = match import.ty {
                            wasmparser::TypeRef::Func(idx)
                            | wasmparser::TypeRef::FuncExact(idx) => {
                                let func_idx = functions.len();
                                num_imported_funcs += 1;
                                functions.push(WasmFunction {
                                    name: format!("func_{}", func_idx),
                                    type_index: idx,
                                    func_id: veloc::ir::FuncId(0),
                                });
                                ExternalKind::Func
                            }
                            wasmparser::TypeRef::Table(ty) => {
                                num_imported_tables += 1;
                                tables.push(WasmTable {
                                    element_type: ty.element_type,
                                    initial: ty.initial as u32,
                                    maximum: ty.maximum.map(|v| v as u32),
                                    init: None,
                                });
                                ExternalKind::Table
                            }
                            wasmparser::TypeRef::Memory(ty) => {
                                num_imported_memories += 1;
                                memories.push(WasmMemory {
                                    initial: ty.initial,
                                    maximum: ty.maximum,
                                });
                                ExternalKind::Memory
                            }
                            wasmparser::TypeRef::Global(ty) => {
                                num_imported_globals += 1;
                                globals.push(WasmGlobal {
                                    ty: ty.content_type,
                                    mutable: ty.mutable,
                                    init: Vec::new().into_boxed_slice(),
                                });
                                ExternalKind::Global
                            }
                            wasmparser::TypeRef::Tag(_) => ExternalKind::Tag,
                        };
                        imports.push(WasmImport {
                            module: import.module.to_string(),
                            field: import.name.to_string(),
                            kind,
                            index: match import.ty {
                                wasmparser::TypeRef::Func(idx)
                                | wasmparser::TypeRef::FuncExact(idx) => idx,
                                _ => 0,
                            },
                        });
                    }
                }
                Payload::TypeSection(reader) => {
                    for rec_group in reader {
                        let rec_group = rec_group?;
                        for ty in rec_group.types() {
                            if let CompositeInnerType::Func(ft) = &ty.composite_type.inner {
                                signatures.push(WasmSignature::new(
                                    ft.params().iter().cloned().collect(),
                                    ft.results().iter().cloned().collect(),
                                ));
                            }
                        }
                    }
                }
                Payload::FunctionSection(reader) => {
                    for ty_idx in reader {
                        let func_idx = functions.len();
                        functions.push(WasmFunction {
                            name: format!("func_{}", func_idx),
                            type_index: ty_idx?,
                            func_id: veloc::ir::FuncId(0),
                        });
                    }
                }
                Payload::TableSection(reader) => {
                    for table in reader {
                        let table = table?;
                        let init = match table.init {
                            wasmparser::TableInit::Expr(expr) => Some(
                                parse_init_expr(expr.get_operators_reader())?.into_boxed_slice(),
                            ),
                            wasmparser::TableInit::RefNull => {
                                Some(vec![GlobalInit::RefNull].into_boxed_slice())
                            }
                        };
                        tables.push(WasmTable {
                            element_type: table.ty.element_type,
                            initial: table.ty.initial as u32,
                            maximum: table.ty.maximum.map(|v| v as u32),
                            init,
                        });
                    }
                }
                Payload::MemorySection(reader) => {
                    for memory in reader {
                        let memory = memory?;
                        memories.push(WasmMemory {
                            initial: memory.initial,
                            maximum: memory.maximum,
                        });
                    }
                }
                Payload::GlobalSection(reader) => {
                    for global in reader {
                        let global = global?;
                        let init = parse_init_expr(global.init_expr.get_operators_reader())?;
                        globals.push(WasmGlobal {
                            ty: global.ty.content_type,
                            mutable: global.ty.mutable,
                            init: init.into_boxed_slice(),
                        });
                    }
                }
                Payload::ExportSection(reader) => {
                    for export in reader {
                        let export = export?;
                        exports.insert(export.name.to_string(), (export.kind, export.index));
                        if let ExternalKind::Func = export.kind {
                            if let Some(func) = functions.get_mut(export.index as usize) {
                                func.name = export.name.to_string();
                            }
                        }
                    }
                }
                Payload::ElementSection(reader) => {
                    for element in reader {
                        let element = element?;
                        let mut offset = Vec::new();
                        let mut is_active = false;
                        let table_index = match element.kind {
                            ElementKind::Active {
                                table_index,
                                offset_expr,
                            } => {
                                is_active = true;
                                offset = parse_init_expr(offset_expr.get_operators_reader())?;
                                table_index.unwrap_or(0)
                            }
                            _ => 0,
                        };

                        let mut items = Vec::new();
                        match element.items {
                            ElementItems::Functions(reader) => {
                                for item in reader {
                                    items.push(vec![GlobalInit::RefFunc(item?)]);
                                }
                            }
                            ElementItems::Expressions(_ty, reader) => {
                                for expr in reader {
                                    items.push(parse_init_expr(expr?.get_operators_reader())?);
                                }
                            }
                        }

                        if !items.is_empty() {
                            elements.push(WasmElement {
                                offset: offset.into_boxed_slice(),
                                items: items
                                    .into_iter()
                                    .map(|v| v.into_boxed_slice())
                                    .collect::<Vec<_>>()
                                    .into_boxed_slice(),
                                table_index,
                                is_active,
                            });
                        }
                    }
                }
                Payload::DataSection(reader) => {
                    for data_payload in reader {
                        let data_payload = data_payload?;
                        let mut offset = Vec::new();
                        let mut is_active = false;
                        let memory_index = match data_payload.kind {
                            DataKind::Active {
                                memory_index,
                                offset_expr,
                            } => {
                                is_active = true;
                                offset = parse_init_expr(offset_expr.get_operators_reader())?;
                                memory_index
                            }
                            _ => 0,
                        };
                        data_list.push(WasmData {
                            offset: offset.into_boxed_slice(),
                            data: data_payload.data.to_vec().into_boxed_slice(),
                            memory_index,
                            is_active,
                        });
                    }
                }
                _ => {}
            }
        }

        Ok(Self {
            exports,
            functions: functions.into_boxed_slice(),
            signatures: signatures.into_boxed_slice(),
            ir_sig_ids: ir_sig_ids.into_boxed_slice(),
            tables: tables.into_boxed_slice(),
            memories: memories.into_boxed_slice(),
            elements: elements.into_boxed_slice(),
            data: data_list.into_boxed_slice(),
            imports: imports.into_boxed_slice(),
            globals: globals.into_boxed_slice(),
            num_imported_funcs,
            num_imported_tables,
            num_imported_memories,
            num_imported_globals,
        })
    }
}

fn parse_init_expr(
    mut reader: wasmparser::OperatorsReader,
) -> crate::error::Result<Vec<GlobalInit>> {
    let mut ops = Vec::new();
    while !reader.eof() {
        let op = reader.read()?;
        match op {
            Operator::I32Const { value } => ops.push(GlobalInit::I32Const(value)),
            Operator::I64Const { value } => ops.push(GlobalInit::I64Const(value)),
            Operator::F32Const { value } => ops.push(GlobalInit::F32Const(value.bits())),
            Operator::F64Const { value } => ops.push(GlobalInit::F64Const(value.bits())),
            Operator::RefNull { .. } => ops.push(GlobalInit::RefNull),
            Operator::RefFunc { function_index } => ops.push(GlobalInit::RefFunc(function_index)),
            Operator::GlobalGet { global_index } => ops.push(GlobalInit::GlobalGet(global_index)),
            Operator::I32Add => ops.push(GlobalInit::I32Add),
            Operator::I32Sub => ops.push(GlobalInit::I32Sub),
            Operator::I32Mul => ops.push(GlobalInit::I32Mul),
            Operator::I64Add => ops.push(GlobalInit::I64Add),
            Operator::I64Sub => ops.push(GlobalInit::I64Sub),
            Operator::I64Mul => ops.push(GlobalInit::I64Mul),
            Operator::End => break,
            _ => {}
        }
    }
    Ok(ops)
}
