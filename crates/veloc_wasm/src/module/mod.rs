pub mod metadata;
pub mod runtime;
pub mod types;

use alloc::format;
use alloc::sync::Arc;
use alloc::vec::Vec;
use elf_loader::Loader;
use elf_loader::image::LoadedObject;
use elf_loader::input::ElfBinary;

use crate::Result;
use crate::engine::{Engine, Strategy};
use crate::translator::WasmTranslator;
use crate::vm::VMOffsets;
use veloc::ir::{CallConv, FuncId, Linkage, Signature, Type as VelocType};
use wasmparser::{Parser, Payload};

pub use self::runtime::*;
pub use self::types::*;

pub enum ModuleArtifact {
    Interpreter(veloc::ir::Module),
    Jit(LoadedObject<()>),
}

pub struct ModuleInner {
    pub engine: Arc<Engine>,
    pub artifact: ModuleArtifact,
    pub metadata: ModuleMetadata,
    pub strategy: Strategy,
    pub offsets: VMOffsets,
    pub init_func_id: FuncId,
}

#[derive(Debug, Clone, Copy)]
pub struct RuntimeFunctions {
    pub trap_handler: FuncId,
    pub memory_grow: FuncId,
    pub table_init: FuncId,
    pub table_copy: FuncId,
    pub table_grow: FuncId,
    pub table_size: FuncId,
    pub table_fill: FuncId,
    pub elem_drop: FuncId,
    pub memory_init: FuncId,
    pub data_drop: FuncId,
    pub memory_copy: FuncId,
    pub memory_fill: FuncId,
    pub init_table_element: FuncId,
    pub init_memory_data: FuncId,
    pub init_table: FuncId,
}

impl RuntimeFunctions {
    pub fn declare(ir: &mut veloc::ir::ModuleBuilder) -> Self {
        // (ptr, i32) -> void
        let sig_p_i_v = Signature::new(
            vec![VelocType::Ptr, VelocType::I32],
            VelocType::Void,
            CallConv::SystemV,
        );
        // (ptr, i32, i32) -> void
        let sig_p_i_i_v = Signature::new(
            vec![VelocType::Ptr, VelocType::I32, VelocType::I32],
            VelocType::Void,
            CallConv::SystemV,
        );
        // (ptr, i32, i32) -> i32
        let sig_p_i_i_i = Signature::new(
            vec![VelocType::Ptr, VelocType::I32, VelocType::I32],
            VelocType::I32,
            CallConv::SystemV,
        );
        // (ptr, i32, ptr, i32) -> i32
        let sig_p_i_p_i_i = Signature::new(
            vec![
                VelocType::Ptr,
                VelocType::I32,
                VelocType::Ptr,
                VelocType::I32,
            ],
            VelocType::I32,
            CallConv::SystemV,
        );
        // (ptr, i32, i32, ptr, i32) -> void
        let sig_p_i_i_p_i_v = Signature::new(
            vec![
                VelocType::Ptr,
                VelocType::I32,
                VelocType::I32,
                VelocType::Ptr,
                VelocType::I32,
            ],
            VelocType::Void,
            CallConv::SystemV,
        );
        // (ptr, i32, i32, i32, i32, i32) -> void
        let sig_p_i_i_i_i_i_v = Signature::new(
            vec![
                VelocType::Ptr,
                VelocType::I32,
                VelocType::I32,
                VelocType::I32,
                VelocType::I32,
                VelocType::I32,
            ],
            VelocType::Void,
            CallConv::SystemV,
        );
        // (ptr, i32, i32, i32, i32) -> void
        let sig_p_i_i_i_i_v = Signature::new(
            vec![
                VelocType::Ptr,
                VelocType::I32,
                VelocType::I32,
                VelocType::I32,
                VelocType::I32,
            ],
            VelocType::Void,
            CallConv::SystemV,
        );
        // (ptr, i32, ptr) -> void
        let sig_p_i_p_v = Signature::new(
            vec![VelocType::Ptr, VelocType::I32, VelocType::Ptr],
            VelocType::Void,
            CallConv::SystemV,
        );

        Self {
            trap_handler: ir.declare_function(
                "wasm_trap_handler".to_string(),
                sig_p_i_v.clone(),
                Linkage::Import,
            ),
            memory_grow: ir.declare_function(
                "wasm_memory_grow".to_string(),
                sig_p_i_i_i.clone(),
                Linkage::Import,
            ),
            init_table_element: ir.declare_function(
                "wasm_init_table_element".to_string(),
                sig_p_i_i_v.clone(),
                Linkage::Import,
            ),
            init_memory_data: ir.declare_function(
                "wasm_init_memory_data".to_string(),
                sig_p_i_i_v.clone(),
                Linkage::Import,
            ),
            init_table: ir.declare_function(
                "wasm_init_table".to_string(),
                sig_p_i_p_v,
                Linkage::Import,
            ),
            table_init: ir.declare_function(
                "wasm_table_init".to_string(),
                sig_p_i_i_i_i_i_v.clone(),
                Linkage::Import,
            ),
            table_copy: ir.declare_function(
                "wasm_table_copy".to_string(),
                sig_p_i_i_i_i_i_v.clone(),
                Linkage::Import,
            ),
            table_grow: ir.declare_function(
                "wasm_table_grow".to_string(),
                sig_p_i_p_i_i,
                Linkage::Import,
            ),
            table_size: ir.declare_function(
                "wasm_table_size".to_string(),
                // (ptr, i32) -> i32
                Signature::new(
                    vec![VelocType::Ptr, VelocType::I32],
                    VelocType::I32,
                    CallConv::SystemV,
                ),
                Linkage::Import,
            ),
            table_fill: ir.declare_function(
                "wasm_table_fill".to_string(),
                sig_p_i_i_p_i_v,
                Linkage::Import,
            ),
            elem_drop: ir.declare_function(
                "wasm_elem_drop".to_string(),
                sig_p_i_v.clone(),
                Linkage::Import,
            ),
            memory_init: ir.declare_function(
                "wasm_memory_init".to_string(),
                sig_p_i_i_i_i_i_v.clone(),
                Linkage::Import,
            ),
            data_drop: ir.declare_function(
                "wasm_data_drop".to_string(),
                sig_p_i_v,
                Linkage::Import,
            ),
            memory_copy: ir.declare_function(
                "wasm_memory_copy".to_string(),
                sig_p_i_i_i_i_i_v,
                Linkage::Import,
            ),
            memory_fill: ir.declare_function(
                "wasm_memory_fill".to_string(),
                sig_p_i_i_i_i_v,
                Linkage::Import,
            ),
        }
    }
}

#[derive(Clone)]
pub struct Module {
    pub(crate) inner: Arc<ModuleInner>,
}

impl Module {
    pub fn ir(&self) -> &veloc::ir::Module {
        match &self.inner.artifact {
            ModuleArtifact::Interpreter(ir) => ir,
            ModuleArtifact::Jit(_loaded) => {
                unreachable!("Should only access IR in interpreter mode")
            }
        }
    }

    pub fn metadata(&self) -> &ModuleMetadata {
        &self.inner.metadata
    }

    pub(crate) fn loaded(&self) -> Option<&LoadedObject<()>> {
        match &self.inner.artifact {
            ModuleArtifact::Jit(loaded) => Some(loaded),
            _ => None,
        }
    }

    pub fn new(engine: Arc<Engine>, wasm_bin: &[u8]) -> Result<Self> {
        let mut metadata = ModuleMetadata::collect(wasm_bin)?;
        let mut ir = veloc::ir::ModuleBuilder::new();

        // Intern all Wasm signatures
        let mut ir_sig_ids = Vec::with_capacity(metadata.signatures.len());
        for i in 0..metadata.signatures.len() {
            let wasm_sig = &metadata.signatures[i];
            let mut params = vec![VelocType::Ptr]; // vmctx
            params.extend(wasm_sig.params.iter().map(|&t| valtype_to_veloc(t)));
            if wasm_sig.results.len() > 1 {
                params.push(VelocType::Ptr); // result ptr
            }
            let ret = if wasm_sig.results.len() == 1 {
                valtype_to_veloc(wasm_sig.results[0])
            } else {
                VelocType::Void
            };
            let sig = Signature::new(params, ret, CallConv::SystemV);
            ir_sig_ids.push(ir.intern_signature(sig));
        }
        metadata.ir_sig_ids = ir_sig_ids.into_boxed_slice();

        let mut strategy = engine.config.strategy;
        if strategy == Strategy::Auto {
            strategy = Strategy::Jit;
        }

        // 1. Declare runtime functions and offsets
        let runtime = RuntimeFunctions::declare(&mut ir);
        let offsets = VMOffsets::new(
            metadata.memories.len() as u32,
            metadata.tables.len() as u32,
            metadata.globals.len() as u32,
            metadata.functions.len() as u32,
            metadata.signatures.len() as u32,
        );

        // 2. Generate function declarations and trampolines
        generate_trampolines(&mut ir, &mut metadata);

        // 3. Generate __veloc_init function
        let init_func_id = generate_veloc_init(&mut ir, &metadata, &offsets, &runtime);

        // 4. Translate Wasm bytecode to IR
        let mut func_count = 0;
        for payload in Parser::new(0).parse_all(wasm_bin) {
            let payload = payload?;

            if let Payload::CodeSectionEntry(body) = payload {
                let global_idx = metadata.num_imported_funcs + func_count;
                let ty_idx = metadata.functions[global_idx].type_index;
                let sig = &metadata.signatures[ty_idx as usize];

                let params: Vec<VelocType> =
                    sig.params.iter().map(|&p| valtype_to_veloc(p)).collect();
                let returns: Vec<VelocType> =
                    sig.results.iter().map(|&r| valtype_to_veloc(r)).collect();

                let func_id = metadata.functions[global_idx].func_id;

                let mut builder = ir.builder(func_id);
                let mut translator =
                    WasmTranslator::new(&mut builder, returns, &metadata, offsets, runtime);
                translator.translate(body, &params)?;

                func_count += 1;
            }
        }

        if let Err(e) = ir.validate() {
            println!("IR Validation error: {}", e);
            return Err(crate::error::Error::Message(format!(
                "IR validation failed: {}",
                e
            )));
        }

        let ir = ir.build();

        let artifact = if strategy == Strategy::Jit {
            let object_data = engine
                .backend
                .compile_module(&ir)
                .map_err(|e| crate::error::Error::Compile(format!("Codegen error: {}", e)))?;

            // Load JIT object and relocate
            let mut loader = Loader::new();
            let lib = loader.load_object(ElfBinary::new("wasm_module", &object_data))?;
            let loaded = lib
                .relocator()
                .pre_find_fn(|name| match name {
                    "wasm_trap_handler" => Some(runtime::wasm_trap_handler as *const ()),
                    "wasm_memory_size" => Some(runtime::wasm_memory_size as *const ()),
                    "wasm_memory_grow" => Some(runtime::wasm_memory_grow as *const ()),
                    "wasm_table_size" => Some(runtime::wasm_table_size as *const ()),
                    "wasm_table_grow" => Some(runtime::wasm_table_grow as *const ()),
                    "wasm_table_fill" => Some(runtime::wasm_table_fill as *const ()),
                    "wasm_table_copy" => Some(runtime::wasm_table_copy as *const ()),
                    "wasm_table_init" => Some(runtime::wasm_table_init as *const ()),
                    "wasm_elem_drop" => Some(runtime::wasm_elem_drop as *const ()),
                    "wasm_memory_init" => Some(runtime::wasm_memory_init as *const ()),
                    "wasm_data_drop" => Some(runtime::wasm_data_drop as *const ()),
                    "wasm_memory_copy" => Some(runtime::wasm_memory_copy as *const ()),
                    "wasm_memory_fill" => Some(runtime::wasm_memory_fill as *const ()),
                    "wasm_init_table_element" => {
                        Some(runtime::wasm_init_table_element as *const ())
                    }
                    "wasm_init_memory_data" => Some(runtime::wasm_init_memory_data as *const ()),
                    "wasm_init_table" => Some(runtime::wasm_init_table as *const ()),
                    _ => None,
                })
                .relocate()?;
            ModuleArtifact::Jit(loaded)
        } else {
            ModuleArtifact::Interpreter(ir)
        };

        let inner = Arc::new(ModuleInner {
            engine,
            artifact,
            metadata,
            strategy,
            offsets,
            init_func_id,
        });

        Ok(Self { inner })
    }
}

fn generate_init_expr(
    ins: &mut veloc::ir::builder::InstBuilder,
    expr: &[GlobalInit],
    vmctx: veloc::ir::Value,
    offsets: &VMOffsets,
    metadata: &ModuleMetadata,
) -> veloc::ir::Value {
    let mut stack = Vec::new();
    for op in expr {
        match op {
            GlobalInit::I32Const(v) => stack.push(ins.iconst(VelocType::I32, *v as i64)),
            GlobalInit::I64Const(v) => stack.push(ins.iconst(VelocType::I64, *v)),
            GlobalInit::F32Const(v) => {
                let b = ins.iconst(VelocType::I32, *v as i64);
                stack.push(ins.reinterpret(b, VelocType::F32))
            }
            GlobalInit::F64Const(v) => {
                let b = ins.iconst(VelocType::I64, *v as i64);
                stack.push(ins.reinterpret(b, VelocType::F64))
            }
            GlobalInit::RefNull => {
                let null_ptr = ins.iconst(VelocType::I64, 0);
                stack.push(ins.int_to_ptr(null_ptr))
            }
            GlobalInit::RefFunc(idx) => {
                let offset = offsets.function_offset(*idx);
                let offset_val = ins.iconst(VelocType::I64, offset as i64);
                stack.push(ins.gep(vmctx, offset_val))
            }
            GlobalInit::GlobalGet(idx) => {
                let src_ptr = ins.load(VelocType::Ptr, vmctx, offsets.global_offset(*idx));
                let ty = valtype_to_veloc(metadata.globals[*idx as usize].ty);
                stack.push(ins.load(ty, src_ptr, 0))
            }
            GlobalInit::I32Add | GlobalInit::I64Add => {
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(ins.iadd(lhs, rhs))
            }
            GlobalInit::I32Sub | GlobalInit::I64Sub => {
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(ins.isub(lhs, rhs))
            }
            GlobalInit::I32Mul | GlobalInit::I64Mul => {
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(ins.imul(lhs, rhs))
            }
        }
    }
    stack.pop().unwrap_or_else(|| ins.iconst(VelocType::I64, 0))
}

fn generate_trampolines(ir: &mut veloc::ir::ModuleBuilder, metadata: &mut ModuleMetadata) {
    for i in 0..metadata.functions.len() {
        let func_name = metadata.functions[i].name.clone();
        let is_import = i < metadata.num_imported_funcs;
        let linkage = if is_import {
            Linkage::Import
        } else {
            Linkage::Export
        };

        let ty_idx = metadata.functions[i].type_index;
        let wasm_sig = &metadata.signatures[ty_idx as usize];

        // 构造正确的函数签名
        let mut params = vec![VelocType::Ptr]; // vmctx
        params.extend(wasm_sig.params.iter().map(|&t| valtype_to_veloc(t)));
        if wasm_sig.results.len() > 1 {
            params.push(VelocType::Ptr); // result ptr
        }
        let ret = if wasm_sig.results.len() == 1 {
            valtype_to_veloc(wasm_sig.results[0])
        } else {
            VelocType::Void
        };
        let signature = Signature::new(params, ret, CallConv::SystemV);

        let func_id = ir.declare_function(func_name.clone(), signature, linkage);
        metadata.functions[i].func_id = func_id;

        // 仅为本地定义的函数生成 Array-to-Wasm Trampoline
        if !is_import {
            let tramp_name = format!("{}_trampoline", func_name);
            let tramp_ret = if wasm_sig.results.len() == 1 {
                VelocType::I64
            } else {
                VelocType::Void
            };
            let tramp_sig = Signature::new(
                vec![VelocType::Ptr, VelocType::Ptr],
                tramp_ret,
                CallConv::SystemV,
            );
            let tramp_id = ir.declare_function(tramp_name, tramp_sig, Linkage::Export);

            let sig = &metadata.signatures[ty_idx as usize];

            let mut builder = ir.builder(tramp_id);
            let mut ins = builder.ins();
            let params = ins.builder().func_params().to_vec();
            let vmctx = params[0];
            let args_ptr = params[1];

            let mut num_params = sig.params.len();
            if sig.results.len() > 1 {
                num_params += 1;
            }

            let mut call_args = Vec::with_capacity(num_params + 1);
            call_args.push(vmctx);
            for j in 0..num_params {
                let val_i64 = ins.load(VelocType::I64, args_ptr, (j * 8) as u32);
                if j < sig.params.len() {
                    let w_ty = sig.params[j];
                    let v_ty = valtype_to_veloc(w_ty);
                    let val = match v_ty {
                        VelocType::I32 => ins.wrap(val_i64, VelocType::I32),
                        VelocType::F32 => {
                            let b = ins.wrap(val_i64, VelocType::I32);
                            ins.reinterpret(b, VelocType::F32)
                        }
                        VelocType::F64 => ins.reinterpret(val_i64, VelocType::F64),
                        _ => val_i64,
                    };
                    call_args.push(val);
                } else {
                    call_args.push(ins.int_to_ptr(val_i64));
                }
            }

            let res = ins.call(func_id, &call_args);
            if let Some(res_val) = res {
                let res_ty = ins.builder().value_type(res_val);
                let res_i64 = match res_ty {
                    VelocType::I32 => ins.extend_u(res_val, VelocType::I64),
                    VelocType::F32 => {
                        let b = ins.reinterpret(res_val, VelocType::I32);
                        ins.extend_u(b, VelocType::I64)
                    }
                    VelocType::F64 => ins.reinterpret(res_val, VelocType::I64),
                    VelocType::Ptr => ins.ptr_to_int(res_val, VelocType::I64),
                    _ => res_val,
                };
                ins.ret(Some(res_i64));
            } else {
                ins.ret(None);
            }
        }
    }
}

fn generate_veloc_init(
    ir: &mut veloc::ir::ModuleBuilder,
    metadata: &ModuleMetadata,
    offsets: &VMOffsets,
    runtime: &RuntimeFunctions,
) -> veloc::ir::FuncId {
    let init_func_id = ir.declare_function(
        "__veloc_init".to_string(),
        veloc::ir::Signature::new(
            vec![veloc::ir::Type::Ptr],
            veloc::ir::Type::Void,
            CallConv::SystemV,
        ),
        Linkage::Export,
    );

    let mut builder = ir.builder(init_func_id);
    let mut ins = builder.ins();
    let vmctx = ins.builder().func_params()[0];

    // 1. Initialize globals
    for i in metadata.num_imported_globals..metadata.globals.len() {
        let global = &metadata.globals[i];
        let dst_ptr = ins.load(VelocType::Ptr, vmctx, offsets.global_offset(i as u32));
        let val = generate_init_expr(&mut ins, &global.init, vmctx, offsets, metadata);
        ins.store(val, dst_ptr, 0);
    }

    // 1b. Initialize tables with their default initializers
    for i in 0..metadata.tables.len() {
        if let Some(init_ops) = &metadata.tables[i].init {
            let val = generate_init_expr(&mut ins, init_ops, vmctx, offsets, metadata);
            let table_idx = ins.iconst(VelocType::I32, i as i64);
            ins.call(runtime.init_table, &[vmctx, table_idx, val]);
        }
    }

    // 2. Initialize tables (elements)
    for (i, element) in metadata.elements.iter().enumerate() {
        if !element.is_active {
            continue;
        }
        let offset = generate_init_expr(&mut ins, &element.offset, vmctx, offsets, metadata);
        let offset_i32 = if ins.builder().value_type(offset) == VelocType::I64 {
            ins.wrap(offset, VelocType::I32)
        } else {
            offset
        };
        let element_idx = ins.iconst(VelocType::I32, i as i64);
        ins.call(
            runtime.init_table_element,
            &[vmctx, element_idx, offset_i32],
        );

        ins.call(runtime.elem_drop, &[vmctx, element_idx]);
    }

    // 3. Initialize memory (data segments)
    for (i, data) in metadata.data.iter().enumerate() {
        if !data.is_active {
            continue;
        }
        let offset = generate_init_expr(&mut ins, &data.offset, vmctx, offsets, metadata);
        let offset_i32 = if ins.builder().value_type(offset) == VelocType::I64 {
            ins.wrap(offset, VelocType::I32)
        } else {
            offset
        };
        let data_idx = ins.iconst(VelocType::I32, i as i64);
        ins.call(runtime.init_memory_data, &[vmctx, data_idx, offset_i32]);

        ins.call(runtime.data_drop, &[vmctx, data_idx]);
    }

    // 4. Drop declarative segments
    for (i, element) in metadata.elements.iter().enumerate() {
        if !element.is_active && element.offset.is_empty() {
            let element_idx = ins.iconst(VelocType::I32, i as i64);
            ins.call(runtime.elem_drop, &[vmctx, element_idx]);
        }
    }

    ins.ret(None);

    init_func_id
}
