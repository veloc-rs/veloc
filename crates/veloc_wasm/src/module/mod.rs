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
use veloc::ir::{CallConv, FuncId, Linkage, MemFlags, Type as VelocType};
use wasmparser::{Parser, Payload, Validator};

pub use self::runtime::*;
pub use self::types::*;

pub enum ModuleArtifact {
    Interpreter(veloc::ir::Module),
    Jit(LoadedObject<()>),
}

struct ModuleInner {
    engine: Engine,
    artifact: ModuleArtifact,
    metadata: WasmMetadata,
    offsets: VMOffsets,
    init_func_id: FuncId,
}

#[derive(Debug, Clone, Copy)]
pub struct RuntimeFunctions {
    pub trap_handler: FuncId,
    pub memory_size: FuncId,
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
        let p = VelocType::Ptr;
        let i = VelocType::I32;
        let v = VelocType::Void;

        let sig = |ir: &mut veloc::ir::ModuleBuilder, params: Vec<VelocType>, ret: VelocType| {
            ir.make_signature(params, ret, CallConv::SystemV)
        };

        let trap_handler_sig = sig(ir, vec![p, i], v);
        let trap_handler = ir.declare_function(
            "wasm_trap_handler".into(),
            trap_handler_sig,
            Linkage::Import,
        );

        let memory_size_sig = sig(ir, vec![p, i], i);
        let memory_size =
            ir.declare_function("wasm_memory_size".into(), memory_size_sig, Linkage::Import);

        let memory_grow_sig = sig(ir, vec![p, i, i], i);
        let memory_grow =
            ir.declare_function("wasm_memory_grow".into(), memory_grow_sig, Linkage::Import);

        let init_table_element_sig = sig(ir, vec![p, i, i], v);
        let init_table_element = ir.declare_function(
            "wasm_init_table_element".into(),
            init_table_element_sig,
            Linkage::Import,
        );

        let init_memory_data_sig = sig(ir, vec![p, i, i], v);
        let init_memory_data = ir.declare_function(
            "wasm_init_memory_data".into(),
            init_memory_data_sig,
            Linkage::Import,
        );

        let init_table_sig = sig(ir, vec![p, i, p], v);
        let init_table =
            ir.declare_function("wasm_init_table".into(), init_table_sig, Linkage::Import);

        let table_init_sig = sig(ir, vec![p, i, i, i, i, i], v);
        let table_init =
            ir.declare_function("wasm_table_init".into(), table_init_sig, Linkage::Import);

        let table_copy_sig = sig(ir, vec![p, i, i, i, i, i], v);
        let table_copy =
            ir.declare_function("wasm_table_copy".into(), table_copy_sig, Linkage::Import);

        let table_grow_sig = sig(ir, vec![p, i, p, i], i);
        let table_grow =
            ir.declare_function("wasm_table_grow".into(), table_grow_sig, Linkage::Import);

        let table_size_sig = sig(ir, vec![p, i], i);
        let table_size =
            ir.declare_function("wasm_table_size".into(), table_size_sig, Linkage::Import);

        let table_fill_sig = sig(ir, vec![p, i, i, p, i], v);
        let table_fill =
            ir.declare_function("wasm_table_fill".into(), table_fill_sig, Linkage::Import);

        let elem_drop_sig = sig(ir, vec![p, i], v);
        let elem_drop =
            ir.declare_function("wasm_elem_drop".into(), elem_drop_sig, Linkage::Import);

        let memory_init_sig = sig(ir, vec![p, i, i, i, i, i], v);
        let memory_init =
            ir.declare_function("wasm_memory_init".into(), memory_init_sig, Linkage::Import);

        let data_drop_sig = sig(ir, vec![p, i], v);
        let data_drop =
            ir.declare_function("wasm_data_drop".into(), data_drop_sig, Linkage::Import);

        let memory_copy_sig = sig(ir, vec![p, i, i, i, i, i], v);
        let memory_copy =
            ir.declare_function("wasm_memory_copy".into(), memory_copy_sig, Linkage::Import);

        let memory_fill_sig = sig(ir, vec![p, i, i, i, i], v);
        let memory_fill =
            ir.declare_function("wasm_memory_fill".into(), memory_fill_sig, Linkage::Import);

        Self {
            trap_handler,
            memory_size,
            memory_grow,
            init_table_element,
            init_memory_data,
            init_table,
            table_init,
            table_copy,
            table_grow,
            table_size,
            table_fill,
            elem_drop,
            memory_init,
            data_drop,
            memory_copy,
            memory_fill,
        }
    }
}

#[derive(Clone)]
pub struct Module {
    inner: Arc<ModuleInner>,
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

    pub fn metadata(&self) -> &WasmMetadata {
        &self.inner.metadata
    }

    pub(crate) fn artifact(&self) -> &ModuleArtifact {
        &self.inner.artifact
    }

    pub(crate) fn loaded(&self) -> Option<&LoadedObject<()>> {
        match &self.inner.artifact {
            ModuleArtifact::Jit(loaded) => Some(loaded),
            _ => None,
        }
    }

    pub fn new(engine: &Engine, wasm_bin: &[u8]) -> Result<Self> {
        Validator::new().validate_all(wasm_bin)?;
        let mut metadata = WasmMetadata::collect(wasm_bin)?;
        let mut ir = veloc::ir::ModuleBuilder::new();

        let mut ir_sig_ids = Vec::with_capacity(metadata.signatures.len());
        for i in 0..metadata.signatures.len() {
            ir_sig_ids.push(metadata.signatures[i].intern_veloc_sig(&mut ir));
        }

        let mut strategy = engine.strategy();
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
        let parser = Parser::new(0);
        for payload in parser.parse_all(wasm_bin) {
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
                let mut translator = WasmTranslator::new(
                    &mut builder,
                    returns,
                    &metadata,
                    &ir_sig_ids,
                    offsets,
                    runtime,
                    engine.config().ir_names,
                );
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

        if engine.config().dump_ir {
            println!("Generated IR for module:");
            println!("{}", ir);
        }

        let artifact = if strategy == Strategy::Jit {
            let object_data = engine
                .backend()
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
            engine: engine.clone(),
            artifact,
            metadata,
            offsets,
            init_func_id,
        });

        Ok(Self { inner })
    }

    pub(crate) fn vm_offsets(&self) -> &VMOffsets {
        &self.inner.offsets
    }

    pub(crate) fn strategy(&self) -> Strategy {
        self.inner.engine.strategy()
    }

    pub(crate) fn init_func_id(&self) -> FuncId {
        self.inner.init_func_id
    }
}

fn generate_init_expr(
    ins: &mut veloc::ir::builder::InstBuilder,
    expr: &[GlobalInit],
    vmctx: veloc::ir::Value,
    offsets: &VMOffsets,
    metadata: &WasmMetadata,
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
                stack.push(ins.ptr_offset(vmctx, offset as i32))
            }
            GlobalInit::GlobalGet(idx) => {
                let offset = offsets.global_offset(*idx);
                let align = if offset % 16 == 0 { 16 } else { 8 };
                let src_ptr = ins.load(
                    VelocType::Ptr,
                    vmctx,
                    offset,
                    MemFlags::new().with_alignment(align),
                );
                let ty = valtype_to_veloc(metadata.globals[*idx as usize].ty);
                stack.push(ins.load(ty, src_ptr, 0, MemFlags::new().with_alignment(8)))
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

fn generate_trampolines(ir: &mut veloc::ir::ModuleBuilder, metadata: &mut WasmMetadata) {
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
        let sig_id = wasm_sig.intern_veloc_sig(ir);

        let func_id = ir.declare_function(func_name.clone(), sig_id, linkage);
        metadata.functions[i].func_id = func_id;

        // 仅为本地定义的函数生成 Array-to-Wasm Trampoline
        if !is_import {
            let tramp_name = format!("{}_trampoline", func_name);
            let tramp_ret = if wasm_sig.results.len() == 1 {
                VelocType::I64
            } else {
                VelocType::Void
            };
            let tramp_sig_id = ir.make_signature(
                vec![VelocType::Ptr, VelocType::Ptr],
                tramp_ret,
                CallConv::SystemV,
            );
            let tramp_id = ir.declare_function(tramp_name, tramp_sig_id, Linkage::Export);

            let sig = &metadata.signatures[ty_idx as usize];

            let mut builder = ir.builder(tramp_id);
            builder.init_entry_block();
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
                let val_i64 = ins.load(
                    VelocType::I64,
                    args_ptr,
                    (j * 8) as u32,
                    MemFlags::default(),
                );
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
                        VelocType::Ptr => ins.int_to_ptr(val_i64),
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
            builder.seal_all_blocks();
        }
    }
}

fn generate_veloc_init(
    ir: &mut veloc::ir::ModuleBuilder,
    metadata: &WasmMetadata,
    offsets: &VMOffsets,
    runtime: &RuntimeFunctions,
) -> veloc::ir::FuncId {
    let init_sig_id = ir.make_signature(
        vec![veloc::ir::Type::Ptr],
        veloc::ir::Type::Void,
        CallConv::SystemV,
    );
    let init_func_id =
        ir.declare_function("__veloc_init".to_string(), init_sig_id, Linkage::Export);

    let mut builder = ir.builder(init_func_id);
    builder.init_entry_block();
    let mut ins = builder.ins();
    let vmctx = ins.builder().func_params()[0];

    // 1. Initialize globals
    for i in metadata.num_imported_globals..metadata.globals.len() {
        let global = &metadata.globals[i];
        let offset = offsets.global_offset(i as u32);
        let align = if offset % 16 == 0 { 16 } else { 8 };
        let dst_ptr = ins.load(
            VelocType::Ptr,
            vmctx,
            offset,
            MemFlags::new().with_alignment(align),
        );
        let val = generate_init_expr(&mut ins, &global.init, vmctx, offsets, metadata);
        ins.store(val, dst_ptr, 0, MemFlags::new().with_alignment(8));
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
        if element.is_declared {
            let element_idx = ins.iconst(VelocType::I32, i as i64);
            ins.call(runtime.elem_drop, &[vmctx, element_idx]);
        }
    }

    // 5. Call start function
    if let Some(start_idx) = metadata.start_func {
        let start_func_id = metadata.functions[start_idx as usize].func_id;
        ins.call(start_func_id, &[vmctx]);
    }

    ins.ret(None);
    builder.seal_all_blocks();

    init_func_id
}
