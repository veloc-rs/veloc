use crate::module::ModuleMetadata;
use crate::module::RuntimeFunctions;
use crate::vm::{TrapCode, VMFuncRef, VMMemory, VMOffsets, VMTable};
use alloc::vec::Vec;
use hashbrown::HashMap;
use veloc::ir::{FloatCC, FunctionBuilder, InstBuilder};
use veloc::ir::{IntCC, Type as VelocType, Value};
use wasmparser::{BinaryReaderError, Operator};

pub struct WasmTranslator<'a> {
    builder: &'a mut FunctionBuilder<'a>,
    stack: Vec<Value>,
    locals: Vec<(veloc::ir::Variable, VelocType)>,
    next_var_idx: u32,
    control_stack: Vec<ControlFrame>,
    vmctx: Option<Value>,
    results_ptr: Option<Value>,
    results: Vec<VelocType>,
    terminated: bool,
    // Add these
    pub metadata: &'a ModuleMetadata,
    pub offsets: VMOffsets,
    pub runtime: RuntimeFunctions,

    cached_memories: HashMap<u32, Value>,
    cached_tables: HashMap<u32, Value>,
    cached_globals: HashMap<u32, Value>,
}

struct ControlFrame {
    label: veloc::ir::Block,
    end_label: Option<veloc::ir::Block>, // Wasm Block/If 的 end 目标，Loop 的退出目标
    else_label: Option<veloc::ir::Block>,
    is_loop: bool,
    stack_size: usize,
    num_params: usize,
    num_results: usize,
}

impl<'a> WasmTranslator<'a> {
    pub fn new(
        builder: &'a mut FunctionBuilder<'a>,
        results: Vec<VelocType>,
        metadata: &'a ModuleMetadata,
        offsets: VMOffsets,
        runtime: RuntimeFunctions,
    ) -> Self {
        Self {
            builder,
            stack: Vec::new(),
            locals: Vec::new(),
            control_stack: Vec::new(),
            vmctx: None,
            results_ptr: None,
            results,
            terminated: false,
            metadata,
            offsets,
            runtime,
            next_var_idx: 0,
            cached_memories: HashMap::new(),
            cached_tables: HashMap::new(),
            cached_globals: HashMap::new(),
        }
    }

    fn new_var(&mut self, ty: VelocType) -> veloc::ir::Variable {
        let var = veloc::ir::Variable(self.next_var_idx);
        self.next_var_idx += 1;
        self.builder.declare_var(var, ty);
        var
    }

    pub fn translate(
        &mut self,
        code: wasmparser::FunctionBody,
        param_types: &[VelocType],
    ) -> Result<(), BinaryReaderError> {
        let mut reader = code.get_operators_reader()?;
        let entry = self.builder.entry_block().unwrap();
        let end_block = self.builder.create_block();

        self.builder.switch_to_block(entry);

        self.control_stack.push(ControlFrame {
            label: entry,
            end_label: Some(end_block),
            else_label: None,
            is_loop: false,
            stack_size: 0,
            num_params: 0,
            num_results: self.results.len(),
        });

        for &ty in &self.results {
            self.builder.add_block_param(end_block, ty);
        }

        let params = self.builder.func_params().to_vec();
        let mut params_iter = params.into_iter();

        let vmctx = params_iter.next().expect("Missing vmctx parameter");
        self.vmctx = Some(vmctx);

        for (_i, &ty) in param_types.iter().enumerate() {
            let var = self.new_var(ty);
            self.locals.push((var, ty));
            let val = params_iter.next().expect("Missing parameter");
            self.builder.def_var(var, val);
        }

        if self.results.len() > 1 {
            let ptr = params_iter.next().expect("Missing results_ptr parameter");
            self.results_ptr = Some(ptr);
        }

        let mut locals_reader = code.get_locals_reader()?;
        for _ in 0..locals_reader.get_count() {
            let (count, ty) = locals_reader.read()?;
            let ty = match ty {
                wasmparser::ValType::I32 => VelocType::I32,
                wasmparser::ValType::I64 => VelocType::I64,
                wasmparser::ValType::F32 => VelocType::F32,
                wasmparser::ValType::F64 => VelocType::F64,
                _ => VelocType::I64,
            };
            for _ in 0..count {
                let var = self.new_var(ty);
                self.locals.push((var, ty));
                let zero = if ty.is_float() {
                    self.builder.ins().fconst(ty, 0)
                } else if ty == VelocType::Bool {
                    self.builder.ins().bconst(false)
                } else {
                    self.builder.ins().iconst(ty, 0)
                };
                self.builder.def_var(var, zero);
            }
        }

        while !reader.eof() {
            let op = reader.read()?;
            match op {
                Operator::Block { .. }
                | Operator::Loop { .. }
                | Operator::If { .. }
                | Operator::Else
                | Operator::End => {
                    self.translate_operator(op)?;
                }
                _ => {
                    if !self.terminated {
                        self.translate_operator(op)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn translate_operator(&mut self, op: Operator) -> Result<(), BinaryReaderError> {
        match op {
            Operator::I32Const { value } => {
                let v = self.builder.ins().i32const(value);
                self.stack.push(v);
            }
            Operator::I64Const { value } => {
                let v = self.builder.ins().i64const(value);
                self.stack.push(v);
            }
            Operator::F32Const { value } => {
                let v = self.builder.ins().f32const(f32::from_bits(value.bits()));
                self.stack.push(v);
            }
            Operator::F64Const { value } => {
                let v = self.builder.ins().f64const(f64::from_bits(value.bits()));
                self.stack.push(v);
            }
            Operator::RefNull { .. } => {
                let v = self.builder.ins().iconst(VelocType::I64, 0);
                let null_ptr = self.builder.ins().int_to_ptr(v);
                self.stack.push(null_ptr);
            }
            Operator::RefFunc { function_index } => {
                let vmctx = self.vmctx.expect("vmctx not set");
                let offset = self.builder.ins().iconst(
                    VelocType::I64,
                    self.offsets.function_offset(function_index) as i64,
                );
                let ptr_addr = self.builder.ins().gep(vmctx, offset);
                self.stack.push(ptr_addr);
            }
            Operator::RefIsNull => {
                let v = self.pop();
                let v_ty = self.builder.value_type(v);
                let zero = if v_ty == VelocType::Ptr {
                    let z = self.builder.ins().iconst(VelocType::I64, 0);
                    self.builder.ins().int_to_ptr(z)
                } else {
                    self.builder.ins().iconst(v_ty, 0)
                };
                let res = self.builder.ins().icmp(IntCC::Eq, v, zero);
                // Wasm ref.is_null 返回 i32 (0 或 1)
                let true_val = self.builder.ins().iconst(VelocType::I32, 1);
                let false_val = self.builder.ins().iconst(VelocType::I32, 0);
                let res_i32 = self.builder.ins().select(res, true_val, false_val);
                self.stack.push(res_i32);
            }
            Operator::Drop => {
                self.stack.pop();
            }
            Operator::Select | Operator::TypedSelect { .. } => {
                let cond_i32 = self.pop();
                let cond_ty = self.builder.value_type(cond_i32);
                let zero = self.builder.ins().iconst(cond_ty, 0);
                let cond = self.builder.ins().icmp(IntCC::Ne, cond_i32, zero);
                let val2 = self.pop();
                let val1 = self.pop();
                let res = self.builder.ins().select(cond, val1, val2);
                self.stack.push(res);
            }
            Operator::LocalGet { local_index } => {
                let (var, _) = self.locals[local_index as usize];
                let val = self.builder.use_var(var);
                self.stack.push(val);
            }
            Operator::LocalSet { local_index } => {
                let val = self.pop();
                let (var, _) = self.locals[local_index as usize];
                self.builder.def_var(var, val);
            }
            Operator::LocalTee { local_index } => {
                let val = if let Some(&v) = self.stack.last() {
                    v
                } else {
                    let zero = self.builder.ins().iconst(VelocType::I64, 0);
                    self.stack.push(zero);
                    zero
                };
                let (var, _) = self.locals[local_index as usize];
                self.builder.def_var(var, val);
            }
            Operator::GlobalGet { global_index } => {
                let ty = self.metadata.globals[global_index as usize].ty;
                let veloc_ty = self.val_type_to_veloc(ty);
                let global_val_ptr = self.get_global_ptr(global_index);
                let val = self.builder.ins().load(veloc_ty, global_val_ptr, 0);
                self.stack.push(val);
            }
            Operator::GlobalSet { global_index } => {
                let _ty = self.metadata.globals[global_index as usize].ty;
                let val = self.pop();
                let global_val_ptr = self.get_global_ptr(global_index);
                self.builder.ins().store(val, global_val_ptr, 0);
            }
            Operator::TableGet { table } => {
                let index = self.pop();
                self.table_bounds_check(table, index);
                let table_base = self.get_table_base(table);
                let index_i64 = self.builder.ins().extend_u(index, VelocType::I64);
                let stride = self.builder.ins().iconst(VelocType::I64, 8);
                let offset = self.builder.ins().imul(index_i64, stride);
                let addr = self.builder.ins().gep(table_base, offset);

                let res = self.builder.ins().load(VelocType::Ptr, addr, 0);
                self.stack.push(res);
            }
            Operator::TableSet { table } => {
                let func_ref = self.pop(); // This is the pointer (*mut VMFuncRef)
                let index = self.pop();
                self.table_bounds_check(table, index);
                let table_base = self.get_table_base(table);
                let index_i64 = self.builder.ins().extend_u(index, VelocType::I64);
                let stride = self.builder.ins().iconst(VelocType::I64, 8);
                let offset = self.builder.ins().imul(index_i64, stride);
                let entry_addr = self.builder.ins().gep(table_base, offset);

                self.builder.ins().store(func_ref, entry_addr, 0);
                self.terminated = false;
            }
            Operator::I32Load { memarg } => self.translate_load(VelocType::I32, memarg),
            Operator::I64Load { memarg } => self.translate_load(VelocType::I64, memarg),
            Operator::F32Load { memarg } => self.translate_load(VelocType::F32, memarg),
            Operator::F64Load { memarg } => self.translate_load(VelocType::F64, memarg),
            Operator::I32Load8S { memarg } => {
                self.translate_load(VelocType::I8, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extend_s(v, VelocType::I32);
                self.stack.push(extended);
            }
            Operator::I32Load8U { memarg } => {
                self.translate_load(VelocType::I8, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extend_u(v, VelocType::I32);
                self.stack.push(extended);
            }
            Operator::I32Load16S { memarg } => {
                self.translate_load(VelocType::I16, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extend_s(v, VelocType::I32);
                self.stack.push(extended);
            }
            Operator::I32Load16U { memarg } => {
                self.translate_load(VelocType::I16, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extend_u(v, VelocType::I32);
                self.stack.push(extended);
            }
            Operator::I64Load8S { memarg } => {
                self.translate_load(VelocType::I8, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extend_s(v, VelocType::I64);
                self.stack.push(extended);
            }
            Operator::I64Load8U { memarg } => {
                self.translate_load(VelocType::I8, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extend_u(v, VelocType::I64);
                self.stack.push(extended);
            }
            Operator::I64Load16S { memarg } => {
                self.translate_load(VelocType::I16, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extend_s(v, VelocType::I64);
                self.stack.push(extended);
            }
            Operator::I64Load16U { memarg } => {
                self.translate_load(VelocType::I16, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extend_u(v, VelocType::I64);
                self.stack.push(extended);
            }
            Operator::I64Load32S { memarg } => {
                self.translate_load(VelocType::I32, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extend_s(v, VelocType::I64);
                self.stack.push(extended);
            }
            Operator::I64Load32U { memarg } => {
                self.translate_load(VelocType::I32, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extend_u(v, VelocType::I64);
                self.stack.push(extended);
            }
            Operator::I32Store { memarg } => self.translate_store(VelocType::I32, memarg),
            Operator::I64Store { memarg } => self.translate_store(VelocType::I64, memarg),
            Operator::F32Store { memarg } => self.translate_store(VelocType::F32, memarg),
            Operator::F64Store { memarg } => self.translate_store(VelocType::F64, memarg),
            Operator::I32Store8 { memarg } => {
                let val = self.pop();
                let truncated = self.builder.ins().wrap(val, VelocType::I8);
                self.stack.push(truncated);
                self.translate_store(VelocType::I8, memarg);
            }
            Operator::I32Store16 { memarg } => {
                let val = self.pop();
                let truncated = self.builder.ins().wrap(val, VelocType::I16);
                self.stack.push(truncated);
                self.translate_store(VelocType::I16, memarg);
            }
            Operator::I64Store8 { memarg } => {
                let val = self.pop();
                let truncated = self.builder.ins().wrap(val, VelocType::I8);
                self.stack.push(truncated);
                self.translate_store(VelocType::I8, memarg);
            }
            Operator::I64Store16 { memarg } => {
                let val = self.pop();
                let truncated = self.builder.ins().wrap(val, VelocType::I16);
                self.stack.push(truncated);
                self.translate_store(VelocType::I16, memarg);
            }
            Operator::I64Store32 { memarg } => {
                let val = self.pop();
                let truncated = self.builder.ins().wrap(val, VelocType::I32);
                self.stack.push(truncated);
                self.translate_store(VelocType::I32, memarg);
            }
            Operator::I32Add | Operator::I64Add => self.bin(|b, l, r| b.iadd(l, r)),
            Operator::I32Sub | Operator::I64Sub => self.bin(|b, l, r| b.isub(l, r)),
            Operator::I32Mul | Operator::I64Mul => self.bin(|b, l, r| b.imul(l, r)),
            Operator::I32DivS => self.translate_div_s(false),
            Operator::I64DivS => self.translate_div_s(true),
            Operator::I32DivU => self.translate_div_u(false),
            Operator::I64DivU => self.translate_div_u(true),
            Operator::I32RemS => self.translate_rem_s(false),
            Operator::I64RemS => self.translate_rem_s(true),
            Operator::I32RemU => self.translate_rem_u(false),
            Operator::I64RemU => self.translate_rem_u(true),
            Operator::I32And | Operator::I64And => self.bin(|b, l, r| b.and(l, r)),
            Operator::I32Or | Operator::I64Or => self.bin(|b, l, r| b.or(l, r)),
            Operator::I32Xor | Operator::I64Xor => self.bin(|b, l, r| b.xor(l, r)),
            Operator::I32Shl | Operator::I64Shl => self.bin(|b, l, r| b.shl(l, r)),
            Operator::I32ShrS | Operator::I64ShrS => self.bin(|b, l, r| b.shr_s(l, r)),
            Operator::I32ShrU | Operator::I64ShrU => self.bin(|b, l, r| b.shr_u(l, r)),
            Operator::I32Rotl | Operator::I64Rotl => self.bin(|b, l, r| b.rotl(l, r)),
            Operator::I32Rotr | Operator::I64Rotr => self.bin(|b, l, r| b.rotr(l, r)),
            Operator::I32Eq | Operator::I64Eq => self.bin_cmp(|b, l, r| b.eq(l, r)),
            Operator::I32Ne | Operator::I64Ne => self.bin_cmp(|b, l, r| b.ne(l, r)),
            Operator::I32LtS | Operator::I64LtS => self.bin_cmp(|b, l, r| b.lt_s(l, r)),
            Operator::I32LtU | Operator::I64LtU => self.bin_cmp(|b, l, r| b.lt_u(l, r)),
            Operator::I32GtS | Operator::I64GtS => self.bin_cmp(|b, l, r| b.gt_s(l, r)),
            Operator::I32GtU | Operator::I64GtU => self.bin_cmp(|b, l, r| b.gt_u(l, r)),
            Operator::I32LeS | Operator::I64LeS => self.bin_cmp(|b, l, r| b.le_s(l, r)),
            Operator::I32LeU | Operator::I64LeU => self.bin_cmp(|b, l, r| b.le_u(l, r)),
            Operator::I32GeS | Operator::I64GeS => self.bin_cmp(|b, l, r| b.ge_s(l, r)),
            Operator::I32GeU | Operator::I64GeU => self.bin_cmp(|b, l, r| b.ge_u(l, r)),
            Operator::I32Eqz | Operator::I64Eqz => self.un_cmp(|b, v| b.eqz(v)),
            Operator::I32Clz | Operator::I64Clz => self.un(|b, v| b.clz(v)),
            Operator::I32Ctz | Operator::I64Ctz => self.un(|b, v| b.ctz(v)),
            Operator::I32Popcnt | Operator::I64Popcnt => self.un(|b, v| b.popcnt(v)),

            Operator::I32Extend8S => self.un(|b, v| {
                let tmp = b.wrap(v, VelocType::I8);
                b.extend_s(tmp, VelocType::I32)
            }),
            Operator::I32Extend16S => self.un(|b, v| {
                let tmp = b.wrap(v, VelocType::I16);
                b.extend_s(tmp, VelocType::I32)
            }),
            Operator::I64Extend8S => self.un(|b, v| {
                let tmp = b.wrap(v, VelocType::I8);
                b.extend_s(tmp, VelocType::I64)
            }),
            Operator::I64Extend16S => self.un(|b, v| {
                let tmp = b.wrap(v, VelocType::I16);
                b.extend_s(tmp, VelocType::I64)
            }),
            Operator::I64Extend32S => self.un(|b, v| {
                let tmp = b.wrap(v, VelocType::I32);
                b.extend_s(tmp, VelocType::I64)
            }),

            Operator::I32WrapI64 => self.un(|b, v| b.wrap(v, VelocType::I32)),
            Operator::MemorySize { mem, .. } => {
                let vmctx = self.vmctx.expect("vmctx not set");
                let def_ptr =
                    self.builder
                        .ins()
                        .load(VelocType::Ptr, vmctx, self.offsets.memory_offset(mem));
                let size_bytes = self.builder.ins().load(
                    VelocType::I64,
                    def_ptr,
                    VMMemory::current_length_offset(),
                );
                let page_size = self.builder.ins().iconst(VelocType::I64, 65536);
                let size_pages = self.builder.ins().udiv(size_bytes, page_size);
                let size_i32 = self.builder.ins().wrap(size_pages, VelocType::I32);
                self.stack.push(size_i32);
            }
            Operator::MemoryGrow { mem, .. } => {
                let delta = self.pop();
                let vmctx = self.vmctx.expect("vmctx not set");
                let mem_idx = self.builder.ins().iconst(VelocType::I32, mem as i64);
                let res = self
                    .builder
                    .ins()
                    .call(self.runtime.memory_grow, &[vmctx, mem_idx, delta]);
                self.stack
                    .push(res.expect("memory.grow should return a value"));
                // Clear caches because memory grow might relocate memory
                self.cached_memories.clear();
            }
            Operator::I64ExtendI32S => self.un(|b, v| b.extend_s(v, VelocType::I64)),
            Operator::I64ExtendI32U => self.un(|b, v| b.extend_u(v, VelocType::I64)),

            Operator::F64Add => self.bin(|b, l, r| b.fadd(l, r)),
            Operator::F64Sub => self.bin(|b, l, r| b.fsub(l, r)),
            Operator::F64Mul => self.bin(|b, l, r| b.fmul(l, r)),
            Operator::F64Div => self.bin(|b, l, r| b.fdiv(l, r)),
            Operator::F32Add => self.bin(|b, l, r| b.fadd(l, r)),
            Operator::F32Sub => self.bin(|b, l, r| b.fsub(l, r)),
            Operator::F32Mul => self.bin(|b, l, r| b.fmul(l, r)),
            Operator::F32Div => self.bin(|b, l, r| b.fdiv(l, r)),
            Operator::F32Neg => self.un(|b, v| b.fneg(v)),
            Operator::F32Abs => self.un(|b, v| b.abs(v)),
            Operator::F64Neg => self.un(|b, v| b.fneg(v)),
            Operator::F64Abs => self.un(|b, v| b.abs(v)),
            Operator::F32Sqrt => self.un(|b, v| b.sqrt(v)),
            Operator::F64Sqrt => self.un(|b, v| b.sqrt(v)),
            Operator::F32Ceil => self.un(|b, v| b.ceil(v)),
            Operator::F64Ceil => self.un(|b, v| b.ceil(v)),
            Operator::F32Floor => self.un(|b, v| b.floor(v)),
            Operator::F64Floor => self.un(|b, v| b.floor(v)),
            Operator::F32Trunc => self.un(|b, v| b.trunc(v)),
            Operator::F64Trunc => self.un(|b, v| b.trunc(v)),
            Operator::F32Nearest => self.un(|b, v| b.nearest(v)),
            Operator::F64Nearest => self.un(|b, v| b.nearest(v)),

            Operator::F32Min => self.translate_fmin_fmax(false, true),
            Operator::F64Min => self.translate_fmin_fmax(true, true),
            Operator::F32Max => self.translate_fmin_fmax(false, false),
            Operator::F64Max => self.translate_fmin_fmax(true, false),
            Operator::F32Copysign => self.bin(|b, l, r| b.copysign(l, r)),
            Operator::F64Copysign => self.bin(|b, l, r| b.copysign(l, r)),

            Operator::F32Eq => self.bin_cmp(|b, l, r| b.feq(l, r)),
            Operator::F32Ne => self.bin_cmp(|b, l, r| b.fne(l, r)),
            Operator::F32Lt => self.bin_cmp(|b, l, r| b.flt(l, r)),
            Operator::F32Gt => self.bin_cmp(|b, l, r| b.fgt(l, r)),
            Operator::F32Le => self.bin_cmp(|b, l, r| b.fle(l, r)),
            Operator::F32Ge => self.bin_cmp(|b, l, r| b.fge(l, r)),

            Operator::F64Eq => self.bin_cmp(|b, l, r| b.feq(l, r)),
            Operator::F64Ne => self.bin_cmp(|b, l, r| b.fne(l, r)),
            Operator::F64Lt => self.bin_cmp(|b, l, r| b.flt(l, r)),
            Operator::F64Gt => self.bin_cmp(|b, l, r| b.fgt(l, r)),
            Operator::F64Le => self.bin_cmp(|b, l, r| b.fle(l, r)),
            Operator::F64Ge => self.bin_cmp(|b, l, r| b.fge(l, r)),

            Operator::F64PromoteF32 => self.un(|b, v| b.promote(v, VelocType::F64)),
            Operator::F32DemoteF64 => self.un(|b, v| b.demote(v, VelocType::F32)),
            Operator::F64ConvertI32S => self.un(|b, v| b.convert_s(v, VelocType::F64)),
            Operator::F64ConvertI32U => self.un(|b, v| b.convert_u(v, VelocType::F64)),
            Operator::F64ConvertI64S => self.un(|b, v| b.convert_s(v, VelocType::F64)),
            Operator::F64ConvertI64U => self.un(|b, v| b.convert_u(v, VelocType::F64)),
            Operator::I32TruncF64S => self.un(|b, v| b.trunc_s(v, VelocType::I32)),
            Operator::I32TruncF64U => self.un(|b, v| b.trunc_u(v, VelocType::I32)),
            Operator::I64TruncF64S => self.un(|b, v| b.trunc_s(v, VelocType::I64)),
            Operator::I64TruncF64U => self.un(|b, v| b.trunc_u(v, VelocType::I64)),
            Operator::I64TruncF32S => self.un(|b, v| b.trunc_s(v, VelocType::I64)),
            Operator::I64TruncF32U => self.un(|b, v| b.trunc_u(v, VelocType::I64)),
            Operator::Nop => {}
            Operator::Unreachable => {
                self.trap(TrapCode::Unreachable);
            }
            Operator::TableInit { table, elem_index } => {
                let len = self.pop();
                let src = self.pop();
                let dst = self.pop();
                let vmctx = self.vmctx.expect("vmctx not set");
                let table_idx = self.builder.ins().iconst(VelocType::I32, table as i64);
                let elem_idx = self.builder.ins().iconst(VelocType::I32, elem_index as i64);
                self.builder.ins().call(
                    self.runtime.table_init,
                    &[vmctx, table_idx, elem_idx, dst, src, len],
                );
            }
            Operator::ElemDrop { elem_index } => {
                let vmctx = self.vmctx.expect("vmctx not set");
                let elem_idx = self.builder.ins().iconst(VelocType::I32, elem_index as i64);
                self.builder
                    .ins()
                    .call(self.runtime.elem_drop, &[vmctx, elem_idx]);
            }
            Operator::TableCopy {
                dst_table,
                src_table,
            } => {
                let len = self.pop();
                let src = self.pop();
                let dst = self.pop();
                let vmctx = self.vmctx.expect("vmctx not set");
                let dst_table_val = self.builder.ins().iconst(VelocType::I32, dst_table as i64);
                let src_table_val = self.builder.ins().iconst(VelocType::I32, src_table as i64);
                self.builder.ins().call(
                    self.runtime.table_copy,
                    &[vmctx, dst_table_val, src_table_val, dst, src, len],
                );
            }
            Operator::TableGrow { table } => {
                let delta = self.pop();
                let init_val = self.pop();
                let vmctx = self.vmctx.expect("vmctx not set");
                let table_idx = self.builder.ins().iconst(VelocType::I32, table as i64);
                let res = self.builder.ins().call(
                    self.runtime.table_grow,
                    &[vmctx, table_idx, init_val, delta],
                );
                self.stack.push(res.unwrap());
            }
            Operator::TableSize { table } => {
                let vmctx = self.vmctx.expect("vmctx not set");
                let table_idx = self.builder.ins().iconst(VelocType::I32, table as i64);
                let res = self
                    .builder
                    .ins()
                    .call(self.runtime.table_size, &[vmctx, table_idx]);
                self.stack.push(res.unwrap());
            }
            Operator::TableFill { table } => {
                let len = self.pop();
                let val = self.pop();
                let dst = self.pop();
                let vmctx = self.vmctx.expect("vmctx not set");
                let table_idx = self.builder.ins().iconst(VelocType::I32, table as i64);
                self.builder
                    .ins()
                    .call(self.runtime.table_fill, &[vmctx, table_idx, dst, val, len]);
            }
            Operator::MemoryInit { data_index, mem } => {
                let len = self.pop();
                let src = self.pop();
                let dst = self.pop();
                let vmctx = self.vmctx.expect("vmctx not set");
                let mem_idx = self.builder.ins().iconst(VelocType::I32, mem as i64);
                let data_idx = self.builder.ins().iconst(VelocType::I32, data_index as i64);
                self.builder.ins().call(
                    self.runtime.memory_init,
                    &[vmctx, mem_idx, data_idx, dst, src, len],
                );
            }
            Operator::DataDrop { data_index } => {
                let vmctx = self.vmctx.expect("vmctx not set");
                let data_idx = self.builder.ins().iconst(VelocType::I32, data_index as i64);
                self.builder
                    .ins()
                    .call(self.runtime.data_drop, &[vmctx, data_idx]);
            }
            Operator::MemoryCopy { dst_mem, src_mem } => {
                let len = self.pop();
                let src = self.pop();
                let dst = self.pop();
                let vmctx = self.vmctx.expect("vmctx not set");
                let dst_mem_val = self.builder.ins().iconst(VelocType::I32, dst_mem as i64);
                let src_mem_val = self.builder.ins().iconst(VelocType::I32, src_mem as i64);
                self.builder.ins().call(
                    self.runtime.memory_copy,
                    &[vmctx, dst_mem_val, src_mem_val, dst, src, len],
                );
            }
            Operator::MemoryFill { mem } => {
                let len = self.pop();
                let val = self.pop();
                let dst = self.pop();
                let vmctx = self.vmctx.expect("vmctx not set");
                let mem_idx = self.builder.ins().iconst(VelocType::I32, mem as i64);
                self.builder
                    .ins()
                    .call(self.runtime.memory_fill, &[vmctx, mem_idx, dst, val, len]);
            }
            Operator::I32ReinterpretF32 => {
                let v = self.pop();
                let res = self.builder.ins().reinterpret(v, VelocType::I32);
                self.stack.push(res);
            }
            Operator::I64ReinterpretF64 => {
                let v = self.pop();
                let res = self.builder.ins().reinterpret(v, VelocType::I64);
                self.stack.push(res);
            }
            Operator::F32ReinterpretI32 => {
                let v = self.pop();
                let res = self.builder.ins().reinterpret(v, VelocType::F32);
                self.stack.push(res);
            }
            Operator::F64ReinterpretI64 => {
                let v = self.pop();
                let res = self.builder.ins().reinterpret(v, VelocType::F64);
                self.stack.push(res);
            }
            Operator::Block { blockty } => {
                let (params_ty, results_ty) = self.block_params_results(blockty);
                let end_block = self.builder.create_block();
                for &ty in &results_ty {
                    self.builder.add_block_param(end_block, ty);
                }

                self.control_stack.push(ControlFrame {
                    label: end_block,
                    end_label: Some(end_block),
                    else_label: None,
                    is_loop: false,
                    stack_size: self.stack.len() - params_ty.len(),
                    num_params: params_ty.len(),
                    num_results: results_ty.len(),
                });
            }
            Operator::Loop { blockty } => {
                let (params_ty, results_ty) = self.block_params_results(blockty);
                let header_block = self.builder.create_block();
                let end_block = self.builder.create_block();

                for &ty in &params_ty {
                    self.builder.add_block_param(header_block, ty);
                }
                for &ty in &results_ty {
                    self.builder.add_block_param(end_block, ty);
                }

                let mut args = Vec::new();
                for _ in 0..params_ty.len() {
                    args.push(self.pop());
                }
                args.reverse();

                self.builder.ins().jump(header_block, &args);
                self.builder.switch_to_block(header_block);
                if self.terminated {
                    self.builder.ins().unreachable();
                }

                // 将 header_block 的参数重新推入 stack
                for i in 0..params_ty.len() {
                    let val = self.builder.block_params(header_block)[i];
                    self.stack.push(val);
                }

                self.control_stack.push(ControlFrame {
                    label: header_block,
                    end_label: Some(end_block),
                    else_label: None,
                    is_loop: true,
                    stack_size: self.stack.len() - params_ty.len(),
                    num_params: params_ty.len(),
                    num_results: results_ty.len(),
                });
            }
            Operator::Br { relative_depth } => {
                let frame_idx = self.control_stack.len() - 1 - relative_depth as usize;
                let frame = &self.control_stack[frame_idx];
                let (target, params_len) = if frame.is_loop {
                    (frame.label, frame.num_params)
                } else {
                    (frame.end_label.unwrap(), frame.num_results)
                };

                let mut args = Vec::new();
                for _ in 0..params_len {
                    args.push(self.pop());
                }
                args.reverse();

                self.builder.ins().jump(target, &args);
                self.terminated = true;
            }
            Operator::BrIf { relative_depth } => {
                let cond_i32 = self.pop();
                let cond_ty = self.builder.value_type(cond_i32);
                let zero = self.builder.ins().iconst(cond_ty, 0);
                let cond = self.builder.ins().icmp(IntCC::Ne, cond_i32, zero);
                let frame_idx = self.control_stack.len() - 1 - relative_depth as usize;
                let frame = &self.control_stack[frame_idx];
                let (target, params_len) = if frame.is_loop {
                    (frame.label, frame.num_params)
                } else {
                    (frame.end_label.unwrap(), frame.num_results)
                };

                let next_block = self.builder.create_block();
                let mut args = Vec::new();
                if params_len > 0 {
                    let mut tmp_stack = Vec::new();
                    for _ in 0..params_len {
                        tmp_stack.push(self.pop());
                    }
                    for &val in tmp_stack.iter().rev() {
                        args.push(val);
                        self.stack.push(val);
                    }
                }

                self.builder.ins().br(cond, target, &args, next_block, &[]);
                self.builder.seal_block(next_block);
                self.builder.switch_to_block(next_block);
            }
            Operator::Call { function_index } => {
                let ty_idx = self.metadata.functions[function_index as usize].type_index;
                let sig = &self.metadata.signatures[ty_idx as usize];

                let mut args = Vec::new();
                for _ in sig.params.iter() {
                    args.push(self.pop());
                }
                args.reverse();

                let results = &sig.results;

                if (function_index as usize) < self.metadata.num_imported_funcs {
                    let entry_offset = self.offsets.function_offset(function_index as u32);
                    let vmptr = self.vmctx.expect("vmctx not set");
                    // Functions in VMContext now store VMFuncRef directly
                    let func_ptr = self.builder.ins().load(
                        VelocType::Ptr,
                        vmptr,
                        entry_offset + VMFuncRef::func_ptr_offset(),
                    );
                    let target_vmctx = self.builder.ins().load(
                        VelocType::Ptr,
                        vmptr,
                        entry_offset + VMFuncRef::vmctx_offset(),
                    );
                    args.insert(0, target_vmctx);

                    let func_id = self.metadata.functions[function_index as usize].func_id;
                    let sig_id = self.builder.func_signature(func_id);

                    if results.len() <= 1 {
                        let res = self.builder.ins().call_indirect(sig_id, func_ptr, &args);
                        if results.len() == 1 {
                            self.stack.push(res.expect("Missing call result"));
                        }
                        self.terminated = false;
                        return Ok(());
                    } else {
                        let ss = self.builder.create_stack_slot((results.len() * 8) as u32);
                        let res_ptr = self.builder.ins().stack_addr(ss, 0);
                        args.push(res_ptr);
                        self.builder.ins().call_indirect(sig_id, func_ptr, &args);

                        for i in 0..results.len() {
                            let ty = self.val_type_to_veloc(results[i]);
                            let val = self.builder.ins().stack_load(ty, ss, (i * 8) as u32);
                            self.stack.push(val);
                        }
                        self.terminated = false;
                        return Ok(());
                    }
                }

                args.insert(0, self.vmctx.expect("vmctx not set"));
                let func_id = self.metadata.functions[function_index as usize].func_id;
                if results.len() <= 1 {
                    let res = self.builder.ins().call(func_id, &args);
                    if results.len() == 1 {
                        self.stack.push(res.expect("call should have result"));
                    }
                } else {
                    // 多返回值的处理逻辑
                    let ss = self.builder.create_stack_slot((results.len() * 8) as u32);
                    let res_ptr = self.builder.ins().stack_addr(ss, 0);
                    args.push(res_ptr);
                    self.builder.ins().call(func_id, &args);

                    for i in 0..results.len() {
                        let ty = self.val_type_to_veloc(results[i]);
                        let val = self.builder.ins().stack_load(ty, ss, (i * 8) as u32);
                        self.stack.push(val);
                    }
                }
            }
            Operator::CallIndirect {
                type_index,
                table_index,
                ..
            } => {
                let sig = &self.metadata.signatures[type_index as usize];

                let index = self.pop();
                let mut args = Vec::new();
                for _ in sig.params.iter() {
                    args.push(self.pop());
                }
                args.reverse();

                let vmctx = self.vmctx.expect("vmctx not set");

                // 1. Load table base and length from VMContext (优化：通过 cache)
                let table_base = self.get_table_base(table_index);
                let def_ptr = self.builder.ins().load(
                    VelocType::Ptr,
                    vmctx,
                    self.offsets.table_offset(table_index),
                );
                let table_len = self.builder.ins().load(
                    VelocType::I64,
                    def_ptr,
                    VMTable::current_elements_offset(),
                );

                // 2. Calculate index and bounds check
                let index_i64 = self.builder.ins().extend_u(index, VelocType::I64);
                let is_lt = self.builder.ins().icmp(IntCC::LtU, index_i64, table_len);

                let trap_table_block = self.builder.create_block();
                let check_null_block = self.builder.create_block();

                self.builder
                    .ins()
                    .br(is_lt, check_null_block, &[], trap_table_block, &[]);
                self.builder.seal_block(trap_table_block);
                self.builder.seal_block(check_null_block);

                // Trap: Table Out of Bounds (code 1)
                self.builder.switch_to_block(trap_table_block);
                self.trap(TrapCode::TableOutOfBounds);

                self.builder.switch_to_block(check_null_block);
                self.terminated = false;

                // 3. Calculate offset = index * 8
                let stride = self.builder.ins().iconst(VelocType::I64, 8);
                let offset = self.builder.ins().imul(index_i64, stride);
                let entry_ptr_addr = self.builder.ins().gep(table_base, offset);
                let entry_ptr = self.builder.ins().load(VelocType::Ptr, entry_ptr_addr, 0);

                // 4. Check if null
                let zero = self.builder.ins().iconst(VelocType::I64, 0);
                let zero_ptr = self.builder.ins().int_to_ptr(zero);
                let is_not_null = self.builder.ins().icmp(IntCC::Ne, entry_ptr, zero_ptr);

                let trap_null_block = self.builder.create_block();
                let actual_call_block = self.builder.create_block();

                self.builder
                    .ins()
                    .br(is_not_null, actual_call_block, &[], trap_null_block, &[]);
                self.builder.seal_block(trap_null_block);
                self.builder.seal_block(actual_call_block);

                // Trap: Indirect Call Null (code 2)
                self.builder.switch_to_block(trap_null_block);
                self.trap(TrapCode::IndirectCallNull);

                self.builder.switch_to_block(actual_call_block);
                self.terminated = false;

                // 4.5 Check signature
                let actual_sig_id = self.builder.ins().load(
                    VelocType::I32,
                    entry_ptr,
                    VMFuncRef::type_index_offset(),
                );
                let expected_sig_id = self
                    .builder
                    .ins()
                    .iconst(VelocType::I32, (sig.hash_u64() as u32) as i64);
                let sig_matches =
                    self.builder
                        .ins()
                        .icmp(IntCC::Eq, actual_sig_id, expected_sig_id);

                let trap_sig_block = self.builder.create_block();
                let sig_ok_block = self.builder.create_block();

                self.builder
                    .ins()
                    .br(sig_matches, sig_ok_block, &[], trap_sig_block, &[]);
                self.builder.seal_block(trap_sig_block);
                self.builder.seal_block(sig_ok_block);

                // Trap: Indirect Call Bad Sig (code 6)
                self.builder.switch_to_block(trap_sig_block);
                self.trap(TrapCode::IndirectCallBadSig);

                self.builder.switch_to_block(sig_ok_block);
                self.terminated = false;

                // Load target vmctx and func_ptr from VMFuncRef
                let func_ptr = self.builder.ins().load(
                    VelocType::Ptr,
                    entry_ptr,
                    VMFuncRef::func_ptr_offset(),
                );
                let target_vmctx =
                    self.builder
                        .ins()
                        .load(VelocType::Ptr, entry_ptr, VMFuncRef::vmctx_offset());

                // 5. Setup arguments (add target_vmctx as first arg)
                args.insert(0, target_vmctx);

                let results = &sig.results;
                let sig_id = self.metadata.ir_sig_ids[type_index as usize];
                if results.len() <= 1 {
                    let res = self.builder.ins().call_indirect(sig_id, func_ptr, &args);
                    if results.len() == 1 {
                        self.stack
                            .push(res.expect("call_indirect should have result"));
                    }
                } else {
                    let ss = self.builder.create_stack_slot((results.len() * 8) as u32);
                    let res_ptr = self.builder.ins().stack_addr(ss, 0);
                    args.push(res_ptr);
                    self.builder.ins().call_indirect(sig_id, func_ptr, &args);

                    for i in 0..results.len() {
                        let ty = self.val_type_to_veloc(results[i]);
                        let val = self.builder.ins().stack_load(ty, ss, (i * 8) as u32);
                        self.stack.push(val);
                    }
                }
            }
            Operator::BrTable { targets } => {
                let index = self.pop();

                let (default_target, default_params_len) = {
                    let default_depth = targets.default();
                    let frame_idx = self.control_stack.len() - 1 - default_depth as usize;
                    let frame = &self.control_stack[frame_idx];
                    if frame.is_loop {
                        (frame.label, frame.num_params)
                    } else {
                        (frame.end_label.unwrap(), frame.num_results)
                    }
                };

                let mut default_args = Vec::new();
                if default_params_len > 0 {
                    let mut tmp = Vec::new();
                    for _ in 0..default_params_len {
                        tmp.push(self.pop());
                    }
                    for &v in tmp.iter().rev() {
                        default_args.push(v);
                        self.stack.push(v);
                    }
                }
                let default_call = self.builder.make_block_call(default_target, &default_args);

                let mut table = Vec::new();
                for t in targets.targets() {
                    let depth = t?;
                    let (target, params_len) = {
                        let frame_idx = self.control_stack.len() - 1 - depth as usize;
                        let frame = &self.control_stack[frame_idx];
                        if frame.is_loop {
                            (frame.label, frame.num_params)
                        } else {
                            (frame.end_label.unwrap(), frame.num_results)
                        }
                    };
                    let mut args = Vec::new();
                    if params_len > 0 {
                        let mut tmp = Vec::new();
                        for _ in 0..params_len {
                            tmp.push(self.pop());
                        }
                        for &v in tmp.iter().rev() {
                            args.push(v);
                            self.stack.push(v);
                        }
                    }
                    table.push(self.builder.make_block_call(target, &args));
                }

                self.builder.ins().br_table(index, default_call, &table);
                self.terminated = true;
            }
            Operator::Return => {
                if !self.terminated {
                    if self.results.is_empty() {
                        self.builder.ins().ret(None);
                    } else if self.results.len() == 1 {
                        let val = self.pop();
                        self.builder.ins().ret(Some(val));
                    } else {
                        if let Some(ptr) = self.results_ptr {
                            let mut vals = Vec::new();
                            for _ in 0..self.results.len() {
                                vals.push(self.pop());
                            }
                            for (i, val) in vals.into_iter().rev().enumerate() {
                                let _ty = self.results[i];
                                self.builder.ins().store(val, ptr, (i * 8) as u32);
                            }
                        }
                        self.builder.ins().ret(None);
                    }
                    self.terminated = true;
                }
            }
            Operator::If { blockty } => {
                let cond_i32 = self.pop();
                let (params_ty, results_ty) = self.block_params_results(blockty);

                let cond_ty = self.builder.value_type(cond_i32);
                let zero = self.builder.ins().iconst(cond_ty, 0);
                let cond = self.builder.ins().icmp(IntCC::Ne, cond_i32, zero);

                let then_block = self.builder.create_block();
                let else_block = self.builder.create_block();
                let end_block = self.builder.create_block();

                for &ty in &params_ty {
                    self.builder.add_block_param(then_block, ty);
                    self.builder.add_block_param(else_block, ty);
                }
                for &ty in &results_ty {
                    self.builder.add_block_param(end_block, ty);
                }

                let mut args = Vec::new();
                for _ in 0..params_ty.len() {
                    args.push(self.pop());
                }
                args.reverse();

                self.builder
                    .ins()
                    .br(cond, then_block, &args, else_block, &args);

                self.builder.seal_block(then_block);
                self.builder.seal_block(else_block);

                self.builder.switch_to_block(then_block);
                if self.terminated {
                    self.builder.ins().unreachable();
                }

                // 将 then_block 的参数重新推入 stack
                for i in 0..params_ty.len() {
                    let val = self.builder.block_params(then_block)[i];
                    self.stack.push(val);
                }

                self.control_stack.push(ControlFrame {
                    label: then_block,
                    end_label: Some(end_block),
                    else_label: Some(else_block),
                    is_loop: false,
                    stack_size: self.stack.len() - params_ty.len(),
                    num_params: params_ty.len(),
                    num_results: results_ty.len(),
                });
            }
            Operator::Else => {
                let (end_label, stack_size, else_label, num_params, num_results) = {
                    let frame = self.control_stack.last_mut().expect("no frame for else");
                    (
                        frame.end_label.expect("no end label"),
                        frame.stack_size,
                        frame
                            .else_label
                            .take()
                            .expect("else already handled or not an If"),
                        frame.num_params,
                        frame.num_results,
                    )
                };

                if !self.terminated {
                    let mut args = Vec::new();
                    for _ in 0..num_results {
                        args.push(self.pop());
                    }
                    args.reverse();
                    self.builder.ins().jump(end_label, &args);
                }

                self.builder.switch_to_block(else_label);
                self.terminated = false;
                self.stack.truncate(stack_size);

                // 将 else_block 的参数重新推入 stack
                for i in 0..num_params {
                    let val = self.builder.block_params(else_label)[i];
                    self.stack.push(val);
                }
            }
            Operator::End => {
                let frame = self.control_stack.pop().expect("no frame for end");
                let end_target = frame.end_label.expect("no end label");

                if frame.is_loop {
                    // 循环体结束，可能有回边，现在可以密封 Header 了
                    self.builder.seal_block(frame.label);
                }

                if let Some(else_label) = frame.else_label {
                    // This was an If without an Else
                    if !self.terminated {
                        let mut args = Vec::new();
                        for _ in 0..frame.num_results {
                            args.push(self.pop());
                        }
                        args.reverse();
                        self.builder.ins().jump(end_target, &args);
                    }
                    self.builder.switch_to_block(else_label);
                    self.terminated = false;

                    // If there's no else branch, we must pass the parameters through to the end block
                    // WebAssembly validation ensures that in this case, the results match the parameters.
                    let mut args = Vec::new();
                    for i in 0..frame.num_params {
                        args.push(self.builder.block_params(else_label)[i]);
                    }
                    self.builder.ins().jump(end_target, &args);
                } else {
                    if !self.terminated {
                        let mut args = Vec::new();
                        for _ in 0..frame.num_results {
                            args.push(self.pop());
                        }
                        args.reverse();
                        self.builder.ins().jump(end_target, &args);
                    }
                }

                self.builder.switch_to_block(end_target);
                self.builder.seal_block(end_target);
                self.terminated = false;

                if self.control_stack.is_empty() {
                    // This was the function's implicit block
                    if !self.terminated {
                        if frame.num_results == 0 {
                            self.builder.ins().ret(None);
                        } else if frame.num_results == 1 {
                            let val = self.builder.block_params(end_target)[0];
                            self.builder.ins().ret(Some(val));
                        } else {
                            if let Some(ptr) = self.results_ptr {
                                for i in 0..frame.num_results {
                                    let val = self.builder.block_params(end_target)[i];
                                    self.builder.ins().store(val, ptr, (i * 8) as u32);
                                }
                            }
                            self.builder.ins().ret(None);
                        }
                        self.terminated = true;
                    }
                } else {
                    self.stack.truncate(frame.stack_size);
                    for i in 0..frame.num_results {
                        let val = self.builder.block_params(end_target)[i];
                        self.stack.push(val);
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn pop(&mut self) -> Value {
        self.stack.pop().unwrap_or_else(|| {
            // 发生下溢通常是因为某些指令未实现，导致由于未压入预期的值。
            // 为了保证翻译能够继续，我们压入一个默认值。
            self.builder.ins().iconst(VelocType::I64, 0)
        })
    }

    fn val_type_to_veloc(&self, ty: wasmparser::ValType) -> VelocType {
        match ty {
            wasmparser::ValType::I32 => VelocType::I32,
            wasmparser::ValType::I64 => VelocType::I64,
            wasmparser::ValType::F32 => VelocType::F32,
            wasmparser::ValType::F64 => VelocType::F64,
            wasmparser::ValType::Ref(_) => VelocType::Ptr,
            _ => VelocType::I64,
        }
    }

    fn block_params_results(&self, ty: wasmparser::BlockType) -> (Vec<VelocType>, Vec<VelocType>) {
        match ty {
            wasmparser::BlockType::Empty => (vec![], vec![]),
            wasmparser::BlockType::Type(t) => (vec![], vec![self.val_type_to_veloc(t)]),
            wasmparser::BlockType::FuncType(idx) => {
                let sig = &self.metadata.signatures[idx as usize];
                let params = sig
                    .params
                    .iter()
                    .map(|&t| self.val_type_to_veloc(t))
                    .collect();
                let results = sig
                    .results
                    .iter()
                    .map(|&t| self.val_type_to_veloc(t))
                    .collect();
                (params, results)
            }
        }
    }

    fn bin<F>(&mut self, f: F)
    where
        F: FnOnce(&mut InstBuilder, Value, Value) -> Value,
    {
        let r = self.pop();
        let l = self.pop();
        let res = f(&mut self.builder.ins(), l, r);
        self.stack.push(res);
    }

    fn bin_cmp<F>(&mut self, f: F)
    where
        F: FnOnce(&mut InstBuilder, Value, Value) -> Value,
    {
        let r = self.pop();
        let l = self.pop();
        let bool_res = f(&mut self.builder.ins(), l, r);
        let res = self.builder.ins().extend_u(bool_res, VelocType::I32);
        self.stack.push(res);
    }

    fn un<F>(&mut self, f: F)
    where
        F: FnOnce(&mut InstBuilder, Value) -> Value,
    {
        let v = self.pop();
        let res = f(&mut self.builder.ins(), v);
        self.stack.push(res);
    }

    fn un_cmp<F>(&mut self, f: F)
    where
        F: FnOnce(&mut InstBuilder, Value) -> Value,
    {
        let v = self.pop();
        let bool_res = f(&mut self.builder.ins(), v);
        let res = self.builder.ins().extend_u(bool_res, VelocType::I32);
        self.stack.push(res);
    }

    fn trap(&mut self, code: TrapCode) {
        let entry = self.builder.entry_block().unwrap();
        let vmctx = self.builder.block_params(entry)[0];
        let code_val = self
            .builder
            .ins()
            .iconst(VelocType::I32, code as i32 as i64);
        self.builder
            .ins()
            .call(self.runtime.trap_handler, &[vmctx, code_val]);
        self.builder.ins().unreachable();
        self.terminated = true;
    }

    fn trap_if(&mut self, cond: Value, code: TrapCode) {
        let trap_block = self.builder.create_block();
        let next_block = self.builder.create_block();
        self.builder
            .ins()
            .br(cond, trap_block, &[], next_block, &[]);

        self.builder.seal_block(trap_block);
        self.builder.seal_block(next_block);

        self.builder.switch_to_block(trap_block);
        self.trap(code);

        self.builder.switch_to_block(next_block);
        self.terminated = false;
    }

    fn translate_fmin_fmax(&mut self, is_64: bool, is_min: bool) {
        let ty = if is_64 {
            VelocType::F64
        } else {
            VelocType::F32
        };
        let r = self.pop();
        let l = self.pop();

        let zero = self.builder.ins().fconst(ty, 0);
        let l_is_zero = self.builder.ins().fcmp(FloatCC::Eq, l, zero);
        let r_is_zero = self.builder.ins().fcmp(FloatCC::Eq, r, zero);
        let both_zero = self.builder.ins().and(l_is_zero, r_is_zero);

        let res_var = self.new_var(ty);

        self.builder.if_else(
            both_zero,
            |b| {
                let ity = if is_64 {
                    VelocType::I64
                } else {
                    VelocType::I32
                };
                let l_bits = b.ins().reinterpret(l, ity);
                let r_bits = b.ins().reinterpret(r, ity);
                let res_bits = if is_min {
                    b.ins().or(l_bits, r_bits)
                } else {
                    b.ins().and(l_bits, r_bits)
                };
                let res = b.ins().reinterpret(res_bits, ty);
                b.def_var(res_var, res);
            },
            |b| {
                let res = if is_min {
                    b.ins().min(l, r)
                } else {
                    b.ins().max(l, r)
                };
                b.def_var(res_var, res);
            },
        );

        self.terminated = false;
        let final_res = self.builder.use_var(res_var);
        self.stack.push(final_res);
    }

    fn translate_div_s(&mut self, is_64: bool) {
        let ty = if is_64 {
            VelocType::I64
        } else {
            VelocType::I32
        };
        let r = self.pop();
        let l = self.pop();

        let zero = self.builder.ins().iconst(ty, 0);
        let is_zero = self.builder.ins().icmp(IntCC::Eq, r, zero);
        self.trap_if(is_zero, TrapCode::IntegerDivideByZero);

        let neg_one = self.builder.ins().iconst(ty, -1);
        let min_val = if is_64 { i64::MIN } else { i32::MIN as i64 };
        let min = self.builder.ins().iconst(ty, min_val);

        let is_min = self.builder.ins().icmp(IntCC::Eq, l, min);
        let is_neg_one = self.builder.ins().icmp(IntCC::Eq, r, neg_one);
        let is_overflow = self.builder.ins().and(is_min, is_neg_one);
        self.trap_if(is_overflow, TrapCode::IntegerOverflow);

        let res = self.builder.ins().idiv(l, r);
        self.stack.push(res);
    }

    fn translate_div_u(&mut self, is_64: bool) {
        let ty = if is_64 {
            VelocType::I64
        } else {
            VelocType::I32
        };
        let r = self.pop();
        let l = self.pop();

        let zero = self.builder.ins().iconst(ty, 0);
        let is_zero = self.builder.ins().icmp(IntCC::Eq, r, zero);
        self.trap_if(is_zero, TrapCode::IntegerDivideByZero);

        let res = self.builder.ins().udiv(l, r);
        self.stack.push(res);
    }

    fn translate_rem_s(&mut self, is_64: bool) {
        let ty = if is_64 {
            VelocType::I64
        } else {
            VelocType::I32
        };
        let r = self.pop();
        let l = self.pop();

        let zero = self.builder.ins().iconst(ty, 0);
        let is_zero = self.builder.ins().icmp(IntCC::Eq, r, zero);
        self.trap_if(is_zero, TrapCode::IntegerDivideByZero);

        let neg_one = self.builder.ins().iconst(ty, -1);
        let min_val = if is_64 { i64::MIN } else { i32::MIN as i64 };
        let min = self.builder.ins().iconst(ty, min_val);

        let is_min = self.builder.ins().icmp(IntCC::Eq, l, min);
        let is_neg_one = self.builder.ins().icmp(IntCC::Eq, r, neg_one);
        let is_overflow = self.builder.ins().and(is_min, is_neg_one);

        let res_var = self.new_var(ty);

        self.builder.if_else(
            is_overflow,
            |b| {
                let zero_res = b.ins().iconst(ty, 0);
                b.def_var(res_var, zero_res);
            },
            |b| {
                let rem_res = b.ins().irem(l, r);
                b.def_var(res_var, rem_res);
            },
        );

        let final_res = self.builder.use_var(res_var);
        self.stack.push(final_res);
    }

    fn translate_rem_u(&mut self, is_64: bool) {
        let ty = if is_64 {
            VelocType::I64
        } else {
            VelocType::I32
        };
        let r = self.pop();
        let l = self.pop();

        let zero = self.builder.ins().iconst(ty, 0);
        let is_zero = self.builder.ins().icmp(IntCC::Eq, r, zero);
        self.trap_if(is_zero, TrapCode::IntegerDivideByZero);

        let res = self.builder.ins().urem(l, r);
        self.stack.push(res);
    }

    fn translate_load(&mut self, ty: VelocType, memarg: wasmparser::MemArg) {
        let addr = self.pop();
        let mem_idx = memarg.memory;
        self.memory_bounds_check(mem_idx, addr, memarg.offset, ty.size_bytes());
        let mem_base = self.get_memory_base(mem_idx);

        let addr_i64 = self.builder.ins().extend_u(addr, VelocType::I64);
        let actual_ptr = self.builder.ins().gep(mem_base, addr_i64);

        let res = self
            .builder
            .ins()
            .load(ty, actual_ptr, memarg.offset as u32);
        self.stack.push(res);
    }

    fn translate_store(&mut self, ty: VelocType, memarg: wasmparser::MemArg) {
        let val = self.pop();
        let addr = self.pop();
        let mem_idx = memarg.memory;
        self.memory_bounds_check(mem_idx, addr, memarg.offset, ty.size_bytes());
        let mem_base = self.get_memory_base(mem_idx);

        let addr_i64 = self.builder.ins().extend_u(addr, VelocType::I64);
        let actual_ptr = self.builder.ins().gep(mem_base, addr_i64);

        self.builder
            .ins()
            .store(val, actual_ptr, memarg.offset as u32);
    }

    fn get_memory_base(&mut self, index: u32) -> Value {
        if let Some(&val) = self.cached_memories.get(&index) {
            return val;
        }
        let vmctx = self.vmctx.expect("vmctx not set");
        let def_ptr =
            self.builder
                .ins()
                .load(VelocType::Ptr, vmctx, self.offsets.memory_offset(index));
        let val = self
            .builder
            .ins()
            .load(VelocType::Ptr, def_ptr, VMMemory::base_offset());
        self.cached_memories.insert(index, val);
        val
    }

    fn memory_bounds_check(&mut self, index: u32, addr: Value, offset: u64, access_size: u32) {
        let vmctx = self.vmctx.expect("vmctx not set");
        let def_ptr =
            self.builder
                .ins()
                .load(VelocType::Ptr, vmctx, self.offsets.memory_offset(index));
        let length =
            self.builder
                .ins()
                .load(VelocType::I64, def_ptr, VMMemory::current_length_offset());

        // Calculate addr + offset + access_size
        let addr_i64 = self.builder.ins().extend_u(addr, VelocType::I64);
        let total_offset = self
            .builder
            .ins()
            .iconst(VelocType::I64, (offset + access_size as u64) as i64);
        let effective_end = self.builder.ins().iadd(addr_i64, total_offset);

        // if effective_end > length then trap
        let is_oob = self.builder.ins().icmp(IntCC::GtU, effective_end, length);
        self.trap_if(is_oob, TrapCode::MemoryOutOfBounds);
    }

    fn get_table_base(&mut self, index: u32) -> Value {
        if let Some(&val) = self.cached_tables.get(&index) {
            return val;
        }
        let vmctx = self.vmctx.expect("vmctx not set");
        let def_ptr =
            self.builder
                .ins()
                .load(VelocType::Ptr, vmctx, self.offsets.table_offset(index));
        let val = self
            .builder
            .ins()
            .load(VelocType::Ptr, def_ptr, VMTable::base_offset());
        self.cached_tables.insert(index, val);
        val
    }

    fn table_bounds_check(&mut self, table_index: u32, index: Value) {
        let vmctx = self.vmctx.expect("vmctx not set");
        let def_ptr = self.builder.ins().load(
            VelocType::Ptr,
            vmctx,
            self.offsets.table_offset(table_index),
        );
        let table_len =
            self.builder
                .ins()
                .load(VelocType::I64, def_ptr, VMTable::current_elements_offset());
        let index_i64 = self.builder.ins().extend_u(index, VelocType::I64);
        let is_oob = self.builder.ins().icmp(IntCC::GeU, index_i64, table_len);
        self.trap_if(is_oob, TrapCode::TableOutOfBounds);
    }

    fn get_global_ptr(&mut self, index: u32) -> Value {
        if let Some(&val) = self.cached_globals.get(&index) {
            return val;
        }
        let vmctx = self.vmctx.expect("vmctx not set");
        let val = self
            .builder
            .ins()
            .load(VelocType::Ptr, vmctx, self.offsets.global_offset(index));
        self.cached_globals.insert(index, val);
        val
    }
}
