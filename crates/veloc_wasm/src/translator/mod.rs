use crate::module::{RuntimeFunctions, WasmMetadata};
use crate::vm::VMOffsets;
use alloc::vec::Vec;
use veloc::ir::{Block, FunctionBuilder, MemFlags, SigId, Type as VelocType, Value, Variable};
use wasmparser::{BinaryReaderError, Operator, ValType};

mod control;
mod memory;
mod numeric;
mod table;
mod variable;

pub struct WasmTranslator<'a> {
    builder: &'a mut FunctionBuilder<'a>,
    stack: Vec<Value>,
    locals: Vec<(Variable, VelocType)>,
    next_var_idx: u32,
    control_stack: Vec<ControlFrame>,
    vmctx: Option<Value>,
    results_ptr: Option<Value>,
    results: Vec<VelocType>,
    terminated: bool,
    use_names: bool,

    pub metadata: &'a WasmMetadata,
    pub ir_sig_ids: &'a [SigId],
    pub offsets: VMOffsets,
    pub runtime: RuntimeFunctions,

    // Optimized: Use Vec instead of HashMap for dense indices
    memory_vars: Vec<(Variable, Variable)>,
    table_vars: Vec<(Variable, Variable)>,
    global_ptr_vars: Vec<Variable>,
}

struct ControlFrame {
    label: Block,
    end_label: Option<Block>, // Wasm Block/If 的 end 目标，Loop 的退出目标
    else_label: Option<Block>,
    is_loop: bool,
    stack_size: usize,
    num_params: usize,
    num_results: usize,
    reachable_at_start: bool,
}

impl<'a> WasmTranslator<'a> {
    pub fn new(
        builder: &'a mut FunctionBuilder<'a>,
        results: Vec<VelocType>,
        metadata: &'a WasmMetadata,
        ir_sig_ids: &'a [SigId],
        offsets: VMOffsets,
        runtime: RuntimeFunctions,
        use_names: bool,
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
            ir_sig_ids,
            offsets,
            runtime,
            next_var_idx: 0,
            use_names,
            memory_vars: Vec::new(),
            table_vars: Vec::new(),
            global_ptr_vars: Vec::new(),
        }
    }

    fn new_var(&mut self, ty: VelocType) -> Variable {
        let var = Variable(self.next_var_idx);
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
        let entry = self.builder.init_entry_block();
        let end_block = self.builder.create_block();

        self.control_stack.push(ControlFrame {
            label: entry,
            end_label: Some(end_block),
            else_label: None,
            is_loop: false,
            stack_size: 0,
            num_params: 0,
            num_results: self.results.len(),
            reachable_at_start: true,
        });

        for &ty in &self.results {
            self.builder.add_block_param(end_block, ty);
        }

        let params = self.builder.func_params().to_vec();
        let mut params_iter = params.into_iter();

        let vmctx = params_iter.next().expect("Missing vmctx parameter");
        self.vmctx = Some(vmctx);
        if self.use_names {
            self.builder.set_value_name(vmctx, "vmctx");
        }

        for (i, &ty) in param_types.iter().enumerate() {
            let var = self.new_var(ty);
            self.locals.push((var, ty));
            let val = params_iter.next().expect("Missing parameter");
            self.builder.def_var(var, val);
            if self.use_names {
                self.builder.set_value_name(val, &format!("param{}", i));
            }
        }

        if self.results.len() > 1 {
            let ptr = params_iter.next().expect("Missing results_ptr parameter");
            self.results_ptr = Some(ptr);
            if self.use_names {
                self.builder.set_value_name(ptr, "results_ptr");
            }
        }

        let mut locals_reader = code.get_locals_reader()?;
        let num_params = param_types.len();
        let mut local_idx = num_params;
        for _ in 0..locals_reader.get_count() {
            let (count, ty) = locals_reader.read()?;
            let ty = match ty {
                ValType::I32 => VelocType::I32,
                ValType::I64 => VelocType::I64,
                ValType::F32 => VelocType::F32,
                ValType::F64 => VelocType::F64,
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
                if self.use_names {
                    self.builder
                        .set_value_name(zero, &format!("local{}", local_idx));
                }
                local_idx += 1;
            }
        }

        // Initialize memory and table variables
        for i in 0..self.metadata.memories.len() as u32 {
            let base_var = self.new_var(VelocType::Ptr);
            let len_var = self.new_var(VelocType::I64);
            self.memory_vars.push((base_var, len_var));
            self.reload_memory(i);
        }
        for i in 0..self.metadata.tables.len() as u32 {
            let base_var = self.new_var(VelocType::Ptr);
            let len_var = self.new_var(VelocType::I64);
            self.table_vars.push((base_var, len_var));
            self.reload_table(i);
        }
        for i in 0..self.metadata.globals.len() as u32 {
            let var = self.new_var(VelocType::Ptr);
            self.global_ptr_vars.push(var);
            let vmctx = self.vmctx.expect("vmctx not set");
            let offset = self.offsets.global_offset(i);
            let alignment = if offset % 16 == 0 { 16 } else { 8 };
            let ptr = self.builder.ins().load(
                VelocType::Ptr,
                vmctx,
                offset,
                MemFlags::new().with_alignment(alignment),
            );
            if self.use_names {
                self.builder
                    .set_value_name(ptr, &format!("global{}_ptr", i));
            }
            self.builder.def_var(var, ptr);
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
        self.builder.seal_all_blocks();
        Ok(())
    }

    pub(super) fn translate_operator(&mut self, op: Operator) -> Result<(), BinaryReaderError> {
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
                let ptr_addr = self
                    .builder
                    .ins()
                    .ptr_offset(vmctx, self.offsets.function_offset(function_index) as i32);
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
                let res = self.builder.ins().icmp(veloc::ir::IntCC::Eq, v, zero);
                let true_val = self.builder.ins().iconst(VelocType::I32, 1);
                let false_val = self.builder.ins().iconst(VelocType::I32, 0);
                let res_i32 = self.builder.ins().select(res, true_val, false_val);
                self.stack.push(res_i32);
            }
            Operator::RefAsNonNull => {
                let v = self.pop();
                let v_ty = self.builder.value_type(v);
                let zero = if v_ty == VelocType::Ptr {
                    let z = self.builder.ins().iconst(VelocType::I64, 0);
                    self.builder.ins().int_to_ptr(z)
                } else {
                    self.builder.ins().iconst(v_ty, 0)
                };
                let is_null = self.builder.ins().icmp(veloc::ir::IntCC::Eq, v, zero);
                self.trap_if(is_null, crate::vm::TrapCode::NullReference);
                self.stack.push(v);
            }
            Operator::Drop => {
                self.stack.pop();
            }
            Operator::Select | Operator::TypedSelect { .. } => {
                let cond_i32 = self.pop();
                let cond_ty = self.builder.value_type(cond_i32);
                let zero = self.builder.ins().iconst(cond_ty, 0);
                let cond = self
                    .builder
                    .ins()
                    .icmp(veloc::ir::IntCC::Ne, cond_i32, zero);
                let val2 = self.pop();
                let val1 = self.pop();
                let res = self.builder.ins().select(cond, val1, val2);
                self.stack.push(res);
            }
            Operator::Nop => {}
            Operator::Unreachable => {
                self.trap(crate::vm::TrapCode::Unreachable);
            }

            // Variable operations
            Operator::LocalGet { .. }
            | Operator::LocalSet { .. }
            | Operator::LocalTee { .. }
            | Operator::GlobalGet { .. }
            | Operator::GlobalSet { .. } => self.translate_variable(op)?,

            // Memory operations
            Operator::I32Load { .. }
            | Operator::I64Load { .. }
            | Operator::F32Load { .. }
            | Operator::F64Load { .. }
            | Operator::I32Load8S { .. }
            | Operator::I32Load8U { .. }
            | Operator::I32Load16S { .. }
            | Operator::I32Load16U { .. }
            | Operator::I64Load8S { .. }
            | Operator::I64Load8U { .. }
            | Operator::I64Load16S { .. }
            | Operator::I64Load16U { .. }
            | Operator::I64Load32S { .. }
            | Operator::I64Load32U { .. }
            | Operator::I32Store { .. }
            | Operator::I64Store { .. }
            | Operator::F32Store { .. }
            | Operator::F64Store { .. }
            | Operator::I32Store8 { .. }
            | Operator::I32Store16 { .. }
            | Operator::I64Store8 { .. }
            | Operator::I64Store16 { .. }
            | Operator::I64Store32 { .. }
            | Operator::MemorySize { .. }
            | Operator::MemoryGrow { .. }
            | Operator::MemoryInit { .. }
            | Operator::DataDrop { .. }
            | Operator::MemoryCopy { .. }
            | Operator::MemoryFill { .. } => self.translate_memory(op)?,

            // Table operations
            Operator::TableGet { .. }
            | Operator::TableSet { .. }
            | Operator::TableInit { .. }
            | Operator::TableCopy { .. }
            | Operator::TableGrow { .. }
            | Operator::TableSize { .. }
            | Operator::TableFill { .. }
            | Operator::ElemDrop { .. } => self.translate_table(op)?,

            _ => {
                if is_control_operator(&op) {
                    self.translate_control(op)?
                } else {
                    self.translate_numeric(op)?
                }
            }
        }
        Ok(())
    }

    fn pop(&mut self) -> Value {
        self.stack.pop().expect("stack underflow")
    }

    fn pop_i32(&mut self) -> Value {
        let v = self.pop();
        let ty = self.builder.value_type(v);
        if ty == VelocType::I32 {
            v
        } else {
            self.builder.ins().wrap(v, VelocType::I32)
        }
    }

    fn val_type_to_veloc(&self, ty: ValType) -> VelocType {
        match ty {
            ValType::I32 => VelocType::I32,
            ValType::I64 => VelocType::I64,
            ValType::F32 => VelocType::F32,
            ValType::F64 => VelocType::F64,
            ValType::Ref(_) => VelocType::Ptr,
            _ => VelocType::I64,
        }
    }

    fn block_params_results(
        &self,
        blockty: wasmparser::BlockType,
    ) -> (Vec<VelocType>, Vec<VelocType>) {
        match blockty {
            wasmparser::BlockType::Empty => (vec![], vec![]),
            wasmparser::BlockType::Type(ty) => (vec![], vec![self.val_type_to_veloc(ty)]),
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

    fn trap(&mut self, code: crate::vm::TrapCode) {
        let vmctx = self.vmctx.expect("vmctx not set");
        let trap_code = self.builder.ins().iconst(VelocType::I32, code as i64);
        self.builder
            .ins()
            .call(self.runtime.trap_handler, &[vmctx, trap_code]);
        self.builder.ins().unreachable();
        self.terminated = true;
    }

    fn trap_if(&mut self, cond: Value, code: crate::vm::TrapCode) {
        let next_block = self.builder.create_block();
        let trap_block = self.builder.create_block();

        self.builder
            .ins()
            .br(cond, trap_block, &[], next_block, &[]);
        self.builder.seal_block(trap_block);

        self.builder.switch_to_block(trap_block);
        self.trap(code);

        self.builder.switch_to_block(next_block);
        self.builder.seal_block(next_block);
        self.terminated = false;
    }
}

fn is_control_operator(op: &Operator) -> bool {
    match op {
        Operator::Block { .. }
        | Operator::Loop { .. }
        | Operator::If { .. }
        | Operator::Else
        | Operator::End
        | Operator::Br { .. }
        | Operator::BrIf { .. }
        | Operator::BrTable { .. }
        | Operator::Return
        | Operator::Call { .. }
        | Operator::CallIndirect { .. }
        | Operator::CallRef { .. } => true,
        _ => false,
    }
}
