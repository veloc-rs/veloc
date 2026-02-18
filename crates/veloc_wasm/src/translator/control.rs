use super::{ControlFrame, WasmTranslator};
use crate::vm::{TrapCode, VMFuncRef};
use alloc::vec::Vec;
use veloc::ir::{IntCC, MemFlags, Type as VelocType};
use wasmparser::{BinaryReaderError, Operator};

impl<'a> WasmTranslator<'a> {
    pub(super) fn translate_control(&mut self, op: Operator) -> Result<(), BinaryReaderError> {
        match op {
            Operator::Block { blockty } => {
                let (params_ty, results_ty) = self.block_params_results(blockty);
                let end_block = self.builder.create_block();
                for &ty in &results_ty {
                    self.builder.add_block_param(end_block, ty);
                }
                let reachable_at_start = !self.terminated;
                self.control_stack.push(ControlFrame {
                    label: end_block,
                    end_label: Some(end_block),
                    else_label: None,
                    is_loop: false,
                    stack_size: self.stack.len().saturating_sub(params_ty.len()),
                    num_params: params_ty.len(),
                    num_results: results_ty.len(),
                    reachable_at_start,
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
                if !self.terminated {
                    for _ in 0..params_ty.len() {
                        args.push(self.pop());
                    }
                    args.reverse();
                } else {
                    for &ty in &params_ty {
                        args.push(self.builder.ins().iconst(ty, 0));
                    }
                }
                self.builder.ins().jump(header_block, &args);
                let reachable_at_start = !self.terminated;
                self.builder.switch_to_block(header_block);
                if self.terminated {
                    self.builder.ins().unreachable();
                }
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
                    reachable_at_start,
                });
            }
            Operator::If { blockty } => {
                let cond_i32 = if !self.terminated {
                    self.pop()
                } else {
                    self.builder.ins().i32const(0)
                };
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
                let reachable_at_start = !self.terminated;
                if !self.terminated {
                    for _ in 0..params_ty.len() {
                        args.push(self.pop());
                    }
                    args.reverse();
                } else {
                    for &ty in &params_ty {
                        args.push(self.builder.ins().iconst(ty, 0));
                    }
                }
                self.builder
                    .ins()
                    .br(cond, then_block, &args, else_block, &args);
                self.builder.seal_block(then_block);
                self.builder.seal_block(else_block);
                self.builder.switch_to_block(then_block);
                if self.terminated {
                    self.builder.ins().unreachable();
                }
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
                    reachable_at_start,
                });
            }
            Operator::Else => {
                let (
                    end_label,
                    stack_size,
                    else_label,
                    num_params,
                    num_results,
                    reachable_at_start,
                ) = {
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
                        frame.reachable_at_start,
                    )
                };
                if !self.terminated {
                    let mut args = Vec::new();
                    for _ in 0..num_results {
                        args.push(self.pop());
                    }
                    args.reverse();
                    self.builder.ins().jump(end_label, &args);
                } else if !self.builder.is_current_block_terminated() {
                    self.builder.ins().unreachable();
                }
                self.builder.switch_to_block(else_label);
                self.terminated = !reachable_at_start;
                self.stack.truncate(stack_size);
                for i in 0..num_params {
                    let val = self.builder.block_params(else_label)[i];
                    self.stack.push(val);
                }
            }
            Operator::End => {
                let frame = self.control_stack.pop().expect("no frame for end");
                let end_target = frame.end_label.expect("no end label");
                if frame.is_loop {
                    self.builder.seal_block(frame.label);
                }
                if let Some(else_label) = frame.else_label {
                    if !self.terminated {
                        let mut args = Vec::new();
                        for _ in 0..frame.num_results {
                            args.push(self.pop());
                        }
                        args.reverse();
                        self.builder.ins().jump(end_target, &args);
                    } else if !self.builder.is_current_block_terminated() {
                        self.builder.ins().unreachable();
                    }
                    self.builder.switch_to_block(else_label);
                    self.terminated = !frame.reachable_at_start;
                    let mut args = Vec::new();
                    for i in 0..frame.num_params {
                        args.push(self.builder.block_params(else_label)[i]);
                    }
                    self.builder.ins().jump(end_target, &args);
                } else if !self.terminated {
                    let mut args = Vec::new();
                    for _ in 0..frame.num_results {
                        args.push(self.pop());
                    }
                    args.reverse();
                    self.builder.ins().jump(end_target, &args);
                } else if !self.builder.is_current_block_terminated() {
                    self.builder.ins().unreachable();
                }
                self.builder.switch_to_block(end_target);
                self.builder.seal_block(end_target);
                self.terminated = !frame.reachable_at_start;
                if self.control_stack.is_empty() {
                    if !self.terminated {
                        let mut vals = Vec::with_capacity(frame.num_results);
                        for i in 0..frame.num_results {
                            vals.push(self.builder.block_params(end_target)[i]);
                        }
                        self.builder.ins().ret(&vals);
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
                // BrIf condition must be i32 in WebAssembly
                let zero = self.builder.ins().i32const(0);
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
                    let func_ptr = self.builder.ins().load(
                        VelocType::Ptr,
                        vmptr,
                        entry_offset + VMFuncRef::func_ptr_offset(),
                        MemFlags::new().with_alignment(16),
                    );
                    let target_vmctx = self.builder.ins().load(
                        VelocType::Ptr,
                        vmptr,
                        entry_offset + VMFuncRef::vmctx_offset(),
                        MemFlags::new().with_alignment(8),
                    );
                    args.insert(0, target_vmctx);
                    let func_id = self.metadata.functions[function_index as usize].func_id;
                    let sig_id = self.builder.func_signature(func_id);
                    let call_inst = self.builder.ins().call_indirect(sig_id, func_ptr, &args);
                    for i in 0..results.len() {
                        let res_val = self.builder.func().dfg.inst_results(call_inst)[i];
                        self.stack.push(res_val);
                    }
                } else {
                    args.insert(0, self.vmctx.expect("vmctx not set"));
                    let func_id = self.metadata.functions[function_index as usize].func_id;
                    let call_inst = self.builder.ins().call(func_id, &args);
                    for i in 0..results.len() {
                        let res_val = self.builder.func().dfg.inst_results(call_inst)[i];
                        self.stack.push(res_val);
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
                let table_base = self.get_table_base(table_index);
                let (_, len_var) = self.table_vars[table_index as usize];
                let table_len = self.builder.use_var(len_var);
                let index_i64 = self.builder.ins().extend_u(index, VelocType::I64);
                let is_lt = self.builder.ins().icmp(IntCC::LtU, index_i64, table_len);
                let trap_table_block = self.builder.create_block();
                let check_null_block = self.builder.create_block();
                self.builder
                    .ins()
                    .br(is_lt, check_null_block, &[], trap_table_block, &[]);
                self.builder.seal_block(trap_table_block);
                self.builder.seal_block(check_null_block);
                self.builder.switch_to_block(trap_table_block);
                self.trap(TrapCode::TableOutOfBounds);
                self.builder.switch_to_block(check_null_block);
                self.terminated = false;
                let entry_ptr_addr = self.builder.ins().ptr_index(table_base, index_i64, 8, 0);
                let entry_ptr =
                    self.builder
                        .ins()
                        .load(VelocType::Ptr, entry_ptr_addr, 0, MemFlags::default());
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
                self.builder.switch_to_block(trap_null_block);
                self.trap(TrapCode::IndirectCallNull);
                self.builder.switch_to_block(actual_call_block);
                self.terminated = false;
                let actual_sig_id = self.builder.ins().load(
                    VelocType::I32,
                    entry_ptr,
                    VMFuncRef::type_index_offset(),
                    MemFlags::new().with_alignment(16),
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
                self.builder.switch_to_block(trap_sig_block);
                self.trap(TrapCode::IndirectCallBadSig);
                self.builder.switch_to_block(sig_ok_block);
                self.terminated = false;
                let func_ptr = self.builder.ins().load(
                    VelocType::Ptr,
                    entry_ptr,
                    VMFuncRef::func_ptr_offset(),
                    MemFlags::new().with_alignment(16),
                );
                let target_vmctx = self.builder.ins().load(
                    VelocType::Ptr,
                    entry_ptr,
                    VMFuncRef::vmctx_offset(),
                    MemFlags::new().with_alignment(8),
                );
                args.insert(0, target_vmctx);
                let results = &sig.results;
                let sig_id = self.ir_sig_ids[type_index as usize];
                let call_inst = self.builder.ins().call_indirect(sig_id, func_ptr, &args);
                for i in 0..results.len() {
                    let res_val = self.builder.func().dfg.inst_results(call_inst)[i];
                    self.stack.push(res_val);
                }
            }
            Operator::CallRef { type_index } => {
                let sig = &self.metadata.signatures[type_index as usize];
                let func_ref = self.pop();
                let mut args = Vec::new();
                for _ in sig.params.iter() {
                    args.push(self.pop());
                }
                args.reverse();
                let func_ptr = self.builder.ins().load(
                    VelocType::Ptr,
                    func_ref,
                    VMFuncRef::func_ptr_offset(),
                    MemFlags::new().with_alignment(16),
                );
                let target_vmctx = self.builder.ins().load(
                    VelocType::Ptr,
                    func_ref,
                    VMFuncRef::vmctx_offset(),
                    MemFlags::new().with_alignment(8),
                );
                args.insert(0, target_vmctx);
                let results = &sig.results;
                let sig_id = self.ir_sig_ids[type_index as usize];
                let call_inst = self.builder.ins().call_indirect(sig_id, func_ptr, &args);
                for i in 0..results.len() {
                    let res_val = self.builder.func().dfg.inst_results(call_inst)[i];
                    self.stack.push(res_val);
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
                    let mut vals = Vec::with_capacity(self.results.len());
                    for _ in 0..self.results.len() {
                        vals.push(self.pop());
                    }
                    vals.reverse();
                    self.builder.ins().ret(&vals);
                    self.terminated = true;
                }
            }
            _ => { /* Handled by router */ }
        }
        Ok(())
    }
}
