use super::{ControlFrame, WasmTranslator};
use crate::vm::{TrapCode, VMFuncRef};
use alloc::vec::Vec;
use veloc::mir::{IntCC, MemFlags, Type as VelocType};
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
                    params_types: params_ty,
                    results_types: results_ty,
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
                    for &ty in params_ty.iter().rev() {
                        args.push(self.pop_typed(ty));
                    }
                    args.reverse();
                } else {
                    for &ty in &params_ty {
                        let zero = self.zero_const(ty);
                        args.push(zero);
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
                    params_types: params_ty,
                    results_types: results_ty,
                    reachable_at_start,
                });
            }
            Operator::If { blockty } => {
                let cond = if !self.terminated {
                    self.pop_cond()
                } else {
                    self.builder.ins().bconst(false)
                };
                let (params_ty, results_ty) = self.block_params_results(blockty);
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
                    for &ty in params_ty.iter().rev() {
                        args.push(self.pop_typed(ty));
                    }
                    args.reverse();
                } else {
                    for &ty in &params_ty {
                        let zero = self.zero_const(ty);
                        args.push(zero);
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
                    params_types: params_ty,
                    results_types: results_ty,
                    reachable_at_start,
                });
            }
            Operator::Else => {
                let (
                    end_label,
                    stack_size,
                    else_label,
                    num_params,
                    results_types,
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
                        frame.results_types.clone(),
                        frame.reachable_at_start,
                    )
                };
                if !self.terminated {
                    let mut args = Vec::new();
                    for &ty in results_types.iter().rev() {
                        args.push(self.pop_typed(ty));
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
                        for &ty in frame.results_types.iter().rev() {
                            args.push(self.pop_typed(ty));
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
                    // If we reached Else, it means the Params are passed to the Else block.
                    // But if Else is empty or we are at End, we need to pass the Params to End if nothing else happens?
                    // No, if we are at End of an If without Else, the Else branch should have been handled.
                    // But here else_label is still Some, meaning we haven't seen Operator::Else.
                    // So this is `if ... then ... end` (no else).
                    // In this case, the params are passed to the else_block, and then we just jump to end.
                    self.builder.ins().jump(end_target, &args);
                } else if !self.terminated {
                    let mut args = Vec::new();
                    for &ty in frame.results_types.iter().rev() {
                        args.push(self.pop_typed(ty));
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
                        self.emit_function_return(&vals);
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
                let (target, target_tys) = if frame.is_loop {
                    (frame.label, frame.params_types.clone())
                } else {
                    (frame.end_label.unwrap(), frame.results_types.clone())
                };
                let mut args = Vec::new();
                for &ty in target_tys.iter().rev() {
                    args.push(self.pop_typed(ty));
                }
                args.reverse();
                self.builder.ins().jump(target, &args);
                self.terminated = true;
            }
            Operator::BrIf { relative_depth } => {
                let cond = self.pop_cond();
                let frame_idx = self.control_stack.len() - 1 - relative_depth as usize;
                let frame = &self.control_stack[frame_idx];
                let (target, target_tys) = if frame.is_loop {
                    (frame.label, frame.params_types.clone())
                } else {
                    (frame.end_label.unwrap(), frame.results_types.clone())
                };
                let next_block = self.builder.create_block();
                let mut args = Vec::new();
                if !target_tys.is_empty() {
                    let mut tmp_values = Vec::new();
                    for &ty in target_tys.iter().rev() {
                        tmp_values.push(self.pop_typed(ty));
                    }
                    for &val in tmp_values.iter().rev() {
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
                for &ty in sig.params.iter().rev() {
                    let veloc_ty = self.val_type_to_veloc(ty);
                    args.push(self.pop_typed(veloc_ty));
                }
                args.reverse();
                let results = &sig.results;
                let multi_ret_slot = if results.len() > 1 {
                    let slot = self.builder.create_stack_slot((results.len() * 8) as u32);
                    let result_ptr = self.builder.ins().stack_addr(slot, 0);
                    args.push(result_ptr);
                    Some(slot)
                } else {
                    None
                };
                if (function_index as usize) < self.metadata.num_imported_funcs {
                    let entry_offset = self.offsets.function_offset(function_index as u32);
                    let vmptr = self.vmctx.expect("vmctx not set");
                    let func_ptr = self.builder.ins().load(
                        vmptr,
                        entry_offset + VMFuncRef::func_ptr_offset(),
                        MemFlags::new().with_alignment(16),
                        VelocType::PTR,
                    );
                    let target_vmctx = self.builder.ins().load(
                        vmptr,
                        entry_offset + VMFuncRef::vmctx_offset(),
                        MemFlags::new().with_alignment(8),
                        VelocType::PTR,
                    );
                    args.insert(0, target_vmctx);
                    let func_id = self.metadata.functions[function_index as usize].func_id;
                    let sig_id = self.builder.func_signature(func_id);
                    let call_inst = self.builder.ins().call_indirect(sig_id, func_ptr, &args);
                    if let Some(slot) = multi_ret_slot {
                        for (i, &ty) in results.iter().enumerate() {
                            let bits =
                                self.builder
                                    .ins()
                                    .stack_load(slot, (i * 8) as u32, VelocType::I64);
                            let res_val = self.decode_result_bits(bits, self.val_type_to_veloc(ty));
                            self.stack.push(res_val);
                        }
                    } else {
                        for i in 0..results.len() {
                            let res_val = self.builder.func().dfg.inst_results(call_inst)[i];
                            self.stack.push(res_val);
                        }
                    }
                } else {
                    args.insert(0, self.vmctx.expect("vmctx not set"));
                    let func_id = self.metadata.functions[function_index as usize].func_id;
                    let call_inst = self.builder.ins().call(func_id, &args);
                    if let Some(slot) = multi_ret_slot {
                        for (i, &ty) in results.iter().enumerate() {
                            let bits =
                                self.builder
                                    .ins()
                                    .stack_load(slot, (i * 8) as u32, VelocType::I64);
                            let res_val = self.decode_result_bits(bits, self.val_type_to_veloc(ty));
                            self.stack.push(res_val);
                        }
                    } else {
                        for i in 0..results.len() {
                            let res_val = self.builder.func().dfg.inst_results(call_inst)[i];
                            self.stack.push(res_val);
                        }
                    }
                }
            }
            Operator::CallIndirect {
                type_index,
                table_index,
                ..
            } => {
                let sig = &self.metadata.signatures[type_index as usize];
                let index = self.pop_i32();
                let mut args = Vec::new();
                for &ty in sig.params.iter().rev() {
                    let veloc_ty = self.val_type_to_veloc(ty);
                    args.push(self.pop_typed(veloc_ty));
                }
                args.reverse();
                let table_base = self.get_table_base(table_index);
                let (_, len_var) = self.table_vars[table_index as usize];
                let table_len = self.builder.use_var(len_var);
                let index_i64 = self.builder.ins().extendu(index, VelocType::I64);
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
                let entry_ptr_addr = self.builder.ins().ptr_index(
                    table_base,
                    index_i64,
                    veloc::mir::inst::PtrIndexImm {
                        scale: 8,
                        offset: 0,
                    },
                );
                let entry_ptr =
                    self.builder
                        .ins()
                        .load(entry_ptr_addr, 0, MemFlags::default(), VelocType::PTR);
                let zero = self.builder.ins().iconst(0, VelocType::I64);
                let zero_ptr = self.builder.ins().inttoptr(zero);
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
                    entry_ptr,
                    VMFuncRef::type_index_offset(),
                    MemFlags::new().with_alignment(16),
                    VelocType::I32,
                );
                let expected_sig_id = self
                    .builder
                    .ins()
                    .iconst((sig.hash_u64() as u32) as u64, VelocType::I32);
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
                    entry_ptr,
                    VMFuncRef::func_ptr_offset(),
                    MemFlags::new().with_alignment(16),
                    VelocType::PTR,
                );
                let target_vmctx = self.builder.ins().load(
                    entry_ptr,
                    VMFuncRef::vmctx_offset(),
                    MemFlags::new().with_alignment(8),
                    VelocType::PTR,
                );
                args.insert(0, target_vmctx);
                let results = &sig.results;
                let sig_id = self.ir_sig_ids[type_index as usize];
                let multi_ret_slot = if results.len() > 1 {
                    let slot = self.builder.create_stack_slot((results.len() * 8) as u32);
                    let result_ptr = self.builder.ins().stack_addr(slot, 0);
                    args.push(result_ptr);
                    Some(slot)
                } else {
                    None
                };
                let call_inst = self.builder.ins().call_indirect(sig_id, func_ptr, &args);
                if let Some(slot) = multi_ret_slot {
                    for (i, &ty) in results.iter().enumerate() {
                        let bits =
                            self.builder
                                .ins()
                                .stack_load(slot, (i * 8) as u32, VelocType::I64);
                        let res_val = self.decode_result_bits(bits, self.val_type_to_veloc(ty));
                        self.stack.push(res_val);
                    }
                } else {
                    for i in 0..results.len() {
                        let res_val = self.builder.func().dfg.inst_results(call_inst)[i];
                        self.stack.push(res_val);
                    }
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
                    func_ref,
                    VMFuncRef::func_ptr_offset(),
                    MemFlags::new().with_alignment(16),
                    VelocType::PTR,
                );
                let target_vmctx = self.builder.ins().load(
                    func_ref,
                    VMFuncRef::vmctx_offset(),
                    MemFlags::new().with_alignment(8),
                    VelocType::PTR,
                );
                args.insert(0, target_vmctx);
                let results = &sig.results;
                let sig_id = self.ir_sig_ids[type_index as usize];
                let multi_ret_slot = if results.len() > 1 {
                    let slot = self.builder.create_stack_slot((results.len() * 8) as u32);
                    let result_ptr = self.builder.ins().stack_addr(slot, 0);
                    args.push(result_ptr);
                    Some(slot)
                } else {
                    None
                };
                let call_inst = self.builder.ins().call_indirect(sig_id, func_ptr, &args);
                if let Some(slot) = multi_ret_slot {
                    for (i, &ty) in results.iter().enumerate() {
                        let bits =
                            self.builder
                                .ins()
                                .stack_load(slot, (i * 8) as u32, VelocType::I64);
                        let res_val = self.decode_result_bits(bits, self.val_type_to_veloc(ty));
                        self.stack.push(res_val);
                    }
                } else {
                    for i in 0..results.len() {
                        let res_val = self.builder.func().dfg.inst_results(call_inst)[i];
                        self.stack.push(res_val);
                    }
                }
            }
            Operator::BrTable { targets } => {
                let index = self.pop_i32();
                let (default_target, target_tys) = {
                    let default_depth = targets.default();
                    let frame_idx = self.control_stack.len() - 1 - default_depth as usize;
                    let frame = &self.control_stack[frame_idx];
                    if frame.is_loop {
                        (frame.label, frame.params_types.clone())
                    } else {
                        (frame.end_label.unwrap(), frame.results_types.clone())
                    }
                };

                let mut args = Vec::new();
                for &ty in target_tys.iter().rev() {
                    args.push(self.pop_typed(ty));
                }
                args.reverse();

                let default_call = self.builder.make_block_call(default_target, &args);
                let mut table = Vec::new();
                for t in targets.targets() {
                    let depth = t?;
                    let frame_idx = self.control_stack.len() - 1 - depth as usize;
                    let frame = &self.control_stack[frame_idx];
                    let target = if frame.is_loop {
                        frame.label
                    } else {
                        frame.end_label.unwrap()
                    };
                    // WASM ensures all targets have same arity and types
                    table.push(self.builder.make_block_call(target, &args));
                }
                self.builder.ins().br_table(index, default_call, &table);
                self.terminated = true;
            }
            Operator::Return => {
                if !self.terminated {
                    let mut vals = Vec::with_capacity(self.results.len());
                    let result_types = self.results.clone();
                    for &ty in result_types.iter().rev() {
                        vals.push(self.pop_typed(ty));
                    }
                    vals.reverse();
                    self.emit_function_return(&vals);
                    self.terminated = true;
                }
            }
            _ => { /* Handled by router */ }
        }
        Ok(())
    }
}
