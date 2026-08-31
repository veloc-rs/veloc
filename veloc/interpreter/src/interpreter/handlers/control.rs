use super::super::*;
use super::intrinsic::execute_intrinsic;

define_control_handlers! {
    interpreter, program, frame, ip, next_ip, values_ptr;
    // === Control Flow ===
    Jump { offset } => {
        next_ip = ip.byte_offset(offset as isize);
    }
    JumpWithMoves { data_offset } => {
        let target = &frame.func.data_section.jump_targets[data_offset as usize];
        Interpreter::execute_moves(frame, values_ptr, target);
        next_ip = ip.byte_offset(target.offset as isize);
    }
    Br {
        cond,
        then_offset,
        else_offset,
    } => {
        let offset = if get!(cond).unwrap_bool() {
            then_offset
        } else {
            else_offset
        };
        next_ip = ip.byte_offset(offset as isize);
    }
    BrWithMoves {
        cond,
        then_idx,
        else_idx,
    } => {
        let c = get!(cond).unwrap_bool();
        let target_idx = if c { then_idx } else { else_idx };
        let target = &frame.func.data_section.jump_targets[target_idx as usize];
        Interpreter::execute_moves(frame, values_ptr, target);
        next_ip = ip.byte_offset(target.offset as isize);
    }
    BrTable {
        idx_reg,
        data_offset,
        num_targets,
    } => {
        let idx = get!(idx_reg).unwrap_i32();
        let num = num_targets as usize;
        let target_idx = if idx >= 0 && (idx as usize) < num {
            idx as usize
        } else {
            num - 1
        };
        let target =
            &frame.func.data_section.jump_targets[data_offset as usize + target_idx];
        Interpreter::execute_moves(frame, values_ptr, target);
        next_ip = ip.byte_offset(target.offset as isize);
    }
    Return {
        data_offset,
        num_vals,
    } => {
        let data_off = data_offset as usize;
        let nvals = num_vals as usize;
        let cur_base = frame.base;
        let cur_stack = frame.stack_base;

        if interpreter.frames.is_empty() {
            for i in 0..nvals {
                interpreter.results_buffer
                    .push(get!(frame.func.data_section.return_reg(data_off, i)));
            }
            interpreter.value_stack.truncate(cur_base);
            interpreter.stack_top = cur_stack;
            return DispatchExit::Returned;
        }

        interpreter.args_buffer.clear();
        for i in 0..nvals {
            interpreter.args_buffer
                .push(get!(frame.func.data_section.return_reg(data_off, i)));
        }
        interpreter.value_stack.truncate(cur_base);
        interpreter.stack_top = cur_stack;

        let prev = interpreter.frames.pop().unwrap();
        let dst_start = prev.dst_regs_start;
        let dst_count = prev.dst_regs_count;
        frame.pc = prev.pc;
        frame.base = prev.base;
        frame.stack_base = prev.stack_base;
        frame.func = prev.func;
        frame.module = prev.module;
        next_ip = frame.func.code.as_ptr().add(frame.pc);
        values_ptr = interpreter.value_stack.as_mut_ptr().add(frame.base);

        debug_assert_eq!(dst_count, interpreter.args_buffer.len());
        for i in 0..dst_count {
            let dst_reg = interpreter.dst_regs_buffer[dst_start + i];
            if dst_reg != Reg::NULL {
                *reg!(dst_reg) = interpreter.args_buffer[i];
            }
        }
        interpreter.dst_regs_buffer.truncate(dst_start);
        dispatch_next!(next_ip, values_ptr);
    }
    Call {
        func_id,
        data_offset,
        num_rets,
        num_args,
    } => {
        let dst_start = interpreter.read_call_data(
            &frame.func.data_section,
            values_ptr,
            data_offset as usize,
            num_rets,
            num_args,
        );
        let func = veloc_ir::FuncId::from_u32(func_id);
        let return_pc = next_ip.offset_from(frame.func.code.as_ptr()) as usize;

        match program.call_target(frame.module, func) {
            CallTarget::Bytecode(target_module, target_func) => {
                if let Err(exit) = interpreter.do_call(
                    program,
                    target_module,
                    target_func,
                    dst_start,
                    num_rets as usize,
                    return_pc,
                    frame,
                ) {
                    return exit;
                }
                next_ip = frame.func.code.as_ptr();
                values_ptr = interpreter.value_stack.as_mut_ptr().add(frame.base);
                dispatch_next!(next_ip, values_ptr);
            }
            CallTarget::Host(host) => {
                if program
                    .call_host(
                        host,
                        &mut interpreter.args_buffer,
                        num_args as usize,
                        num_rets as usize,
                    )
                    .is_err()
                {
                    return DispatchExit::InvalidHostCall;
                }
                values_ptr = interpreter.value_stack.as_mut_ptr().add(frame.base);
                for i in 0..num_rets as usize {
                    let dst = interpreter.dst_regs_buffer[dst_start + i];
                    if dst != Reg::NULL {
                        *reg!(dst) = interpreter.args_buffer[i];
                    }
                }
                interpreter.dst_regs_buffer.truncate(dst_start);
            }
        }
    }
    CallIndirect {
        ptr,
        data_offset,
        num_rets,
        num_args,
    } => {
        let dst_start = interpreter.read_call_data(
            &frame.func.data_section,
            values_ptr,
            data_offset as usize,
            num_rets,
            num_args,
        );
        let address = get!(ptr).0 as usize;
        let return_pc = next_ip.offset_from(frame.func.code.as_ptr()) as usize;

        match program.resolve_ref(address) {
            Some(CallTarget::Bytecode(target_module, target_func)) => {
                if let Err(exit) = interpreter.do_call(
                    program,
                    target_module,
                    target_func,
                    dst_start,
                    num_rets as usize,
                    return_pc,
                    frame,
                ) {
                    return exit;
                }
                next_ip = frame.func.code.as_ptr();
                values_ptr = interpreter.value_stack.as_mut_ptr().add(frame.base);
                dispatch_next!(next_ip, values_ptr);
            }
            Some(CallTarget::Host(host)) => {
                if program
                    .call_host(
                        host,
                        &mut interpreter.args_buffer,
                        num_args as usize,
                        num_rets as usize,
                    )
                    .is_err()
                {
                    return DispatchExit::InvalidHostCall;
                }
                values_ptr = interpreter.value_stack.as_mut_ptr().add(frame.base);
                for i in 0..num_rets as usize {
                    let dst = interpreter.dst_regs_buffer[dst_start + i];
                    if dst != Reg::NULL {
                        *reg!(dst) = interpreter.args_buffer[i];
                    }
                }
                interpreter.dst_regs_buffer.truncate(dst_start);
            }
            None => {
                return DispatchExit::InvalidFunctionReference;
            }
        }
    }
    PtrIndex {
        dst,
        ptr,
        index,
        scale,
        offset,
    } => {
        let p = get!(ptr).unwrap_i64();
        let idx = get!(index).unwrap_i64();
        let s = scale as i64;
        let o = offset as i32 as i64;
        set!(
            dst,
            InterpreterValue::i64(p.wrapping_add(idx.wrapping_mul(s)).wrapping_add(o))
        );
    }
    Select {
        dst,
        cond,
        then_reg,
        else_reg,
    } => {
        set!(
            dst,
            if get!(cond).unwrap_bool() {
                get!(then_reg)
            } else {
                get!(else_reg)
            }
        );
    }
    RegMove { dst, src } => { set!(dst, get!(src)); }
    CallIntrinsic {
        intrinsic,
        data_offset,
        num_rets,
        num_args,
    } => {
        let dst_start = interpreter.read_call_data(
            &frame.func.data_section,
            values_ptr,
            data_offset as usize,
            num_rets,
            num_args,
        );
        let res = execute_intrinsic(intrinsic, &interpreter.args_buffer);
        values_ptr = interpreter.value_stack.as_mut_ptr().add(frame.base);
        if num_rets > 0 {
            let dst = interpreter.dst_regs_buffer[dst_start];
            if dst != Reg::NULL {
                *reg!(dst) = res;
            }
        }
        interpreter.dst_regs_buffer.truncate(dst_start);
    }
    Unreachable {} => {
        return DispatchExit::Unreachable;
    }
}
