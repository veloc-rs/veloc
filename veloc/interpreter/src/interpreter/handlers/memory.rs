use super::super::*;

define_memory_handlers! {
    interpreter, mem, frame;
    // === Memory Access ===
    I32Load { dst, ptr, offset } => {
        let addr = (get!(ptr).0 as usize).wrapping_add(offset as usize);
        let Some(value) = Interpreter::load_memory::<M, i32>(mem, addr) else {
            return DispatchExit::OutOfBounds;
        };
        set!(dst, InterpreterValue::i32(value));
    }
    I64Load { dst, ptr, offset } => {
        let addr = (get!(ptr).0 as usize).wrapping_add(offset as usize);
        let Some(value) = Interpreter::load_memory::<M, i64>(mem, addr) else {
            return DispatchExit::OutOfBounds;
        };
        set!(dst, InterpreterValue::i64(value));
    }
    I8Load { dst, ptr, offset } => {
        let addr = (get!(ptr).0 as usize).wrapping_add(offset as usize);
        let Some(value) = Interpreter::load_memory::<M, u8>(mem, addr) else {
            return DispatchExit::OutOfBounds;
        };
        set!(dst, InterpreterValue::i64(value as i64));
    }
    I16Load { dst, ptr, offset } => {
        let addr = (get!(ptr).0 as usize).wrapping_add(offset as usize);
        let Some(value) = Interpreter::load_memory::<M, u16>(mem, addr) else {
            return DispatchExit::OutOfBounds;
        };
        set!(dst, InterpreterValue::i64(value as i64));
    }
    F32Load { dst, ptr, offset } => {
        let addr = (get!(ptr).0 as usize).wrapping_add(offset as usize);
        let Some(value) = Interpreter::load_memory::<M, f32>(mem, addr) else {
            return DispatchExit::OutOfBounds;
        };
        set!(dst, InterpreterValue::f32(value));
    }
    F64Load { dst, ptr, offset } => {
        let addr = (get!(ptr).0 as usize).wrapping_add(offset as usize);
        let Some(value) = Interpreter::load_memory::<M, f64>(mem, addr) else {
            return DispatchExit::OutOfBounds;
        };
        set!(dst, InterpreterValue::f64(value));
    }
    I32Store { val, ptr, offset } => {
        let addr = (get!(ptr).0 as usize).wrapping_add(offset as usize);
        if !Interpreter::store_memory(mem, addr, get!(val).unwrap_i32()) {
            return DispatchExit::OutOfBounds;
        }
    }
    I64Store { val, ptr, offset } => {
        let addr = (get!(ptr).0 as usize).wrapping_add(offset as usize);
        if !Interpreter::store_memory(mem, addr, get!(val).unwrap_i64()) {
            return DispatchExit::OutOfBounds;
        }
    }
    I8Store { val, ptr, offset } => {
        let addr = (get!(ptr).0 as usize).wrapping_add(offset as usize);
        if !Interpreter::store_memory(mem, addr, get!(val).unwrap_i64() as u8) {
            return DispatchExit::OutOfBounds;
        }
    }
    I16Store { val, ptr, offset } => {
        let addr = (get!(ptr).0 as usize).wrapping_add(offset as usize);
        if !Interpreter::store_memory(mem, addr, get!(val).unwrap_i64() as u16) {
            return DispatchExit::OutOfBounds;
        }
    }
    F32Store { val, ptr, offset } => {
        let addr = (get!(ptr).0 as usize).wrapping_add(offset as usize);
        if !Interpreter::store_memory(mem, addr, get!(val).unwrap_f32()) {
            return DispatchExit::OutOfBounds;
        }
    }
    F64Store { val, ptr, offset } => {
        let addr = (get!(ptr).0 as usize).wrapping_add(offset as usize);
        if !Interpreter::store_memory(mem, addr, get!(val).unwrap_f64()) {
            return DispatchExit::OutOfBounds;
        }
    }

    // === Stack Operations ===
    StackAddr { dst, offset } => {
        let ptr = interpreter
            .stack_memory
            .as_ptr()
            .add(frame.stack_base + offset as usize);
        set!(dst, InterpreterValue::i64(ptr as i64));
    }
    StackLoad { dst, ty, offset } => {
        let addr = frame.stack_base + offset as usize;
        let ptr = interpreter.stack_memory.as_ptr().add(addr);
        set!(
            dst,
            match ty {
                ScalarType::I8 =>
                    InterpreterValue::i32((ptr as *const i8).read_unaligned() as i32),
                ScalarType::I16 =>
                    InterpreterValue::i32((ptr as *const i16).read_unaligned() as i32),
                ScalarType::I32 =>
                    InterpreterValue::i32((ptr as *const i32).read_unaligned()),
                ScalarType::I64 | ScalarType::Ptr =>
                    InterpreterValue::i64((ptr as *const i64).read_unaligned()),
                ScalarType::F32 =>
                    InterpreterValue::f32((ptr as *const f32).read_unaligned()),
                ScalarType::F64 =>
                    InterpreterValue::f64((ptr as *const f64).read_unaligned()),
                _ => panic!("Unknown type {:?} in StackLoad", ty),
            }
        );
    }
    StackStore { val, ty, offset } => {
        let addr = frame.stack_base + offset as usize;
        let ptr = interpreter.stack_memory.as_mut_ptr().add(addr);
        let v = get!(val);
        match ty {
            ScalarType::I8 => (ptr as *mut i8).write_unaligned(v.unwrap_i32() as i8),
            ScalarType::I16 => (ptr as *mut i16).write_unaligned(v.unwrap_i32() as i16),
            ScalarType::I32 => (ptr as *mut i32).write_unaligned(v.unwrap_i32()),
            ScalarType::I64 | ScalarType::Ptr => {
                (ptr as *mut i64).write_unaligned(v.unwrap_i64())
            }
            ScalarType::F32 => (ptr as *mut f32).write_unaligned(v.unwrap_f32()),
            ScalarType::F64 => (ptr as *mut f64).write_unaligned(v.unwrap_f64()),
            _ => panic!("Unknown type {:?} in StackStore", ty),
        }
    }
}
