use crate::bytecode::{
    CompiledFunction, EXTEND_TYPE_I8, EXTEND_TYPE_I16, EXTEND_TYPE_I32, Opcode,
    STACK_TYPE_F32, STACK_TYPE_F64, STACK_TYPE_I8, STACK_TYPE_I16, STACK_TYPE_I32, STACK_TYPE_I64,
};
use crate::host::{ModuleId, Program};
use crate::value::InterpreterValue;
use ::alloc::vec::Vec;
use veloc_ir::Intrinsic;

pub trait VirtualMemory {
    fn translate_addr(&self, logical_addr: usize, size: usize) -> Option<*mut u8>;
}

pub struct Interpreter {
    pub value_stack: Vec<InterpreterValue>,
    pub stack_memory: Vec<u8>,
    pub frames: Vec<StackFrame>,
    args_buffer: Vec<InterpreterValue>,
}

pub struct StackFrame {
    pub mid: ModuleId,
    pub func: ::alloc::sync::Arc<CompiledFunction>,
    pub pc: usize,
    pub base: usize,
    pub stack_base: usize,
    pub dst_regs: Vec<u16>, // Where to put the results after returning (support multi-value)
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            value_stack: Vec::with_capacity(4096),
            stack_memory: Vec::with_capacity(1024 * 1024),
            frames: Vec::with_capacity(128),
            args_buffer: Vec::with_capacity(32),
        }
    }

    pub fn run_function(
        &mut self,
        program: &Program,
        mem: &dyn VirtualMemory,
        mid: ModuleId,
        fid: veloc_ir::FuncId,
        args: &[InterpreterValue],
    ) -> Vec<InterpreterValue> {
        let func = program.get_compiled_func(mid, fid);
        let base = self.value_stack.len();
        self.value_stack
            .resize(base + func.register_count, InterpreterValue::none());

        let stack_base = self.stack_memory.len();
        let mut total_stack_size = 0;
        for &size in &func.stack_slots_sizes {
            total_stack_size += size;
        }
        self.stack_memory.resize(stack_base + total_stack_size, 0);

        // Initialize parameters
        for (i, &new_idx) in func.param_indices.iter().enumerate() {
            if i < args.len() {
                self.value_stack[base + new_idx as usize] = args[i];
            }
        }

        self.frames.push(StackFrame {
            mid,
            func,
            pc: 0,
            base,
            stack_base,
            dst_regs: Vec::new(),
        });

        self.execute(program, mem)
    }

    #[inline(always)]
    fn execute(&mut self, program: &Program, mem: &dyn VirtualMemory) -> Vec<InterpreterValue> {
        let frame = self.frames.pop().unwrap();
        let mut pc = frame.pc;
        let mut base = frame.base;
        let mut stack_base = frame.stack_base;
        let mut func = frame.func.clone();
        let mut mid = frame.mid;

        'main_loop: loop {
            let code_ptr = func.code.as_ptr();
            let code_len = func.code.len();
            let mut values_ptr = self.value_stack.as_mut_ptr();

            macro_rules! read_u16 {
                () => {{
                    let v = unsafe {
                        let ptr = code_ptr.add(pc) as *const u16;
                        pc += 2;
                        u16::from_le(ptr.read_unaligned())
                    };
                    v
                }};
            }
            macro_rules! reg {
                ($r:expr) => {
                    unsafe { &mut *values_ptr.add(base + $r as usize) }
                };
            }
            macro_rules! reg_val {
                ($r:expr) => {
                    unsafe { *values_ptr.add(base + $r as usize) }
                };
            }

            macro_rules! bin_op {
                ($op:ident, $u:ident, $w:ident, $method:ident) => {{
                    let (d, l, r) = decode_into!($op, pc, code_ptr);
                    let lv = reg_val!(l).$u();
                    let rv = reg_val!(r).$u();
                    *reg!(d) = InterpreterValue::$w(lv.$method(rv as _));
                }};
                ($op:ident, $u:ident, $w:ident, $builtin:tt) => {{
                    let (d, l, r) = decode_into!($op, pc, code_ptr);
                    let lv = reg_val!(l).$u();
                    let rv = reg_val!(r).$u();
                    *reg!(d) = InterpreterValue::$w(lv $builtin rv);
                }};
                ($op:ident, $u:ident, $w:ident, $builtin:tt, $as_ty:ty) => {{
                    let (d, l, r) = decode_into!($op, pc, code_ptr);
                    let lv = reg_val!(l).$u() as $as_ty;
                    let rv = reg_val!(r).$u() as $as_ty;
                    *reg!(d) = InterpreterValue::$w((lv $builtin rv) as _);
                }};
            }

            macro_rules! imm_op {
                ($op:ident, $u:ident, $w:ident, $method:ident, $ity:ty) => {{
                    let (d, l, i) = decode_into!($op, pc, code_ptr);
                    let lv = reg_val!(l).$u();
                    *reg!(d) = InterpreterValue::$w(lv.$method(i as $ity as _));
                }};
                ($op:ident, $u:ident, $w:ident, $builtin:tt, $ity:ty) => {{
                    let (d, l, i) = decode_into!($op, pc, code_ptr);
                    let lv = reg_val!(l).$u();
                    *reg!(d) = InterpreterValue::$w(lv $builtin (i as $ity));
                }};
            }

            macro_rules! cmp_op {
                ($op:ident, $u:ident, $builtin:tt) => {{
                    let (d, l, r) = decode_into!($op, pc, code_ptr);
                    *reg!(d) = InterpreterValue::bool(reg_val!(l).$u() $builtin reg_val!(r).$u());
                }};
                ($op:ident, $u:ident, $as_ty:ty, $builtin:tt) => {{
                    let (d, l, r) = decode_into!($op, pc, code_ptr);
                    *reg!(d) = InterpreterValue::bool((reg_val!(l).$u() as $as_ty) $builtin (reg_val!(r).$u() as $as_ty));
                }};
            }

            macro_rules! unary_op {
                ($op:ident, $u:ident, $w:ident, $method:ident) => {{
                    let (d, s) = decode_into!($op, pc, code_ptr);
                    *reg!(d) = InterpreterValue::$w(reg_val!(s).$u().$method() as _);
                }};
                ($op:ident, $u:ident, $w:ident, $builtin:tt) => {{
                    let (d, s) = decode_into!($op, pc, code_ptr);
                    *reg!(d) = InterpreterValue::$w($builtin reg_val!(s).$u());
                }};
            }

            macro_rules! cmp_zero_op {
                ($op:ident, $u:ident) => {{
                    let (d, s) = decode_into!($op, pc, code_ptr);
                    *reg!(d) = InterpreterValue::bool(reg_val!(s).$u() == 0);
                }};
            }

            macro_rules! conv_op {
                ($op:ident, $u:ident, $w:ident, $ty:ty) => {{
                    let (d, s) = decode_into!($op, pc, code_ptr);
                    *reg!(d) = InterpreterValue::$w(reg_val!(s).$u() as $ty);
                }};
                ($op:ident, $u:ident, $w:ident, $ity:ty, $ty:ty) => {{
                    let (d, s) = decode_into!($op, pc, code_ptr);
                    *reg!(d) = InterpreterValue::$w(reg_val!(s).$u() as $ity as $ty);
                }};
            }

            macro_rules! move_op {
                ($op:ident) => {{
                    let (d, s) = decode_into!($op, pc, code_ptr);
                    *reg!(d) = reg_val!(s);
                }};
            }

            macro_rules! load_op {
                ($op:ident, $w:ident, $ptr_ty:ty, $size:expr) => {{
                    let (d, p, o) = decode_into!($op, pc, code_ptr);
                    let addr = reg_val!(p).unwarp_i64() as usize + o as usize;
                    let ptr = mem.translate_addr(addr, $size).expect("Segment fault");
                    *reg!(d) =
                        InterpreterValue::$w(
                            unsafe { (ptr as *const $ptr_ty).read_unaligned() } as _
                        );
                }};
            }

            macro_rules! store_op {
                ($op:ident, $u:ident, $ptr_ty:ty, $size:expr) => {{
                    let (v, p, o) = decode_into!($op, pc, code_ptr);
                    let addr = reg_val!(p).unwarp_i64() as usize + o as usize;
                    let ptr = mem.translate_addr(addr, $size).expect("Segment fault");
                    #[allow(unused_unsafe)]
                    unsafe {
                        (ptr as *mut $ptr_ty).write_unaligned(reg_val!(v).$u() as _)
                    };
                }};
            }

            macro_rules! shift_op {
                ($op:ident, $u:ident, $w:ident, $method:ident) => {{
                    let (d, l, r) = decode_into!($op, pc, code_ptr);
                    let lv = reg_val!(l).$u();
                    let rv = reg_val!(r).$u() as u32;
                    *reg!(d) = InterpreterValue::$w(lv.$method(rv));
                }};
                ($op:ident, $u:ident, $w:ident, $method:ident, $uty:ty) => {{
                    let (d, l, r) = decode_into!($op, pc, code_ptr);
                    let lv = reg_val!(l).$u() as $uty;
                    let rv = reg_val!(r).$u() as u32;
                    *reg!(d) = InterpreterValue::$w(lv.$method(rv) as _);
                }};
            }

            macro_rules! shift_imm_op {
                ($op:ident, $u:ident, $w:ident, $method:ident) => {{
                    let (d, l, i) = decode_into!($op, pc, code_ptr);
                    let lv = reg_val!(l).$u();
                    *reg!(d) = InterpreterValue::$w(lv.$method(i as u32));
                }};
                ($op:ident, $u:ident, $w:ident, $method:ident, $uty:ty) => {{
                    let (d, l, i) = decode_into!($op, pc, code_ptr);
                    let lv = reg_val!(l).$u() as $uty;
                    *reg!(d) = InterpreterValue::$w(lv.$method(i as u32) as _);
                }};
            }

            // --- High-level Semantic Macros for Clarity ---
            macro_rules! i32_bin {
                ($op:ident, $method:ident) => {
                    bin_op!($op, unwarp_i32, i32, $method)
                };
                ($op:ident, $oper:tt) => {
                    bin_op!($op, unwarp_i32, i32, $oper)
                };
                ($op:ident, $oper:tt, $as_ty:ty) => {
                    bin_op!($op, unwarp_i32, i32, $oper, $as_ty)
                };
            }
            macro_rules! i32_imm {
                ($op:ident, $method:ident) => {
                    imm_op!($op, unwarp_i32, i32, $method, i32)
                };
                ($op:ident, $oper:tt) => {
                    imm_op!($op, unwarp_i32, i32, $oper, i32)
                };
            }
            macro_rules! i32_shift {
                ($op:ident, $method:ident) => {
                    shift_op!($op, unwarp_i32, i32, $method)
                };
                ($op:ident, $method:ident, $uty:ty) => {
                    shift_op!($op, unwarp_i32, i32, $method, $uty)
                };
            }
            macro_rules! i32_shift_imm {
                ($op:ident, $method:ident) => {
                    shift_imm_op!($op, unwarp_i32, i32, $method)
                };
                ($op:ident, $method:ident, $uty:ty) => {
                    shift_imm_op!($op, unwarp_i32, i32, $method, $uty)
                };
            }
            macro_rules! i32_cmp {
                ($op:ident, $oper:tt) => {
                    cmp_op!($op, unwarp_i32, $oper)
                };
                ($op:ident, $as_ty:ty, $oper:tt) => {
                    cmp_op!($op, unwarp_i32, $as_ty, $oper)
                };
            }
            macro_rules! i32_unary {
                ($op:ident, $method:ident) => {
                    unary_op!($op, unwarp_i32, i32, $method)
                };
                ($op:ident, $oper:tt) => {
                    unary_op!($op, unwarp_i32, i32, $oper)
                };
            }
            macro_rules! i32_eqz {
                ($op:ident) => {
                    cmp_zero_op!($op, unwarp_i32)
                };
            }
            macro_rules! i64_bin {
                ($op:ident, $method:ident) => {
                    bin_op!($op, unwarp_i64, i64, $method)
                };
                ($op:ident, $oper:tt) => {
                    bin_op!($op, unwarp_i64, i64, $oper)
                };
                ($op:ident, $oper:tt, $as_ty:ty) => {
                    bin_op!($op, unwarp_i64, i64, $oper, $as_ty)
                };
            }
            macro_rules! i64_imm {
                ($op:ident, $method:ident) => {
                    imm_op!($op, unwarp_i64, i64, $method, i64)
                };
                ($op:ident, $oper:tt) => {
                    imm_op!($op, unwarp_i64, i64, $oper, i64)
                };
            }
            macro_rules! i64_shift {
                ($op:ident, $method:ident) => {
                    shift_op!($op, unwarp_i64, i64, $method)
                };
                ($op:ident, $method:ident, $uty:ty) => {
                    shift_op!($op, unwarp_i64, i64, $method, $uty)
                };
            }
            macro_rules! i64_shift_imm {
                ($op:ident, $method:ident) => {
                    shift_imm_op!($op, unwarp_i64, i64, $method)
                };
                ($op:ident, $method:ident, $uty:ty) => {
                    shift_imm_op!($op, unwarp_i64, i64, $method, $uty)
                };
            }
            macro_rules! i64_cmp {
                ($op:ident, $oper:tt) => {
                    cmp_op!($op, unwarp_i64, $oper)
                };
                ($op:ident, $as_ty:ty, $oper:tt) => {
                    cmp_op!($op, unwarp_i64, $as_ty, $oper)
                };
            }
            macro_rules! i64_unary {
                ($op:ident, $method:ident) => {
                    unary_op!($op, unwarp_i64, i64, $method)
                };
                ($op:ident, $oper:tt) => {
                    unary_op!($op, unwarp_i64, i64, $oper)
                };
            }
            macro_rules! i64_eqz {
                ($op:ident) => {
                    cmp_zero_op!($op, unwarp_i64)
                };
            }

            macro_rules! f32_bin {
                ($op:ident, $method:ident) => {
                    bin_op!($op, unwarp_f32, f32, $method)
                };
                ($op:ident, $oper:tt) => {
                    bin_op!($op, unwarp_f32, f32, $oper)
                };
            }
            macro_rules! f32_unary {
                ($op:ident, $method:ident) => {
                    unary_op!($op, unwarp_f32, f32, $method)
                };
                ($op:ident, $oper:tt) => {
                    unary_op!($op, unwarp_f32, f32, $oper)
                };
            }
            macro_rules! f32_cmp {
                ($op:ident, $oper:tt) => {
                    cmp_op!($op, unwarp_f32, $oper)
                };
            }
            macro_rules! f64_bin {
                ($op:ident, $method:ident) => {
                    bin_op!($op, unwarp_f64, f64, $method)
                };
                ($op:ident, $oper:tt) => {
                    bin_op!($op, unwarp_f64, f64, $oper)
                };
            }
            macro_rules! f64_unary {
                ($op:ident, $method:ident) => {
                    unary_op!($op, unwarp_f64, f64, $method)
                };
                ($op:ident, $oper:tt) => {
                    unary_op!($op, unwarp_f64, f64, $oper)
                };
            }
            macro_rules! f64_cmp {
                ($op:ident, $oper:tt) => {
                    cmp_op!($op, unwarp_f64, $oper)
                };
            }

            macro_rules! prepare_call {
                ($target_mid:expr, $target_fid:expr, $dst_regs:expr, $args:expr) => {{
                    if program.compiled_modules[$target_mid.0][$target_fid.0 as usize].is_none() {
                        panic!(
                            "Calling uncompiled function: mid={:?}, fid={:?}",
                            $target_mid, $target_fid
                        );
                    }

                    let next_func = program.get_compiled_func($target_mid, $target_fid);
                    self.frames.push(StackFrame {
                        mid,
                        func: func.clone(),
                        pc,
                        base,
                        stack_base,
                        dst_regs: $dst_regs,
                    });

                    mid = $target_mid;
                    func = next_func;
                    pc = 0;
                    base = self.value_stack.len();
                    self.value_stack
                        .resize(base + func.register_count, InterpreterValue::none());

                    stack_base = self.stack_memory.len();
                    let mut current_offset = 0;
                    for &size in &func.stack_slots_sizes {
                        current_offset += size;
                    }
                    self.stack_memory.resize(stack_base + current_offset, 0);

                    // Initialize parameters
                    for (i, &new_idx) in func.param_indices.iter().enumerate() {
                        if i < $args.len() {
                            self.value_stack[base + new_idx as usize] = $args[i];
                        }
                    }
                    continue 'main_loop;
                }};
            }

            while pc < code_len {
                let opcode_byte = unsafe { *code_ptr.add(pc) };
                pc += 1;
                let opcode: Opcode = unsafe { core::mem::transmute(opcode_byte) };

                match opcode {
                    // --- Constants ---
                    Opcode::Iconst => {
                        let (d, v) = decode_into!(Iconst, pc, code_ptr);
                        *reg!(d) = InterpreterValue::i64(v as i64);
                    }
                    Opcode::Fconst => {
                        let (d, v) = decode_into!(Fconst, pc, code_ptr);
                        *reg!(d) = InterpreterValue(v);
                    }
                    Opcode::Bconst => {
                        let (d, v) = decode_into!(Bconst, pc, code_ptr);
                        *reg!(d) = InterpreterValue::bool(v != 0);
                    }

                    // --- I32 Operations ---
                    Opcode::I32Add => i32_bin!(I32Add, wrapping_add),
                    Opcode::I32AddImm => i32_imm!(I32AddImm, wrapping_add),
                    Opcode::I32Sub => i32_bin!(I32Sub, wrapping_sub),
                    Opcode::I32SubImm => i32_imm!(I32SubImm, wrapping_sub),
                    Opcode::I32Mul => i32_bin!(I32Mul, wrapping_mul),
                    Opcode::I32DivS => i32_bin!(I32DivS, /),
                    Opcode::I32DivU => i32_bin!(I32DivU, /, u32),
                    Opcode::I32RemS => i32_bin!(I32RemS, %),
                    Opcode::I32RemU => i32_bin!(I32RemU, %, u32),
                    Opcode::I32And => i32_bin!(I32And, &),
                    Opcode::I32AndImm => i32_imm!(I32AndImm, &),
                    Opcode::I32Or => i32_bin!(I32Or, |),
                    Opcode::I32OrImm => i32_imm!(I32OrImm, |),
                    Opcode::I32Xor => i32_bin!(I32Xor, ^),
                    Opcode::I32XorImm => i32_imm!(I32XorImm, ^),
                    Opcode::I32Shl => i32_shift!(I32Shl, wrapping_shl),
                    Opcode::I32ShlImm => i32_shift_imm!(I32ShlImm, wrapping_shl),
                    Opcode::I32ShrS => i32_shift!(I32ShrS, wrapping_shr),
                    Opcode::I32ShrSImm => i32_shift_imm!(I32ShrSImm, wrapping_shr),
                    Opcode::I32ShrU => i32_shift!(I32ShrU, wrapping_shr, u32),
                    Opcode::I32ShrUImm => i32_shift_imm!(I32ShrUImm, wrapping_shr, u32),
                    Opcode::I32RotL => i32_shift!(I32RotL, rotate_left),
                    Opcode::I32RotR => i32_shift!(I32RotR, rotate_right),
                    Opcode::I32Clz => i32_unary!(I32Clz, leading_zeros),
                    Opcode::I32Ctz => i32_unary!(I32Ctz, trailing_zeros),
                    Opcode::I32Popcnt => i32_unary!(I32Popcnt, count_ones),
                    Opcode::I32Eqz => i32_eqz!(I32Eqz),
                    Opcode::I32Eq => i32_cmp!(I32Eq, ==),
                    Opcode::I32Ne => i32_cmp!(I32Ne, !=),
                    Opcode::I32LtS => i32_cmp!(I32LtS, <),
                    Opcode::I32LtU => i32_cmp!(I32LtU, u32, <),
                    Opcode::I32LeS => i32_cmp!(I32LeS, <=),
                    Opcode::I32LeU => i32_cmp!(I32LeU, u32, <=),
                    Opcode::I32GtS => i32_cmp!(I32GtS, >),
                    Opcode::I32GtU => i32_cmp!(I32GtU, u32, >),
                    Opcode::I32GeS => i32_cmp!(I32GeS, >=),
                    Opcode::I32GeU => i32_cmp!(I32GeU, u32, >=),

                    // --- I64 Operations ---
                    Opcode::I64Add => i64_bin!(I64Add, wrapping_add),
                    Opcode::I64AddImm => i64_imm!(I64AddImm, wrapping_add),
                    Opcode::I64Sub => i64_bin!(I64Sub, wrapping_sub),
                    Opcode::I64SubImm => i64_imm!(I64SubImm, wrapping_sub),
                    Opcode::I64Mul => i64_bin!(I64Mul, wrapping_mul),
                    Opcode::I64DivS => i64_bin!(I64DivS, /),
                    Opcode::I64DivU => i64_bin!(I64DivU, /, u64),
                    Opcode::I64RemS => i64_bin!(I64RemS, %),
                    Opcode::I64RemU => i64_bin!(I64RemU, %, u64),
                    Opcode::I64And => i64_bin!(I64And, &),
                    Opcode::I64AndImm => i64_imm!(I64AndImm, &),
                    Opcode::I64Or => i64_bin!(I64Or, |),
                    Opcode::I64OrImm => i64_imm!(I64OrImm, |),
                    Opcode::I64Xor => i64_bin!(I64Xor, ^),
                    Opcode::I64XorImm => i64_imm!(I64XorImm, ^),
                    Opcode::I64Shl => i64_shift!(I64Shl, wrapping_shl),
                    Opcode::I64ShlImm => i64_shift_imm!(I64ShlImm, wrapping_shl),
                    Opcode::I64ShrS => i64_shift!(I64ShrS, wrapping_shr),
                    Opcode::I64ShrSImm => i64_shift_imm!(I64ShrSImm, wrapping_shr),
                    Opcode::I64ShrU => i64_shift!(I64ShrU, wrapping_shr, u64),
                    Opcode::I64ShrUImm => i64_shift_imm!(I64ShrUImm, wrapping_shr, u64),
                    Opcode::I64RotL => i64_shift!(I64RotL, rotate_left),
                    Opcode::I64RotR => i64_shift!(I64RotR, rotate_right),
                    Opcode::I64Clz => i64_unary!(I64Clz, leading_zeros),
                    Opcode::I64Ctz => i64_unary!(I64Ctz, trailing_zeros),
                    Opcode::I64Popcnt => i64_unary!(I64Popcnt, count_ones),
                    Opcode::I64Eqz => i64_eqz!(I64Eqz),
                    Opcode::I64Eq => i64_cmp!(I64Eq, ==),
                    Opcode::I64Ne => i64_cmp!(I64Ne, !=),
                    Opcode::I64LtS => i64_cmp!(I64LtS, <),
                    Opcode::I64LtU => i64_cmp!(I64LtU, u64, <),
                    Opcode::I64LeS => i64_cmp!(I64LeS, <=),
                    Opcode::I64LeU => i64_cmp!(I64LeU, u64, <=),
                    Opcode::I64GtS => i64_cmp!(I64GtS, >),
                    Opcode::I64GtU => i64_cmp!(I64GtU, u64, >),
                    Opcode::I64GeS => i64_cmp!(I64GeS, >=),
                    Opcode::I64GeU => i64_cmp!(I64GeU, u64, >=),

                    // --- F32 Operations ---
                    Opcode::F32Add => f32_bin!(F32Add, +),
                    Opcode::F32Sub => f32_bin!(F32Sub, -),
                    Opcode::F32Mul => f32_bin!(F32Mul, *),
                    Opcode::F32Div => f32_bin!(F32Div, /),
                    Opcode::F32Abs => f32_unary!(F32Abs, abs),
                    Opcode::F32Neg => f32_unary!(F32Neg, -),
                    Opcode::F32Sqrt => f32_unary!(F32Sqrt, sqrt),
                    Opcode::F32Ceil => f32_unary!(F32Ceil, ceil),
                    Opcode::F32Floor => f32_unary!(F32Floor, floor),
                    Opcode::F32Trunc => f32_unary!(F32Trunc, trunc),
                    Opcode::F32Nearest => f32_unary!(F32Nearest, round_ties_even),
                    Opcode::F32Min => f32_bin!(F32Min, min),
                    Opcode::F32Max => f32_bin!(F32Max, max),
                    Opcode::F32CopySign => f32_bin!(F32CopySign, copysign),
                    Opcode::F32Eq => f32_cmp!(F32Eq, ==),
                    Opcode::F32Ne => f32_cmp!(F32Ne, !=),
                    Opcode::F32Lt => f32_cmp!(F32Lt, <),
                    Opcode::F32Le => f32_cmp!(F32Le, <=),
                    Opcode::F32Gt => f32_cmp!(F32Gt, >),
                    Opcode::F32Ge => f32_cmp!(F32Ge, >=),

                    // --- F64 Operations ---
                    Opcode::F64Add => f64_bin!(F64Add, +),
                    Opcode::F64Sub => f64_bin!(F64Sub, -),
                    Opcode::F64Mul => f64_bin!(F64Mul, *),
                    Opcode::F64Div => f64_bin!(F64Div, /),
                    Opcode::F64Abs => f64_unary!(F64Abs, abs),
                    Opcode::F64Neg => f64_unary!(F64Neg, -),
                    Opcode::F64Sqrt => f64_unary!(F64Sqrt, sqrt),
                    Opcode::F64Ceil => f64_unary!(F64Ceil, ceil),
                    Opcode::F64Floor => f64_unary!(F64Floor, floor),
                    Opcode::F64Trunc => f64_unary!(F64Trunc, trunc),
                    Opcode::F64Nearest => f64_unary!(F64Nearest, round_ties_even),
                    Opcode::F64Min => f64_bin!(F64Min, min),
                    Opcode::F64Max => f64_bin!(F64Max, max),
                    Opcode::F64CopySign => f64_bin!(F64CopySign, copysign),
                    Opcode::F64Eq => f64_cmp!(F64Eq, ==),
                    Opcode::F64Ne => f64_cmp!(F64Ne, !=),
                    Opcode::F64Lt => f64_cmp!(F64Lt, <),
                    Opcode::F64Le => f64_cmp!(F64Le, <=),
                    Opcode::F64Gt => f64_cmp!(F64Gt, >),
                    Opcode::F64Ge => f64_cmp!(F64Ge, >=),

                    // --- Conversions ---
                    Opcode::I32TruncF32S => conv_op!(I32TruncF32S, unwarp_f32, i32, i32),
                    Opcode::I32TruncF32U => conv_op!(I32TruncF32U, unwarp_f32, i32, u32, i32),
                    Opcode::I32TruncF64S => conv_op!(I32TruncF64S, unwarp_f64, i32, i32),
                    Opcode::I32TruncF64U => conv_op!(I32TruncF64U, unwarp_f64, i32, u32, i32),
                    Opcode::I64TruncF32S => conv_op!(I64TruncF32S, unwarp_f32, i64, i64),
                    Opcode::I64TruncF32U => conv_op!(I64TruncF32U, unwarp_f32, i64, u64, i64),
                    Opcode::I64TruncF64S => conv_op!(I64TruncF64S, unwarp_f64, i64, i64),
                    Opcode::I64TruncF64U => conv_op!(I64TruncF64U, unwarp_f64, i64, u64, i64),
                    Opcode::F32ConvertI32S => conv_op!(F32ConvertI32S, unwarp_i32, f32, f32),
                    Opcode::F32ConvertI32U => conv_op!(F32ConvertI32U, unwarp_i32, f32, u32, f32),
                    Opcode::F32ConvertI64S => conv_op!(F32ConvertI64S, unwarp_i64, f32, i64, f32),
                    Opcode::F32ConvertI64U => conv_op!(F32ConvertI64U, unwarp_i64, f32, u64, f32),
                    Opcode::F64ConvertI32S => conv_op!(F64ConvertI32S, unwarp_i32, f64, f64),
                    Opcode::F64ConvertI32U => conv_op!(F64ConvertI32U, unwarp_i32, f64, u32, f64),
                    Opcode::F64ConvertI64S => conv_op!(F64ConvertI64S, unwarp_i64, f64, i64, f64),
                    Opcode::F64ConvertI64U => conv_op!(F64ConvertI64U, unwarp_i64, f64, u64, f64),
                    Opcode::F32DemoteF64 => conv_op!(F32DemoteF64, unwarp_f64, f32, f32),
                    Opcode::F64PromoteF32 => conv_op!(F64PromoteF32, unwarp_f32, f64, f64),
                    Opcode::Wrap => conv_op!(Wrap, unwarp_i64, i32, i32),
                    Opcode::Bitcast => move_op!(Bitcast),
                    Opcode::ExtendS => {
                        let (d, s, from_ty) = decode_into!(ExtendS, pc, code_ptr);
                        let val = reg_val!(s).unwarp_i64();
                        let res = match from_ty {
                            EXTEND_TYPE_I8 => val as i8 as i64,
                            EXTEND_TYPE_I16 => val as i16 as i64,
                            EXTEND_TYPE_I32 => val as i32 as i64,
                            _ => unreachable!(),
                        };
                        *reg!(d) = InterpreterValue::i64(res);
                    }
                    Opcode::ExtendU => {
                        let (d, s, from_ty) = decode_into!(ExtendU, pc, code_ptr);
                        let val = reg_val!(s).unwarp_i64();
                        let res = match from_ty {
                            EXTEND_TYPE_I8 => (val as u8) as u64,
                            EXTEND_TYPE_I16 => (val as u16) as u64,
                            EXTEND_TYPE_I32 => (val as u32) as u64,
                            _ => unreachable!(),
                        };
                        *reg!(d) = InterpreterValue::i64(res as i64);
                    }

                    // --- Memory Access ---
                    Opcode::I32Load => load_op!(I32Load, i32, i32, 4),
                    Opcode::I64Load => load_op!(I64Load, i64, i64, 8),
                    Opcode::I8Load => load_op!(I8Load, i64, u8, 1),
                    Opcode::I16Load => load_op!(I16Load, i64, u16, 2),
                    Opcode::F32Load => load_op!(F32Load, f32, f32, 4),
                    Opcode::F64Load => load_op!(F64Load, f64, f64, 8),
                    Opcode::I32Store => store_op!(I32Store, unwarp_i32, i32, 4),
                    Opcode::I64Store => store_op!(I64Store, unwarp_i64, i64, 8),
                    Opcode::I8Store => store_op!(I8Store, unwarp_i64, u8, 1),
                    Opcode::I16Store => store_op!(I16Store, unwarp_i64, u16, 2),
                    Opcode::F32Store => store_op!(F32Store, unwarp_f32, f32, 4),
                    Opcode::F64Store => store_op!(F64Store, unwarp_f64, f64, 8),

                    // --- Stack Operations ---
                    Opcode::StackAddr => {
                        let (d, o) = decode_into!(StackAddr, pc, code_ptr);
                        let addr = stack_base + o as usize;
                        let ptr = unsafe { self.stack_memory.as_ptr().add(addr) };
                        *reg!(d) = InterpreterValue::i64(ptr as i64);
                    }
                    Opcode::StackLoad => {
                        let (d, o, ty) = decode_into!(StackLoad, pc, code_ptr);
                        let addr = stack_base + o as usize;
                        let ptr = unsafe { self.stack_memory.as_ptr().add(addr) };
                        *reg!(d) = match ty {
                            STACK_TYPE_I8 => {
                                InterpreterValue::i32(
                                    unsafe { (ptr as *const i8).read_unaligned() } as i32,
                                )
                            }
                            STACK_TYPE_I16 => InterpreterValue::i32(unsafe {
                                (ptr as *const i16).read_unaligned()
                            }
                                as i32),
                            STACK_TYPE_I32 => InterpreterValue::i32(unsafe {
                                (ptr as *const i32).read_unaligned()
                            }),
                            STACK_TYPE_I64 => InterpreterValue::i64(unsafe {
                                (ptr as *const i64).read_unaligned()
                            }),
                            STACK_TYPE_F32 => InterpreterValue::f32(unsafe {
                                (ptr as *const f32).read_unaligned()
                            }),
                            STACK_TYPE_F64 => InterpreterValue::f64(unsafe {
                                (ptr as *const f64).read_unaligned()
                            }),
                            _ => panic!("Unknown type {} in StackLoad", ty),
                        };
                    }
                    Opcode::StackStore => {
                        let (v, o, ty) = decode_into!(StackStore, pc, code_ptr);
                        let addr = stack_base + o as usize;
                        let ptr = unsafe { self.stack_memory.as_mut_ptr().add(addr) };
                        let val = reg_val!(v);
                        match ty {
                            STACK_TYPE_I8 => unsafe {
                                (ptr as *mut i8).write_unaligned(val.unwarp_i32() as i8)
                            },
                            STACK_TYPE_I16 => unsafe {
                                (ptr as *mut i16).write_unaligned(val.unwarp_i32() as i16)
                            },
                            STACK_TYPE_I32 => unsafe {
                                (ptr as *mut i32).write_unaligned(val.unwarp_i32())
                            },
                            STACK_TYPE_I64 => unsafe {
                                (ptr as *mut i64).write_unaligned(val.unwarp_i64())
                            },
                            STACK_TYPE_F32 => unsafe {
                                (ptr as *mut f32).write_unaligned(val.unwarp_f32())
                            },
                            STACK_TYPE_F64 => unsafe {
                                (ptr as *mut f64).write_unaligned(val.unwarp_f64())
                            },
                            _ => panic!("Unknown type {} in StackStore", ty),
                        }
                    }

                    // --- Control Flow & Pointers ---
                    Opcode::Jump => {
                        let target = decode_into!(Jump, pc, code_ptr);
                        pc = target as usize;
                    }
                    Opcode::BrIf => {
                        let (cond, target) = decode_into!(BrIf, pc, code_ptr);
                        if reg_val!(cond).unwarp_bool() {
                            pc = target as usize;
                        }
                    }
                    Opcode::BrTable => {
                        let (idx_reg, num_targets) = decode_into!(BrTable, pc, code_ptr);
                        let idx = reg_val!(idx_reg).unwarp_i32();

                        let target_idx = if idx >= 0 && (idx as u32) < (num_targets - 1) {
                            (idx as usize) + 1
                        } else {
                            0
                        };

                        let target_pc_pos = pc + target_idx * 4;
                        let target_pc = unsafe {
                            let ptr = code_ptr.add(target_pc_pos) as *const u32;
                            u32::from_le(ptr.read_unaligned())
                        };
                        pc = target_pc as usize;
                    }
                    Opcode::Return => {
                        let num_vals: u16 = read_u16!();
                        let mut return_vals = Vec::with_capacity(num_vals as usize);
                        for _ in 0..num_vals {
                            let reg = read_u16!();
                            return_vals.push(reg_val!(reg));
                        }

                        self.value_stack.truncate(base);
                        self.stack_memory.truncate(stack_base);

                        // Pop the previous frame first to get caller's context
                        let prev_frame = match self.frames.pop() {
                            Some(f) => f,
                            None => return return_vals,
                        };

                        // Restore caller's context
                        let dst_regs = prev_frame.dst_regs;
                        pc = prev_frame.pc;
                        base = prev_frame.base;
                        stack_base = prev_frame.stack_base;
                        func = prev_frame.func.clone();
                        mid = prev_frame.mid;

                        // Refresh the values pointer after stack operations
                        values_ptr = self.value_stack.as_mut_ptr();

                        // Write return values to caller's destination registers
                        for (i, &dst_reg) in dst_regs.iter().enumerate() {
                            if i < return_vals.len() {
                                *reg!(dst_reg) = return_vals[i];
                            }
                        }
                        continue 'main_loop;
                    }
                    Opcode::Call => {
                        let num_rets: u16 = read_u16!();
                        let f_id: u32 = unsafe {
                            let v = (code_ptr.add(pc) as *const u32).read_unaligned();
                            pc += 4;
                            u32::from_le(v)
                        };
                        let num_args: u16 = read_u16!();
                        let mut dst_regs = Vec::with_capacity(num_rets as usize);
                        for _ in 0..num_rets {
                            dst_regs.push(read_u16!());
                        }
                        self.args_buffer.clear();
                        for _ in 0..num_args {
                            let arg_reg = read_u16!();
                            self.args_buffer.push(reg_val!(arg_reg));
                        }

                        let target_fid = veloc_ir::FuncId::from_u32(f_id);
                        let (call_mid, call_fid) =
                            if let Some(link) = program.import_links.get(&(mid, target_fid)) {
                                match link {
                                    crate::host::ImportTarget::Module(m, f) => (*m, *f),
                                    crate::host::ImportTarget::Host(h_id) => {
                                        let host_func = &program.host_functions_list[*h_id];
                                        let res = host_func.call(&mut self.args_buffer);
                                        values_ptr = self.value_stack.as_mut_ptr();
                                        if let Some(&dst) = dst_regs.first() {
                                            if dst != 0 {
                                                *reg!(dst) = res;
                                            }
                                        }
                                        continue;
                                    }
                                }
                            } else {
                                (mid, target_fid)
                            };

                        prepare_call!(call_mid, call_fid, dst_regs, self.args_buffer);
                    }
                    Opcode::CallIndirect => {
                        let num_rets: u16 = read_u16!();
                        let ptr_reg: u16 = read_u16!();
                        let num_args: u16 = read_u16!();
                        let mut dst_regs = Vec::with_capacity(num_rets as usize);
                        for _ in 0..num_rets {
                            dst_regs.push(read_u16!());
                        }
                        self.args_buffer.clear();
                        for _ in 0..num_args {
                            let arg_reg = read_u16!();
                            self.args_buffer.push(reg_val!(arg_reg));
                        }
                        let ptr_val = reg_val!(ptr_reg).0 as usize;

                        let (call_mid, call_fid) = if let Some((target_mid, target_fid)) =
                            program.decode_interpreter_ptr(ptr_val)
                        {
                            (target_mid, target_fid)
                        } else if let Some(host_id) = program.decode_host_ptr(ptr_val) {
                            let host_func = &program.host_functions_list[host_id];
                            let res = host_func.call(&mut self.args_buffer);
                            values_ptr = self.value_stack.as_mut_ptr();
                            if let Some(&dst) = dst_regs.first() {
                                if dst != 0 {
                                    *reg!(dst) = res;
                                }
                            }
                            continue;
                        } else {
                            panic!("Invalid function pointer: {:x}", ptr_val);
                        };

                        prepare_call!(call_mid, call_fid, dst_regs, self.args_buffer);
                    }
                    Opcode::PtrIndex => {
                        let (d, p, i, s, o) = decode_into!(PtrIndex, pc, code_ptr);
                        let ptr = reg_val!(p).unwarp_i64();
                        let idx = reg_val!(i).unwarp_i64();
                        let res = ptr
                            .wrapping_add(idx.wrapping_mul(s as i64))
                            .wrapping_add(o as i64);
                        *reg!(d) = InterpreterValue::i64(res);
                    }
                    Opcode::Select => {
                        let (dst, cond, t, e) = decode_into!(Select, pc, code_ptr);
                        if reg_val!(cond).unwarp_bool() {
                            *reg!(dst) = reg_val!(t);
                        } else {
                            *reg!(dst) = reg_val!(e);
                        }
                    }
                    Opcode::RegMove => move_op!(RegMove),
                    Opcode::CallIntrinsic => {
                        let num_rets: u16 = read_u16!();
                        let intrinsic_id: u16 = read_u16!();
                        let num_args: u16 = read_u16!();
                        let mut dst_regs = Vec::with_capacity(num_rets as usize);
                        for _ in 0..num_rets {
                            dst_regs.push(read_u16!());
                        }
                        self.args_buffer.clear();
                        for _ in 0..num_args {
                            let arg_reg = read_u16!();
                            self.args_buffer.push(reg_val!(arg_reg));
                        }

                        let result = execute_intrinsic(intrinsic_id, &self.args_buffer);
                        values_ptr = self.value_stack.as_mut_ptr();
                        if let Some(&dst) = dst_regs.first() {
                            if dst != 0 {
                                *reg!(dst) = result;
                            }
                        }
                    }
                    Opcode::Unreachable => panic!("Unreachable code executed"),
                }
            }
            panic!("Function reached end without Return");
        }
    }
}

/// Execute an intrinsic function at runtime.
fn execute_intrinsic(id: u16, args: &[InterpreterValue]) -> InterpreterValue {
    use veloc_ir::intrinsic_ids::*;
    let f = |i: usize| args[i].unwarp_f32();
    let d = |i: usize| args[i].unwarp_f64();

    match Intrinsic::from_u16(id) {
        // Math
        SIN_F32 => InterpreterValue::f32(libm::sinf(f(0))),
        SIN_F64 => InterpreterValue::f64(libm::sin(d(0))),
        COS_F32 => InterpreterValue::f32(libm::cosf(f(0))),
        COS_F64 => InterpreterValue::f64(libm::cos(d(0))),
        POW_F32 => InterpreterValue::f32(libm::powf(f(0), f(1))),
        POW_F64 => InterpreterValue::f64(libm::pow(d(0), d(1))),
        EXP_F32 => InterpreterValue::f32(libm::expf(f(0))),
        EXP_F64 => InterpreterValue::f64(libm::exp(d(0))),
        LOG_F32 => InterpreterValue::f32(libm::logf(f(0))),
        LOG_F64 => InterpreterValue::f64(libm::log(d(0))),
        LOG2_F32 => InterpreterValue::f32(libm::log2f(f(0))),
        LOG2_F64 => InterpreterValue::f64(libm::log2(d(0))),
        LOG10_F32 => InterpreterValue::f32(libm::log10f(f(0))),
        LOG10_F64 => InterpreterValue::f64(libm::log10(d(0))),
        // Memory (stubs)
        MEMCPY | MEMMOVE | MEMSET => InterpreterValue::none(),
        MEMCMP => InterpreterValue::i32(0),
        // Sync (no-op in interpreter)
        FENCE | FENCE_ACQ | FENCE_REL | FENCE_SEQ => InterpreterValue::none(),
        // Debug
        ASSUME => InterpreterValue::none(),
        EXPECT => args[0],
        TRAP => panic!("trap"),
        _ => panic!("Unknown intrinsic: {}", id),
    }
}
