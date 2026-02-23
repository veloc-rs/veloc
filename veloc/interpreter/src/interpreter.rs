use crate::bytecode::{
    CompiledFunction, EXTEND_TYPE_I8, EXTEND_TYPE_I16, EXTEND_TYPE_I32, Opcode, STACK_TYPE_F32,
    STACK_TYPE_F64, STACK_TYPE_I8, STACK_TYPE_I16, STACK_TYPE_I32, STACK_TYPE_I64,
};
use crate::host::{ModuleId, Program};
use crate::value::InterpreterValue;
use ::alloc::vec::Vec;
use veloc_ir::Intrinsic;

/// 解释器执行错误（与 WASM 无关的通用错误类型）
#[derive(Debug, Clone)]
pub enum RunError {
    /// 内存访问越界
    OutOfBounds,
    /// 执行了 unreachable 指令
    Unreachable,
}

pub type Result<T> = core::result::Result<T, RunError>;

pub trait VirtualMemory {
    fn translate_addr(&self, logical_addr: usize, size: usize) -> Option<*mut u8>;
}

pub struct Interpreter {
    pub value_stack: Vec<InterpreterValue>,
    pub stack_memory: Vec<u8>,
    pub frames: Vec<StackFrame>,
    args_buffer: Vec<InterpreterValue>,
    dst_regs_buffer: Vec<u16>,
}

pub struct StackFrame {
    pub mid: ModuleId,
    pub func: ::alloc::sync::Arc<CompiledFunction>,
    pub pc: usize,
    pub base: usize,
    pub stack_base: usize,
    pub dst_regs_start: usize,
    pub dst_regs_count: usize,
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            value_stack: Vec::with_capacity(4096),
            stack_memory: Vec::with_capacity(1024 * 1024),
            frames: Vec::with_capacity(128),
            args_buffer: Vec::with_capacity(32),
            dst_regs_buffer: Vec::with_capacity(128),
        }
    }

    pub fn run_function(
        &mut self,
        program: &Program,
        mem: &dyn VirtualMemory,
        mid: ModuleId,
        fid: veloc_ir::FuncId,
        args: &[InterpreterValue],
    ) -> Result<Vec<InterpreterValue>> {
        let func = program.get_compiled_func(mid, fid);
        let base = self.value_stack.len();
        self.value_stack
            .resize(base + func.register_count, InterpreterValue::none());

        let stack_base = self.stack_memory.len();
        let total_stack_size: usize = func.stack_slots_sizes.iter().sum();
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
            dst_regs_start: 0,
            dst_regs_count: 0,
        });

        self.execute(program, mem)
    }

    #[inline(always)]
    fn execute(
        &mut self,
        program: &Program,
        mem: &dyn VirtualMemory,
    ) -> Result<Vec<InterpreterValue>> {
        let frame = self.frames.pop().unwrap();
        let mut pc = frame.pc;
        let mut base = frame.base;
        let mut stack_base = frame.stack_base;
        let mut func = frame.func.clone();
        let mut mid = frame.mid;

        'main_loop: loop {
            let code_len = func.code.len();
            let mut values_ptr = self.value_stack.as_mut_ptr();

            // === Core Register Access Helpers ===
            macro_rules! reg {
                ($r:expr) => {
                    &mut *values_ptr.add(base + $r as usize)
                };
            }
            macro_rules! get {
                ($r:expr) => {
                    *values_ptr.add(base + $r as usize)
                };
            }
            macro_rules! set {
                ($d:expr, $v:expr) => {
                    *reg!($d) = $v
                };
            }

            // Binary operation helper - takes a closure
            macro_rules! bin {
                ($d:expr, $l:expr, $r:expr, $op:expr) => {
                    set!($d, $op(get!($l), get!($r)))
                };
            }
            macro_rules! bin_imm {
                ($d:expr, $l:expr, $i:expr, $op:expr) => {
                    set!($d, $op(get!($l), $i))
                };
            }

            // Unary: dst = op(src)
            macro_rules! unary {
                ($d:expr, $s:expr, $op:expr) => {
                    set!($d, $op(get!($s)))
                };
            }
            // Compare: dst = src1 op src2 (boolean result)
            macro_rules! cmp {
                ($d:expr, $l:expr, $r:expr, $op:expr) => {
                    set!($d, InterpreterValue::bool($op(get!($l), get!($r))))
                };
            }
            // Compare with zero
            macro_rules! cmpz {
                ($d:expr, $s:expr, $ty:ident) => {
                    set!($d, InterpreterValue::bool(get!($s).$ty() == 0))
                };
            }
            // Convert: dst = convert(src)
            macro_rules! conv {
                ($d:expr, $s:expr, $op:expr) => {
                    set!($d, $op(get!($s)))
                };
            }
            // Memory load/store
            macro_rules! load {
                ($d:expr, $p:expr, $off:expr, $rd:ty, $wrap:expr) => {{
                    let addr = (get!($p).0 as usize).wrapping_add($off as usize);
                    match mem.translate_addr(addr, core::mem::size_of::<$rd>()) {
                        Some(ptr) => set!($d, $wrap((ptr as *const $rd).read_unaligned())),
                        None => return Err(RunError::OutOfBounds),
                    }
                }};
            }
            macro_rules! store {
                ($v:expr, $p:expr, $off:expr, $ty:ident, $wr:ty) => {{
                    let addr = (get!($p).0 as usize).wrapping_add($off as usize);
                    match mem.translate_addr(addr, core::mem::size_of::<$wr>()) {
                        Some(ptr) => (ptr as *mut $wr).write_unaligned(get!($v).$ty() as _),
                        None => return Err(RunError::OutOfBounds),
                    }
                }};
            }
            // Shift operations
            macro_rules! shift {
                ($d:expr, $l:expr, $r:expr, $op:expr) => {
                    set!($d, $op(get!($l), get!($r)))
                };
            }
            macro_rules! shift_imm {
                ($d:expr, $l:expr, $i:expr, $op:expr) => {
                    set!($d, $op(get!($l), $i))
                };
            }
            // Move
            macro_rules! mov {
                ($d:expr, $s:expr) => {
                    set!($d, get!($s))
                };
            }

            // Execute register moves from data section
            macro_rules! execute_moves {
                ($target:expr) => {{
                    let num_moves = $target.num_moves as usize;
                    if num_moves > 0 {
                        let moves_offset = $target.moves_offset as usize;
                        for i in 0..num_moves {
                            let dst = func.data_section.u16_data[moves_offset + i * 2];
                            let src = func.data_section.u16_data[moves_offset + i * 2 + 1];
                            set!(dst, get!(src));
                        }
                    }
                }};
            }

            // === Call Preparation Macro ===
            macro_rules! prepare_call {
                ($target_mid:expr, $target_fid:expr, $dst_regs_start:expr, $dst_regs_count:expr, $args:expr) => {{
                    if program.compiled_modules[$target_mid.0][$target_fid.0 as usize].is_none() {
                        panic!(
                            "Calling uncompiled function: mid={:?}, fid={:?}",
                            $target_mid, $target_fid
                        );
                    }
                    let next_func = program.get_compiled_func($target_mid, $target_fid);
                    let saved_pc = pc;
                    self.frames.push(StackFrame {
                        mid,
                        func: func.clone(),
                        pc: saved_pc,
                        base,
                        stack_base,
                        dst_regs_start: $dst_regs_start,
                        dst_regs_count: $dst_regs_count,
                    });
                    mid = $target_mid;
                    func = next_func;
                    pc = 0;
                    base = self.value_stack.len();
                    self.value_stack
                        .resize(base + func.register_count, InterpreterValue::none());
                    stack_base = self.stack_memory.len();
                    let total_size: usize = func.stack_slots_sizes.iter().sum();
                    self.stack_memory.resize(stack_base + total_size, 0);
                    for (i, &new_idx) in func.param_indices.iter().enumerate() {
                        if i < $args.len() {
                            self.value_stack[base + new_idx as usize] = $args[i];
                        }
                    }
                    continue 'main_loop;
                }};
            }

            // === Read call data from data section ===
            // Layout: [ret_count: u16, arg_count: u16, ret_regs..., arg_regs...]
            macro_rules! read_call_data {
                ($data_sec:expr, $off:expr) => {{
                    let rets = $data_sec.u16_data[$off];
                    let args = $data_sec.u16_data[$off + 1];
                    let dst_start = self.dst_regs_buffer.len();
                    for i in 0..rets {
                        self.dst_regs_buffer
                            .push($data_sec.u16_data[$off + 2 + i as usize]);
                    }
                    self.args_buffer.clear();
                    for i in 0..args {
                        let reg = $data_sec.u16_data[$off + 2 + rets as usize + i as usize];
                        self.args_buffer.push(get!(reg));
                    }
                    (rets, args, dst_start)
                }};
            }

            while pc < code_len {
                unsafe {
                    let inst = &func.code[pc];
                    pc += 1;
                    let opcode: Opcode = core::mem::transmute(inst.opcode);

                    match opcode {
                        // === Constants ===
                        Opcode::Iconst => set!(inst.dst, InterpreterValue::i64(inst.imm64 as i64)),
                        Opcode::Fconst => set!(inst.dst, InterpreterValue(inst.imm64)),
                        Opcode::Bconst => set!(inst.dst, InterpreterValue::bool(inst.src2 != 0)),
                        Opcode::Vconst => {
                            todo!("Vector constants")
                        }

                        // === I32 Arithmetic ===
                        Opcode::I32Add => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i32(
                                a.unwarp_i32().wrapping_add(b.unwarp_i32())
                            )
                        ),
                        Opcode::I32AddImm => bin_imm!(
                            inst.dst,
                            inst.src1,
                            inst.imm32() as i32,
                            |a: InterpreterValue, b: i32| InterpreterValue::i32(
                                a.unwarp_i32().wrapping_add(b)
                            )
                        ),
                        Opcode::I32Sub => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i32(
                                a.unwarp_i32().wrapping_sub(b.unwarp_i32())
                            )
                        ),
                        Opcode::I32SubImm => bin_imm!(
                            inst.dst,
                            inst.src1,
                            inst.imm32() as i32,
                            |a: InterpreterValue, b: i32| InterpreterValue::i32(
                                a.unwarp_i32().wrapping_sub(b)
                            )
                        ),
                        Opcode::I32Mul => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i32(
                                a.unwarp_i32().wrapping_mul(b.unwarp_i32())
                            )
                        ),
                        Opcode::I32DivS => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i32(
                                a.unwarp_i32().wrapping_div(b.unwarp_i32())
                            )
                        ),
                        Opcode::I32DivU => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i32(
                                (a.unwarp_i32() as u32).wrapping_div(b.unwarp_i32() as u32) as i32
                            )
                        ),
                        Opcode::I32RemS => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i32(
                                a.unwarp_i32().wrapping_rem(b.unwarp_i32())
                            )
                        ),
                        Opcode::I32RemU => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i32(
                                (a.unwarp_i32() as u32).wrapping_rem(b.unwarp_i32() as u32) as i32
                            )
                        ),
                        Opcode::I32And => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i32(
                                a.unwarp_i32() & b.unwarp_i32()
                            )
                        ),
                        Opcode::I32AndImm => bin_imm!(
                            inst.dst,
                            inst.src1,
                            inst.imm32() as i32,
                            |a: InterpreterValue, b: i32| InterpreterValue::i32(a.unwarp_i32() & b)
                        ),
                        Opcode::I32Or => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i32(
                                a.unwarp_i32() | b.unwarp_i32()
                            )
                        ),
                        Opcode::I32OrImm => bin_imm!(
                            inst.dst,
                            inst.src1,
                            inst.imm32() as i32,
                            |a: InterpreterValue, b: i32| InterpreterValue::i32(a.unwarp_i32() | b)
                        ),
                        Opcode::I32Xor => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i32(
                                a.unwarp_i32() ^ b.unwarp_i32()
                            )
                        ),
                        Opcode::I32XorImm => bin_imm!(
                            inst.dst,
                            inst.src1,
                            inst.imm32() as i32,
                            |a: InterpreterValue, b: i32| InterpreterValue::i32(a.unwarp_i32() ^ b)
                        ),
                        Opcode::I32Shl => shift!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i32(
                                a.unwarp_i32().wrapping_shl(b.unwarp_i32() as u32)
                            )
                        ),
                        Opcode::I32ShlImm => shift_imm!(
                            inst.dst,
                            inst.src1,
                            inst.imm32() as u32,
                            |a: InterpreterValue, b: u32| InterpreterValue::i32(
                                a.unwarp_i32().wrapping_shl(b)
                            )
                        ),
                        Opcode::I32ShrS => shift!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i32(
                                a.unwarp_i32().wrapping_shr(b.unwarp_i32() as u32)
                            )
                        ),
                        Opcode::I32ShrSImm => shift_imm!(
                            inst.dst,
                            inst.src1,
                            inst.imm32() as u32,
                            |a: InterpreterValue, b: u32| InterpreterValue::i32(
                                a.unwarp_i32().wrapping_shr(b)
                            )
                        ),
                        Opcode::I32ShrU => shift!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i32(
                                (a.unwarp_i32() as u32).wrapping_shr(b.unwarp_i32() as u32) as i32
                            )
                        ),
                        Opcode::I32ShrUImm => shift_imm!(
                            inst.dst,
                            inst.src1,
                            inst.imm32() as u32,
                            |a: InterpreterValue, b: u32| InterpreterValue::i32(
                                (a.unwarp_i32() as u32).wrapping_shr(b) as i32
                            )
                        ),
                        Opcode::I32RotL => shift!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i32(
                                a.unwarp_i32().rotate_left(b.unwarp_i32() as u32)
                            )
                        ),
                        Opcode::I32RotR => shift!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i32(
                                a.unwarp_i32().rotate_right(b.unwarp_i32() as u32)
                            )
                        ),
                        Opcode::I32Clz => unary!(inst.dst, inst.src1, |a: InterpreterValue| {
                            InterpreterValue::i32(a.unwarp_i32().leading_zeros() as i32)
                        }),
                        Opcode::I32Ctz => unary!(inst.dst, inst.src1, |a: InterpreterValue| {
                            InterpreterValue::i32(a.unwarp_i32().trailing_zeros() as i32)
                        }),
                        Opcode::I32Popcnt => unary!(inst.dst, inst.src1, |a: InterpreterValue| {
                            InterpreterValue::i32(a.unwarp_i32().count_ones() as i32)
                        }),
                        Opcode::I32Eqz => cmpz!(inst.dst, inst.src1, unwarp_i32),
                        Opcode::I32Eq => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| a.unwarp_i32()
                                == b.unwarp_i32()
                        ),
                        Opcode::I32Ne => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| a.unwarp_i32()
                                != b.unwarp_i32()
                        ),
                        Opcode::I32LtS => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| a.unwarp_i32()
                                < b.unwarp_i32()
                        ),
                        Opcode::I32LtU => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| (a.unwarp_i32() as u32)
                                < (b.unwarp_i32() as u32)
                        ),
                        Opcode::I32LeS => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| a.unwarp_i32()
                                <= b.unwarp_i32()
                        ),
                        Opcode::I32LeU => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| (a.unwarp_i32() as u32)
                                <= (b.unwarp_i32() as u32)
                        ),
                        Opcode::I32GtS => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| a.unwarp_i32()
                                > b.unwarp_i32()
                        ),
                        Opcode::I32GtU => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| (a.unwarp_i32() as u32)
                                > (b.unwarp_i32() as u32)
                        ),
                        Opcode::I32GeS => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| a.unwarp_i32()
                                >= b.unwarp_i32()
                        ),
                        Opcode::I32GeU => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| (a.unwarp_i32() as u32)
                                >= (b.unwarp_i32() as u32)
                        ),

                        // === I64 Operations ===
                        Opcode::I64Add => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i64(
                                a.unwarp_i64().wrapping_add(b.unwarp_i64())
                            )
                        ),
                        Opcode::I64AddImm => bin_imm!(
                            inst.dst,
                            inst.src1,
                            inst.imm64 as i64,
                            |a: InterpreterValue, b: i64| InterpreterValue::i64(
                                a.unwarp_i64().wrapping_add(b)
                            )
                        ),
                        Opcode::I64Sub => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i64(
                                a.unwarp_i64().wrapping_sub(b.unwarp_i64())
                            )
                        ),
                        Opcode::I64SubImm => bin_imm!(
                            inst.dst,
                            inst.src1,
                            inst.imm64 as i64,
                            |a: InterpreterValue, b: i64| InterpreterValue::i64(
                                a.unwarp_i64().wrapping_sub(b)
                            )
                        ),
                        Opcode::I64Mul => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i64(
                                a.unwarp_i64().wrapping_mul(b.unwarp_i64())
                            )
                        ),
                        Opcode::I64DivS => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i64(
                                a.unwarp_i64().wrapping_div(b.unwarp_i64())
                            )
                        ),
                        Opcode::I64DivU => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i64(
                                (a.unwarp_i64() as u64).wrapping_div(b.unwarp_i64() as u64) as i64
                            )
                        ),
                        Opcode::I64RemS => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i64(
                                a.unwarp_i64().wrapping_rem(b.unwarp_i64())
                            )
                        ),
                        Opcode::I64RemU => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i64(
                                (a.unwarp_i64() as u64).wrapping_rem(b.unwarp_i64() as u64) as i64
                            )
                        ),
                        Opcode::I64And => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i64(
                                a.unwarp_i64() & b.unwarp_i64()
                            )
                        ),
                        Opcode::I64AndImm => bin_imm!(
                            inst.dst,
                            inst.src1,
                            inst.imm64 as i64,
                            |a: InterpreterValue, b: i64| InterpreterValue::i64(a.unwarp_i64() & b)
                        ),
                        Opcode::I64Or => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i64(
                                a.unwarp_i64() | b.unwarp_i64()
                            )
                        ),
                        Opcode::I64OrImm => bin_imm!(
                            inst.dst,
                            inst.src1,
                            inst.imm64 as i64,
                            |a: InterpreterValue, b: i64| InterpreterValue::i64(a.unwarp_i64() | b)
                        ),
                        Opcode::I64Xor => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i64(
                                a.unwarp_i64() ^ b.unwarp_i64()
                            )
                        ),
                        Opcode::I64XorImm => bin_imm!(
                            inst.dst,
                            inst.src1,
                            inst.imm64 as i64,
                            |a: InterpreterValue, b: i64| InterpreterValue::i64(a.unwarp_i64() ^ b)
                        ),
                        Opcode::I64Shl => shift!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i64(
                                a.unwarp_i64().wrapping_shl(b.unwarp_i64() as u32)
                            )
                        ),
                        Opcode::I64ShlImm => shift_imm!(
                            inst.dst,
                            inst.src1,
                            inst.imm64 as u32,
                            |a: InterpreterValue, b: u32| InterpreterValue::i64(
                                a.unwarp_i64().wrapping_shl(b)
                            )
                        ),
                        Opcode::I64ShrS => shift!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i64(
                                a.unwarp_i64().wrapping_shr(b.unwarp_i64() as u32)
                            )
                        ),
                        Opcode::I64ShrSImm => shift_imm!(
                            inst.dst,
                            inst.src1,
                            inst.imm64 as u32,
                            |a: InterpreterValue, b: u32| InterpreterValue::i64(
                                a.unwarp_i64().wrapping_shr(b)
                            )
                        ),
                        Opcode::I64ShrU => shift!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i64(
                                (a.unwarp_i64() as u64).wrapping_shr(b.unwarp_i64() as u32) as i64
                            )
                        ),
                        Opcode::I64ShrUImm => shift_imm!(
                            inst.dst,
                            inst.src1,
                            inst.imm64 as u32,
                            |a: InterpreterValue, b: u32| InterpreterValue::i64(
                                (a.unwarp_i64() as u64).wrapping_shr(b) as i64
                            )
                        ),
                        Opcode::I64RotL => shift!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i64(
                                a.unwarp_i64().rotate_left(b.unwarp_i64() as u32)
                            )
                        ),
                        Opcode::I64RotR => shift!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::i64(
                                a.unwarp_i64().rotate_right(b.unwarp_i64() as u32)
                            )
                        ),
                        Opcode::I64Clz => unary!(inst.dst, inst.src1, |a: InterpreterValue| {
                            InterpreterValue::i64(a.unwarp_i64().leading_zeros() as i64)
                        }),
                        Opcode::I64Ctz => unary!(inst.dst, inst.src1, |a: InterpreterValue| {
                            InterpreterValue::i64(a.unwarp_i64().trailing_zeros() as i64)
                        }),
                        Opcode::I64Popcnt => unary!(inst.dst, inst.src1, |a: InterpreterValue| {
                            InterpreterValue::i64(a.unwarp_i64().count_ones() as i64)
                        }),
                        Opcode::I64Eqz => cmpz!(inst.dst, inst.src1, unwarp_i64),
                        Opcode::I64Eq => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| a.unwarp_i64()
                                == b.unwarp_i64()
                        ),
                        Opcode::I64Ne => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| a.unwarp_i64()
                                != b.unwarp_i64()
                        ),
                        Opcode::I64LtS => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| a.unwarp_i64()
                                < b.unwarp_i64()
                        ),
                        Opcode::I64LtU => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| (a.unwarp_i64() as u64)
                                < (b.unwarp_i64() as u64)
                        ),
                        Opcode::I64LeS => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| a.unwarp_i64()
                                <= b.unwarp_i64()
                        ),
                        Opcode::I64LeU => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| (a.unwarp_i64() as u64)
                                <= (b.unwarp_i64() as u64)
                        ),
                        Opcode::I64GtS => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| a.unwarp_i64()
                                > b.unwarp_i64()
                        ),
                        Opcode::I64GtU => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| (a.unwarp_i64() as u64)
                                > (b.unwarp_i64() as u64)
                        ),
                        Opcode::I64GeS => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| a.unwarp_i64()
                                >= b.unwarp_i64()
                        ),
                        Opcode::I64GeU => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| (a.unwarp_i64() as u64)
                                >= (b.unwarp_i64() as u64)
                        ),

                        // === F32 Operations ===
                        Opcode::F32Add => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::f32(
                                a.unwarp_f32() + b.unwarp_f32()
                            )
                        ),
                        Opcode::F32Sub => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::f32(
                                a.unwarp_f32() - b.unwarp_f32()
                            )
                        ),
                        Opcode::F32Mul => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::f32(
                                a.unwarp_f32() * b.unwarp_f32()
                            )
                        ),
                        Opcode::F32Div => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::f32(
                                a.unwarp_f32() / b.unwarp_f32()
                            )
                        ),
                        Opcode::F32Abs => unary!(inst.dst, inst.src1, |a: InterpreterValue| {
                            InterpreterValue::f32(a.unwarp_f32().abs())
                        }),
                        Opcode::F32Neg => unary!(inst.dst, inst.src1, |a: InterpreterValue| {
                            InterpreterValue::f32(-a.unwarp_f32())
                        }),
                        Opcode::F32Sqrt => unary!(inst.dst, inst.src1, |a: InterpreterValue| {
                            InterpreterValue::f32(a.unwarp_f32().sqrt())
                        }),
                        Opcode::F32Ceil => unary!(inst.dst, inst.src1, |a: InterpreterValue| {
                            InterpreterValue::f32(a.unwarp_f32().ceil())
                        }),
                        Opcode::F32Floor => unary!(inst.dst, inst.src1, |a: InterpreterValue| {
                            InterpreterValue::f32(a.unwarp_f32().floor())
                        }),
                        Opcode::F32Trunc => unary!(inst.dst, inst.src1, |a: InterpreterValue| {
                            InterpreterValue::f32(a.unwarp_f32().trunc())
                        }),
                        Opcode::F32Nearest => unary!(inst.dst, inst.src1, |a: InterpreterValue| {
                            InterpreterValue::f32(a.unwarp_f32().round_ties_even())
                        }),
                        Opcode::F32Min => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::f32(
                                a.unwarp_f32().min(b.unwarp_f32())
                            )
                        ),
                        Opcode::F32Max => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::f32(
                                a.unwarp_f32().max(b.unwarp_f32())
                            )
                        ),
                        Opcode::F32CopySign => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::f32(
                                a.unwarp_f32().copysign(b.unwarp_f32())
                            )
                        ),
                        Opcode::F32Eq => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| a.unwarp_f32()
                                == b.unwarp_f32()
                        ),
                        Opcode::F32Ne => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| a.unwarp_f32()
                                != b.unwarp_f32()
                        ),
                        Opcode::F32Lt => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| a.unwarp_f32()
                                < b.unwarp_f32()
                        ),
                        Opcode::F32Le => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| a.unwarp_f32()
                                <= b.unwarp_f32()
                        ),
                        Opcode::F32Gt => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| a.unwarp_f32()
                                > b.unwarp_f32()
                        ),
                        Opcode::F32Ge => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| a.unwarp_f32()
                                >= b.unwarp_f32()
                        ),

                        // === F64 Operations ===
                        Opcode::F64Add => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::f64(
                                a.unwarp_f64() + b.unwarp_f64()
                            )
                        ),
                        Opcode::F64Sub => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::f64(
                                a.unwarp_f64() - b.unwarp_f64()
                            )
                        ),
                        Opcode::F64Mul => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::f64(
                                a.unwarp_f64() * b.unwarp_f64()
                            )
                        ),
                        Opcode::F64Div => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::f64(
                                a.unwarp_f64() / b.unwarp_f64()
                            )
                        ),
                        Opcode::F64Abs => unary!(inst.dst, inst.src1, |a: InterpreterValue| {
                            InterpreterValue::f64(a.unwarp_f64().abs())
                        }),
                        Opcode::F64Neg => unary!(inst.dst, inst.src1, |a: InterpreterValue| {
                            InterpreterValue::f64(-a.unwarp_f64())
                        }),
                        Opcode::F64Sqrt => unary!(inst.dst, inst.src1, |a: InterpreterValue| {
                            InterpreterValue::f64(a.unwarp_f64().sqrt())
                        }),
                        Opcode::F64Ceil => unary!(inst.dst, inst.src1, |a: InterpreterValue| {
                            InterpreterValue::f64(a.unwarp_f64().ceil())
                        }),
                        Opcode::F64Floor => unary!(inst.dst, inst.src1, |a: InterpreterValue| {
                            InterpreterValue::f64(a.unwarp_f64().floor())
                        }),
                        Opcode::F64Trunc => unary!(inst.dst, inst.src1, |a: InterpreterValue| {
                            InterpreterValue::f64(a.unwarp_f64().trunc())
                        }),
                        Opcode::F64Nearest => unary!(inst.dst, inst.src1, |a: InterpreterValue| {
                            InterpreterValue::f64(a.unwarp_f64().round_ties_even())
                        }),
                        Opcode::F64Min => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::f64(
                                a.unwarp_f64().min(b.unwarp_f64())
                            )
                        ),
                        Opcode::F64Max => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::f64(
                                a.unwarp_f64().max(b.unwarp_f64())
                            )
                        ),
                        Opcode::F64CopySign => bin!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| InterpreterValue::f64(
                                a.unwarp_f64().copysign(b.unwarp_f64())
                            )
                        ),
                        Opcode::F64Eq => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| a.unwarp_f64()
                                == b.unwarp_f64()
                        ),
                        Opcode::F64Ne => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| a.unwarp_f64()
                                != b.unwarp_f64()
                        ),
                        Opcode::F64Lt => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| a.unwarp_f64()
                                < b.unwarp_f64()
                        ),
                        Opcode::F64Le => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| a.unwarp_f64()
                                <= b.unwarp_f64()
                        ),
                        Opcode::F64Gt => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| a.unwarp_f64()
                                > b.unwarp_f64()
                        ),
                        Opcode::F64Ge => cmp!(
                            inst.dst,
                            inst.src1,
                            inst.src2,
                            |a: InterpreterValue, b: InterpreterValue| a.unwarp_f64()
                                >= b.unwarp_f64()
                        ),

                        // === Conversions ===
                        Opcode::I32TruncF32S => {
                            conv!(inst.dst, inst.src1, |a: InterpreterValue| {
                                InterpreterValue::i32(a.unwarp_f32() as i32)
                            })
                        }
                        Opcode::I32TruncF32U => {
                            conv!(inst.dst, inst.src1, |a: InterpreterValue| {
                                InterpreterValue::i32(a.unwarp_f32() as u32 as i32)
                            })
                        }
                        Opcode::I32TruncF64S => {
                            conv!(inst.dst, inst.src1, |a: InterpreterValue| {
                                InterpreterValue::i32(a.unwarp_f64() as i32)
                            })
                        }
                        Opcode::I32TruncF64U => {
                            conv!(inst.dst, inst.src1, |a: InterpreterValue| {
                                InterpreterValue::i32(a.unwarp_f64() as u32 as i32)
                            })
                        }
                        Opcode::I64TruncF32S => {
                            conv!(inst.dst, inst.src1, |a: InterpreterValue| {
                                InterpreterValue::i64(a.unwarp_f32() as i64)
                            })
                        }
                        Opcode::I64TruncF32U => {
                            conv!(inst.dst, inst.src1, |a: InterpreterValue| {
                                InterpreterValue::i64(a.unwarp_f32() as u64 as i64)
                            })
                        }
                        Opcode::I64TruncF64S => {
                            conv!(inst.dst, inst.src1, |a: InterpreterValue| {
                                InterpreterValue::i64(a.unwarp_f64() as i64)
                            })
                        }
                        Opcode::I64TruncF64U => {
                            conv!(inst.dst, inst.src1, |a: InterpreterValue| {
                                InterpreterValue::i64(a.unwarp_f64() as u64 as i64)
                            })
                        }
                        Opcode::I32TruncSatF32S => {
                            let val = get!(inst.src1).unwarp_f32();
                            set!(
                                inst.dst,
                                InterpreterValue::i32(if val.is_nan() { 0 } else { val as i32 })
                            );
                        }
                        Opcode::I32TruncSatF32U => {
                            let val = get!(inst.src1).unwarp_f32();
                            set!(
                                inst.dst,
                                InterpreterValue::i32(if val.is_nan() || val < 0.0 {
                                    0
                                } else {
                                    val as u32
                                } as i32)
                            );
                        }
                        Opcode::I32TruncSatF64S => {
                            let val = get!(inst.src1).unwarp_f64();
                            set!(
                                inst.dst,
                                InterpreterValue::i32(if val.is_nan() { 0 } else { val as i32 })
                            );
                        }
                        Opcode::I32TruncSatF64U => {
                            let val = get!(inst.src1).unwarp_f64();
                            set!(
                                inst.dst,
                                InterpreterValue::i32(if val.is_nan() || val < 0.0 {
                                    0
                                } else {
                                    val as u32
                                } as i32)
                            );
                        }
                        Opcode::I64TruncSatF32S => {
                            let val = get!(inst.src1).unwarp_f32();
                            set!(
                                inst.dst,
                                InterpreterValue::i64(if val.is_nan() { 0 } else { val as i64 })
                            );
                        }
                        Opcode::I64TruncSatF32U => {
                            let val = get!(inst.src1).unwarp_f32();
                            set!(
                                inst.dst,
                                InterpreterValue::i64(if val.is_nan() || val < 0.0 {
                                    0
                                } else {
                                    val as u64
                                } as i64)
                            );
                        }
                        Opcode::I64TruncSatF64S => {
                            let val = get!(inst.src1).unwarp_f64();
                            set!(
                                inst.dst,
                                InterpreterValue::i64(if val.is_nan() { 0 } else { val as i64 })
                            );
                        }
                        Opcode::I64TruncSatF64U => {
                            let val = get!(inst.src1).unwarp_f64();
                            set!(
                                inst.dst,
                                InterpreterValue::i64(if val.is_nan() || val < 0.0 {
                                    0
                                } else {
                                    val as u64
                                } as i64)
                            );
                        }
                        Opcode::F32ConvertI32S => {
                            conv!(inst.dst, inst.src1, |a: InterpreterValue| {
                                InterpreterValue::f32(a.unwarp_i32() as f32)
                            })
                        }
                        Opcode::F32ConvertI32U => {
                            conv!(inst.dst, inst.src1, |a: InterpreterValue| {
                                InterpreterValue::f32(a.unwarp_i32() as u32 as f32)
                            })
                        }
                        Opcode::F32ConvertI64S => {
                            conv!(inst.dst, inst.src1, |a: InterpreterValue| {
                                InterpreterValue::f32(a.unwarp_i64() as f32)
                            })
                        }
                        Opcode::F32ConvertI64U => {
                            conv!(inst.dst, inst.src1, |a: InterpreterValue| {
                                InterpreterValue::f32(a.unwarp_i64() as u64 as f32)
                            })
                        }
                        Opcode::F64ConvertI32S => {
                            conv!(inst.dst, inst.src1, |a: InterpreterValue| {
                                InterpreterValue::f64(a.unwarp_i32() as f64)
                            })
                        }
                        Opcode::F64ConvertI32U => {
                            conv!(inst.dst, inst.src1, |a: InterpreterValue| {
                                InterpreterValue::f64(a.unwarp_i32() as u32 as f64)
                            })
                        }
                        Opcode::F64ConvertI64S => {
                            conv!(inst.dst, inst.src1, |a: InterpreterValue| {
                                InterpreterValue::f64(a.unwarp_i64() as f64)
                            })
                        }
                        Opcode::F64ConvertI64U => {
                            conv!(inst.dst, inst.src1, |a: InterpreterValue| {
                                InterpreterValue::f64(a.unwarp_i64() as u64 as f64)
                            })
                        }
                        Opcode::F32DemoteF64 => {
                            conv!(inst.dst, inst.src1, |a: InterpreterValue| {
                                InterpreterValue::f32(a.unwarp_f64() as f32)
                            })
                        }
                        Opcode::F64PromoteF32 => {
                            conv!(inst.dst, inst.src1, |a: InterpreterValue| {
                                InterpreterValue::f64(a.unwarp_f32() as f64)
                            })
                        }
                        Opcode::Wrap => conv!(inst.dst, inst.src1, |a: InterpreterValue| {
                            InterpreterValue::i32(a.unwarp_i64() as i32)
                        }),
                        Opcode::Bitcast => mov!(inst.dst, inst.src1),
                        Opcode::ExtendS => {
                            let val = get!(inst.src1).unwarp_i64();
                            set!(
                                inst.dst,
                                InterpreterValue::i64(match inst.src2 as u8 {
                                    EXTEND_TYPE_I8 => val as i8 as i64,
                                    EXTEND_TYPE_I16 => val as i16 as i64,
                                    EXTEND_TYPE_I32 => val as i32 as i64,
                                    _ => unreachable!(),
                                })
                            );
                        }
                        Opcode::ExtendU => {
                            let val = get!(inst.src1).unwarp_i64();
                            set!(
                                inst.dst,
                                InterpreterValue::i64(match inst.src2 as u8 {
                                    EXTEND_TYPE_I8 => (val as u8) as u64,
                                    EXTEND_TYPE_I16 => (val as u16) as u64,
                                    EXTEND_TYPE_I32 => (val as u32) as u64,
                                    _ => unreachable!(),
                                } as i64)
                            );
                        }

                        // === Memory Access ===
                        Opcode::I32Load => load!(inst.dst, inst.src1, inst.imm32(), i32, |v| {
                            InterpreterValue::i32(v)
                        }),
                        Opcode::I64Load => load!(inst.dst, inst.src1, inst.imm32(), i64, |v| {
                            InterpreterValue::i64(v)
                        }),
                        Opcode::I8Load => load!(inst.dst, inst.src1, inst.imm32(), u8, |v| {
                            InterpreterValue::i64(v as i64)
                        }),
                        Opcode::I16Load => load!(inst.dst, inst.src1, inst.imm32(), u16, |v| {
                            InterpreterValue::i64(v as i64)
                        }),
                        Opcode::F32Load => load!(inst.dst, inst.src1, inst.imm32(), f32, |v| {
                            InterpreterValue::f32(v)
                        }),
                        Opcode::F64Load => load!(inst.dst, inst.src1, inst.imm32(), f64, |v| {
                            InterpreterValue::f64(v)
                        }),
                        Opcode::I32Store => {
                            store!(inst.src1, inst.src2, inst.imm32(), unwarp_i32, i32)
                        }
                        Opcode::I64Store => {
                            store!(inst.src1, inst.src2, inst.imm32(), unwarp_i64, i64)
                        }
                        Opcode::I8Store => {
                            store!(inst.src1, inst.src2, inst.imm32(), unwarp_i64, u8)
                        }
                        Opcode::I16Store => {
                            store!(inst.src1, inst.src2, inst.imm32(), unwarp_i64, u16)
                        }
                        Opcode::F32Store => {
                            store!(inst.src1, inst.src2, inst.imm32(), unwarp_f32, f32)
                        }
                        Opcode::F64Store => {
                            store!(inst.src1, inst.src2, inst.imm32(), unwarp_f64, f64)
                        }

                        // === Stack Operations ===
                        Opcode::StackAddr => {
                            let ptr = self
                                .stack_memory
                                .as_ptr()
                                .add(stack_base + inst.imm32() as usize);
                            set!(inst.dst, InterpreterValue::i64(ptr as i64));
                        }
                        Opcode::StackLoad => {
                            let addr = stack_base + inst.imm32() as usize;
                            let ptr = self.stack_memory.as_ptr().add(addr);
                            set!(
                                inst.dst,
                                match inst.src2 as u8 {
                                    STACK_TYPE_I8 => InterpreterValue::i32(
                                        (ptr as *const i8).read_unaligned() as i32,
                                    ),
                                    STACK_TYPE_I16 => InterpreterValue::i32(
                                        (ptr as *const i16).read_unaligned() as i32,
                                    ),
                                    STACK_TYPE_I32 => {
                                        InterpreterValue::i32((ptr as *const i32).read_unaligned())
                                    }
                                    STACK_TYPE_I64 => {
                                        InterpreterValue::i64((ptr as *const i64).read_unaligned())
                                    }
                                    STACK_TYPE_F32 => {
                                        InterpreterValue::f32((ptr as *const f32).read_unaligned())
                                    }
                                    STACK_TYPE_F64 => {
                                        InterpreterValue::f64((ptr as *const f64).read_unaligned())
                                    }
                                    _ => panic!("Unknown type {} in StackLoad", inst.src2),
                                }
                            );
                        }
                        Opcode::StackStore => {
                            let addr = stack_base + inst.imm32() as usize;
                            let ptr = self.stack_memory.as_mut_ptr().add(addr);
                            let val = get!(inst.src1);
                            match inst.src2 as u8 {
                                STACK_TYPE_I8 => {
                                    (ptr as *mut i8).write_unaligned(val.unwarp_i32() as i8)
                                }
                                STACK_TYPE_I16 => {
                                    (ptr as *mut i16).write_unaligned(val.unwarp_i32() as i16)
                                }
                                STACK_TYPE_I32 => {
                                    (ptr as *mut i32).write_unaligned(val.unwarp_i32())
                                }
                                STACK_TYPE_I64 => {
                                    (ptr as *mut i64).write_unaligned(val.unwarp_i64())
                                }
                                STACK_TYPE_F32 => {
                                    (ptr as *mut f32).write_unaligned(val.unwarp_f32())
                                }
                                STACK_TYPE_F64 => {
                                    (ptr as *mut f64).write_unaligned(val.unwarp_f64())
                                }
                                _ => panic!("Unknown type {} in StackStore", inst.src2),
                            }
                        }

                        // === Control Flow ===
                        Opcode::Jump => pc = inst.imm32() as usize,
                        Opcode::JumpWithMoves => {
                            let target = &func.data_section.jump_targets[inst.imm32() as usize];
                            execute_moves!(target);
                            pc = target.pc as usize;
                        }
                        Opcode::Br => {
                            let cond = get!(inst.dst).unwarp_bool();
                            let target_idx = if cond { inst.imm32() } else { inst.aux() };
                            let target = &func.data_section.jump_targets[target_idx as usize];
                            execute_moves!(target);
                            pc = target.pc as usize;
                        }
                        Opcode::BrTable => {
                            let idx = get!(inst.dst).unwarp_i32();
                            let num = inst.aux() as usize;
                            let target_idx = if idx >= 0 && (idx as usize) < num {
                                idx as usize
                            } else {
                                num - 1
                            };

                            let target =
                                &func.data_section.jump_targets[inst.imm32() as usize + target_idx];
                            execute_moves!(target);
                            pc = target.pc as usize;
                        }
                        Opcode::Return => {
                            let data_off = inst.imm32() as usize;
                            let num_vals = inst.aux() as usize;
                            let cur_base = base;
                            let cur_stack = stack_base;

                            if self.frames.is_empty() {
                                let mut res = Vec::with_capacity(num_vals);
                                for i in 0..num_vals {
                                    res.push(get!(func.data_section.u16_data[data_off + 1 + i]));
                                }
                                self.value_stack.truncate(cur_base);
                                self.stack_memory.truncate(cur_stack);
                                return Ok(res);
                            }

                            self.args_buffer.clear();
                            for i in 0..num_vals {
                                self.args_buffer
                                    .push(get!(func.data_section.u16_data[data_off + 1 + i]));
                            }
                            self.value_stack.truncate(cur_base);
                            self.stack_memory.truncate(cur_stack);

                            let prev = self.frames.pop().unwrap();
                            let dst_start = prev.dst_regs_start;
                            let dst_count = prev.dst_regs_count;
                            pc = prev.pc;
                            base = prev.base;
                            stack_base = prev.stack_base;
                            func = prev.func.clone();
                            mid = prev.mid;
                            values_ptr = self.value_stack.as_mut_ptr();

                            for i in 0..dst_count {
                                if i < self.args_buffer.len() {
                                    let dst_reg = self.dst_regs_buffer[dst_start + i];
                                    if dst_reg != 0 {
                                        *reg!(dst_reg) = self.args_buffer[i];
                                    }
                                }
                            }
                            self.dst_regs_buffer.truncate(dst_start);
                            continue 'main_loop;
                        }
                        Opcode::Call => {
                            let (rets, _args, dst_start) =
                                read_call_data!(func.data_section, inst.aux() as usize);
                            let f_id = veloc_ir::FuncId::from_u32(inst.imm32());

                            match program.import_links.get(&(mid, f_id)) {
                                Some(crate::host::ImportTarget::Module(m, f)) => {
                                    prepare_call!(
                                        *m,
                                        *f,
                                        dst_start,
                                        rets as usize,
                                        self.args_buffer
                                    );
                                }
                                Some(crate::host::ImportTarget::Host(h_id)) => {
                                    let res = program.host_functions_list[*h_id]
                                        .call(&mut self.args_buffer);
                                    values_ptr = self.value_stack.as_mut_ptr();
                                    if rets > 0 {
                                        let dst = self.dst_regs_buffer[dst_start];
                                        if dst != 0 {
                                            *reg!(dst) = res;
                                        }
                                    }
                                    self.dst_regs_buffer.truncate(dst_start);
                                }
                                None => {
                                    prepare_call!(
                                        mid,
                                        f_id,
                                        dst_start,
                                        rets as usize,
                                        self.args_buffer
                                    )
                                }
                            }
                        }
                        Opcode::CallIndirect => {
                            let (rets, _args, dst_start) =
                                read_call_data!(func.data_section, inst.imm32() as usize);
                            let ptr = get!(inst.src1).0 as usize;

                            if let Some((m, f)) = program.decode_interpreter_ptr(ptr) {
                                prepare_call!(m, f, dst_start, rets as usize, self.args_buffer);
                            } else if let Some(h) = program.decode_host_ptr(ptr) {
                                let res =
                                    program.host_functions_list[h].call(&mut self.args_buffer);
                                values_ptr = self.value_stack.as_mut_ptr();
                                if rets > 0 {
                                    let dst = self.dst_regs_buffer[dst_start];
                                    if dst != 0 {
                                        *reg!(dst) = res;
                                    }
                                }
                                self.dst_regs_buffer.truncate(dst_start);
                            } else {
                                panic!("Invalid function pointer: {:x}", ptr);
                            }
                        }
                        Opcode::PtrIndex => {
                            let ptr = get!(inst.src1).unwarp_i64();
                            let idx = get!(inst.src2).unwarp_i64();
                            let s = inst.imm32() as i64;
                            let o = inst.aux() as i64;
                            set!(
                                inst.dst,
                                InterpreterValue::i64(
                                    ptr.wrapping_add(idx.wrapping_mul(s)).wrapping_add(o)
                                )
                            );
                        }
                        Opcode::Select => {
                            let e = (inst.imm32() & 0xFFFF) as u16;
                            set!(
                                inst.dst,
                                if get!(inst.src1).unwarp_bool() {
                                    get!(inst.src2)
                                } else {
                                    get!(e)
                                }
                            );
                        }
                        Opcode::RegMove => mov!(inst.dst, inst.src1),
                        Opcode::GlobalAddr => {
                            todo!("GlobalAddr in interpreter");
                        }
                        Opcode::CallIntrinsic => {
                            let (rets, _args, dst_start) =
                                read_call_data!(func.data_section, inst.imm32() as usize);
                            let res = execute_intrinsic(inst.src1, &self.args_buffer);
                            values_ptr = self.value_stack.as_mut_ptr();
                            if rets > 0 {
                                let dst = self.dst_regs_buffer[dst_start];
                                if dst != 0 {
                                    *reg!(dst) = res;
                                }
                            }
                            self.dst_regs_buffer.truncate(dst_start);
                        }
                        Opcode::Unreachable => return Err(RunError::Unreachable),
                    }
                }
            }
            return Err(RunError::Unreachable);
        }
    }
}

fn execute_intrinsic(id: u16, args: &[InterpreterValue]) -> InterpreterValue {
    use veloc_ir::intrinsic_ids::*;
    let f = |i: usize| args[i].unwarp_f32();
    let d = |i: usize| args[i].unwarp_f64();

    match Intrinsic::from_u16(id) {
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
        MEMCPY | MEMMOVE | MEMSET => InterpreterValue::none(),
        MEMCMP => InterpreterValue::i32(0),
        FENCE | FENCE_ACQ | FENCE_REL | FENCE_SEQ => InterpreterValue::none(),
        ASSUME => InterpreterValue::none(),
        EXPECT => args[0],
        TRAP => panic!("trap"),
        _ => panic!("Unknown intrinsic: {}", id),
    }
}
