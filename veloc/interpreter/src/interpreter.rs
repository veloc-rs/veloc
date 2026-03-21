use crate::bytecode::{CompiledFunction, DecodedInstruction, Instruction, Reg};
use crate::error::Result;
use crate::runtime::{ImportTarget, Program};
use crate::value::InterpreterValue;
use ::alloc::vec::Vec;
use cranelift_entity::EntityRef;
use veloc_ir::{Intrinsic, ModuleId, ScalarType};

pub trait VirtualMemory {
    fn translate_addr(&self, logical_addr: usize, size: usize) -> Option<*mut u8>;
}

pub struct Interpreter {
    value_stack: Vec<InterpreterValue>,
    stack_memory: Vec<u8>,
    frames: Vec<StackFrame>,
    args_buffer: Vec<InterpreterValue>,
    dst_regs_buffer: Vec<Reg>,
}

struct StackFrame {
    mid: ModuleId,
    func: ::alloc::sync::Arc<CompiledFunction>,
    pc: usize,
    base: usize,
    stack_base: usize,
    dst_regs_start: usize,
    dst_regs_count: usize,
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

    pub fn run_function<M>(
        &mut self,
        program: &Program,
        mem: &M,
        mid: ModuleId,
        fid: veloc_ir::FuncId,
        args: &[InterpreterValue],
    ) -> Result<Vec<InterpreterValue>>
    where
        M: VirtualMemory,
    {
        let func = program.get_compiled_func(mid, fid);
        let base = self.value_stack.len();
        self.value_stack
            .resize(base + func.register_count, InterpreterValue::none());

        let stack_base = self.stack_memory.len();
        let total_stack_size: usize = func.stack_slots_sizes.iter().sum();
        self.stack_memory.resize(stack_base + total_stack_size, 0);

        // Initialize parameters
        for (i, &new_idx) in func.param_indices.iter().enumerate() {
            self.value_stack[base + new_idx.0 as usize] = args[i];
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
    fn do_call(
        &mut self,
        program: &Program,
        target_mid: ModuleId,
        target_fid: veloc_ir::FuncId,
        dst_regs_start: usize,
        dst_regs_count: usize,
        frame: &mut StackFrame,
    ) {
        if program.modules[target_mid].compiled[target_fid].is_none() {
            panic!(
                "Calling uncompiled function: mid={:?}, fid={:?}",
                target_mid, target_fid
            );
        }
        let next_func = program.get_compiled_func(target_mid, target_fid);
        self.frames.push(StackFrame {
            mid: frame.mid,
            func: frame.func.clone(),
            pc: frame.pc,
            base: frame.base,
            stack_base: frame.stack_base,
            dst_regs_start,
            dst_regs_count,
        });
        frame.mid = target_mid;
        frame.func = next_func;
        frame.pc = 0;
        frame.base = self.value_stack.len();
        self.value_stack.resize(
            frame.base + frame.func.register_count,
            InterpreterValue::none(),
        );
        frame.stack_base = self.stack_memory.len();
        let total_size: usize = frame.func.stack_slots_sizes.iter().sum();
        self.stack_memory.resize(frame.stack_base + total_size, 0);
        for (i, &new_idx) in frame.func.param_indices.iter().enumerate() {
            let val = self.args_buffer[i];
            self.value_stack[frame.base + new_idx.0 as usize] = val;
        }
    }

    fn execute<M>(&mut self, program: &Program, mem: &M) -> Result<Vec<InterpreterValue>>
    where
        M: VirtualMemory,
    {
        let mut frame = self.frames.pop().unwrap();

        'main_loop: loop {
            let mut values_ptr = self.value_stack.as_mut_ptr();

            // === Core Register Access Helpers ===
            macro_rules! reg {
                ($r:expr) => {
                    &mut *values_ptr.add(frame.base + ($r).index() as usize)
                };
            }
            macro_rules! get {
                ($r:expr) => {
                    *values_ptr.add(frame.base + ($r).index() as usize)
                };
            }
            macro_rules! set {
                ($d:expr, $v:expr) => {
                    *reg!($d) = $v
                };
            }

            // Memory load/store
            macro_rules! load {
                ($d:expr, $p:expr, $off:expr, $rd:ty, $wrap:expr) => {{
                    let addr = (get!($p).0 as usize).wrapping_add($off as usize);
                    match mem.translate_addr(addr, core::mem::size_of::<$rd>()) {
                        Some(ptr) => set!($d, $wrap((ptr as *const $rd).read_unaligned())),
                        None => return Err(crate::error::Error::OutOfBounds),
                    }
                }};
            }
            macro_rules! store {
                ($v:expr, $p:expr, $off:expr, $ty:ident, $wr:ty) => {{
                    let addr = (get!($p).0 as usize).wrapping_add($off as usize);
                    match mem.translate_addr(addr, core::mem::size_of::<$wr>()) {
                        Some(ptr) => (ptr as *mut $wr).write_unaligned(get!($v).$ty() as _),
                        None => return Err(crate::error::Error::OutOfBounds),
                    }
                }};
            }

            loop {
                unsafe {
                    let mut execute_moves = |target: &crate::bytecode::JumpTarget| {
                        let num_moves = target.num_moves as usize;
                        for i in 0..num_moves {
                            let (dst, src) = frame.func.data_section.jump_move_pair(target, i);
                            set!(dst, get!(src));
                        }
                    };

                    let mut read_call_data =
                        |data_sec: &crate::bytecode::DataSection,
                         off: usize,
                         num_rets: u16,
                         num_args: u16| {
                            let dst_start = self.dst_regs_buffer.len();
                            for i in 0..num_rets {
                                self.dst_regs_buffer
                                    .push(data_sec.call_ret_reg(off, i as usize));
                            }
                            self.args_buffer.clear();
                            for i in 0..num_args {
                                let reg = data_sec.call_arg_reg(off, num_rets as usize, i as usize);
                                self.args_buffer.push(get!(reg));
                            }
                            dst_start
                        };

                    let inst: Instruction = *frame.func.code.get_unchecked(frame.pc);
                    frame.pc += 1;
                    let inst = inst.decode();

                    match inst {
                        // === Constants ===
                        DecodedInstruction::Iconst { dst, imm64 } => {
                            set!(dst, InterpreterValue::i64(imm64 as i64))
                        }
                        DecodedInstruction::Fconst { dst, imm64 } => {
                            set!(dst, InterpreterValue(imm64))
                        }
                        DecodedInstruction::Bconst { dst, val } => {
                            set!(dst, InterpreterValue::bool(val))
                        }
                        DecodedInstruction::Vconst { .. } => {
                            todo!("Vector constants")
                        }

                        // === I32 Arithmetic ===
                        DecodedInstruction::I32Add { dst, src1, src2 } => {
                            let (a, b) = (get!(src1).unwrap_i32(), get!(src2).unwrap_i32());
                            set!(dst, InterpreterValue::i32(a.wrapping_add(b)));
                        }
                        DecodedInstruction::I32AddImm { dst, src1, imm } => {
                            let a = get!(src1).unwrap_i32();
                            set!(dst, InterpreterValue::i32(a.wrapping_add(imm as i32)));
                        }
                        DecodedInstruction::I32Sub { dst, src1, src2 } => {
                            let (a, b) = (get!(src1).unwrap_i32(), get!(src2).unwrap_i32());
                            set!(dst, InterpreterValue::i32(a.wrapping_sub(b)));
                        }
                        DecodedInstruction::I32SubImm { dst, src1, imm } => {
                            let a = get!(src1).unwrap_i32();
                            set!(dst, InterpreterValue::i32(a.wrapping_sub(imm as i32)));
                        }
                        DecodedInstruction::I32Mul { dst, src1, src2 } => {
                            let (a, b) = (get!(src1).unwrap_i32(), get!(src2).unwrap_i32());
                            set!(dst, InterpreterValue::i32(a.wrapping_mul(b)));
                        }
                        DecodedInstruction::I32DivS { dst, src1, src2 } => {
                            let (a, b) = (get!(src1).unwrap_i32(), get!(src2).unwrap_i32());
                            set!(dst, InterpreterValue::i32(a.wrapping_div(b)));
                        }
                        DecodedInstruction::I32DivU { dst, src1, src2 } => {
                            let (a, b) = (
                                get!(src1).unwrap_i32() as u32,
                                get!(src2).unwrap_i32() as u32,
                            );
                            set!(dst, InterpreterValue::i32(a.wrapping_div(b) as i32));
                        }
                        DecodedInstruction::I32RemS { dst, src1, src2 } => {
                            let (a, b) = (get!(src1).unwrap_i32(), get!(src2).unwrap_i32());
                            set!(dst, InterpreterValue::i32(a.wrapping_rem(b)));
                        }
                        DecodedInstruction::I32RemU { dst, src1, src2 } => {
                            let (a, b) = (
                                get!(src1).unwrap_i32() as u32,
                                get!(src2).unwrap_i32() as u32,
                            );
                            set!(dst, InterpreterValue::i32(a.wrapping_rem(b) as i32));
                        }
                        DecodedInstruction::I32And { dst, src1, src2 } => {
                            let (a, b) = (get!(src1).unwrap_i32(), get!(src2).unwrap_i32());
                            set!(dst, InterpreterValue::i32(a & b));
                        }
                        DecodedInstruction::I32AndImm { dst, src1, imm } => {
                            let a = get!(src1).unwrap_i32();
                            set!(dst, InterpreterValue::i32(a & imm as i32));
                        }
                        DecodedInstruction::I32Or { dst, src1, src2 } => {
                            let (a, b) = (get!(src1).unwrap_i32(), get!(src2).unwrap_i32());
                            set!(dst, InterpreterValue::i32(a | b));
                        }
                        DecodedInstruction::I32OrImm { dst, src1, imm } => {
                            let a = get!(src1).unwrap_i32();
                            set!(dst, InterpreterValue::i32(a | imm as i32));
                        }
                        DecodedInstruction::I32Xor { dst, src1, src2 } => {
                            let (a, b) = (get!(src1).unwrap_i32(), get!(src2).unwrap_i32());
                            set!(dst, InterpreterValue::i32(a ^ b));
                        }
                        DecodedInstruction::I32XorImm { dst, src1, imm } => {
                            let a = get!(src1).unwrap_i32();
                            set!(dst, InterpreterValue::i32(a ^ imm as i32));
                        }
                        DecodedInstruction::I32Shl { dst, src1, src2 } => {
                            let (a, b) = (get!(src1).unwrap_i32(), get!(src2).unwrap_i32());
                            set!(dst, InterpreterValue::i32(a.wrapping_shl(b as u32)));
                        }
                        DecodedInstruction::I32ShlImm { dst, src1, imm } => {
                            let a = get!(src1).unwrap_i32();
                            set!(dst, InterpreterValue::i32(a.wrapping_shl(imm as u32)));
                        }
                        DecodedInstruction::I32ShrS { dst, src1, src2 } => {
                            let (a, b) = (get!(src1).unwrap_i32(), get!(src2).unwrap_i32());
                            set!(dst, InterpreterValue::i32(a.wrapping_shr(b as u32)));
                        }
                        DecodedInstruction::I32ShrSImm { dst, src1, imm } => {
                            let a = get!(src1).unwrap_i32();
                            set!(dst, InterpreterValue::i32(a.wrapping_shr(imm as u32)));
                        }
                        DecodedInstruction::I32ShrU { dst, src1, src2 } => {
                            let (a, b) = (
                                get!(src1).unwrap_i32() as u32,
                                get!(src2).unwrap_i32() as u32,
                            );
                            set!(dst, InterpreterValue::i32(a.wrapping_shr(b) as i32));
                        }
                        DecodedInstruction::I32ShrUImm { dst, src1, imm } => {
                            let a = get!(src1).unwrap_i32() as u32;
                            set!(
                                dst,
                                InterpreterValue::i32(a.wrapping_shr(imm as u32) as i32)
                            );
                        }
                        DecodedInstruction::I32RotL { dst, src1, src2 } => {
                            let (a, b) = (get!(src1).unwrap_i32(), get!(src2).unwrap_i32());
                            set!(dst, InterpreterValue::i32(a.rotate_left(b as u32)));
                        }
                        DecodedInstruction::I32RotR { dst, src1, src2 } => {
                            let (a, b) = (get!(src1).unwrap_i32(), get!(src2).unwrap_i32());
                            set!(dst, InterpreterValue::i32(a.rotate_right(b as u32)));
                        }
                        DecodedInstruction::I32Clz { dst, src } => {
                            set!(
                                dst,
                                InterpreterValue::i32(get!(src).unwrap_i32().leading_zeros() as i32)
                            )
                        }
                        DecodedInstruction::I32Ctz { dst, src } => {
                            set!(
                                dst,
                                InterpreterValue::i32(
                                    get!(src).unwrap_i32().trailing_zeros() as i32
                                )
                            )
                        }
                        DecodedInstruction::I32Popcnt { dst, src } => {
                            set!(
                                dst,
                                InterpreterValue::i32(get!(src).unwrap_i32().count_ones() as i32)
                            )
                        }
                        DecodedInstruction::I32Eqz { dst, src_val } => {
                            set!(dst, InterpreterValue::bool(get!(src_val).unwrap_i32() == 0))
                        }
                        DecodedInstruction::I32Eq { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    get!(src1).unwrap_i32() == get!(src2).unwrap_i32()
                                )
                            )
                        }
                        DecodedInstruction::I32Ne { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    get!(src1).unwrap_i32() != get!(src2).unwrap_i32()
                                )
                            )
                        }
                        DecodedInstruction::I32LtS { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    get!(src1).unwrap_i32() < get!(src2).unwrap_i32()
                                )
                            )
                        }
                        DecodedInstruction::I32LtU { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    (get!(src1).unwrap_i32() as u32)
                                        < (get!(src2).unwrap_i32() as u32)
                                )
                            )
                        }
                        DecodedInstruction::I32LeS { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    get!(src1).unwrap_i32() <= get!(src2).unwrap_i32()
                                )
                            )
                        }
                        DecodedInstruction::I32LeU { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    (get!(src1).unwrap_i32() as u32)
                                        <= (get!(src2).unwrap_i32() as u32)
                                )
                            )
                        }
                        DecodedInstruction::I32GtS { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    get!(src1).unwrap_i32() > get!(src2).unwrap_i32()
                                )
                            )
                        }
                        DecodedInstruction::I32GtU { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    (get!(src1).unwrap_i32() as u32)
                                        > (get!(src2).unwrap_i32() as u32)
                                )
                            )
                        }
                        DecodedInstruction::I32GeS { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    get!(src1).unwrap_i32() >= get!(src2).unwrap_i32()
                                )
                            )
                        }
                        DecodedInstruction::I32GeU { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    (get!(src1).unwrap_i32() as u32)
                                        >= (get!(src2).unwrap_i32() as u32)
                                )
                            )
                        }

                        // === I64 Operations ===
                        DecodedInstruction::I64Add { dst, src1, src2 } => {
                            let (a, b) = (get!(src1).unwrap_i64(), get!(src2).unwrap_i64());
                            set!(dst, InterpreterValue::i64(a.wrapping_add(b)));
                        }
                        DecodedInstruction::I64AddImm { dst, src1, imm64 } => {
                            let a = get!(src1).unwrap_i64();
                            set!(dst, InterpreterValue::i64(a.wrapping_add(imm64 as i64)));
                        }
                        DecodedInstruction::I64Sub { dst, src1, src2 } => {
                            let (a, b) = (get!(src1).unwrap_i64(), get!(src2).unwrap_i64());
                            set!(dst, InterpreterValue::i64(a.wrapping_sub(b)));
                        }
                        DecodedInstruction::I64SubImm { dst, src1, imm64 } => {
                            let a = get!(src1).unwrap_i64();
                            set!(dst, InterpreterValue::i64(a.wrapping_sub(imm64 as i64)));
                        }
                        DecodedInstruction::I64Mul { dst, src1, src2 } => {
                            let (a, b) = (get!(src1).unwrap_i64(), get!(src2).unwrap_i64());
                            set!(dst, InterpreterValue::i64(a.wrapping_mul(b)));
                        }
                        DecodedInstruction::I64DivS { dst, src1, src2 } => {
                            let (a, b) = (get!(src1).unwrap_i64(), get!(src2).unwrap_i64());
                            set!(dst, InterpreterValue::i64(a.wrapping_div(b)));
                        }
                        DecodedInstruction::I64DivU { dst, src1, src2 } => {
                            let (a, b) = (
                                get!(src1).unwrap_i64() as u64,
                                get!(src2).unwrap_i64() as u64,
                            );
                            set!(dst, InterpreterValue::i64(a.wrapping_div(b) as i64));
                        }
                        DecodedInstruction::I64RemS { dst, src1, src2 } => {
                            let (a, b) = (get!(src1).unwrap_i64(), get!(src2).unwrap_i64());
                            set!(dst, InterpreterValue::i64(a.wrapping_rem(b)));
                        }
                        DecodedInstruction::I64RemU { dst, src1, src2 } => {
                            let (a, b) = (
                                get!(src1).unwrap_i64() as u64,
                                get!(src2).unwrap_i64() as u64,
                            );
                            set!(dst, InterpreterValue::i64(a.wrapping_rem(b) as i64));
                        }
                        DecodedInstruction::I64And { dst, src1, src2 } => {
                            let (a, b) = (get!(src1).unwrap_i64(), get!(src2).unwrap_i64());
                            set!(dst, InterpreterValue::i64(a & b));
                        }
                        DecodedInstruction::I64AndImm { dst, src1, imm64 } => {
                            set!(
                                dst,
                                InterpreterValue::i64(get!(src1).unwrap_i64() & imm64 as i64)
                            )
                        }
                        DecodedInstruction::I64Or { dst, src1, src2 } => {
                            let (a, b) = (get!(src1).unwrap_i64(), get!(src2).unwrap_i64());
                            set!(dst, InterpreterValue::i64(a | b));
                        }
                        DecodedInstruction::I64OrImm { dst, src1, imm64 } => {
                            set!(
                                dst,
                                InterpreterValue::i64(get!(src1).unwrap_i64() | imm64 as i64)
                            )
                        }
                        DecodedInstruction::I64Xor { dst, src1, src2 } => {
                            let (a, b) = (get!(src1).unwrap_i64(), get!(src2).unwrap_i64());
                            set!(dst, InterpreterValue::i64(a ^ b));
                        }
                        DecodedInstruction::I64XorImm { dst, src1, imm64 } => {
                            set!(
                                dst,
                                InterpreterValue::i64(get!(src1).unwrap_i64() ^ imm64 as i64)
                            )
                        }
                        DecodedInstruction::I64Shl { dst, src1, src2 } => {
                            let (a, b) = (get!(src1).unwrap_i64(), get!(src2).unwrap_i64());
                            set!(dst, InterpreterValue::i64(a.wrapping_shl(b as u32)));
                        }
                        DecodedInstruction::I64ShlImm { dst, src1, imm64 } => {
                            set!(
                                dst,
                                InterpreterValue::i64(
                                    get!(src1).unwrap_i64().wrapping_shl(imm64 as u32)
                                )
                            )
                        }
                        DecodedInstruction::I64ShrS { dst, src1, src2 } => {
                            let (a, b) = (get!(src1).unwrap_i64(), get!(src2).unwrap_i64());
                            set!(dst, InterpreterValue::i64(a.wrapping_shr(b as u32)));
                        }
                        DecodedInstruction::I64ShrSImm { dst, src1, imm64 } => {
                            set!(
                                dst,
                                InterpreterValue::i64(
                                    get!(src1).unwrap_i64().wrapping_shr(imm64 as u32)
                                )
                            )
                        }
                        DecodedInstruction::I64ShrU { dst, src1, src2 } => {
                            let (a, b) = (
                                get!(src1).unwrap_i64() as u64,
                                get!(src2).unwrap_i64() as u32,
                            );
                            set!(dst, InterpreterValue::i64(a.wrapping_shr(b) as i64));
                        }
                        DecodedInstruction::I64ShrUImm { dst, src1, imm64 } => {
                            set!(
                                dst,
                                InterpreterValue::i64(
                                    (get!(src1).unwrap_i64() as u64).wrapping_shr(imm64 as u32)
                                        as i64
                                )
                            )
                        }
                        DecodedInstruction::I64RotL { dst, src1, src2 } => {
                            let (a, b) = (get!(src1).unwrap_i64(), get!(src2).unwrap_i64());
                            set!(dst, InterpreterValue::i64(a.rotate_left(b as u32)));
                        }
                        DecodedInstruction::I64RotR { dst, src1, src2 } => {
                            let (a, b) = (get!(src1).unwrap_i64(), get!(src2).unwrap_i64());
                            set!(dst, InterpreterValue::i64(a.rotate_right(b as u32)));
                        }
                        DecodedInstruction::I64Clz { dst, src } => {
                            set!(
                                dst,
                                InterpreterValue::i64(get!(src).unwrap_i64().leading_zeros() as i64)
                            )
                        }
                        DecodedInstruction::I64Ctz { dst, src } => {
                            set!(
                                dst,
                                InterpreterValue::i64(
                                    get!(src).unwrap_i64().trailing_zeros() as i64
                                )
                            )
                        }
                        DecodedInstruction::I64Popcnt { dst, src } => {
                            set!(
                                dst,
                                InterpreterValue::i64(get!(src).unwrap_i64().count_ones() as i64)
                            )
                        }
                        DecodedInstruction::I64Eqz { dst, src_val } => {
                            set!(dst, InterpreterValue::bool(get!(src_val).unwrap_i64() == 0))
                        }
                        DecodedInstruction::I64Eq { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    get!(src1).unwrap_i64() == get!(src2).unwrap_i64()
                                )
                            )
                        }
                        DecodedInstruction::I64Ne { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    get!(src1).unwrap_i64() != get!(src2).unwrap_i64()
                                )
                            )
                        }
                        DecodedInstruction::I64LtS { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    get!(src1).unwrap_i64() < get!(src2).unwrap_i64()
                                )
                            )
                        }
                        DecodedInstruction::I64LtU { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    (get!(src1).unwrap_i64() as u64)
                                        < (get!(src2).unwrap_i64() as u64)
                                )
                            )
                        }
                        DecodedInstruction::I64LeS { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    get!(src1).unwrap_i64() <= get!(src2).unwrap_i64()
                                )
                            )
                        }
                        DecodedInstruction::I64LeU { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    (get!(src1).unwrap_i64() as u64)
                                        <= (get!(src2).unwrap_i64() as u64)
                                )
                            )
                        }
                        DecodedInstruction::I64GtS { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    get!(src1).unwrap_i64() > get!(src2).unwrap_i64()
                                )
                            )
                        }
                        DecodedInstruction::I64GtU { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    (get!(src1).unwrap_i64() as u64)
                                        > (get!(src2).unwrap_i64() as u64)
                                )
                            )
                        }
                        DecodedInstruction::I64GeS { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    get!(src1).unwrap_i64() >= get!(src2).unwrap_i64()
                                )
                            )
                        }
                        DecodedInstruction::I64GeU { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    (get!(src1).unwrap_i64() as u64)
                                        >= (get!(src2).unwrap_i64() as u64)
                                )
                            )
                        }

                        // === F32 Operations ===
                        DecodedInstruction::F32Add { dst, src1, src2 } => {
                            let lhs = get!(src1).unwrap_f32();
                            let rhs = get!(src2).unwrap_f32();
                            set!(dst, InterpreterValue::f32(lhs + rhs));
                        }
                        DecodedInstruction::F32Sub { dst, src1, src2 } => {
                            let lhs = get!(src1).unwrap_f32();
                            let rhs = get!(src2).unwrap_f32();
                            set!(dst, InterpreterValue::f32(lhs - rhs));
                        }
                        DecodedInstruction::F32Mul { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::f32(
                                    get!(src1).unwrap_f32() * get!(src2).unwrap_f32()
                                )
                            )
                        }
                        DecodedInstruction::F32Div { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::f32(
                                    get!(src1).unwrap_f32() / get!(src2).unwrap_f32()
                                )
                            )
                        }
                        DecodedInstruction::F32Abs { dst, src1 } => {
                            set!(dst, InterpreterValue::f32(get!(src1).unwrap_f32().abs()))
                        }
                        DecodedInstruction::F32Neg { dst, src1 } => {
                            set!(dst, InterpreterValue::f32(-get!(src1).unwrap_f32()))
                        }
                        DecodedInstruction::F32Sqrt { dst, src1 } => {
                            set!(dst, InterpreterValue::f32(get!(src1).unwrap_f32().sqrt()))
                        }
                        DecodedInstruction::F32Ceil { dst, src1 } => {
                            set!(dst, InterpreterValue::f32(get!(src1).unwrap_f32().ceil()))
                        }
                        DecodedInstruction::F32Floor { dst, src1 } => {
                            set!(dst, InterpreterValue::f32(get!(src1).unwrap_f32().floor()))
                        }
                        DecodedInstruction::F32Trunc { dst, src1 } => {
                            set!(dst, InterpreterValue::f32(get!(src1).unwrap_f32().trunc()))
                        }
                        DecodedInstruction::F32Nearest { dst, src1 } => {
                            set!(
                                dst,
                                InterpreterValue::f32(get!(src1).unwrap_f32().round_ties_even())
                            )
                        }
                        DecodedInstruction::F32Min { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::f32(
                                    get!(src1).unwrap_f32().min(get!(src2).unwrap_f32())
                                )
                            )
                        }
                        DecodedInstruction::F32Max { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::f32(
                                    get!(src1).unwrap_f32().max(get!(src2).unwrap_f32())
                                )
                            )
                        }
                        DecodedInstruction::F32CopySign { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::f32(
                                    get!(src1).unwrap_f32().copysign(get!(src2).unwrap_f32())
                                )
                            )
                        }
                        DecodedInstruction::F32Eq { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    get!(src1).unwrap_f32() == get!(src2).unwrap_f32()
                                )
                            )
                        }
                        DecodedInstruction::F32Ne { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    get!(src1).unwrap_f32() != get!(src2).unwrap_f32()
                                )
                            )
                        }
                        DecodedInstruction::F32Lt { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    get!(src1).unwrap_f32() < get!(src2).unwrap_f32()
                                )
                            )
                        }
                        DecodedInstruction::F32Le { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    get!(src1).unwrap_f32() <= get!(src2).unwrap_f32()
                                )
                            )
                        }
                        DecodedInstruction::F32Gt { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    get!(src1).unwrap_f32() > get!(src2).unwrap_f32()
                                )
                            )
                        }
                        DecodedInstruction::F32Ge { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    get!(src1).unwrap_f32() >= get!(src2).unwrap_f32()
                                )
                            )
                        }

                        // === F64 Operations ===
                        DecodedInstruction::F64Add { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::f64(
                                    get!(src1).unwrap_f64() + get!(src2).unwrap_f64()
                                )
                            )
                        }
                        DecodedInstruction::F64Sub { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::f64(
                                    get!(src1).unwrap_f64() - get!(src2).unwrap_f64()
                                )
                            )
                        }
                        DecodedInstruction::F64Mul { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::f64(
                                    get!(src1).unwrap_f64() * get!(src2).unwrap_f64()
                                )
                            )
                        }
                        DecodedInstruction::F64Div { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::f64(
                                    get!(src1).unwrap_f64() / get!(src2).unwrap_f64()
                                )
                            )
                        }
                        DecodedInstruction::F64Abs { dst, src1 } => {
                            set!(dst, InterpreterValue::f64(get!(src1).unwrap_f64().abs()))
                        }
                        DecodedInstruction::F64Neg { dst, src1 } => {
                            set!(dst, InterpreterValue::f64(-get!(src1).unwrap_f64()))
                        }
                        DecodedInstruction::F64Sqrt { dst, src1 } => {
                            set!(dst, InterpreterValue::f64(get!(src1).unwrap_f64().sqrt()))
                        }
                        DecodedInstruction::F64Ceil { dst, src1 } => {
                            set!(dst, InterpreterValue::f64(get!(src1).unwrap_f64().ceil()))
                        }
                        DecodedInstruction::F64Floor { dst, src1 } => {
                            set!(dst, InterpreterValue::f64(get!(src1).unwrap_f64().floor()))
                        }
                        DecodedInstruction::F64Trunc { dst, src1 } => {
                            set!(dst, InterpreterValue::f64(get!(src1).unwrap_f64().trunc()))
                        }
                        DecodedInstruction::F64Nearest { dst, src1 } => {
                            set!(
                                dst,
                                InterpreterValue::f64(get!(src1).unwrap_f64().round_ties_even())
                            )
                        }
                        DecodedInstruction::F64Min { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::f64(
                                    get!(src1).unwrap_f64().min(get!(src2).unwrap_f64())
                                )
                            )
                        }
                        DecodedInstruction::F64Max { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::f64(
                                    get!(src1).unwrap_f64().max(get!(src2).unwrap_f64())
                                )
                            )
                        }
                        DecodedInstruction::F64CopySign { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::f64(
                                    get!(src1).unwrap_f64().copysign(get!(src2).unwrap_f64())
                                )
                            )
                        }
                        DecodedInstruction::F64Eq { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    get!(src1).unwrap_f64() == get!(src2).unwrap_f64()
                                )
                            )
                        }
                        DecodedInstruction::F64Ne { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    get!(src1).unwrap_f64() != get!(src2).unwrap_f64()
                                )
                            )
                        }
                        DecodedInstruction::F64Lt { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    get!(src1).unwrap_f64() < get!(src2).unwrap_f64()
                                )
                            )
                        }
                        DecodedInstruction::F64Le { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    get!(src1).unwrap_f64() <= get!(src2).unwrap_f64()
                                )
                            )
                        }
                        DecodedInstruction::F64Gt { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    get!(src1).unwrap_f64() > get!(src2).unwrap_f64()
                                )
                            )
                        }
                        DecodedInstruction::F64Ge { dst, src1, src2 } => {
                            set!(
                                dst,
                                InterpreterValue::bool(
                                    get!(src1).unwrap_f64() >= get!(src2).unwrap_f64()
                                )
                            )
                        }

                        // === Conversions ===
                        DecodedInstruction::ExtendS { dst, src, ty } => {
                            let val = get!(src).unwrap_i64();
                            let res = match ty.from {
                                ScalarType::I8 => val as i8 as i64,
                                ScalarType::I16 => val as i16 as i64,
                                ScalarType::I32 => val as i32 as i64,
                                _ => panic!("Unsupported ExtendS from_ty: {:?}", ty.from),
                            };
                            set!(
                                dst,
                                if ty.to == ScalarType::I32 {
                                    InterpreterValue::i32(res as i32)
                                } else {
                                    InterpreterValue::i64(res)
                                }
                            );
                        }
                        DecodedInstruction::ExtendU { dst, src, ty } => {
                            let val = get!(src).unwrap_i64();
                            let res = match ty.from {
                                ScalarType::I8 => (val as u8) as u64 as i64,
                                ScalarType::I16 => (val as u16) as u64 as i64,
                                ScalarType::I32 => (val as u32) as u64 as i64,
                                _ => panic!("Unsupported ExtendU from_ty: {:?}", ty.from),
                            };
                            set!(
                                dst,
                                if ty.to == ScalarType::I32 {
                                    InterpreterValue::i32(res as i32)
                                } else {
                                    InterpreterValue::i64(res)
                                }
                            );
                        }
                        DecodedInstruction::Wrap { dst, src, ty } => {
                            let val = get!(src).unwrap_i64();
                            let res = match ty.to {
                                ScalarType::I8 => val as i8 as i64,
                                ScalarType::I16 => val as i16 as i64,
                                ScalarType::I32 => val as i32 as i64,
                                _ => val,
                            };
                            set!(
                                dst,
                                if ty.to == ScalarType::I32 {
                                    InterpreterValue::i32(res as i32)
                                } else {
                                    InterpreterValue::i64(res)
                                }
                            );
                        }

                        DecodedInstruction::I32TruncF32S { dst, src } => {
                            set!(dst, InterpreterValue::i32(get!(src).unwrap_f32() as i32))
                        }
                        DecodedInstruction::I32TruncF32U { dst, src } => {
                            set!(
                                dst,
                                InterpreterValue::i32(get!(src).unwrap_f32() as u32 as i32)
                            )
                        }
                        DecodedInstruction::I32TruncF64S { dst, src } => {
                            set!(dst, InterpreterValue::i32(get!(src).unwrap_f64() as i32))
                        }
                        DecodedInstruction::I32TruncF64U { dst, src } => {
                            set!(
                                dst,
                                InterpreterValue::i32(get!(src).unwrap_f64() as u32 as i32)
                            )
                        }
                        DecodedInstruction::I64TruncF32S { dst, src } => {
                            set!(dst, InterpreterValue::i64(get!(src).unwrap_f32() as i64))
                        }
                        DecodedInstruction::I64TruncF32U { dst, src } => {
                            set!(
                                dst,
                                InterpreterValue::i64(get!(src).unwrap_f32() as u64 as i64)
                            )
                        }
                        DecodedInstruction::I64TruncF64S { dst, src } => {
                            set!(dst, InterpreterValue::i64(get!(src).unwrap_f64() as i64))
                        }
                        DecodedInstruction::I64TruncF64U { dst, src } => {
                            set!(
                                dst,
                                InterpreterValue::i64(get!(src).unwrap_f64() as u64 as i64)
                            )
                        }

                        DecodedInstruction::F32DemoteF64 { dst, src } => {
                            set!(dst, InterpreterValue::f32(get!(src).unwrap_f64() as f32))
                        }
                        DecodedInstruction::F64PromoteF32 { dst, src } => {
                            set!(dst, InterpreterValue::f64(get!(src).unwrap_f32() as f64))
                        }
                        DecodedInstruction::Bitcast { dst, src } => set!(dst, get!(src)),

                        DecodedInstruction::I32TruncSatF32S { dst, src } => {
                            let val = get!(src).unwrap_f32();
                            set!(
                                dst,
                                InterpreterValue::i32(if val.is_nan() { 0 } else { val as i32 })
                            );
                        }
                        DecodedInstruction::I32TruncSatF32U { dst, src } => {
                            let val = get!(src).unwrap_f32();
                            set!(
                                dst,
                                InterpreterValue::i32(if val.is_nan() || val < 0.0 {
                                    0
                                } else {
                                    val as u32
                                } as i32)
                            );
                        }
                        DecodedInstruction::I32TruncSatF64S { dst, src } => {
                            let val = get!(src).unwrap_f64();
                            set!(
                                dst,
                                InterpreterValue::i32(if val.is_nan() { 0 } else { val as i32 })
                            );
                        }
                        DecodedInstruction::I32TruncSatF64U { dst, src } => {
                            let val = get!(src).unwrap_f64();
                            set!(
                                dst,
                                InterpreterValue::i32(if val.is_nan() || val < 0.0 {
                                    0
                                } else {
                                    val as u32
                                } as i32)
                            );
                        }
                        DecodedInstruction::I64TruncSatF32S { dst, src } => {
                            let val = get!(src).unwrap_f32();
                            set!(
                                dst,
                                InterpreterValue::i64(if val.is_nan() { 0 } else { val as i64 })
                            );
                        }
                        DecodedInstruction::I64TruncSatF32U { dst, src } => {
                            let val = get!(src).unwrap_f32();
                            set!(
                                dst,
                                InterpreterValue::i64(if val.is_nan() || val < 0.0 {
                                    0
                                } else {
                                    val as u64
                                } as i64)
                            );
                        }
                        DecodedInstruction::I64TruncSatF64S { dst, src } => {
                            let val = get!(src).unwrap_f64();
                            set!(
                                dst,
                                InterpreterValue::i64(if val.is_nan() { 0 } else { val as i64 })
                            );
                        }
                        DecodedInstruction::I64TruncSatF64U { dst, src } => {
                            let val = get!(src).unwrap_f64();
                            set!(
                                dst,
                                InterpreterValue::i64(if val.is_nan() || val < 0.0 {
                                    0
                                } else {
                                    val as u64
                                } as i64)
                            );
                        }
                        DecodedInstruction::F32ConvertI32S { dst, src } => {
                            set!(dst, InterpreterValue::f32(get!(src).unwrap_i32() as f32))
                        }
                        DecodedInstruction::F32ConvertI32U { dst, src } => {
                            set!(
                                dst,
                                InterpreterValue::f32(get!(src).unwrap_i32() as u32 as f32)
                            )
                        }
                        DecodedInstruction::F32ConvertI64S { dst, src } => {
                            set!(dst, InterpreterValue::f32(get!(src).unwrap_i64() as f32))
                        }
                        DecodedInstruction::F32ConvertI64U { dst, src } => {
                            set!(
                                dst,
                                InterpreterValue::f32(get!(src).unwrap_i64() as u64 as f32)
                            )
                        }
                        DecodedInstruction::F64ConvertI32S { dst, src } => {
                            set!(dst, InterpreterValue::f64(get!(src).unwrap_i32() as f64))
                        }
                        DecodedInstruction::F64ConvertI32U { dst, src } => {
                            set!(
                                dst,
                                InterpreterValue::f64(get!(src).unwrap_i32() as u32 as f64)
                            )
                        }
                        DecodedInstruction::F64ConvertI64S { dst, src } => {
                            set!(dst, InterpreterValue::f64(get!(src).unwrap_i64() as f64))
                        }
                        DecodedInstruction::F64ConvertI64U { dst, src } => {
                            set!(
                                dst,
                                InterpreterValue::f64(get!(src).unwrap_i64() as u64 as f64)
                            )
                        }

                        // === Memory Access ===
                        DecodedInstruction::I32Load { dst, ptr, offset } => {
                            load!(dst, ptr, offset, i32, |v| InterpreterValue::i32(v))
                        }
                        DecodedInstruction::I64Load { dst, ptr, offset } => {
                            load!(dst, ptr, offset, i64, |v| InterpreterValue::i64(v))
                        }
                        DecodedInstruction::I8Load { dst, ptr, offset } => {
                            load!(dst, ptr, offset, u8, |v| InterpreterValue::i64(v as i64))
                        }
                        DecodedInstruction::I16Load { dst, ptr, offset } => {
                            load!(dst, ptr, offset, u16, |v| InterpreterValue::i64(v as i64))
                        }
                        DecodedInstruction::F32Load { dst, ptr, offset } => {
                            load!(dst, ptr, offset, f32, |v| InterpreterValue::f32(v))
                        }
                        DecodedInstruction::F64Load { dst, ptr, offset } => {
                            load!(dst, ptr, offset, f64, |v| InterpreterValue::f64(v))
                        }
                        DecodedInstruction::I32Store { val, ptr, offset } => {
                            store!(val, ptr, offset, unwrap_i32, i32)
                        }
                        DecodedInstruction::I64Store { val, ptr, offset } => {
                            store!(val, ptr, offset, unwrap_i64, i64)
                        }
                        DecodedInstruction::I8Store { val, ptr, offset } => {
                            store!(val, ptr, offset, unwrap_i64, u8)
                        }
                        DecodedInstruction::I16Store { val, ptr, offset } => {
                            store!(val, ptr, offset, unwrap_i64, u16)
                        }
                        DecodedInstruction::F32Store { val, ptr, offset } => {
                            store!(val, ptr, offset, unwrap_f32, f32)
                        }
                        DecodedInstruction::F64Store { val, ptr, offset } => {
                            store!(val, ptr, offset, unwrap_f64, f64)
                        }

                        // === Stack Operations ===
                        DecodedInstruction::StackAddr { dst, offset } => {
                            let ptr = self
                                .stack_memory
                                .as_ptr()
                                .add(frame.stack_base + offset as usize);
                            set!(dst, InterpreterValue::i64(ptr as i64));
                        }
                        DecodedInstruction::StackLoad { dst, ty, offset } => {
                            let addr = frame.stack_base + offset as usize;
                            let ptr = self.stack_memory.as_ptr().add(addr);
                            set!(
                                dst,
                                match ty {
                                    ScalarType::I8 => InterpreterValue::i32(
                                        (ptr as *const i8).read_unaligned() as i32
                                    ),
                                    ScalarType::I16 => InterpreterValue::i32(
                                        (ptr as *const i16).read_unaligned() as i32
                                    ),
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
                        DecodedInstruction::StackStore { val, ty, offset } => {
                            let addr = frame.stack_base + offset as usize;
                            let ptr = self.stack_memory.as_mut_ptr().add(addr);
                            let v = get!(val);
                            match ty {
                                ScalarType::I8 => {
                                    (ptr as *mut i8).write_unaligned(v.unwrap_i32() as i8)
                                }
                                ScalarType::I16 => {
                                    (ptr as *mut i16).write_unaligned(v.unwrap_i32() as i16)
                                }
                                ScalarType::I32 => {
                                    (ptr as *mut i32).write_unaligned(v.unwrap_i32())
                                }
                                ScalarType::I64 | ScalarType::Ptr => {
                                    (ptr as *mut i64).write_unaligned(v.unwrap_i64())
                                }
                                ScalarType::F32 => {
                                    (ptr as *mut f32).write_unaligned(v.unwrap_f32())
                                }
                                ScalarType::F64 => {
                                    (ptr as *mut f64).write_unaligned(v.unwrap_f64())
                                }
                                _ => panic!("Unknown type {:?} in StackStore", ty),
                            }
                        }

                        // === Control Flow ===
                        DecodedInstruction::Jump { pc: target_pc } => frame.pc = target_pc as usize,
                        DecodedInstruction::JumpWithMoves { data_offset } => {
                            let target =
                                &frame.func.data_section.jump_targets[data_offset as usize];
                            execute_moves(target);
                            frame.pc = target.pc as usize;
                        }
                        DecodedInstruction::Br {
                            cond,
                            then_idx,
                            else_idx,
                        } => {
                            let c = get!(cond).unwrap_bool();
                            let target_idx = if c { then_idx } else { else_idx };
                            let target = &frame.func.data_section.jump_targets[target_idx as usize];
                            execute_moves(target);
                            frame.pc = target.pc as usize;
                        }
                        DecodedInstruction::BrTable {
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
                            let target = &frame.func.data_section.jump_targets
                                [data_offset as usize + target_idx];
                            execute_moves(target);
                            frame.pc = target.pc as usize;
                        }
                        DecodedInstruction::Return {
                            data_offset,
                            num_vals,
                        } => {
                            let data_off = data_offset as usize;
                            let nvals = num_vals as usize;
                            let cur_base = frame.base;
                            let cur_stack = frame.stack_base;

                            if self.frames.is_empty() {
                                let mut res = Vec::with_capacity(nvals);
                                for i in 0..nvals {
                                    res.push(get!(frame.func.data_section.return_reg(data_off, i)));
                                }
                                self.value_stack.truncate(cur_base);
                                self.stack_memory.truncate(cur_stack);
                                return Ok(res);
                            }

                            self.args_buffer.clear();
                            for i in 0..nvals {
                                self.args_buffer
                                    .push(get!(frame.func.data_section.return_reg(data_off, i)));
                            }
                            self.value_stack.truncate(cur_base);
                            self.stack_memory.truncate(cur_stack);

                            let prev = self.frames.pop().unwrap();
                            let dst_start = prev.dst_regs_start;
                            let dst_count = prev.dst_regs_count;
                            frame.pc = prev.pc;
                            frame.base = prev.base;
                            frame.stack_base = prev.stack_base;
                            frame.func = prev.func;
                            frame.mid = prev.mid;
                            values_ptr = self.value_stack.as_mut_ptr();

                            debug_assert_eq!(dst_count, self.args_buffer.len());
                            for i in 0..dst_count {
                                let dst_reg = self.dst_regs_buffer[dst_start + i];
                                if dst_reg != Reg::NULL {
                                    *reg!(dst_reg) = self.args_buffer[i];
                                }
                            }
                            self.dst_regs_buffer.truncate(dst_start);
                            continue 'main_loop;
                        }
                        DecodedInstruction::Call {
                            func_id,
                            data_offset,
                            num_rets,
                            num_args,
                        } => {
                            let dst_start = read_call_data(
                                &frame.func.data_section,
                                data_offset as usize,
                                num_rets,
                                num_args,
                            );
                            let f_id = veloc_ir::FuncId::from_u32(func_id);

                            match program.modules[frame.mid].links[f_id] {
                                ImportTarget::Module(m, f) => {
                                    self.do_call(
                                        program,
                                        m,
                                        f,
                                        dst_start,
                                        num_rets as usize,
                                        &mut frame,
                                    );
                                    continue 'main_loop;
                                }
                                ImportTarget::Host(h_id) => {
                                    let args = self.args_buffer.len();
                                    program.host_functions_list[h_id]
                                        .call(&mut self.args_buffer, args);
                                    values_ptr = self.value_stack.as_mut_ptr();
                                    for i in 0..num_rets as usize {
                                        let dst = self.dst_regs_buffer[dst_start + i];
                                        if dst != Reg::NULL {
                                            *reg!(dst) = self.args_buffer[i];
                                        }
                                    }
                                    self.dst_regs_buffer.truncate(dst_start);
                                }
                                ImportTarget::None => {
                                    self.do_call(
                                        program,
                                        frame.mid,
                                        f_id,
                                        dst_start,
                                        num_rets as usize,
                                        &mut frame,
                                    );
                                    continue 'main_loop;
                                }
                            }
                        }
                        DecodedInstruction::CallIndirect {
                            ptr,
                            data_offset,
                            num_rets,
                            num_args,
                        } => {
                            let dst_start = read_call_data(
                                &frame.func.data_section,
                                data_offset as usize,
                                num_rets,
                                num_args,
                            );
                            let p = get!(ptr).0 as usize;

                            match program.decode_ptr(p) {
                                Some(ImportTarget::Module(m, f)) => {
                                    self.do_call(
                                        program,
                                        m,
                                        f,
                                        dst_start,
                                        num_rets as usize,
                                        &mut frame,
                                    );
                                    continue 'main_loop;
                                }
                                Some(ImportTarget::Host(h_id)) => {
                                    let args = self.args_buffer.len();
                                    program.host_functions_list[h_id]
                                        .call(&mut self.args_buffer, args);
                                    values_ptr = self.value_stack.as_mut_ptr();
                                    for i in 0..num_rets as usize {
                                        let dst = self.dst_regs_buffer[dst_start + i];
                                        if dst != Reg::NULL {
                                            *reg!(dst) = self.args_buffer[i];
                                        }
                                    }
                                    self.dst_regs_buffer.truncate(dst_start);
                                }
                                _ => panic!("Invalid function pointer: {:x}", p),
                            }
                        }
                        DecodedInstruction::PtrIndex {
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
                                InterpreterValue::i64(
                                    p.wrapping_add(idx.wrapping_mul(s)).wrapping_add(o)
                                )
                            );
                        }
                        DecodedInstruction::Select {
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
                        DecodedInstruction::RegMove { dst, src } => set!(dst, get!(src)),
                        DecodedInstruction::GlobalAddr { .. } => {
                            todo!("GlobalAddr in interpreter");
                        }
                        DecodedInstruction::CallIntrinsic {
                            intrinsic,
                            data_offset,
                            num_rets,
                            num_args,
                        } => {
                            let dst_start = read_call_data(
                                &frame.func.data_section,
                                data_offset as usize,
                                num_rets,
                                num_args,
                            );
                            let res = execute_intrinsic(intrinsic, &self.args_buffer);
                            values_ptr = self.value_stack.as_mut_ptr();
                            if num_rets > 0 {
                                let dst = self.dst_regs_buffer[dst_start];
                                if dst != Reg::NULL {
                                    *reg!(dst) = res;
                                }
                            }
                            self.dst_regs_buffer.truncate(dst_start);
                        }
                        DecodedInstruction::Unreachable {} => {
                            return Err(crate::error::Error::Unreachable);
                        }
                    }
                }
            }
        }
    }
}

fn execute_intrinsic(id: u16, args: &[InterpreterValue]) -> InterpreterValue {
    use veloc_ir::intrinsic_ids::*;
    let f = |i: usize| args[i].unwrap_f32();
    let d = |i: usize| args[i].unwrap_f64();

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
