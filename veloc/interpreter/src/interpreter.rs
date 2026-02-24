use crate::bytecode::{CompiledFunction, Opcode};
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
    dst_regs_buffer: Vec<u16>,
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

    fn execute<M>(&mut self, program: &Program, mem: &M) -> Result<Vec<InterpreterValue>>
    where
        M: VirtualMemory,
    {
        let frame = self.frames.pop().unwrap();
        let mut pc = frame.pc;
        let mut base = frame.base;
        let mut stack_base = frame.stack_base;
        let mut func = frame.func.clone();
        let mut mid = frame.mid;

        'main_loop: loop {
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

            // Execute register moves from data section
            macro_rules! execute_moves {
                ($target:expr) => {{
                    let num_moves = $target.num_moves as usize;
                    let moves_offset = $target.moves_offset as usize;
                    for i in 0..num_moves {
                        let dst = func.data_section.u16_data[moves_offset + i * 2];
                        let src = func.data_section.u16_data[moves_offset + i * 2 + 1];
                        set!(dst, get!(src));
                    }
                }};
            }

            // === Call Preparation Macro ===
            macro_rules! prepare_call {
                ($target_mid:expr, $target_fid:expr, $dst_regs_start:expr, $dst_regs_count:expr, $args:expr) => {{
                    if program.modules[$target_mid].compiled[$target_fid].is_none() {
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

            loop {
                unsafe {
                    let inst = func.code.get_unchecked(pc);
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
                        Opcode::I32Add => {
                            let (a, b) =
                                (get!(inst.src1).unwrap_i32(), get!(inst.src2).unwrap_i32());
                            set!(inst.dst, InterpreterValue::i32(a.wrapping_add(b)));
                        }
                        Opcode::I32AddImm => {
                            let a = get!(inst.src1).unwrap_i32();
                            set!(
                                inst.dst,
                                InterpreterValue::i32(a.wrapping_add(inst.imm32() as i32))
                            );
                        }
                        Opcode::I32Sub => {
                            let (a, b) =
                                (get!(inst.src1).unwrap_i32(), get!(inst.src2).unwrap_i32());
                            set!(inst.dst, InterpreterValue::i32(a.wrapping_sub(b)));
                        }
                        Opcode::I32SubImm => {
                            let a = get!(inst.src1).unwrap_i32();
                            set!(
                                inst.dst,
                                InterpreterValue::i32(a.wrapping_sub(inst.imm32() as i32))
                            );
                        }
                        Opcode::I32Mul => {
                            let (a, b) =
                                (get!(inst.src1).unwrap_i32(), get!(inst.src2).unwrap_i32());
                            set!(inst.dst, InterpreterValue::i32(a.wrapping_mul(b)));
                        }
                        Opcode::I32DivS => {
                            let (a, b) =
                                (get!(inst.src1).unwrap_i32(), get!(inst.src2).unwrap_i32());
                            set!(inst.dst, InterpreterValue::i32(a.wrapping_div(b)));
                        }
                        Opcode::I32DivU => {
                            let (a, b) = (
                                get!(inst.src1).unwrap_i32() as u32,
                                get!(inst.src2).unwrap_i32() as u32,
                            );
                            set!(inst.dst, InterpreterValue::i32(a.wrapping_div(b) as i32));
                        }
                        Opcode::I32RemS => {
                            let (a, b) =
                                (get!(inst.src1).unwrap_i32(), get!(inst.src2).unwrap_i32());
                            set!(inst.dst, InterpreterValue::i32(a.wrapping_rem(b)));
                        }
                        Opcode::I32RemU => {
                            let (a, b) = (
                                get!(inst.src1).unwrap_i32() as u32,
                                get!(inst.src2).unwrap_i32() as u32,
                            );
                            set!(inst.dst, InterpreterValue::i32(a.wrapping_rem(b) as i32));
                        }
                        Opcode::I32And => {
                            let (a, b) =
                                (get!(inst.src1).unwrap_i32(), get!(inst.src2).unwrap_i32());
                            set!(inst.dst, InterpreterValue::i32(a & b));
                        }
                        Opcode::I32AndImm => {
                            let a = get!(inst.src1).unwrap_i32();
                            set!(inst.dst, InterpreterValue::i32(a & inst.imm32() as i32));
                        }
                        Opcode::I32Or => {
                            let (a, b) =
                                (get!(inst.src1).unwrap_i32(), get!(inst.src2).unwrap_i32());
                            set!(inst.dst, InterpreterValue::i32(a | b));
                        }
                        Opcode::I32OrImm => {
                            let a = get!(inst.src1).unwrap_i32();
                            set!(inst.dst, InterpreterValue::i32(a | inst.imm32() as i32));
                        }
                        Opcode::I32Xor => {
                            let (a, b) =
                                (get!(inst.src1).unwrap_i32(), get!(inst.src2).unwrap_i32());
                            set!(inst.dst, InterpreterValue::i32(a ^ b));
                        }
                        Opcode::I32XorImm => {
                            let a = get!(inst.src1).unwrap_i32();
                            set!(inst.dst, InterpreterValue::i32(a ^ inst.imm32() as i32));
                        }
                        Opcode::I32Shl => {
                            let (a, b) =
                                (get!(inst.src1).unwrap_i32(), get!(inst.src2).unwrap_i32());
                            set!(inst.dst, InterpreterValue::i32(a.wrapping_shl(b as u32)));
                        }
                        Opcode::I32ShlImm => {
                            let a = get!(inst.src1).unwrap_i32();
                            set!(
                                inst.dst,
                                InterpreterValue::i32(a.wrapping_shl(inst.imm32() as u32))
                            );
                        }
                        Opcode::I32ShrS => {
                            let (a, b) =
                                (get!(inst.src1).unwrap_i32(), get!(inst.src2).unwrap_i32());
                            set!(inst.dst, InterpreterValue::i32(a.wrapping_shr(b as u32)));
                        }
                        Opcode::I32ShrSImm => {
                            let a = get!(inst.src1).unwrap_i32();
                            set!(
                                inst.dst,
                                InterpreterValue::i32(a.wrapping_shr(inst.imm32() as u32))
                            );
                        }
                        Opcode::I32ShrU => {
                            let (a, b) = (
                                get!(inst.src1).unwrap_i32() as u32,
                                get!(inst.src2).unwrap_i32() as u32,
                            );
                            set!(inst.dst, InterpreterValue::i32(a.wrapping_shr(b) as i32));
                        }
                        Opcode::I32ShrUImm => {
                            let a = get!(inst.src1).unwrap_i32() as u32;
                            set!(
                                inst.dst,
                                InterpreterValue::i32(a.wrapping_shr(inst.imm32() as u32) as i32)
                            );
                        }
                        Opcode::I32RotL => {
                            let (a, b) =
                                (get!(inst.src1).unwrap_i32(), get!(inst.src2).unwrap_i32());
                            set!(inst.dst, InterpreterValue::i32(a.rotate_left(b as u32)));
                        }
                        Opcode::I32RotR => {
                            let (a, b) =
                                (get!(inst.src1).unwrap_i32(), get!(inst.src2).unwrap_i32());
                            set!(inst.dst, InterpreterValue::i32(a.rotate_right(b as u32)));
                        }
                        Opcode::I32Clz => set!(
                            inst.dst,
                            InterpreterValue::i32(
                                get!(inst.src1).unwrap_i32().leading_zeros() as i32
                            )
                        ),
                        Opcode::I32Ctz => set!(
                            inst.dst,
                            InterpreterValue::i32(
                                get!(inst.src1).unwrap_i32().trailing_zeros() as i32
                            )
                        ),
                        Opcode::I32Popcnt => set!(
                            inst.dst,
                            InterpreterValue::i32(get!(inst.src1).unwrap_i32().count_ones() as i32)
                        ),
                        Opcode::I32Eqz => set!(
                            inst.dst,
                            InterpreterValue::bool(get!(inst.src1).unwrap_i32() == 0)
                        ),
                        Opcode::I32Eq => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                get!(inst.src1).unwrap_i32() == get!(inst.src2).unwrap_i32()
                            )
                        ),
                        Opcode::I32Ne => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                get!(inst.src1).unwrap_i32() != get!(inst.src2).unwrap_i32()
                            )
                        ),
                        Opcode::I32LtS => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                get!(inst.src1).unwrap_i32() < get!(inst.src2).unwrap_i32()
                            )
                        ),
                        Opcode::I32LtU => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                (get!(inst.src1).unwrap_i32() as u32)
                                    < (get!(inst.src2).unwrap_i32() as u32)
                            )
                        ),
                        Opcode::I32LeS => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                get!(inst.src1).unwrap_i32() <= get!(inst.src2).unwrap_i32()
                            )
                        ),
                        Opcode::I32LeU => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                (get!(inst.src1).unwrap_i32() as u32)
                                    <= (get!(inst.src2).unwrap_i32() as u32)
                            )
                        ),
                        Opcode::I32GtS => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                get!(inst.src1).unwrap_i32() > get!(inst.src2).unwrap_i32()
                            )
                        ),
                        Opcode::I32GtU => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                (get!(inst.src1).unwrap_i32() as u32)
                                    > (get!(inst.src2).unwrap_i32() as u32)
                            )
                        ),
                        Opcode::I32GeS => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                get!(inst.src1).unwrap_i32() >= get!(inst.src2).unwrap_i32()
                            )
                        ),
                        Opcode::I32GeU => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                (get!(inst.src1).unwrap_i32() as u32)
                                    >= (get!(inst.src2).unwrap_i32() as u32)
                            )
                        ),

                        // === I64 Operations ===
                        Opcode::I64Add => {
                            let (a, b) =
                                (get!(inst.src1).unwrap_i64(), get!(inst.src2).unwrap_i64());
                            set!(inst.dst, InterpreterValue::i64(a.wrapping_add(b)));
                        }
                        Opcode::I64AddImm => {
                            let a = get!(inst.src1).unwrap_i64();
                            set!(
                                inst.dst,
                                InterpreterValue::i64(a.wrapping_add(inst.imm64 as i64))
                            );
                        }
                        Opcode::I64Sub => {
                            let (a, b) =
                                (get!(inst.src1).unwrap_i64(), get!(inst.src2).unwrap_i64());
                            set!(inst.dst, InterpreterValue::i64(a.wrapping_sub(b)));
                        }
                        Opcode::I64SubImm => {
                            let a = get!(inst.src1).unwrap_i64();
                            set!(
                                inst.dst,
                                InterpreterValue::i64(a.wrapping_sub(inst.imm64 as i64))
                            );
                        }
                        Opcode::I64Mul => {
                            let (a, b) =
                                (get!(inst.src1).unwrap_i64(), get!(inst.src2).unwrap_i64());
                            set!(inst.dst, InterpreterValue::i64(a.wrapping_mul(b)));
                        }
                        Opcode::I64DivS => {
                            let (a, b) =
                                (get!(inst.src1).unwrap_i64(), get!(inst.src2).unwrap_i64());
                            set!(inst.dst, InterpreterValue::i64(a.wrapping_div(b)));
                        }
                        Opcode::I64DivU => {
                            let (a, b) = (
                                get!(inst.src1).unwrap_i64() as u64,
                                get!(inst.src2).unwrap_i64() as u64,
                            );
                            set!(inst.dst, InterpreterValue::i64(a.wrapping_div(b) as i64));
                        }
                        Opcode::I64RemS => {
                            let (a, b) =
                                (get!(inst.src1).unwrap_i64(), get!(inst.src2).unwrap_i64());
                            set!(inst.dst, InterpreterValue::i64(a.wrapping_rem(b)));
                        }

                        Opcode::I64RemU => {
                            let (a, b) = (
                                get!(inst.src1).unwrap_i64() as u64,
                                get!(inst.src2).unwrap_i64() as u64,
                            );
                            set!(inst.dst, InterpreterValue::i64(a.wrapping_rem(b) as i64));
                        }
                        Opcode::I64And => {
                            let (a, b) =
                                (get!(inst.src1).unwrap_i64(), get!(inst.src2).unwrap_i64());
                            set!(inst.dst, InterpreterValue::i64(a & b));
                        }
                        Opcode::I64AndImm => set!(
                            inst.dst,
                            InterpreterValue::i64(get!(inst.src1).unwrap_i64() & inst.imm64 as i64)
                        ),
                        Opcode::I64Or => {
                            let (a, b) =
                                (get!(inst.src1).unwrap_i64(), get!(inst.src2).unwrap_i64());
                            set!(inst.dst, InterpreterValue::i64(a | b));
                        }
                        Opcode::I64OrImm => set!(
                            inst.dst,
                            InterpreterValue::i64(get!(inst.src1).unwrap_i64() | inst.imm64 as i64)
                        ),
                        Opcode::I64Xor => {
                            let (a, b) =
                                (get!(inst.src1).unwrap_i64(), get!(inst.src2).unwrap_i64());
                            set!(inst.dst, InterpreterValue::i64(a ^ b));
                        }
                        Opcode::I64XorImm => set!(
                            inst.dst,
                            InterpreterValue::i64(get!(inst.src1).unwrap_i64() ^ inst.imm64 as i64)
                        ),
                        Opcode::I64Shl => {
                            let (a, b) =
                                (get!(inst.src1).unwrap_i64(), get!(inst.src2).unwrap_i64());
                            set!(inst.dst, InterpreterValue::i64(a.wrapping_shl(b as u32)));
                        }
                        Opcode::I64ShlImm => set!(
                            inst.dst,
                            InterpreterValue::i64(
                                get!(inst.src1).unwrap_i64().wrapping_shl(inst.imm64 as u32)
                            )
                        ),
                        Opcode::I64ShrS => {
                            let (a, b) =
                                (get!(inst.src1).unwrap_i64(), get!(inst.src2).unwrap_i64());
                            set!(inst.dst, InterpreterValue::i64(a.wrapping_shr(b as u32)));
                        }
                        Opcode::I64ShrSImm => set!(
                            inst.dst,
                            InterpreterValue::i64(
                                get!(inst.src1).unwrap_i64().wrapping_shr(inst.imm64 as u32)
                            )
                        ),
                        Opcode::I64ShrU => {
                            let (a, b) = (
                                get!(inst.src1).unwrap_i64() as u64,
                                get!(inst.src2).unwrap_i64() as u32,
                            );
                            set!(inst.dst, InterpreterValue::i64(a.wrapping_shr(b) as i64));
                        }
                        Opcode::I64ShrUImm => set!(
                            inst.dst,
                            InterpreterValue::i64(
                                (get!(inst.src1).unwrap_i64() as u64)
                                    .wrapping_shr(inst.imm64 as u32)
                                    as i64
                            )
                        ),
                        Opcode::I64RotL => {
                            let (a, b) =
                                (get!(inst.src1).unwrap_i64(), get!(inst.src2).unwrap_i64());
                            set!(inst.dst, InterpreterValue::i64(a.rotate_left(b as u32)));
                        }
                        Opcode::I64RotR => {
                            let (a, b) =
                                (get!(inst.src1).unwrap_i64(), get!(inst.src2).unwrap_i64());
                            set!(inst.dst, InterpreterValue::i64(a.rotate_right(b as u32)));
                        }
                        Opcode::I64Clz => set!(
                            inst.dst,
                            InterpreterValue::i64(
                                get!(inst.src1).unwrap_i64().leading_zeros() as i64
                            )
                        ),
                        Opcode::I64Ctz => set!(
                            inst.dst,
                            InterpreterValue::i64(
                                get!(inst.src1).unwrap_i64().trailing_zeros() as i64
                            )
                        ),
                        Opcode::I64Popcnt => set!(
                            inst.dst,
                            InterpreterValue::i64(get!(inst.src1).unwrap_i64().count_ones() as i64)
                        ),
                        Opcode::I64Eqz => set!(
                            inst.dst,
                            InterpreterValue::bool(get!(inst.src1).unwrap_i64() == 0)
                        ),
                        Opcode::I64Eq => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                get!(inst.src1).unwrap_i64() == get!(inst.src2).unwrap_i64()
                            )
                        ),
                        Opcode::I64Ne => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                get!(inst.src1).unwrap_i64() != get!(inst.src2).unwrap_i64()
                            )
                        ),
                        Opcode::I64LtS => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                get!(inst.src1).unwrap_i64() < get!(inst.src2).unwrap_i64()
                            )
                        ),
                        Opcode::I64LtU => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                (get!(inst.src1).unwrap_i64() as u64)
                                    < (get!(inst.src2).unwrap_i64() as u64)
                            )
                        ),
                        Opcode::I64LeS => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                get!(inst.src1).unwrap_i64() <= get!(inst.src2).unwrap_i64()
                            )
                        ),
                        Opcode::I64LeU => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                (get!(inst.src1).unwrap_i64() as u64)
                                    <= (get!(inst.src2).unwrap_i64() as u64)
                            )
                        ),
                        Opcode::I64GtS => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                get!(inst.src1).unwrap_i64() > get!(inst.src2).unwrap_i64()
                            )
                        ),
                        Opcode::I64GtU => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                (get!(inst.src1).unwrap_i64() as u64)
                                    > (get!(inst.src2).unwrap_i64() as u64)
                            )
                        ),
                        Opcode::I64GeS => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                get!(inst.src1).unwrap_i64() >= get!(inst.src2).unwrap_i64()
                            )
                        ),
                        Opcode::I64GeU => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                (get!(inst.src1).unwrap_i64() as u64)
                                    >= (get!(inst.src2).unwrap_i64() as u64)
                            )
                        ),

                        // === F32 Operations ===
                        Opcode::F32Add => set!(
                            inst.dst,
                            InterpreterValue::f32(
                                get!(inst.src1).unwrap_f32() + get!(inst.src2).unwrap_f32()
                            )
                        ),
                        Opcode::F32Sub => set!(
                            inst.dst,
                            InterpreterValue::f32(
                                get!(inst.src1).unwrap_f32() - get!(inst.src2).unwrap_f32()
                            )
                        ),
                        Opcode::F32Mul => set!(
                            inst.dst,
                            InterpreterValue::f32(
                                get!(inst.src1).unwrap_f32() * get!(inst.src2).unwrap_f32()
                            )
                        ),
                        Opcode::F32Div => set!(
                            inst.dst,
                            InterpreterValue::f32(
                                get!(inst.src1).unwrap_f32() / get!(inst.src2).unwrap_f32()
                            )
                        ),
                        Opcode::F32Abs => set!(
                            inst.dst,
                            InterpreterValue::f32(get!(inst.src1).unwrap_f32().abs())
                        ),
                        Opcode::F32Neg => set!(
                            inst.dst,
                            InterpreterValue::f32(-get!(inst.src1).unwrap_f32())
                        ),
                        Opcode::F32Sqrt => set!(
                            inst.dst,
                            InterpreterValue::f32(get!(inst.src1).unwrap_f32().sqrt())
                        ),
                        Opcode::F32Ceil => set!(
                            inst.dst,
                            InterpreterValue::f32(get!(inst.src1).unwrap_f32().ceil())
                        ),
                        Opcode::F32Floor => set!(
                            inst.dst,
                            InterpreterValue::f32(get!(inst.src1).unwrap_f32().floor())
                        ),
                        Opcode::F32Trunc => set!(
                            inst.dst,
                            InterpreterValue::f32(get!(inst.src1).unwrap_f32().trunc())
                        ),
                        Opcode::F32Nearest => set!(
                            inst.dst,
                            InterpreterValue::f32(get!(inst.src1).unwrap_f32().round_ties_even())
                        ),
                        Opcode::F32Min => set!(
                            inst.dst,
                            InterpreterValue::f32(
                                get!(inst.src1)
                                    .unwrap_f32()
                                    .min(get!(inst.src2).unwrap_f32())
                            )
                        ),
                        Opcode::F32Max => set!(
                            inst.dst,
                            InterpreterValue::f32(
                                get!(inst.src1)
                                    .unwrap_f32()
                                    .max(get!(inst.src2).unwrap_f32())
                            )
                        ),
                        Opcode::F32CopySign => set!(
                            inst.dst,
                            InterpreterValue::f32(
                                get!(inst.src1)
                                    .unwrap_f32()
                                    .copysign(get!(inst.src2).unwrap_f32())
                            )
                        ),
                        Opcode::F32Eq => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                get!(inst.src1).unwrap_f32() == get!(inst.src2).unwrap_f32()
                            )
                        ),
                        Opcode::F32Ne => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                get!(inst.src1).unwrap_f32() != get!(inst.src2).unwrap_f32()
                            )
                        ),
                        Opcode::F32Lt => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                get!(inst.src1).unwrap_f32() < get!(inst.src2).unwrap_f32()
                            )
                        ),
                        Opcode::F32Le => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                get!(inst.src1).unwrap_f32() <= get!(inst.src2).unwrap_f32()
                            )
                        ),
                        Opcode::F32Gt => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                get!(inst.src1).unwrap_f32() > get!(inst.src2).unwrap_f32()
                            )
                        ),
                        Opcode::F32Ge => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                get!(inst.src1).unwrap_f32() >= get!(inst.src2).unwrap_f32()
                            )
                        ),

                        // === F64 Operations ===
                        Opcode::F64Add => set!(
                            inst.dst,
                            InterpreterValue::f64(
                                get!(inst.src1).unwrap_f64() + get!(inst.src2).unwrap_f64()
                            )
                        ),
                        Opcode::F64Sub => set!(
                            inst.dst,
                            InterpreterValue::f64(
                                get!(inst.src1).unwrap_f64() - get!(inst.src2).unwrap_f64()
                            )
                        ),
                        Opcode::F64Mul => set!(
                            inst.dst,
                            InterpreterValue::f64(
                                get!(inst.src1).unwrap_f64() * get!(inst.src2).unwrap_f64()
                            )
                        ),
                        Opcode::F64Div => set!(
                            inst.dst,
                            InterpreterValue::f64(
                                get!(inst.src1).unwrap_f64() / get!(inst.src2).unwrap_f64()
                            )
                        ),
                        Opcode::F64Abs => set!(
                            inst.dst,
                            InterpreterValue::f64(get!(inst.src1).unwrap_f64().abs())
                        ),
                        Opcode::F64Neg => set!(
                            inst.dst,
                            InterpreterValue::f64(-get!(inst.src1).unwrap_f64())
                        ),
                        Opcode::F64Sqrt => set!(
                            inst.dst,
                            InterpreterValue::f64(get!(inst.src1).unwrap_f64().sqrt())
                        ),
                        Opcode::F64Ceil => set!(
                            inst.dst,
                            InterpreterValue::f64(get!(inst.src1).unwrap_f64().ceil())
                        ),
                        Opcode::F64Floor => set!(
                            inst.dst,
                            InterpreterValue::f64(get!(inst.src1).unwrap_f64().floor())
                        ),
                        Opcode::F64Trunc => set!(
                            inst.dst,
                            InterpreterValue::f64(get!(inst.src1).unwrap_f64().trunc())
                        ),
                        Opcode::F64Nearest => set!(
                            inst.dst,
                            InterpreterValue::f64(get!(inst.src1).unwrap_f64().round_ties_even())
                        ),
                        Opcode::F64Min => set!(
                            inst.dst,
                            InterpreterValue::f64(
                                get!(inst.src1)
                                    .unwrap_f64()
                                    .min(get!(inst.src2).unwrap_f64())
                            )
                        ),
                        Opcode::F64Max => set!(
                            inst.dst,
                            InterpreterValue::f64(
                                get!(inst.src1)
                                    .unwrap_f64()
                                    .max(get!(inst.src2).unwrap_f64())
                            )
                        ),
                        Opcode::F64CopySign => set!(
                            inst.dst,
                            InterpreterValue::f64(
                                get!(inst.src1)
                                    .unwrap_f64()
                                    .copysign(get!(inst.src2).unwrap_f64())
                            )
                        ),
                        Opcode::F64Eq => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                get!(inst.src1).unwrap_f64() == get!(inst.src2).unwrap_f64()
                            )
                        ),
                        Opcode::F64Ne => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                get!(inst.src1).unwrap_f64() != get!(inst.src2).unwrap_f64()
                            )
                        ),
                        Opcode::F64Lt => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                get!(inst.src1).unwrap_f64() < get!(inst.src2).unwrap_f64()
                            )
                        ),
                        Opcode::F64Le => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                get!(inst.src1).unwrap_f64() <= get!(inst.src2).unwrap_f64()
                            )
                        ),
                        Opcode::F64Gt => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                get!(inst.src1).unwrap_f64() > get!(inst.src2).unwrap_f64()
                            )
                        ),
                        Opcode::F64Ge => set!(
                            inst.dst,
                            InterpreterValue::bool(
                                get!(inst.src1).unwrap_f64() >= get!(inst.src2).unwrap_f64()
                            )
                        ),

                        // === Conversions ===
                        Opcode::ExtendS => {
                            let val = get!(inst.src1).unwrap_i64();
                            let (from_ty, to_ty) = inst.conv_types();
                            let res = match from_ty {
                                ScalarType::I8 => val as i8 as i64,
                                ScalarType::I16 => val as i16 as i64,
                                ScalarType::I32 => val as i32 as i64,
                                _ => panic!("Unsupported ExtendS from_ty: {:?}", from_ty),
                            };
                            set!(
                                inst.dst,
                                if to_ty == ScalarType::I32 {
                                    InterpreterValue::i32(res as i32)
                                } else {
                                    InterpreterValue::i64(res)
                                }
                            );
                        }
                        Opcode::ExtendU => {
                            let val = get!(inst.src1).unwrap_i64();
                            let (from_ty, to_ty) = inst.conv_types();
                            let res = match from_ty {
                                ScalarType::I8 => (val as u8) as u64 as i64,
                                ScalarType::I16 => (val as u16) as u64 as i64,
                                ScalarType::I32 => (val as u32) as u64 as i64,
                                _ => panic!("Unsupported ExtendU from_ty: {:?}", from_ty),
                            };
                            set!(
                                inst.dst,
                                if to_ty == ScalarType::I32 {
                                    InterpreterValue::i32(res as i32)
                                } else {
                                    InterpreterValue::i64(res)
                                }
                            );
                        }
                        Opcode::Wrap => set!(
                            inst.dst,
                            InterpreterValue::i32(get!(inst.src1).unwrap_i64() as i32)
                        ),

                        Opcode::I32TruncF32S => set!(
                            inst.dst,
                            InterpreterValue::i32(get!(inst.src1).unwrap_f32() as i32)
                        ),
                        Opcode::I32TruncF32U => set!(
                            inst.dst,
                            InterpreterValue::i32(get!(inst.src1).unwrap_f32() as u32 as i32)
                        ),
                        Opcode::I32TruncF64S => set!(
                            inst.dst,
                            InterpreterValue::i32(get!(inst.src1).unwrap_f64() as i32)
                        ),
                        Opcode::I32TruncF64U => set!(
                            inst.dst,
                            InterpreterValue::i32(get!(inst.src1).unwrap_f64() as u32 as i32)
                        ),
                        Opcode::I64TruncF32S => set!(
                            inst.dst,
                            InterpreterValue::i64(get!(inst.src1).unwrap_f32() as i64)
                        ),
                        Opcode::I64TruncF32U => set!(
                            inst.dst,
                            InterpreterValue::i64(get!(inst.src1).unwrap_f32() as u64 as i64)
                        ),
                        Opcode::I64TruncF64S => set!(
                            inst.dst,
                            InterpreterValue::i64(get!(inst.src1).unwrap_f64() as i64)
                        ),
                        Opcode::I64TruncF64U => set!(
                            inst.dst,
                            InterpreterValue::i64(get!(inst.src1).unwrap_f64() as u64 as i64)
                        ),

                        Opcode::F32DemoteF64 => set!(
                            inst.dst,
                            InterpreterValue::f32(get!(inst.src1).unwrap_f64() as f32)
                        ),
                        Opcode::F64PromoteF32 => set!(
                            inst.dst,
                            InterpreterValue::f64(get!(inst.src1).unwrap_f32() as f64)
                        ),
                        Opcode::Bitcast => set!(inst.dst, get!(inst.src1)),

                        Opcode::I32TruncSatF32S => {
                            let val = get!(inst.src1).unwrap_f32();
                            set!(
                                inst.dst,
                                InterpreterValue::i32(if val.is_nan() { 0 } else { val as i32 })
                            );
                        }
                        Opcode::I32TruncSatF32U => {
                            let val = get!(inst.src1).unwrap_f32();
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
                            let val = get!(inst.src1).unwrap_f64();
                            set!(
                                inst.dst,
                                InterpreterValue::i32(if val.is_nan() { 0 } else { val as i32 })
                            );
                        }
                        Opcode::I32TruncSatF64U => {
                            let val = get!(inst.src1).unwrap_f64();
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
                            let val = get!(inst.src1).unwrap_f32();
                            set!(
                                inst.dst,
                                InterpreterValue::i64(if val.is_nan() { 0 } else { val as i64 })
                            );
                        }
                        Opcode::I64TruncSatF32U => {
                            let val = get!(inst.src1).unwrap_f32();
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
                            let val = get!(inst.src1).unwrap_f64();
                            set!(
                                inst.dst,
                                InterpreterValue::i64(if val.is_nan() { 0 } else { val as i64 })
                            );
                        }
                        Opcode::I64TruncSatF64U => {
                            let val = get!(inst.src1).unwrap_f64();
                            set!(
                                inst.dst,
                                InterpreterValue::i64(if val.is_nan() || val < 0.0 {
                                    0
                                } else {
                                    val as u64
                                } as i64)
                            );
                        }
                        Opcode::F32ConvertI32S => set!(
                            inst.dst,
                            InterpreterValue::f32(get!(inst.src1).unwrap_i32() as f32)
                        ),
                        Opcode::F32ConvertI32U => set!(
                            inst.dst,
                            InterpreterValue::f32(get!(inst.src1).unwrap_i32() as u32 as f32)
                        ),
                        Opcode::F32ConvertI64S => set!(
                            inst.dst,
                            InterpreterValue::f32(get!(inst.src1).unwrap_i64() as f32)
                        ),
                        Opcode::F32ConvertI64U => set!(
                            inst.dst,
                            InterpreterValue::f32(get!(inst.src1).unwrap_i64() as u64 as f32)
                        ),
                        Opcode::F64ConvertI32S => set!(
                            inst.dst,
                            InterpreterValue::f64(get!(inst.src1).unwrap_i32() as f64)
                        ),
                        Opcode::F64ConvertI32U => set!(
                            inst.dst,
                            InterpreterValue::f64(get!(inst.src1).unwrap_i32() as u32 as f64)
                        ),
                        Opcode::F64ConvertI64S => set!(
                            inst.dst,
                            InterpreterValue::f64(get!(inst.src1).unwrap_i64() as f64)
                        ),
                        Opcode::F64ConvertI64U => set!(
                            inst.dst,
                            InterpreterValue::f64(get!(inst.src1).unwrap_i64() as u64 as f64)
                        ),

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
                            store!(inst.src1, inst.src2, inst.imm32(), unwrap_i32, i32)
                        }
                        Opcode::I64Store => {
                            store!(inst.src1, inst.src2, inst.imm32(), unwrap_i64, i64)
                        }
                        Opcode::I8Store => {
                            store!(inst.src1, inst.src2, inst.imm32(), unwrap_i64, u8)
                        }
                        Opcode::I16Store => {
                            store!(inst.src1, inst.src2, inst.imm32(), unwrap_i64, u16)
                        }
                        Opcode::F32Store => {
                            store!(inst.src1, inst.src2, inst.imm32(), unwrap_f32, f32)
                        }
                        Opcode::F64Store => {
                            store!(inst.src1, inst.src2, inst.imm32(), unwrap_f64, f64)
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
                            let ty = inst.stack_type();
                            set!(
                                inst.dst,
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
                        Opcode::StackStore => {
                            let addr = stack_base + inst.imm32() as usize;
                            let ptr = self.stack_memory.as_mut_ptr().add(addr);
                            let val = get!(inst.src1);
                            let ty = inst.stack_type();
                            match ty {
                                ScalarType::I8 => {
                                    (ptr as *mut i8).write_unaligned(val.unwrap_i32() as i8)
                                }
                                ScalarType::I16 => {
                                    (ptr as *mut i16).write_unaligned(val.unwrap_i32() as i16)
                                }
                                ScalarType::I32 => {
                                    (ptr as *mut i32).write_unaligned(val.unwrap_i32())
                                }
                                ScalarType::I64 | ScalarType::Ptr => {
                                    (ptr as *mut i64).write_unaligned(val.unwrap_i64())
                                }
                                ScalarType::F32 => {
                                    (ptr as *mut f32).write_unaligned(val.unwrap_f32())
                                }
                                ScalarType::F64 => {
                                    (ptr as *mut f64).write_unaligned(val.unwrap_f64())
                                }
                                _ => panic!("Unknown type {:?} in StackStore", ty),
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
                            let cond = get!(inst.dst).unwrap_bool();
                            let target_idx = inst.br_target(cond);
                            let target = &func.data_section.jump_targets[target_idx as usize];
                            execute_moves!(target);
                            pc = target.pc as usize;
                        }
                        Opcode::BrTable => {
                            let idx = get!(inst.dst).unwrap_i32();
                            let num = inst.br_table_num_targets() as usize;
                            let target_idx = if idx >= 0 && (idx as usize) < num {
                                idx as usize
                            } else {
                                num - 1
                            };

                            let target = &func.data_section.jump_targets
                                [inst.br_table_base_idx() as usize + target_idx];
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

                            match program.modules[mid].links[f_id] {
                                ImportTarget::Module(m, f) => {
                                    prepare_call!(m, f, dst_start, rets as usize, self.args_buffer);
                                }
                                ImportTarget::Host(h_id) => {
                                    let args = self.args_buffer.len();
                                    program.host_functions_list[h_id]
                                        .call(&mut self.args_buffer, args);
                                    values_ptr = self.value_stack.as_mut_ptr();
                                    for i in 0..rets as usize {
                                        let dst = self.dst_regs_buffer[dst_start + i];
                                        if dst != 0 {
                                            *reg!(dst) = self.args_buffer[i];
                                        }
                                    }
                                    self.dst_regs_buffer.truncate(dst_start);
                                }
                                ImportTarget::None => {
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

                            match program.decode_ptr(ptr) {
                                Some(ImportTarget::Module(m, f)) => {
                                    prepare_call!(m, f, dst_start, rets as usize, self.args_buffer);
                                }
                                Some(ImportTarget::Host(h_id)) => {
                                    let args = self.args_buffer.len();
                                    program.host_functions_list[h_id]
                                        .call(&mut self.args_buffer, args);
                                    values_ptr = self.value_stack.as_mut_ptr();
                                    for i in 0..rets as usize {
                                        let dst = self.dst_regs_buffer[dst_start + i];
                                        if dst != 0 {
                                            *reg!(dst) = self.args_buffer[i];
                                        }
                                    }
                                    self.dst_regs_buffer.truncate(dst_start);
                                }
                                _ => panic!("Invalid function pointer: {:x}", ptr),
                            }
                        }
                        Opcode::PtrIndex => {
                            let ptr = get!(inst.src1).unwrap_i64();
                            let idx = get!(inst.src2).unwrap_i64();
                            let s = inst.ptr_index_scale();
                            let o = inst.ptr_index_offset();
                            set!(
                                inst.dst,
                                InterpreterValue::i64(
                                    ptr.wrapping_add(idx.wrapping_mul(s)).wrapping_add(o)
                                )
                            );
                        }
                        Opcode::Select => {
                            let e = inst.select_false_reg();
                            set!(
                                inst.dst,
                                if get!(inst.src1).unwrap_bool() {
                                    get!(inst.src2)
                                } else {
                                    get!(e)
                                }
                            );
                        }
                        Opcode::RegMove => set!(inst.dst, get!(inst.src1)),
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
                        Opcode::Unreachable => return Err(crate::error::Error::Unreachable),
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
