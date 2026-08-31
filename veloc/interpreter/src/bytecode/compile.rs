use crate::bytecode::inst::{Instruction, Reg, TypePair, emit};
use cranelift_entity::SecondaryMap;
use smallvec::SmallVec;
use veloc_analyzer::{LiveInterval, UseDefAnalysis, analyze_liveness};
use veloc_ir::{
    Block, BlockCall, FuncId, Function, Inst, InstructionData, Intrinsic, ModuleId,
    Opcode as IrOpcode, ScalarType, StackSlot, Type, Value, ValueList,
};

macro_rules! unary_dispatch_op {
    ($arg_reg:expr, $dst:expr, $ty:expr, $code:expr, $f32_op:ident, $f64_op:ident) => {
        match $ty {
            ScalarType::F32 => emit::$f32_op($code, $dst, $arg_reg),
            ScalarType::F64 => emit::$f64_op($code, $dst, $arg_reg),
            _ => panic!("Unsupported op for type {:?}", $ty),
        }
    };
    ($arg_reg:expr, $dst:expr, $ty:expr, $code:expr, $i32_op:ident, $i64_op:ident, int) => {
        match $ty {
            ScalarType::I32 => emit::$i32_op($code, $dst, $arg_reg),
            ScalarType::I64 => emit::$i64_op($code, $dst, $arg_reg),
            _ => panic!("Unsupported op for type {:?}", $ty),
        }
    };
}

macro_rules! convert_op {
    ($arg:expr, $dst:expr, $code:expr, $from_ty:expr, $to_ty:expr, 
     $f32_i32:ident, $f64_i32:ident, $f32_i64:ident, $f64_i64:ident) => {
        match ($from_ty, $to_ty) {
            (ScalarType::F32, ScalarType::I32) => emit::$f32_i32($code, $dst, $arg),
            (ScalarType::F64, ScalarType::I32) => emit::$f64_i32($code, $dst, $arg),
            (ScalarType::F32, ScalarType::I64) => emit::$f32_i64($code, $dst, $arg),
            (ScalarType::F64, ScalarType::I64) => emit::$f64_i64($code, $dst, $arg),
            _ => panic!("Unsupported conversion: {:?} -> {:?}", $from_ty, $to_ty),
        }
    };
}

macro_rules! convert_int_to_float_op {
    ($arg:expr, $dst:expr, $code:expr, $from_ty:expr, $to_ty:expr, 
     $i32_f32:ident, $i64_f32:ident, $i32_f64:ident, $i64_f64:ident) => {
        match ($from_ty, $to_ty) {
            (ScalarType::I32, ScalarType::F32) => emit::$i32_f32($code, $dst, $arg),
            (ScalarType::I64, ScalarType::F32) => emit::$i64_f32($code, $dst, $arg),
            (ScalarType::I32, ScalarType::F64) => emit::$i32_f64($code, $dst, $arg),
            (ScalarType::I64, ScalarType::F64) => emit::$i64_f64($code, $dst, $arg),
            _ => panic!("Unsupported conversion: {:?} -> {:?}", $from_ty, $to_ty),
        }
    };
}

/// A control-flow target relative to its source instruction.
#[derive(Debug, Clone, Default)]
#[repr(C)]
pub struct JumpTarget {
    /// Signed byte offset from the source instruction.
    pub offset: i32,
    /// Number of register moves
    pub num_moves: u16,
    /// Offset into regs where moves are stored (as pairs of dst, src)
    pub moves_offset: u32,
}

const _: () = assert!(core::mem::size_of::<JumpTarget>() == 12);

fn relative_byte_offset(source_pc: usize, target_pc: u32) -> i64 {
    let source_pc = i64::try_from(source_pc).expect("bytecode source PC overflow");
    let instruction_offset = i64::from(target_pc) - source_pc;
    instruction_offset
        .checked_mul(core::mem::size_of::<Instruction>() as i64)
        .expect("bytecode jump offset overflow")
}

fn relative_i32_byte_offset(source_pc: usize, target_pc: u32) -> i32 {
    i32::try_from(relative_byte_offset(source_pc, target_pc))
        .expect("bytecode jump offset exceeds i32")
}

/// Data section stored as u16 words for efficient register access
/// For Jump targets, they are stored in a separate Vec<JumpTarget>
#[derive(Debug, Clone, Default)]
pub struct DataSection {
    /// Register lists stored as Reg (counts are encoded in instructions)
    pub regs: Vec<Reg>,
    /// Jump targets with direct PC and moves
    pub jump_targets: Vec<JumpTarget>,
}

impl DataSection {
    pub fn new() -> Self {
        Self {
            regs: Vec::new(),
            jump_targets: Vec::new(),
        }
    }

    /// Add return registers for Return instruction
    /// Returns the offset (in u16 words) where data starts
    /// Note: count is encoded in instruction, regs only stores registers
    pub fn add_return_regs(&mut self, regs: &[Reg]) -> u32 {
        let offset = self.regs.len();
        self.regs.extend_from_slice(regs);
        offset as u32
    }

    /// Add call data (ret_regs + arg_regs) for Call/CallIndirect/CallIntrinsic
    /// Returns the offset (in u16 words) where data starts
    /// Note: counts are encoded in instruction, regs only stores registers
    pub fn add_call_data(&mut self, ret_regs: &[Reg], arg_regs: &[Reg]) -> u32 {
        let offset = self.regs.len();
        self.regs.extend_from_slice(ret_regs);
        self.regs.extend_from_slice(arg_regs);
        offset as u32
    }

    /// Add Jump targets with their register moves
    /// Returns (offset in jump_targets, num_targets)
    pub fn add_jump_target(&mut self, target: JumpTarget) -> u32 {
        let offset = self.jump_targets.len();
        self.jump_targets.push(target);
        offset as u32
    }

    #[inline(always)]
    /// Get return register (count is encoded in instruction)
    pub fn return_reg(&self, offset: usize, index: usize) -> Reg {
        self.regs[offset + index]
    }

    #[inline(always)]
    /// Get return register for call data (counts are encoded in instruction)
    pub fn call_ret_reg(&self, offset: usize, index: usize) -> Reg {
        self.regs[offset + index]
    }

    #[inline(always)]
    /// Get argument register for call data (counts are encoded in instruction)
    pub fn call_arg_reg(&self, offset: usize, ret_count: usize, index: usize) -> Reg {
        self.regs[offset + ret_count + index]
    }

    #[inline(always)]
    pub fn jump_move_pair(&self, target: &JumpTarget, index: usize) -> (Reg, Reg) {
        let base = target.moves_offset as usize + index * 2;
        (self.regs[base], self.regs[base + 1])
    }
}

pub struct CompiledFunction {
    pub(crate) module_id: ModuleId,
    pub(crate) func_id: FuncId,
    pub(crate) code: Vec<Instruction>,
    /// Data section: regs for register lists, jump_targets for jump targets
    pub(crate) data_section: DataSection,
    pub(crate) stack_slots_sizes: Vec<usize>,
    pub(crate) param_indices: Vec<Reg>,
    pub(crate) register_count: usize,
}

struct ValueMapper<'a> {
    map: SecondaryMap<Value, Reg>,
    free_registers: Vec<Reg>,
    next_register: u16,
    intervals: &'a SecondaryMap<Value, LiveInterval>,
    block_params: &'a std::collections::HashSet<Value>,
    fused_values: &'a std::collections::HashSet<Value>,
}

impl<'a> ValueMapper<'a> {
    fn use_val(&self, val: Value) -> Reg {
        let reg = self.map[val];
        if reg != Reg::NULL {
            return reg;
        }

        if self.fused_values.contains(&val) {
            panic!("Value {:?} is fused as constant and has no register", val);
        }
        panic!("Value {:?} used before defined or mapping missing", val);
    }

    fn free_if_last_use(&mut self, val: Value, current_pc: u32) {
        // Block params must keep stable registers across all incoming edges.
        if self.block_params.contains(&val) {
            return;
        }

        if let Some(range) = self.intervals.get(val) {
            if current_pc >= range.end().saturating_sub(1) {
                let reg = self.map[val];
                if reg != Reg::NULL {
                    self.map[val] = Reg::NULL;
                    self.free_registers.push(reg);
                }
            }
        }
    }

    fn alloc_val(&mut self, val: Value) -> Reg {
        // Assert that we are not re-allocating a register for the same value
        debug_assert_eq!(
            self.map[val],
            Reg::NULL,
            "Value {:?} already has a register assigned",
            val
        );

        let reg = self.free_registers.pop().unwrap_or_else(|| {
            let r = Reg(self.next_register);
            self.next_register += 1;
            r
        });

        self.map[val] = reg;
        reg
    }

    fn use_block_param(&mut self, param: Value) -> Reg {
        if self.map[param] != Reg::NULL {
            return self.map[param];
        }

        // All block parameters should have been pre-allocated in apply_rpo,
        // but for forward jumps we might not have reached that block yet.
        self.alloc_val(param)
    }
}

impl<'a> ValueMapper<'a> {
    fn new(
        intervals_map: &'a SecondaryMap<Value, LiveInterval>,
        block_params: &'a std::collections::HashSet<Value>,
        fused_values: &'a std::collections::HashSet<Value>,
    ) -> Self {
        Self {
            map: SecondaryMap::new(),
            free_registers: Vec::new(),
            next_register: 1,
            intervals: intervals_map,
            block_params,
            fused_values,
        }
    }
}

/// Try to emit inline bytecode for intrinsics that map directly to opcodes.
/// Returns true if the intrinsic was successfully inlined.
fn try_emit_inline_intrinsic(
    _code: &mut Vec<Instruction>,
    _dst: u16,
    _intrinsic: Intrinsic,
    _args: &[Reg],
) -> bool {
    // Currently no intrinsics are inlineable as bytecode.
    // Math intrinsics (sin, cos, pow, etc.) need runtime libm calls.
    // Memory intrinsics need runtime support.
    false
}

/// Check if a value is a constant that can be fused into a given user instruction.
fn can_fuse_operand(func: &Function, user_inst: Inst, val: Value) -> bool {
    use IrOpcode::*;
    let idata = &func.dfg.instructions[user_inst];
    let constant = func.dfg.as_const(val);

    match idata {
        InstructionData::Binary { opcode, args } => {
            let res = func.dfg.first_result(user_inst).unwrap();
            let ty = func.dfg.value_type(res);
            // Only I32 and I64 binary operations currently support immediate operands in bytecode
            if ty != Type::I32 && ty != Type::I64 {
                return false;
            }

            // Binary instructions in our bytecode only support integer immediates.
            if constant.and_then(|c| c.as_i64()).is_none() {
                return false;
            }

            match opcode {
                IAdd | IAnd | IOr | IXor => {
                    debug_assert!(args[0] == val || args[1] == val);
                    true
                }
                ISub | IShl | IShrS | IShrU => args[1] == val,
                _ => false,
            }
        }
        InstructionData::Unary { opcode, .. } => {
            // Unary instructions can fuse integer, boolean, or float constants.
            if constant.is_none() {
                return false;
            }
            matches!(
                opcode,
                ExtendS | ExtendU | Wrap | FloatDemote | FloatPromote
            )
        }
        _ => false,
    }
}

/// Check if a value is already zero-extended to at least the given bit width.
/// Identify constants that can be fully fused into their user instructions and thus do not need a register.
fn identify_fused_values(func: &Function, rpo: &[Block]) -> std::collections::HashSet<Value> {
    let mut fused_values = std::collections::HashSet::new();
    let use_def = UseDefAnalysis::new(func);
    let mut insts_with_fused_op = std::collections::HashSet::new();

    for &block in rpo {
        for &inst in &func.layout.blocks[block].insts {
            let idata = &func.dfg.instructions[inst];
            if matches!(
                idata,
                InstructionData::Iconst { .. } | InstructionData::Bconst { .. }
            ) {
                let res = func.dfg.first_result(inst).unwrap();
                let users = use_def.users_of(res);

                // A constant can be fused if all its uses support fusion
                // and haven't fused another operand yet.
                let mut all_fusable = true;
                for &user_inst in users {
                    if !can_fuse_operand(func, user_inst, res)
                        || insts_with_fused_op.contains(&user_inst)
                    {
                        all_fusable = false;
                        break;
                    }
                }
                if all_fusable {
                    for &user_inst in users {
                        insts_with_fused_op.insert(user_inst);
                    }
                    fused_values.insert(res);
                }
            }
        }
    }
    fused_values
}

struct Compiler<'a> {
    func: &'a Function,
    liveness: &'a veloc_analyzer::Liveness,
    mapper: ValueMapper<'a>,
    code: Vec<Instruction>,
    data_section: DataSection,
    slot_to_offset: SecondaryMap<StackSlot, u32>,
    block_to_pc: SecondaryMap<Block, u32>,
    jump_fixups: Vec<(usize, Block)>,
    br_fixups: Vec<(usize, Block, Block)>,
    data_fixups: Vec<(usize, Block, usize)>,
    param_indices: Vec<Reg>,
}

impl<'a> Compiler<'a> {
    fn new(
        func: &'a Function,
        liveness: &'a veloc_analyzer::Liveness,
        mapper: ValueMapper<'a>,
    ) -> Self {
        let mut slot_to_offset = SecondaryMap::new();
        let mut current_offset = 0u32;
        for (id, data) in &func.stack_slots {
            slot_to_offset[id] = current_offset;
            current_offset += data.size;
        }

        Self {
            func,
            liveness,
            mapper,
            code: Vec::new(),
            data_section: DataSection::new(),
            slot_to_offset,
            block_to_pc: SecondaryMap::new(),
            jump_fixups: Vec::new(),
            br_fixups: Vec::new(),
            data_fixups: Vec::new(),
            param_indices: Vec::new(),
        }
    }

    fn push_target(
        &mut self,
        target_block: Block,
        moves: Vec<(Reg, Reg)>,
        source_pc: usize,
    ) -> u32 {
        let num_moves = moves.len() as u16;
        let moves_offset = if num_moves == 0 {
            0
        } else {
            let offset = self.data_section.regs.len() as u32;
            for (dst, src) in &moves {
                self.data_section.regs.push(*dst);
                self.data_section.regs.push(*src);
            }
            offset
        };

        let br_offset = self.data_section.add_jump_target(JumpTarget {
            offset: 0,
            num_moves,
            moves_offset,
        });

        self.data_fixups
            .push((br_offset as usize, target_block, source_pc));
        br_offset
    }

    fn emit_binary(&mut self, inst: Inst, opcode: IrOpcode, args: &[Value; 2]) {
        let res = self.func.dfg.first_result(inst).unwrap();
        let ty = self.func.dfg.value_type(res);
        let mut bin = |imm_f: &dyn Fn(&mut Vec<Instruction>, Reg, Reg, i64),
                       reg_f: &dyn Fn(&mut Vec<Instruction>, Reg, Reg, Reg),
                       commutative: bool| {
            let lhs_fused = self.mapper.fused_values.contains(&args[0]);
            let rhs_fused = self.mapper.fused_values.contains(&args[1]);

            if rhs_fused {
                let imm = self.func.dfg.as_const(args[1]).unwrap().as_i64().unwrap();
                let lhs = self.mapper.use_val(args[0]);
                let dst = self.mapper.alloc_val(res);
                imm_f(&mut self.code, dst, lhs, imm);
            } else if commutative && lhs_fused {
                let imm = self.func.dfg.as_const(args[0]).unwrap().as_i64().unwrap();
                let rhs = self.mapper.use_val(args[1]);
                let dst = self.mapper.alloc_val(res);
                imm_f(&mut self.code, dst, rhs, imm);
            } else {
                let lhs = self.mapper.use_val(args[0]);
                let rhs = self.mapper.use_val(args[1]);
                let dst = self.mapper.alloc_val(res);
                reg_f(&mut self.code, dst, lhs, rhs);
            }
        };

        match (ty, opcode) {
            // Bool type uses I32 instructions for bitwise operations
            (Type::BOOL, IrOpcode::IAnd) => bin(
                &|c, d, l, i| emit::I32AndImm(c, d, l, i as u32),
                &emit::I32And,
                true,
            ),
            (Type::BOOL, IrOpcode::IOr) => bin(
                &|c, d, l, i| emit::I32OrImm(c, d, l, i as u32),
                &emit::I32Or,
                true,
            ),
            (Type::BOOL, IrOpcode::IXor) => bin(
                &|c, d, l, i| emit::I32XorImm(c, d, l, i as u32),
                &emit::I32Xor,
                true,
            ),
            (Type::I32, IrOpcode::IAdd) => bin(
                &|c, d, l, i| emit::I32AddImm(c, d, l, i as u32),
                &emit::I32Add,
                true,
            ),
            (Type::I32, IrOpcode::ISub) => bin(
                &|c, d, l, i| emit::I32SubImm(c, d, l, i as u32),
                &emit::I32Sub,
                false,
            ),
            (Type::I32, IrOpcode::IMul) => {
                let lhs = self.mapper.use_val(args[0]);
                let rhs = self.mapper.use_val(args[1]);
                let dst = self.mapper.alloc_val(res);
                emit::I32Mul(&mut self.code, dst, lhs, rhs);
            }
            (Type::I32, IrOpcode::IAnd) => bin(
                &|c, d, l, i| emit::I32AndImm(c, d, l, i as u32),
                &emit::I32And,
                true,
            ),
            (Type::I32, IrOpcode::IOr) => bin(
                &|c, d, l, i| emit::I32OrImm(c, d, l, i as u32),
                &emit::I32Or,
                true,
            ),
            (Type::I32, IrOpcode::IXor) => bin(
                &|c, d, l, i| emit::I32XorImm(c, d, l, i as u32),
                &emit::I32Xor,
                true,
            ),
            (Type::I32, IrOpcode::IShl) => bin(
                &|c, d, l, i| emit::I32ShlImm(c, d, l, i as u32),
                &emit::I32Shl,
                false,
            ),
            (Type::I32, IrOpcode::IShrS) => bin(
                &|c, d, l, i| emit::I32ShrSImm(c, d, l, i as u32),
                &emit::I32ShrS,
                false,
            ),
            (Type::I32, IrOpcode::IShrU) => bin(
                &|c, d, l, i| emit::I32ShrUImm(c, d, l, i as u32),
                &emit::I32ShrU,
                false,
            ),
            (Type::I32, IrOpcode::IDivS) => {
                let lhs = self.mapper.use_val(args[0]);
                let rhs = self.mapper.use_val(args[1]);
                let dst = self.mapper.alloc_val(res);
                emit::I32DivS(&mut self.code, dst, lhs, rhs);
            }
            (Type::I32, IrOpcode::IDivU) => {
                let lhs = self.mapper.use_val(args[0]);
                let rhs = self.mapper.use_val(args[1]);
                let dst = self.mapper.alloc_val(res);
                emit::I32DivU(&mut self.code, dst, lhs, rhs);
            }
            (Type::I32, IrOpcode::IRemS) => {
                let lhs = self.mapper.use_val(args[0]);
                let rhs = self.mapper.use_val(args[1]);
                let dst = self.mapper.alloc_val(res);
                emit::I32RemS(&mut self.code, dst, lhs, rhs);
            }
            (Type::I32, IrOpcode::IRemU) => {
                let lhs = self.mapper.use_val(args[0]);
                let rhs = self.mapper.use_val(args[1]);
                let dst = self.mapper.alloc_val(res);
                emit::I32RemU(&mut self.code, dst, lhs, rhs);
            }
            (Type::I32, IrOpcode::IRotl) => {
                let lhs = self.mapper.use_val(args[0]);
                let rhs = self.mapper.use_val(args[1]);
                let dst = self.mapper.alloc_val(res);
                emit::I32RotL(&mut self.code, dst, lhs, rhs);
            }
            (Type::I32, IrOpcode::IRotr) => {
                let lhs = self.mapper.use_val(args[0]);
                let rhs = self.mapper.use_val(args[1]);
                let dst = self.mapper.alloc_val(res);
                emit::I32RotR(&mut self.code, dst, lhs, rhs);
            }
            (Type::I64, IrOpcode::IAdd) => bin(
                &|c, d, l, i| emit::I64AddImm(c, d, l, i as u64),
                &emit::I64Add,
                true,
            ),
            (Type::I64, IrOpcode::ISub) => bin(
                &|c, d, l, i| emit::I64SubImm(c, d, l, i as u64),
                &emit::I64Sub,
                false,
            ),
            (Type::I64, IrOpcode::IMul) => {
                let lhs = self.mapper.use_val(args[0]);
                let rhs = self.mapper.use_val(args[1]);
                let dst = self.mapper.alloc_val(res);
                emit::I64Mul(&mut self.code, dst, lhs, rhs);
            }
            (Type::I64, IrOpcode::IAnd) => bin(
                &|c, d, l, i| emit::I64AndImm(c, d, l, i as u64),
                &emit::I64And,
                true,
            ),
            (Type::I64, IrOpcode::IOr) => bin(
                &|c, d, l, i| emit::I64OrImm(c, d, l, i as u64),
                &emit::I64Or,
                true,
            ),
            (Type::I64, IrOpcode::IXor) => bin(
                &|c, d, l, i| emit::I64XorImm(c, d, l, i as u64),
                &emit::I64Xor,
                true,
            ),
            (Type::I64, IrOpcode::IShl) => bin(
                &|c, d, l, i| emit::I64ShlImm(c, d, l, i as u64),
                &emit::I64Shl,
                false,
            ),
            (Type::I64, IrOpcode::IShrS) => bin(
                &|c, d, l, i| emit::I64ShrSImm(c, d, l, i as u64),
                &emit::I64ShrS,
                false,
            ),
            (Type::I64, IrOpcode::IShrU) => bin(
                &|c, d, l, i| emit::I64ShrUImm(c, d, l, i as u64),
                &emit::I64ShrU,
                false,
            ),
            (Type::I64, IrOpcode::IDivS) => {
                let lhs = self.mapper.use_val(args[0]);
                let rhs = self.mapper.use_val(args[1]);
                let dst = self.mapper.alloc_val(res);
                emit::I64DivS(&mut self.code, dst, lhs, rhs);
            }
            (Type::I64, IrOpcode::IDivU) => {
                let lhs = self.mapper.use_val(args[0]);
                let rhs = self.mapper.use_val(args[1]);
                let dst = self.mapper.alloc_val(res);
                emit::I64DivU(&mut self.code, dst, lhs, rhs);
            }
            (Type::I64, IrOpcode::IRemS) => {
                let lhs = self.mapper.use_val(args[0]);
                let rhs = self.mapper.use_val(args[1]);
                let dst = self.mapper.alloc_val(res);
                emit::I64RemS(&mut self.code, dst, lhs, rhs);
            }
            (Type::I64, IrOpcode::IRemU) => {
                let lhs = self.mapper.use_val(args[0]);
                let rhs = self.mapper.use_val(args[1]);
                let dst = self.mapper.alloc_val(res);
                emit::I64RemU(&mut self.code, dst, lhs, rhs);
            }
            (Type::I64, IrOpcode::IRotl) => {
                let lhs = self.mapper.use_val(args[0]);
                let rhs = self.mapper.use_val(args[1]);
                let dst = self.mapper.alloc_val(res);
                emit::I64RotL(&mut self.code, dst, lhs, rhs);
            }
            (Type::I64, IrOpcode::IRotr) => {
                let lhs = self.mapper.use_val(args[0]);
                let rhs = self.mapper.use_val(args[1]);
                let dst = self.mapper.alloc_val(res);
                emit::I64RotR(&mut self.code, dst, lhs, rhs);
            }
            (ty, opcode) if ty.is_float() => {
                let lhs = self.mapper.use_val(args[0]);
                let rhs = self.mapper.use_val(args[1]);
                let dst = self.mapper.alloc_val(res);
                let is_f32 = ty == Type::F32;
                match (opcode, is_f32) {
                    (IrOpcode::FAdd, true) => emit::F32Add(&mut self.code, dst, lhs, rhs),
                    (IrOpcode::FAdd, false) => emit::F64Add(&mut self.code, dst, lhs, rhs),
                    (IrOpcode::FSub, true) => emit::F32Sub(&mut self.code, dst, lhs, rhs),
                    (IrOpcode::FSub, false) => emit::F64Sub(&mut self.code, dst, lhs, rhs),
                    (IrOpcode::FMul, true) => emit::F32Mul(&mut self.code, dst, lhs, rhs),
                    (IrOpcode::FMul, false) => emit::F64Mul(&mut self.code, dst, lhs, rhs),
                    (IrOpcode::FDiv, true) => emit::F32Div(&mut self.code, dst, lhs, rhs),
                    (IrOpcode::FDiv, false) => emit::F64Div(&mut self.code, dst, lhs, rhs),
                    (IrOpcode::FMin, true) => emit::F32Min(&mut self.code, dst, lhs, rhs),
                    (IrOpcode::FMin, false) => emit::F64Min(&mut self.code, dst, lhs, rhs),
                    (IrOpcode::FMax, true) => emit::F32Max(&mut self.code, dst, lhs, rhs),
                    (IrOpcode::FMax, false) => emit::F64Max(&mut self.code, dst, lhs, rhs),
                    (IrOpcode::FCopysign, true) => emit::F32CopySign(&mut self.code, dst, lhs, rhs),
                    (IrOpcode::FCopysign, false) => {
                        emit::F64CopySign(&mut self.code, dst, lhs, rhs)
                    }
                    _ => todo!("Unsupported float opcode {:?} for {:?}", opcode, ty),
                }
            }
            _ => todo!("Unsupported binary opcode {:?} for type {:?}", opcode, ty),
        }
    }

    fn emit_icmp(&mut self, inst: Inst, kind: veloc_ir::IntCC, args: &[Value; 2]) {
        let lhs = self.mapper.use_val(args[0]);
        let rhs = self.mapper.use_val(args[1]);
        let dst = self
            .mapper
            .alloc_val(self.func.dfg.first_result(inst).unwrap());
        let ty = self.func.dfg.value_type(args[0]);

        use veloc_ir::IntCC::*;
        match (ty, kind) {
            (Type::I32, Eq) => emit::I32Eq(&mut self.code, dst, lhs, rhs),
            (Type::I32, Ne) => emit::I32Ne(&mut self.code, dst, lhs, rhs),
            (Type::I32, LtS) => emit::I32LtS(&mut self.code, dst, lhs, rhs),
            (Type::I32, LtU) => emit::I32LtU(&mut self.code, dst, lhs, rhs),
            (Type::I32, LeS) => emit::I32LeS(&mut self.code, dst, lhs, rhs),
            (Type::I32, LeU) => emit::I32LeU(&mut self.code, dst, lhs, rhs),
            (Type::I32, GtS) => emit::I32GtS(&mut self.code, dst, lhs, rhs),
            (Type::I32, GtU) => emit::I32GtU(&mut self.code, dst, lhs, rhs),
            (Type::I32, GeS) => emit::I32GeS(&mut self.code, dst, lhs, rhs),
            (Type::I32, GeU) => emit::I32GeU(&mut self.code, dst, lhs, rhs),
            (Type::I64, Eq) => emit::I64Eq(&mut self.code, dst, lhs, rhs),
            (Type::I64, Ne) => emit::I64Ne(&mut self.code, dst, lhs, rhs),
            (Type::I64, LtS) => emit::I64LtS(&mut self.code, dst, lhs, rhs),
            (Type::I64, LtU) => emit::I64LtU(&mut self.code, dst, lhs, rhs),
            (Type::I64, LeS) => emit::I64LeS(&mut self.code, dst, lhs, rhs),
            (Type::I64, LeU) => emit::I64LeU(&mut self.code, dst, lhs, rhs),
            (Type::I64, GtS) => emit::I64GtS(&mut self.code, dst, lhs, rhs),
            (Type::I64, GtU) => emit::I64GtU(&mut self.code, dst, lhs, rhs),
            (Type::I64, GeS) => emit::I64GeS(&mut self.code, dst, lhs, rhs),
            (Type::I64, GeU) => emit::I64GeU(&mut self.code, dst, lhs, rhs),
            (Type::BOOL, Eq) => emit::I32Eq(&mut self.code, dst, lhs, rhs),
            (Type::BOOL, Ne) => emit::I32Ne(&mut self.code, dst, lhs, rhs),
            (Type::PTR, Eq) => emit::I64Eq(&mut self.code, dst, lhs, rhs),
            (Type::PTR, Ne) => emit::I64Ne(&mut self.code, dst, lhs, rhs),
            (Type::PTR, LtU) => emit::I64LtU(&mut self.code, dst, lhs, rhs),
            (Type::PTR, LeU) => emit::I64LeU(&mut self.code, dst, lhs, rhs),
            (Type::PTR, GtU) => emit::I64GtU(&mut self.code, dst, lhs, rhs),
            (Type::PTR, GeU) => emit::I64GeU(&mut self.code, dst, lhs, rhs),
            _ => unreachable!("Invalid icmp type or kind: ty={:?}, kind={:?}", ty, kind),
        }
    }

    fn emit_fcmp(&mut self, inst: Inst, kind: veloc_ir::FloatCC, args: &[Value; 2]) {
        let lhs = self.mapper.use_val(args[0]);
        let rhs = self.mapper.use_val(args[1]);
        let dst = self
            .mapper
            .alloc_val(self.func.dfg.first_result(inst).unwrap());
        let ty = self.func.dfg.value_type(args[0]);

        use veloc_ir::FloatCC::*;
        match (ty, kind) {
            (Type::F32, Eq) => emit::F32Eq(&mut self.code, dst, lhs, rhs),
            (Type::F32, Ne) => emit::F32Ne(&mut self.code, dst, lhs, rhs),
            (Type::F32, Lt) => emit::F32Lt(&mut self.code, dst, lhs, rhs),
            (Type::F32, Le) => emit::F32Le(&mut self.code, dst, lhs, rhs),
            (Type::F32, Gt) => emit::F32Gt(&mut self.code, dst, lhs, rhs),
            (Type::F32, Ge) => emit::F32Ge(&mut self.code, dst, lhs, rhs),
            (Type::F64, Eq) => emit::F64Eq(&mut self.code, dst, lhs, rhs),
            (Type::F64, Ne) => emit::F64Ne(&mut self.code, dst, lhs, rhs),
            (Type::F64, Lt) => emit::F64Lt(&mut self.code, dst, lhs, rhs),
            (Type::F64, Le) => emit::F64Le(&mut self.code, dst, lhs, rhs),
            (Type::F64, Gt) => emit::F64Gt(&mut self.code, dst, lhs, rhs),
            (Type::F64, Ge) => emit::F64Ge(&mut self.code, dst, lhs, rhs),
            _ => unreachable!("Invalid fcmp type or kind: {:?}", ty),
        }
    }

    fn emit_load(&mut self, inst: Inst, ptr: Value, offset: u32) {
        let ptr_reg = self.mapper.use_val(ptr);
        let res = self.func.dfg.first_result(inst).unwrap();
        let dst = self.mapper.alloc_val(res);
        let ty = self.val_ty(res);

        match ty {
            ScalarType::I32 => emit::I32Load(&mut self.code, dst, ptr_reg, offset),
            ScalarType::I64 | ScalarType::Ptr => {
                emit::I64Load(&mut self.code, dst, ptr_reg, offset)
            }
            ScalarType::F32 => emit::F32Load(&mut self.code, dst, ptr_reg, offset),
            ScalarType::F64 => emit::F64Load(&mut self.code, dst, ptr_reg, offset),
            ScalarType::I8 => emit::I8Load(&mut self.code, dst, ptr_reg, offset),
            ScalarType::I16 => emit::I16Load(&mut self.code, dst, ptr_reg, offset),
            _ => panic!("Unsupported load type {:?}", ty),
        }
    }

    fn emit_jump(&mut self, dest: BlockCall) {
        let target_block = self.func.dfg.block_calls[dest].block;
        let moves = calculate_moves(self.func, dest, &mut self.mapper);

        if moves.is_empty() {
            let inst_idx = self.code.len();
            emit::Jump(&mut self.code, 0);
            self.jump_fixups.push((inst_idx, target_block));
        } else {
            let id = self.push_target(target_block, moves, self.code.len());
            emit::JumpWithMoves(&mut self.code, id);
        }
    }

    fn emit_br(&mut self, condition: Value, then_dest: BlockCall, else_dest: BlockCall) {
        let then_block = self.func.dfg.block_calls[then_dest].block;
        let then_moves = calculate_moves(self.func, then_dest, &mut self.mapper);
        let else_block = self.func.dfg.block_calls[else_dest].block;
        let else_moves = calculate_moves(self.func, else_dest, &mut self.mapper);
        let source_pc = self.code.len();
        let cond_reg = self.mapper.use_val(condition);

        if then_moves.is_empty() && else_moves.is_empty() {
            emit::Br(&mut self.code, cond_reg, 0, 0);
            self.br_fixups
                .push((source_pc, then_block, else_block));
        } else {
            let then_idx = self.push_target(then_block, then_moves, source_pc);
            let else_idx = self.push_target(else_block, else_moves, source_pc);
            emit::BrWithMoves(&mut self.code, cond_reg, then_idx, else_idx);
        }
    }

    fn emit_br_table(&mut self, index: Value, table: veloc_ir::JumpTable) {
        let table_data = self.func.dfg.jump_tables[table].targets.clone();
        let num_targets = table_data.len() as u32;
        let source_pc = self.code.len();

        let mut br_offset = 0;
        for (i, &target_call) in table_data.iter().enumerate() {
            let target_block = self.func.dfg.block_calls[target_call].block;
            let moves = calculate_moves(self.func, target_call, &mut self.mapper);
            let id = self.push_target(target_block, moves, source_pc);
            if i == 0 {
                br_offset = id;
            }
        }
        // Consume index after building all targets to avoid premature free.
        let index_reg = self.mapper.use_val(index);
        emit::BrTable(&mut self.code, index_reg, br_offset, num_targets);
    }

    fn emit_return(&mut self, values: veloc_ir::ValueList) {
        let ret_vals = self.func.dfg.get_value_list(values);
        let ret_regs: SmallVec<[Reg; 2]> =
            ret_vals.iter().map(|&v| self.mapper.use_val(v)).collect();
        let num_vals = ret_regs.len() as u32;
        let data_offset = self.data_section.add_return_regs(&ret_regs);
        emit::Return(&mut self.code, data_offset, num_vals);
    }

    fn emit_call(&mut self, inst: Inst, func_id: FuncId, args: veloc_ir::ValueList) {
        let args_regs: SmallVec<[Reg; 4]> = self
            .func
            .dfg
            .get_value_list(args)
            .iter()
            .map(|&v| self.mapper.use_val(v))
            .collect();

        let res_vals = self.func.dfg.inst_results(inst);
        let mut ret_regs: SmallVec<[Reg; 2]> = SmallVec::with_capacity(res_vals.len());
        for &v in res_vals {
            ret_regs.push(self.mapper.alloc_val(v));
        }

        let data_offset = self.data_section.add_call_data(&ret_regs, &args_regs);
        emit::Call(
            &mut self.code,
            func_id.as_u32(),
            data_offset,
            ret_regs.len() as u16,
            args_regs.len() as u16,
        );
    }

    fn emit_call_indirect(&mut self, inst: Inst, ptr: Value, args: ValueList) {
        let ptr_reg = self.mapper.use_val(ptr);
        let args_regs: SmallVec<[Reg; 4]> = self
            .func
            .dfg
            .get_value_list(args)
            .iter()
            .map(|&v| self.mapper.use_val(v))
            .collect();

        let res_vals = self.func.dfg.inst_results(inst);
        let mut ret_regs: SmallVec<[Reg; 2]> = SmallVec::with_capacity(res_vals.len());
        for &v in res_vals {
            ret_regs.push(self.mapper.alloc_val(v));
        }
        let data_offset = self.data_section.add_call_data(&ret_regs, &args_regs);
        emit::CallIndirect(
            &mut self.code,
            ptr_reg,
            data_offset,
            ret_regs.len() as u16,
            args_regs.len() as u16,
        );
    }

    fn emit_call_intrinsic(&mut self, inst: Inst, intrinsic: Intrinsic, args: ValueList) {
        let args_regs: SmallVec<[Reg; 4]> = self
            .func
            .dfg
            .get_value_list(args)
            .iter()
            .map(|&v| self.mapper.use_val(v))
            .collect();
        let res_vals = self.func.dfg.inst_results(inst);
        let mut ret_regs: SmallVec<[Reg; 2]> = SmallVec::with_capacity(res_vals.len());
        for &v in res_vals {
            ret_regs.push(self.mapper.alloc_val(v));
        }

        if ret_regs.len() == 1
            && try_emit_inline_intrinsic(&mut self.code, ret_regs[0].0, intrinsic, &args_regs)
        {
            // Inlined
        } else {
            let data_offset = self.data_section.add_call_data(&ret_regs, &args_regs);
            emit::CallIntrinsic(
                &mut self.code,
                intrinsic.as_u16(),
                data_offset,
                ret_regs.len() as u16,
                args_regs.len() as u16,
            );
        }
    }

    fn emit_unary(&mut self, inst: Inst, opcode: IrOpcode, arg: Value) {
        let from_ty = self.val_ty(arg);
        let res = self.func.dfg.first_result(inst).unwrap();
        let to_ty = self.val_ty(res);

        // Try to handle constant operands first (fusion)
        if let Some(c) = self.func.dfg.as_const(arg) {
            if let Some(val) = c.as_i64().or_else(|| c.as_bool().map(|b| b as i64)) {
                match opcode {
                    IrOpcode::ExtendS => {
                        let res_val = match from_ty {
                            ScalarType::I8 => val as i8 as i64,
                            ScalarType::I16 => val as i16 as i64,
                            ScalarType::I32 => val as i32 as i64,
                            _ => panic!("Unsupported ExtendS from_ty: {:?}", from_ty),
                        };
                        let dst = self.mapper.alloc_val(res);
                        emit::Iconst(&mut self.code, dst, res_val as u64);
                        return;
                    }
                    IrOpcode::ExtendU => {
                        let res_val = match from_ty {
                            ScalarType::I8 => (val as u8) as u64 as i64,
                            ScalarType::I16 => (val as u16) as u64 as i64,
                            ScalarType::I32 => (val as u32) as u64 as i64,
                            _ => panic!("Unsupported ExtendU from_ty: {:?}", from_ty),
                        };
                        let dst = self.mapper.alloc_val(res);
                        emit::Iconst(&mut self.code, dst, res_val as u64);
                        return;
                    }
                    IrOpcode::Wrap => {
                        let dst = self.mapper.alloc_val(res);
                        emit::Iconst(&mut self.code, dst, (val as u32) as u64);
                        return;
                    }
                    _ => {}
                }
            } else {
                match opcode {
                    IrOpcode::FloatDemote => {
                        if let Some(f64_val) = c.as_f64() {
                            let f = f64_val as f32;
                            let dst = self.mapper.alloc_val(res);
                            emit::Fconst(&mut self.code, dst, f.to_bits() as u64);
                            return;
                        }
                    }
                    IrOpcode::FloatPromote => {
                        if let Some(f32_val) = c.as_f32() {
                            let f = f32_val as f64;
                            let dst = self.mapper.alloc_val(res);
                            emit::Fconst(&mut self.code, dst, f.to_bits());
                            return;
                        }
                    }
                    _ => {}
                }
            }
        }

        let arg_reg = self.mapper.use_val(arg);
        let dst = self.mapper.alloc_val(res);

        match opcode {
            IrOpcode::ExtendS => {
                emit::ExtendS(
                    &mut self.code,
                    dst,
                    arg_reg,
                    TypePair {
                        from: from_ty,
                        to: to_ty,
                    },
                );
            }
            IrOpcode::ExtendU => {
                emit::ExtendU(
                    &mut self.code,
                    dst,
                    arg_reg,
                    TypePair {
                        from: from_ty,
                        to: to_ty,
                    },
                );
            }
            IrOpcode::Wrap => {
                emit::Wrap(
                    &mut self.code,
                    dst,
                    arg_reg,
                    TypePair {
                        from: from_ty,
                        to: to_ty,
                    },
                );
            }
            IrOpcode::FloatToIntS => convert_op!(
                arg_reg,
                dst,
                &mut self.code,
                from_ty,
                to_ty,
                I32TruncF32S,
                I32TruncF64S,
                I64TruncF32S,
                I64TruncF64S
            ),
            IrOpcode::FloatToIntU => convert_op!(
                arg_reg,
                dst,
                &mut self.code,
                from_ty,
                to_ty,
                I32TruncF32U,
                I32TruncF64U,
                I64TruncF32U,
                I64TruncF64U
            ),
            IrOpcode::FloatToIntSatS => convert_op!(
                arg_reg,
                dst,
                &mut self.code,
                from_ty,
                to_ty,
                I32TruncSatF32S,
                I32TruncSatF64S,
                I64TruncSatF32S,
                I64TruncSatF64S
            ),
            IrOpcode::FloatToIntSatU => convert_op!(
                arg_reg,
                dst,
                &mut self.code,
                from_ty,
                to_ty,
                I32TruncSatF32U,
                I32TruncSatF64U,
                I64TruncSatF32U,
                I64TruncSatF64U
            ),
            IrOpcode::IntToFloatS => convert_int_to_float_op!(
                arg_reg,
                dst,
                &mut self.code,
                from_ty,
                to_ty,
                F32ConvertI32S,
                F32ConvertI64S,
                F64ConvertI32S,
                F64ConvertI64S
            ),
            IrOpcode::IntToFloatU => convert_int_to_float_op!(
                arg_reg,
                dst,
                &mut self.code,
                from_ty,
                to_ty,
                F32ConvertI32U,
                F32ConvertI64U,
                F64ConvertI32U,
                F64ConvertI64U
            ),
            IrOpcode::FloatDemote => emit::F32DemoteF64(&mut self.code, dst, arg_reg),
            IrOpcode::FloatPromote => emit::F64PromoteF32(&mut self.code, dst, arg_reg),
            IrOpcode::Reinterpret => {
                emit::RegMove(&mut self.code, dst, arg_reg);
            }
            IrOpcode::FAbs => {
                unary_dispatch_op!(arg_reg, dst, from_ty, &mut self.code, F32Abs, F64Abs)
            }
            IrOpcode::FNeg => {
                unary_dispatch_op!(arg_reg, dst, from_ty, &mut self.code, F32Neg, F64Neg)
            }
            IrOpcode::FSqrt => {
                unary_dispatch_op!(arg_reg, dst, from_ty, &mut self.code, F32Sqrt, F64Sqrt)
            }
            IrOpcode::FCeil => {
                unary_dispatch_op!(arg_reg, dst, from_ty, &mut self.code, F32Ceil, F64Ceil)
            }
            IrOpcode::FFloor => {
                unary_dispatch_op!(arg_reg, dst, from_ty, &mut self.code, F32Floor, F64Floor)
            }
            IrOpcode::FTrunc => {
                unary_dispatch_op!(arg_reg, dst, from_ty, &mut self.code, F32Trunc, F64Trunc)
            }
            IrOpcode::FNearest => {
                unary_dispatch_op!(
                    arg_reg,
                    dst,
                    from_ty,
                    &mut self.code,
                    F32Nearest,
                    F64Nearest
                )
            }
            IrOpcode::IClz => {
                unary_dispatch_op!(arg_reg, dst, from_ty, &mut self.code, I32Clz, I64Clz, int)
            }
            IrOpcode::ICtz => {
                unary_dispatch_op!(arg_reg, dst, from_ty, &mut self.code, I32Ctz, I64Ctz, int)
            }
            IrOpcode::IPopcnt => {
                unary_dispatch_op!(
                    arg_reg,
                    dst,
                    from_ty,
                    &mut self.code,
                    I32Popcnt,
                    I64Popcnt,
                    int
                )
            }
            IrOpcode::IEqz => {
                unary_dispatch_op!(arg_reg, dst, from_ty, &mut self.code, I32Eqz, I64Eqz, int)
            }
            _ => todo!("Implement other unary ops in emit_unary: {:?}", opcode),
        }
    }

    fn val_ty(&self, v: Value) -> ScalarType {
        self.func.dfg.value_type(v).scalar_type()
    }

    fn emit_store(&mut self, ptr: Value, value: Value, offset: u32) {
        let ptr_reg = self.mapper.use_val(ptr);
        let val_reg = self.mapper.use_val(value);
        let ty = self.val_ty(value);

        match ty {
            ScalarType::I8 => emit::I8Store(&mut self.code, val_reg, ptr_reg, offset),
            ScalarType::I16 => emit::I16Store(&mut self.code, val_reg, ptr_reg, offset),
            ScalarType::I32 => emit::I32Store(&mut self.code, val_reg, ptr_reg, offset),
            ScalarType::I64 | ScalarType::Ptr => {
                emit::I64Store(&mut self.code, val_reg, ptr_reg, offset)
            }
            ScalarType::F32 => emit::F32Store(&mut self.code, val_reg, ptr_reg, offset),
            ScalarType::F64 => emit::F64Store(&mut self.code, val_reg, ptr_reg, offset),
            _ => panic!("Unsupported store type {:?}", ty),
        }
    }

    fn finish(mut self, module_id: ModuleId, func_id: FuncId) -> CompiledFunction {
        // Patch jump targets
        for (inst_idx, target_block) in self.jump_fixups {
            let target_pc = self.block_to_pc[target_block];
            let byte_offset = relative_byte_offset(inst_idx, target_pc);
            self.code[inst_idx].imm64 = byte_offset as u64;
        }

        // Patch direct conditional branch targets.
        for (inst_idx, then_block, else_block) in self.br_fixups {
            let then_offset = relative_i32_byte_offset(inst_idx, self.block_to_pc[then_block]);
            let else_offset = relative_i32_byte_offset(inst_idx, self.block_to_pc[else_block]);
            self.code[inst_idx].imm64 =
                u64::from(then_offset as u32) | (u64::from(else_offset as u32) << 32);
        }

        // Patch data section targets
        for (data_idx, target_block, source_pc) in self.data_fixups {
            let target_pc = self.block_to_pc[target_block];
            self.data_section.jump_targets[data_idx].offset =
                relative_i32_byte_offset(source_pc, target_pc);
        }

        CompiledFunction {
            module_id,
            func_id,
            code: self.code,
            data_section: self.data_section,
            stack_slots_sizes: self
                .func
                .stack_slots
                .iter()
                .map(|(_, d)| d.size as usize)
                .collect(),
            param_indices: self.param_indices,
            register_count: self.mapper.next_register as usize,
        }
    }
}

pub(crate) fn compile_function(
    module_id: ModuleId,
    func_id: FuncId,
    func: &Function,
) -> CompiledFunction {
    let entry = func.entry_block.expect("Function must have entry block");
    let rpo = func.layout.compute_rpo(entry);

    let liveness = analyze_liveness(func);
    let block_params: std::collections::HashSet<Value> = func
        .layout
        .blocks
        .values()
        .flat_map(|b| b.params.iter().copied())
        .collect();
    let fused_values = identify_fused_values(func, &rpo);

    let mapper = ValueMapper::new(&liveness.intervals, &block_params, &fused_values);
    let mut compiler = Compiler::new(func, &liveness, mapper);

    compiler.apply_rpo(&rpo);
    compiler.finish(module_id, func_id)
}

impl<'a> Compiler<'a> {
    fn apply_rpo(&mut self, rpo: &[Block]) {
        let entry_block = self.func.entry_block.unwrap();
        for &param in &self.func.layout.blocks[entry_block].params {
            self.param_indices.push(self.mapper.alloc_val(param));
        }

        for &block in rpo {
            self.block_to_pc[block] = self.code.len() as u32;
            let block_data = &self.func.layout.blocks[block];

            if block != entry_block {
                for &param in &block_data.params {
                    if self.mapper.map[param] == Reg::NULL {
                        self.mapper.alloc_val(param);
                    }
                }
            }

            for &inst in &block_data.insts {
                self.compile_inst(inst);
            }

            // Also free parameters if they are not used anymore after this block
            let pc = self.liveness.block_ends[block];
            for &param in &block_data.params {
                self.mapper.free_if_last_use(param, pc);
            }
        }
    }

    fn compile_inst(&mut self, inst: Inst) {
        let pc = self.liveness.inst_pcs[inst];
        let idata = &self.func.dfg.instructions[inst];

        match idata {
            InstructionData::Iconst { value } => {
                let res = self.func.dfg.first_result(inst).unwrap();
                if !self.mapper.fused_values.contains(&res) {
                    let dst = self.mapper.alloc_val(res);
                    emit::Iconst(&mut self.code, dst, *value);
                }
            }
            InstructionData::Fconst { value } => {
                let res = self.func.dfg.first_result(inst).unwrap();
                let dst = self.mapper.alloc_val(res);
                emit::Fconst(&mut self.code, dst, *value);
            }
            InstructionData::Vconst { pool_id } => {
                let res = self.func.dfg.first_result(inst).unwrap();
                let dst = self.mapper.alloc_val(res);
                emit::Vconst(&mut self.code, dst, pool_id.as_u32());
            }
            InstructionData::Bconst { value } => {
                let res = self.func.dfg.first_result(inst).unwrap();
                if !self.mapper.fused_values.contains(&res) {
                    let dst = self.mapper.alloc_val(res);
                    emit::Bconst(&mut self.code, dst, *value);
                }
            }
            InstructionData::Binary { opcode, args } => self.emit_binary(inst, *opcode, args),
            InstructionData::IntCompare { kind, args, .. } => self.emit_icmp(inst, *kind, args),
            InstructionData::FloatCompare { kind, args, .. } => self.emit_fcmp(inst, *kind, args),
            InstructionData::StackAddr { slot, offset, .. } => {
                let res = self.func.dfg.first_result(inst).unwrap();
                let dst = self.mapper.alloc_val(res);
                let base_offset = self.slot_to_offset[*slot];
                emit::StackAddr(&mut self.code, dst, base_offset + *offset);
            }
            InstructionData::StackLoad { slot, offset } => {
                let res = self.func.dfg.first_result(inst).unwrap();
                let dst = self.mapper.alloc_val(res);
                let base_offset = self.slot_to_offset[*slot];
                let ty = self.val_ty(res);
                emit::StackLoad(&mut self.code, dst, ty, base_offset + *offset);
            }
            InstructionData::StackStore {
                slot,
                value,
                offset,
                ..
            } => {
                let base_offset = self.slot_to_offset[*slot];
                let val_reg = self.mapper.use_val(*value);
                let ty = self.val_ty(*value);
                emit::StackStore(&mut self.code, val_reg, ty, base_offset + *offset);
            }
            InstructionData::Load { ptr, offset, .. } => self.emit_load(inst, *ptr, *offset as u32),
            InstructionData::Store {
                ptr, value, offset, ..
            } => self.emit_store(*ptr, *value, *offset as u32),
            InstructionData::Jump { dest } => self.emit_jump(*dest),
            InstructionData::Br {
                condition,
                then_dest,
                else_dest,
            } => {
                self.emit_br(*condition, *then_dest, *else_dest);
            }
            InstructionData::BrTable { index, table } => self.emit_br_table(*index, *table),
            InstructionData::Return { values } => self.emit_return(*values),
            InstructionData::Unary { opcode, arg, .. } => self.emit_unary(inst, *opcode, *arg),
            InstructionData::IntToPtr { arg } | InstructionData::PtrToInt { arg, .. } => {
                let arg_reg = self.mapper.use_val(*arg);
                let res = self.func.dfg.first_result(inst).unwrap();
                let dst = self.mapper.alloc_val(res);
                emit::RegMove(&mut self.code, dst, arg_reg);
            }
            InstructionData::Call { func_id, args, .. } => self.emit_call(inst, *func_id, *args),
            InstructionData::CallIndirect { ptr, args, .. } => {
                self.emit_call_indirect(inst, *ptr, *args)
            }
            InstructionData::CallIntrinsic {
                intrinsic, args, ..
            } => self.emit_call_intrinsic(inst, *intrinsic, *args),
            InstructionData::PtrIndex { ptr, index, imm_id } => {
                let ptr_reg = self.mapper.use_val(*ptr);
                let index_reg = self.mapper.use_val(*index);
                let res = self.func.dfg.first_result(inst).unwrap();
                let dst = self.mapper.alloc_val(res);
                let imm = self.func.dfg.get_ptr_imm(*imm_id);
                emit::PtrIndex(
                    &mut self.code,
                    dst,
                    ptr_reg,
                    index_reg,
                    imm.scale as u32,
                    imm.offset as u32,
                );
            }
            InstructionData::PtrOffset { ptr, offset } => {
                let ptr_reg = self.mapper.use_val(*ptr);
                let res = self.func.dfg.first_result(inst).unwrap();
                let dst = self.mapper.alloc_val(res);
                emit::I64AddImm(&mut self.code, dst, ptr_reg, *offset as u64);
            }
            InstructionData::Unreachable => {
                emit::Unreachable(&mut self.code);
            }
            InstructionData::Ternary { opcode, args } if *opcode == IrOpcode::Select => {
                let cond_reg = self.mapper.use_val(args[0]);
                let then_reg = self.mapper.use_val(args[1]);
                let else_reg = self.mapper.use_val(args[2]);
                let res = self.func.dfg.first_result(inst).unwrap();
                let dst = self.mapper.alloc_val(res);
                emit::Select(&mut self.code, dst, cond_reg, then_reg, else_reg);
            }
            InstructionData::Nop => {}
            _ => todo!("Unsupported instruction: {:?}", idata),
        }

        idata.visit_operands(&self.func.dfg, |v| self.mapper.free_if_last_use(v, pc));
    }
}

/// Calculate register moves needed for block argument passing without emitting instructions.
/// Returns a vector of (dst_reg, src_reg) pairs representing the moves in order.
/// Note: This function uses a simple ordering strategy; the interpreter will execute
/// the moves in the order they are stored.
fn calculate_moves(
    func: &Function,
    call: veloc_ir::types::BlockCall,
    mapper: &mut ValueMapper,
) -> Vec<(Reg, Reg)> {
    fn find_non_conflicting_move_index(pending: &[(Reg, Reg)]) -> Option<usize> {
        // Find a move whose destination is not used as a source by any other move.
        for (i, &(dst, _)) in pending.iter().enumerate() {
            let dst_used_as_source = pending
                .iter()
                .enumerate()
                .any(|(j, &(_, src))| i != j && dst == src);
            if !dst_used_as_source {
                return Some(i);
            }
        }
        None
    }

    fn alloc_temp_reg(mapper: &mut ValueMapper) -> Reg {
        mapper.free_registers.pop().unwrap_or_else(|| {
            let r = Reg(mapper.next_register);
            mapper.next_register += 1;
            r
        })
    }

    let target_block = func.dfg.block_calls[call].block;
    let args = func.dfg.get_value_list(func.dfg.block_calls[call].args);
    let params = &func.layout.blocks[target_block].params;

    // 1. Collect all move requests with pre-allocated capacity
    let mut pending: Vec<(Reg, Reg)> = Vec::with_capacity(params.len());
    for (&p, &a) in params.iter().zip(args.iter()) {
        let src = mapper.use_val(a);
        let dst = mapper.use_block_param(p);
        if dst != src {
            pending.push((dst, src));
        }
    }

    // 2. Resolve parallel moves into an ordered sequence
    let mut result: Vec<(Reg, Reg)> = Vec::with_capacity(pending.len() + 1); // +1 for potential temp move

    while !pending.is_empty() {
        if let Some(i) = find_non_conflicting_move_index(&pending) {
            // Use swap_remove for O(1) removal instead of O(n)
            let (d, s) = pending.swap_remove(i);
            result.push((d, s));
            continue;
        }

        // 3. Cycle detected. We need to break it by using a temporary register.
        // Pick the first move (d, s) and save s to a temp register.
        let (d, s) = pending.swap_remove(0);

        // Prefer reusing a free register over allocating a new one.
        let temp = alloc_temp_reg(mapper);

        result.push((temp, s));

        // Replace (d, s) with (d, temp). Since temp is fresh, (d, temp)
        // will eventually be considered safe to move into d.
        pending.push((d, temp));
    }

    result
}
