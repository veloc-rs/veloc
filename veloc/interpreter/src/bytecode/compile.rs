use crate::bytecode::inst::{Instruction, emit};
use cranelift_entity::SecondaryMap;
use veloc_analyzer::{LiveInterval, analyze_liveness};
use veloc_ir::{
    Block, FuncId, Function, InstructionData, Intrinsic, ModuleId, Opcode as IrOpcode, ScalarType,
    StackSlot, Type, Value,
};

/// Represents a single BrTable target with direct PC and register moves for argument passing
#[derive(Debug, Clone, Default)]
pub struct JumpTarget {
    /// Target PC (direct block address)
    pub pc: u32,
    /// Number of register moves
    pub num_moves: u16,
    /// Offset into u16_data where moves are stored (as pairs of dst, src)
    pub moves_offset: u32,
}

/// Data section stored as u16 words for efficient register access
/// For Jump targets, they are stored in a separate Vec<JumpTarget>
#[derive(Debug, Clone, Default)]
pub struct DataSection {
    /// Register lists and counts stored as u16 words
    pub u16_data: Vec<u16>,
    /// Jump targets with direct PC and moves
    pub jump_targets: Vec<JumpTarget>,
}

impl DataSection {
    pub fn new() -> Self {
        Self {
            u16_data: Vec::new(),
            jump_targets: Vec::new(),
        }
    }

    /// Add return registers for Return instruction
    /// Returns the offset (in u16 words) where data starts
    pub fn add_return_regs(&mut self, regs: &[u16]) -> u32 {
        let offset = self.u16_data.len();
        self.u16_data.push(regs.len() as u16);
        self.u16_data.extend_from_slice(regs);
        offset as u32
    }

    /// Add call data (ret_regs + arg_regs) for Call/CallIndirect/CallIntrinsic
    /// Returns the offset (in u16 words) where data starts
    pub fn add_call_data(&mut self, ret_regs: &[u16], arg_regs: &[u16]) -> u32 {
        let offset = self.u16_data.len();
        self.u16_data.push(ret_regs.len() as u16);
        self.u16_data.push(arg_regs.len() as u16);
        self.u16_data.extend_from_slice(ret_regs);
        self.u16_data.extend_from_slice(arg_regs);
        offset as u32
    }

    /// Add Jump targets with their register moves
    /// Returns (offset in jump_targets, num_targets)
    pub fn add_jump_target(&mut self, target: JumpTarget) -> u32 {
        let offset = self.jump_targets.len();
        self.jump_targets.push(target);
        offset as u32
    }
}

pub struct CompiledFunction {
    pub(crate) module_id: ModuleId,
    pub(crate) func_id: FuncId,
    pub(crate) code: Vec<Instruction>,
    /// Data section: u16_data for register lists, jump_targets for jump targets
    pub(crate) data_section: DataSection,
    pub(crate) stack_slots_sizes: Vec<usize>,
    pub(crate) param_indices: Vec<u16>,
    pub(crate) ret_indices: Vec<u16>, // Return value register indices (support multi-value)
    pub(crate) register_count: usize,
    pub(crate) constant_pool: Vec<Vec<u8>>,
}

struct ValueMapper {
    map: SecondaryMap<Value, u16>,
    free_registers: Vec<u16>,
    next_register: u16,
    intervals: SecondaryMap<Value, LiveInterval>,
}

impl ValueMapper {
    fn new(intervals_map: SecondaryMap<Value, LiveInterval>) -> Self {
        Self {
            map: SecondaryMap::new(),
            free_registers: Vec::new(),
            next_register: 1,
            intervals: intervals_map,
        }
    }

    fn get_mapped(&self, val: Value) -> u16 {
        let reg = self.map[val];
        if reg == 0 {
            panic!("Value {:?} used before defined or mapping missing", val);
        }
        reg
    }

    fn alloc_and_map(&mut self, val: Value) -> u16 {
        let existing = self.map[val];
        if existing != 0 {
            return existing;
        }

        let reg = if let Some(r) = self.free_registers.pop() {
            r
        } else {
            let r = self.next_register;
            self.next_register += 1;
            r
        };
        self.map[val] = reg;
        reg
    }

    fn free_if_last_use(&mut self, val: Value, pc: u32) {
        let interval = &self.intervals[val];
        if !interval.ranges.is_empty() {
            if interval.end() <= pc {
                let reg = self.map[val];
                if reg != 0 {
                    self.map[val] = 0;
                    self.free_registers.push(reg);
                }
            }
        }
    }
}

/// Try to emit inline bytecode for intrinsics that map directly to opcodes.
/// Returns true if the intrinsic was successfully inlined.
fn try_emit_inline_intrinsic(
    _code: &mut Vec<Instruction>,
    _dst: u16,
    _intrinsic: Intrinsic,
    _args: &[u16],
) -> bool {
    // Currently no intrinsics are inlineable as bytecode.
    // Math intrinsics (sin, cos, pow, etc.) need runtime libm calls.
    // Memory intrinsics need runtime support.
    false
}

pub(crate) fn compile_function(
    module_id: ModuleId,
    func_id: FuncId,
    func: &Function,
) -> CompiledFunction {
    let liveness = analyze_liveness(func);
    let entry = func.entry_block.expect("Function must have entry block");
    let rpo = func.layout.compute_rpo(entry);

    let mut slot_to_offset: SecondaryMap<StackSlot, u32> = SecondaryMap::new();
    let mut current_offset = 0u32;
    for (id, data) in &func.stack_slots {
        slot_to_offset[id] = current_offset;
        current_offset += data.size;
    }

    let mut mapper = ValueMapper::new(liveness.intervals);
    let mut code = Vec::new();
    let mut data_section = DataSection::new();
    let mut param_indices = Vec::new();
    let mut block_to_pc: SecondaryMap<Block, u32> = SecondaryMap::new();
    let mut jump_fixups = Vec::new();
    let mut data_fixups = Vec::new();

    let push_target = |call: veloc_ir::types::BlockCall,
                       mapper: &mut ValueMapper,
                       data_section: &mut DataSection,
                       data_fixups: &mut Vec<(usize, Block)>|
     -> u32 {
        let target_block = func.dfg.block_calls[call].block;
        let moves = calculate_moves(func, call, mapper);
        let num_moves = moves.len() as u16;
        let moves_offset = if num_moves == 0 {
            0
        } else {
            let offset = data_section.u16_data.len() as u32;
            for (dst, src) in moves {
                data_section.u16_data.push(dst);
                data_section.u16_data.push(src);
            }
            offset
        };

        let br_offset = data_section.add_jump_target(JumpTarget {
            pc: 0,
            num_moves,
            moves_offset,
        });

        data_fixups.push((br_offset as usize, target_block));
        br_offset
    };

    let val_ty = |v: Value| func.dfg.value_type(v).scalar_type();

    macro_rules! binary_op {
        ($imm_op:ident, $reg_op:ident, $imm_ty:ty, $lhs:expr, $rhs:expr, $args:expr, $code:expr, $dst:expr) => {
            if let Some(v) = func.dfg.as_const($args[1]).and_then(|c| c.as_i64()) {
                emit::$imm_op($code, $dst, $lhs, v as $imm_ty);
            } else {
                emit::$reg_op($code, $dst, $lhs, $rhs);
            }
        };
        ($imm_op:ident, $reg_op:ident, $imm_ty:ty, $lhs:expr, $rhs:expr, $args:expr, $code:expr, $dst:expr, commutative) => {
            if let Some(v) = func.dfg.as_const($args[1]).and_then(|c| c.as_i64()) {
                emit::$imm_op($code, $dst, $lhs, v as $imm_ty);
            } else if let Some(v) = func.dfg.as_const($args[0]).and_then(|c| c.as_i64()) {
                emit::$imm_op($code, $dst, $rhs, v as $imm_ty);
            } else {
                emit::$reg_op($code, $dst, $lhs, $rhs);
            }
        };
    }

    macro_rules! icmp_op {
        ($kind:expr, $lhs:expr, $rhs:expr, $code:expr, $dst:expr, $Eq:ident, $Ne:ident, $LtS:ident, $LtU:ident, $LeS:ident, $LeU:ident, $GtS:ident, $GtU:ident, $GeS:ident, $GeU:ident) => {
            match $kind {
                veloc_ir::IntCC::Eq => emit::$Eq($code, $dst, $lhs, $rhs),
                veloc_ir::IntCC::Ne => emit::$Ne($code, $dst, $lhs, $rhs),
                veloc_ir::IntCC::LtS => emit::$LtS($code, $dst, $lhs, $rhs),
                veloc_ir::IntCC::LtU => emit::$LtU($code, $dst, $lhs, $rhs),
                veloc_ir::IntCC::LeS => emit::$LeS($code, $dst, $lhs, $rhs),
                veloc_ir::IntCC::LeU => emit::$LeU($code, $dst, $lhs, $rhs),
                veloc_ir::IntCC::GtS => emit::$GtS($code, $dst, $lhs, $rhs),
                veloc_ir::IntCC::GtU => emit::$GtU($code, $dst, $lhs, $rhs),
                veloc_ir::IntCC::GeS => emit::$GeS($code, $dst, $lhs, $rhs),
                veloc_ir::IntCC::GeU => emit::$GeU($code, $dst, $lhs, $rhs),
            }
        };
    }

    macro_rules! fcmp_op {
        ($kind:expr, $lhs:expr, $rhs:expr, $code:expr, $dst:expr, $Eq:ident, $Ne:ident, $Lt:ident, $Le:ident, $Gt:ident, $Ge:ident) => {
            match $kind {
                veloc_ir::FloatCC::Eq => emit::$Eq($code, $dst, $lhs, $rhs),
                veloc_ir::FloatCC::Ne => emit::$Ne($code, $dst, $lhs, $rhs),
                veloc_ir::FloatCC::Lt => emit::$Lt($code, $dst, $lhs, $rhs),
                veloc_ir::FloatCC::Le => emit::$Le($code, $dst, $lhs, $rhs),
                veloc_ir::FloatCC::Gt => emit::$Gt($code, $dst, $lhs, $rhs),
                veloc_ir::FloatCC::Ge => emit::$Ge($code, $dst, $lhs, $rhs),
            }
        };
    }

    if let Some(entry_block) = func.entry_block {
        for &param in &func.layout.blocks[entry_block].params {
            param_indices.push(mapper.alloc_and_map(param));
        }
    }

    for &block in &rpo {
        block_to_pc[block] = code.len() as u32;
        let block_data = &func.layout.blocks[block];

        // Ensure all parameters of this block are mapped
        for &param in &block_data.params {
            mapper.alloc_and_map(param);
        }

        for &inst in &block_data.insts {
            let current_ir_inst_idx = liveness.inst_pcs[inst];
            let idata = &func.dfg.instructions[inst];
            let res_vals = func.dfg.inst_results(inst);

            // Allocate registers for all results
            for &res in res_vals {
                mapper.alloc_and_map(res);
            }

            let dst = res_vals.first().map(|&v| mapper.map[v]).unwrap_or(0);

            match idata {
                InstructionData::Iconst { value } => {
                    emit::Iconst(&mut code, dst, *value);
                }
                InstructionData::Fconst { value } => {
                    emit::Fconst(&mut code, dst, *value);
                }
                InstructionData::Vconst { pool_id } => {
                    emit::Vconst(&mut code, dst, pool_id.as_u32());
                }
                InstructionData::Bconst { value } => {
                    emit::Bconst(&mut code, dst, if *value { 1 } else { 0 });
                }
                InstructionData::Binary { opcode, args } => {
                    let lhs = mapper.get_mapped(args[0]);
                    let rhs = mapper.get_mapped(args[1]);
                    let ty = res_vals
                        .first()
                        .map(|&v| func.dfg.value_type(v))
                        .unwrap_or(Type::VOID);

                    if ty == Type::I32 {
                        match *opcode {
                            IrOpcode::IAdd => binary_op!(
                                I32AddImm,
                                I32Add,
                                u32,
                                lhs,
                                rhs,
                                args,
                                &mut code,
                                dst,
                                commutative
                            ),
                            IrOpcode::ISub => {
                                binary_op!(I32SubImm, I32Sub, u32, lhs, rhs, args, &mut code, dst)
                            }
                            IrOpcode::IMul => emit::I32Mul(&mut code, dst, lhs, rhs),
                            IrOpcode::IAnd => binary_op!(
                                I32AndImm,
                                I32And,
                                u32,
                                lhs,
                                rhs,
                                args,
                                &mut code,
                                dst,
                                commutative
                            ),
                            IrOpcode::IOr => binary_op!(
                                I32OrImm,
                                I32Or,
                                u32,
                                lhs,
                                rhs,
                                args,
                                &mut code,
                                dst,
                                commutative
                            ),
                            IrOpcode::IXor => binary_op!(
                                I32XorImm,
                                I32Xor,
                                u32,
                                lhs,
                                rhs,
                                args,
                                &mut code,
                                dst,
                                commutative
                            ),
                            IrOpcode::IShl => {
                                binary_op!(I32ShlImm, I32Shl, u32, lhs, rhs, args, &mut code, dst)
                            }
                            IrOpcode::IShrS => {
                                binary_op!(I32ShrSImm, I32ShrS, u32, lhs, rhs, args, &mut code, dst)
                            }
                            IrOpcode::IShrU => {
                                binary_op!(I32ShrUImm, I32ShrU, u32, lhs, rhs, args, &mut code, dst)
                            }
                            IrOpcode::IDivS => emit::I32DivS(&mut code, dst, lhs, rhs),
                            IrOpcode::IDivU => emit::I32DivU(&mut code, dst, lhs, rhs),
                            IrOpcode::IRemS => emit::I32RemS(&mut code, dst, lhs, rhs),
                            IrOpcode::IRemU => emit::I32RemU(&mut code, dst, lhs, rhs),
                            IrOpcode::IRotl => emit::I32RotL(&mut code, dst, lhs, rhs),
                            IrOpcode::IRotr => emit::I32RotR(&mut code, dst, lhs, rhs),
                            _ => panic!(
                                "Unsupported binary opcode {:?} for type {:?}",
                                opcode,
                                ty.clone()
                            ),
                        }
                    } else if ty == Type::I64 {
                        match *opcode {
                            IrOpcode::IAdd => binary_op!(
                                I64AddImm,
                                I64Add,
                                u64,
                                lhs,
                                rhs,
                                args,
                                &mut code,
                                dst,
                                commutative
                            ),
                            IrOpcode::ISub => {
                                binary_op!(I64SubImm, I64Sub, u64, lhs, rhs, args, &mut code, dst)
                            }
                            IrOpcode::IMul => emit::I64Mul(&mut code, dst, lhs, rhs),
                            IrOpcode::IAnd => binary_op!(
                                I64AndImm,
                                I64And,
                                u64,
                                lhs,
                                rhs,
                                args,
                                &mut code,
                                dst,
                                commutative
                            ),
                            IrOpcode::IOr => binary_op!(
                                I64OrImm,
                                I64Or,
                                u64,
                                lhs,
                                rhs,
                                args,
                                &mut code,
                                dst,
                                commutative
                            ),
                            IrOpcode::IXor => binary_op!(
                                I64XorImm,
                                I64Xor,
                                u64,
                                lhs,
                                rhs,
                                args,
                                &mut code,
                                dst,
                                commutative
                            ),
                            IrOpcode::IShl => {
                                binary_op!(I64ShlImm, I64Shl, u64, lhs, rhs, args, &mut code, dst)
                            }
                            IrOpcode::IShrS => {
                                binary_op!(I64ShrSImm, I64ShrS, u64, lhs, rhs, args, &mut code, dst)
                            }
                            IrOpcode::IShrU => {
                                binary_op!(I64ShrUImm, I64ShrU, u64, lhs, rhs, args, &mut code, dst)
                            }
                            IrOpcode::IDivS => emit::I64DivS(&mut code, dst, lhs, rhs),
                            IrOpcode::IDivU => emit::I64DivU(&mut code, dst, lhs, rhs),
                            IrOpcode::IRemS => emit::I64RemS(&mut code, dst, lhs, rhs),
                            IrOpcode::IRemU => emit::I64RemU(&mut code, dst, lhs, rhs),
                            IrOpcode::IRotl => emit::I64RotL(&mut code, dst, lhs, rhs),
                            IrOpcode::IRotr => emit::I64RotR(&mut code, dst, lhs, rhs),
                            _ => panic!(
                                "Unsupported binary opcode {:?} for type {:?}",
                                opcode,
                                ty.clone()
                            ),
                        }
                    } else if ty == Type::BOOL {
                        // Bool uses I32 operations
                        match *opcode {
                            IrOpcode::IAnd => binary_op!(
                                I32AndImm,
                                I32And,
                                u32,
                                lhs,
                                rhs,
                                args,
                                &mut code,
                                dst,
                                commutative
                            ),
                            IrOpcode::IOr => binary_op!(
                                I32OrImm,
                                I32Or,
                                u32,
                                lhs,
                                rhs,
                                args,
                                &mut code,
                                dst,
                                commutative
                            ),
                            IrOpcode::IXor => binary_op!(
                                I32XorImm,
                                I32Xor,
                                u32,
                                lhs,
                                rhs,
                                args,
                                &mut code,
                                dst,
                                commutative
                            ),
                            _ => panic!(
                                "Unsupported binary opcode {:?} for type {:?}",
                                opcode,
                                ty.clone()
                            ),
                        }
                    } else if ty == Type::F32 {
                        match *opcode {
                            IrOpcode::FAdd => emit::F32Add(&mut code, dst, lhs, rhs),
                            IrOpcode::FSub => emit::F32Sub(&mut code, dst, lhs, rhs),
                            IrOpcode::FMul => emit::F32Mul(&mut code, dst, lhs, rhs),
                            IrOpcode::FDiv => emit::F32Div(&mut code, dst, lhs, rhs),
                            IrOpcode::FMin => emit::F32Min(&mut code, dst, lhs, rhs),
                            IrOpcode::FMax => emit::F32Max(&mut code, dst, lhs, rhs),
                            IrOpcode::FCopysign => emit::F32CopySign(&mut code, dst, lhs, rhs),
                            _ => panic!(
                                "Unsupported binary opcode {:?} for type {:?}",
                                opcode,
                                ty.clone()
                            ),
                        }
                    } else if ty == Type::F64 {
                        match *opcode {
                            IrOpcode::FAdd => emit::F64Add(&mut code, dst, lhs, rhs),
                            IrOpcode::FSub => emit::F64Sub(&mut code, dst, lhs, rhs),
                            IrOpcode::FMul => emit::F64Mul(&mut code, dst, lhs, rhs),
                            IrOpcode::FDiv => emit::F64Div(&mut code, dst, lhs, rhs),
                            IrOpcode::FMin => emit::F64Min(&mut code, dst, lhs, rhs),
                            IrOpcode::FMax => emit::F64Max(&mut code, dst, lhs, rhs),
                            IrOpcode::FCopysign => emit::F64CopySign(&mut code, dst, lhs, rhs),
                            _ => panic!(
                                "Unsupported binary opcode {:?} for type {:?}",
                                opcode,
                                ty.clone()
                            ),
                        }
                    } else {
                        panic!(
                            "Unsupported binary opcode {:?} for type {:?}",
                            opcode,
                            ty.clone()
                        );
                    }
                }
                InstructionData::IntCompare { kind, args, .. } => {
                    let lhs = mapper.get_mapped(args[0]);
                    let rhs = mapper.get_mapped(args[1]);
                    let operand_ty = func.dfg.values[args[0]].ty.clone();
                    if operand_ty == Type::I32 {
                        icmp_op!(
                            kind, lhs, rhs, &mut code, dst, I32Eq, I32Ne, I32LtS, I32LtU, I32LeS,
                            I32LeU, I32GtS, I32GtU, I32GeS, I32GeU
                        )
                    } else if operand_ty == Type::I64
                        || operand_ty == Type::PTR
                        || operand_ty == Type::BOOL
                    {
                        icmp_op!(
                            kind, lhs, rhs, &mut code, dst, I64Eq, I64Ne, I64LtS, I64LtU, I64LeS,
                            I64LeU, I64GtS, I64GtU, I64GeS, I64GeU
                        )
                    } else {
                        panic!(
                            "Unsupported icmp kind {:?} for type {:?}",
                            kind,
                            operand_ty.clone()
                        );
                    }
                }
                InstructionData::StackAddr { slot, offset, .. } => {
                    let base_offset = slot_to_offset[*slot];
                    emit::StackAddr(&mut code, dst, base_offset + *offset);
                }
                InstructionData::StackLoad { slot, offset } => {
                    let base_offset = slot_to_offset[*slot];
                    let ty_val = val_ty(res_vals[0]) as u16;
                    emit::StackLoad(&mut code, dst, ty_val, base_offset + *offset);
                }
                InstructionData::StackStore { slot, value, .. } => {
                    let base_offset = slot_to_offset[*slot];
                    let val_reg = mapper.get_mapped(*value);
                    let ty_val = val_ty(*value) as u16;
                    emit::StackStore(&mut code, val_reg, ty_val, base_offset);
                }
                InstructionData::Load { ptr, offset, .. } => {
                    let ptr_reg = mapper.get_mapped(*ptr);
                    match val_ty(res_vals[0]) {
                        ScalarType::I32 => emit::I32Load(&mut code, dst, ptr_reg, *offset as u32),
                        ScalarType::I64 | ScalarType::Ptr => {
                            emit::I64Load(&mut code, dst, ptr_reg, *offset as u32)
                        }
                        ScalarType::F32 => emit::F32Load(&mut code, dst, ptr_reg, *offset as u32),
                        ScalarType::F64 => emit::F64Load(&mut code, dst, ptr_reg, *offset as u32),
                        ScalarType::I8 => emit::I8Load(&mut code, dst, ptr_reg, *offset as u32),
                        ScalarType::I16 => emit::I16Load(&mut code, dst, ptr_reg, *offset as u32),
                        ty => panic!("Unsupported load type {:?}", ty),
                    }
                }
                InstructionData::Store {
                    ptr: m_ptr,
                    value,
                    offset,
                    ..
                } => {
                    let ptr_reg = mapper.get_mapped(*m_ptr);
                    let val_reg = mapper.get_mapped(*value);
                    match val_ty(*value) {
                        ScalarType::I32 => {
                            emit::I32Store(&mut code, val_reg, ptr_reg, *offset as u32)
                        }
                        ScalarType::I64 | ScalarType::Ptr => {
                            emit::I64Store(&mut code, val_reg, ptr_reg, *offset as u32)
                        }
                        ScalarType::F32 => {
                            emit::F32Store(&mut code, val_reg, ptr_reg, *offset as u32)
                        }
                        ScalarType::F64 => {
                            emit::F64Store(&mut code, val_reg, ptr_reg, *offset as u32)
                        }
                        ScalarType::I8 => {
                            emit::I8Store(&mut code, val_reg, ptr_reg, *offset as u32)
                        }
                        ScalarType::I16 => {
                            emit::I16Store(&mut code, val_reg, ptr_reg, *offset as u32)
                        }
                        ty => panic!("Unsupported store type {:?}", ty),
                    }
                }
                InstructionData::Jump { dest } => {
                    let target_call = *dest;
                    let target_block = func.dfg.block_calls[target_call].block;
                    let moves = calculate_moves(func, target_call, &mut mapper);

                    if moves.is_empty() {
                        let inst_idx = code.len();
                        emit::Jump(&mut code, 0);
                        jump_fixups.push((inst_idx, target_block));
                    } else {
                        let id = push_target(
                            target_call,
                            &mut mapper,
                            &mut data_section,
                            &mut data_fixups,
                        );
                        emit::JumpWithMoves(&mut code, id);
                    }
                }
                InstructionData::Br {
                    condition,
                    then_dest,
                    else_dest,
                } => {
                    let cond_reg = mapper.get_mapped(*condition);
                    let then_idx =
                        push_target(*then_dest, &mut mapper, &mut data_section, &mut data_fixups);
                    let else_idx =
                        push_target(*else_dest, &mut mapper, &mut data_section, &mut data_fixups);
                    emit::Br(&mut code, cond_reg, then_idx, else_idx);
                }
                InstructionData::BrTable { index, table } => {
                    let index_reg = mapper.get_mapped(*index);
                    let table_data = func.dfg.jump_tables[*table].targets.clone();
                    let num_targets = table_data.len() as u32;

                    let mut br_offset = 0;
                    for (i, &target_call) in table_data.iter().enumerate() {
                        let id = push_target(
                            target_call,
                            &mut mapper,
                            &mut data_section,
                            &mut data_fixups,
                        );
                        if i == 0 {
                            br_offset = id;
                        }
                    }

                    emit::BrTable(&mut code, index_reg, br_offset, num_targets);
                }
                InstructionData::FloatCompare { kind, args, .. } => {
                    let lhs = mapper.get_mapped(args[0]);
                    let rhs = mapper.get_mapped(args[1]);
                    let operand_ty = func.dfg.values[args[0]].ty.clone();
                    if operand_ty == Type::F32 {
                        fcmp_op!(
                            kind, lhs, rhs, &mut code, dst, F32Eq, F32Ne, F32Lt, F32Le, F32Gt,
                            F32Ge
                        )
                    } else if operand_ty == Type::F64 {
                        fcmp_op!(
                            kind, lhs, rhs, &mut code, dst, F64Eq, F64Ne, F64Lt, F64Le, F64Gt,
                            F64Ge
                        )
                    } else {
                        panic!(
                            "Unsupported float compare {:?} for type {:?}",
                            kind, operand_ty
                        );
                    }
                }
                InstructionData::Return { values } => {
                    let ret_vals: Vec<_> =
                        func.dfg.get_value_list(*values).iter().copied().collect();
                    let ret_regs: Vec<u16> =
                        ret_vals.iter().map(|&v| mapper.get_mapped(v)).collect();
                    let num_vals = ret_regs.len() as u32;

                    // Store return registers in data section
                    let data_offset = data_section.add_return_regs(&ret_regs);

                    emit::Return(&mut code, data_offset, num_vals);
                }
                InstructionData::Unary { opcode, arg, .. } => {
                    let arg_reg = mapper.get_mapped(*arg);
                    let from_ty = val_ty(*arg);
                    let to_ty = val_ty(res_vals[0]);
                    match opcode {
                        IrOpcode::ExtendS => {
                            let f = from_ty as u16;
                            let t = to_ty as u16;
                            emit::ExtendS(&mut code, dst, arg_reg, (t << 8) | f);
                        }
                        IrOpcode::ExtendU => {
                            let f = from_ty as u16;
                            let t = to_ty as u16;
                            emit::ExtendU(&mut code, dst, arg_reg, (t << 8) | f);
                        }
                        IrOpcode::Wrap => {
                            emit::Wrap(&mut code, dst, arg_reg);
                        }
                        IrOpcode::FloatToIntS => match (from_ty, to_ty) {
                            (ScalarType::F32, ScalarType::I32) => {
                                emit::I32TruncF32S(&mut code, dst, arg_reg)
                            }
                            (ScalarType::F64, ScalarType::I32) => {
                                emit::I32TruncF64S(&mut code, dst, arg_reg)
                            }
                            (ScalarType::F32, ScalarType::I64) => {
                                emit::I64TruncF32S(&mut code, dst, arg_reg)
                            }
                            (ScalarType::F64, ScalarType::I64) => {
                                emit::I64TruncF64S(&mut code, dst, arg_reg)
                            }
                            _ => panic!("Unsupported TruncS: {:?} -> {:?}", from_ty, to_ty),
                        },
                        IrOpcode::FloatToIntU => match (from_ty, to_ty) {
                            (ScalarType::F32, ScalarType::I32) => {
                                emit::I32TruncF32U(&mut code, dst, arg_reg)
                            }
                            (ScalarType::F64, ScalarType::I32) => {
                                emit::I32TruncF64U(&mut code, dst, arg_reg)
                            }
                            (ScalarType::F32, ScalarType::I64) => {
                                emit::I64TruncF32U(&mut code, dst, arg_reg)
                            }
                            (ScalarType::F64, ScalarType::I64) => {
                                emit::I64TruncF64U(&mut code, dst, arg_reg)
                            }
                            _ => panic!("Unsupported TruncU: {:?} -> {:?}", from_ty, to_ty),
                        },
                        IrOpcode::FloatToIntSatS => match (from_ty, to_ty) {
                            (ScalarType::F32, ScalarType::I32) => {
                                emit::I32TruncSatF32S(&mut code, dst, arg_reg)
                            }
                            (ScalarType::F64, ScalarType::I32) => {
                                emit::I32TruncSatF64S(&mut code, dst, arg_reg)
                            }
                            (ScalarType::F32, ScalarType::I64) => {
                                emit::I64TruncSatF32S(&mut code, dst, arg_reg)
                            }
                            (ScalarType::F64, ScalarType::I64) => {
                                emit::I64TruncSatF64S(&mut code, dst, arg_reg)
                            }
                            _ => panic!("Unsupported TruncSatS: {:?} -> {:?}", from_ty, to_ty),
                        },
                        IrOpcode::FloatToIntSatU => match (from_ty, to_ty) {
                            (ScalarType::F32, ScalarType::I32) => {
                                emit::I32TruncSatF32U(&mut code, dst, arg_reg)
                            }
                            (ScalarType::F64, ScalarType::I32) => {
                                emit::I32TruncSatF64U(&mut code, dst, arg_reg)
                            }
                            (ScalarType::F32, ScalarType::I64) => {
                                emit::I64TruncSatF32U(&mut code, dst, arg_reg)
                            }
                            (ScalarType::F64, ScalarType::I64) => {
                                emit::I64TruncSatF64U(&mut code, dst, arg_reg)
                            }
                            _ => panic!("Unsupported TruncSatU: {:?} -> {:?}", from_ty, to_ty),
                        },
                        IrOpcode::IntToFloatS => match (from_ty, to_ty) {
                            (ScalarType::I32, ScalarType::F32) => {
                                emit::F32ConvertI32S(&mut code, dst, arg_reg)
                            }
                            (ScalarType::I64, ScalarType::F32) => {
                                emit::F32ConvertI64S(&mut code, dst, arg_reg)
                            }
                            (ScalarType::I32, ScalarType::F64) => {
                                emit::F64ConvertI32S(&mut code, dst, arg_reg)
                            }
                            (ScalarType::I64, ScalarType::F64) => {
                                emit::F64ConvertI64S(&mut code, dst, arg_reg)
                            }
                            _ => panic!("Unsupported ConvertS: {:?} -> {:?}", from_ty, to_ty),
                        },
                        IrOpcode::IntToFloatU => match (from_ty, to_ty) {
                            (ScalarType::I32, ScalarType::F32) => {
                                emit::F32ConvertI32U(&mut code, dst, arg_reg)
                            }
                            (ScalarType::I64, ScalarType::F32) => {
                                emit::F32ConvertI64U(&mut code, dst, arg_reg)
                            }
                            (ScalarType::I32, ScalarType::F64) => {
                                emit::F64ConvertI32U(&mut code, dst, arg_reg)
                            }
                            (ScalarType::I64, ScalarType::F64) => {
                                emit::F64ConvertI64U(&mut code, dst, arg_reg)
                            }
                            _ => panic!("Unsupported ConvertU: {:?} -> {:?}", from_ty, to_ty),
                        },
                        IrOpcode::FloatDemote => emit::F32DemoteF64(&mut code, dst, arg_reg),
                        IrOpcode::FloatPromote => emit::F64PromoteF32(&mut code, dst, arg_reg),
                        IrOpcode::Reinterpret => {
                            emit::RegMove(&mut code, dst, arg_reg);
                        }
                        IrOpcode::FAbs => match from_ty {
                            ScalarType::F32 => emit::F32Abs(&mut code, dst, arg_reg),
                            ScalarType::F64 => emit::F64Abs(&mut code, dst, arg_reg),
                            _ => panic!("Unsupported Abs for type {:?}", from_ty),
                        },
                        IrOpcode::FNeg => match from_ty {
                            ScalarType::F32 => emit::F32Neg(&mut code, dst, arg_reg),
                            ScalarType::F64 => emit::F64Neg(&mut code, dst, arg_reg),
                            _ => panic!("Unsupported Fneg for type {:?}", from_ty),
                        },
                        IrOpcode::INeg => {
                            // ... implement if needed
                            todo!("Ineg not implemented");
                        }
                        IrOpcode::FSqrt => match from_ty {
                            ScalarType::F32 => emit::F32Sqrt(&mut code, dst, arg_reg),
                            ScalarType::F64 => emit::F64Sqrt(&mut code, dst, arg_reg),
                            _ => panic!("Unsupported Sqrt for type {:?}", from_ty),
                        },
                        IrOpcode::FCeil => match from_ty {
                            ScalarType::F32 => emit::F32Ceil(&mut code, dst, arg_reg),
                            ScalarType::F64 => emit::F64Ceil(&mut code, dst, arg_reg),
                            _ => panic!("Unsupported Ceil for type {:?}", from_ty),
                        },
                        IrOpcode::FFloor => match from_ty {
                            ScalarType::F32 => emit::F32Floor(&mut code, dst, arg_reg),
                            ScalarType::F64 => emit::F64Floor(&mut code, dst, arg_reg),
                            _ => panic!("Unsupported Floor for type {:?}", from_ty),
                        },
                        IrOpcode::FTrunc => match from_ty {
                            ScalarType::F32 => emit::F32Trunc(&mut code, dst, arg_reg),
                            ScalarType::F64 => emit::F64Trunc(&mut code, dst, arg_reg),
                            _ => panic!("Unsupported Trunc for type {:?}", from_ty),
                        },
                        IrOpcode::FNearest => match from_ty {
                            ScalarType::F32 => emit::F32Nearest(&mut code, dst, arg_reg),
                            ScalarType::F64 => emit::F64Nearest(&mut code, dst, arg_reg),
                            _ => panic!("Unsupported Nearest for type {:?}", from_ty),
                        },
                        IrOpcode::IClz => match from_ty {
                            ScalarType::I32 => emit::I32Clz(&mut code, dst, arg_reg),
                            ScalarType::I64 => emit::I64Clz(&mut code, dst, arg_reg),
                            _ => panic!("Unsupported Clz for type {:?}", from_ty),
                        },
                        IrOpcode::ICtz => match from_ty {
                            ScalarType::I32 => emit::I32Ctz(&mut code, dst, arg_reg),
                            ScalarType::I64 => emit::I64Ctz(&mut code, dst, arg_reg),
                            _ => panic!("Unsupported Ctz for type {:?}", from_ty),
                        },
                        IrOpcode::IPopcnt => match from_ty {
                            ScalarType::I32 => emit::I32Popcnt(&mut code, dst, arg_reg),
                            ScalarType::I64 => emit::I64Popcnt(&mut code, dst, arg_reg),
                            _ => panic!("Unsupported Popcnt for type {:?}", from_ty),
                        },
                        IrOpcode::IEqz => match from_ty {
                            ScalarType::I32 => emit::I32Eqz(&mut code, dst, arg_reg),
                            ScalarType::I64 => emit::I64Eqz(&mut code, dst, arg_reg),
                            _ => panic!("Unsupported Eqz for type {:?}", from_ty),
                        },
                        _ => {
                            todo!("Unsupported unary op {:?}", opcode);
                        }
                    }
                }
                InstructionData::IntToPtr { arg } | InstructionData::PtrToInt { arg, .. } => {
                    let arg_reg = mapper.get_mapped(*arg);
                    emit::RegMove(&mut code, dst, arg_reg);
                }
                InstructionData::Call { func_id, args, .. } => {
                    // Get result values and allocate registers for them
                    let res_vals: Vec<_> = func.dfg.inst_results(inst).iter().copied().collect();
                    let mut ret_regs = Vec::with_capacity(res_vals.len());
                    for &v in &res_vals {
                        ret_regs.push(mapper.alloc_and_map(v));
                    }
                    let args_regs: Vec<u16> = func
                        .dfg
                        .get_value_list(*args)
                        .iter()
                        .map(|&v| mapper.get_mapped(v))
                        .collect();

                    // Store register lists in data section (func_id goes in instruction)
                    let data_offset = data_section.add_call_data(&ret_regs, &args_regs);

                    // imm32 = func_id, aux = data_offset
                    emit::Call(&mut code, func_id.as_u32(), data_offset);
                }
                InstructionData::CallIndirect { ptr, args, .. } => {
                    let ptr_reg = mapper.get_mapped(*ptr);
                    // Get result values and allocate registers for them
                    let res_vals: Vec<_> = func.dfg.inst_results(inst).iter().copied().collect();
                    let mut ret_regs = Vec::with_capacity(res_vals.len());
                    for &v in &res_vals {
                        ret_regs.push(mapper.alloc_and_map(v));
                    }
                    let args_regs: Vec<u16> = func
                        .dfg
                        .get_value_list(*args)
                        .iter()
                        .map(|&v| mapper.get_mapped(v))
                        .collect();
                    let num_args = args_regs.len() as u16;

                    // Store in data section
                    let data_offset = data_section.add_call_data(&ret_regs, &args_regs);

                    let packed = Instruction::pack_counts(ret_regs.len() as u16, num_args);
                    emit::CallIndirect(&mut code, ptr_reg, data_offset, packed);
                }
                InstructionData::CallIntrinsic { intrinsic, .. } => {
                    // Note: args are stored in ValueList side table
                    let args_regs: Vec<u16> = vec![]; // Would need to retrieve from side table
                    // Get result values and allocate registers for them
                    let res_vals: Vec<_> = func.dfg.inst_results(inst).iter().copied().collect();
                    let mut ret_regs = Vec::with_capacity(res_vals.len());
                    for &v in &res_vals {
                        ret_regs.push(mapper.alloc_and_map(v));
                    }

                    // Try to inline common math intrinsics
                    if ret_regs.len() <= 1
                        && try_emit_inline_intrinsic(&mut code, dst, *intrinsic, &args_regs)
                    {
                        // Successfully inlined
                    } else {
                        // Fall back to generic intrinsic call
                        let num_args = args_regs.len() as u16;
                        let data_offset = data_section.add_call_data(&ret_regs, &args_regs);

                        let packed = Instruction::pack_counts(ret_regs.len() as u16, num_args);
                        emit::CallIntrinsic(&mut code, intrinsic.as_u16(), data_offset, packed);
                    }
                }
                InstructionData::PtrIndex { ptr, index, imm_id } => {
                    let ptr_reg = mapper.get_mapped(*ptr);
                    let index_reg = mapper.get_mapped(*index);
                    let imm = func.dfg.get_ptr_imm(*imm_id);
                    // Store offset as u32 bits (will be reinterpreted as i32 by interpreter)
                    emit::PtrIndex(
                        &mut code,
                        dst,
                        ptr_reg,
                        index_reg,
                        imm.scale as u32,
                        imm.offset as u32,
                    );
                }
                InstructionData::PtrOffset { ptr, offset } => {
                    let ptr_reg = mapper.get_mapped(*ptr);
                    let imm = *offset as i64;
                    emit::I64AddImm(&mut code, dst, ptr_reg, imm as u64);
                }
                InstructionData::Unreachable => {
                    emit::Unreachable(&mut code);
                }
                // Vector operations - not yet implemented in interpreter
                InstructionData::Ternary { opcode, args } => {
                    if *opcode == IrOpcode::Select {
                        let cond_reg = mapper.get_mapped(args[0]);
                        let then_reg = mapper.get_mapped(args[1]);
                        let else_reg = mapper.get_mapped(args[2]);
                        emit::Select(&mut code, dst, cond_reg, then_reg, else_reg as u32);
                    } else {
                        todo!("Implement interpreter for ternary vector operations")
                    }
                }
                InstructionData::VectorOpWithExt { .. } => {
                    todo!("Implement interpreter for masked vector operations")
                }
                InstructionData::VectorLoadStrided { .. } => {
                    todo!("Implement interpreter for vector strided load")
                }
                InstructionData::VectorStoreStrided { .. } => {
                    todo!("Implement interpreter for vector strided store")
                }
                InstructionData::VectorGather { .. } => {
                    todo!("Implement interpreter for vector gather")
                }
                InstructionData::VectorScatter { .. } => {
                    todo!("Implement interpreter for vector scatter")
                }
                InstructionData::Shuffle { .. } => {
                    todo!("Implement interpreter for vector shuffle")
                }
                InstructionData::Nop => {}
            }

            // Free registers
            idata.visit_operands(&func.dfg, |v| {
                mapper.free_if_last_use(v, current_ir_inst_idx)
            });
            for &rv in res_vals {
                // Results are defined at pc + 1
                mapper.free_if_last_use(rv, current_ir_inst_idx + 1);
            }
        }
    }

    // Patch jump targets - inst_idx is instruction index
    for (inst_idx, target_block) in jump_fixups {
        let target_pc = block_to_pc[target_block];
        if target_pc == 0 && target_block != entry {
            panic!("Missing block");
        }
        // Patch the imm32 field (low 32 bits of imm64)
        code[inst_idx].imm64 = target_pc as u64;
    }

    // Patch data section targets
    for (data_idx, target_block) in data_fixups {
        let target_pc = block_to_pc[target_block];
        if target_pc == 0 && target_block != entry {
            panic!("Missing block in data_fixups");
        }
        data_section.jump_targets[data_idx].pc = target_pc;
    }

    CompiledFunction {
        module_id,
        func_id,
        code,
        data_section,
        stack_slots_sizes: func
            .stack_slots
            .iter()
            .map(|(_, d)| d.size as usize)
            .collect(),
        param_indices,
        ret_indices: Vec::new(), // TODO: collect return indices if needed
        register_count: mapper.next_register as usize,
        constant_pool: func
            .dfg
            .constant_pool
            .values()
            .map(|d| match d {
                veloc_ir::ConstantPoolData::Bytes(b) => b.clone(),
            })
            .collect(),
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
) -> Vec<(u16, u16)> {
    let target_block = func.dfg.block_calls[call].block;
    let args = func.dfg.get_value_list(func.dfg.block_calls[call].args);
    let params = &func.layout.blocks[target_block].params;

    // 1. Collect all move requests with pre-allocated capacity
    let mut pending: Vec<(u16, u16)> = Vec::with_capacity(params.len());
    for (&p, &a) in params.iter().zip(args.iter()) {
        let d = mapper.alloc_and_map(p);
        let s = mapper.get_mapped(a);
        if d != s {
            pending.push((d, s));
        }
    }

    // 2. Resolve parallel moves into an ordered sequence
    let mut result: Vec<(u16, u16)> = Vec::with_capacity(pending.len() + 1); // +1 for potential temp move

    while !pending.is_empty() {
        // Try to find a move whose destination is not used as a source by any other move
        let mut best_i = None;
        for i in 0..pending.len() {
            let (dst, _) = pending[i];
            let is_read_by_others = pending
                .iter()
                .enumerate()
                .any(|(j, &(_, other_src))| i != j && dst == other_src);

            if !is_read_by_others {
                best_i = Some(i);
                break;
            }
        }

        if let Some(i) = best_i {
            // Use swap_remove for O(1) removal instead of O(n)
            let (d, s) = pending.swap_remove(i);
            result.push((d, s));
            continue;
        }

        // 3. Cycle detected. We need to break it by using a temporary register.
        // Pick the first move (d, s) and save s to a temp register.
        let (d, s) = pending.swap_remove(0);

        // Prefer reusing a free register over allocating a new one
        let temp = mapper.free_registers.pop().unwrap_or_else(|| {
            let r = mapper.next_register;
            mapper.next_register += 1;
            r
        });

        result.push((temp, s));

        // Replace (d, s) with (d, temp). Since temp is fresh, (d, temp)
        // will eventually be considered safe to move into d.
        pending.push((d, temp));
    }

    result
}
