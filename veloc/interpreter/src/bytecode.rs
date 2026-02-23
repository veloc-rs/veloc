use crate::host::ModuleId;
use ::alloc::vec::Vec;
use cranelift_entity::SecondaryMap;
use veloc_analyzer::{LiveInterval, analyze_liveness};
use veloc_ir::{
    Block, FuncId, Function, InstructionData, Intrinsic, Opcode as IrOpcode, StackSlot, Type, Value,
};

pub const STACK_TYPE_I8: u8 = 1;
pub const STACK_TYPE_I16: u8 = 2;
pub const STACK_TYPE_I32: u8 = 3;
pub const STACK_TYPE_I64: u8 = 4;
pub const STACK_TYPE_F32: u8 = 5;
pub const STACK_TYPE_F64: u8 = 6;

pub const EXTEND_TYPE_I8: u8 = 0;
pub const EXTEND_TYPE_I16: u8 = 1;
pub const EXTEND_TYPE_I32: u8 = 2;

pub const RETURN_VOID: u8 = 0;
pub const RETURN_HAS_VALUE: u8 = 1;

/// Fixed instruction size in bytes - all instructions are exactly this size
pub const INSTRUCTION_SIZE: usize = 16;

/// Represents a fixed-size bytecode instruction (16 bytes)
/// Layout: opcode(1) + pad(1) + dst(2) + src1(2) + src2(2) + imm64(8) = 16 bytes
#[derive(Clone, Copy, Debug)]
pub struct Instruction {
    /// Opcode (1 byte)
    pub opcode: u8,
    /// Destination register (2 bytes)
    pub dst: u16,
    /// First source register (2 bytes)
    pub src1: u16,
    /// Second source register (2 bytes)
    pub src2: u16,
    /// 64-bit immediate/offset/aux data
    pub imm64: u64,
}

impl Instruction {
    /// Create a new instruction from split imm32/aux (for macro compatibility)
    #[inline(always)]
    pub const fn new(opcode: u8, dst: u16, src1: u16, src2: u16, imm32: u32, aux: u32) -> Self {
        Self {
            opcode,
            dst,
            src1,
            src2,
            imm64: (imm32 as u64) | ((aux as u64) << 32),
        }
    }

    /// Create a new instruction with full 64-bit immediate
    #[inline(always)]
    pub const fn with_imm64(opcode: u8, dst: u16, src1: u16, imm64: u64) -> Self {
        Self {
            opcode,
            dst,
            src1,
            src2: 0,
            imm64,
        }
    }

    /// Get low 32 bits of immediate
    #[inline(always)]
    pub const fn imm32(&self) -> u32 {
        self.imm64 as u32
    }

    /// Get high 32 bits of immediate (aux)
    #[inline(always)]
    pub const fn aux(&self) -> u32 {
        (self.imm64 >> 32) as u32
    }

    /// Pack register counts for call instructions: (num_rets << 16) | num_args
    #[inline(always)]
    pub const fn pack_counts(num_rets: u16, num_args: u16) -> u32 {
        ((num_rets as u32) << 16) | (num_args as u32)
    }

    /// Unpack register counts: returns (num_rets, num_args)
    #[inline(always)]
    pub const fn unpack_counts(&self) -> (u16, u16) {
        ((self.aux() >> 16) as u16, self.aux() as u16)
    }

    /// Decode an instruction from a raw pointer
    #[inline(always)]
    pub unsafe fn decode(ptr: *const u8) -> Self {
        unsafe { (ptr as *const Instruction).read_unaligned() }
    }
}

macro_rules! define_opcodes {
    ($(
        $name:ident {
            $( $arg:ident : $ty:ty $( => $field:ident )? ),*
        }
    );* $(;)?) => {
        #[repr(u8)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum Opcode {
            $($name),*
        }

        pub mod emit {
            use super::*;
            $(
                #[allow(non_snake_case)]
                #[inline(always)]
                pub fn $name(
                    code: &mut Vec<Instruction>,
                    $($arg : $ty),*
                ) {
                    #[allow(unused_mut, unused_assignments)]
                    let mut inst = Instruction {
                        opcode: Opcode::$name as u8,
                        dst: 0,
                        src1: 0,
                        src2: 0,
                        imm64: 0,
                    };

                    $(
                        define_opcodes!(@assign inst, $arg, $( $field )? );
                    )*

                    code.push(inst);
                }
            )*
        }
    };

    // Helper to assign fields
    (@assign $inst:ident, $val:ident, dst) => { $inst.dst = $val; };
    (@assign $inst:ident, $val:ident, src1) => { $inst.src1 = $val; };
    (@assign $inst:ident, $val:ident, src2) => { $inst.src2 = $val; };
    (@assign $inst:ident, $val:ident, imm32) => { $inst.imm64 = ($inst.imm64 & 0xFFFFFFFF00000000) | ($val as u64); };
    (@assign $inst:ident, $val:ident, aux) => { $inst.imm64 = ($inst.imm64 & 0x00000000FFFFFFFF) | (($val as u64) << 32); };
    (@assign $inst:ident, $val:ident, imm64) => { $inst.imm64 = $val; };
    // If no field mapped, assume it's one of the standard ones by name
    (@assign $inst:ident, $val:ident, ) => {
        define_opcodes!(@assign $inst, $val, $val);
    };
}

// Field mapping conventions:
// - dst: destination register (for 2-result ops, aux may hold 2nd dst)
// - src1: first source register
// - src2: second source register
// - imm32: 32-bit immediate, offset, or low 32-bits of 64-bit immediate
// - aux: high 32-bits of 64-bit immediate, data section offset, or packed counts

define_opcodes! {
    // === Constants ===
    Iconst { dst: u16, imm64: u64 };
    Fconst { dst: u16, imm64: u64 };
    Bconst { dst: u16, val: u16 => src2 };
    Vconst { dst: u16, pool_id: u32 => imm32 };

    // === I32 Arithmetic ===
    I32Add { dst: u16, src1: u16, src2: u16 };
    I32Sub { dst: u16, src1: u16, src2: u16 };
    I32Mul { dst: u16, src1: u16, src2: u16 };
    I32DivS { dst: u16, src1: u16, src2: u16 };
    I32DivU { dst: u16, src1: u16, src2: u16 };
    I32RemS { dst: u16, src1: u16, src2: u16 };
    I32RemU { dst: u16, src1: u16, src2: u16 };
    I32And { dst: u16, src1: u16, src2: u16 };
    I32Or { dst: u16, src1: u16, src2: u16 };
    I32Xor { dst: u16, src1: u16, src2: u16 };
    I32Shl { dst: u16, src1: u16, src2: u16 };
    I32ShrS { dst: u16, src1: u16, src2: u16 };
    I32ShrU { dst: u16, src1: u16, src2: u16 };
    I32RotL { dst: u16, src1: u16, src2: u16 };
    I32RotR { dst: u16, src1: u16, src2: u16 };

    I32AddImm { dst: u16, src1: u16, imm: u32 => imm32 };
    I32SubImm { dst: u16, src1: u16, imm: u32 => imm32 };
    I32AndImm { dst: u16, src1: u16, imm: u32 => imm32 };
    I32OrImm { dst: u16, src1: u16, imm: u32 => imm32 };
    I32XorImm { dst: u16, src1: u16, imm: u32 => imm32 };
    I32ShlImm { dst: u16, src1: u16, imm: u32 => imm32 };
    I32ShrSImm { dst: u16, src1: u16, imm: u32 => imm32 };
    I32ShrUImm { dst: u16, src1: u16, imm: u32 => imm32 };

    // === I64 Arithmetic ===
    I64Add { dst: u16, src1: u16, src2: u16 };
    I64Sub { dst: u16, src1: u16, src2: u16 };
    I64Mul { dst: u16, src1: u16, src2: u16 };
    I64DivS { dst: u16, src1: u16, src2: u16 };
    I64DivU { dst: u16, src1: u16, src2: u16 };
    I64RemS { dst: u16, src1: u16, src2: u16 };
    I64RemU { dst: u16, src1: u16, src2: u16 };
    I64And { dst: u16, src1: u16, src2: u16 };
    I64Or { dst: u16, src1: u16, src2: u16 };
    I64Xor { dst: u16, src1: u16, src2: u16 };
    I64Shl { dst: u16, src1: u16, src2: u16 };
    I64ShrS { dst: u16, src1: u16, src2: u16 };
    I64ShrU { dst: u16, src1: u16, src2: u16 };
    I64RotL { dst: u16, src1: u16, src2: u16 };
    I64RotR { dst: u16, src1: u16, src2: u16 };

    I64AddImm { dst: u16, src1: u16, imm64: u64 };
    I64SubImm { dst: u16, src1: u16, imm64: u64 };
    I64AndImm { dst: u16, src1: u16, imm64: u64 };
    I64OrImm { dst: u16, src1: u16, imm64: u64 };
    I64XorImm { dst: u16, src1: u16, imm64: u64 };
    I64ShlImm { dst: u16, src1: u16, imm64: u64 };
    I64ShrSImm { dst: u16, src1: u16, imm64: u64 };
    I64ShrUImm { dst: u16, src1: u16, imm64: u64 };

    // === Comparisons ===
    I32Eq { dst: u16, src1: u16, src2: u16 };
    I32Ne { dst: u16, src1: u16, src2: u16 };
    I32LtS { dst: u16, src1: u16, src2: u16 };
    I32LtU { dst: u16, src1: u16, src2: u16 };
    I32LeS { dst: u16, src1: u16, src2: u16 };
    I32LeU { dst: u16, src1: u16, src2: u16 };
    I32GtS { dst: u16, src1: u16, src2: u16 };
    I32GtU { dst: u16, src1: u16, src2: u16 };
    I32GeS { dst: u16, src1: u16, src2: u16 };
    I32GeU { dst: u16, src1: u16, src2: u16 };

    I64Eq { dst: u16, src1: u16, src2: u16 };
    I64Ne { dst: u16, src1: u16, src2: u16 };
    I64LtS { dst: u16, src1: u16, src2: u16 };
    I64LtU { dst: u16, src1: u16, src2: u16 };
    I64LeS { dst: u16, src1: u16, src2: u16 };
    I64LeU { dst: u16, src1: u16, src2: u16 };
    I64GtS { dst: u16, src1: u16, src2: u16 };
    I64GtU { dst: u16, src1: u16, src2: u16 };
    I64GeS { dst: u16, src1: u16, src2: u16 };
    I64GeU { dst: u16, src1: u16, src2: u16 };

    // === Float Arithmetic ===
    F32Add { dst: u16, src1: u16, src2: u16 };
    F32Sub { dst: u16, src1: u16, src2: u16 };
    F32Mul { dst: u16, src1: u16, src2: u16 };
    F32Div { dst: u16, src1: u16, src2: u16 };
    F32Neg { dst: u16, src1: u16 };
    F32Abs { dst: u16, src1: u16 };
    F32Sqrt { dst: u16, src1: u16 };
    F32Ceil { dst: u16, src1: u16 };
    F32Floor { dst: u16, src1: u16 };
    F32Trunc { dst: u16, src1: u16 };
    F32Nearest { dst: u16, src1: u16 };
    F32Min { dst: u16, src1: u16, src2: u16 };
    F32Max { dst: u16, src1: u16, src2: u16 };
    F32CopySign { dst: u16, src1: u16, src2: u16 };

    F64Add { dst: u16, src1: u16, src2: u16 };
    F64Sub { dst: u16, src1: u16, src2: u16 };
    F64Mul { dst: u16, src1: u16, src2: u16 };
    F64Div { dst: u16, src1: u16, src2: u16 };
    F64Neg { dst: u16, src1: u16 };
    F64Abs { dst: u16, src1: u16 };
    F64Sqrt { dst: u16, src1: u16 };
    F64Ceil { dst: u16, src1: u16 };
    F64Floor { dst: u16, src1: u16 };
    F64Trunc { dst: u16, src1: u16 };
    F64Nearest { dst: u16, src1: u16 };
    F64Min { dst: u16, src1: u16, src2: u16 };
    F64Max { dst: u16, src1: u16, src2: u16 };
    F64CopySign { dst: u16, src1: u16, src2: u16 };

    F32Eq { dst: u16, src1: u16, src2: u16 };
    F32Ne { dst: u16, src1: u16, src2: u16 };
    F32Lt { dst: u16, src1: u16, src2: u16 };
    F32Le { dst: u16, src1: u16, src2: u16 };
    F32Gt { dst: u16, src1: u16, src2: u16 };
    F32Ge { dst: u16, src1: u16, src2: u16 };
    F64Eq { dst: u16, src1: u16, src2: u16 };
    F64Ne { dst: u16, src1: u16, src2: u16 };
    F64Lt { dst: u16, src1: u16, src2: u16 };
    F64Le { dst: u16, src1: u16, src2: u16 };
    F64Gt { dst: u16, src1: u16, src2: u16 };
    F64Ge { dst: u16, src1: u16, src2: u16 };

    // === Memory ===
    I32Load { dst: u16, ptr: u16 => src1, offset: u32 => imm32 };
    I64Load { dst: u16, ptr: u16 => src1, offset: u32 => imm32 };
    F32Load { dst: u16, ptr: u16 => src1, offset: u32 => imm32 };
    F64Load { dst: u16, ptr: u16 => src1, offset: u32 => imm32 };
    I8Load { dst: u16, ptr: u16 => src1, offset: u32 => imm32 };
    I16Load { dst: u16, ptr: u16 => src1, offset: u32 => imm32 };

    I32Store { val: u16 => src1, ptr: u16 => src2, offset: u32 => imm32 };
    I64Store { val: u16 => src1, ptr: u16 => src2, offset: u32 => imm32 };
    F32Store { val: u16 => src1, ptr: u16 => src2, offset: u32 => imm32 };
    F64Store { val: u16 => src1, ptr: u16 => src2, offset: u32 => imm32 };
    I8Store { val: u16 => src1, ptr: u16 => src2, offset: u32 => imm32 };
    I16Store { val: u16 => src1, ptr: u16 => src2, offset: u32 => imm32 };

    // === Conversions ===
    ExtendS { dst: u16, src: u16 => src1, ty: u16 => src2 };
    ExtendU { dst: u16, src: u16 => src1, ty: u16 => src2 };
    Wrap { dst: u16, src: u16 => src1 };

    I32TruncF32S { dst: u16, src: u16 => src1 };
    I32TruncF32U { dst: u16, src: u16 => src1 };
    I32TruncF64S { dst: u16, src: u16 => src1 };
    I32TruncF64U { dst: u16, src: u16 => src1 };
    I64TruncF32S { dst: u16, src: u16 => src1 };
    I64TruncF32U { dst: u16, src: u16 => src1 };
    I64TruncF64S { dst: u16, src: u16 => src1 };
    I64TruncF64U { dst: u16, src: u16 => src1 };
    I32TruncSatF32S { dst: u16, src: u16 => src1 };
    I32TruncSatF32U { dst: u16, src: u16 => src1 };
    I32TruncSatF64S { dst: u16, src: u16 => src1 };
    I32TruncSatF64U { dst: u16, src: u16 => src1 };
    I64TruncSatF32S { dst: u16, src: u16 => src1 };
    I64TruncSatF32U { dst: u16, src: u16 => src1 };
    I64TruncSatF64S { dst: u16, src: u16 => src1 };
    I64TruncSatF64U { dst: u16, src: u16 => src1 };

    F32ConvertI32S { dst: u16, src: u16 => src1 };
    F32ConvertI32U { dst: u16, src: u16 => src1 };
    F32ConvertI64S { dst: u16, src: u16 => src1 };
    F32ConvertI64U { dst: u16, src: u16 => src1 };
    F64ConvertI32S { dst: u16, src: u16 => src1 };
    F64ConvertI32U { dst: u16, src: u16 => src1 };
    F64ConvertI64S { dst: u16, src: u16 => src1 };
    F64ConvertI64U { dst: u16, src: u16 => src1 };
    F32DemoteF64 { dst: u16, src: u16 => src1 };
    F64PromoteF32 { dst: u16, src: u16 => src1 };
    Bitcast { dst: u16, src: u16 => src1 };

    // === Bitwise ===
    I32Clz { dst: u16, src: u16 => src1 };
    I32Ctz { dst: u16, src: u16 => src1 };
    I32Popcnt { dst: u16, src: u16 => src1 };
    I64Clz { dst: u16, src: u16 => src1 };
    I64Ctz { dst: u16, src: u16 => src1 };
    I64Popcnt { dst: u16, src: u16 => src1 };
    I32Eqz { dst: u16, src_val: u16 => src1 };
    I64Eqz { dst: u16, src_val: u16 => src1 };

    // === Stack ===
    StackAddr { dst: u16, offset: u32 => imm32 };
    StackLoad { dst: u16, ty: u16 => src2, offset: u32 => imm32 };
    StackStore { val: u16 => src1, ty: u16 => src2, offset: u32 => imm32 };

    PtrIndex { dst: u16, ptr: u16 => src1, index: u16 => src2, scale: u32 => imm32, offset: u32 => aux };

    // === Control Flow ===
    Jump { pc: u32 => imm32 };
    JumpWithMoves { data_offset: u32 => imm32 };
    Br { cond: u16 => dst, then_idx: u32 => imm32, else_idx: u32 => aux };
    BrTable { idx_reg: u16 => dst, data_offset: u32 => imm32, num_targets: u32 => aux };

    Select { dst: u16, cond: u16 => src1, then_reg: u16 => src2, else_reg: u32 => imm32 };
    Return { data_offset: u32 => imm32, num_vals: u32 => aux };

    Call { func_id: u32 => imm32, data_offset: u32 => aux };
    CallIndirect { ptr: u16 => src1, data_offset: u32 => imm32, counts: u32 => aux };
    CallIntrinsic { intrinsic: u16 => src1, data_offset: u32 => imm32, counts: u32 => aux };

    GlobalAddr { dst: u16, global_idx: u32 => imm32 };

    RegMove { dst: u16, src: u16 => src1 };
    Unreachable {};
}

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
    pub fn add_jump_targets(&mut self, targets: &[JumpTarget]) -> (u32, u32) {
        let offset = self.jump_targets.len();
        let num_targets = targets.len() as u32;
        self.jump_targets.extend_from_slice(targets);
        (offset as u32, num_targets)
    }
}

pub struct CompiledFunction {
    pub module_id: ModuleId,
    pub func_id: FuncId,
    pub code: Vec<Instruction>,
    /// Data section: u16_data for register lists, jump_targets for jump targets
    pub data_section: DataSection,
    pub stack_slots_sizes: Vec<usize>,
    pub param_indices: Vec<u16>,
    pub ret_indices: Vec<u16>, // Return value register indices (support multi-value)
    pub register_count: usize,
    pub constant_pool: Vec<Vec<u8>>,
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

pub fn compile_function(module_id: ModuleId, func_id: FuncId, func: &Function) -> CompiledFunction {
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

        let (br_offset, _) = data_section.add_jump_targets(&[JumpTarget {
            pc: 0,
            num_moves,
            moves_offset,
        }]);

        data_fixups.push((br_offset as usize, target_block));
        br_offset
    };

    let get_stack_type = |ty: &Type| -> u16 {
        if *ty == Type::I32 {
            STACK_TYPE_I32 as u16
        } else if *ty == Type::I64 || *ty == Type::PTR {
            STACK_TYPE_I64 as u16
        } else if *ty == Type::F32 {
            STACK_TYPE_F32 as u16
        } else if *ty == Type::F64 {
            STACK_TYPE_F64 as u16
        } else if *ty == Type::I8 {
            STACK_TYPE_I8 as u16
        } else if *ty == Type::I16 {
            STACK_TYPE_I16 as u16
        } else {
            panic!("Unsupported type: {:?}", ty);
        }
    };

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
                    code.push(Instruction::with_imm64(
                        Opcode::Iconst as u8,
                        dst,
                        0,
                        *value,
                    ));
                }
                InstructionData::Fconst { value } => {
                    code.push(Instruction::with_imm64(
                        Opcode::Fconst as u8,
                        dst,
                        0,
                        *value,
                    ));
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
                    let ty = res_vals
                        .first()
                        .map(|&v| func.dfg.value_type(v))
                        .unwrap_or(Type::VOID);
                    let ty_val = get_stack_type(&ty);
                    emit::StackLoad(&mut code, dst, ty_val, base_offset + *offset);
                }
                InstructionData::StackStore { slot, value, .. } => {
                    let base_offset = slot_to_offset[*slot];
                    let val_reg = mapper.get_mapped(*value);
                    let ty = func.dfg.values[*value].ty.clone();
                    let ty_val = get_stack_type(&ty);
                    emit::StackStore(&mut code, val_reg, ty_val, base_offset);
                }
                InstructionData::Load { ptr, offset, .. } => {
                    let ptr_reg = mapper.get_mapped(*ptr);
                    let ty = res_vals
                        .first()
                        .map(|&v| func.dfg.value_type(v))
                        .unwrap_or(Type::VOID);
                    let ty_val = get_stack_type(&ty);

                    match ty_val as u8 {
                        STACK_TYPE_I32 => emit::I32Load(&mut code, dst, ptr_reg, *offset as u32),
                        STACK_TYPE_I64 => emit::I64Load(&mut code, dst, ptr_reg, *offset as u32),
                        STACK_TYPE_F32 => emit::F32Load(&mut code, dst, ptr_reg, *offset as u32),
                        STACK_TYPE_F64 => emit::F64Load(&mut code, dst, ptr_reg, *offset as u32),
                        STACK_TYPE_I8 => emit::I8Load(&mut code, dst, ptr_reg, *offset as u32),
                        STACK_TYPE_I16 => emit::I16Load(&mut code, dst, ptr_reg, *offset as u32),
                        _ => panic!("Unsupported load type {:?}", ty),
                    }
                }
                InstructionData::Store {
                    ptr, value, offset, ..
                } => {
                    let ptr_reg = mapper.get_mapped(*ptr);
                    let val_reg = mapper.get_mapped(*value);
                    let ty = func.dfg.values[*value].ty.clone();
                    let ty_val = get_stack_type(&ty);

                    match ty_val as u8 {
                        STACK_TYPE_I32 => {
                            emit::I32Store(&mut code, val_reg, ptr_reg, *offset as u32)
                        }
                        STACK_TYPE_I64 => {
                            emit::I64Store(&mut code, val_reg, ptr_reg, *offset as u32)
                        }
                        STACK_TYPE_F32 => {
                            emit::F32Store(&mut code, val_reg, ptr_reg, *offset as u32)
                        }
                        STACK_TYPE_F64 => {
                            emit::F64Store(&mut code, val_reg, ptr_reg, *offset as u32)
                        }
                        STACK_TYPE_I8 => emit::I8Store(&mut code, val_reg, ptr_reg, *offset as u32),
                        STACK_TYPE_I16 => {
                            emit::I16Store(&mut code, val_reg, ptr_reg, *offset as u32)
                        }
                        _ => panic!("Unsupported store type {:?}", ty),
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
                    let from_ty = func.dfg.values[*arg].ty.clone();
                    match opcode {
                        IrOpcode::ExtendS => {
                            let ty_val = if from_ty == Type::I8 {
                                EXTEND_TYPE_I8
                            } else if from_ty == Type::I16 {
                                EXTEND_TYPE_I16
                            } else if from_ty == Type::I32 {
                                EXTEND_TYPE_I32
                            } else {
                                panic!("Unsupported extend from type {:?}", from_ty);
                            };
                            emit::ExtendS(&mut code, dst, arg_reg, ty_val as u16);
                        }
                        IrOpcode::ExtendU => {
                            let ty_val = if from_ty == Type::I8 {
                                EXTEND_TYPE_I8
                            } else if from_ty == Type::I16 {
                                EXTEND_TYPE_I16
                            } else if from_ty == Type::I32 {
                                EXTEND_TYPE_I32
                            } else {
                                panic!("Unsupported extend from type {:?}", from_ty);
                            };
                            emit::ExtendU(&mut code, dst, arg_reg, ty_val as u16);
                        }
                        IrOpcode::Wrap => {
                            emit::Wrap(&mut code, dst, arg_reg);
                        }
                        IrOpcode::FloatToIntS => {
                            let to_ty = func.dfg.values[res_vals[0]].ty.clone();
                            if to_ty == Type::I32 && from_ty == Type::F32 {
                                emit::I32TruncF32S(&mut code, dst, arg_reg);
                            } else if to_ty == Type::I32 && from_ty == Type::F64 {
                                emit::I32TruncF64S(&mut code, dst, arg_reg);
                            } else if to_ty == Type::I64 && from_ty == Type::F32 {
                                emit::I64TruncF32S(&mut code, dst, arg_reg);
                            } else if to_ty == Type::I64 && from_ty == Type::F64 {
                                emit::I64TruncF64S(&mut code, dst, arg_reg);
                            } else {
                                panic!(
                                    "Unsupported TruncS: {:?} -> {:?}",
                                    from_ty.clone(),
                                    to_ty.clone()
                                );
                            }
                        }
                        IrOpcode::FloatToIntU => {
                            let to_ty = func.dfg.values[res_vals[0]].ty.clone();
                            if to_ty == Type::I32 && from_ty == Type::F32 {
                                emit::I32TruncF32U(&mut code, dst, arg_reg);
                            } else if to_ty == Type::I32 && from_ty == Type::F64 {
                                emit::I32TruncF64U(&mut code, dst, arg_reg);
                            } else if to_ty == Type::I64 && from_ty == Type::F32 {
                                emit::I64TruncF32U(&mut code, dst, arg_reg);
                            } else if to_ty == Type::I64 && from_ty == Type::F64 {
                                emit::I64TruncF64U(&mut code, dst, arg_reg);
                            } else {
                                panic!(
                                    "Unsupported TruncU: {:?} -> {:?}",
                                    from_ty.clone(),
                                    to_ty.clone()
                                );
                            }
                        }
                        IrOpcode::FloatToIntSatS => {
                            let to_ty = func.dfg.values[res_vals[0]].ty.clone();
                            if to_ty == Type::I32 && from_ty == Type::F32 {
                                emit::I32TruncSatF32S(&mut code, dst, arg_reg);
                            } else if to_ty == Type::I32 && from_ty == Type::F64 {
                                emit::I32TruncSatF64S(&mut code, dst, arg_reg);
                            } else if to_ty == Type::I64 && from_ty == Type::F32 {
                                emit::I64TruncSatF32S(&mut code, dst, arg_reg);
                            } else if to_ty == Type::I64 && from_ty == Type::F64 {
                                emit::I64TruncSatF64S(&mut code, dst, arg_reg);
                            } else {
                                panic!(
                                    "Unsupported TruncSatS: {:?} -> {:?}",
                                    from_ty.clone(),
                                    to_ty.clone()
                                );
                            }
                        }
                        IrOpcode::FloatToIntSatU => {
                            let to_ty = func.dfg.values[res_vals[0]].ty.clone();
                            if to_ty == Type::I32 && from_ty == Type::F32 {
                                emit::I32TruncSatF32U(&mut code, dst, arg_reg);
                            } else if to_ty == Type::I32 && from_ty == Type::F64 {
                                emit::I32TruncSatF64U(&mut code, dst, arg_reg);
                            } else if to_ty == Type::I64 && from_ty == Type::F32 {
                                emit::I64TruncSatF32U(&mut code, dst, arg_reg);
                            } else if to_ty == Type::I64 && from_ty == Type::F64 {
                                emit::I64TruncSatF64U(&mut code, dst, arg_reg);
                            } else {
                                panic!(
                                    "Unsupported TruncSatU: {:?} -> {:?}",
                                    from_ty.clone(),
                                    to_ty.clone()
                                );
                            }
                        }
                        IrOpcode::IntToFloatS => {
                            let to_ty = func.dfg.values[res_vals[0]].ty.clone();
                            if to_ty == Type::F32 && from_ty == Type::I32 {
                                emit::F32ConvertI32S(&mut code, dst, arg_reg);
                            } else if to_ty == Type::F32 && from_ty == Type::I64 {
                                emit::F32ConvertI64S(&mut code, dst, arg_reg);
                            } else if to_ty == Type::F64 && from_ty == Type::I32 {
                                emit::F64ConvertI32S(&mut code, dst, arg_reg);
                            } else if to_ty == Type::F64 && from_ty == Type::I64 {
                                emit::F64ConvertI64S(&mut code, dst, arg_reg);
                            } else {
                                panic!(
                                    "Unsupported ConvertS: {:?} -> {:?}",
                                    from_ty.clone(),
                                    to_ty.clone()
                                );
                            }
                        }
                        IrOpcode::IntToFloatU => {
                            let to_ty = func.dfg.values[res_vals[0]].ty.clone();
                            if to_ty == Type::F32 && from_ty == Type::I32 {
                                emit::F32ConvertI32U(&mut code, dst, arg_reg);
                            } else if to_ty == Type::F32 && from_ty == Type::I64 {
                                emit::F32ConvertI64U(&mut code, dst, arg_reg);
                            } else if to_ty == Type::F64 && from_ty == Type::I32 {
                                emit::F64ConvertI32U(&mut code, dst, arg_reg);
                            } else if to_ty == Type::F64 && from_ty == Type::I64 {
                                emit::F64ConvertI64U(&mut code, dst, arg_reg);
                            } else {
                                panic!(
                                    "Unsupported ConvertU: {:?} -> {:?}",
                                    from_ty.clone(),
                                    to_ty.clone()
                                );
                            }
                        }
                        IrOpcode::FloatDemote => emit::F32DemoteF64(&mut code, dst, arg_reg),
                        IrOpcode::FloatPromote => emit::F64PromoteF32(&mut code, dst, arg_reg),
                        IrOpcode::Reinterpret => {
                            emit::RegMove(&mut code, dst, arg_reg);
                        }
                        IrOpcode::FAbs => {
                            if from_ty == Type::F32 {
                                emit::F32Abs(&mut code, dst, arg_reg);
                            } else if from_ty == Type::F64 {
                                emit::F64Abs(&mut code, dst, arg_reg);
                            } else {
                                panic!("Unsupported Abs for type {:?}", from_ty);
                            }
                        }
                        IrOpcode::FNeg => {
                            if from_ty == Type::F32 {
                                emit::F32Neg(&mut code, dst, arg_reg);
                            } else if from_ty == Type::F64 {
                                emit::F64Neg(&mut code, dst, arg_reg);
                            } else {
                                panic!("Unsupported Fneg for type {:?}", from_ty);
                            }
                        }
                        IrOpcode::INeg => {
                            // ... implement if needed
                            todo!("Ineg not implemented");
                        }
                        IrOpcode::FSqrt => {
                            if from_ty == Type::F32 {
                                emit::F32Sqrt(&mut code, dst, arg_reg);
                            } else if from_ty == Type::F64 {
                                emit::F64Sqrt(&mut code, dst, arg_reg);
                            } else {
                                panic!("Unsupported Sqrt for type {:?}", from_ty);
                            }
                        }
                        IrOpcode::FCeil => {
                            if from_ty == Type::F32 {
                                emit::F32Ceil(&mut code, dst, arg_reg);
                            } else if from_ty == Type::F64 {
                                emit::F64Ceil(&mut code, dst, arg_reg);
                            } else {
                                panic!("Unsupported Ceil for type {:?}", from_ty);
                            }
                        }
                        IrOpcode::FFloor => {
                            if from_ty == Type::F32 {
                                emit::F32Floor(&mut code, dst, arg_reg);
                            } else if from_ty == Type::F64 {
                                emit::F64Floor(&mut code, dst, arg_reg);
                            } else {
                                panic!("Unsupported Floor for type {:?}", from_ty);
                            }
                        }
                        IrOpcode::FTrunc => {
                            if from_ty == Type::F32 {
                                emit::F32Trunc(&mut code, dst, arg_reg);
                            } else if from_ty == Type::F64 {
                                emit::F64Trunc(&mut code, dst, arg_reg);
                            } else {
                                panic!("Unsupported Trunc for type {:?}", from_ty);
                            }
                        }
                        IrOpcode::FNearest => {
                            if from_ty == Type::F32 {
                                emit::F32Nearest(&mut code, dst, arg_reg);
                            } else if from_ty == Type::F64 {
                                emit::F64Nearest(&mut code, dst, arg_reg);
                            } else {
                                panic!("Unsupported Nearest for type {:?}", from_ty);
                            }
                        }
                        IrOpcode::IClz => {
                            if from_ty == Type::I32 {
                                emit::I32Clz(&mut code, dst, arg_reg);
                            } else if from_ty == Type::I64 {
                                emit::I64Clz(&mut code, dst, arg_reg);
                            } else {
                                panic!("Unsupported Clz for type {:?}", from_ty);
                            }
                        }
                        IrOpcode::ICtz => {
                            if from_ty == Type::I32 {
                                emit::I32Ctz(&mut code, dst, arg_reg);
                            } else if from_ty == Type::I64 {
                                emit::I64Ctz(&mut code, dst, arg_reg);
                            } else {
                                panic!("Unsupported Ctz for type {:?}", from_ty);
                            }
                        }
                        IrOpcode::IPopcnt => {
                            if from_ty == Type::I32 {
                                emit::I32Popcnt(&mut code, dst, arg_reg);
                            } else if from_ty == Type::I64 {
                                emit::I64Popcnt(&mut code, dst, arg_reg);
                            } else {
                                panic!("Unsupported Popcnt for type {:?}", from_ty);
                            }
                        }
                        IrOpcode::IEqz => {
                            if from_ty == Type::I32 {
                                emit::I32Eqz(&mut code, dst, arg_reg);
                            } else if from_ty == Type::I64 {
                                emit::I64Eqz(&mut code, dst, arg_reg);
                            } else {
                                panic!("Unsupported Eqz for type {:?}", from_ty);
                            }
                        }
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
