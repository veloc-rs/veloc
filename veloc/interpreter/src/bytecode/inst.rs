use veloc_ir::ScalarType;

/// Represents a fixed-size bytecode instruction (16 bytes)
/// Layout: opcode(1) + pad(1) + dst(2) + src1(2) + src2(2) + imm64(8) = 16 bytes
#[derive(Clone, Copy, Debug)]
pub(crate) struct Instruction {
    /// Opcode
    pub opcode: Opcode,
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

    /// Get conversion types (from_type, to_type)
    #[inline(always)]
    pub fn conv_types(&self) -> (ScalarType, ScalarType) {
        unsafe {
            (
                core::mem::transmute::<u8, ScalarType>((self.src2 & 0xFF) as u8),
                core::mem::transmute::<u8, ScalarType>((self.src2 >> 8) as u8),
            )
        }
    }

    /// Get stack type for stack operations
    #[inline(always)]
    pub fn stack_type(&self) -> ScalarType {
        unsafe { core::mem::transmute::<u8, ScalarType>(self.src2 as u8) }
    }

    /// Get jump target based on condition
    #[inline(always)]
    pub const fn br_target(&self, cond: bool) -> u32 {
        if cond { self.imm32() } else { self.aux() }
    }

    /// Get number of targets in BrTable
    #[inline(always)]
    pub const fn br_table_num_targets(&self) -> u32 {
        self.aux()
    }

    /// Get base index in jump_targets for BrTable
    #[inline(always)]
    pub const fn br_table_base_idx(&self) -> u32 {
        self.imm32()
    }

    /// Get scale for PtrIndex
    #[inline(always)]
    pub const fn ptr_index_scale(&self) -> i64 {
        self.imm32() as i64
    }

    /// Get offset for PtrIndex
    #[inline(always)]
    pub const fn ptr_index_offset(&self) -> i64 {
        self.aux() as i32 as i64
    }

    /// Get false register for Select
    #[inline(always)]
    pub const fn select_false_reg(&self) -> u16 {
        (self.imm32() & 0xFFFF) as u16
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
        pub(crate) enum Opcode {
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
                        opcode: Opcode::$name,
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
    Wrap { dst: u16, src: u16 => src1, ty: u16 => src2 };

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
