use veloc_ir::ScalarType;

/// Register index wrapper for type-safe printing
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub(crate) struct Reg(pub u16);

impl Reg {
    /// Sentinel "no register" value (register slot is unused / unallocated).
    pub(crate) const NULL: Reg = Reg(0);
}

impl core::fmt::Display for Reg {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.0 == 0 {
            write!(f, "_")
        } else {
            write!(f, "r{}", self.0)
        }
    }
}

impl From<Reg> for u16 {
    fn from(reg: Reg) -> u16 {
        reg.0
    }
}

impl Default for Reg {
    fn default() -> Self {
        Reg::NULL
    }
}

impl Reg {
    pub(crate) fn index(&self) -> u16 {
        self.0
    }
}

/// Decode a raw `u32` immediate-slot value into the logical field type.
/// Works symmetrically with [`FromRawReg`] but for immediate/offset slots.
pub(crate) trait FromRawImm: Sized {
    fn from_raw_u32(v: u32) -> Self;
}

impl FromRawImm for u32 {
    #[inline(always)]
    fn from_raw_u32(v: u32) -> Self {
        v
    }
}

impl FromRawImm for u16 {
    #[inline(always)]
    fn from_raw_u32(v: u32) -> Self {
        v as u16
    }
}

impl FromRawImm for Reg {
    #[inline(always)]
    fn from_raw_u32(v: u32) -> Self {
        Reg(v as u16)
    }
}

/// Convert a logical field value into a raw `u32` for storage in an immediate slot.
pub(crate) trait IntoRawImm {
    fn into_raw_imm(self) -> u32;
}

impl IntoRawImm for u32 {
    #[inline(always)]
    fn into_raw_imm(self) -> u32 {
        self
    }
}

impl IntoRawImm for u16 {
    #[inline(always)]
    fn into_raw_imm(self) -> u32 {
        self as u32
    }
}

impl IntoRawImm for Reg {
    #[inline(always)]
    fn into_raw_imm(self) -> u32 {
        self.0 as u32
    }
}

/// Decode a raw `u16` register-slot value into the logical field type.
/// - `Reg`  → wraps the index in `Reg(...)`
/// - `u16`  → returns the raw value as-is
pub(crate) trait FromRawReg: Sized {
    fn from_raw_reg(v: u16) -> Self;
}

impl FromRawReg for Reg {
    #[inline(always)]
    fn from_raw_reg(v: u16) -> Self {
        Reg(v)
    }
}

impl FromRawReg for u16 {
    #[inline(always)]
    fn from_raw_reg(v: u16) -> Self {
        v
    }
}

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

        /// Decoded view of a bytecode instruction with logical field names.
        ///
        /// Obtained via [`Instruction::decode`]. Each variant mirrors the argument
        /// list of the corresponding `define_opcodes!` entry, so you get names like
        /// `ptr`, `offset`, `val` instead of raw `src1`, `imm32()`, etc.
        #[derive(Debug, Clone, Copy)]
        #[allow(dead_code)]
        pub(crate) enum DecodedInstruction {
            $(
                $name { $($arg: $ty),* }
            ),*
        }

        impl Instruction {
            /// Decode this instruction into a [`DecodedInstruction`] with logical
            /// field names. Always inlined — zero run-time overhead.
            #[inline(always)]
            pub(crate) fn decode(self) -> DecodedInstruction {
                let inst = self;
                match inst.opcode {
                    $(
                        Opcode::$name => DecodedInstruction::$name {
                            $(
                                $arg: define_opcodes!(@decode_field inst, $arg, $ty, $($field)?)
                            ),*
                        }
                    ),*
                }
            }
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
                        define_opcodes!(@assign inst, $arg, $ty, $( $field )? );
                    )*

                    code.push(inst);
                }
            )*
        }
    };

    // --- Decode helpers: reverse-map raw fields back to logical arg names ---
    // Register slots: FromRawReg dispatches on $ty — Reg-typed fields become Reg(...),
    // non-register types (e.g. u16) receive the raw value directly. No per-name special cases needed.
    (@decode_field $inst:ident, $arg:ident, $ty:ty, dst)   => { <$ty as FromRawReg>::from_raw_reg($inst.dst) };
    (@decode_field $inst:ident, $arg:ident, $ty:ty, src1)  => { <$ty as FromRawReg>::from_raw_reg($inst.src1) };
    (@decode_field $inst:ident, $arg:ident, $ty:ty, src2)  => { <$ty as FromRawReg>::from_raw_reg($inst.src2) };
    // Other fields use direct cast with trait dispatch
    (@decode_field $inst:ident, $arg:ident, $ty:ty, imm32) => { <$ty as FromRawImm>::from_raw_u32($inst.imm32()) };
    (@decode_field $inst:ident, $arg:ident, $ty:ty, aux)   => { <$ty as FromRawImm>::from_raw_u32($inst.aux()) };
    (@decode_field $inst:ident, $arg:ident, $ty:ty, imm64) => { $inst.imm64 as $ty };
    // No explicit mapping: the arg name itself is the raw field name.
    (@decode_field $inst:ident, $arg:ident, $ty:ty, ) => {
        define_opcodes!(@decode_field $inst, $arg, $ty, $arg)
    };

    // --- Assign helpers (emit side) ---
    // Register fields (dst/src1/src2): use Into<u16> to handle both Reg and u16
    // Reg implements Into<u16> via the From impl, u16 has it built-in
    (@assign $inst:ident, $val:ident, $ty:ty, dst) => { $inst.dst = $val.into(); };
    (@assign $inst:ident, $val:ident, $ty:ty, src1) => { $inst.src1 = $val.into(); };
    (@assign $inst:ident, $val:ident, $ty:ty, src2) => { $inst.src2 = $val.into(); };
    // Immediate slots: use IntoRawImm for type-safe conversion
    (@assign $inst:ident, $val:ident, $ty:ty, imm32) => { $inst.imm64 = ($inst.imm64 & 0xFFFFFFFF00000000) | ($val.into_raw_imm() as u64); };
    (@assign $inst:ident, $val:ident, $ty:ty, aux)   => { $inst.imm64 = ($inst.imm64 & 0x00000000FFFFFFFF) | (($val.into_raw_imm() as u64) << 32); };
    (@assign $inst:ident, $val:ident, $ty:ty, imm64) => { $inst.imm64 = $val; };
    // If no field mapped, assume it's one of the standard ones by name (treat as register)
    (@assign $inst:ident, $val:ident, $ty:ty, ) => {
        $inst.$val = $val.into();
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
    Iconst { dst: Reg, imm64: u64 };
    Fconst { dst: Reg, imm64: u64 };
    Bconst { dst: Reg, val: u16 => src2 };
    Vconst { dst: Reg, pool_id: u32 => imm32 };

    // === I32 Arithmetic ===
    I32Add { dst: Reg, src1: Reg, src2: Reg };
    I32Sub { dst: Reg, src1: Reg, src2: Reg };
    I32Mul { dst: Reg, src1: Reg, src2: Reg };
    I32DivS { dst: Reg, src1: Reg, src2: Reg };
    I32DivU { dst: Reg, src1: Reg, src2: Reg };
    I32RemS { dst: Reg, src1: Reg, src2: Reg };
    I32RemU { dst: Reg, src1: Reg, src2: Reg };
    I32And { dst: Reg, src1: Reg, src2: Reg };
    I32Or { dst: Reg, src1: Reg, src2: Reg };
    I32Xor { dst: Reg, src1: Reg, src2: Reg };
    I32Shl { dst: Reg, src1: Reg, src2: Reg };
    I32ShrS { dst: Reg, src1: Reg, src2: Reg };
    I32ShrU { dst: Reg, src1: Reg, src2: Reg };
    I32RotL { dst: Reg, src1: Reg, src2: Reg };
    I32RotR { dst: Reg, src1: Reg, src2: Reg };

    I32AddImm { dst: Reg, src1: Reg, imm: u32 => imm32 };
    I32SubImm { dst: Reg, src1: Reg, imm: u32 => imm32 };
    I32AndImm { dst: Reg, src1: Reg, imm: u32 => imm32 };
    I32OrImm { dst: Reg, src1: Reg, imm: u32 => imm32 };
    I32XorImm { dst: Reg, src1: Reg, imm: u32 => imm32 };
    I32ShlImm { dst: Reg, src1: Reg, imm: u32 => imm32 };
    I32ShrSImm { dst: Reg, src1: Reg, imm: u32 => imm32 };
    I32ShrUImm { dst: Reg, src1: Reg, imm: u32 => imm32 };

    // === I64 Arithmetic ===
    I64Add { dst: Reg, src1: Reg, src2: Reg };
    I64Sub { dst: Reg, src1: Reg, src2: Reg };
    I64Mul { dst: Reg, src1: Reg, src2: Reg };
    I64DivS { dst: Reg, src1: Reg, src2: Reg };
    I64DivU { dst: Reg, src1: Reg, src2: Reg };
    I64RemS { dst: Reg, src1: Reg, src2: Reg };
    I64RemU { dst: Reg, src1: Reg, src2: Reg };
    I64And { dst: Reg, src1: Reg, src2: Reg };
    I64Or { dst: Reg, src1: Reg, src2: Reg };
    I64Xor { dst: Reg, src1: Reg, src2: Reg };
    I64Shl { dst: Reg, src1: Reg, src2: Reg };
    I64ShrS { dst: Reg, src1: Reg, src2: Reg };
    I64ShrU { dst: Reg, src1: Reg, src2: Reg };
    I64RotL { dst: Reg, src1: Reg, src2: Reg };
    I64RotR { dst: Reg, src1: Reg, src2: Reg };

    I64AddImm { dst: Reg, src1: Reg, imm64: u64 };
    I64SubImm { dst: Reg, src1: Reg, imm64: u64 };
    I64AndImm { dst: Reg, src1: Reg, imm64: u64 };
    I64OrImm { dst: Reg, src1: Reg, imm64: u64 };
    I64XorImm { dst: Reg, src1: Reg, imm64: u64 };
    I64ShlImm { dst: Reg, src1: Reg, imm64: u64 };
    I64ShrSImm { dst: Reg, src1: Reg, imm64: u64 };
    I64ShrUImm { dst: Reg, src1: Reg, imm64: u64 };

    // === Comparisons ===
    I32Eq { dst: Reg, src1: Reg, src2: Reg };
    I32Ne { dst: Reg, src1: Reg, src2: Reg };
    I32LtS { dst: Reg, src1: Reg, src2: Reg };
    I32LtU { dst: Reg, src1: Reg, src2: Reg };
    I32LeS { dst: Reg, src1: Reg, src2: Reg };
    I32LeU { dst: Reg, src1: Reg, src2: Reg };
    I32GtS { dst: Reg, src1: Reg, src2: Reg };
    I32GtU { dst: Reg, src1: Reg, src2: Reg };
    I32GeS { dst: Reg, src1: Reg, src2: Reg };
    I32GeU { dst: Reg, src1: Reg, src2: Reg };

    I64Eq { dst: Reg, src1: Reg, src2: Reg };
    I64Ne { dst: Reg, src1: Reg, src2: Reg };
    I64LtS { dst: Reg, src1: Reg, src2: Reg };
    I64LtU { dst: Reg, src1: Reg, src2: Reg };
    I64LeS { dst: Reg, src1: Reg, src2: Reg };
    I64LeU { dst: Reg, src1: Reg, src2: Reg };
    I64GtS { dst: Reg, src1: Reg, src2: Reg };
    I64GtU { dst: Reg, src1: Reg, src2: Reg };
    I64GeS { dst: Reg, src1: Reg, src2: Reg };
    I64GeU { dst: Reg, src1: Reg, src2: Reg };

    // === Float Arithmetic ===
    F32Add { dst: Reg, src1: Reg, src2: Reg };
    F32Sub { dst: Reg, src1: Reg, src2: Reg };
    F32Mul { dst: Reg, src1: Reg, src2: Reg };
    F32Div { dst: Reg, src1: Reg, src2: Reg };
    F32Neg { dst: Reg, src1: Reg };
    F32Abs { dst: Reg, src1: Reg };
    F32Sqrt { dst: Reg, src1: Reg };
    F32Ceil { dst: Reg, src1: Reg };
    F32Floor { dst: Reg, src1: Reg };
    F32Trunc { dst: Reg, src1: Reg };
    F32Nearest { dst: Reg, src1: Reg };
    F32Min { dst: Reg, src1: Reg, src2: Reg };
    F32Max { dst: Reg, src1: Reg, src2: Reg };
    F32CopySign { dst: Reg, src1: Reg, src2: Reg };

    F64Add { dst: Reg, src1: Reg, src2: Reg };
    F64Sub { dst: Reg, src1: Reg, src2: Reg };
    F64Mul { dst: Reg, src1: Reg, src2: Reg };
    F64Div { dst: Reg, src1: Reg, src2: Reg };
    F64Neg { dst: Reg, src1: Reg };
    F64Abs { dst: Reg, src1: Reg };
    F64Sqrt { dst: Reg, src1: Reg };
    F64Ceil { dst: Reg, src1: Reg };
    F64Floor { dst: Reg, src1: Reg };
    F64Trunc { dst: Reg, src1: Reg };
    F64Nearest { dst: Reg, src1: Reg };
    F64Min { dst: Reg, src1: Reg, src2: Reg };
    F64Max { dst: Reg, src1: Reg, src2: Reg };
    F64CopySign { dst: Reg, src1: Reg, src2: Reg };

    F32Eq { dst: Reg, src1: Reg, src2: Reg };
    F32Ne { dst: Reg, src1: Reg, src2: Reg };
    F32Lt { dst: Reg, src1: Reg, src2: Reg };
    F32Le { dst: Reg, src1: Reg, src2: Reg };
    F32Gt { dst: Reg, src1: Reg, src2: Reg };
    F32Ge { dst: Reg, src1: Reg, src2: Reg };
    F64Eq { dst: Reg, src1: Reg, src2: Reg };
    F64Ne { dst: Reg, src1: Reg, src2: Reg };
    F64Lt { dst: Reg, src1: Reg, src2: Reg };
    F64Le { dst: Reg, src1: Reg, src2: Reg };
    F64Gt { dst: Reg, src1: Reg, src2: Reg };
    F64Ge { dst: Reg, src1: Reg, src2: Reg };

    // === Memory ===
    I32Load { dst: Reg, ptr: Reg => src1, offset: u32 => imm32 };
    I64Load { dst: Reg, ptr: Reg => src1, offset: u32 => imm32 };
    F32Load { dst: Reg, ptr: Reg => src1, offset: u32 => imm32 };
    F64Load { dst: Reg, ptr: Reg => src1, offset: u32 => imm32 };
    I8Load { dst: Reg, ptr: Reg => src1, offset: u32 => imm32 };
    I16Load { dst: Reg, ptr: Reg => src1, offset: u32 => imm32 };

    I32Store { val: Reg => src1, ptr: Reg => src2, offset: u32 => imm32 };
    I64Store { val: Reg => src1, ptr: Reg => src2, offset: u32 => imm32 };
    F32Store { val: Reg => src1, ptr: Reg => src2, offset: u32 => imm32 };
    F64Store { val: Reg => src1, ptr: Reg => src2, offset: u32 => imm32 };
    I8Store { val: Reg => src1, ptr: Reg => src2, offset: u32 => imm32 };
    I16Store { val: Reg => src1, ptr: Reg => src2, offset: u32 => imm32 };

    // === Conversions ===
    ExtendS { dst: Reg, src: Reg => src1, ty: u16 => src2 };
    ExtendU { dst: Reg, src: Reg => src1, ty: u16 => src2 };
    Wrap { dst: Reg, src: Reg => src1, ty: u16 => src2 };

    I32TruncF32S { dst: Reg, src: Reg => src1 };
    I32TruncF32U { dst: Reg, src: Reg => src1 };
    I32TruncF64S { dst: Reg, src: Reg => src1 };
    I32TruncF64U { dst: Reg, src: Reg => src1 };
    I64TruncF32S { dst: Reg, src: Reg => src1 };
    I64TruncF32U { dst: Reg, src: Reg => src1 };
    I64TruncF64S { dst: Reg, src: Reg => src1 };
    I64TruncF64U { dst: Reg, src: Reg => src1 };
    I32TruncSatF32S { dst: Reg, src: Reg => src1 };
    I32TruncSatF32U { dst: Reg, src: Reg => src1 };
    I32TruncSatF64S { dst: Reg, src: Reg => src1 };
    I32TruncSatF64U { dst: Reg, src: Reg => src1 };
    I64TruncSatF32S { dst: Reg, src: Reg => src1 };
    I64TruncSatF32U { dst: Reg, src: Reg => src1 };
    I64TruncSatF64S { dst: Reg, src: Reg => src1 };
    I64TruncSatF64U { dst: Reg, src: Reg => src1 };

    F32ConvertI32S { dst: Reg, src: Reg => src1 };
    F32ConvertI32U { dst: Reg, src: Reg => src1 };
    F32ConvertI64S { dst: Reg, src: Reg => src1 };
    F32ConvertI64U { dst: Reg, src: Reg => src1 };
    F64ConvertI32S { dst: Reg, src: Reg => src1 };
    F64ConvertI32U { dst: Reg, src: Reg => src1 };
    F64ConvertI64S { dst: Reg, src: Reg => src1 };
    F64ConvertI64U { dst: Reg, src: Reg => src1 };
    F32DemoteF64 { dst: Reg, src: Reg => src1 };
    F64PromoteF32 { dst: Reg, src: Reg => src1 };
    Bitcast { dst: Reg, src: Reg => src1 };

    // === Bitwise ===
    I32Clz { dst: Reg, src: Reg => src1 };
    I32Ctz { dst: Reg, src: Reg => src1 };
    I32Popcnt { dst: Reg, src: Reg => src1 };
    I64Clz { dst: Reg, src: Reg => src1 };
    I64Ctz { dst: Reg, src: Reg => src1 };
    I64Popcnt { dst: Reg, src: Reg => src1 };
    I32Eqz { dst: Reg, src_val: Reg => src1 };
    I64Eqz { dst: Reg, src_val: Reg => src1 };

    // === Stack ===
    StackAddr { dst: Reg, offset: u32 => imm32 };
    StackLoad { dst: Reg, ty: u16 => src2, offset: u32 => imm32 };
    StackStore { val: Reg => src1, ty: u16 => src2, offset: u32 => imm32 };

    PtrIndex { dst: Reg, ptr: Reg => src1, index: Reg => src2, scale: u32 => imm32, offset: u32 => aux };

    // === Control Flow ===
    Jump { pc: u32 => imm32 };
    JumpWithMoves { data_offset: u32 => imm32 };
    Br { cond: Reg => dst, then_idx: u32 => imm32, else_idx: u32 => aux };
    BrTable { idx_reg: Reg => dst, data_offset: u32 => imm32, num_targets: u32 => aux };

    Select { dst: Reg, cond: Reg => src1, then_reg: Reg => src2, else_reg: Reg => imm32 };
    Return { data_offset: u32 => imm32, num_vals: u32 => aux };

    Call { func_id: u32 => imm32, data_offset: u32 => aux };
    CallIndirect { ptr: Reg => src1, data_offset: u32 => imm32, counts: u32 => aux };
    CallIntrinsic { intrinsic: u16 => src1, data_offset: u32 => imm32, counts: u32 => aux };

    GlobalAddr { dst: Reg, global_idx: u32 => imm32 };

    RegMove { dst: Reg, src: Reg => src1 };
    Unreachable {};
}
