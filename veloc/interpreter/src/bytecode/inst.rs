use veloc_ir::ScalarType;

/// Packed type pair for Extend/Convert operations: (to_ty, from_ty)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TypePair {
    pub from: ScalarType,
    pub to: ScalarType,
}

impl TypePair {
    /// Pack two types into u16: (to_ty << 8) | from_ty
    #[inline(always)]
    pub const fn pack(from: ScalarType, to: ScalarType) -> u16 {
        ((to as u16) << 8) | (from as u16)
    }

    /// Unpack u16 into TypePair
    #[inline(always)]
    pub const fn unpack(raw: u16) -> Self {
        Self {
            from: unsafe { core::mem::transmute((raw & 0xFF) as u8) },
            to: unsafe { core::mem::transmute((raw >> 8) as u8) },
        }
    }
}

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
    #[inline(always)]
    fn from(reg: Reg) -> u16 {
        reg.0
    }
}

impl Default for Reg {
    #[inline(always)]
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

impl FromRawImm for i32 {
    #[inline(always)]
    fn from_raw_u32(v: u32) -> Self {
        v as i32
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

impl IntoRawImm for i32 {
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

/// Convert a logical field value into a raw `u16` for storage in a register slot.
pub(crate) trait IntoRawReg {
    fn into_raw_reg(self) -> u16;
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

impl FromRawReg for bool {
    #[inline(always)]
    fn from_raw_reg(v: u16) -> Self {
        v != 0
    }
}

impl IntoRawReg for Reg {
    #[inline(always)]
    fn into_raw_reg(self) -> u16 {
        self.0
    }
}

impl IntoRawReg for u16 {
    #[inline(always)]
    fn into_raw_reg(self) -> u16 {
        self
    }
}

impl IntoRawReg for bool {
    #[inline(always)]
    fn into_raw_reg(self) -> u16 {
        self as u16
    }
}

impl FromRawReg for TypePair {
    #[inline(always)]
    fn from_raw_reg(v: u16) -> Self {
        TypePair::unpack(v)
    }
}

impl IntoRawReg for TypePair {
    #[inline(always)]
    fn into_raw_reg(self) -> u16 {
        TypePair::pack(self.from, self.to)
    }
}

impl FromRawReg for ScalarType {
    #[inline(always)]
    fn from_raw_reg(v: u16) -> Self {
        unsafe { core::mem::transmute(v as u8) }
    }
}

impl IntoRawReg for ScalarType {
    #[inline(always)]
    fn into_raw_reg(self) -> u16 {
        self as u16
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct HeaderWord {
    opcode: u8,
    reserved: u8,
    dst: u16,
    src1: u16,
    src2: u16,
}

/// One aligned word in the executable bytecode stream.
///
/// A word is either an instruction header or the payload belonging to the
/// preceding header. Register-only instructions contain just a header; an
/// instruction with an immediate appends one payload word.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) union CodeWord {
    header: HeaderWord,
    raw: u64,
}

impl core::fmt::Debug for CodeWord {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // SAFETY: both union variants occupy the same fully initialized bytes.
        f.debug_tuple("CodeWord")
            .field(&unsafe { self.raw })
            .finish()
    }
}

const _: () = {
    assert!(core::mem::size_of::<HeaderWord>() == 8);
    assert!(core::mem::offset_of!(HeaderWord, opcode) == 0);
    assert!(core::mem::offset_of!(HeaderWord, reserved) == 1);
    assert!(core::mem::offset_of!(HeaderWord, dst) == 2);
    assert!(core::mem::offset_of!(HeaderWord, src1) == 4);
    assert!(core::mem::offset_of!(HeaderWord, src2) == 6);
    assert!(core::mem::size_of::<CodeWord>() == 8);
    assert!(core::mem::align_of::<CodeWord>() == 8);
};

impl CodeWord {
    #[inline(always)]
    const fn header(opcode: Opcode, dst: u16, src1: u16, src2: u16) -> Self {
        Self {
            header: HeaderWord {
                opcode: opcode as u8,
                reserved: 0,
                dst,
                src1,
                src2,
            },
        }
    }

    #[inline(always)]
    pub(crate) const fn from_payload(payload: u64) -> Self {
        Self { raw: payload }
    }

    #[inline(always)]
    pub(crate) unsafe fn opcode(self) -> Opcode {
        // SAFETY: every header initializes all eight bytes with integer fields.
        let raw = unsafe { self.header.opcode };
        debug_assert!((raw as usize) < Opcode::COUNT);
        // SAFETY: executable instruction headers are only constructed from an
        // Opcode. Payload words are never passed to dispatch as headers.
        unsafe { core::mem::transmute(raw) }
    }

    #[inline(always)]
    unsafe fn read_dst(ip: *const CodeWord) -> u16 {
        // SAFETY: `ip` points to an instruction header.
        unsafe { (*ip).header.dst }
    }

    #[inline(always)]
    unsafe fn read_src1(ip: *const CodeWord) -> u16 {
        // SAFETY: `ip` points to an instruction header.
        unsafe { (*ip).header.src1 }
    }

    #[inline(always)]
    unsafe fn read_src2(ip: *const CodeWord) -> u16 {
        // SAFETY: `ip` points to an instruction header.
        unsafe { (*ip).header.src2 }
    }

    #[inline(always)]
    unsafe fn payload(ip: *const CodeWord, opcode: Opcode) -> u64 {
        if opcode.has_payload() {
            // SAFETY: payload-bearing opcodes always emit a second word.
            unsafe { (*ip.add(1)).raw }
        } else {
            0
        }
    }

    #[inline(always)]
    fn set_dst(&mut self, value: u16) {
        self.header.dst = value;
    }

    #[inline(always)]
    fn set_src1(&mut self, value: u16) {
        self.header.src1 = value;
    }

    #[inline(always)]
    fn set_src2(&mut self, value: u16) {
        self.header.src2 = value;
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

        impl Opcode {
            /// Number of entries in the dense opcode space.
            pub(crate) const COUNT: usize = 0 $(+ { let _ = stringify!($name); 1 })*;

            /// Whether this opcode is followed by a 64-bit payload word.
            #[inline(always)]
            pub(crate) const fn has_payload(self) -> bool {
                match self {
                    $(
                        Opcode::$name => false $(
                            || define_opcodes!(@uses_payload $arg $(, $field)?)
                        )*
                    ),*
                }
            }

            /// Number of eight-byte code words occupied by this instruction.
            #[inline(always)]
            pub(crate) const fn words(self) -> usize {
                1 + self.has_payload() as usize
            }
        }

        /// Direct-threaded dispatch table. Keeping this generated beside
        /// `Opcode` guarantees that discriminants and handler slots cannot drift.
        #[repr(transparent)]
        pub(crate) struct OpcodeHandlers<M>(
            [crate::interpreter::OpcodeHandler<M>; Opcode::COUNT],
        );

        impl<M: crate::interpreter::VirtualMemory> OpcodeHandlers<M> {
            pub(crate) const TABLE: Self = Self([
                $(crate::interpreter::handlers::$name::<M>),*
            ]);

            #[inline(always)]
            pub(crate) unsafe fn get(
                &self,
                opcode: Opcode,
            ) -> crate::interpreter::OpcodeHandler<M> {
                unsafe { *self.0.get_unchecked(opcode as usize) }
            }
        }

        /// Decoded view of a bytecode instruction with logical field names.
        /// Each variant mirrors the argument list of the corresponding
        /// `define_opcodes!` entry.
        #[derive(Debug, Clone, Copy)]
        #[allow(dead_code)]
        pub(crate) enum DecodedInstruction {
            $(
                $name { $($arg: $ty),* }
            ),*
        }

        /// Per-opcode typed decoders used by direct-threaded handlers. These
        /// functions read fields straight from the compact code words and contain
        /// no secondary opcode dispatch.
        #[allow(non_snake_case)]
        pub(crate) mod decode {
            use super::*;

            $(
                #[allow(unused_variables)]
                #[inline(always)]
                pub(crate) unsafe fn $name(ip: *const CodeWord) -> ($($ty,)*) {
                    // SAFETY: handlers call their decoder with an instruction header.
                    debug_assert_eq!(unsafe { (*ip).opcode() }, Opcode::$name);
                    // SAFETY: `ip` refers to the expected logical instruction.
                    let payload = unsafe { CodeWord::payload(ip, Opcode::$name) };
                    ($(
                        define_opcodes!(@decode_field ip, payload, $arg, $ty, $($field)?),
                    )*)
                }
            )*
        }

        impl DecodedInstruction {
            /// Decode the instruction at `ip` for diagnostics and tests.
            #[inline(always)]
            pub(crate) unsafe fn read(ip: *const CodeWord) -> Self {
                // SAFETY: callers pass a logical instruction header.
                match unsafe { (*ip).opcode() } {
                    $(
                        Opcode::$name => {
                            // SAFETY: opcode dispatch selected the matching decoder.
                            let ($($arg,)*) = unsafe { decode::$name(ip) };
                            DecodedInstruction::$name { $($arg),* }
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
                    code: &mut Vec<CodeWord>,
                    $($arg : $ty),*
                ) {
                    #[allow(unused_mut, unused_assignments)]
                    let mut header = CodeWord::header(Opcode::$name, 0, 0, 0);
                    #[allow(unused_mut, unused_assignments)]
                    let mut payload = 0u64;

                    $(
                        define_opcodes!(@assign header, payload, $arg, $ty, $( $field )? );
                    )*

                    code.push(header);
                    if Opcode::$name.has_payload() {
                        code.push(CodeWord::from_payload(payload));
                    }
                }
            )*
        }
    };

    // --- Decode helpers: reverse-map raw fields back to logical arg names ---
    // Register slots: FromRawReg dispatches on $ty — Reg-typed fields become Reg(...),
    // non-register types (e.g. u16) receive the raw value directly. No per-name special cases needed.
    (@decode_field $ip:ident, $payload:ident, $arg:ident, $ty:ty, dst)   => { <$ty as FromRawReg>::from_raw_reg(unsafe { CodeWord::read_dst($ip) }) };
    (@decode_field $ip:ident, $payload:ident, $arg:ident, $ty:ty, src1)  => { <$ty as FromRawReg>::from_raw_reg(unsafe { CodeWord::read_src1($ip) }) };
    (@decode_field $ip:ident, $payload:ident, $arg:ident, $ty:ty, src2)  => { <$ty as FromRawReg>::from_raw_reg(unsafe { CodeWord::read_src2($ip) }) };
    // Other fields use direct cast with trait dispatch
    (@decode_field $ip:ident, $payload:ident, $arg:ident, $ty:ty, lo32) => { <$ty as FromRawImm>::from_raw_u32($payload as u32) };
    (@decode_field $ip:ident, $payload:ident, $arg:ident, $ty:ty, hi32) => { <$ty as FromRawImm>::from_raw_u32(($payload >> 32) as u32) };
    (@decode_field $ip:ident, $payload:ident, $arg:ident, $ty:ty, imm64) => { $payload as $ty };
    // No explicit mapping: the arg name itself is the raw field name.
    (@decode_field $ip:ident, $payload:ident, $arg:ident, $ty:ty, ) => {
        define_opcodes!(@decode_field $ip, $payload, $arg, $ty, $arg)
    };

    // Determine whether a logical field lives in the optional payload word.
    (@uses_payload $arg:ident, lo32) => { true };
    (@uses_payload $arg:ident, hi32) => { true };
    (@uses_payload $arg:ident, imm64) => { true };
    (@uses_payload $arg:ident, $field:ident) => { false };
    (@uses_payload imm64) => { true };
    (@uses_payload $arg:ident) => { false };

    // --- Assign helpers (emit side) ---
    // Register fields (dst/src1/src2): use IntoRawReg for type-safe conversion
    (@assign $header:ident, $payload:ident, $val:ident, $ty:ty, dst) => { $header.set_dst($val.into_raw_reg()); };
    (@assign $header:ident, $payload:ident, $val:ident, $ty:ty, src1) => { $header.set_src1($val.into_raw_reg()); };
    (@assign $header:ident, $payload:ident, $val:ident, $ty:ty, src2) => { $header.set_src2($val.into_raw_reg()); };
    // Immediate slots: use IntoRawImm for type-safe conversion
    (@assign $header:ident, $payload:ident, $val:ident, $ty:ty, lo32) => { $payload = ($payload & 0xFFFFFFFF00000000) | ($val.into_raw_imm() as u64); };
    (@assign $header:ident, $payload:ident, $val:ident, $ty:ty, hi32) => { $payload = ($payload & 0x00000000FFFFFFFF) | (($val.into_raw_imm() as u64) << 32); };
    (@assign $header:ident, $payload:ident, $val:ident, $ty:ty, imm64) => { $payload = $val as u64; };
    // If no field mapped, assume it's one of the standard ones by name (treat as register)
    (@assign $header:ident, $payload:ident, $val:ident, $ty:ty, ) => {
        define_opcodes!(@assign $header, $payload, $val, $ty, $val)
    };
}

// Field mapping conventions:
// - dst: destination register (for 2-result ops, hi32 may hold 2nd dst)
// - src1: first source register
// - src2: second source register
// - lo32: low 32-bits of 64-bit immediate, also used for 32-bit immediates/offsets
// - hi32: high 32-bits of 64-bit immediate, also used for data section offsets or packed counts

define_opcodes! {
    // === Constants ===
    Iconst { dst: Reg, imm64: u64 };
    Fconst { dst: Reg, imm64: u64 };
    Bconst { dst: Reg, val: bool => src2 };
    Vconst { dst: Reg, pool_id: u32 => lo32 };

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

    I32AddImm { dst: Reg, src1: Reg, imm: u32 => lo32 };
    I32SubImm { dst: Reg, src1: Reg, imm: u32 => lo32 };
    I32AndImm { dst: Reg, src1: Reg, imm: u32 => lo32 };
    I32OrImm { dst: Reg, src1: Reg, imm: u32 => lo32 };
    I32XorImm { dst: Reg, src1: Reg, imm: u32 => lo32 };
    I32ShlImm { dst: Reg, src1: Reg, imm: u32 => lo32 };
    I32ShrSImm { dst: Reg, src1: Reg, imm: u32 => lo32 };
    I32ShrUImm { dst: Reg, src1: Reg, imm: u32 => lo32 };

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
    I32Load { dst: Reg, ptr: Reg => src1, offset: u32 => lo32 };
    I64Load { dst: Reg, ptr: Reg => src1, offset: u32 => lo32 };
    F32Load { dst: Reg, ptr: Reg => src1, offset: u32 => lo32 };
    F64Load { dst: Reg, ptr: Reg => src1, offset: u32 => lo32 };
    I8Load { dst: Reg, ptr: Reg => src1, offset: u32 => lo32 };
    I16Load { dst: Reg, ptr: Reg => src1, offset: u32 => lo32 };

    I32Store { val: Reg => src1, ptr: Reg => src2, offset: u32 => lo32 };
    I64Store { val: Reg => src1, ptr: Reg => src2, offset: u32 => lo32 };
    F32Store { val: Reg => src1, ptr: Reg => src2, offset: u32 => lo32 };
    F64Store { val: Reg => src1, ptr: Reg => src2, offset: u32 => lo32 };
    I8Store { val: Reg => src1, ptr: Reg => src2, offset: u32 => lo32 };
    I16Store { val: Reg => src1, ptr: Reg => src2, offset: u32 => lo32 };

    // === Conversions ===
    ExtendS { dst: Reg, src: Reg => src1, ty: TypePair => src2 };
    ExtendU { dst: Reg, src: Reg => src1, ty: TypePair => src2 };
    Wrap { dst: Reg, src: Reg => src1, ty: TypePair => src2 };

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
    StackAddr { dst: Reg, offset: u32 => lo32 };
    StackLoad { dst: Reg, ty: ScalarType => src2, offset: u32 => lo32 };
    StackStore { val: Reg => src1, ty: ScalarType => src2, offset: u32 => lo32 };

    PtrIndex { dst: Reg, ptr: Reg => src1, index: Reg => src2, scale: u32 => lo32, offset: u32 => hi32 };

    // === Control Flow ===
    Jump { offset: i64 => imm64 };
    JumpWithMoves { data_offset: u32 => lo32 };
    Br { cond: Reg => dst, then_offset: i32 => lo32, else_offset: i32 => hi32 };
    BrWithMoves { cond: Reg => dst, then_idx: u32 => lo32, else_idx: u32 => hi32 };
    BrTable { idx_reg: Reg => dst, data_offset: u32 => lo32, num_targets: u32 => hi32 };

    Select { dst: Reg, cond: Reg => src1, then_reg: Reg => src2, else_reg: Reg => lo32 };
    Return { data_offset: u32 => lo32, num_vals: u32 => hi32 };

    Call { func_id: u32 => lo32, data_offset: u32 => hi32, num_rets: u16 => src2, num_args: u16 => src1 };
    CallIndirect { ptr: Reg => dst, data_offset: u32 => lo32, num_rets: u16 => src2, num_args: u16 => src1 };
    CallIntrinsic { intrinsic: u16 => src2, data_offset: u32 => lo32, num_rets: u16 => dst, num_args: u16 => src1 };

    // Note: Call/CallIndirect/CallIntrinsic layout:
    // - data_offset: offset in regs where ret_regs + arg_regs are stored
    // - num_rets/num_args: encoded in src1/src2/dst slots to avoid memory access

    RegMove { dst: Reg, src: Reg => src1 };
    Unreachable {};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_only_instruction_uses_one_word() {
        let mut code = Vec::new();
        emit::I32Add(&mut code, Reg(60_000), Reg(50_000), Reg(40_000));

        assert_eq!(core::mem::size_of::<CodeWord>(), 8);
        assert_eq!(code.len(), 1);
        assert!(matches!(
            unsafe { DecodedInstruction::read(code.as_ptr()) },
            DecodedInstruction::I32Add {
                dst: Reg(60_000),
                src1: Reg(50_000),
                src2: Reg(40_000),
            }
        ));
    }

    #[test]
    fn immediate_instruction_appends_payload_word() {
        let mut code = Vec::new();
        emit::Iconst(&mut code, Reg(7), 0xfedc_ba98_7654_3210);

        assert_eq!(code.len(), 2);
        assert!(matches!(
            unsafe { DecodedInstruction::read(code.as_ptr()) },
            DecodedInstruction::Iconst {
                dst: Reg(7),
                imm64: 0xfedc_ba98_7654_3210,
            }
        ));
    }

    #[test]
    fn mixed_width_stream_preserves_header_boundaries() {
        let mut code = Vec::new();
        emit::RegMove(&mut code, Reg(1), Reg(2));
        emit::Br(&mut code, Reg(3), -24, 40);
        emit::Unreachable(&mut code);

        assert_eq!(code.len(), 4);
        assert_eq!(unsafe { code[0].opcode() }, Opcode::RegMove);
        assert_eq!(unsafe { code[1].opcode() }, Opcode::Br);
        assert_eq!(unsafe { code[3].opcode() }, Opcode::Unreachable);
        assert_eq!(Opcode::Br.words(), 2);
        assert!(matches!(
            unsafe { DecodedInstruction::read(code.as_ptr().add(1)) },
            DecodedInstruction::Br {
                cond: Reg(3),
                then_offset: -24,
                else_offset: 40,
            }
        ));
    }
}
