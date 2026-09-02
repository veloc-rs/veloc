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
        let Some(from) = ScalarType::from_code((raw & 0xFF) as u8) else {
            panic!("invalid source scalar type in bytecode")
        };
        let Some(to) = ScalarType::from_code((raw >> 8) as u8) else {
            panic!("invalid destination scalar type in bytecode")
        };
        Self { from, to }
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
/// Works symmetrically with [`FromSlot`] but for immediate/offset slots.
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

trait FromRawImm64: Sized {
    fn from_raw_u64(v: u64) -> Self;
}

impl FromRawImm64 for u64 {
    #[inline(always)]
    fn from_raw_u64(v: u64) -> Self {
        v
    }
}

impl FromRawImm64 for i64 {
    #[inline(always)]
    fn from_raw_u64(v: u64) -> Self {
        v as i64
    }
}

trait IntoRawImm64 {
    fn into_raw_imm64(self) -> u64;
}

impl IntoRawImm64 for u64 {
    #[inline(always)]
    fn into_raw_imm64(self) -> u64 {
        self
    }
}

impl IntoRawImm64 for i64 {
    #[inline(always)]
    fn into_raw_imm64(self) -> u64 {
        self as u64
    }
}

trait FromInlineImm16: Sized {
    fn from_inline_imm16(v: u16) -> Self;
}

impl FromInlineImm16 for i16 {
    #[inline(always)]
    fn from_inline_imm16(v: u16) -> Self {
        v as i16
    }
}

trait IntoInlineImm16 {
    fn into_inline_imm16(self) -> u16;
}

impl IntoInlineImm16 for i16 {
    #[inline(always)]
    fn into_inline_imm16(self) -> u16 {
        self as u16
    }
}

/// Decode a raw `u16` slot into the logical field type.
/// - `Reg`  → wraps the index in `Reg(...)`
/// - `u16`  → returns the raw value as-is
pub(crate) trait FromSlot: Sized {
    fn from_slot(v: u16) -> Self;
}

/// Convert a logical field value into a raw `u16` slot.
pub(crate) trait IntoSlot {
    fn into_slot(self) -> u16;
}

impl FromSlot for Reg {
    #[inline(always)]
    fn from_slot(v: u16) -> Self {
        Reg(v)
    }
}

impl FromSlot for u16 {
    #[inline(always)]
    fn from_slot(v: u16) -> Self {
        v
    }
}

impl FromSlot for bool {
    #[inline(always)]
    fn from_slot(v: u16) -> Self {
        v != 0
    }
}

impl IntoSlot for Reg {
    #[inline(always)]
    fn into_slot(self) -> u16 {
        self.0
    }
}

impl IntoSlot for u16 {
    #[inline(always)]
    fn into_slot(self) -> u16 {
        self
    }
}

impl IntoSlot for bool {
    #[inline(always)]
    fn into_slot(self) -> u16 {
        self as u16
    }
}

impl FromSlot for TypePair {
    #[inline(always)]
    fn from_slot(v: u16) -> Self {
        TypePair::unpack(v)
    }
}

impl IntoSlot for TypePair {
    #[inline(always)]
    fn into_slot(self) -> u16 {
        TypePair::pack(self.from, self.to)
    }
}

impl FromSlot for ScalarType {
    #[inline(always)]
    fn from_slot(v: u16) -> Self {
        Self::from_code(v as u8).expect("invalid scalar type in bytecode")
    }
}

impl IntoSlot for ScalarType {
    #[inline(always)]
    fn into_slot(self) -> u16 {
        self as u16
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct SlotsWord {
    opcode: u16,
    slot0: u16,
    slot1: u16,
    slot2: u16,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Const32Word {
    opcode: u16,
    dst: u16,
    imm: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct BinaryImm16Word {
    opcode: u16,
    dst: u16,
    src: u16,
    imm: u16,
}

/// One aligned word in the executable bytecode stream.
///
/// An instruction is either fully encoded in one word or followed by one
/// payload word. The opcode always occupies the first `u16` of its first word.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) union CodeWord {
    slots: SlotsWord,
    const32: Const32Word,
    binary_imm16: BinaryImm16Word,
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
    assert!(core::mem::size_of::<SlotsWord>() == 8);
    assert!(core::mem::offset_of!(SlotsWord, opcode) == 0);
    assert!(core::mem::offset_of!(SlotsWord, slot0) == 2);
    assert!(core::mem::offset_of!(SlotsWord, slot1) == 4);
    assert!(core::mem::offset_of!(SlotsWord, slot2) == 6);
    assert!(core::mem::size_of::<Const32Word>() == 8);
    assert!(core::mem::offset_of!(Const32Word, opcode) == 0);
    assert!(core::mem::offset_of!(Const32Word, dst) == 2);
    assert!(core::mem::offset_of!(Const32Word, imm) == 4);
    assert!(core::mem::size_of::<BinaryImm16Word>() == 8);
    assert!(core::mem::offset_of!(BinaryImm16Word, opcode) == 0);
    assert!(core::mem::offset_of!(BinaryImm16Word, dst) == 2);
    assert!(core::mem::offset_of!(BinaryImm16Word, src) == 4);
    assert!(core::mem::offset_of!(BinaryImm16Word, imm) == 6);
    assert!(core::mem::size_of::<CodeWord>() == 8);
    assert!(core::mem::align_of::<CodeWord>() == 8);
};

impl CodeWord {
    #[inline(always)]
    const fn slots(opcode: Opcode, slot0: u16, slot1: u16, slot2: u16) -> Self {
        Self {
            slots: SlotsWord {
                opcode: opcode as u16,
                slot0,
                slot1,
                slot2,
            },
        }
    }

    #[inline(always)]
    pub(crate) const fn from_payload(payload: u64) -> Self {
        Self { raw: payload }
    }

    #[inline(always)]
    pub(crate) unsafe fn opcode(self) -> Opcode {
        // SAFETY: every instruction word initializes all eight bytes.
        let raw = unsafe { self.slots.opcode };
        debug_assert!((raw as usize) < Opcode::COUNT);
        // SAFETY: executable instruction headers are only constructed from an
        // Opcode. Payload words are never passed to dispatch as headers.
        unsafe { core::mem::transmute(raw) }
    }

    #[inline(always)]
    unsafe fn read_slot0(ip: *const CodeWord) -> u16 {
        // SAFETY: `ip` points to an instruction header.
        unsafe { (*ip).slots.slot0 }
    }

    #[inline(always)]
    unsafe fn read_slot1(ip: *const CodeWord) -> u16 {
        // SAFETY: `ip` points to an instruction header.
        unsafe { (*ip).slots.slot1 }
    }

    #[inline(always)]
    unsafe fn read_slot2(ip: *const CodeWord) -> u16 {
        // SAFETY: `ip` points to an instruction header.
        unsafe { (*ip).slots.slot2 }
    }

    #[inline(always)]
    unsafe fn read_inline_imm32(ip: *const CodeWord) -> u32 {
        // SAFETY: `ip` points to an instruction using `Const32Word`.
        unsafe { (*ip).const32.imm }
    }

    #[inline(always)]
    unsafe fn read_inline_imm16(ip: *const CodeWord) -> u16 {
        // SAFETY: `ip` points to an instruction using `BinaryImm16Word`.
        unsafe { (*ip).binary_imm16.imm }
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
    fn set_slot0(&mut self, value: u16) {
        self.slots.slot0 = value;
    }

    #[inline(always)]
    fn set_slot1(&mut self, value: u16) {
        self.slots.slot1 = value;
    }

    #[inline(always)]
    fn set_slot2(&mut self, value: u16) {
        self.slots.slot2 = value;
    }

    #[inline(always)]
    fn set_inline_imm32(&mut self, value: u32) {
        self.const32.imm = value;
    }

    #[inline(always)]
    fn set_inline_imm16(&mut self, value: u16) {
        self.binary_imm16.imm = value;
    }
}

macro_rules! define_opcodes {
    ($(
        $name:ident [$format:ident] {
            $( $arg:ident : $ty:ty ),*
        }
    );* $(;)?) => {
        #[repr(u16)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub(crate) enum Opcode {
            $($name),*
        }

        const _: () = assert!(Opcode::COUNT <= u16::MAX as usize + 1);

        impl Opcode {
            /// Number of entries in the dense opcode space.
            pub(crate) const COUNT: usize = 0 $(+ { let _ = stringify!($name); 1 })*;

            /// Whether this opcode is followed by a 64-bit payload word.
            #[inline(always)]
            pub(crate) const fn has_payload(self) -> bool {
                match self {
                    $(Opcode::$name => define_opcodes!(@has_payload $format)),*
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
                    define_opcodes!(@decode $format, ip, payload; $($arg),*);
                    ($($arg,)*)
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
                    let mut header = CodeWord::slots(Opcode::$name, 0, 0, 0);
                    #[allow(unused_mut, unused_assignments)]
                    let mut payload = 0u64;

                    define_opcodes!(@encode $format, header, payload; $($arg),*);

                    code.push(header);
                    if Opcode::$name.has_payload() {
                        code.push(CodeWord::from_payload(payload));
                    }
                }
            )*
        }
    };

    (@has_payload Slots) => { false };
    (@has_payload Const32) => { false };
    (@has_payload BinaryImm16) => { false };
    (@has_payload Payload32) => { true };
    (@has_payload Payload32x2) => { true };
    (@has_payload Payload64) => { true };

    // Types are inferred from each generated decoder's return tuple. Format
    // helpers recursively consume register slots, leaving trailing immediates.
    (@decode Slots, $ip:ident, $payload:ident; $($arg:ident),*) => {
        define_opcodes!(@decode_slots $ip; [read_slot0 read_slot1 read_slot2]; $($arg),*);
    };
    (@encode Slots, $word:ident, $payload:ident; $($arg:ident),*) => {
        define_opcodes!(@encode_slots $word; [set_slot0 set_slot1 set_slot2]; $($arg),*);
    };

    // Inline layouts: opcode:u16 + dst:u16 + immediate/register fields.
    (@decode Const32, $ip:ident, $payload:ident; $dst:ident, $imm:ident) => {
        let $dst = FromSlot::from_slot(unsafe { CodeWord::read_slot0($ip) });
        let $imm =
            FromRawImm::from_raw_u32(unsafe { CodeWord::read_inline_imm32($ip) });
    };
    (@encode Const32, $word:ident, $payload:ident; $dst:ident, $imm:ident) => {
        $word.set_slot0($dst.into_slot());
        $word.set_inline_imm32($imm.into_raw_imm());
    };

    (@decode BinaryImm16, $ip:ident, $payload:ident; $dst:ident, $src:ident, $imm:ident) => {
        let $dst = FromSlot::from_slot(unsafe { CodeWord::read_slot0($ip) });
        let $src = FromSlot::from_slot(unsafe { CodeWord::read_slot1($ip) });
        let $imm = FromInlineImm16::from_inline_imm16(unsafe {
            CodeWord::read_inline_imm16($ip)
        });
    };
    (@encode BinaryImm16, $word:ident, $payload:ident; $dst:ident, $src:ident, $imm:ident) => {
        $word.set_slot0($dst.into_slot());
        $word.set_slot1($src.into_slot());
        $word.set_inline_imm16($imm.into_inline_imm16());
    };

    (@decode Payload32, $ip:ident, $payload:ident; $($arg:ident),*) => {
        define_opcodes!(
            @decode_regs_then_imm32 $ip, $payload;
            [read_slot0 read_slot1 read_slot2];
            $($arg),*
        );
    };
    (@encode Payload32, $word:ident, $payload:ident; $($arg:ident),*) => {
        define_opcodes!(
            @encode_regs_then_imm32 $word, $payload;
            [set_slot0 set_slot1 set_slot2];
            $($arg),*
        );
    };

    (@decode Payload32x2, $ip:ident, $payload:ident; $($arg:ident),*) => {
        define_opcodes!(
            @decode_regs_then_imm32x2 $ip, $payload;
            [read_slot0 read_slot1 read_slot2];
            $($arg),*
        );
    };
    (@encode Payload32x2, $word:ident, $payload:ident; $($arg:ident),*) => {
        define_opcodes!(
            @encode_regs_then_imm32x2 $word, $payload;
            [set_slot0 set_slot1 set_slot2];
            $($arg),*
        );
    };

    (@decode Payload64, $ip:ident, $payload:ident; $($arg:ident),*) => {
        define_opcodes!(
            @decode_regs_then_imm64 $ip, $payload;
            [read_slot0 read_slot1 read_slot2];
            $($arg),*
        );
    };
    (@encode Payload64, $word:ident, $payload:ident; $($arg:ident),*) => {
        define_opcodes!(
            @encode_regs_then_imm64 $word, $payload;
            [set_slot0 set_slot1 set_slot2];
            $($arg),*
        );
    };

    // Recursively map leading operands to the available u16 slots.
    (@decode_slots $ip:ident; [$($read:ident)*];) => {};
    (@decode_slots $ip:ident; [$read:ident $($reads:ident)*]; $arg:ident $(, $rest:ident)*) => {
        let $arg = FromSlot::from_slot(unsafe { CodeWord::$read($ip) });
        define_opcodes!(@decode_slots $ip; [$($reads)*]; $($rest),*);
    };
    (@encode_slots $word:ident; [$($set:ident)*];) => {};
    (@encode_slots $word:ident; [$set:ident $($sets:ident)*]; $arg:ident $(, $rest:ident)*) => {
        $word.$set($arg.into_slot());
        define_opcodes!(@encode_slots $word; [$($sets)*]; $($rest),*);
    };

    // The final operand is a 32-bit payload; preceding operands use slots.
    (@decode_regs_then_imm32 $ip:ident, $payload:ident; [$($read:ident)*]; $imm:ident) => {
        let $imm = FromRawImm::from_raw_u32($payload as u32);
    };
    (@decode_regs_then_imm32 $ip:ident, $payload:ident; [$read:ident $($reads:ident)*]; $reg:ident, $($rest:ident),+) => {
        let $reg = FromSlot::from_slot(unsafe { CodeWord::$read($ip) });
        define_opcodes!(
            @decode_regs_then_imm32 $ip, $payload; [$($reads)*]; $($rest),+
        );
    };
    (@encode_regs_then_imm32 $word:ident, $payload:ident; [$($set:ident)*]; $imm:ident) => {
        $payload = $imm.into_raw_imm() as u64;
    };
    (@encode_regs_then_imm32 $word:ident, $payload:ident; [$set:ident $($sets:ident)*]; $reg:ident, $($rest:ident),+) => {
        $word.$set($reg.into_slot());
        define_opcodes!(
            @encode_regs_then_imm32 $word, $payload; [$($sets)*]; $($rest),+
        );
    };

    // The final two operands occupy the low/high halves of one payload word.
    (@decode_regs_then_imm32x2 $ip:ident, $payload:ident; [$($read:ident)*]; $lo:ident, $hi:ident) => {
        let $lo = FromRawImm::from_raw_u32($payload as u32);
        let $hi = FromRawImm::from_raw_u32(($payload >> 32) as u32);
    };
    (@decode_regs_then_imm32x2 $ip:ident, $payload:ident; [$read:ident $($reads:ident)*]; $reg:ident, $next:ident, $($rest:ident),+) => {
        let $reg = FromSlot::from_slot(unsafe { CodeWord::$read($ip) });
        define_opcodes!(
            @decode_regs_then_imm32x2 $ip, $payload;
            [$($reads)*];
            $next, $($rest),+
        );
    };
    (@encode_regs_then_imm32x2 $word:ident, $payload:ident; [$($set:ident)*]; $lo:ident, $hi:ident) => {
        $payload = ($lo.into_raw_imm() as u64) | (($hi.into_raw_imm() as u64) << 32);
    };
    (@encode_regs_then_imm32x2 $word:ident, $payload:ident; [$set:ident $($sets:ident)*]; $reg:ident, $next:ident, $($rest:ident),+) => {
        $word.$set($reg.into_slot());
        define_opcodes!(
            @encode_regs_then_imm32x2 $word, $payload;
            [$($sets)*];
            $next, $($rest),+
        );
    };

    // The final operand is a full 64-bit payload; preceding operands use slots.
    (@decode_regs_then_imm64 $ip:ident, $payload:ident; [$($read:ident)*]; $imm:ident) => {
        let $imm = FromRawImm64::from_raw_u64($payload);
    };
    (@decode_regs_then_imm64 $ip:ident, $payload:ident; [$read:ident $($reads:ident)*]; $reg:ident, $($rest:ident),+) => {
        let $reg = FromSlot::from_slot(unsafe { CodeWord::$read($ip) });
        define_opcodes!(
            @decode_regs_then_imm64 $ip, $payload; [$($reads)*]; $($rest),+
        );
    };
    (@encode_regs_then_imm64 $word:ident, $payload:ident; [$($set:ident)*]; $imm:ident) => {
        $payload = $imm.into_raw_imm64();
    };
    (@encode_regs_then_imm64 $word:ident, $payload:ident; [$set:ident $($sets:ident)*]; $reg:ident, $($rest:ident),+) => {
        $word.$set($reg.into_slot());
        define_opcodes!(
            @encode_regs_then_imm64 $word, $payload; [$($sets)*]; $($rest),+
        );
    };
}

define_opcodes! {
    // === Constants ===
    Iconst [Payload64] { dst: Reg, imm64: u64 };
    Iconst32 [Const32] { dst: Reg, imm32: u32 };
    Fconst [Payload64] { dst: Reg, imm64: u64 };
    Fconst32 [Const32] { dst: Reg, bits32: u32 };
    Bconst [Slots] { dst: Reg, val: bool };
    Vconst [Payload32] { dst: Reg, pool_id: u32 };

    // === I32 Arithmetic ===
    I32Add [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I32AddImm16 [BinaryImm16] { dst: Reg, src1: Reg, imm16: i16 };
    I32Sub [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I32Mul [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I32DivS [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I32DivU [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I32RemS [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I32RemU [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I32And [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I32Or [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I32Xor [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I32Shl [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I32ShrS [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I32ShrU [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I32RotL [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I32RotR [Slots] { dst: Reg, src1: Reg, src2: Reg };

    I32AddImm [Payload32] { dst: Reg, src1: Reg, imm: u32 };
    I32SubImm [Payload32] { dst: Reg, src1: Reg, imm: u32 };
    I32AndImm [Payload32] { dst: Reg, src1: Reg, imm: u32 };
    I32OrImm [Payload32] { dst: Reg, src1: Reg, imm: u32 };
    I32XorImm [Payload32] { dst: Reg, src1: Reg, imm: u32 };
    I32ShlImm [Payload32] { dst: Reg, src1: Reg, imm: u32 };
    I32ShrSImm [Payload32] { dst: Reg, src1: Reg, imm: u32 };
    I32ShrUImm [Payload32] { dst: Reg, src1: Reg, imm: u32 };

    // === I64 Arithmetic ===
    I64Add [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I64AddImm16 [BinaryImm16] { dst: Reg, src1: Reg, imm16: i16 };
    I64Sub [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I64Mul [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I64DivS [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I64DivU [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I64RemS [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I64RemU [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I64And [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I64Or [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I64Xor [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I64Shl [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I64ShrS [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I64ShrU [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I64RotL [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I64RotR [Slots] { dst: Reg, src1: Reg, src2: Reg };

    I64AddImm [Payload64] { dst: Reg, src1: Reg, imm64: u64 };
    I64SubImm [Payload64] { dst: Reg, src1: Reg, imm64: u64 };
    I64AndImm [Payload64] { dst: Reg, src1: Reg, imm64: u64 };
    I64OrImm [Payload64] { dst: Reg, src1: Reg, imm64: u64 };
    I64XorImm [Payload64] { dst: Reg, src1: Reg, imm64: u64 };
    I64ShlImm [Payload64] { dst: Reg, src1: Reg, imm64: u64 };
    I64ShrSImm [Payload64] { dst: Reg, src1: Reg, imm64: u64 };
    I64ShrUImm [Payload64] { dst: Reg, src1: Reg, imm64: u64 };

    // === Comparisons ===
    I32Eq [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I32Ne [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I32LtS [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I32LtU [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I32LeS [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I32LeU [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I32GtS [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I32GtU [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I32GeS [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I32GeU [Slots] { dst: Reg, src1: Reg, src2: Reg };

    I64Eq [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I64Ne [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I64LtS [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I64LtU [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I64LeS [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I64LeU [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I64GtS [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I64GtU [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I64GeS [Slots] { dst: Reg, src1: Reg, src2: Reg };
    I64GeU [Slots] { dst: Reg, src1: Reg, src2: Reg };

    // === Float Arithmetic ===
    F32Add [Slots] { dst: Reg, src1: Reg, src2: Reg };
    F32Sub [Slots] { dst: Reg, src1: Reg, src2: Reg };
    F32Mul [Slots] { dst: Reg, src1: Reg, src2: Reg };
    F32Div [Slots] { dst: Reg, src1: Reg, src2: Reg };
    F32Neg [Slots] { dst: Reg, src1: Reg };
    F32Abs [Slots] { dst: Reg, src1: Reg };
    F32Sqrt [Slots] { dst: Reg, src1: Reg };
    F32Ceil [Slots] { dst: Reg, src1: Reg };
    F32Floor [Slots] { dst: Reg, src1: Reg };
    F32Trunc [Slots] { dst: Reg, src1: Reg };
    F32Nearest [Slots] { dst: Reg, src1: Reg };
    F32Min [Slots] { dst: Reg, src1: Reg, src2: Reg };
    F32Max [Slots] { dst: Reg, src1: Reg, src2: Reg };
    F32CopySign [Slots] { dst: Reg, src1: Reg, src2: Reg };

    F64Add [Slots] { dst: Reg, src1: Reg, src2: Reg };
    F64Sub [Slots] { dst: Reg, src1: Reg, src2: Reg };
    F64Mul [Slots] { dst: Reg, src1: Reg, src2: Reg };
    F64Div [Slots] { dst: Reg, src1: Reg, src2: Reg };
    F64Neg [Slots] { dst: Reg, src1: Reg };
    F64Abs [Slots] { dst: Reg, src1: Reg };
    F64Sqrt [Slots] { dst: Reg, src1: Reg };
    F64Ceil [Slots] { dst: Reg, src1: Reg };
    F64Floor [Slots] { dst: Reg, src1: Reg };
    F64Trunc [Slots] { dst: Reg, src1: Reg };
    F64Nearest [Slots] { dst: Reg, src1: Reg };
    F64Min [Slots] { dst: Reg, src1: Reg, src2: Reg };
    F64Max [Slots] { dst: Reg, src1: Reg, src2: Reg };
    F64CopySign [Slots] { dst: Reg, src1: Reg, src2: Reg };

    F32Eq [Slots] { dst: Reg, src1: Reg, src2: Reg };
    F32Ne [Slots] { dst: Reg, src1: Reg, src2: Reg };
    F32Lt [Slots] { dst: Reg, src1: Reg, src2: Reg };
    F32Le [Slots] { dst: Reg, src1: Reg, src2: Reg };
    F32Gt [Slots] { dst: Reg, src1: Reg, src2: Reg };
    F32Ge [Slots] { dst: Reg, src1: Reg, src2: Reg };
    F64Eq [Slots] { dst: Reg, src1: Reg, src2: Reg };
    F64Ne [Slots] { dst: Reg, src1: Reg, src2: Reg };
    F64Lt [Slots] { dst: Reg, src1: Reg, src2: Reg };
    F64Le [Slots] { dst: Reg, src1: Reg, src2: Reg };
    F64Gt [Slots] { dst: Reg, src1: Reg, src2: Reg };
    F64Ge [Slots] { dst: Reg, src1: Reg, src2: Reg };

    // === Memory ===
    I32Load [Payload32] { dst: Reg, ptr: Reg, offset: u32 };
    I64Load [Payload32] { dst: Reg, ptr: Reg, offset: u32 };
    F32Load [Payload32] { dst: Reg, ptr: Reg, offset: u32 };
    F64Load [Payload32] { dst: Reg, ptr: Reg, offset: u32 };
    I8Load [Payload32] { dst: Reg, ptr: Reg, offset: u32 };
    I16Load [Payload32] { dst: Reg, ptr: Reg, offset: u32 };

    I32Store [Payload32] { val: Reg, ptr: Reg, offset: u32 };
    I64Store [Payload32] { val: Reg, ptr: Reg, offset: u32 };
    F32Store [Payload32] { val: Reg, ptr: Reg, offset: u32 };
    F64Store [Payload32] { val: Reg, ptr: Reg, offset: u32 };
    I8Store [Payload32] { val: Reg, ptr: Reg, offset: u32 };
    I16Store [Payload32] { val: Reg, ptr: Reg, offset: u32 };

    // === Conversions ===
    ExtendS [Slots] { dst: Reg, src: Reg, ty: TypePair };
    ExtendU [Slots] { dst: Reg, src: Reg, ty: TypePair };
    Wrap [Slots] { dst: Reg, src: Reg, ty: TypePair };

    I32TruncF32S [Slots] { dst: Reg, src: Reg };
    I32TruncF32U [Slots] { dst: Reg, src: Reg };
    I32TruncF64S [Slots] { dst: Reg, src: Reg };
    I32TruncF64U [Slots] { dst: Reg, src: Reg };
    I64TruncF32S [Slots] { dst: Reg, src: Reg };
    I64TruncF32U [Slots] { dst: Reg, src: Reg };
    I64TruncF64S [Slots] { dst: Reg, src: Reg };
    I64TruncF64U [Slots] { dst: Reg, src: Reg };
    I32TruncSatF32S [Slots] { dst: Reg, src: Reg };
    I32TruncSatF32U [Slots] { dst: Reg, src: Reg };
    I32TruncSatF64S [Slots] { dst: Reg, src: Reg };
    I32TruncSatF64U [Slots] { dst: Reg, src: Reg };
    I64TruncSatF32S [Slots] { dst: Reg, src: Reg };
    I64TruncSatF32U [Slots] { dst: Reg, src: Reg };
    I64TruncSatF64S [Slots] { dst: Reg, src: Reg };
    I64TruncSatF64U [Slots] { dst: Reg, src: Reg };

    F32ConvertI32S [Slots] { dst: Reg, src: Reg };
    F32ConvertI32U [Slots] { dst: Reg, src: Reg };
    F32ConvertI64S [Slots] { dst: Reg, src: Reg };
    F32ConvertI64U [Slots] { dst: Reg, src: Reg };
    F64ConvertI32S [Slots] { dst: Reg, src: Reg };
    F64ConvertI32U [Slots] { dst: Reg, src: Reg };
    F64ConvertI64S [Slots] { dst: Reg, src: Reg };
    F64ConvertI64U [Slots] { dst: Reg, src: Reg };
    F32DemoteF64 [Slots] { dst: Reg, src: Reg };
    F64PromoteF32 [Slots] { dst: Reg, src: Reg };

    // === Bitwise ===
    I32Clz [Slots] { dst: Reg, src: Reg };
    I32Ctz [Slots] { dst: Reg, src: Reg };
    I32Popcnt [Slots] { dst: Reg, src: Reg };
    I64Clz [Slots] { dst: Reg, src: Reg };
    I64Ctz [Slots] { dst: Reg, src: Reg };
    I64Popcnt [Slots] { dst: Reg, src: Reg };
    I32Eqz [Slots] { dst: Reg, src_val: Reg };
    I64Eqz [Slots] { dst: Reg, src_val: Reg };

    // === Stack ===
    StackAddr [Payload32] { dst: Reg, offset: u32 };
    StackLoad [Payload32] { dst: Reg, ty: ScalarType, offset: u32 };
    StackStore [Payload32] { val: Reg, ty: ScalarType, offset: u32 };

    PtrIndex [Payload32x2] { dst: Reg, ptr: Reg, index: Reg, scale: u32, offset: u32 };

    // === Control Flow ===
    Jump [Payload64] { offset: i64 };
    JumpWithMoves [Payload32] { data_offset: u32 };
    Br [Payload32x2] { cond: Reg, then_offset: i32, else_offset: i32 };
    BrWithMoves [Payload32x2] { cond: Reg, then_idx: u32, else_idx: u32 };
    BrTable [Payload32x2] { idx_reg: Reg, data_offset: u32, num_targets: u32 };

    Select [Payload32] { dst: Reg, cond: Reg, then_reg: Reg, else_reg: Reg };
    Return [Payload32x2] { data_offset: u32, num_vals: u32 };

    // Slots fields come first; data-section identifiers occupy the payload.
    Call [Payload32x2] { num_rets: u16, num_args: u16, func_id: u32, data_offset: u32 };
    CallIndirect [Payload32] { ptr: Reg, num_rets: u16, num_args: u16, data_offset: u32 };
    CallIntrinsic [Payload32] { intrinsic: u16, num_rets: u16, num_args: u16, data_offset: u32 };

    RegMove [Slots] { dst: Reg, src: Reg };
    Unreachable [Slots] {};
}

/// Select the narrowest lossless encoding for instructions with compact forms.
/// The choice happens while compiling bytecode, so dispatch remains one opcode
/// lookup with no encoding-mode branch in the hot interpreter loop.
#[allow(non_snake_case)]
pub(crate) mod emit_auto {
    use super::{CodeWord, Reg, emit};

    #[inline(always)]
    pub(crate) fn Iconst(code: &mut Vec<CodeWord>, dst: Reg, imm64: u64) {
        if let Ok(imm32) = u32::try_from(imm64) {
            emit::Iconst32(code, dst, imm32);
        } else {
            emit::Iconst(code, dst, imm64);
        }
    }

    #[inline(always)]
    pub(crate) fn Fconst(code: &mut Vec<CodeWord>, dst: Reg, bits64: u64) {
        if let Ok(bits32) = u32::try_from(bits64) {
            emit::Fconst32(code, dst, bits32);
        } else {
            emit::Fconst(code, dst, bits64);
        }
    }

    #[inline(always)]
    pub(crate) fn I32AddImm(code: &mut Vec<CodeWord>, dst: Reg, src1: Reg, imm: u32) {
        if let Ok(imm16) = i16::try_from(imm as i32) {
            emit::I32AddImm16(code, dst, src1, imm16);
        } else {
            emit::I32AddImm(code, dst, src1, imm);
        }
    }

    #[inline(always)]
    pub(crate) fn I64AddImm(code: &mut Vec<CodeWord>, dst: Reg, src1: Reg, imm64: u64) {
        if let Ok(imm16) = i16::try_from(imm64 as i64) {
            emit::I64AddImm16(code, dst, src1, imm16);
        } else {
            emit::I64AddImm(code, dst, src1, imm64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_only_instruction_uses_one_word() {
        let mut code = Vec::new();
        emit::I32Add(&mut code, Reg(60_000), Reg(50_000), Reg(40_000));

        assert_eq!(core::mem::size_of::<CodeWord>(), 8);
        assert_eq!(core::mem::size_of::<Opcode>(), 2);
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
    fn auto_emit_uses_inline_immediates_at_the_boundaries() {
        let mut code = Vec::new();

        emit_auto::Iconst(&mut code, Reg(1), u32::MAX as u64);
        emit_auto::Fconst(&mut code, Reg(2), u32::MAX as u64);
        emit_auto::I32AddImm(&mut code, Reg(3), Reg(4), (-32_768_i32) as u32);
        emit_auto::I64AddImm(&mut code, Reg(5), Reg(6), (-32_768_i64) as u64);

        assert_eq!(code.len(), 4);
        assert!(matches!(
            unsafe { DecodedInstruction::read(code.as_ptr()) },
            DecodedInstruction::Iconst32 {
                dst: Reg(1),
                imm32: u32::MAX,
            }
        ));
        assert!(matches!(
            unsafe { DecodedInstruction::read(code.as_ptr().add(1)) },
            DecodedInstruction::Fconst32 {
                dst: Reg(2),
                bits32: u32::MAX,
            }
        ));
        assert!(matches!(
            unsafe { DecodedInstruction::read(code.as_ptr().add(2)) },
            DecodedInstruction::I32AddImm16 {
                dst: Reg(3),
                src1: Reg(4),
                imm16: -32_768,
            }
        ));
        assert!(matches!(
            unsafe { DecodedInstruction::read(code.as_ptr().add(3)) },
            DecodedInstruction::I64AddImm16 {
                dst: Reg(5),
                src1: Reg(6),
                imm16: -32_768,
            }
        ));
    }

    #[test]
    fn auto_emit_falls_back_when_immediate_does_not_fit() {
        let mut code = Vec::new();

        emit_auto::Iconst(&mut code, Reg(1), u32::MAX as u64 + 1);
        emit_auto::Fconst(&mut code, Reg(2), u32::MAX as u64 + 1);
        emit_auto::I32AddImm(&mut code, Reg(3), Reg(4), 32_768);
        emit_auto::I64AddImm(&mut code, Reg(5), Reg(6), 32_768);

        assert_eq!(code.len(), 8);
        assert_eq!(unsafe { code[0].opcode() }, Opcode::Iconst);
        assert_eq!(unsafe { code[2].opcode() }, Opcode::Fconst);
        assert_eq!(unsafe { code[4].opcode() }, Opcode::I32AddImm);
        assert_eq!(unsafe { code[6].opcode() }, Opcode::I64AddImm);
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
