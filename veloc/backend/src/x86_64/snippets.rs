use crate::RegisterHole;

// === Register Hole Helpers ===

const fn r_hole(offset: usize, shift: u8, mask: u8, rex: Option<(usize, u8)>) -> RegisterHole {
    RegisterHole {
        offset,
        shift,
        mask,
        extra_bit: rex,
    }
}

/// REX.B + bits 0-2 (Opcode or ModRM.rm)
const fn r_low3(offset: usize, rex_off: usize) -> RegisterHole {
    r_hole(offset, 0, 0x07, Some((rex_off, 0x01)))
}

/// REX.R + ModRM.reg (bits 3-5)
const fn r_mod_reg(offset: usize, rex_off: usize) -> RegisterHole {
    r_hole(offset, 3, 0x38, Some((rex_off, 0x04)))
}

/// REX.B + ModRM.rm (bits 0-2)
const fn r_mod_rm(offset: usize, rex_off: usize) -> RegisterHole {
    r_hole(offset, 0, 0x07, Some((rex_off, 0x01)))
}

// 32-bit versions without REX bits
const fn r32_low3(offset: usize) -> RegisterHole {
    r_hole(offset, 0, 0x07, None)
}
const fn r32_mod_reg(offset: usize) -> RegisterHole {
    r_hole(offset, 3, 0x38, None)
}
const fn r32_mod_rm(offset: usize) -> RegisterHole {
    r_hole(offset, 0, 0x07, None)
}

// === Macro Simplification ===

macro_rules! x86_insts {
    ($($variant:ident ($name:expr, $bytes:expr $(; $($key:ident ($val:expr)),*)?);)*) => {
        $crate::define_backend_insts! {
            X86_64Inst {
                $($variant ($name, $bytes $(; $($key = $val),*)?);)*
            }
        }
    };
}

x86_insts! {
    // === Control Flow & Function ===
    Prologue        ("prologue",    &[0x55, 0x48, 0x89, 0xe5]);
    Ret             ("ret",         &[0x5d, 0xc3]);
    Ud2             ("ud2",         &[0x0f, 0x0b]);
    CallRel32       ("call rel32",  &[0xe8, 0, 0, 0, 0]; h(1), s(4));
    CallReg         ("call r64",    &[0x40, 0xff, 0xd0]; r1(r_low3(2, 0)));

    // === Integer Move / Load / Store ===
    MovR32Imm32     ("mov r32, imm32",  &[0xb8, 0, 0, 0, 0]; h(1), s(4), r1(r32_low3(0)));
    MovR64Imm64     ("mov r64, imm64",  &[0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0]; h(2), s(8), r1(r_low3(1, 0)));
    MovRRbpOff      ("mov r, [rbp-off]",&[0x48, 0x8b, 0x85, 0, 0, 0, 0]; h(3), s(4), r1(r_mod_reg(2, 0)));
    MovR32RbpOff    ("mov r32, [rbp-off]",&[0x8b, 0x85, 0, 0, 0, 0]; h(2), s(4), r1(r32_mod_reg(1)));
    MovRbpOffR      ("mov [rbp-off], r",&[0x48, 0x89, 0x85, 0, 0, 0, 0]; h(3), s(4), r1(r_mod_reg(2, 0)));
    MovRbpOffR32    ("mov [rbp-off], r32",&[0x89, 0x85, 0, 0, 0, 0]; h(2), s(4), r1(r32_mod_reg(1)));
    MovRaxRbpOff    ("mov rax, [rbp-off]", &[0x48, 0x8b, 0x85, 0, 0, 0, 0]; h(3), s(4));
    MovRbpOffRax    ("mov [rbp-off], rax", &[0x48, 0x89, 0x85, 0, 0, 0, 0]; h(3), s(4));
    MovR64R64Off    ("mov r64, [r64+off]", &[0x48, 0x8b, 0x80, 0, 0, 0, 0]; h(3), s(4), r1(r_mod_reg(2, 0)), r2(r_mod_rm(2, 0)));
    MovR32R64Off    ("mov r32, [r64+off]", &[0x8b, 0x80, 0, 0, 0, 0]; h(2), s(4), r1(r32_mod_reg(1)), r2(r32_mod_rm(1)));
    MovR64OffR64    ("mov [r64+off], r64", &[0x48, 0x89, 0x80, 0, 0, 0, 0]; h(3), s(4), r1(r_mod_reg(2, 0)), r2(r_mod_rm(2, 0)));
    MovR64OffR32    ("mov [r64+off], r32", &[0x89, 0x80, 0, 0, 0, 0]; h(2), s(4), r1(r32_mod_reg(1)), r2(r32_mod_rm(1)));
    MovR64OffR16    ("mov [r64+off], r16", &[0x66, 0x89, 0x80, 0, 0, 0, 0]; h(3), s(4), r1(r32_mod_reg(2)), r2(r32_mod_rm(2)));
    MovR64OffR8     ("mov [r64+off], r8",  &[0x88, 0x80, 0, 0, 0, 0]; h(2), s(4), r1(r32_mod_reg(1)), r2(r32_mod_rm(1)));
    LeaRaxRbpOff    ("lea rax, [rbp-off]", &[0x48, 0x8d, 0x85, 0, 0, 0, 0]; h(3), s(4));
    PushRax         ("push rax",    &[0x50]);
    PopRax          ("pop rax",     &[0x58]);
    PopRbx          ("pop rbx",     &[0x5b]);

    // === Integer Arithmetic ===
    AddR32R32       ("add r32, r32", &[0x01, 0xd8]; r1(r32_mod_rm(1)), r2(r32_mod_reg(1)));
    SubR32R32       ("sub r32, r32", &[0x29, 0xd8]; r1(r32_mod_rm(1)), r2(r32_mod_reg(1)));
    MulR32R32       ("imul r32, r32",&[0x0f, 0xaf, 0xc3]; r1(r32_mod_reg(2)), r2(r32_mod_rm(2)));
    DivRcx          ("div rcx",      &[0xf7, 0xf1]);
    IdivRcx         ("idiv rcx",     &[0xf7, 0xf9]);
    DivRcx64        ("div rcx 64",   &[0x48, 0xf7, 0xf1]);
    IdivRcx64       ("idiv rcx 64",  &[0x48, 0xf7, 0xf9]);
    MovEcxEcx       ("mov ecx, ecx", &[0x89, 0xc9]);
    SubRspImm32     ("sub rsp, imm32", &[0x48, 0x81, 0xec, 0, 0, 0, 0]; h(3), s(4));
    AddRspImm32     ("add rsp, imm32", &[0x48, 0x81, 0xc4, 0, 0, 0, 0]; h(3), s(4));
    Cdq             ("cdq",          &[0x99]);
    Cqo             ("cqo",          &[0x48, 0x99]);

    // === Logical & Shift ===
    AndR32R32       ("and r32, r32", &[0x21, 0xd8]; r1(r32_mod_rm(1)), r2(r32_mod_reg(1)));
    OrR32R32        ("or r32, r32",  &[0x09, 0xd8]; r1(r32_mod_rm(1)), r2(r32_mod_reg(1)));
    XorR32R32       ("xor r32, r32", &[0x31, 0xd8]; r1(r32_mod_rm(1)), r2(r32_mod_reg(1)));
    XorEdxEdx       ("xor edx, edx", &[0x31, 0xd2]);
    XorRdxRdx64     ("xor rdx, rdx 64",&[0x48, 0x31, 0xd2]);
    ShlEaxCl        ("shl eax, cl",  &[0xd3, 0xe0]);
    ShrEaxCl        ("shr eax, cl",  &[0xd3, 0xe8]);
    SarEaxCl        ("sar eax, cl",  &[0xd3, 0xf8]);
    RolEaxCl        ("rol eax, cl",  &[0xd3, 0xc0]);
    RorEaxCl        ("ror eax, cl",  &[0xd3, 0xc8]);

    // === Comparison & SetCC ===
    CmpR32R32       ("cmp r32, r32", &[0x39, 0xd8]; r1(r32_mod_rm(1)), r2(r32_mod_reg(1)));
    CmpRaxImm8      ("cmp rax, imm8", &[0x48, 0x83, 0xf8, 0]; h(3), s(1));
    CmpRaxImm32     ("cmp rax, imm32",&[0x48, 0x3d, 0, 0, 0, 0]; h(2), s(4));
    TestEaxEax      ("test eax, eax",&[0x85, 0xc0]);
    TestRaxRax      ("test rax, rax",&[0x48, 0x85, 0xc0]);
    SeteAl          ("sete al",      &[0x0f, 0x94, 0xc0]);
    SetneAl         ("setne al",     &[0x0f, 0x95, 0xc0]);
    SetlAl          ("setl al",      &[0x0f, 0x9c, 0xc0]);
    SetgAl          ("setg al",      &[0x0f, 0x9f, 0xc0]);
    SetleAl         ("setle al",     &[0x0f, 0x9e, 0xc0]);
    SetgeAl         ("setge al",     &[0x0f, 0x9d, 0xc0]);
    SetbAl          ("setb al",      &[0x0f, 0x92, 0xc0]);
    SetaAl          ("seta al",      &[0x0f, 0x97, 0xc0]);
    SetbeAl         ("setbe al",     &[0x0f, 0x96, 0xc0]);
    SetaeAl         ("setae al",     &[0x0f, 0x93, 0xc0]);
    SetpAl          ("setp al",      &[0x0f, 0x9a, 0xc0]);
    SetnpAl         ("setnp al",     &[0x0f, 0x9b, 0xc0]);
    SetpBl          ("setp bl",      &[0x0f, 0x9a, 0xc3]);
    SetnpBl         ("setnp bl",     &[0x0f, 0x9b, 0xc3]);
    SeteBl          ("sete bl",      &[0x0f, 0x94, 0xc3]);
    SetneBl         ("setne bl",     &[0x0f, 0x95, 0xc3]);
    AndAlBl         ("and al, bl",   &[0x20, 0xd8]);
    OrAlBl          ("or al, bl",    &[0x08, 0xd8]);
    XorRaxRax       ("xor rax, rax", &[0x48, 0x31, 0xc0]);
    XorRbxRbx       ("xor rbx, rbx", &[0x48, 0x31, 0xdb]);

    // === Conditional Move ===
    CmovnzRaxRbx    ("cmovnz eax, ebx",&[0x0f, 0x45, 0xc3]);
    CmovnzRaxRbx64  ("cmovnz rax, rbx",&[0x48, 0x0f, 0x45, 0xc3]);

    // === Jumps ===
    JnzRel8         ("jnz rel8",     &[0x75, 0x00]; h(1), s(1));
    JnzRel32        ("jnz rel32",    &[0x0f, 0x85, 0, 0, 0, 0]; h(2), s(4));
    JzRel8          ("jz rel8",      &[0x74, 0x00]; h(1), s(1));
    JzRel32         ("jz rel32",     &[0x0f, 0x84, 0, 0, 0, 0]; h(2), s(4));
    JmpRel8         ("jmp rel8",     &[0xeb, 0x00]; h(1), s(1));
    JmpRel32        ("jmp rel32",    &[0xe9, 0, 0, 0, 0]; h(1), s(4));

    // === Special Moves & Zero Extension ===
    MovEcxEbx       ("mov ecx, ebx", &[0x89, 0xd9]);
    MovEaxEdx       ("mov eax, edx", &[0x89, 0xd0]);
    MovRaxRdx64     ("mov rax, rdx 64",&[0x48, 0x89, 0xd0]);
    MovEaxEax       ("mov eax, eax", &[0x89, 0xc0]);

    // === Floating Point (XMM) ===
    MovsdXRbpOff    ("movsd xmm, [rbp-off]", &[0xf2, 0x0f, 0x10, 0x85, 0, 0, 0, 0]; h(4), s(4), r1(r_mod_reg(3, 0)));
    MovsdRbpOffX    ("movsd [rbp-off], xmm", &[0xf2, 0x0f, 0x11, 0x85, 0, 0, 0, 0]; h(4), s(4), r1(r_mod_reg(3, 0)));
    MovssXRbpOff    ("movss xmm, [rbp-off]", &[0xf3, 0x0f, 0x10, 0x85, 0, 0, 0, 0]; h(4), s(4), r1(r_mod_reg(3, 0)));
    MovssRbpOffX    ("movss [rbp-off], xmm", &[0xf3, 0x0f, 0x11, 0x85, 0, 0, 0, 0]; h(4), s(4), r1(r_mod_reg(3, 0)));
    MovssXR64Off    ("movss xmm, [r64+off]", &[0xf3, 0x0f, 0x10, 0x80, 0, 0, 0, 0]; h(4), s(4), r1(r32_mod_reg(3)), r2(r32_mod_rm(3)));
    MovsdXR64Off    ("movsd xmm, [r64+off]", &[0xf2, 0x0f, 0x10, 0x80, 0, 0, 0, 0]; h(4), s(4), r1(r32_mod_reg(3)), r2(r32_mod_rm(3)));
    MovssR64OffX    ("movss [r64+off], xmm", &[0xf3, 0x0f, 0x11, 0x80, 0, 0, 0, 0]; h(4), s(4), r1(r32_mod_reg(3)), r2(r32_mod_rm(3)));
    MovsdR64OffX    ("movsd [r64+off], xmm", &[0xf2, 0x0f, 0x11, 0x80, 0, 0, 0, 0]; h(4), s(4), r1(r32_mod_reg(3)), r2(r32_mod_rm(3)));

    AddsdXX         ("addsd xmm, xmm", &[0xf2, 0x0f, 0x58, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));
    SubsdXX         ("subsd xmm, xmm", &[0xf2, 0x0f, 0x5c, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));
    MulsdXX         ("mulsd xmm, xmm", &[0xf2, 0x0f, 0x59, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));
    DivsdXX         ("divsd xmm, xmm", &[0xf2, 0x0f, 0x5e, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));

    AddssXX         ("addss xmm, xmm", &[0xf3, 0x0f, 0x58, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));
    SubssXX         ("subss xmm, xmm", &[0xf3, 0x0f, 0x5c, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));
    MulssXX         ("mulss xmm, xmm", &[0xf3, 0x0f, 0x59, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));
    DivssXX         ("divss xmm, xmm", &[0xf3, 0x0f, 0x5e, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));
    MinssXX         ("minss xmm, xmm", &[0xf3, 0x0f, 0x5d, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));
    MaxssXX         ("maxss xmm, xmm", &[0xf3, 0x0f, 0x5f, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));
    MinsdXX         ("minsd xmm, xmm", &[0xf2, 0x0f, 0x5d, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));
    MaxsdXX         ("maxsd xmm, xmm", &[0xf2, 0x0f, 0x5f, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));
    UcomissXX       ("ucomiss xmm, xmm", &[0x0f, 0x2e, 0xc0]; r1(r32_mod_reg(2)), r2(r32_mod_rm(2)));
    UcomisdXX       ("ucomisd xmm, xmm", &[0x66, 0x0f, 0x2e, 0xc0]; r1(r32_mod_reg(3)), r2(r32_mod_rm(3)));
    XorpsXX         ("xorps xmm, xmm", &[0x0f, 0x57, 0xc0]; r1(r32_mod_reg(2)), r2(r32_mod_rm(2)));
    XorpdXX         ("xorpd xmm, xmm", &[0x66, 0x0f, 0x57, 0xc0]; r1(r32_mod_reg(3)), r2(r32_mod_rm(3)));
    SqrtssXX        ("sqrtss xmm, xmm", &[0xf3, 0x0f, 0x51, 0xc0]; r1(r32_mod_reg(3)), r2(r32_mod_rm(3)));
    SqrtsdXX        ("sqrtsd xmm, xmm", &[0xf2, 0x0f, 0x51, 0xc0]; r1(r32_mod_reg(3)), r2(r32_mod_reg(3)));

    // === Type Conversion & Bitwise ===
    Cvtsi2ssXR      ("cvtsi2ss xmm, r32", &[0xf3, 0x0f, 0x2a, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));
    Cvtsi2sdXR      ("cvtsi2sd xmm, r32", &[0xf2, 0x0f, 0x2a, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));
    Cvtsi2ssXR64    ("cvtsi2ss xmm, r64", &[0xf3, 0x48, 0x0f, 0x2a, 0xc0]; r1(r_mod_rm(4, 1)), r2(r32_mod_reg(4)));
    Cvtsi2sdXR64    ("cvtsi2sd xmm, r64", &[0xf2, 0x48, 0x0f, 0x2a, 0xc0]; r1(r_mod_rm(4, 1)), r2(r32_mod_reg(4)));

    Cvttss2siRX     ("cvttss2si r32, xmm", &[0xf3, 0x0f, 0x2c, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));
    Cvttsd2siRX     ("cvttsd2si r32, xmm", &[0xf2, 0x0f, 0x2c, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));
    Cvttss2siRX64    ("cvttss2si r64, xmm", &[0xf3, 0x48, 0x0f, 0x2c, 0xc0]; r1(r_mod_rm(4, 1)), r2(r32_mod_reg(4)));
    Cvttsd2siRX64    ("cvttsd2si r64, xmm", &[0xf2, 0x48, 0x0f, 0x2c, 0xc0]; r1(r_mod_rm(4, 1)), r2(r32_mod_reg(4)));

    Cvtss2sdXX      ("cvtss2sd xmm, xmm",  &[0xf3, 0x0f, 0x5a, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));
    Cvtsd2ssXX      ("cvtsd2ss xmm, xmm",  &[0xf2, 0x0f, 0x5a, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));

    MovdRX          ("movd r32, xmm",  &[0x66, 0x0f, 0x7e, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));
    MovqRX64        ("movq r64, xmm",  &[0x66, 0x48, 0x0f, 0x7e, 0xc0]; r1(r_mod_rm(4, 1)), r2(r32_mod_reg(4)));
    MovdXR          ("movd xmm, r32",  &[0x66, 0x0f, 0x6e, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));
    MovqXR64        ("movq xmm, r64",  &[0x66, 0x48, 0x0f, 0x6e, 0xc0]; r1(r_mod_rm(4, 1)), r2(r32_mod_reg(4)));

    AndpsXX         ("andps xmm, xmm", &[0x0f, 0x54, 0xc0]; r1(r32_mod_rm(2)), r2(r32_mod_reg(2)));
    AndpdXX         ("andpd xmm, xmm", &[0x66, 0x0f, 0x54, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));
    AndnpsXX        ("andnps xmm, xmm", &[0x0f, 0x55, 0xc0]; r1(r32_mod_rm(2)), r2(r32_mod_reg(2)));
    AndnpdXX        ("andnpd xmm, xmm", &[0x66, 0x0f, 0x55, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));
    OrpsXX          ("orps xmm, xmm", &[0x0f, 0x56, 0xc0]; r1(r32_mod_rm(2)), r2(r32_mod_reg(2)));
    OrpdXX          ("orpd xmm, xmm", &[0x66, 0x0f, 0x56, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));

    RoundssXXI      ("roundss xmm, xmm, imm8", &[0x66, 0x0f, 0x3a, 0x0a, 0xc0, 0]; h(5), s(1), r1(r32_mod_rm(4)), r2(r32_mod_reg(4)));
    RoundsdXXI      ("roundsd xmm, xmm, imm8", &[0x66, 0x0f, 0x3a, 0x0b, 0xc0, 0]; h(5), s(1), r1(r32_mod_rm(4)), r2(r32_mod_reg(4)));

    MovzxEaxAl      ("movzx eax, al",  &[0x0f, 0xb6, 0xc0]);
    Movzx16EaxAx    ("movzx eax, ax",  &[0x0f, 0xb7, 0xc0]);
    Movsx8RaxRax    ("movsx rax, al",  &[0x0f, 0xbe, 0xc0]);
    Movsx16RaxRax   ("movsx rax, ax",  &[0x0f, 0xbf, 0xc0]);
    MovsxdRaxEax    ("movsxd rax, eax",&[0x48, 0x63, 0xc0]);
    Movzx8RaxRax    ("movzx rax, al",  &[0x48, 0x0f, 0xb6, 0xc0]);
    Movzx16RaxRax   ("movzx rax, ax",  &[0x48, 0x0f, 0xb7, 0xc0]);
    Movzx8R64R64Off ("movzx r64, byte [r64+off]", &[0x48, 0x0f, 0xb6, 0x80, 0, 0, 0, 0]; h(4), s(4), r1(r_mod_reg(3, 0)), r2(r_mod_rm(3, 0)));
    Movzx16R64R64Off("movzx r64, word [r64+off]", &[0x48, 0x0f, 0xb7, 0x80, 0, 0, 0, 0]; h(4), s(4), r1(r_mod_reg(3, 0)), r2(r_mod_rm(3, 0)));

    LzcntRR         ("lzcnt r32, r32", &[0xf3, 0x0f, 0xbd, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));
    TzcntRR         ("tzcnt r32, r32", &[0xf3, 0x0f, 0xbc, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));
    PopcntRR        ("popcnt r32, r32", &[0xf3, 0x0f, 0xb8, 0xc0]; r1(r32_mod_rm(3)), r2(r32_mod_reg(3)));
    LzcntR64R64     ("lzcnt r64, r64", &[0xf3, 0x48, 0x0f, 0xbd, 0xc0]; r1(r_mod_rm(4, 1)), r2(r_mod_reg(4, 1)));
    TzcntR64R64     ("tzcnt r64, r64", &[0xf3, 0x48, 0x0f, 0xbc, 0xc0]; r1(r_mod_rm(4, 1)), r2(r_mod_reg(4, 1)));
    PopcntR64R64    ("popcnt r64, r64", &[0xf3, 0x48, 0x0f, 0xb8, 0xc0]; r1(r_mod_rm(4, 1)), r2(r_mod_reg(4, 1)));
}
