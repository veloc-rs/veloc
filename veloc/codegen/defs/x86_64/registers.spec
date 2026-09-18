// Machine constants use the target schema's Register, RegisterView and
// RegisterClass types. Views share a root identity and declare write effects.
template Root(Name: ident, Bits: expr, Id: expr, Encoding: expr) {
    const Name: Register = Register {
        bits: Bits,
        id: Id,
        encoding: Encoding,
        reserved: false,
        roles: [],
    };
}

template View(Name: ident, Base: ident, Bits: expr, Write: expr) {
    const Name: RegisterView = RegisterView {
        base: Base,
        offset: 0,
        bits: Bits,
        write: Write,
    };
}

expand Root(RAX, 64, 0, 0);
expand Root(RCX, 64, 1, 1);
expand Root(RDX, 64, 2, 2);
expand Root(RBX, 64, 3, 3);
const RSP: Register = Register {
    bits: 64,
    id: 4,
    encoding: 4,
    reserved: true,
    roles: [RegisterRole::StackPointer],
};
const RBP: Register = Register {
    bits: 64,
    id: 5,
    encoding: 5,
    reserved: true,
    roles: [RegisterRole::FramePointer],
};
expand Root(RSI, 64, 6, 6);
expand Root(RDI, 64, 7, 7);
expand Root(R8, 64, 8, 8);
expand Root(R9, 64, 9, 9);
expand Root(R10, 64, 10, 10);
expand Root(R11, 64, 11, 11);
expand Root(R12, 64, 12, 12);
expand Root(R13, 64, 13, 13);
expand Root(R14, 64, 14, 14);
expand Root(R15, 64, 15, 15);
expand Root(XMM0, 128, 16, 0);
expand Root(XMM1, 128, 17, 1);
expand Root(XMM2, 128, 18, 2);
expand Root(XMM3, 128, 19, 3);
expand Root(XMM4, 128, 20, 4);
expand Root(XMM5, 128, 21, 5);
expand Root(XMM6, 128, 22, 6);
expand Root(XMM7, 128, 23, 7);
expand Root(XMM8, 128, 24, 8);
expand Root(XMM9, 128, 25, 9);
expand Root(XMM10, 128, 26, 10);
expand Root(XMM11, 128, 27, 11);
expand Root(XMM12, 128, 28, 12);
expand Root(XMM13, 128, 29, 13);
expand Root(XMM14, 128, 30, 14);
expand Root(XMM15, 128, 31, 15);
expand View(EAX, RAX, 32, WriteEffect::ZeroExtend);
expand View(ECX, RCX, 32, WriteEffect::ZeroExtend);
expand View(EDX, RDX, 32, WriteEffect::ZeroExtend);
expand View(EBX, RBX, 32, WriteEffect::ZeroExtend);
expand View(ESP, RSP, 32, WriteEffect::ZeroExtend);
expand View(EBP, RBP, 32, WriteEffect::ZeroExtend);
expand View(ESI, RSI, 32, WriteEffect::ZeroExtend);
expand View(EDI, RDI, 32, WriteEffect::ZeroExtend);
expand View(AL, RAX, 8, WriteEffect::Preserve);
expand View(AX, RAX, 16, WriteEffect::Preserve);
expand View(CL, RCX, 8, WriteEffect::Preserve);
expand View(CX, RCX, 16, WriteEffect::Preserve);
expand View(DL, RDX, 8, WriteEffect::Preserve);
expand View(DX, RDX, 16, WriteEffect::Preserve);
expand View(BL, RBX, 8, WriteEffect::Preserve);
expand View(BX, RBX, 16, WriteEffect::Preserve);
expand View(SPL, RSP, 8, WriteEffect::Preserve);
expand View(SP, RSP, 16, WriteEffect::Preserve);
expand View(BPL, RBP, 8, WriteEffect::Preserve);
expand View(BP, RBP, 16, WriteEffect::Preserve);
expand View(SIL, RSI, 8, WriteEffect::Preserve);
expand View(SI, RSI, 16, WriteEffect::Preserve);
expand View(DIL, RDI, 8, WriteEffect::Preserve);
expand View(DI, RDI, 16, WriteEffect::Preserve);
expand View(R8B, R8, 8, WriteEffect::Preserve);
expand View(R8W, R8, 16, WriteEffect::Preserve);
expand View(R8D, R8, 32, WriteEffect::ZeroExtend);
expand View(R9B, R9, 8, WriteEffect::Preserve);
expand View(R9W, R9, 16, WriteEffect::Preserve);
expand View(R9D, R9, 32, WriteEffect::ZeroExtend);
expand View(R10B, R10, 8, WriteEffect::Preserve);
expand View(R10W, R10, 16, WriteEffect::Preserve);
expand View(R10D, R10, 32, WriteEffect::ZeroExtend);
expand View(R11B, R11, 8, WriteEffect::Preserve);
expand View(R11W, R11, 16, WriteEffect::Preserve);
expand View(R11D, R11, 32, WriteEffect::ZeroExtend);
expand View(R12B, R12, 8, WriteEffect::Preserve);
expand View(R12W, R12, 16, WriteEffect::Preserve);
expand View(R12D, R12, 32, WriteEffect::ZeroExtend);
expand View(R13B, R13, 8, WriteEffect::Preserve);
expand View(R13W, R13, 16, WriteEffect::Preserve);
expand View(R13D, R13, 32, WriteEffect::ZeroExtend);
expand View(R14B, R14, 8, WriteEffect::Preserve);
expand View(R14W, R14, 16, WriteEffect::Preserve);
expand View(R14D, R14, 32, WriteEffect::ZeroExtend);
expand View(R15B, R15, 8, WriteEffect::Preserve);
expand View(R15W, R15, 16, WriteEffect::Preserve);
expand View(R15D, R15, 32, WriteEffect::ZeroExtend);

const GPR64: RegisterClass = RegisterClass {
    members: [RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15],
};

const GPR32: RegisterClass = RegisterClass {
    members: [EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI],
};

const FPR128: RegisterClass = RegisterClass {
    members: [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15],
};

// Flags are a dependency resource, never an allocatable register operand.
const EFLAGS: Register = Register {
    bits: 64, id: 32, encoding: 0, reserved: true, roles: [],
};
