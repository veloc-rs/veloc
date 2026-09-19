import "../../../defs/type_sets.spec";

// Ordered actions: exhausted register lists fall through to the next rule.
// Domains reuse ordinary Spec typesets; there is no ABI classifier registry.
typeset AbiWord = Type::BOOL | Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::PTR;
typeset AbiFloat = Type::F32 | Type::F64;
typeset AbiVector = Type::I8X16 | Type::I16X8 | Type::I32X4 | Type::I64X2 | Type::F32X4 | Type::F64X2;

abi X86_64SystemV {
    arch = X86_64;
    stack = { align: 16 };
    args = [
        assign(AbiWord, [RDI, RSI, RDX, RCX, R8, R9]),
        assign(AbiFloat | AbiVector, [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7]),
        stack(AbiWord | AbiFloat, 8, 8),
        stack(AbiVector, 16, 16),
    ];
    returns = [
        assign(AbiWord, [RAX, RDX]),
        assign(AbiFloat | AbiVector, [XMM0, XMM1]),
    ];
    preserved = [RBX, RBP, R12, R13, R14, R15];
}

abi X86_64WindowsFastcall {
    arch = X86_64;
    stack = { align: 16, reserved: 32 };
    args = [
        shadow(AbiWord, [RCX, RDX, R8, R9], [XMM0, XMM1, XMM2, XMM3]),
        shadow(AbiFloat, [XMM0, XMM1, XMM2, XMM3], [RCX, RDX, R8, R9]),
        stack(AbiWord | AbiFloat, 8, 8),
    ];
    returns = [
        assign(AbiWord, [RAX]),
        assign(AbiFloat | AbiVector, [XMM0]),
    ];
    preserved = [
        RBX, RBP, RDI, RSI, R12, R13, R14, R15,
        XMM6, XMM7, XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15,
    ];
}
