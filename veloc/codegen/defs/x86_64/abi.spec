abi X86_64SystemV {
    arch = X86_64;
    stack = {
        align: 16,
        incoming: base(RBP, 16),
        outgoing: slot(8, 8),
    };
    args = { Integer: [RDI, RSI, RDX, RCX, R8, R9], Float: [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7], Vector: [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7] };
    returns = { Integer: [RAX, RDX], Float: [XMM0, XMM1], Vector: [XMM0, XMM1] };
    preserved = { gpr: [RBX, RBP, R12, R13, R14, R15] };
    classifier = x86_64_sysv_classifier;
}
abi X86_64WindowsFastcall {
    arch = X86_64;
    stack = {
        align: 16,
        outgoing: slot(8, 8),
    };
    args = { Integer: [RCX, RDX, R8, R9], Float: [XMM0, XMM1, XMM2, XMM3], Vector: [XMM0, XMM1, XMM2, XMM3] };
    returns = { Integer: [RAX, RDX], Float: [XMM0], Vector: [XMM0] };
    preserved = { gpr: [RBX, RBP, RDI, RSI, R12, R13, R14, R15] };
    classifier = x86_64_win64_classifier;
}
