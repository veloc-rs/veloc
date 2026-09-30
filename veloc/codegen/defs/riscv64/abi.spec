import "../../../defs/type_sets.spec";
import "registers.spec";
typeset AbiWord = Type::BOOL | ScalarInteger | Type::PTR;
abi Rv64Lp64d {
    arch = Riscv64;
    stack = { align: 16 };
    args = [
        assign(ScalarFloat, [F10,F11,F12,F13,F14,F15,F16,F17]),
        assign(AbiWord | ScalarFloat, [X10,X11,X12,X13,X14,X15,X16,X17]),
        stack(AbiWord | ScalarFloat, 8, 8),
    ];
    returns = [assign(ScalarFloat, [F10,F11]), assign(AbiWord, [X10,X11])];
    preserved = [X8, X9, X18, X19, X20, X21, X22, X23, X24, X25, X26, X27, F8, F9, F18, F19, F20, F21, F22, F23, F24, F25, F26, F27];
}
