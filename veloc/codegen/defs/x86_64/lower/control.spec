import "../common.spec";

op X86Call(sp: Value<AddressValue>, target: Global, info: CallInfo) -> () {
    schedule = Call;
    encoding = Emission::relative(target, Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0xE8, wide: false }, Form::None, 0);
    registers = { sp: fixed(RSP, GPR64) };
    clobbers = [CF, PF, ZF, SF, OF, AF];
    flow = Call;
    assembly = {
        lines: [{ mnemonic: "call", operands: [target(target)] }]
    };
}

op X86CallReg(sp: Value<AddressValue>, target: Value<AddressValue>, info: CallInfo) -> () {
    schedule = Call;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0xFF, wide: false },
        Form::ModRm(RegField::Extension(2), Rm::Register(target)),
        Immediate::None,
    );
    registers = {
        sp: fixed(RSP, GPR64),
        target: GPR64,
    };
    clobbers = [CF, PF, ZF, SF, OF, AF];
    flow = Call;
    assembly = {
        lines: [{ mnemonic: "call", operands: [reg(target, 64)] }]
    };
}

op X86Ret() -> () {
    schedule = Return;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0xC3, wide: false },
        Form::None,
        Immediate::None,
    );
    flow = Return;
    assembly = {
        lines: [{ mnemonic: "ret", operands: [] }]
    };
}

op X86Ud2() -> () {
    schedule = Trap;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Map0F, opcode: 0xB, wide: false },
        Form::None,
        Immediate::None,
    );
    flow = Trap;
    assembly = {
        lines: [{ mnemonic: "ud2", operands: [] }]
    };
}

op X86Jmp(target: Successor) -> () {
    schedule = Branch;
    encoding = Emission::branch(
        target,
        Branch { map: OpcodeMap::Primary, near: 0xE9, short: 0xEB },
    );
    flow = Jump;
    assembly = {
        lines: [{ mnemonic: "jmp", operands: [target(target)] }]
    };
}

template ConditionalBranch1(Opcode: ident, Near: expr, Short: expr, Mnemonic: expr, Flag0: type) {
    op Opcode(f0: Value<Flag0>, target: Successor) -> () {
        schedule = Branch;
        encoding = Emission::branch(target, Branch { map: OpcodeMap::Map0F, near: Near, short: Short });
        flow = Branch;
        assembly = { lines: [{ mnemonic: Mnemonic, operands: [target(target)] }] };
    }
}

template ConditionalBranch2(Opcode: ident, Near: expr, Short: expr, Mnemonic: expr, Flag0: type, Flag1: type) {
    op Opcode(f0: Value<Flag0>, f1: Value<Flag1>, target: Successor) -> () {
        schedule = Branch;
        encoding = Emission::branch(target, Branch { map: OpcodeMap::Map0F, near: Near, short: Short });
        flow = Branch;
        assembly = { lines: [{ mnemonic: Mnemonic, operands: [target(target)] }] };
    }
}

template ConditionalBranch3(Opcode: ident, Near: expr, Short: expr, Mnemonic: expr, Flag0: type, Flag1: type, Flag2: type) {
    op Opcode(f0: Value<Flag0>, f1: Value<Flag1>, f2: Value<Flag2>, target: Successor) -> () {
        schedule = Branch;
        encoding = Emission::branch(target, Branch { map: OpcodeMap::Map0F, near: Near, short: Short });
        flow = Branch;
        assembly = { lines: [{ mnemonic: Mnemonic, operands: [target(target)] }] };
    }
}

expand ConditionalBranch1(X86Je, 0x84, 0x74, "je", ZERO);

expand ConditionalBranch1(X86Jne, 0x85, 0x75, "jne", ZERO);

expand ConditionalBranch1(X86Jb, 0x82, 0x72, "jb", CARRY);

expand ConditionalBranch1(X86Jae, 0x83, 0x73, "jae", CARRY);

expand ConditionalBranch2(X86Jbe, 0x86, 0x76, "jbe", CARRY, ZERO);

expand ConditionalBranch2(X86Ja, 0x87, 0x77, "ja", CARRY, ZERO);

expand ConditionalBranch2(X86Jl, 0x8C, 0x7C, "jl", SIGN, OVERFLOW);

expand ConditionalBranch2(X86Jge, 0x8D, 0x7D, "jge", SIGN, OVERFLOW);

expand ConditionalBranch3(X86Jle, 0x8E, 0x7E, "jle", ZERO, SIGN, OVERFLOW);

expand ConditionalBranch3(X86Jg, 0x8F, 0x7F, "jg", ZERO, SIGN, OVERFLOW);

op X86PushRbp(rbp: Value<GprValue>) -> () {
    schedule = Push;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0x50, wide: false },
        Form::OpcodeReg(rbp),
        Immediate::None,
    );
    registers = {
        rbp: fixed(RBP, GPR64),
    };
    assembly = {
        lines: [{ mnemonic: "push", operands: [reg(rbp, 64)] }]
    };
}

op X86PopRbp() -> (rbp: Value<GprValue>) {
    schedule = Pop;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0x58, wide: false },
        Form::OpcodeReg(rbp),
        Immediate::None,
    );
    registers = {
        rbp: fixed(RBP, GPR64),
    };
    assembly = {
        lines: [{ mnemonic: "pop", operands: [reg(rbp, 64)] }]
    };
}

op X86MovRbpRsp(rsp: Value<GprValue>) -> (rbp: Value<GprValue>) {
    schedule = Copy;
    encoding = legacy_rr(0x89, true, rsp, rbp);
    registers = {
        rbp: fixed(RBP, GPR64),
        rsp: fixed(RSP, GPR64),
    };
    assembly = {
        lines: [{ mnemonic: "mov", operands: [reg(rbp, 64), reg(rsp, 64)] }]
    };
}

select(n: lir::Ret) {
    replace(n, build(X86Ret()));
}

select(n: lir::Call) {
    choose {
        case {
            replace(n, build(X86Call(reg(RSP), n.callee, n.info)));
        }
    }
}

select(n: lir::Callind) {
    choose {
        case {
            replace(n, build(X86CallReg(reg(RSP), n.callee, n.info)));
        }
    }
}

// Select a value using false ^ ((true ^ false) & -bool(cond)).
template Select32(ResultType: expr, CondType: expr, Test: ident) {
    select(n: lir::Select) {
        choose {
        case {
            require(type_is<ResultType>(n.dst));
            require(type_is<CondType>(n.cond));
            let cond_byte = temp(Type::I8);
            let cond32 = temp(Type::I32);
            let zero = temp(Type::I32);
            let mask = temp(Type::I32);
            let diff = temp(Type::I32);
            let masked = temp(Type::I32);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [
                build(Test(cf, pf, zf, sf, of, n.cond, n.cond)),
                build(X86Setne(cond_byte, zf)),
                build(X86Movzx8to32(cond32, cond_byte)),
                build(X86Mov32Imm(zero, 0)),
                build(X86Sub32(mask, cond32, zero)),
                build(X86Xor32(diff, n.v1, n.v2)),
                build(X86And32(masked, diff, mask)),
                build(X86Xor32(n.dst, n.v2, masked)),
            ]);
        }
        }
    }
}

template Select64(ResultType: expr, CondType: expr, Test: ident) {
    select(n: lir::Select) {
        choose {
        case {
            require(type_is<ResultType>(n.dst));
            require(type_is<CondType>(n.cond));
            let cond_byte = temp(Type::I8);
            let cond32 = temp(Type::I32);
            let zero = temp(Type::I64);
            let mask = temp(Type::I64);
            let diff = temp(Type::I64);
            let masked = temp(Type::I64);
            let wide = temp(Type::I64);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [
                build(Test(cf, pf, zf, sf, of, n.cond, n.cond)),
                build(X86Setne(cond_byte, zf)),
                build(X86Movzx8to32(cond32, cond_byte)),
                build(X86Mov32(wide, cond32)),
                build(X86Mov64Imm32(zero, 0)),
                build(X86Sub64(mask, wide, zero)),
                build(X86Xor64(diff, n.v1, n.v2)),
                build(X86And64(masked, diff, mask)),
                build(X86Xor64(n.dst, n.v2, masked)),
            ]);
        }
        }
    }
}

template SelectF32(ResultType: expr, CondType: expr, Test: ident) {
    select(n: lir::Select) {
        choose {
        case {
            require(type_is<ResultType>(n.dst));
            require(type_is<CondType>(n.cond));
            let cond_byte = temp(Type::I8);
            let cond32 = temp(Type::I32);
            let zero = temp(Type::I32);
            let mask = temp(Type::I32);
            let diff = temp(Type::I32);
            let masked = temp(Type::I32);
            let true_bits = temp(Type::I32);
            let false_bits = temp(Type::I32);
            let result_bits = temp(Type::I32);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [
                build(Test(cf, pf, zf, sf, of, n.cond, n.cond)),
                build(X86Setne(cond_byte, zf)),
                build(X86Movzx8to32(cond32, cond_byte)),
                build(X86MovdFromXmm(true_bits, n.v1)),
                build(X86MovdFromXmm(false_bits, n.v2)),
                build(X86Mov32Imm(zero, 0)),
                build(X86Sub32(mask, cond32, zero)),
                build(X86Xor32(diff, true_bits, false_bits)),
                build(X86And32(masked, diff, mask)),
                build(X86Xor32(result_bits, false_bits, masked)),
                build(X86MovdToXmm(n.dst, result_bits)),
            ]);
        }
        }
    }
}

template SelectF64(ResultType: expr, CondType: expr, Test: ident) {
    select(n: lir::Select) {
        choose {
        case {
            require(type_is<ResultType>(n.dst));
            require(type_is<CondType>(n.cond));
            let cond_byte = temp(Type::I8);
            let cond32 = temp(Type::I32);
            let zero = temp(Type::I64);
            let mask = temp(Type::I64);
            let diff = temp(Type::I64);
            let masked = temp(Type::I64);
            let wide = temp(Type::I64);
            let true_bits = temp(Type::I64);
            let false_bits = temp(Type::I64);
            let result_bits = temp(Type::I64);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [
                build(Test(cf, pf, zf, sf, of, n.cond, n.cond)),
                build(X86Setne(cond_byte, zf)),
                build(X86Movzx8to32(cond32, cond_byte)),
                build(X86Mov32(wide, cond32)),
                build(X86MovqFromXmm(true_bits, n.v1)),
                build(X86MovqFromXmm(false_bits, n.v2)),
                build(X86Mov64Imm32(zero, 0)),
                build(X86Sub64(mask, wide, zero)),
                build(X86Xor64(diff, true_bits, false_bits)),
                build(X86And64(masked, diff, mask)),
                build(X86Xor64(result_bits, false_bits, masked)),
                build(X86MovqToXmm(n.dst, result_bits)),
            ]);
        }
        }
    }
}

expand Select32(SmallInt | Type::BOOL, SmallInt | Type::BOOL, X86Test32);

expand Select32(SmallInt | Type::BOOL, WordOrPtr, X86Test64);

expand Select64(WordOrPtr, SmallInt | Type::BOOL, X86Test32);

expand Select64(WordOrPtr, WordOrPtr, X86Test64);

expand SelectF32(Type::F32, SmallInt | Type::BOOL, X86Test32);

expand SelectF32(Type::F32, WordOrPtr, X86Test64);

expand SelectF64(Type::F64, SmallInt | Type::BOOL, X86Test32);

expand SelectF64(Type::F64, WordOrPtr, X86Test64);

select(n: lir::Br) {
    choose {
        case {
            replace(n, build(X86Jmp(n.target)));
        }
    }
}

template IntBranchZf(InputType: expr, Compare: ident, Condition: expr, Jump: ident) {
    select(n: lir::Brcond) {
        choose {
            case {
                let cmp = def<lir::Icmp<InputType>>(n.cond);
                require(matches(cmp.cc, Condition));
                let cf = temp(Type::BOOL);
                let pf = temp(Type::BOOL);
                let zf = temp(Type::BOOL);
                let sf = temp(Type::BOOL);
                let of = temp(Type::BOOL);
                replace(n, [build(Compare(cf, pf, zf, sf, of, cmp.lhs, cmp.rhs)), build(Jump(zf, n.then_blk)), build(X86Jmp(n.else_blk))]);
            }
        }
    }
}

template IntBranchCf(InputType: expr, Compare: ident, Condition: expr, Jump: ident) {
    select(n: lir::Brcond) {
        choose {
            case {
                let cmp = def<lir::Icmp<InputType>>(n.cond);
                require(matches(cmp.cc, Condition));
                let cf = temp(Type::BOOL);
                let pf = temp(Type::BOOL);
                let zf = temp(Type::BOOL);
                let sf = temp(Type::BOOL);
                let of = temp(Type::BOOL);
                replace(n, [build(Compare(cf, pf, zf, sf, of, cmp.lhs, cmp.rhs)), build(Jump(cf, n.then_blk)), build(X86Jmp(n.else_blk))]);
            }
        }
    }
}

template IntBranchCfZf(InputType: expr, Compare: ident, Condition: expr, Jump: ident) {
    select(n: lir::Brcond) {
        choose {
            case {
                let cmp = def<lir::Icmp<InputType>>(n.cond);
                require(matches(cmp.cc, Condition));
                let cf = temp(Type::BOOL);
                let pf = temp(Type::BOOL);
                let zf = temp(Type::BOOL);
                let sf = temp(Type::BOOL);
                let of = temp(Type::BOOL);
                replace(n, [build(Compare(cf, pf, zf, sf, of, cmp.lhs, cmp.rhs)), build(Jump(cf, zf, n.then_blk)), build(X86Jmp(n.else_blk))]);
            }
        }
    }
}

template IntBranchSfOf(InputType: expr, Compare: ident, Condition: expr, Jump: ident) {
    select(n: lir::Brcond) {
        choose {
            case {
                let cmp = def<lir::Icmp<InputType>>(n.cond);
                require(matches(cmp.cc, Condition));
                let cf = temp(Type::BOOL);
                let pf = temp(Type::BOOL);
                let zf = temp(Type::BOOL);
                let sf = temp(Type::BOOL);
                let of = temp(Type::BOOL);
                replace(n, [build(Compare(cf, pf, zf, sf, of, cmp.lhs, cmp.rhs)), build(Jump(sf, of, n.then_blk)), build(X86Jmp(n.else_blk))]);
            }
        }
    }
}

template IntBranchZfSfOf(InputType: expr, Compare: ident, Condition: expr, Jump: ident) {
    select(n: lir::Brcond) {
        choose {
            case {
                let cmp = def<lir::Icmp<InputType>>(n.cond);
                require(matches(cmp.cc, Condition));
                let cf = temp(Type::BOOL);
                let pf = temp(Type::BOOL);
                let zf = temp(Type::BOOL);
                let sf = temp(Type::BOOL);
                let of = temp(Type::BOOL);
                replace(n, [build(Compare(cf, pf, zf, sf, of, cmp.lhs, cmp.rhs)), build(Jump(zf, sf, of, n.then_blk)), build(X86Jmp(n.else_blk))]);
            }
        }
    }
}

expand IntBranchZf(Type::I8, X86Cmp32, CC::E, X86Je);

expand IntBranchZf(Type::I8, X86Cmp32, CC::NE, X86Jne);

expand IntBranchSfOf(Type::I8, X86Cmp32, CC::L, X86Jl);

expand IntBranchZfSfOf(Type::I8, X86Cmp32, CC::LE, X86Jle);

expand IntBranchZfSfOf(Type::I8, X86Cmp32, CC::G, X86Jg);

expand IntBranchSfOf(Type::I8, X86Cmp32, CC::GE, X86Jge);

expand IntBranchCf(Type::I8, X86Cmp32, CC::B, X86Jb);

expand IntBranchCfZf(Type::I8, X86Cmp32, CC::BE, X86Jbe);

expand IntBranchCfZf(Type::I8, X86Cmp32, CC::A, X86Ja);

expand IntBranchCf(Type::I8, X86Cmp32, CC::AE, X86Jae);

expand IntBranchZf(Type::I16, X86Cmp32, CC::E, X86Je);

expand IntBranchZf(Type::I16, X86Cmp32, CC::NE, X86Jne);

expand IntBranchSfOf(Type::I16, X86Cmp32, CC::L, X86Jl);

expand IntBranchZfSfOf(Type::I16, X86Cmp32, CC::LE, X86Jle);

expand IntBranchZfSfOf(Type::I16, X86Cmp32, CC::G, X86Jg);

expand IntBranchSfOf(Type::I16, X86Cmp32, CC::GE, X86Jge);

expand IntBranchCf(Type::I16, X86Cmp32, CC::B, X86Jb);

expand IntBranchCfZf(Type::I16, X86Cmp32, CC::BE, X86Jbe);

expand IntBranchCfZf(Type::I16, X86Cmp32, CC::A, X86Ja);

expand IntBranchCf(Type::I16, X86Cmp32, CC::AE, X86Jae);

expand IntBranchZf(Type::I32, X86Cmp32, CC::E, X86Je);

expand IntBranchZf(Type::I32, X86Cmp32, CC::NE, X86Jne);

expand IntBranchSfOf(Type::I32, X86Cmp32, CC::L, X86Jl);

expand IntBranchZfSfOf(Type::I32, X86Cmp32, CC::LE, X86Jle);

expand IntBranchZfSfOf(Type::I32, X86Cmp32, CC::G, X86Jg);

expand IntBranchSfOf(Type::I32, X86Cmp32, CC::GE, X86Jge);

expand IntBranchCf(Type::I32, X86Cmp32, CC::B, X86Jb);

expand IntBranchCfZf(Type::I32, X86Cmp32, CC::BE, X86Jbe);

expand IntBranchCfZf(Type::I32, X86Cmp32, CC::A, X86Ja);

expand IntBranchCf(Type::I32, X86Cmp32, CC::AE, X86Jae);

expand IntBranchZf(Type::I64, X86Cmp64, CC::E, X86Je);

expand IntBranchZf(Type::I64, X86Cmp64, CC::NE, X86Jne);

expand IntBranchSfOf(Type::I64, X86Cmp64, CC::L, X86Jl);

expand IntBranchZfSfOf(Type::I64, X86Cmp64, CC::LE, X86Jle);

expand IntBranchZfSfOf(Type::I64, X86Cmp64, CC::G, X86Jg);

expand IntBranchSfOf(Type::I64, X86Cmp64, CC::GE, X86Jge);

expand IntBranchCf(Type::I64, X86Cmp64, CC::B, X86Jb);

expand IntBranchCfZf(Type::I64, X86Cmp64, CC::BE, X86Jbe);

expand IntBranchCfZf(Type::I64, X86Cmp64, CC::A, X86Ja);

expand IntBranchCf(Type::I64, X86Cmp64, CC::AE, X86Jae);

expand IntBranchZf(Type::PTR, X86Cmp64, CC::E, X86Je);

expand IntBranchZf(Type::PTR, X86Cmp64, CC::NE, X86Jne);

expand IntBranchSfOf(Type::PTR, X86Cmp64, CC::L, X86Jl);

expand IntBranchZfSfOf(Type::PTR, X86Cmp64, CC::LE, X86Jle);

expand IntBranchZfSfOf(Type::PTR, X86Cmp64, CC::G, X86Jg);

expand IntBranchSfOf(Type::PTR, X86Cmp64, CC::GE, X86Jge);

expand IntBranchCf(Type::PTR, X86Cmp64, CC::B, X86Jb);

expand IntBranchCfZf(Type::PTR, X86Cmp64, CC::BE, X86Jbe);

expand IntBranchCfZf(Type::PTR, X86Cmp64, CC::A, X86Ja);

expand IntBranchCf(Type::PTR, X86Cmp64, CC::AE, X86Jae);

select(n: lir::Brcond) {
    choose {
        case {
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Test32(cf, pf, zf, sf, of, n.cond, n.cond)), build(X86Jne(zf, n.then_blk)), build(X86Jmp(n.else_blk))]);
        }
    }
}

select(n: lir::Ret) {
    choose {
        case {
            replace(n, build(X86Ret()));
        }
    }
}

select(n: lir::Trap) {
    choose {
        case {
            replace(n, build(X86Ud2()));
        }
    }
}
