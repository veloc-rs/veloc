import "../common.spec";

// Machine values retain their source type; register classes constrain placement.
// Each instruction owns its encoding; templates share static family structure.

template GprBinary(Opcode: ident, Byte: expr, Wide: expr, Mnemonic: expr, Bits: expr) {
    op Opcode(src2: Value<GprValue>, src1: Value<GprValue>) -> (dst: Value<GprValue>, cf: Value<CARRY>, pf: Value<PARITY>, zf: Value<ZERO>, sf: Value<SIGN>, of: Value<OVERFLOW>) {
        encoding = legacy_rr(Byte, Wide, src2, dst);
        registers = {
            dst: tied(src1, GPR64),
            src2: GPR64,
            src1: GPR64,
        };
        clobbers = [AF];
        rematerializable = true;
        schedule = IntAlu;
        movable = true;
        assembly = {
            lines: [{ mnemonic: Mnemonic, operands: [reg(dst, Bits), reg(src2, Bits)] }]
        };
    }
}

expand GprBinary(X86Add32, 0x01, false, "add", 32);

expand GprBinary(X86Sub32, 0x29, false, "sub", 32);

expand GprBinary(X86And32, 0x21, false, "and", 32);

expand GprBinary(X86Or32, 0x09, false, "or", 32);

expand GprBinary(X86Xor32, 0x31, false, "xor", 32);

expand GprBinary(X86Add64, 0x01, true, "add", 64);

expand GprBinary(X86Sub64, 0x29, true, "sub", 64);

template GprBinaryImm(Opcode: ident, Wide: expr, Extension: expr, Imm: ident, Mnemonic: expr, Bits: expr) {
    op Opcode(imm: i64, src: Value<GprValue>) -> (dst: Value<GprValue>, cf: Value<CARRY>, pf: Value<PARITY>, zf: Value<ZERO>, sf: Value<SIGN>, of: Value<OVERFLOW>) {
        encoding = Emission::legacy(
            Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0x81, wide: Wide },
            Form::ModRm(RegField::Extension(Extension), Rm::Register(dst)),
            Imm(imm),
        );
        registers = {
            dst: tied(src, GPR64),
            src: GPR64,
        };
        clobbers = [AF];
        rematerializable = true;
        schedule = IntAlu;
        movable = true;
        assembly = {
            lines: [{ mnemonic: Mnemonic, operands: [reg(dst, Bits), imm(imm)] }]
        };
    }
}

expand GprBinaryImm(X86Add64ri, true, 0, Immediate::Signed32, "add", 64);

expand GprBinaryImm(X86Sub64ri, true, 5, Immediate::Signed32, "sub", 64);

expand GprBinary(X86And64, 0x21, true, "and", 64);

expand GprBinaryImm(X86And32ri, false, 4, Immediate::Bits32, "and", 32);

expand GprBinaryImm(X86And64ri, true, 4, Immediate::Signed32, "and", 64);

expand GprBinary(X86Or64, 0x09, true, "or", 64);

expand GprBinary(X86Xor64, 0x31, true, "xor", 64);

template GprMultiply(Opcode: ident, Wide: expr, Mnemonic: expr, Bits: expr) {
    op Opcode(src2: Value<GprValue>, src1: Value<GprValue>) -> (dst: Value<GprValue>, cf: Value<CARRY>, of: Value<OVERFLOW>) {
        encoding = Emission::legacy(
            Legacy { prefix: Prefix::None, map: OpcodeMap::Map0F, opcode: 0xAF, wide: Wide },
            Form::ModRm(RegField::Register(dst), Rm::Register(src2)),
            Immediate::None,
        );
        registers = {
            dst: tied(src1, GPR64),
            src2: GPR64,
            src1: GPR64,
        };
        clobbers = [PF, ZF, SF, AF];
        rematerializable = true;
        schedule = IntMul;
        movable = true;
        assembly = {
            lines: [{ mnemonic: Mnemonic, operands: [reg(dst, Bits), reg(src2, Bits)] }]
        };
    }
}

expand GprMultiply(X86IMul32, false, "imul", 32);

expand GprMultiply(X86IMul64, true, "imul", 64);

template GprShiftCl(Opcode: ident, Wide: expr, Extension: expr, Mnemonic: expr, Bits: expr) {
    op Opcode(count: Value<GprValue>, src1: Value<GprValue>) -> (dst: Value<GprValue>) {
        schedule = IntShift;
        encoding = Emission::legacy(
            Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0xD3, wide: Wide },
            Form::ModRm(RegField::Extension(Extension), Rm::Register(dst)),
            Immediate::None,
        );
        registers = {
            dst: tied(src1, GPR64),
            count: fixed(RCX, GPR64),
            src1: GPR64,
        };
        clobbers = [CF, PF, ZF, SF, OF, AF];
        assembly = {
            lines: [{ mnemonic: Mnemonic, operands: [reg(dst, Bits), reg(count, 8)] }]
        };
    }
}

expand GprShiftCl(X86Rol32Cl, false, 0, "rol", 32);

expand GprShiftCl(X86Rol64Cl, true, 0, "rol", 64);

expand GprShiftCl(X86Ror32Cl, false, 1, "ror", 32);

expand GprShiftCl(X86Ror64Cl, true, 1, "ror", 64);

expand GprShiftCl(X86Shl32Cl, false, 4, "shl", 32);

expand GprShiftCl(X86Shl64Cl, true, 4, "shl", 64);

expand GprShiftCl(X86Shr32Cl, false, 5, "shr", 32);

expand GprShiftCl(X86Shr64Cl, true, 5, "shr", 64);

expand GprShiftCl(X86Sar32Cl, false, 7, "sar", 32);

expand GprShiftCl(X86Sar64Cl, true, 7, "sar", 64);

template GprShiftImm(Opcode: ident, Wide: expr, Extension: expr, Mnemonic: expr, Bits: expr) {
    op Opcode(imm: i64, src: Value<GprValue>) -> (dst: Value<GprValue>) {
        schedule = IntAlu;
        encoding = Emission::legacy(
            Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0xC1, wide: Wide },
            Form::ModRm(RegField::Extension(Extension), Rm::Register(dst)),
            Immediate::Bits8(imm),
        );
        registers = {
            dst: tied(src, GPR64),
            src: GPR64,
        };
        clobbers = [CF, PF, ZF, SF, OF, AF];
        assembly = {
            lines: [{ mnemonic: Mnemonic, operands: [reg(dst, Bits), imm(imm)] }]
        };
    }
}

expand GprShiftImm(X86Shl32ri, false, 4, "shl", 32);

expand GprShiftImm(X86Shl64ri, true, 4, "shl", 64);

expand GprShiftImm(X86Sar32ri, false, 7, "sar", 32);

expand GprShiftImm(X86Sar64ri, true, 7, "sar", 64);

// Hardware register operands remain SSA values until allocation. The low/high
// inputs and quotient/remainder results occupy the same roots at different times.
template Divide(Opcode: ident, Ty: expr, Wide: expr, Bits: expr, Extension: expr, Mnemonic: expr, Scheduling: ident) {
    op Opcode(low: Value<Ty>, high: Value<Ty>, divisor: Value<Ty>) -> (quotient: Value<Ty>, remainder: Value<Ty>) {
        schedule = Scheduling;
        encoding = Emission::legacy(
            Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0xF7, wide: Wide },
            Form::ModRm(RegField::Extension(Extension), Rm::Register(divisor)),
            Immediate::None,
        );
        registers = {
            low: fixed(RAX, GPR64),
            high: fixed(RDX, GPR64),
            divisor: GPR64,
            quotient: fixed(RAX, GPR64),
            remainder: fixed(RDX, GPR64),
        };
        clobbers = [CF, PF, ZF, SF, OF, AF];
        assembly = {
            lines: [{ mnemonic: Mnemonic, operands: [reg(divisor, Bits)] }]
        };
    }
}

expand Divide(X86IDiv32, Type::I32, false, 32, 7, "idiv", IntDiv32);
expand Divide(X86IDiv64, Type::I64, true, 64, 7, "idiv", IntDiv64);
expand Divide(X86Div32, Type::I32, false, 32, 6, "div", IntDiv32);
expand Divide(X86Div64, Type::I64, true, 64, 6, "div", IntDiv64);

template SignExtendDividend(Opcode: ident, Ty: expr, Wide: expr, Mnemonic: expr) {
    op Opcode(low: Value<Ty>) -> (high: Value<Ty>) {
        schedule = IntAlu;
        encoding = Emission::legacy(
            Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0x99, wide: Wide },
            Form::None,
            Immediate::None,
        );
        registers = {
            low: fixed(RAX, GPR64),
            high: fixed(RDX, GPR64),
        };
        assembly = {
            lines: [{ mnemonic: Mnemonic, operands: [] }]
        };
    }
}

expand SignExtendDividend(X86Cqo, Type::I64, true, "cqo");
expand SignExtendDividend(X86Cdq, Type::I32, false, "cdq");

select(n: lir::PtrAdd) {
    choose {
        case {
            require(type_is<Type::I64>(n.lhs));
            require(type_is<Type::I64>(n.rhs));
            replace(n, build(X86Add64(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::PTR>(n.lhs));
            require(type_is<Type::I64>(n.rhs));
            replace(n, build(X86Add64(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::PTR>(n.dst));
            require(type_is<Type::PTR>(n.lhs));
            require(type_is<Type::I64>(n.rhs));
            replace(n, build(X86Add64(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Add) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(X86Add32(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86Add64(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Sub) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(X86Sub32(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86Sub64(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Mul) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(X86IMul32(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86IMul64(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::And) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(X86And32(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86And64(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            replace(n, build(X86And32(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Or) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(X86Or32(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86Or64(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            replace(n, build(X86Or32(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Xor) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(X86Xor32(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86Xor64(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            replace(n, build(X86Xor32(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Shl) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(X86Shl32Cl(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86Shl64Cl(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Lshr) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(X86Shr32Cl(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86Shr64Cl(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Ashr) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(X86Sar32Cl(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86Sar64Cl(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Sdiv) {
    choose {
        case {
            require(type_is<Type::I64>(n.rhs));
            let high = temp(Type::I64);
            let remainder = temp(Type::I64);
            replace(n, [build(X86Cqo(high, n.lhs)), build(X86IDiv64(n.dst, remainder, n.lhs, high, n.rhs))]);
        }
        case {
            require(type_is<Type::I32>(n.rhs));
            let high = temp(Type::I32);
            let remainder = temp(Type::I32);
            replace(n, [build(X86Cdq(high, n.lhs)), build(X86IDiv32(n.dst, remainder, n.lhs, high, n.rhs))]);
        }
    }
}

select(n: lir::Srem) {
    choose {
        case {
            require(type_is<Type::I64>(n.rhs));
            let high = temp(Type::I64);
            let quotient = temp(Type::I64);
            replace(n, [build(X86Cqo(high, n.lhs)), build(X86IDiv64(quotient, n.dst, n.lhs, high, n.rhs))]);
        }
        case {
            require(type_is<Type::I32>(n.rhs));
            let high = temp(Type::I32);
            let quotient = temp(Type::I32);
            replace(n, [build(X86Cdq(high, n.lhs)), build(X86IDiv32(quotient, n.dst, n.lhs, high, n.rhs))]);
        }
    }
}

select(n: lir::Udiv) {
    choose {
        case {
            require(type_is<Type::I64>(n.rhs));
            let high = temp(Type::I64);
            let remainder = temp(Type::I64);
            // A 32-bit zero write also defines the full 64-bit high word.
            replace(n, [build(X86Mov32Imm(high, 0)), build(X86Div64(n.dst, remainder, n.lhs, high, n.rhs))]);
        }
        case {
            require(type_is<Type::I32>(n.rhs));
            let high = temp(Type::I32);
            let remainder = temp(Type::I32);
            replace(n, [build(X86Mov32Imm(high, 0)), build(X86Div32(n.dst, remainder, n.lhs, high, n.rhs))]);
        }
    }
}

select(n: lir::Urem) {
    choose {
        case {
            require(type_is<Type::I64>(n.rhs));
            let high = temp(Type::I64);
            let quotient = temp(Type::I64);
            replace(n, [build(X86Mov32Imm(high, 0)), build(X86Div64(quotient, n.dst, n.lhs, high, n.rhs))]);
        }
        case {
            require(type_is<Type::I32>(n.rhs));
            let high = temp(Type::I32);
            let quotient = temp(Type::I32);
            replace(n, [build(X86Mov32Imm(high, 0)), build(X86Div32(quotient, n.dst, n.lhs, high, n.rhs))]);
        }
    }
}

select(n: lir::Rotl) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(X86Rol32Cl(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86Rol64Cl(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Rotr) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(X86Ror32Cl(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86Ror64Cl(n.rhs, n.lhs)));
        }
    }
}
