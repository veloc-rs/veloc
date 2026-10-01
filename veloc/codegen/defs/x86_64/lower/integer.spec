import "../common.spec";

// Machine values retain their source type; register classes constrain placement.
// Each instruction owns its encoding; templates share static family structure.

template GprBinary(Opcode: ident, Byte: expr, Wide: expr, Mnemonic: expr, Bits: expr) {
    op Opcode(src2: Value<GprValue>, src1: Value<GprValue>) -> (dst: Value<GprValue>) {
        encoding = legacy_rr(Byte, Wide, src2, dst);
        registers = {
            dst: tied(src1, GPR64),
            src2: GPR64,
            src1: GPR64,
        };
        implicit = {
            clobbers: [EFLAGS],
        };
        schedule = "IntAlu";
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
    op Opcode(imm: i64, src: Value<GprValue>) -> (dst: Value<GprValue>) {
        encoding = Emission::legacy(
            Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0x81, wide: Wide },
            Form::ModRm(RegField::Extension(Extension), Rm::Register(dst)),
            Imm(imm),
        );
        registers = {
            dst: tied(src, GPR64),
            src: GPR64,
        };
        implicit = {
            clobbers: [EFLAGS],
        };
        schedule = "IntAlu";
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
    op Opcode(src2: Value<GprValue>, src1: Value<GprValue>) -> (dst: Value<GprValue>) {
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
        implicit = {
            clobbers: [EFLAGS],
        };
        schedule = "IntMul";
        assembly = {
            lines: [{ mnemonic: Mnemonic, operands: [reg(dst, Bits), reg(src2, Bits)] }]
        };
    }
}

expand GprMultiply(X86IMul32, false, "imul", 32);

expand GprMultiply(X86IMul64, true, "imul", 64);

template GprShiftCl(Opcode: ident, Wide: expr, Extension: expr, Mnemonic: expr, Bits: expr) {
    op Opcode(count: Value<GprValue>, src1: Value<GprValue>) -> (dst: Value<GprValue>) {
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
        implicit = {
            clobbers: [EFLAGS],
        };
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
        encoding = Emission::legacy(
            Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0xC1, wide: Wide },
            Form::ModRm(RegField::Extension(Extension), Rm::Register(dst)),
            Immediate::Bits8(imm),
        );
        registers = {
            dst: tied(src, GPR64),
            src: GPR64,
        };
        implicit = {
            clobbers: [EFLAGS],
        };
        assembly = {
            lines: [{ mnemonic: Mnemonic, operands: [reg(dst, Bits), imm(imm)] }]
        };
    }
}

expand GprShiftImm(X86Shl32ri, false, 4, "shl", 32);

expand GprShiftImm(X86Shl64ri, true, 4, "shl", 64);

expand GprShiftImm(X86Sar32ri, false, 7, "sar", 32);

expand GprShiftImm(X86Sar64ri, true, 7, "sar", 64);

template Divide32(Opcode: ident, Extension: expr, Mnemonic: expr) {
    op Opcode(src: Value<GprValue>) -> () {
        encoding = Emission::legacy(
            Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0xF7, wide: false },
            Form::ModRm(RegField::Extension(Extension), Rm::Register(src)),
            Immediate::None,
        );
        registers = {
            src: GPR64,
        };
        implicit = {
            reads: [EAX, EDX],
            writes: [EAX, EDX],
            clobbers: [EFLAGS],
        };
        assembly = {
            lines: [{ mnemonic: Mnemonic, operands: [reg(src, 32)] }]
        };
    }
}

expand Divide32(X86IDiv32, 7, "idiv");

template Divide64(Opcode: ident, Extension: expr, Mnemonic: expr) {
    op Opcode(src: Value<GprValue>) -> () {
        encoding = Emission::legacy(
            Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0xF7, wide: true },
            Form::ModRm(RegField::Extension(Extension), Rm::Register(src)),
            Immediate::None,
        );
        registers = {
            src: GPR64,
        };
        implicit = {
            reads: [RAX, RDX],
            writes: [RAX, RDX],
            clobbers: [EFLAGS],
        };
        assembly = {
            lines: [{ mnemonic: Mnemonic, operands: [reg(src, 64)] }]
        };
    }
}

expand Divide64(X86IDiv64, 7, "idiv");

expand Divide32(X86Div32, 6, "div");

expand Divide64(X86Div64, 6, "div");

op X86Cqo() -> () {
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0x99, wide: true },
        Form::None,
        Immediate::None,
    );
    implicit = {
        reads: [RAX],
        writes: [RDX],
    };
    assembly = {
        lines: [{ mnemonic: "cqo", operands: [] }]
    };
}

op X86Cdq() -> () {
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0x99, wide: false },
        Form::None,
        Immediate::None,
    );
    implicit = {
        reads: [EAX],
        writes: [EDX],
    };
    assembly = {
        lines: [{ mnemonic: "cdq", operands: [] }]
    };
}

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
            replace(n, [build(X86Mov64(reg(RAX), n.lhs)), build(X86Cqo()), build(X86IDiv64(n.rhs)), build(X86Mov64(n.dst, reg(RAX)))]);
        }
        case {
            require(type_is<Type::I32>(n.rhs));
            replace(n, [build(X86Mov32(reg(RAX), n.lhs)), build(X86Cdq()), build(X86IDiv32(n.rhs)), build(X86Mov32(n.dst, reg(RAX)))]);
        }
    }
}

select(n: lir::Srem) {
    choose {
        case {
            require(type_is<Type::I64>(n.rhs));
            replace(n, [build(X86Mov64(reg(RAX), n.lhs)), build(X86Cqo()), build(X86IDiv64(n.rhs)), build(X86Mov64(n.dst, reg(RDX)))]);
        }
        case {
            require(type_is<Type::I32>(n.rhs));
            replace(n, [build(X86Mov32(reg(RAX), n.lhs)), build(X86Cdq()), build(X86IDiv32(n.rhs)), build(X86Mov32(n.dst, reg(RDX)))]);
        }
    }
}

select(n: lir::Udiv) {
    choose {
        case {
            require(type_is<Type::I64>(n.rhs));
            replace(n, [build(X86Mov64(reg(RAX), n.lhs)), build(X86Xor64(reg(RDX), reg(RDX), reg(RDX))), build(X86Div64(n.rhs)), build(X86Mov64(n.dst, reg(RAX)))]);
        }
        case {
            require(type_is<Type::I32>(n.rhs));
            replace(n, [build(X86Mov32(reg(RAX), n.lhs)), build(X86Xor32(reg(RDX), reg(RDX), reg(RDX))), build(X86Div32(n.rhs)), build(X86Mov32(n.dst, reg(RAX)))]);
        }
    }
}

select(n: lir::Urem) {
    choose {
        case {
            require(type_is<Type::I64>(n.rhs));
            replace(n, [build(X86Mov64(reg(RAX), n.lhs)), build(X86Xor64(reg(RDX), reg(RDX), reg(RDX))), build(X86Div64(n.rhs)), build(X86Mov64(n.dst, reg(RDX)))]);
        }
        case {
            require(type_is<Type::I32>(n.rhs));
            replace(n, [build(X86Mov32(reg(RAX), n.lhs)), build(X86Xor32(reg(RDX), reg(RDX), reg(RDX))), build(X86Div32(n.rhs)), build(X86Mov32(n.dst, reg(RDX)))]);
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
