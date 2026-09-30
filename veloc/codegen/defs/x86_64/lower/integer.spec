import "../common.spec";

// Machine values retain their source type; register classes constrain placement.
// Each instruction owns its encoding; templates share static family structure.

template GprBinary(Opcode: ident, Byte: expr, Wide: expr) {
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
        schedule = { latency: 1 };
    }
}

expand GprBinary(X86Add32, 0x01, false);

expand Asm(X86Add32, "add", [reg(dst, 32), reg(src2, 32)]);

expand GprBinary(X86Sub32, 0x29, false);

expand Asm(X86Sub32, "sub", [reg(dst, 32), reg(src2, 32)]);

expand GprBinary(X86And32, 0x21, false);

expand Asm(X86And32, "and", [reg(dst, 32), reg(src2, 32)]);

expand GprBinary(X86Or32, 0x09, false);

expand Asm(X86Or32, "or", [reg(dst, 32), reg(src2, 32)]);

expand GprBinary(X86Xor32, 0x31, false);

expand Asm(X86Xor32, "xor", [reg(dst, 32), reg(src2, 32)]);

expand GprBinary(X86Add64, 0x01, true);

expand Asm(X86Add64, "add", [reg(dst, 64), reg(src2, 64)]);

expand GprBinary(X86Sub64, 0x29, true);

expand Asm(X86Sub64, "sub", [reg(dst, 64), reg(src2, 64)]);

template GprBinaryImm(Opcode: ident, Wide: expr, Extension: expr, Imm: ident) {
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
        schedule = { latency: 1 };
    }
}

expand GprBinaryImm(X86Add64ri, true, 0, Immediate::Signed32);

expand Asm(X86Add64ri, "add", [reg(dst, 64), imm(imm)]);

expand GprBinaryImm(X86Sub64ri, true, 5, Immediate::Signed32);

expand Asm(X86Sub64ri, "sub", [reg(dst, 64), imm(imm)]);

expand GprBinary(X86And64, 0x21, true);

expand Asm(X86And64, "and", [reg(dst, 64), reg(src2, 64)]);

expand GprBinaryImm(X86And32ri, false, 4, Immediate::Bits32);

expand Asm(X86And32ri, "and", [reg(dst, 32), imm(imm)]);

expand GprBinaryImm(X86And64ri, true, 4, Immediate::Signed32);

expand Asm(X86And64ri, "and", [reg(dst, 64), imm(imm)]);

expand GprBinary(X86Or64, 0x09, true);

expand Asm(X86Or64, "or", [reg(dst, 64), reg(src2, 64)]);

expand GprBinary(X86Xor64, 0x31, true);

expand Asm(X86Xor64, "xor", [reg(dst, 64), reg(src2, 64)]);

template GprMultiply(Opcode: ident, Wide: expr) {
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
        schedule = { latency: 3 };
    }
}

expand GprMultiply(X86IMul32, false);

expand Asm(X86IMul32, "imul", [reg(dst, 32), reg(src2, 32)]);

expand GprMultiply(X86IMul64, true);

expand Asm(X86IMul64, "imul", [reg(dst, 64), reg(src2, 64)]);

template GprShiftCl(Opcode: ident, Wide: expr, Extension: expr) {
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
    }
}

expand GprShiftCl(X86Rol32Cl, false, 0);

expand Asm(X86Rol32Cl, "rol", [reg(dst, 32), reg(count, 8)]);

expand GprShiftCl(X86Rol64Cl, true, 0);

expand Asm(X86Rol64Cl, "rol", [reg(dst, 64), reg(count, 8)]);

expand GprShiftCl(X86Ror32Cl, false, 1);

expand Asm(X86Ror32Cl, "ror", [reg(dst, 32), reg(count, 8)]);

expand GprShiftCl(X86Ror64Cl, true, 1);

expand Asm(X86Ror64Cl, "ror", [reg(dst, 64), reg(count, 8)]);

expand GprShiftCl(X86Shl32Cl, false, 4);

expand Asm(X86Shl32Cl, "shl", [reg(dst, 32), reg(count, 8)]);

expand GprShiftCl(X86Shl64Cl, true, 4);

expand Asm(X86Shl64Cl, "shl", [reg(dst, 64), reg(count, 8)]);

expand GprShiftCl(X86Shr32Cl, false, 5);

expand Asm(X86Shr32Cl, "shr", [reg(dst, 32), reg(count, 8)]);

expand GprShiftCl(X86Shr64Cl, true, 5);

expand Asm(X86Shr64Cl, "shr", [reg(dst, 64), reg(count, 8)]);

expand GprShiftCl(X86Sar32Cl, false, 7);

expand Asm(X86Sar32Cl, "sar", [reg(dst, 32), reg(count, 8)]);

expand GprShiftCl(X86Sar64Cl, true, 7);

expand Asm(X86Sar64Cl, "sar", [reg(dst, 64), reg(count, 8)]);

template GprShiftImm(Opcode: ident, Wide: expr, Extension: expr) {
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
    }
}

expand GprShiftImm(X86Shl32ri, false, 4);

expand Asm(X86Shl32ri, "shl", [reg(dst, 32), imm(imm)]);

expand GprShiftImm(X86Shl64ri, true, 4);

expand Asm(X86Shl64ri, "shl", [reg(dst, 64), imm(imm)]);

expand GprShiftImm(X86Sar32ri, false, 7);

expand Asm(X86Sar32ri, "sar", [reg(dst, 32), imm(imm)]);

expand GprShiftImm(X86Sar64ri, true, 7);

expand Asm(X86Sar64ri, "sar", [reg(dst, 64), imm(imm)]);

template Divide32(Opcode: ident, Extension: expr) {
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
    }
}

expand Divide32(X86IDiv32, 7);

expand Asm(X86IDiv32, "idiv", [reg(src, 32)]);

template Divide64(Opcode: ident, Extension: expr) {
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
    }
}

expand Divide64(X86IDiv64, 7);

expand Asm(X86IDiv64, "idiv", [reg(src, 64)]);

expand Divide32(X86Div32, 6);

expand Asm(X86Div32, "div", [reg(src, 32)]);

expand Divide64(X86Div64, 6);

expand Asm(X86Div64, "div", [reg(src, 64)]);

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
}

expand Asm(X86Cqo, "cqo", []);

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
}

expand Asm(X86Cdq, "cdq", []);

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
