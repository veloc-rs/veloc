import "../common.spec";

template GprExtend(Opcode: ident, Map: expr, Byte: expr, Wide: expr, RmKind: ident) {
    op Opcode(src: Value<GprValue>) -> (dst: Value<GprValue>) {
        encoding = Emission::legacy(
            Legacy { prefix: Prefix::None, map: Map, opcode: Byte, wide: Wide },
            Form::ModRm(RegField::Register(dst), RmKind(src)),
            Immediate::None,
        );
        registers = {
            dst: GPR64,
            src: GPR64,
        };
        schedule = { latency: 1 };
    }
}

template GprMove(Opcode: ident, Bits: expr, Wide: expr) {
    op Opcode(src: Value<GprValue>) -> (dst: Value<GprValue>) {
        encoding = legacy_rr(0x89, Wide, src, dst);
        registers = {
            dst: GPR64,
            src: GPR64,
        };
        schedule = { latency: 1 };
    }
}

expand GprMove(X86Mov32, 32, false);

expand Asm(X86Mov32, "mov", [reg(dst, 32), reg(src, 32)]);

expand GprMove(X86Mov64, 64, true);

expand Asm(X86Mov64, "mov", [reg(dst, 64), reg(src, 64)]);

template FloatMove(Opcode: ident, Bits: expr, Prefix: expr, Ty: expr) {
    op Opcode(src2: Value<Ty>) -> (dst: Value<Ty>) {
        encoding = Emission::legacy(
            Legacy { prefix: Prefix, map: OpcodeMap::Map0F, opcode: 0x10, wide: false },
            Form::ModRm(RegField::Register(dst), Rm::Register(src2)),
            Immediate::None,
        );
        registers = {
            dst: FPR128,
            src2: FPR128,
        };
        schedule = { latency: 1 };
    }
}

expand FloatMove(X86Movss, 32, Prefix::F3, Type::F32);

expand Asm(X86Movss, "movss", [reg(dst, 128), reg(src2, 128)]);

expand FloatMove(X86Movsd, 64, Prefix::F2, Type::F64);

expand Asm(X86Movsd, "movsd", [reg(dst, 128), reg(src2, 128)]);

template GprToXmmMove(Opcode: ident, Bits: expr, Wide: expr, FloatType: expr) {
    op Opcode(src: Value<GprValue>) -> (dst: Value<FloatType>) {
        encoding = Emission::legacy(
            Legacy { prefix: Prefix::P66, map: OpcodeMap::Map0F, opcode: 0x6E, wide: Wide },
            Form::ModRm(RegField::Register(dst), Rm::Register(src)),
            Immediate::None,
        );
        registers = {
            dst: FPR128,
            src: GPR64,
        };
        schedule = { latency: 1 };
    }
}

expand GprToXmmMove(X86MovdToXmm, 32, false, Type::F32);

expand Asm(X86MovdToXmm, "movd", [reg(dst, 128), reg(src, 32)]);

template XmmToGprMove(Opcode: ident, Bits: expr, Wide: expr, FloatType: expr) {
    op Opcode(src: Value<FloatType>) -> (dst: Value<GprValue>) {
        encoding = Emission::legacy(
            Legacy { prefix: Prefix::P66, map: OpcodeMap::Map0F, opcode: 0x7E, wide: Wide },
            Form::ModRm(RegField::Register(src), Rm::Register(dst)),
            Immediate::None,
        );
        registers = {
            dst: GPR64,
            src: FPR128,
        };
        schedule = { latency: 1 };
    }
}

expand XmmToGprMove(X86MovdFromXmm, 32, false, Type::F32);

expand Asm(X86MovdFromXmm, "movd", [reg(dst, 32), reg(src, 128)]);

expand GprToXmmMove(X86MovqToXmm, 64, true, Type::F64);

expand Asm(X86MovqToXmm, "movq", [reg(dst, 128), reg(src, 64)]);

expand XmmToGprMove(X86MovqFromXmm, 64, true, Type::F64);

expand Asm(X86MovqFromXmm, "movq", [reg(dst, 64), reg(src, 128)]);

expand GprExtend(X86Movzx8to32, OpcodeMap::Map0F, 0xB6, false, Rm::ByteRegister);

expand Asm(X86Movzx8to32, "movzx", [reg(dst, 32), reg(src, 8)]);

expand GprExtend(X86Movzx16to32, OpcodeMap::Map0F, 0xB7, false, Rm::Register);

expand Asm(X86Movzx16to32, "movzx", [reg(dst, 32), reg(src, 16)]);

expand GprExtend(X86Movsx8to32, OpcodeMap::Map0F, 0xBE, false, Rm::ByteRegister);

expand Asm(X86Movsx8to32, "movsx", [reg(dst, 32), reg(src, 8)]);

expand GprExtend(X86Movsx16to32, OpcodeMap::Map0F, 0xBF, false, Rm::Register);

expand Asm(X86Movsx16to32, "movsx", [reg(dst, 32), reg(src, 16)]);

expand GprExtend(X86Movsx8to64, OpcodeMap::Map0F, 0xBE, true, Rm::ByteRegister);

expand Asm(X86Movsx8to64, "movsx", [reg(dst, 64), reg(src, 8)]);

expand GprExtend(X86Movsx16to64, OpcodeMap::Map0F, 0xBF, true, Rm::Register);

expand Asm(X86Movsx16to64, "movsx", [reg(dst, 64), reg(src, 16)]);

expand GprExtend(X86Movsxd32to64, OpcodeMap::Primary, 0x63, true, Rm::Register);

expand Asm(X86Movsxd32to64, "movsxd", [reg(dst, 64), reg(src, 32)]);

op X86Mov32Imm(imm: i64) -> (dst: Value<GprValue>) {
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0xB8, wide: false },
        Form::OpcodeReg(dst),
        Immediate::Bits32(imm),
    );
    registers = {
        dst: GPR64,
    };
    schedule = { latency: 1 };
}

expand Asm(X86Mov32Imm, "mov", [reg(dst, 32), imm(imm)]);

op X86Mov64Imm32(imm: i64) -> (dst: Value<GprValue>) {
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0xC7, wide: true },
        Form::ModRm(RegField::Extension(0), Rm::Register(dst)),
        Immediate::Signed32(imm),
    );
    registers = {
        dst: GPR64,
    };
    schedule = { latency: 1 };
}

expand Asm(X86Mov64Imm32, "mov", [reg(dst, 64), imm(imm)]);

op X86Mov64Imm64(imm: i64) -> (dst: Value<GprValue>) {
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0xB8, wide: true },
        Form::OpcodeReg(dst),
        Immediate::Bits64(imm),
    );
    registers = {
        dst: GPR64,
    };
    schedule = { latency: 1 };
}

expand Asm(X86Mov64Imm64, "mov", [reg(dst, 64), imm(imm)]);

select(n: lir::Constant) {
    choose {
        case {
            require(type_is<SmallInt>(n.dst));
            replace(n, build(X86Mov32Imm(n.imm)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86Mov64Imm64(n.imm)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            replace(n, build(X86Mov32Imm(n.imm)));
        }
    }
}

select(n: lir::Copy) {
    choose {
        case {
            require(type_is<Type::BOOL | SmallInt>(n.dst));
            replace(n, build(X86Mov32(n.src)));
        }
        case {
            require(type_is<WordOrPtr>(n.dst));
            replace(n, build(X86Mov64(n.src)));
        }
        case {
            require(type_is<Type::F32>(n.dst));
            replace(n, build(X86Movss(n.src)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            replace(n, build(X86Movsd(n.src)));
        }
    }
}

select(n: lir::Inttoptr) {
    choose {
        case {
            require(type_is<Type::PTR>(n.dst));
            require(type_is<Type::I64>(n.src));
            replace(n, build(X86Mov64(n.src)));
        }
    }
}

select(n: lir::Ptrtoint) {
    choose {
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::PTR>(n.src));
            replace(n, build(X86Mov64(n.src)));
        }
    }
}

select(n: lir::Bitcast) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::I32>(n.src));
            replace(n, build(X86Mov32(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::I64>(n.src));
            replace(n, build(X86Mov64(n.src)));
        }
        case {
            require(type_is<Type::PTR>(n.dst));
            require(type_is<Type::PTR>(n.src));
            replace(n, build(X86Mov64(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::PTR>(n.src));
            replace(n, build(X86Mov64(n.src)));
        }
        case {
            require(type_is<Type::PTR>(n.dst));
            require(type_is<Type::I64>(n.src));
            replace(n, build(X86Mov64(n.src)));
        }
        case {
            require(type_is<Type::F32>(n.dst));
            require(type_is<Type::F32>(n.src));
            replace(n, build(X86Movss(n.src)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            require(type_is<Type::F64>(n.src));
            replace(n, build(X86Movsd(n.src)));
        }
        case {
            require(type_is<Type::F32>(n.dst));
            require(type_is<Type::I32>(n.src));
            replace(n, build(X86MovdToXmm(n.src)));
        }
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::F32>(n.src));
            replace(n, build(X86MovdFromXmm(n.src)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            require(type_is<Type::I64>(n.src));
            replace(n, build(X86MovqToXmm(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::F64>(n.src));
            replace(n, build(X86MovqFromXmm(n.src)));
        }
    }
}

select(n: lir::Trunc) {
    choose {
        case {
            replace(n, build(X86Mov32(n.src)));
        }
    }
}

select(n: lir::Zext) {
    choose {
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::I32>(n.src));
            replace(n, build(X86Mov32(n.src)));
        }
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::I8>(n.src));
            replace(n, build(X86Movzx8to32(n.src)));
        }
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::I16>(n.src));
            replace(n, build(X86Movzx16to32(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::I8>(n.src));
            replace(n, build(X86Movzx8to32(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::I16>(n.src));
            replace(n, build(X86Movzx16to32(n.src)));
        }
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::BOOL>(n.src));
            replace(n, build(X86Movzx8to32(n.dst, n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::BOOL>(n.src));
            replace(n, build(X86Movzx8to32(n.dst, n.src)));
        }
    }
}

select(n: lir::Sext) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::I8>(n.src));
            replace(n, build(X86Movsx8to32(n.src)));
        }
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::I16>(n.src));
            replace(n, build(X86Movsx16to32(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::I8>(n.src));
            replace(n, build(X86Movsx8to64(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::I16>(n.src));
            replace(n, build(X86Movsx16to64(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::I32>(n.src));
            replace(n, build(X86Movsxd32to64(n.src)));
        }
    }
}
