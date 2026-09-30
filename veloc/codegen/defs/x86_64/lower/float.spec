import "../common.spec";

template FloatBinary(Opcode: ident, Prefix: expr, Byte: expr, Ty: expr) {
    op Opcode(rhs: Value<Ty>, lhs: Value<Ty>) -> (dst: Value<Ty>) {
        encoding = Emission::legacy(
            Legacy { prefix: Prefix, map: OpcodeMap::Map0F, opcode: Byte, wide: false },
            Form::ModRm(RegField::Register(dst), Rm::Register(rhs)),
            Immediate::None,
        );
        registers = {
            dst: tied(lhs, FPR128),
            rhs: FPR128,
            lhs: FPR128,
        };
    }
}

expand FloatBinary(X86FAdd32, Prefix::F3, 0x58, Type::F32);

expand Asm(X86FAdd32, "addss", [reg(dst, 128), reg(rhs, 128)]);

expand FloatBinary(X86FAdd64, Prefix::F2, 0x58, Type::F64);

expand Asm(X86FAdd64, "addsd", [reg(dst, 128), reg(rhs, 128)]);

expand FloatBinary(X86FSub32, Prefix::F3, 0x5C, Type::F32);

expand Asm(X86FSub32, "subss", [reg(dst, 128), reg(rhs, 128)]);

expand FloatBinary(X86FSub64, Prefix::F2, 0x5C, Type::F64);

expand Asm(X86FSub64, "subsd", [reg(dst, 128), reg(rhs, 128)]);

expand FloatBinary(X86FMul32, Prefix::F3, 0x59, Type::F32);

expand Asm(X86FMul32, "mulss", [reg(dst, 128), reg(rhs, 128)]);

expand FloatBinary(X86FMul64, Prefix::F2, 0x59, Type::F64);

expand Asm(X86FMul64, "mulsd", [reg(dst, 128), reg(rhs, 128)]);

expand FloatBinary(X86FDiv32, Prefix::F3, 0x5E, Type::F32);

expand Asm(X86FDiv32, "divss", [reg(dst, 128), reg(rhs, 128)]);

expand FloatBinary(X86FDiv64, Prefix::F2, 0x5E, Type::F64);

expand Asm(X86FDiv64, "divsd", [reg(dst, 128), reg(rhs, 128)]);

template IntToFloat(Opcode: ident, Prefix: expr, Wide: expr, Src: expr, Dst: expr) {
    op Opcode(src: Value<Src>) -> (dst: Value<Dst>) {
        encoding = Emission::legacy(
            Legacy { prefix: Prefix, map: OpcodeMap::Map0F, opcode: 0x2A, wide: Wide },
            Form::ModRm(RegField::Register(dst), Rm::Register(src)),
            Immediate::None,
        );
        registers = {
            dst: FPR128,
            src: GPR64,
        };
        schedule = { latency: 4 };
    }
}

expand IntToFloat(X86I32ToF32, Prefix::F3, false, Type::I32, Type::F32);

expand Asm(X86I32ToF32, "cvtsi2ss", [reg(dst, 128), reg(src, 32)]);

expand IntToFloat(X86I64ToF32, Prefix::F3, true, Type::I64, Type::F32);

expand Asm(X86I64ToF32, "cvtsi2ss", [reg(dst, 128), reg(src, 64)]);

expand IntToFloat(X86I32ToF64, Prefix::F2, false, Type::I32, Type::F64);

expand Asm(X86I32ToF64, "cvtsi2sd", [reg(dst, 128), reg(src, 32)]);

expand IntToFloat(X86I64ToF64, Prefix::F2, true, Type::I64, Type::F64);

expand Asm(X86I64ToF64, "cvtsi2sd", [reg(dst, 128), reg(src, 64)]);

template FloatToInt(Opcode: ident, Prefix: expr, Wide: expr, Src: expr, Dst: expr) {
    op Opcode(src: Value<Src>) -> (dst: Value<Dst>) {
        encoding = Emission::legacy(
            Legacy { prefix: Prefix, map: OpcodeMap::Map0F, opcode: 0x2C, wide: Wide },
            Form::ModRm(RegField::Register(dst), Rm::Register(src)),
            Immediate::None,
        );
        registers = {
            dst: GPR64,
            src: FPR128,
        };
        schedule = { latency: 4 };
    }
}

expand FloatToInt(X86F32ToI32, Prefix::F3, false, Type::F32, Type::I32);

expand Asm(X86F32ToI32, "cvttss2si", [reg(dst, 32), reg(src, 128)]);

expand FloatToInt(X86F32ToI64, Prefix::F3, true, Type::F32, Type::I64);

expand Asm(X86F32ToI64, "cvttss2si", [reg(dst, 64), reg(src, 128)]);

expand FloatToInt(X86F64ToI32, Prefix::F2, false, Type::F64, Type::I32);

expand Asm(X86F64ToI32, "cvttsd2si", [reg(dst, 32), reg(src, 128)]);

expand FloatToInt(X86F64ToI64, Prefix::F2, true, Type::F64, Type::I64);

expand Asm(X86F64ToI64, "cvttsd2si", [reg(dst, 64), reg(src, 128)]);

template FloatUnary(Opcode: ident, Prefix: expr, Byte: expr, Src: expr, Dst: expr) {
    op Opcode(src: Value<Src>) -> (dst: Value<Dst>) {
        encoding = Emission::legacy(
            Legacy { prefix: Prefix, map: OpcodeMap::Map0F, opcode: Byte, wide: false },
            Form::ModRm(RegField::Register(dst), Rm::Register(src)),
            Immediate::None,
        );
        registers = {
            dst: FPR128,
            src: FPR128,
        };
        schedule = { latency: 4 };
    }
}

expand FloatUnary(X86F32ToF64, Prefix::F3, 0x5A, Type::F32, Type::F64);

expand Asm(X86F32ToF64, "cvtss2sd", [reg(dst, 128), reg(src, 128)]);

expand FloatUnary(X86F64ToF32, Prefix::F2, 0x5A, Type::F64, Type::F32);

expand Asm(X86F64ToF32, "cvtsd2ss", [reg(dst, 128), reg(src, 128)]);

expand FloatUnary(X86SqrtF32, Prefix::F3, 0x51, Type::F32, Type::F32);

expand Asm(X86SqrtF32, "sqrtss", [reg(dst, 128), reg(src, 128)]);

expand FloatUnary(X86SqrtF64, Prefix::F2, 0x51, Type::F64, Type::F64);

expand Asm(X86SqrtF64, "sqrtsd", [reg(dst, 128), reg(src, 128)]);

select(n: lir::Fadd) {
    choose {
        case {
            require(type_is<Type::F32>(n.dst));
            replace(n, build(X86FAdd32(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            replace(n, build(X86FAdd64(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Fsub) {
    choose {
        case {
            require(type_is<Type::F32>(n.dst));
            replace(n, build(X86FSub32(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            replace(n, build(X86FSub64(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Fmul) {
    choose {
        case {
            require(type_is<Type::F32>(n.dst));
            replace(n, build(X86FMul32(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            replace(n, build(X86FMul64(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Fdiv) {
    choose {
        case {
            require(type_is<Type::F32>(n.dst));
            replace(n, build(X86FDiv32(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            replace(n, build(X86FDiv64(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Sitofp) {
    choose {
        case {
            require(type_is<Type::F32>(n.dst));
            require(type_is<Type::I32>(n.src));
            replace(n, build(X86I32ToF32(n.src)));
        }
        case {
            require(type_is<Type::F32>(n.dst));
            require(type_is<Type::I64>(n.src));
            replace(n, build(X86I64ToF32(n.src)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            require(type_is<Type::I32>(n.src));
            replace(n, build(X86I32ToF64(n.src)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            require(type_is<Type::I64>(n.src));
            replace(n, build(X86I64ToF64(n.src)));
        }
    }
}

select(n: lir::Fptosi) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::F32>(n.src));
            replace(n, build(X86F32ToI32(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::F32>(n.src));
            replace(n, build(X86F32ToI64(n.src)));
        }
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::F64>(n.src));
            replace(n, build(X86F64ToI32(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::F64>(n.src));
            replace(n, build(X86F64ToI64(n.src)));
        }
    }
}

select(n: lir::Fpext) {
    choose {
        case {
            require(type_is<Type::F64>(n.dst));
            require(type_is<Type::F32>(n.src));
            replace(n, build(X86F32ToF64(n.src)));
        }
    }
}

select(n: lir::Fptrunc) {
    choose {
        case {
            require(type_is<Type::F32>(n.dst));
            require(type_is<Type::F64>(n.src));
            replace(n, build(X86F64ToF32(n.src)));
        }
    }
}

select(n: lir::Fsqrt) {
    choose {
        case {
            require(type_is<Type::F32>(n.dst));
            replace(n, build(X86SqrtF32(n.src)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            replace(n, build(X86SqrtF64(n.src)));
        }
    }
}
