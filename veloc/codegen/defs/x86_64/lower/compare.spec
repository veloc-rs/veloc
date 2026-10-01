import "../common.spec";

template GprCompare(Opcode: ident, Byte: expr, Wide: expr, Mnemonic: expr, Bits: expr) {
    op Opcode(lhs: Value<GprValue>, rhs: Value<GprValue>) -> () {
        encoding = legacy_rr(Byte, Wide, rhs, lhs);
        registers = {
            lhs: GPR64,
            rhs: GPR64,
        };
        implicit = {
            clobbers: [EFLAGS],
        };
        schedule = "IntAlu";
        assembly = {
            lines: [{ mnemonic: Mnemonic, operands: [reg(lhs, Bits), reg(rhs, Bits)] }]
        };
    }
}

expand GprCompare(X86Cmp32, 0x39, false, "cmp", 32);

op X86Cmp32ri(src: Value<GprValue>, imm: i64) -> () {
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0x81, wide: false },
        Form::ModRm(RegField::Extension(7), Rm::Register(src)),
        Immediate::Bits32(imm),
    );
    registers = {
        src: GPR64,
    };
    implicit = {
        clobbers: [EFLAGS],
    };
    schedule = "IntAlu";
    assembly = {
        lines: [{ mnemonic: "cmp", operands: [reg(src, 32), imm(imm)] }]
    };
}

expand GprCompare(X86Test32, 0x85, false, "test", 32);

template FloatCompare(Opcode: ident, Prefix: expr, Ty: expr, Mnemonic: expr) {
    op Opcode(lhs: Value<Ty>, rhs: Value<Ty>) -> () {
        encoding = Emission::legacy(
            Legacy { prefix: Prefix, map: OpcodeMap::Map0F, opcode: 0x2E, wide: false },
            Form::ModRm(RegField::Register(lhs), Rm::Register(rhs)),
            Immediate::None,
        );
        registers = {
            lhs: FPR128,
            rhs: FPR128,
        };
        implicit = {
            clobbers: [EFLAGS],
        };
        assembly = {
            lines: [{ mnemonic: Mnemonic, operands: [reg(lhs, 128), reg(rhs, 128)] }]
        };
    }
}

expand FloatCompare(X86Ucomiss, Prefix::None, Type::F32, "ucomiss");

expand GprCompare(X86Cmp64, 0x39, true, "cmp", 64);

expand GprCompare(X86Test64, 0x85, true, "test", 64);

expand FloatCompare(X86Ucomisd, Prefix::P66, Type::F64, "ucomisd");

template SetCondition(Opcode: ident, Byte: expr, Mnemonic: expr) {
    op Opcode() -> (dst: Value<GprValue>) {
        implicit = { reads: [EFLAGS] };
        encoding = Emission::legacy(
            Legacy { prefix: Prefix::None, map: OpcodeMap::Map0F, opcode: Byte, wide: false },
            Form::ModRm(RegField::Extension(0), Rm::ByteRegister(dst)),
            Immediate::None,
        );
        registers = {
            dst: GPR64,
        };
        assembly = {
            lines: [{ mnemonic: Mnemonic, operands: [reg(dst, 8)] }]
        };
    }
}

expand SetCondition(X86Sete, 0x94, "sete");

expand SetCondition(X86Setne, 0x95, "setne");

expand SetCondition(X86Setb, 0x92, "setb");

expand SetCondition(X86Seta, 0x97, "seta");

expand SetCondition(X86Setbe, 0x96, "setbe");

expand SetCondition(X86Setae, 0x93, "setae");

expand SetCondition(X86Setl, 0x9C, "setl");

expand SetCondition(X86Setg, 0x9F, "setg");

expand SetCondition(X86Setle, 0x9E, "setle");

expand SetCondition(X86Setge, 0x9D, "setge");

expand SetCondition(X86Setp, 0x9A, "setp");

expand SetCondition(X86Setnp, 0x9B, "setnp");

select(n: lir::Icmp) {
    choose {
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::E));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp32(n.lhs, n.rhs)), build(X86Sete(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::NE));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp32(n.lhs, n.rhs)), build(X86Setne(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::L));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp32(n.lhs, n.rhs)), build(X86Setl(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::LE));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp32(n.lhs, n.rhs)), build(X86Setle(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::G));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp32(n.lhs, n.rhs)), build(X86Setg(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::GE));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp32(n.lhs, n.rhs)), build(X86Setge(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::B));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp32(n.lhs, n.rhs)), build(X86Setb(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::BE));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp32(n.lhs, n.rhs)), build(X86Setbe(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::A));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp32(n.lhs, n.rhs)), build(X86Seta(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::AE));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp32(n.lhs, n.rhs)), build(X86Setae(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::E));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp64(n.lhs, n.rhs)), build(X86Sete(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::NE));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp64(n.lhs, n.rhs)), build(X86Setne(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::L));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp64(n.lhs, n.rhs)), build(X86Setl(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::LE));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp64(n.lhs, n.rhs)), build(X86Setle(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::G));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp64(n.lhs, n.rhs)), build(X86Setg(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::GE));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp64(n.lhs, n.rhs)), build(X86Setge(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::B));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp64(n.lhs, n.rhs)), build(X86Setb(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::BE));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp64(n.lhs, n.rhs)), build(X86Setbe(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::A));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp64(n.lhs, n.rhs)), build(X86Seta(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::AE));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp64(n.lhs, n.rhs)), build(X86Setae(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
    }
}

// Ordered comparisons exclude NaN; inequality includes unordered inputs.
template SelectFloatCompare(InputType: expr, Compare: ident, Condition: expr, Set: ident, Order: ident, Combine: ident) {
    select(n: lir::Fcmp) {
        choose {
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<InputType>(n.lhs));
            require(matches(n.cc, Condition));
            let bit = temp(Type::I8);
            let ordered_bit = temp(Type::I8);
            let predicate = temp(Type::I32);
            let ordered = temp(Type::I32);
            replace(n, [
                build(Compare(n.lhs, n.rhs)),
                build(Set(bit)),
                build(X86Movzx8to32(predicate, bit)),
                build(Order(ordered_bit)),
                build(X86Movzx8to32(ordered, ordered_bit)),
                build(Combine(n.dst, ordered, predicate)),
            ]);
        }
        }
    }
}

expand SelectFloatCompare(Type::F32, X86Ucomiss, CC::E, X86Sete, X86Setnp, X86And32);

expand SelectFloatCompare(Type::F32, X86Ucomiss, CC::NE, X86Setne, X86Setp, X86Or32);

expand SelectFloatCompare(Type::F32, X86Ucomiss, CC::B, X86Setb, X86Setnp, X86And32);

expand SelectFloatCompare(Type::F32, X86Ucomiss, CC::BE, X86Setbe, X86Setnp, X86And32);

expand SelectFloatCompare(Type::F64, X86Ucomisd, CC::E, X86Sete, X86Setnp, X86And32);

expand SelectFloatCompare(Type::F64, X86Ucomisd, CC::NE, X86Setne, X86Setp, X86Or32);

expand SelectFloatCompare(Type::F64, X86Ucomisd, CC::B, X86Setb, X86Setnp, X86And32);

expand SelectFloatCompare(Type::F64, X86Ucomisd, CC::BE, X86Setbe, X86Setnp, X86And32);

select(n: lir::Fcmp) {
    choose {
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<Type::F32>(n.lhs));
            require(matches(n.cc, CC::A));
            let bit = temp(Type::I8);
            replace(n, [build(X86Ucomiss(n.lhs, n.rhs)), build(X86Seta(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<Type::F32>(n.lhs));
            require(matches(n.cc, CC::AE));
            let bit = temp(Type::I8);
            replace(n, [build(X86Ucomiss(n.lhs, n.rhs)), build(X86Setae(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<Type::F64>(n.lhs));
            require(matches(n.cc, CC::A));
            let bit = temp(Type::I8);
            replace(n, [build(X86Ucomisd(n.lhs, n.rhs)), build(X86Seta(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<Type::F64>(n.lhs));
            require(matches(n.cc, CC::AE));
            let bit = temp(Type::I8);
            replace(n, [build(X86Ucomisd(n.lhs, n.rhs)), build(X86Setae(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
    }
}

select(n: lir::Ieqz) {
    choose {
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.src));
            let bit = temp(Type::I8);
            replace(n, [build(X86Test32(n.src, n.src)), build(X86Sete(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.src));
            let bit = temp(Type::I8);
            replace(n, [build(X86Test64(n.src, n.src)), build(X86Sete(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
    }
}
