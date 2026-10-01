import "../common.spec";

template GprCompare(Opcode: ident, Byte: expr, Wide: expr, Mnemonic: expr, Bits: expr) {
    op Opcode(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (cf: Value<CARRY>, pf: Value<PARITY>, zf: Value<ZERO>, sf: Value<SIGN>, of: Value<OVERFLOW>) {
        encoding = legacy_rr(Byte, Wide, rhs, lhs);
        registers = {
            lhs: GPR64,
            rhs: GPR64,
        };
        clobbers = [AF];
        rematerializable = true;
        schedule = IntAlu;
        assembly = {
            lines: [{ mnemonic: Mnemonic, operands: [reg(lhs, Bits), reg(rhs, Bits)] }]
        };
    }
}

expand GprCompare(X86Cmp32, 0x39, false, "cmp", 32);

op X86Cmp32ri(src: Value<GprValue>, imm: i64) -> (cf: Value<CARRY>, pf: Value<PARITY>, zf: Value<ZERO>, sf: Value<SIGN>, of: Value<OVERFLOW>) {
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0x81, wide: false },
        Form::ModRm(RegField::Extension(7), Rm::Register(src)),
        Immediate::Bits32(imm),
    );
    registers = {
        src: GPR64,
    };
    clobbers = [AF];
        rematerializable = true;
    schedule = IntAlu;
    assembly = {
        lines: [{ mnemonic: "cmp", operands: [reg(src, 32), imm(imm)] }]
    };
}

expand GprCompare(X86Test32, 0x85, false, "test", 32);

template FloatCompare(Opcode: ident, Prefix: expr, Ty: expr, Mnemonic: expr) {
    op Opcode(lhs: Value<Ty>, rhs: Value<Ty>) -> (cf: Value<CARRY>, pf: Value<PARITY>, zf: Value<ZERO>, sf: Value<SIGN>, of: Value<OVERFLOW>) {
        encoding = Emission::legacy(
            Legacy { prefix: Prefix, map: OpcodeMap::Map0F, opcode: 0x2E, wide: false },
            Form::ModRm(RegField::Register(lhs), Rm::Register(rhs)),
            Immediate::None,
        );
        registers = {
            lhs: FPR128,
            rhs: FPR128,
        };
        clobbers = [AF];
        rematerializable = true;
        assembly = {
            lines: [{ mnemonic: Mnemonic, operands: [reg(lhs, 128), reg(rhs, 128)] }]
        };
    }
}

expand FloatCompare(X86Ucomiss, Prefix::None, Type::F32, "ucomiss");

expand GprCompare(X86Cmp64, 0x39, true, "cmp", 64);

expand GprCompare(X86Test64, 0x85, true, "test", 64);

expand FloatCompare(X86Ucomisd, Prefix::P66, Type::F64, "ucomisd");

template SetCondition1(Opcode: ident, Byte: expr, Mnemonic: expr, Flag0: type) {
    op Opcode(f0: Value<Flag0>) -> (dst: Value<GprValue>) {
        encoding = Emission::legacy(
            Legacy { prefix: Prefix::None, map: OpcodeMap::Map0F, opcode: Byte, wide: false },
            Form::ModRm(RegField::Extension(0), Rm::ByteRegister(dst)), Immediate::None,
        );
        registers = { dst: GPR64, };
        schedule = IntAlu;
        assembly = { lines: [{ mnemonic: Mnemonic, operands: [reg(dst, 8)] }] };
    }
}

template SetCondition2(Opcode: ident, Byte: expr, Mnemonic: expr, Flag0: type, Flag1: type) {
    op Opcode(f0: Value<Flag0>, f1: Value<Flag1>) -> (dst: Value<GprValue>) {
        encoding = Emission::legacy(
            Legacy { prefix: Prefix::None, map: OpcodeMap::Map0F, opcode: Byte, wide: false },
            Form::ModRm(RegField::Extension(0), Rm::ByteRegister(dst)), Immediate::None,
        );
        registers = { dst: GPR64, };
        schedule = IntAlu;
        assembly = { lines: [{ mnemonic: Mnemonic, operands: [reg(dst, 8)] }] };
    }
}

template SetCondition3(Opcode: ident, Byte: expr, Mnemonic: expr, Flag0: type, Flag1: type, Flag2: type) {
    op Opcode(f0: Value<Flag0>, f1: Value<Flag1>, f2: Value<Flag2>) -> (dst: Value<GprValue>) {
        encoding = Emission::legacy(
            Legacy { prefix: Prefix::None, map: OpcodeMap::Map0F, opcode: Byte, wide: false },
            Form::ModRm(RegField::Extension(0), Rm::ByteRegister(dst)), Immediate::None,
        );
        registers = { dst: GPR64, };
        schedule = IntAlu;
        assembly = { lines: [{ mnemonic: Mnemonic, operands: [reg(dst, 8)] }] };
    }
}

expand SetCondition1(X86Sete, 0x94, "sete", ZERO);

expand SetCondition1(X86Setne, 0x95, "setne", ZERO);

expand SetCondition1(X86Setb, 0x92, "setb", CARRY);

expand SetCondition2(X86Seta, 0x97, "seta", CARRY, ZERO);

expand SetCondition2(X86Setbe, 0x96, "setbe", CARRY, ZERO);

expand SetCondition1(X86Setae, 0x93, "setae", CARRY);

expand SetCondition2(X86Setl, 0x9C, "setl", SIGN, OVERFLOW);

expand SetCondition3(X86Setg, 0x9F, "setg", ZERO, SIGN, OVERFLOW);

expand SetCondition3(X86Setle, 0x9E, "setle", ZERO, SIGN, OVERFLOW);

expand SetCondition2(X86Setge, 0x9D, "setge", SIGN, OVERFLOW);

expand SetCondition1(X86Setp, 0x9A, "setp", PARITY);

expand SetCondition1(X86Setnp, 0x9B, "setnp", PARITY);

select(n: lir::Icmp) {
    choose {
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::E));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Cmp32(cf, pf, zf, sf, of, n.lhs, n.rhs)), build(X86Sete(bit, zf)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::NE));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Cmp32(cf, pf, zf, sf, of, n.lhs, n.rhs)), build(X86Setne(bit, zf)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::L));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Cmp32(cf, pf, zf, sf, of, n.lhs, n.rhs)), build(X86Setl(bit, sf, of)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::LE));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Cmp32(cf, pf, zf, sf, of, n.lhs, n.rhs)), build(X86Setle(bit, zf, sf, of)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::G));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Cmp32(cf, pf, zf, sf, of, n.lhs, n.rhs)), build(X86Setg(bit, zf, sf, of)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::GE));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Cmp32(cf, pf, zf, sf, of, n.lhs, n.rhs)), build(X86Setge(bit, sf, of)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::B));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Cmp32(cf, pf, zf, sf, of, n.lhs, n.rhs)), build(X86Setb(bit, cf)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::BE));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Cmp32(cf, pf, zf, sf, of, n.lhs, n.rhs)), build(X86Setbe(bit, cf, zf)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::A));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Cmp32(cf, pf, zf, sf, of, n.lhs, n.rhs)), build(X86Seta(bit, cf, zf)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::AE));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Cmp32(cf, pf, zf, sf, of, n.lhs, n.rhs)), build(X86Setae(bit, cf)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::E));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Cmp64(cf, pf, zf, sf, of, n.lhs, n.rhs)), build(X86Sete(bit, zf)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::NE));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Cmp64(cf, pf, zf, sf, of, n.lhs, n.rhs)), build(X86Setne(bit, zf)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::L));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Cmp64(cf, pf, zf, sf, of, n.lhs, n.rhs)), build(X86Setl(bit, sf, of)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::LE));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Cmp64(cf, pf, zf, sf, of, n.lhs, n.rhs)), build(X86Setle(bit, zf, sf, of)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::G));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Cmp64(cf, pf, zf, sf, of, n.lhs, n.rhs)), build(X86Setg(bit, zf, sf, of)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::GE));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Cmp64(cf, pf, zf, sf, of, n.lhs, n.rhs)), build(X86Setge(bit, sf, of)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::B));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Cmp64(cf, pf, zf, sf, of, n.lhs, n.rhs)), build(X86Setb(bit, cf)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::BE));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Cmp64(cf, pf, zf, sf, of, n.lhs, n.rhs)), build(X86Setbe(bit, cf, zf)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::A));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Cmp64(cf, pf, zf, sf, of, n.lhs, n.rhs)), build(X86Seta(bit, cf, zf)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::AE));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Cmp64(cf, pf, zf, sf, of, n.lhs, n.rhs)), build(X86Setae(bit, cf)), build(X86Movzx8to32(n.dst, bit))]);
        }
    }
}

// Ordered comparisons exclude NaN; inequality includes unordered inputs.
template SelectFloatCompareZf(InputType: expr, Compare: ident, Condition: expr, Set: ident, Order: ident, Combine: ident) {
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
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [
                build(Compare(cf, pf, zf, sf, of, n.lhs, n.rhs)),
                build(Set(bit, zf)),
                build(X86Movzx8to32(predicate, bit)),
                build(Order(ordered_bit, pf)),
                build(X86Movzx8to32(ordered, ordered_bit)),
                build(Combine(n.dst, ordered, predicate)),
            ]);
        }
        }
    }
}

template SelectFloatCompareCf(InputType: expr, Compare: ident, Condition: expr, Set: ident, Order: ident, Combine: ident) {
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
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [
                build(Compare(cf, pf, zf, sf, of, n.lhs, n.rhs)),
                build(Set(bit, cf)),
                build(X86Movzx8to32(predicate, bit)),
                build(Order(ordered_bit, pf)),
                build(X86Movzx8to32(ordered, ordered_bit)),
                build(Combine(n.dst, ordered, predicate)),
            ]);
        }
        }
    }
}

template SelectFloatCompareCfZf(InputType: expr, Compare: ident, Condition: expr, Set: ident, Order: ident, Combine: ident) {
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
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [
                build(Compare(cf, pf, zf, sf, of, n.lhs, n.rhs)),
                build(Set(bit, cf, zf)),
                build(X86Movzx8to32(predicate, bit)),
                build(Order(ordered_bit, pf)),
                build(X86Movzx8to32(ordered, ordered_bit)),
                build(Combine(n.dst, ordered, predicate)),
            ]);
        }
        }
    }
}

expand SelectFloatCompareZf(Type::F32, X86Ucomiss, CC::E, X86Sete, X86Setnp, X86And32);

expand SelectFloatCompareZf(Type::F32, X86Ucomiss, CC::NE, X86Setne, X86Setp, X86Or32);

expand SelectFloatCompareCf(Type::F32, X86Ucomiss, CC::B, X86Setb, X86Setnp, X86And32);

expand SelectFloatCompareCfZf(Type::F32, X86Ucomiss, CC::BE, X86Setbe, X86Setnp, X86And32);

expand SelectFloatCompareZf(Type::F64, X86Ucomisd, CC::E, X86Sete, X86Setnp, X86And32);

expand SelectFloatCompareZf(Type::F64, X86Ucomisd, CC::NE, X86Setne, X86Setp, X86Or32);

expand SelectFloatCompareCf(Type::F64, X86Ucomisd, CC::B, X86Setb, X86Setnp, X86And32);

expand SelectFloatCompareCfZf(Type::F64, X86Ucomisd, CC::BE, X86Setbe, X86Setnp, X86And32);

select(n: lir::Fcmp) {
    choose {
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<Type::F32>(n.lhs));
            require(matches(n.cc, CC::A));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Ucomiss(cf, pf, zf, sf, of, n.lhs, n.rhs)), build(X86Seta(bit, cf, zf)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<Type::F32>(n.lhs));
            require(matches(n.cc, CC::AE));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Ucomiss(cf, pf, zf, sf, of, n.lhs, n.rhs)), build(X86Setae(bit, cf)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<Type::F64>(n.lhs));
            require(matches(n.cc, CC::A));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Ucomisd(cf, pf, zf, sf, of, n.lhs, n.rhs)), build(X86Seta(bit, cf, zf)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<Type::F64>(n.lhs));
            require(matches(n.cc, CC::AE));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Ucomisd(cf, pf, zf, sf, of, n.lhs, n.rhs)), build(X86Setae(bit, cf)), build(X86Movzx8to32(n.dst, bit))]);
        }
    }
}

select(n: lir::Ieqz) {
    choose {
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.src));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Test32(cf, pf, zf, sf, of, n.src, n.src)), build(X86Sete(bit, zf)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.src));
            let bit = temp(Type::I8);
            let cf = temp(Type::BOOL);
            let pf = temp(Type::BOOL);
            let zf = temp(Type::BOOL);
            let sf = temp(Type::BOOL);
            let of = temp(Type::BOOL);
            replace(n, [build(X86Test64(cf, pf, zf, sf, of, n.src, n.src)), build(X86Sete(bit, zf)), build(X86Movzx8to32(n.dst, bit))]);
        }
    }
}
