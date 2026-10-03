import "../common.spec";

expand Binary(RvMinSigned, GprValue, GPR, 51, 4, 5, "min", Zbb, IntAlu, true);
expand Binary(RvMaxSigned, GprValue, GPR, 51, 6, 5, "max", Zbb, IntAlu, true);
expand Binary(RvMinUnsigned, GprValue, GPR, 51, 5, 5, "minu", Zbb, IntAlu, true);
expand Binary(RvMaxUnsigned, GprValue, GPR, 51, 7, 5, "maxu", Zbb, IntAlu, true);

// RV64's canonical sign-extended i32 representation preserves both signed
// and unsigned ordering. Selecting either operand also preserves that form.
template MinMaxSelect(Condition: expr, Direct: ident, Reversed: ident) {
    select(n: lir::Select<Type::I32 | Type::I64>) {
        choose {
            case {
                let cmp = def<lir::Icmp>(n.cond);
                require(matches(cmp.cc, Condition));
                require(same_value(n.v1, cmp.lhs));
                require(same_value(n.v2, cmp.rhs));
                replace(n, build(Direct(n.v1, n.v2)));
            }
            case {
                let cmp = def<lir::Icmp>(n.cond);
                require(matches(cmp.cc, Condition));
                require(same_value(n.v1, cmp.rhs));
                require(same_value(n.v2, cmp.lhs));
                replace(n, build(Reversed(n.v1, n.v2)));
            }
        }
    }
}
expand MinMaxSelect(CC::L, RvMinSigned, RvMaxSigned);
expand MinMaxSelect(CC::LE, RvMinSigned, RvMaxSigned);
expand MinMaxSelect(CC::G, RvMaxSigned, RvMinSigned);
expand MinMaxSelect(CC::GE, RvMaxSigned, RvMinSigned);
expand MinMaxSelect(CC::B, RvMinUnsigned, RvMaxUnsigned);
expand MinMaxSelect(CC::BE, RvMinUnsigned, RvMaxUnsigned);
expand MinMaxSelect(CC::A, RvMaxUnsigned, RvMinUnsigned);
expand MinMaxSelect(CC::AE, RvMaxUnsigned, RvMinUnsigned);

expand Binary(RvAndNot, GprValue, GPR, 51, 7, 32, "andn", Zbb, IntAlu, true);
expand Binary(RvOrNot, GprValue, GPR, 51, 6, 32, "orn", Zbb, IntAlu, true);
expand Binary(RvXnor, GprValue, GPR, 51, 4, 32, "xnor", Zbb, IntAlu, true);

// Fuse either operand's complement. Shared producers retain their other uses;
// the selector removes the original XOR only when all results are dead.
template ComplementBinary(Source: ident, Target: ident) {
    select(n: Source<Type::I32 | Type::I64>) {
        choose {
            case {
                let inverted = def<lir::Xor>(n.rhs);
                let mask = def<lir::Constant>(inverted.rhs);
                require(matches(mask.imm, -1));
                replace(n, build(Target(n.lhs, inverted.lhs)));
            }
            case {
                let inverted = def<lir::Xor>(n.rhs);
                let mask = def<lir::Constant>(inverted.lhs);
                require(matches(mask.imm, -1));
                replace(n, build(Target(n.lhs, inverted.rhs)));
            }
            case {
                let inverted = def<lir::Xor>(n.lhs);
                let mask = def<lir::Constant>(inverted.rhs);
                require(matches(mask.imm, -1));
                replace(n, build(Target(n.rhs, inverted.lhs)));
            }
            case {
                let inverted = def<lir::Xor>(n.lhs);
                let mask = def<lir::Constant>(inverted.lhs);
                require(matches(mask.imm, -1));
                replace(n, build(Target(n.rhs, inverted.rhs)));
            }
        }
    }
}

expand ComplementBinary(lir::And, RvAndNot);
expand ComplementBinary(lir::Or, RvOrNot);
expand ComplementBinary(lir::Xor, RvXnor);

expand Binary(RvRol32, GprValue, GPR, 59, 1, 48, "rolw", Zbb, IntAlu, true);

expand Binary(RvRor32, GprValue, GPR, 59, 5, 48, "rorw", Zbb, IntAlu, true);

expand Binary(RvRol64, GprValue, GPR, 51, 1, 48, "rol", Zbb, IntAlu, true);

expand Binary(RvRor64, GprValue, GPR, 51, 5, 48, "ror", Zbb, IntAlu, true);

op RvSext8Zbb(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::I(19,dst,1,src,1540)]);
    registers = { dst: GPR, src: GPR };
    schedule = IntAlu;
    movable = true;
    requires = [Zbb];
    assembly = {
        lines: [{ mnemonic: "sext.b", operands: [reg(dst,64),reg(src,64)] }]
    };
}

op RvSext16Zbb(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::I(19,dst,1,src,1541)]);
    registers = { dst: GPR, src: GPR };
    schedule = IntAlu;
    movable = true;
    requires = [Zbb];
    assembly = {
        lines: [{ mnemonic: "sext.h", operands: [reg(dst,64),reg(src,64)] }]
    };
}

op RvZext16Zbb(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::R(59,dst,4,src,Reg::X0,4)]);
    registers = { dst: GPR, src: GPR };
    schedule = IntAlu;
    movable = true;
    requires = [Zbb];
    assembly = {
        lines: [{ mnemonic: "zext.h", operands: [reg(dst,64),reg(src,64)] }]
    };
}

// Frontends also express narrow casts as masks or paired shifts. Recognize
// these value-preserving forms before the generic single-op shift rules.
select(n: lir::And<Type::I32 | Type::I64>) {
    choose {
        case {
            let mask = def<lir::Constant>(n.rhs);
            require(matches(mask.imm, 65535));
            replace(n, build(RvZext16Zbb(n.lhs)));
        }
        case {
            let mask = def<lir::Constant>(n.lhs);
            require(matches(mask.imm, 65535));
            replace(n, build(RvZext16Zbb(n.rhs)));
        }
    }
}

template SignExtendShifts(Ty: type, Amount: expr, Target: ident) {
    select(n: lir::Ashr<Ty>) {
        let right = def<lir::Constant>(n.rhs);
        require(matches(right.imm, Amount));
        let shift = def<lir::Shl<Ty>>(n.lhs);
        let left = def<lir::Constant>(shift.rhs);
        require(matches(left.imm, Amount));
        replace(n, build(Target(shift.lhs)));
    }
}
expand SignExtendShifts(Type::I32, 24, RvSext8Zbb);
expand SignExtendShifts(Type::I32, 16, RvSext16Zbb);
expand SignExtendShifts(Type::I64, 56, RvSext8Zbb);
expand SignExtendShifts(Type::I64, 48, RvSext16Zbb);

select(n: lir::Rotl<Type::I32>) {
    replace(n, build(RvRol32(n.lhs, n.rhs)));
}

select(n: lir::Rotl<Type::I64>) {
    replace(n, build(RvRol64(n.lhs, n.rhs)));
}

select(n: lir::Rotr<Type::I32>) {
    replace(n, build(RvRor32(n.lhs, n.rhs)));
}

select(n: lir::Rotr<Type::I64>) {
    replace(n, build(RvRor64(n.lhs, n.rhs)));
}

select(n: lir::Sext) {
    require(type_is<Type::I8>(n.src));
    let truncated = def<lir::Trunc>(n.src);
    replace(n, build(RvSext8Zbb(truncated.src)));
}

select(n: lir::Sext) {
    require(type_is<Type::I16>(n.src));
    let truncated = def<lir::Trunc>(n.src);
    replace(n, build(RvSext16Zbb(truncated.src)));
}

select(n: lir::Trunc) {
    require(type_is<Type::I16>(n.dst));
    replace(n, build(RvZext16Zbb(n.src)));
}

select(n: lir::Sext) {
    require(type_is<Type::I8>(n.src));
    replace(n, build(RvSext8Zbb(n.src)));
}

select(n: lir::Sext) {
    require(type_is<Type::I16>(n.src));
    replace(n, build(RvSext16Zbb(n.src)));
}

select(n: lir::Zext) {
    require(type_is<Type::I16>(n.src));
    let load = def<lir::Load>(n.src);
    // LHU already returns a zero-extended register. Keep the original load at
    // its original position; only the redundant pure conversion disappears.
    replace(n, build(RvMove64(n.src)));
}

select(n: lir::Zext) {
    require(type_is<Type::I16>(n.src));
    let truncated = def<lir::Trunc>(n.src);
    replace(n, build(RvZext16Zbb(truncated.src)));
}

select(n: lir::Zext) {
    require(type_is<Type::I16>(n.src));
    replace(n, build(RvZext16Zbb(n.src)));
}
