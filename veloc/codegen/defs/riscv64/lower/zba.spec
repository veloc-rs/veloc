import "../common.spec";

expand Binary(RvSh1Add, GprValue, GPR, 51, 2, 16, "sh1add", "Zba");

expand Binary(RvSh2Add, GprValue, GPR, 51, 4, 16, "sh2add", "Zba");

expand Binary(RvSh3Add, GprValue, GPR, 51, 6, 16, "sh3add", "Zba");

op RvZext32Zba(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::R(59,dst,0,src,Reg::X0,4)]);
    registers = { dst: GPR, src: GPR };
    requires = ["Zba"];
    assembly = {
        lines: [{ mnemonic: "zext.w", operands: [reg(dst,64),reg(src,64)] }]
    };
}

select(n: lir::Zext) {
    require(type_is<Type::I32>(n.src));
    replace(n, build(RvZext32Zba(n.src)));
}

template ShiftAdd(Source: ident, Amount: expr, Target: ident) {
    select(n: Source<Type::I64>) {
        choose {
            case {
                let shift = def<lir::Shl<Type::I64>>(n.rhs);
                let c = def<lir::Constant>(shift.rhs);
                require(matches(c.imm, Amount));
                replace(n, build(Target(shift.lhs, n.lhs)));
            }
        }
    }
}

expand ShiftAdd(lir::Add, 1, RvSh1Add);

expand ShiftAdd(lir::Add, 2, RvSh2Add);

expand ShiftAdd(lir::Add, 3, RvSh3Add);

expand ShiftAdd(lir::PtrAdd, 1, RvSh1Add);

expand ShiftAdd(lir::PtrAdd, 2, RvSh2Add);

expand ShiftAdd(lir::PtrAdd, 3, RvSh3Add);
