import "../common.spec";

op RvZext32Zba(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    movable = true;
    schedule = IntAlu;
    // add.uw dst, src, x0: opcode=OP-32, funct3=000, funct7=0000100.
    encoding = Emission::instructions([
        Instruction::R(0b0111011, dst, 0b000, src, Reg::X0, 0b0000100)
    ]);
    registers = { dst: GPR, src: GPR };
    requires = [Zba];
    assembly = {
        lines: [{ mnemonic: "zext.w", operands: [reg(dst,64),reg(src,64)] }]
    };
}

select(n: lir::Zext) {
    require(type_is<Type::I32>(n.src));
    replace(n, build(RvZext32Zba(n.src)));
}

template SelectShiftAdd(Source: ident, Amount: expr, Target: ident) {
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

template ShiftAdd(Name: ident, Amount: expr, Funct3: expr, Mnemonic: expr) {
    // R-type: opcode=OP (0110011), funct7=0010000; funct3 selects the shift.
    expand Binary(Name, GprValue, GPR, 0b0110011, Funct3, 0b0010000, Mnemonic, Zba, IntAlu, true);
    expand SelectShiftAdd(lir::Add, Amount, Name);
    expand SelectShiftAdd(lir::PtrAdd, Amount, Name);
    // Integer addition is commutative; pointer addition's lhs remains a pointer.
    select(n: lir::Add<Type::I64>) {
        let shift = def<lir::Shl<Type::I64>>(n.lhs);
        let c = def<lir::Constant>(shift.rhs);
        require(matches(c.imm, Amount));
        replace(n, build(Name(shift.lhs, n.rhs)));
    }
}

expand ShiftAdd(RvSh1Add, 1, 0b010, "sh1add");
expand ShiftAdd(RvSh2Add, 2, 0b100, "sh2add");
expand ShiftAdd(RvSh3Add, 3, 0b110, "sh3add");
