import "../common.spec";

expand Binary(RvRol32, GprValue, GPR, 59, 1, 48, "rolw", Zbb, IntAlu);

expand Binary(RvRor32, GprValue, GPR, 59, 5, 48, "rorw", Zbb, IntAlu);

expand Binary(RvRol64, GprValue, GPR, 51, 1, 48, "rol", Zbb, IntAlu);

expand Binary(RvRor64, GprValue, GPR, 51, 5, 48, "ror", Zbb, IntAlu);

op RvSext8Zbb(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::I(19,dst,1,src,1540)]);
    registers = { dst: GPR, src: GPR };
    schedule = IntAlu;
    requires = [Zbb];
    assembly = {
        lines: [{ mnemonic: "sext.b", operands: [reg(dst,64),reg(src,64)] }]
    };
}

op RvSext16Zbb(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::I(19,dst,1,src,1541)]);
    registers = { dst: GPR, src: GPR };
    schedule = IntAlu;
    requires = [Zbb];
    assembly = {
        lines: [{ mnemonic: "sext.h", operands: [reg(dst,64),reg(src,64)] }]
    };
}

op RvZext16Zbb(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::R(59,dst,4,src,Reg::X0,4)]);
    registers = { dst: GPR, src: GPR };
    schedule = IntAlu;
    requires = [Zbb];
    assembly = {
        lines: [{ mnemonic: "zext.h", operands: [reg(dst,64),reg(src,64)] }]
    };
}

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
    replace(n, build(RvSext8Zbb(n.src)));
}

select(n: lir::Sext) {
    require(type_is<Type::I16>(n.src));
    replace(n, build(RvSext16Zbb(n.src)));
}

select(n: lir::Zext) {
    require(type_is<Type::I16>(n.src));
    replace(n, build(RvZext16Zbb(n.src)));
}
