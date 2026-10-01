import "../common.spec";

expand Binary(RvFadd32, Type::F32, FPR, 83, 0, 0, "fadd.s", F, None);

expand Binary(RvFsub32, Type::F32, FPR, 83, 0, 4, "fsub.s", F, None);

expand Binary(RvFmul32, Type::F32, FPR, 83, 0, 8, "fmul.s", F, None);

expand Binary(RvFdiv32, Type::F32, FPR, 83, 0, 12, "fdiv.s", F, None);

op RvLoadF32(base: Value<Type::PTR>, offset: i64) -> (dst: Value<Type::F32>) {
    encoding = Emission::instructions([Instruction::Load(7,dst,Address { base: base, offset: offset },2)]);
    registers = { dst: FPR, base: GPR };
    memory = { kind: Read, bytes: 4 };
    assembly = {
        lines: [{ mnemonic: "loadf32", operands: [] }]
    };
}

op RvLoadF32Stack(slot: StackSlot) -> (dst: Value<Type::F32>) {
    encoding = Emission::instructions([Instruction::Load(7,dst,slot,2)]);
    registers = { dst: FPR };
    memory = { kind: Read, bytes: 4 };
    assembly = {
        lines: [{ mnemonic: "loadf32stack", operands: [] }]
    };
}

op RvStoreF32(src: Value<Type::F32>, base: Value<Type::PTR>, offset: i64) -> () {
    encoding = Emission::instructions([Instruction::Store(39,src,Address { base: base, offset: offset },2)]);
    registers = { src: FPR, base: GPR };
    memory = { kind: Write, bytes: 4 };
    assembly = {
        lines: [{ mnemonic: "storef32", operands: [] }]
    };
}

op RvStoreF32Stack(src: Value<Type::F32>, slot: StackSlot) -> () {
    encoding = Emission::instructions([Instruction::Store(39,src,slot,2)]);
    registers = { src: FPR };
    memory = { kind: Write, bytes: 4 };
    assembly = {
        lines: [{ mnemonic: "storef32stack", operands: [] }]
    };
}

op RvSelectF32(cond: Value<Type::BOOL>, v1: Value<Type::F32>, v2: Value<Type::F32>) -> (dst: Value<Type::F32>) {
    encoding = Emission::instructions([Instruction::B(0,cond,Reg::X0,12), Instruction::Move(dst,v1,32), Instruction::J(Reg::X0,8), Instruction::Move(dst,v2,32)]);
    registers = { dst: FPR, cond: GPR, v1: FPR, v2: FPR };
    assembly = {
        lines: [{ mnemonic: "selectf32", operands: [] }]
    };
}

op RvFcmpEq32(lhs: Value<Type::F32>, rhs: Value<Type::F32>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(83,dst,2,lhs,rhs,80)]);
    registers = { dst: GPR, lhs: FPR, rhs: FPR };
    assembly = {
        lines: [{ mnemonic: "fcmpeq32", operands: [] }]
    };
}

op RvFcmpNe32(lhs: Value<Type::F32>, rhs: Value<Type::F32>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(83,dst,2,lhs,rhs,80), Instruction::I(19,dst,4,dst,1)]);
    registers = { dst: GPR, lhs: FPR, rhs: FPR };
    assembly = {
        lines: [{ mnemonic: "fcmpne32", operands: [] }]
    };
}

op RvFcmpLt32(lhs: Value<Type::F32>, rhs: Value<Type::F32>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(83,dst,1,lhs,rhs,80)]);
    registers = { dst: GPR, lhs: FPR, rhs: FPR };
    assembly = {
        lines: [{ mnemonic: "fcmplt32", operands: [] }]
    };
}

op RvFcmpLe32(lhs: Value<Type::F32>, rhs: Value<Type::F32>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,lhs,rhs,80)]);
    registers = { dst: GPR, lhs: FPR, rhs: FPR };
    assembly = {
        lines: [{ mnemonic: "fcmple32", operands: [] }]
    };
}

op RvFcmpGt32(lhs: Value<Type::F32>, rhs: Value<Type::F32>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(83,dst,1,rhs,lhs,80)]);
    registers = { dst: GPR, lhs: FPR, rhs: FPR };
    assembly = {
        lines: [{ mnemonic: "fcmpgt32", operands: [] }]
    };
}

op RvFcmpGe32(lhs: Value<Type::F32>, rhs: Value<Type::F32>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,rhs,lhs,80)]);
    registers = { dst: GPR, lhs: FPR, rhs: FPR };
    assembly = {
        lines: [{ mnemonic: "fcmpge32", operands: [] }]
    };
}

op RvFsqrt32(src: Value<Type::F32>) -> (dst: Value<Type::F32>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,src,Reg::X0,44)]);
    registers = { dst: FPR, src: FPR };
    assembly = {
        lines: [{ mnemonic: "fsqrt32", operands: [] }]
    };
}

op RvSitofp32F32(src: Value<Type::I32>) -> (dst: Value<Type::F32>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,src,Reg::X0,104)]);
    registers = { dst: FPR, src: GPR };
    assembly = {
        lines: [{ mnemonic: "sitofp32f32", operands: [] }]
    };
}

op RvSitofp64F32(src: Value<Type::I64>) -> (dst: Value<Type::F32>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,src,Reg::X2,104)]);
    registers = { dst: FPR, src: GPR };
    assembly = {
        lines: [{ mnemonic: "sitofp64f32", operands: [] }]
    };
}

op RvUitofp32F32(src: Value<Type::I32>) -> (dst: Value<Type::F32>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,src,Reg::X1,104)]);
    registers = { dst: FPR, src: GPR };
    assembly = {
        lines: [{ mnemonic: "uitofp32f32", operands: [] }]
    };
}

op RvUitofp64F32(src: Value<Type::I64>) -> (dst: Value<Type::F32>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,src,Reg::X3,104)]);
    registers = { dst: FPR, src: GPR };
    assembly = {
        lines: [{ mnemonic: "uitofp64f32", operands: [] }]
    };
}

op RvFptosi32F32(src: Value<Type::F32>) -> (dst: Value<Type::I32>) {
    encoding = Emission::instructions([Instruction::R(83,dst,1,src,Reg::X0,96)]);
    registers = { dst: GPR, src: FPR };
    assembly = {
        lines: [{ mnemonic: "fptosi32f32", operands: [] }]
    };
}

op RvFptosi64F32(src: Value<Type::F32>) -> (dst: Value<Type::I64>) {
    encoding = Emission::instructions([Instruction::R(83,dst,1,src,Reg::X2,96)]);
    registers = { dst: GPR, src: FPR };
    assembly = {
        lines: [{ mnemonic: "fptosi64f32", operands: [] }]
    };
}

op RvFptoui32F32(src: Value<Type::F32>) -> (dst: Value<Type::I32>) {
    encoding = Emission::instructions([Instruction::R(83,dst,1,src,Reg::X1,96)]);
    registers = { dst: GPR, src: FPR };
    assembly = {
        lines: [{ mnemonic: "fptoui32f32", operands: [] }]
    };
}

op RvFptoui64F32(src: Value<Type::F32>) -> (dst: Value<Type::I64>) {
    encoding = Emission::instructions([Instruction::R(83,dst,1,src,Reg::X3,96)]);
    registers = { dst: GPR, src: FPR };
    assembly = {
        lines: [{ mnemonic: "fptoui64f32", operands: [] }]
    };
}

select(n: lir::Fadd) {
    require(type_is<Type::F32>(n.dst));
    replace(n, build(RvFadd32(n.lhs, n.rhs)));
}

select(n: lir::Fsub) {
    require(type_is<Type::F32>(n.dst));
    replace(n, build(RvFsub32(n.lhs, n.rhs)));
}

select(n: lir::Fmul) {
    require(type_is<Type::F32>(n.dst));
    replace(n, build(RvFmul32(n.lhs, n.rhs)));
}

select(n: lir::Fdiv) {
    require(type_is<Type::F32>(n.dst));
    replace(n, build(RvFdiv32(n.lhs, n.rhs)));
}

select(n: lir::Load) {
    require(type_is<Type::F32>(n.dst));
    replace(n, build(RvLoadF32(n.base, n.offset)));
}

select(n: lir::Store) {
    require(type_is<Type::F32>(n.src));
    replace(n, build(RvStoreF32(n.src, n.base, n.offset)));
}

select(n: lir::Select) {
    require(type_is<Type::F32>(n.dst));
    replace(n, build(RvSelectF32(n.cond, n.v1, n.v2)));
}

select(n: lir::Fcmp) {
    choose {
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<Type::F32>(n.lhs));
            require(matches(n.cc, CC::E));
            replace(n, build(RvFcmpEq32(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<Type::F32>(n.lhs));
            require(matches(n.cc, CC::NE));
            replace(n, build(RvFcmpNe32(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<Type::F32>(n.lhs));
            require(matches(n.cc, CC::B));
            replace(n, build(RvFcmpLt32(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<Type::F32>(n.lhs));
            require(matches(n.cc, CC::BE));
            replace(n, build(RvFcmpLe32(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<Type::F32>(n.lhs));
            require(matches(n.cc, CC::A));
            replace(n, build(RvFcmpGt32(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<Type::F32>(n.lhs));
            require(matches(n.cc, CC::AE));
            replace(n, build(RvFcmpGe32(n.lhs, n.rhs)));
        }
    }
}

select(n: lir::Fsqrt) {
    require(type_is<Type::F32>(n.dst));
    replace(n, build(RvFsqrt32(n.src)));
}

select(n: lir::Sitofp) {
    choose {
        case {
            require(type_is<Type::F32>(n.dst));
            require(type_is<Type::I32>(n.src));
            replace(n, build(RvSitofp32F32(n.src)));
        }
        case {
            require(type_is<Type::F32>(n.dst));
            require(type_is<Type::I64>(n.src));
            replace(n, build(RvSitofp64F32(n.src)));
        }
    }
}

select(n: lir::Uitofp) {
    choose {
        case {
            require(type_is<Type::F32>(n.dst));
            require(type_is<Type::I32>(n.src));
            replace(n, build(RvUitofp32F32(n.src)));
        }
        case {
            require(type_is<Type::F32>(n.dst));
            require(type_is<Type::I64>(n.src));
            replace(n, build(RvUitofp64F32(n.src)));
        }
    }
}

select(n: lir::Fptosi) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::F32>(n.src));
            replace(n, build(RvFptosi32F32(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::F32>(n.src));
            replace(n, build(RvFptosi64F32(n.src)));
        }
    }
}

select(n: lir::Fptoui) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::F32>(n.src));
            replace(n, build(RvFptoui32F32(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::F32>(n.src));
            replace(n, build(RvFptoui64F32(n.src)));
        }
    }
}
