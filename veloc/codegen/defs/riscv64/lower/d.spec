import "../common.spec";

expand Binary(RvFadd64, Type::F64, FPR, 83, 0, 1, "fadd.d", "D");

expand Binary(RvFsub64, Type::F64, FPR, 83, 0, 5, "fsub.d", "D");

expand Binary(RvFmul64, Type::F64, FPR, 83, 0, 9, "fmul.d", "D");

expand Binary(RvFdiv64, Type::F64, FPR, 83, 0, 13, "fdiv.d", "D");

op RvLoadF64(base: Value<Type::PTR>, offset: i64) -> (dst: Value<Type::F64>) {
    encoding = Emission::instructions([Instruction::Load(7,dst,Address { base: base, offset: offset },3)]);
    registers = { dst: FPR, base: GPR };
    memory = { kind: Read, bytes: 8 };
}

assembly RvLoadF64 { lines = [{ mnemonic: "loadf64", operands: [] }]; }

op RvLoadF64Stack(slot: StackSlot) -> (dst: Value<Type::F64>) {
    encoding = Emission::instructions([Instruction::Load(7,dst,slot,3)]);
    registers = { dst: FPR };
    memory = { kind: Read, bytes: 8 };
}

assembly RvLoadF64Stack { lines = [{ mnemonic: "loadf64stack", operands: [] }]; }

op RvStoreF64(src: Value<Type::F64>, base: Value<Type::PTR>, offset: i64) -> () {
    encoding = Emission::instructions([Instruction::Store(39,src,Address { base: base, offset: offset },3)]);
    registers = { src: FPR, base: GPR };
    memory = { kind: Write, bytes: 8 };
}

assembly RvStoreF64 { lines = [{ mnemonic: "storef64", operands: [] }]; }

op RvStoreF64Stack(src: Value<Type::F64>, slot: StackSlot) -> () {
    encoding = Emission::instructions([Instruction::Store(39,src,slot,3)]);
    registers = { src: FPR };
    memory = { kind: Write, bytes: 8 };
}

assembly RvStoreF64Stack { lines = [{ mnemonic: "storef64stack", operands: [] }]; }

op RvSelectF64(cond: Value<Type::BOOL>, v1: Value<Type::F64>, v2: Value<Type::F64>) -> (dst: Value<Type::F64>) {
    encoding = Emission::instructions([Instruction::B(0,cond,Reg::X0,12), Instruction::Move(dst,v1,64), Instruction::J(Reg::X0,8), Instruction::Move(dst,v2,64)]);
    registers = { dst: FPR, cond: GPR, v1: FPR, v2: FPR };

}

assembly RvSelectF64 { lines = [{ mnemonic: "selectf64", operands: [] }]; }

op RvFcmpEq64(lhs: Value<Type::F64>, rhs: Value<Type::F64>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(83,dst,2,lhs,rhs,81)]);
    registers = { dst: GPR, lhs: FPR, rhs: FPR };

}

assembly RvFcmpEq64 { lines = [{ mnemonic: "fcmpeq64", operands: [] }]; }

op RvFcmpNe64(lhs: Value<Type::F64>, rhs: Value<Type::F64>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(83,dst,2,lhs,rhs,81), Instruction::I(19,dst,4,dst,1)]);
    registers = { dst: GPR, lhs: FPR, rhs: FPR };

}

assembly RvFcmpNe64 { lines = [{ mnemonic: "fcmpne64", operands: [] }]; }

op RvFcmpLt64(lhs: Value<Type::F64>, rhs: Value<Type::F64>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(83,dst,1,lhs,rhs,81)]);
    registers = { dst: GPR, lhs: FPR, rhs: FPR };

}

assembly RvFcmpLt64 { lines = [{ mnemonic: "fcmplt64", operands: [] }]; }

op RvFcmpLe64(lhs: Value<Type::F64>, rhs: Value<Type::F64>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,lhs,rhs,81)]);
    registers = { dst: GPR, lhs: FPR, rhs: FPR };

}

assembly RvFcmpLe64 { lines = [{ mnemonic: "fcmple64", operands: [] }]; }

op RvFcmpGt64(lhs: Value<Type::F64>, rhs: Value<Type::F64>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(83,dst,1,rhs,lhs,81)]);
    registers = { dst: GPR, lhs: FPR, rhs: FPR };

}

assembly RvFcmpGt64 { lines = [{ mnemonic: "fcmpgt64", operands: [] }]; }

op RvFcmpGe64(lhs: Value<Type::F64>, rhs: Value<Type::F64>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,rhs,lhs,81)]);
    registers = { dst: GPR, lhs: FPR, rhs: FPR };

}

assembly RvFcmpGe64 { lines = [{ mnemonic: "fcmpge64", operands: [] }]; }

op RvFsqrt64(src: Value<Type::F64>) -> (dst: Value<Type::F64>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,src,Reg::X0,45)]);
    registers = { dst: FPR, src: FPR };

}

assembly RvFsqrt64 { lines = [{ mnemonic: "fsqrt64", operands: [] }]; }

op RvSitofp32F64(src: Value<Type::I32>) -> (dst: Value<Type::F64>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,src,Reg::X0,105)]);
    registers = { dst: FPR, src: GPR };

}

assembly RvSitofp32F64 { lines = [{ mnemonic: "sitofp32f64", operands: [] }]; }

op RvSitofp64F64(src: Value<Type::I64>) -> (dst: Value<Type::F64>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,src,Reg::X2,105)]);
    registers = { dst: FPR, src: GPR };

}

assembly RvSitofp64F64 { lines = [{ mnemonic: "sitofp64f64", operands: [] }]; }

op RvUitofp32F64(src: Value<Type::I32>) -> (dst: Value<Type::F64>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,src,Reg::X1,105)]);
    registers = { dst: FPR, src: GPR };

}

assembly RvUitofp32F64 { lines = [{ mnemonic: "uitofp32f64", operands: [] }]; }

op RvUitofp64F64(src: Value<Type::I64>) -> (dst: Value<Type::F64>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,src,Reg::X3,105)]);
    registers = { dst: FPR, src: GPR };

}

assembly RvUitofp64F64 { lines = [{ mnemonic: "uitofp64f64", operands: [] }]; }

op RvFptosi32F64(src: Value<Type::F64>) -> (dst: Value<Type::I32>) {
    encoding = Emission::instructions([Instruction::R(83,dst,1,src,Reg::X0,97)]);
    registers = { dst: GPR, src: FPR };

}

assembly RvFptosi32F64 { lines = [{ mnemonic: "fptosi32f64", operands: [] }]; }

op RvFptosi64F64(src: Value<Type::F64>) -> (dst: Value<Type::I64>) {
    encoding = Emission::instructions([Instruction::R(83,dst,1,src,Reg::X2,97)]);
    registers = { dst: GPR, src: FPR };

}

assembly RvFptosi64F64 { lines = [{ mnemonic: "fptosi64f64", operands: [] }]; }

op RvFptoui32F64(src: Value<Type::F64>) -> (dst: Value<Type::I32>) {
    encoding = Emission::instructions([Instruction::R(83,dst,1,src,Reg::X1,97)]);
    registers = { dst: GPR, src: FPR };

}

assembly RvFptoui32F64 { lines = [{ mnemonic: "fptoui32f64", operands: [] }]; }

op RvFptoui64F64(src: Value<Type::F64>) -> (dst: Value<Type::I64>) {
    encoding = Emission::instructions([Instruction::R(83,dst,1,src,Reg::X3,97)]);
    registers = { dst: GPR, src: FPR };

}

assembly RvFptoui64F64 { lines = [{ mnemonic: "fptoui64f64", operands: [] }]; }

op RvFpext(src: Value<Type::F32>) -> (dst: Value<Type::F64>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,src,Reg::X0,33)]);
    registers = { dst: FPR, src: FPR };

}

assembly RvFpext { lines = [{ mnemonic: "fpext", operands: [] }]; }

op RvFptrunc(src: Value<Type::F64>) -> (dst: Value<Type::F32>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,src,Reg::X1,32)]);
    registers = { dst: FPR, src: FPR };

}

assembly RvFptrunc { lines = [{ mnemonic: "fptrunc", operands: [] }]; }

select(n: lir::Fadd) {
    require(type_is<Type::F64>(n.dst));
    replace(n, build(RvFadd64(n.lhs, n.rhs)));
}

select(n: lir::Fsub) {
    require(type_is<Type::F64>(n.dst));
    replace(n, build(RvFsub64(n.lhs, n.rhs)));
}

select(n: lir::Fmul) {
    require(type_is<Type::F64>(n.dst));
    replace(n, build(RvFmul64(n.lhs, n.rhs)));
}

select(n: lir::Fdiv) {
    require(type_is<Type::F64>(n.dst));
    replace(n, build(RvFdiv64(n.lhs, n.rhs)));
}

select(n: lir::Load) {
    require(type_is<Type::F64>(n.dst));
    replace(n, build(RvLoadF64(n.base, n.offset)));
}

select(n: lir::Store) {
    require(type_is<Type::F64>(n.src));
    replace(n, build(RvStoreF64(n.src, n.base, n.offset)));
}

select(n: lir::Select) {
    require(type_is<Type::F64>(n.dst));
    replace(n, build(RvSelectF64(n.cond, n.v1, n.v2)));
}

select(n: lir::Fcmp) {
    choose {
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<Type::F64>(n.lhs));
            require(matches(n.cc, CC::E));
            replace(n, build(RvFcmpEq64(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<Type::F64>(n.lhs));
            require(matches(n.cc, CC::NE));
            replace(n, build(RvFcmpNe64(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<Type::F64>(n.lhs));
            require(matches(n.cc, CC::B));
            replace(n, build(RvFcmpLt64(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<Type::F64>(n.lhs));
            require(matches(n.cc, CC::BE));
            replace(n, build(RvFcmpLe64(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<Type::F64>(n.lhs));
            require(matches(n.cc, CC::A));
            replace(n, build(RvFcmpGt64(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<Type::F64>(n.lhs));
            require(matches(n.cc, CC::AE));
            replace(n, build(RvFcmpGe64(n.lhs, n.rhs)));
        }
    }
}

select(n: lir::Fsqrt) {
    require(type_is<Type::F64>(n.dst));
    replace(n, build(RvFsqrt64(n.src)));
}

select(n: lir::Sitofp) {
    choose {
        case {
            require(type_is<Type::F64>(n.dst));
            require(type_is<Type::I32>(n.src));
            replace(n, build(RvSitofp32F64(n.src)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            require(type_is<Type::I64>(n.src));
            replace(n, build(RvSitofp64F64(n.src)));
        }
    }
}

select(n: lir::Uitofp) {
    choose {
        case {
            require(type_is<Type::F64>(n.dst));
            require(type_is<Type::I32>(n.src));
            replace(n, build(RvUitofp32F64(n.src)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            require(type_is<Type::I64>(n.src));
            replace(n, build(RvUitofp64F64(n.src)));
        }
    }
}

select(n: lir::Fptosi) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::F64>(n.src));
            replace(n, build(RvFptosi32F64(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::F64>(n.src));
            replace(n, build(RvFptosi64F64(n.src)));
        }
    }
}

select(n: lir::Fptoui) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::F64>(n.src));
            replace(n, build(RvFptoui32F64(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::F64>(n.src));
            replace(n, build(RvFptoui64F64(n.src)));
        }
    }
}

select(n: lir::Fpext) {
    replace(n, build(RvFpext(n.src)));
}

select(n: lir::Fptrunc) {
    replace(n, build(RvFptrunc(n.src)));
}
