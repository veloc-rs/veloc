import "../common.spec";

expand Binary(RvAdd32, GprValue, GPR, 59, 0, 0, "addw", "I", "IntAlu");

expand Binary(RvAdd64, GprValue, GPR, 51, 0, 0, "add", "I", "IntAlu");

expand Binary(RvSub32, GprValue, GPR, 59, 0, 32, "subw", "I", "IntAlu");

expand Binary(RvSub64, GprValue, GPR, 51, 0, 32, "sub", "I", "IntAlu");

expand Binary(RvShl32, GprValue, GPR, 59, 1, 0, "sllw", "I", "IntAlu");

expand Binary(RvShl64, GprValue, GPR, 51, 1, 0, "sll", "I", "IntAlu");

expand Binary(RvLshr32, GprValue, GPR, 59, 5, 0, "srlw", "I", "IntAlu");

expand Binary(RvLshr64, GprValue, GPR, 51, 5, 0, "srl", "I", "IntAlu");

expand Binary(RvAshr32, GprValue, GPR, 59, 5, 32, "sraw", "I", "IntAlu");

expand Binary(RvAshr64, GprValue, GPR, 51, 5, 32, "sra", "I", "IntAlu");

expand Binary(RvAnd32, GprValue, GPR, 51, 7, 0, "and", "I", "IntAlu");

expand Binary(RvAnd64, GprValue, GPR, 51, 7, 0, "and", "I", "IntAlu");

expand Binary(RvOr32, GprValue, GPR, 51, 6, 0, "or", "I", "IntAlu");

expand Binary(RvOr64, GprValue, GPR, 51, 6, 0, "or", "I", "IntAlu");

expand Binary(RvXor32, GprValue, GPR, 51, 4, 0, "xor", "I", "IntAlu");

expand Binary(RvXor64, GprValue, GPR, 51, 4, 0, "xor", "I", "IntAlu");

op RvMove32(src: Value<ScalarValue>) -> (dst: Value<ScalarValue>) {
    encoding = Emission::instructions([Instruction::Move(dst,src,32)]);
    registers = { dst: SCALAR, src: SCALAR };
    assembly = {
        lines: [{ mnemonic: "move32", operands: [reg(dst,32), reg(src,32)] }]
    };
}

op RvLi32(imm: i64) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::Constant(dst,imm), Instruction::I(27,dst,0,dst,0)]);
    registers = { dst: GPR };
    assembly = {
        lines: [{ mnemonic: "li32", operands: [reg(dst,32), imm(imm)] }]
    };
}

op RvRotl32(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::R(51,Reg::X28,0,Reg::X0,rhs,32), Instruction::R(59,Reg::X29,1,lhs,rhs,0), Instruction::R(59,Reg::X30,5,lhs,Reg::X28,0), Instruction::R(51,dst,6,Reg::X29,Reg::X30,0)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    implicit = { writes: [X28,X29,X30] };
    assembly = {
        lines: [{ mnemonic: "rotl32", operands: [] }]
    };
}

op RvRotr32(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::R(51,Reg::X28,0,Reg::X0,rhs,32), Instruction::R(59,Reg::X29,5,lhs,rhs,0), Instruction::R(59,Reg::X30,1,lhs,Reg::X28,0), Instruction::R(51,dst,6,Reg::X29,Reg::X30,0)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    implicit = { writes: [X28,X29,X30] };
    assembly = {
        lines: [{ mnemonic: "rotr32", operands: [] }]
    };
}

op RvMove64(src: Value<ScalarValue>) -> (dst: Value<ScalarValue>) {
    encoding = Emission::instructions([Instruction::Move(dst,src,64)]);
    registers = { dst: SCALAR, src: SCALAR };
    assembly = {
        lines: [{ mnemonic: "move64", operands: [reg(dst,64), reg(src,64)] }]
    };
}

op RvLi64(imm: i64) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::Constant(dst,imm)]);
    registers = { dst: GPR };
    assembly = {
        lines: [{ mnemonic: "li64", operands: [reg(dst,64), imm(imm)] }]
    };
}

op RvRotl64(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::R(51,Reg::X28,0,Reg::X0,rhs,32), Instruction::R(51,Reg::X29,1,lhs,rhs,0), Instruction::R(51,Reg::X30,5,lhs,Reg::X28,0), Instruction::R(51,dst,6,Reg::X29,Reg::X30,0)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    implicit = { writes: [X28,X29,X30] };
    assembly = {
        lines: [{ mnemonic: "rotl64", operands: [] }]
    };
}

op RvRotr64(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::R(51,Reg::X28,0,Reg::X0,rhs,32), Instruction::R(51,Reg::X29,5,lhs,rhs,0), Instruction::R(51,Reg::X30,1,lhs,Reg::X28,0), Instruction::R(51,dst,6,Reg::X29,Reg::X30,0)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    implicit = { writes: [X28,X29,X30] };
    assembly = {
        lines: [{ mnemonic: "rotr64", operands: [] }]
    };
}

op RvZext1(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::I(19,dst,7,src,1)]);
    registers = { dst: GPR, src: GPR };
    schedule = "IntAlu";
    assembly = {
        lines: [{ mnemonic: "zext1", operands: [] }]
    };
}

op RvSext1(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::I(19,dst,1,src,63), Instruction::I(19,dst,5,dst,1087)]);
    registers = { dst: GPR, src: GPR };
    assembly = {
        lines: [{ mnemonic: "sext1", operands: [] }]
    };
}

op RvZext8(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::I(19,dst,7,src,255)]);
    registers = { dst: GPR, src: GPR };
    schedule = "IntAlu";
    assembly = {
        lines: [{ mnemonic: "zext8", operands: [] }]
    };
}

op RvSext8(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::I(19,dst,1,src,56), Instruction::I(19,dst,5,dst,1080)]);
    registers = { dst: GPR, src: GPR };
    assembly = {
        lines: [{ mnemonic: "sext8", operands: [] }]
    };
}

op RvZext16(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::I(19,dst,1,src,48), Instruction::I(19,dst,5,dst,48)]);
    registers = { dst: GPR, src: GPR };
    assembly = {
        lines: [{ mnemonic: "zext16", operands: [] }]
    };
}

op RvSext16(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::I(19,dst,1,src,48), Instruction::I(19,dst,5,dst,1072)]);
    registers = { dst: GPR, src: GPR };
    assembly = {
        lines: [{ mnemonic: "sext16", operands: [] }]
    };
}

op RvZext32(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::I(19,dst,1,src,32), Instruction::I(19,dst,5,dst,32)]);
    registers = { dst: GPR, src: GPR };
    assembly = {
        lines: [{ mnemonic: "zext32", operands: [] }]
    };
}

op RvSext32(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::I(27,dst,0,src,0)]);
    registers = { dst: GPR, src: GPR };
    schedule = "IntAlu";
    assembly = {
        lines: [{ mnemonic: "sext32", operands: [] }]
    };
}

op RvCmpEq(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(51,dst,4,lhs,rhs,0), Instruction::I(19,dst,3,dst,1)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    assembly = {
        lines: [{ mnemonic: "cmpeq", operands: [] }]
    };
}

op RvCmpNe(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(51,dst,4,lhs,rhs,0), Instruction::I(19,dst,3,dst,1), Instruction::I(19,dst,4,dst,1)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    assembly = {
        lines: [{ mnemonic: "cmpne", operands: [] }]
    };
}

op RvCmpLtS(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(51,dst,2,lhs,rhs,0)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    assembly = {
        lines: [{ mnemonic: "cmplts", operands: [] }]
    };
}

op RvCmpGtS(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(51,dst,2,rhs,lhs,0)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    assembly = {
        lines: [{ mnemonic: "cmpgts", operands: [] }]
    };
}

op RvCmpLeS(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(51,dst,2,rhs,lhs,0), Instruction::I(19,dst,4,dst,1)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    assembly = {
        lines: [{ mnemonic: "cmples", operands: [] }]
    };
}

op RvCmpGeS(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(51,dst,2,lhs,rhs,0), Instruction::I(19,dst,4,dst,1)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    assembly = {
        lines: [{ mnemonic: "cmpges", operands: [] }]
    };
}

op RvCmpLtU(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(51,dst,3,lhs,rhs,0)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    assembly = {
        lines: [{ mnemonic: "cmpltu", operands: [] }]
    };
}

op RvCmpGtU(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(51,dst,3,rhs,lhs,0)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    assembly = {
        lines: [{ mnemonic: "cmpgtu", operands: [] }]
    };
}

op RvCmpLeU(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(51,dst,3,rhs,lhs,0), Instruction::I(19,dst,4,dst,1)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    assembly = {
        lines: [{ mnemonic: "cmpleu", operands: [] }]
    };
}

op RvCmpGeU(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(51,dst,3,lhs,rhs,0), Instruction::I(19,dst,4,dst,1)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    assembly = {
        lines: [{ mnemonic: "cmpgeu", operands: [] }]
    };
}

op RvEqz(src: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::I(19,dst,3,src,1)]);
    registers = { dst: GPR, src: GPR };
    assembly = {
        lines: [{ mnemonic: "eqz", operands: [] }]
    };
}

op RvLoad8(base: Value<Type::PTR>, offset: i64) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::Load(3,dst,Address { base: base, offset: offset },4)]);
    registers = { dst: GPR, base: GPR };
    memory = { kind: Read, bytes: 1 };
    assembly = {
        lines: [{ mnemonic: "load8", operands: [] }]
    };
}

op RvLoad8Stack(slot: StackSlot) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::Load(3,dst,slot,4)]);
    registers = { dst: GPR };
    memory = { kind: Read, bytes: 1 };
    assembly = {
        lines: [{ mnemonic: "load8stack", operands: [] }]
    };
}

op RvStore8(src: Value<GprValue>, base: Value<Type::PTR>, offset: i64) -> () {
    encoding = Emission::instructions([Instruction::Store(35,src,Address { base: base, offset: offset },0)]);
    registers = { src: GPR, base: GPR };
    memory = { kind: Write, bytes: 1 };
    assembly = {
        lines: [{ mnemonic: "store8", operands: [] }]
    };
}

op RvStore8Stack(src: Value<GprValue>, slot: StackSlot) -> () {
    encoding = Emission::instructions([Instruction::Store(35,src,slot,0)]);
    registers = { src: GPR };
    memory = { kind: Write, bytes: 1 };
    assembly = {
        lines: [{ mnemonic: "store8stack", operands: [] }]
    };
}

op RvLoad16(base: Value<Type::PTR>, offset: i64) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::Load(3,dst,Address { base: base, offset: offset },5)]);
    registers = { dst: GPR, base: GPR };
    memory = { kind: Read, bytes: 2 };
    assembly = {
        lines: [{ mnemonic: "load16", operands: [] }]
    };
}

op RvLoad16Stack(slot: StackSlot) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::Load(3,dst,slot,5)]);
    registers = { dst: GPR };
    memory = { kind: Read, bytes: 2 };
    assembly = {
        lines: [{ mnemonic: "load16stack", operands: [] }]
    };
}

op RvStore16(src: Value<GprValue>, base: Value<Type::PTR>, offset: i64) -> () {
    encoding = Emission::instructions([Instruction::Store(35,src,Address { base: base, offset: offset },1)]);
    registers = { src: GPR, base: GPR };
    memory = { kind: Write, bytes: 2 };
    assembly = {
        lines: [{ mnemonic: "store16", operands: [] }]
    };
}

op RvStore16Stack(src: Value<GprValue>, slot: StackSlot) -> () {
    encoding = Emission::instructions([Instruction::Store(35,src,slot,1)]);
    registers = { src: GPR };
    memory = { kind: Write, bytes: 2 };
    assembly = {
        lines: [{ mnemonic: "store16stack", operands: [] }]
    };
}

op RvLoad32(base: Value<Type::PTR>, offset: i64) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::Load(3,dst,Address { base: base, offset: offset },2)]);
    registers = { dst: GPR, base: GPR };
    memory = { kind: Read, bytes: 4 };
    assembly = {
        lines: [{ mnemonic: "load32", operands: [] }]
    };
}

op RvLoad32Stack(slot: StackSlot) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::Load(3,dst,slot,2)]);
    registers = { dst: GPR };
    memory = { kind: Read, bytes: 4 };
    assembly = {
        lines: [{ mnemonic: "load32stack", operands: [] }]
    };
}

op RvStore32(src: Value<GprValue>, base: Value<Type::PTR>, offset: i64) -> () {
    encoding = Emission::instructions([Instruction::Store(35,src,Address { base: base, offset: offset },2)]);
    registers = { src: GPR, base: GPR };
    memory = { kind: Write, bytes: 4 };
    assembly = {
        lines: [{ mnemonic: "store32", operands: [] }]
    };
}

op RvStore32Stack(src: Value<GprValue>, slot: StackSlot) -> () {
    encoding = Emission::instructions([Instruction::Store(35,src,slot,2)]);
    registers = { src: GPR };
    memory = { kind: Write, bytes: 4 };
    assembly = {
        lines: [{ mnemonic: "store32stack", operands: [] }]
    };
}

op RvLoad64(base: Value<Type::PTR>, offset: i64) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::Load(3,dst,Address { base: base, offset: offset },3)]);
    registers = { dst: GPR, base: GPR };
    memory = { kind: Read, bytes: 8 };
    assembly = {
        lines: [{ mnemonic: "load64", operands: [] }]
    };
}

op RvLoad64Stack(slot: StackSlot) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::Load(3,dst,slot,3)]);
    registers = { dst: GPR };
    memory = { kind: Read, bytes: 8 };
    assembly = {
        lines: [{ mnemonic: "load64stack", operands: [] }]
    };
}

op RvStore64(src: Value<GprValue>, base: Value<Type::PTR>, offset: i64) -> () {
    encoding = Emission::instructions([Instruction::Store(35,src,Address { base: base, offset: offset },3)]);
    registers = { src: GPR, base: GPR };
    memory = { kind: Write, bytes: 8 };
    assembly = {
        lines: [{ mnemonic: "store64", operands: [] }]
    };
}

op RvStore64Stack(src: Value<GprValue>, slot: StackSlot) -> () {
    encoding = Emission::instructions([Instruction::Store(35,src,slot,3)]);
    registers = { src: GPR };
    memory = { kind: Write, bytes: 8 };
    assembly = {
        lines: [{ mnemonic: "store64stack", operands: [] }]
    };
}

op RvStackAddr(slot: StackSlot) -> (dst: Value<Type::PTR>) {
    encoding = Emission::instructions([Instruction::Address(dst,slot)]);
    registers = { dst: GPR };
    assembly = {
        lines: [{ mnemonic: "stackaddr", operands: [] }]
    };
}

op RvAddOffset(base: Value<GprValue>, offset: i64) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::Address(dst,Address {base:base,offset:offset})]);
    registers = { dst: GPR, base: GPR };
    assembly = {
        lines: [{ mnemonic: "addoffset", operands: [] }]
    };
}

op RvSelect32(cond: Value<Type::BOOL>, v1: Value<Type::I32 | Type::BOOL>, v2: Value<Type::I32 | Type::BOOL>) -> (dst: Value<Type::I32 | Type::BOOL>) {
    encoding = Emission::instructions([Instruction::B(0,cond,Reg::X0,12), Instruction::Move(dst,v1,32), Instruction::J(Reg::X0,8), Instruction::Move(dst,v2,32)]);
    registers = { dst: GPR, cond: GPR, v1: GPR, v2: GPR };
    assembly = {
        lines: [{ mnemonic: "select32", operands: [] }]
    };
}

op RvSelect64(cond: Value<Type::BOOL>, v1: Value<Type::I64 | Type::PTR>, v2: Value<Type::I64 | Type::PTR>) -> (dst: Value<Type::I64 | Type::PTR>) {
    encoding = Emission::instructions([Instruction::B(0,cond,Reg::X0,12), Instruction::Move(dst,v1,64), Instruction::J(Reg::X0,8), Instruction::Move(dst,v2,64)]);
    registers = { dst: GPR, cond: GPR, v1: GPR, v2: GPR };
    assembly = {
        lines: [{ mnemonic: "select64", operands: [] }]
    };
}

op RvJump(target: Successor) -> () {
    encoding = Emission::jump(target);
    registers = {  };
    flow = Jump;
    assembly = {
        lines: [{ mnemonic: "jump", operands: [] }]
    };
}

op RvBranch(cond: Value<Type::BOOL>, target: Successor) -> () {
    encoding = Emission::branch(1,cond,Reg::X0,target);
    registers = { cond: GPR };
    flow = Branch;
    assembly = {
        lines: [{ mnemonic: "branch", operands: [] }]
    };
}

op RvCall(target: Global, info: CallInfo) -> () {
    encoding = Emission::call(target);
    registers = {  };
    flow = Call; implicit = {reads:[X2]};
    assembly = {
        lines: [{ mnemonic: "call", operands: [] }]
    };
}

op RvCallReg(target: Value<GprValue>, info: CallInfo) -> () {
    encoding = Emission::instructions([Instruction::I(103,Reg::X1,0,target,0)]);
    registers = { target: GPR };
    flow = Call; implicit = {reads:[X2]};
    assembly = {
        lines: [{ mnemonic: "callreg", operands: [] }]
    };
}

op RvRet() -> () {
    encoding = Emission::instructions([Instruction::I(103,Reg::X0,0,Reg::X1,0)]);
    registers = {  };
    flow = Return;
    assembly = {
        lines: [{ mnemonic: "ret", operands: [] }]
    };
}

op RvTrap() -> () {
    encoding = Emission::instructions([Instruction::I(115,Reg::X0,0,Reg::X0,1)]);
    registers = {  };
    flow = Trap;
    assembly = {
        lines: [{ mnemonic: "trap", operands: [] }]
    };
}

// Immediate encodings share the same field layout across integer operations.
template Immediate(Name: ident, Major: expr, F3: expr, Bias: expr, Mnemonic: expr) {
    op Name(src: Value<GprValue>, imm: i64) -> (dst: Value<GprValue>) {
        encoding = Emission::instructions([Instruction::I(Major,dst,F3,src,imm | Bias)]);
        registers = { dst: GPR, src: GPR };
        schedule = "IntAlu";
        requires = ["I"];
        assembly = {
            lines: [{ mnemonic: Mnemonic, operands: [reg(dst,64),reg(src,64),imm(imm)] }]
        };
    }
}

expand Immediate(RvAdd32Imm, 27, 0, 0, "addiw");

expand Immediate(RvAdd64Imm, 19, 0, 0, "addi");

expand Immediate(RvAndImm, 19, 7, 0, "andi");

expand Immediate(RvOrImm, 19, 6, 0, "ori");

expand Immediate(RvXorImm, 19, 4, 0, "xori");

expand Immediate(RvShl32Imm, 27, 1, 0, "slliw");

expand Immediate(RvShl64Imm, 19, 1, 0, "slli");

expand Immediate(RvLshr32Imm, 27, 5, 0, "srliw");

expand Immediate(RvLshr64Imm, 19, 5, 0, "srli");

expand Immediate(RvAshr32Imm, 27, 5, 1024, "sraiw");

expand Immediate(RvAshr64Imm, 19, 5, 1024, "srai");

template CompareBranch(Name: ident, F3: expr, Mnemonic: expr) {
    op Name(lhs: Value<GprValue>, rhs: Value<GprValue>, target: Successor) -> () {
        encoding = Emission::branch(F3,lhs,rhs,target);
        registers = { lhs: GPR, rhs: GPR };
        flow = Branch;
        requires = ["I"];
        assembly = {
            lines: [{ mnemonic: Mnemonic, operands: [reg(lhs,64),reg(rhs,64)] }]
        };
    }
}

expand CompareBranch(RvBranchEq, 0, "beq");

expand CompareBranch(RvBranchNe, 1, "bne");

expand CompareBranch(RvBranchLtS, 4, "blt");

expand CompareBranch(RvBranchGeS, 5, "bge");

expand CompareBranch(RvBranchLtU, 6, "bltu");

expand CompareBranch(RvBranchGeU, 7, "bgeu");

// Constants are folded only when the immediate fits the actual encoding.
template BinaryImmediate(Source: ident, Ty: expr, Target: ident, Bits: expr) {
    select(n: Source<Ty>) {
        choose {
            case {
                let c = def<lir::Constant>(n.rhs);
                require(fits_signed(c.imm, Bits));
                replace(n, build(Target(n.lhs, c.imm)));
            }
            case {
                let c = def<lir::Constant>(n.lhs);
                require(fits_signed(c.imm, Bits));
                replace(n, build(Target(n.rhs, c.imm)));
            }
        }
    }
}

expand BinaryImmediate(lir::Add, Type::I32, RvAdd32Imm, 12);

expand BinaryImmediate(lir::Add, Type::I64, RvAdd64Imm, 12);

expand BinaryImmediate(lir::PtrAdd, Type::I64, RvAdd64Imm, 12);

expand BinaryImmediate(lir::And, Type::I32 | Type::I64 | Type::BOOL, RvAndImm, 12);

expand BinaryImmediate(lir::Or, Type::I32 | Type::I64 | Type::BOOL, RvOrImm, 12);

expand BinaryImmediate(lir::Xor, Type::I32 | Type::I64 | Type::BOOL, RvXorImm, 12);

template ShiftImmediate(Source: ident, Ty: expr, Target: ident, Bits: expr) {
    select(n: Source<Ty>) {
        choose {
            case {
                let c = def<lir::Constant>(n.rhs);
                require(fits_unsigned(c.imm, Bits));
                replace(n, build(Target(n.lhs, c.imm)));
            }
        }
    }
}

expand ShiftImmediate(lir::Shl, Type::I32, RvShl32Imm, 5);

expand ShiftImmediate(lir::Shl, Type::I64, RvShl64Imm, 6);

expand ShiftImmediate(lir::Lshr, Type::I32, RvLshr32Imm, 5);

expand ShiftImmediate(lir::Lshr, Type::I64, RvLshr64Imm, 6);

expand ShiftImmediate(lir::Ashr, Type::I32, RvAshr32Imm, 5);

expand ShiftImmediate(lir::Ashr, Type::I64, RvAshr64Imm, 6);

// i32 registers are sign-extended to XLEN, preserving signed and unsigned order.
template IntBranch(Condition: expr, Target: ident) {
    select(n: lir::Brcond) {
        choose {
            case {
                let cmp = def<lir::Icmp<Type::I32 | Type::I64 | Type::PTR>>(n.cond);
                require(matches(cmp.cc, Condition));
                replace(n, [build(Target(cmp.lhs, cmp.rhs, n.then_blk)), build(RvJump(n.else_blk))]);
            }
        }
    }
}

expand IntBranch(CC::E, RvBranchEq);

expand IntBranch(CC::NE, RvBranchNe);

expand IntBranch(CC::L, RvBranchLtS);

expand IntBranch(CC::GE, RvBranchGeS);

expand IntBranch(CC::B, RvBranchLtU);

expand IntBranch(CC::AE, RvBranchGeU);

template IntBranchReversed(Condition: expr, Target: ident) {
    select(n: lir::Brcond) {
        choose {
            case {
                let cmp = def<lir::Icmp<Type::I32 | Type::I64 | Type::PTR>>(n.cond);
                require(matches(cmp.cc, Condition));
                replace(n, [build(Target(cmp.rhs, cmp.lhs, n.then_blk)), build(RvJump(n.else_blk))]);
            }
        }
    }
}

expand IntBranchReversed(CC::G, RvBranchLtS);

expand IntBranchReversed(CC::LE, RvBranchGeS);

expand IntBranchReversed(CC::A, RvBranchLtU);

expand IntBranchReversed(CC::BE, RvBranchGeU);

select(n: lir::Add) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(RvAdd32(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(RvAdd64(n.lhs, n.rhs)));
        }
    }
}

select(n: lir::Sub) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(RvSub32(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(RvSub64(n.lhs, n.rhs)));
        }
    }
}

select(n: lir::Shl) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(RvShl32(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(RvShl64(n.lhs, n.rhs)));
        }
    }
}

select(n: lir::Lshr) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(RvLshr32(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(RvLshr64(n.lhs, n.rhs)));
        }
    }
}

select(n: lir::Ashr) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(RvAshr32(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(RvAshr64(n.lhs, n.rhs)));
        }
    }
}

select(n: lir::And) {
    choose {
        case {
            require(type_is<Type::I32 | Type::BOOL>(n.dst));
            replace(n, build(RvAnd32(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(RvAnd64(n.lhs, n.rhs)));
        }
    }
}

select(n: lir::Or) {
    choose {
        case {
            require(type_is<Type::I32 | Type::BOOL>(n.dst));
            replace(n, build(RvOr32(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(RvOr64(n.lhs, n.rhs)));
        }
    }
}

select(n: lir::Xor) {
    choose {
        case {
            require(type_is<Type::I32 | Type::BOOL>(n.dst));
            replace(n, build(RvXor32(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(RvXor64(n.lhs, n.rhs)));
        }
    }
}

select(n: lir::Copy) {
    choose {
        case {
            require(type_is<Type::BOOL | Type::I8 | Type::I16 | Type::I32 | Type::F32>(n.dst));
            replace(n, build(RvMove32(n.src)));
        }
        case {
            require(type_is<Type::I64 | Type::PTR | Type::F64>(n.dst));
            replace(n, build(RvMove64(n.src)));
        }
    }
}

select(n: lir::Bitcast) {
    choose {
        case {
            require(type_is<Type::BOOL | Type::I8 | Type::I16 | Type::I32 | Type::F32>(n.dst));
            replace(n, build(RvMove32(n.src)));
        }
        case {
            require(type_is<Type::I64 | Type::PTR | Type::F64>(n.dst));
            replace(n, build(RvMove64(n.src)));
        }
    }
}

select(n: lir::Constant) {
    choose {
        case {
            require(type_is<Type::BOOL | Type::I8 | Type::I16 | Type::I32>(n.dst));
            replace(n, build(RvLi32(n.imm)));
        }
        case {
            require(type_is<Type::I64 | Type::PTR>(n.dst));
            replace(n, build(RvLi64(n.imm)));
        }
    }
}

select(n: lir::Rotl) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(RvRotl32(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(RvRotl64(n.lhs, n.rhs)));
        }
    }
}

select(n: lir::Rotr) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(RvRotr32(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(RvRotr64(n.lhs, n.rhs)));
        }
    }
}

select(n: lir::Inttoptr) {
    replace(n, build(RvMove64(n.src)));
}

select(n: lir::Ptrtoint) {
    replace(n, build(RvMove64(n.src)));
}

select(n: lir::Zext) {
    choose {
        case {
            require(type_is<Type::BOOL>(n.src));
            replace(n, build(RvZext1(n.src)));
        }
        case {
            require(type_is<Type::I8>(n.src));
            replace(n, build(RvZext8(n.src)));
        }
        case {
            require(type_is<Type::I16>(n.src));
            replace(n, build(RvZext16(n.src)));
        }
        case {
            require(type_is<Type::I32>(n.src));
            replace(n, build(RvZext32(n.src)));
        }
    }
}

select(n: lir::Sext) {
    choose {
        case {
            require(type_is<Type::BOOL>(n.src));
            replace(n, build(RvSext1(n.src)));
        }
        case {
            require(type_is<Type::I8>(n.src));
            replace(n, build(RvSext8(n.src)));
        }
        case {
            require(type_is<Type::I16>(n.src));
            replace(n, build(RvSext16(n.src)));
        }
        case {
            require(type_is<Type::I32>(n.src));
            replace(n, build(RvSext32(n.src)));
        }
    }
}

select(n: lir::Trunc) {
    choose {
        case {
            require(type_is<Type::I8>(n.dst));
            replace(n, build(RvZext8(n.src)));
        }
        case {
            require(type_is<Type::I16>(n.dst));
            replace(n, build(RvZext16(n.src)));
        }
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(RvSext32(n.src)));
        }
    }
}

select(n: lir::PtrAdd) {
    replace(n, build(RvAdd64(n.lhs, n.rhs)));
}

select(n: lir::Icmp) {
    choose {
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(matches(n.cc, CC::E));
            replace(n, build(RvCmpEq(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(matches(n.cc, CC::NE));
            replace(n, build(RvCmpNe(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(matches(n.cc, CC::L));
            replace(n, build(RvCmpLtS(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(matches(n.cc, CC::G));
            replace(n, build(RvCmpGtS(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(matches(n.cc, CC::LE));
            replace(n, build(RvCmpLeS(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(matches(n.cc, CC::GE));
            replace(n, build(RvCmpGeS(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(matches(n.cc, CC::B));
            replace(n, build(RvCmpLtU(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(matches(n.cc, CC::A));
            replace(n, build(RvCmpGtU(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(matches(n.cc, CC::BE));
            replace(n, build(RvCmpLeU(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(matches(n.cc, CC::AE));
            replace(n, build(RvCmpGeU(n.lhs, n.rhs)));
        }
    }
}

select(n: lir::Ieqz) {
    require(type_is<Type::BOOL>(n.dst));
    replace(n, build(RvEqz(n.src)));
}

select(n: lir::Load) {
    choose {
        case {
            require(type_is<Type::BOOL | Type::I8>(n.dst));
            replace(n, build(RvLoad8(n.base, n.offset)));
        }
        case {
            require(type_is<Type::I16>(n.dst));
            replace(n, build(RvLoad16(n.base, n.offset)));
        }
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(RvLoad32(n.base, n.offset)));
        }
        case {
            require(type_is<Type::I64 | Type::PTR>(n.dst));
            replace(n, build(RvLoad64(n.base, n.offset)));
        }
    }
}

select(n: lir::Store) {
    choose {
        case {
            require(type_is<Type::BOOL | Type::I8>(n.src));
            replace(n, build(RvStore8(n.src, n.base, n.offset)));
        }
        case {
            require(type_is<Type::I16>(n.src));
            replace(n, build(RvStore16(n.src, n.base, n.offset)));
        }
        case {
            require(type_is<Type::I32>(n.src));
            replace(n, build(RvStore32(n.src, n.base, n.offset)));
        }
        case {
            require(type_is<Type::I64 | Type::PTR>(n.src));
            replace(n, build(RvStore64(n.src, n.base, n.offset)));
        }
    }
}

select(n: lir::StackAddr) {
    replace(n, build(RvStackAddr(n.slot)));
}

select(n: lir::Select) {
    choose {
        case {
            require(type_is<Type::I32 | Type::BOOL>(n.dst));
            replace(n, build(RvSelect32(n.cond, n.v1, n.v2)));
        }
        case {
            require(type_is<Type::I64 | Type::PTR>(n.dst));
            replace(n, build(RvSelect64(n.cond, n.v1, n.v2)));
        }
    }
}

select(n: lir::Br) {
    replace(n, build(RvJump(n.target)));
}

select(n: lir::Brcond) {
    replace(n, [build(RvBranch(n.cond, n.then_blk)), build(RvJump(n.else_blk))]);
}

select(n: lir::Call) {
    replace(n, build(RvCall(n.callee, n.info)));
}

select(n: lir::Callind) {
    replace(n, build(RvCallReg(n.callee, n.info)));
}

select(n: lir::Ret) {
    replace(n, build(RvRet()));
}

select(n: lir::Trap) {
    replace(n, build(RvTrap()));
}
