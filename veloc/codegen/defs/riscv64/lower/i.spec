import "../common.spec";

op RvSymbolAddr(target: Global) -> (dst: Value<Type::PTR>) {
    movable = true;
    schedule = Address;
    registers = { dst: GPR };
    encoding = Emission::address(dst, target);
    assembly = { lines: [{ mnemonic: "la", operands: [reg(dst,64), target(target)] }] };
}

select(n: lir::SymbolAddr) {
    replace(n, build(RvSymbolAddr(n.target)));
}

expand Binary(RvAdd32, GprValue, GPR, 59, 0, 0, "addw", I, IntAlu, true);

expand Binary(RvAdd64, GprValue, GPR, 51, 0, 0, "add", I, IntAlu, true);

expand Binary(RvSub32, GprValue, GPR, 59, 0, 32, "subw", I, IntAlu, true);

expand Binary(RvSub64, GprValue, GPR, 51, 0, 32, "sub", I, IntAlu, true);

expand Binary(RvShl32, GprValue, GPR, 59, 1, 0, "sllw", I, IntAlu, true);

expand Binary(RvShl64, GprValue, GPR, 51, 1, 0, "sll", I, IntAlu, true);

expand Binary(RvLshr32, GprValue, GPR, 59, 5, 0, "srlw", I, IntAlu, true);

expand Binary(RvLshr64, GprValue, GPR, 51, 5, 0, "srl", I, IntAlu, true);

expand Binary(RvAshr32, GprValue, GPR, 59, 5, 32, "sraw", I, IntAlu, true);

expand Binary(RvAshr64, GprValue, GPR, 51, 5, 32, "sra", I, IntAlu, true);

expand Binary(RvAnd32, GprValue, GPR, 51, 7, 0, "and", I, IntAlu, true);

expand Binary(RvAnd64, GprValue, GPR, 51, 7, 0, "and", I, IntAlu, true);

expand Binary(RvOr32, GprValue, GPR, 51, 6, 0, "or", I, IntAlu, true);

expand Binary(RvOr64, GprValue, GPR, 51, 6, 0, "or", I, IntAlu, true);

expand Binary(RvXor32, GprValue, GPR, 51, 4, 0, "xor", I, IntAlu, true);

expand Binary(RvXor64, GprValue, GPR, 51, 4, 0, "xor", I, IntAlu, true);

op RvMove32(src: Value<ScalarValue>) -> (dst: Value<ScalarValue>) {
    movable = true;
    schedule = Copy;
    encoding = Emission::instructions([Instruction::Copy(dst,src,32)]);
    registers = { dst: SCALAR, src: SCALAR };
    assembly = {
        lines: [{ mnemonic: "move32", operands: [reg(dst,32), reg(src,32)] }]
    };
}

op RvLi32(imm: i64) -> (dst: Value<GprValue>) {
    movable = true;
    schedule = Constant32;
    encoding = Emission::instructions([Instruction::Constant(dst,imm,32)]);
    registers = { dst: GPR };
    assembly = {
        lines: [{ mnemonic: "li32", operands: [reg(dst,32), imm(imm)] }]
    };
}



op RvMove64(src: Value<ScalarValue>) -> (dst: Value<ScalarValue>) {
    movable = true;
    schedule = Copy;
    encoding = Emission::instructions([Instruction::Copy(dst,src,64)]);
    registers = { dst: SCALAR, src: SCALAR };
    assembly = {
        lines: [{ mnemonic: "move64", operands: [reg(dst,64), reg(src,64)] }]
    };
}

op RvLi64(imm: i64) -> (dst: Value<GprValue>) {
    movable = true;
    schedule = Constant64;
    encoding = Emission::instructions([Instruction::Constant(dst,imm,64)]);
    registers = { dst: GPR };
    assembly = {
        lines: [{ mnemonic: "li64", operands: [reg(dst,64), imm(imm)] }]
    };
}



op RvZext1(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::I(19,dst,7,src,1)]);
    registers = { dst: GPR, src: GPR };
    schedule = IntAlu;
    movable = true;
    assembly = {
        lines: [{ mnemonic: "zext1", operands: [] }]
    };
}

op RvSext1(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    movable = true;
    schedule = IntPair;
    encoding = Emission::instructions([Instruction::I(19,dst,1,src,63), Instruction::I(19,dst,5,dst,1087)]);
    registers = { dst: GPR, src: GPR };
    assembly = {
        lines: [{ mnemonic: "sext1", operands: [] }]
    };
}

op RvZext8(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::I(19,dst,7,src,255)]);
    registers = { dst: GPR, src: GPR };
    schedule = IntAlu;
    movable = true;
    assembly = {
        lines: [{ mnemonic: "zext8", operands: [] }]
    };
}

op RvSext8(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    movable = true;
    schedule = IntPair;
    encoding = Emission::instructions([Instruction::I(19,dst,1,src,56), Instruction::I(19,dst,5,dst,1080)]);
    registers = { dst: GPR, src: GPR };
    assembly = {
        lines: [{ mnemonic: "sext8", operands: [] }]
    };
}

op RvZext16(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    movable = true;
    schedule = IntPair;
    encoding = Emission::instructions([Instruction::I(19,dst,1,src,48), Instruction::I(19,dst,5,dst,48)]);
    registers = { dst: GPR, src: GPR };
    assembly = {
        lines: [{ mnemonic: "zext16", operands: [] }]
    };
}

op RvSext16(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    movable = true;
    schedule = IntPair;
    encoding = Emission::instructions([Instruction::I(19,dst,1,src,48), Instruction::I(19,dst,5,dst,1072)]);
    registers = { dst: GPR, src: GPR };
    assembly = {
        lines: [{ mnemonic: "sext16", operands: [] }]
    };
}

op RvZext32(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    movable = true;
    schedule = IntPair;
    encoding = Emission::instructions([Instruction::I(19,dst,1,src,32), Instruction::I(19,dst,5,dst,32)]);
    registers = { dst: GPR, src: GPR };
    assembly = {
        lines: [{ mnemonic: "zext32", operands: [] }]
    };
}

op RvSext32(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::I(27,dst,0,src,0)]);
    registers = { dst: GPR, src: GPR };
    schedule = IntAlu;
    movable = true;
    assembly = {
        lines: [{ mnemonic: "sext32", operands: [] }]
    };
}

op RvCmpEq(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    movable = true;
    schedule = IntPair;
    encoding = Emission::instructions([Instruction::R(51,dst,4,lhs,rhs,0), Instruction::I(19,dst,3,dst,1)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    assembly = {
        lines: [{ mnemonic: "cmpeq", operands: [] }]
    };
}

op RvCmpNe(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    movable = true;
    schedule = IntPair;
    encoding = Emission::instructions([Instruction::R(51,dst,4,lhs,rhs,0), Instruction::R(51,dst,3,Reg::X0,dst,0)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    assembly = {
        lines: [{ mnemonic: "cmpne", operands: [] }]
    };
}

op RvCmpLtS(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    movable = true;
    schedule = IntAlu;
    encoding = Emission::instructions([Instruction::R(51,dst,2,lhs,rhs,0)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    assembly = {
        lines: [{ mnemonic: "cmplts", operands: [] }]
    };
}

op RvCmpGtS(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    movable = true;
    schedule = IntAlu;
    encoding = Emission::instructions([Instruction::R(51,dst,2,rhs,lhs,0)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    assembly = {
        lines: [{ mnemonic: "cmpgts", operands: [] }]
    };
}

op RvCmpLeS(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    movable = true;
    schedule = IntPair;
    encoding = Emission::instructions([Instruction::R(51,dst,2,rhs,lhs,0), Instruction::I(19,dst,4,dst,1)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    assembly = {
        lines: [{ mnemonic: "cmples", operands: [] }]
    };
}

op RvCmpGeS(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    movable = true;
    schedule = IntPair;
    encoding = Emission::instructions([Instruction::R(51,dst,2,lhs,rhs,0), Instruction::I(19,dst,4,dst,1)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    assembly = {
        lines: [{ mnemonic: "cmpges", operands: [] }]
    };
}

op RvCmpLtU(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    movable = true;
    schedule = IntAlu;
    encoding = Emission::instructions([Instruction::R(51,dst,3,lhs,rhs,0)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    assembly = {
        lines: [{ mnemonic: "cmpltu", operands: [] }]
    };
}

op RvCmpGtU(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    movable = true;
    schedule = IntAlu;
    encoding = Emission::instructions([Instruction::R(51,dst,3,rhs,lhs,0)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    assembly = {
        lines: [{ mnemonic: "cmpgtu", operands: [] }]
    };
}

op RvCmpLeU(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    movable = true;
    schedule = IntPair;
    encoding = Emission::instructions([Instruction::R(51,dst,3,rhs,lhs,0), Instruction::I(19,dst,4,dst,1)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    assembly = {
        lines: [{ mnemonic: "cmpleu", operands: [] }]
    };
}

op RvCmpGeU(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    movable = true;
    schedule = IntPair;
    encoding = Emission::instructions([Instruction::R(51,dst,3,lhs,rhs,0), Instruction::I(19,dst,4,dst,1)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    assembly = {
        lines: [{ mnemonic: "cmpgeu", operands: [] }]
    };
}

op RvEqz(src: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    movable = true;
    schedule = IntAlu;
    encoding = Emission::instructions([Instruction::I(19,dst,3,src,1)]);
    registers = { dst: GPR, src: GPR };
    assembly = {
        lines: [{ mnemonic: "eqz", operands: [] }]
    };
}

op RvLoad8(base: Value<Type::PTR>, offset: i64, flags: MemFlags) -> (dst: Value<GprValue>) {
    clobbers = [X31];
    movable = true;
    schedule = Load;
    encoding = Emission::instructions([Instruction::Load(3,dst,Address { base: base, offset: offset },4)]);
    registers = { dst: GPR, base: GPR };
    memory = { kind: Read, bytes: 1 };
    assembly = {
        lines: [{ mnemonic: "load8", operands: [] }]
    };
}

op RvLoad8Stack(slot: StackSlot, flags: MemFlags) -> (dst: Value<GprValue>) {
    clobbers = [X31];
    schedule = Load;
    encoding = Emission::instructions([Instruction::Load(3,dst,slot,4)]);
    registers = { dst: GPR };
    memory = { kind: Read, bytes: 1 };
    assembly = {
        lines: [{ mnemonic: "load8stack", operands: [] }]
    };
}

op RvStore8(src: Value<GprValue>, base: Value<Type::PTR>, offset: i64, flags: MemFlags) -> () {
    clobbers = [X31];
    movable = true;
    schedule = Store;
    encoding = Emission::instructions([Instruction::Store(35,src,Address { base: base, offset: offset },0)]);
    registers = { src: GPR, base: GPR };
    memory = { kind: Write, bytes: 1 };
    assembly = {
        lines: [{ mnemonic: "store8", operands: [] }]
    };
}

op RvStore8Stack(src: Value<GprValue>, slot: StackSlot, flags: MemFlags) -> () {
    clobbers = [X31];
    schedule = Store;
    encoding = Emission::instructions([Instruction::Store(35,src,slot,0)]);
    registers = { src: GPR };
    memory = { kind: Write, bytes: 1 };
    assembly = {
        lines: [{ mnemonic: "store8stack", operands: [] }]
    };
}

op RvLoad16(base: Value<Type::PTR>, offset: i64, flags: MemFlags) -> (dst: Value<GprValue>) {
    clobbers = [X31];
    movable = true;
    schedule = Load;
    encoding = Emission::instructions([Instruction::Load(3,dst,Address { base: base, offset: offset },5)]);
    registers = { dst: GPR, base: GPR };
    memory = { kind: Read, bytes: 2 };
    assembly = {
        lines: [{ mnemonic: "load16", operands: [] }]
    };
}

op RvLoad16Stack(slot: StackSlot, flags: MemFlags) -> (dst: Value<GprValue>) {
    clobbers = [X31];
    schedule = Load;
    encoding = Emission::instructions([Instruction::Load(3,dst,slot,5)]);
    registers = { dst: GPR };
    memory = { kind: Read, bytes: 2 };
    assembly = {
        lines: [{ mnemonic: "load16stack", operands: [] }]
    };
}

// Signed extending loads used by post-selection producer/consumer folding.
// The byte range and access flags are identical to the original narrow load.
template SignedLoad(Name: ident, Funct3: expr, Bytes: expr, Bits: expr, Mnemonic: expr) {
    op Name(base: Value<Type::PTR>, offset: i64, flags: MemFlags) -> (dst: Value<GprValue>) {
        movable = true;
        schedule = Load;
        encoding = Emission::instructions([Instruction::Load(3,dst,Address { base: base, offset: offset },Funct3)]);
        registers = { dst: GPR, base: GPR };
        clobbers = [X31];
        memory = { kind: Read, bytes: Bytes };
        assembly = { lines: [{ mnemonic: Mnemonic, operands: [reg(dst,64),mem(base,offset,Bits)] }] };
    }
}
expand SignedLoad(RvLoad8Signed, 0, 1, 8, "lb");
expand SignedLoad(RvLoad16Signed, 1, 2, 16, "lh");

op RvStore16(src: Value<GprValue>, base: Value<Type::PTR>, offset: i64, flags: MemFlags) -> () {
    clobbers = [X31];
    movable = true;
    schedule = Store;
    encoding = Emission::instructions([Instruction::Store(35,src,Address { base: base, offset: offset },1)]);
    registers = { src: GPR, base: GPR };
    memory = { kind: Write, bytes: 2 };
    assembly = {
        lines: [{ mnemonic: "store16", operands: [] }]
    };
}

op RvStore16Stack(src: Value<GprValue>, slot: StackSlot, flags: MemFlags) -> () {
    clobbers = [X31];
    schedule = Store;
    encoding = Emission::instructions([Instruction::Store(35,src,slot,1)]);
    registers = { src: GPR };
    memory = { kind: Write, bytes: 2 };
    assembly = {
        lines: [{ mnemonic: "store16stack", operands: [] }]
    };
}

op RvLoad32(base: Value<Type::PTR>, offset: i64, flags: MemFlags) -> (dst: Value<GprValue>) {
    clobbers = [X31];
    movable = true;
    schedule = Load;
    encoding = Emission::instructions([Instruction::Load(3,dst,Address { base: base, offset: offset },2)]);
    registers = { dst: GPR, base: GPR };
    memory = { kind: Read, bytes: 4 };
    assembly = {
        lines: [{ mnemonic: "load32", operands: [] }]
    };
}

op RvLoad32Stack(slot: StackSlot, flags: MemFlags) -> (dst: Value<GprValue>) {
    clobbers = [X31];
    schedule = Load;
    encoding = Emission::instructions([Instruction::Load(3,dst,slot,2)]);
    registers = { dst: GPR };
    memory = { kind: Read, bytes: 4 };
    assembly = {
        lines: [{ mnemonic: "load32stack", operands: [] }]
    };
}

op RvStore32(src: Value<GprValue>, base: Value<Type::PTR>, offset: i64, flags: MemFlags) -> () {
    clobbers = [X31];
    movable = true;
    schedule = Store;
    encoding = Emission::instructions([Instruction::Store(35,src,Address { base: base, offset: offset },2)]);
    registers = { src: GPR, base: GPR };
    memory = { kind: Write, bytes: 4 };
    assembly = {
        lines: [{ mnemonic: "store32", operands: [] }]
    };
}

op RvStore32Stack(src: Value<GprValue>, slot: StackSlot, flags: MemFlags) -> () {
    clobbers = [X31];
    schedule = Store;
    encoding = Emission::instructions([Instruction::Store(35,src,slot,2)]);
    registers = { src: GPR };
    memory = { kind: Write, bytes: 4 };
    assembly = {
        lines: [{ mnemonic: "store32stack", operands: [] }]
    };
}

op RvLoad64(base: Value<Type::PTR>, offset: i64, flags: MemFlags) -> (dst: Value<GprValue>) {
    clobbers = [X31];
    movable = true;
    schedule = Load;
    encoding = Emission::instructions([Instruction::Load(3,dst,Address { base: base, offset: offset },3)]);
    registers = { dst: GPR, base: GPR };
    memory = { kind: Read, bytes: 8 };
    assembly = {
        lines: [{ mnemonic: "load64", operands: [] }]
    };
}

op RvLoad64Stack(slot: StackSlot, flags: MemFlags) -> (dst: Value<GprValue>) {
    clobbers = [X31];
    schedule = Load;
    encoding = Emission::instructions([Instruction::Load(3,dst,slot,3)]);
    registers = { dst: GPR };
    memory = { kind: Read, bytes: 8 };
    assembly = {
        lines: [{ mnemonic: "load64stack", operands: [] }]
    };
}

op RvStore64(src: Value<GprValue>, base: Value<Type::PTR>, offset: i64, flags: MemFlags) -> () {
    clobbers = [X31];
    movable = true;
    schedule = Store;
    encoding = Emission::instructions([Instruction::Store(35,src,Address { base: base, offset: offset },3)]);
    registers = { src: GPR, base: GPR };
    memory = { kind: Write, bytes: 8 };
    assembly = {
        lines: [{ mnemonic: "store64", operands: [] }]
    };
}

op RvStore64Stack(src: Value<GprValue>, slot: StackSlot, flags: MemFlags) -> () {
    clobbers = [X31];
    schedule = Store;
    encoding = Emission::instructions([Instruction::Store(35,src,slot,3)]);
    registers = { src: GPR };
    memory = { kind: Write, bytes: 8 };
    assembly = {
        lines: [{ mnemonic: "store64stack", operands: [] }]
    };
}

op RvStackAddr(slot: StackSlot) -> (dst: Value<Type::PTR>) {
    clobbers = [X31];
    schedule = Address;
    encoding = Emission::instructions([Instruction::Address(dst,slot)]);
    registers = { dst: GPR };
    assembly = {
        lines: [{ mnemonic: "stackaddr", operands: [] }]
    };
}

op RvAddOffset(base: Value<GprValue>, offset: i64) -> (dst: Value<GprValue>) {
    clobbers = [X31];
    movable = true;
    schedule = Address;
    encoding = Emission::instructions([Instruction::Address(dst,Address {base:base,offset:offset})]);
    registers = { dst: GPR, base: GPR };
    assembly = {
        lines: [{ mnemonic: "addoffset", operands: [] }]
    };
}

op RvSelect32(cond: Value<Type::BOOL>, v1: Value<Type::I32 | Type::BOOL>, v2: Value<Type::I32 | Type::BOOL>) -> (dst: Value<Type::I32 | Type::BOOL>) {
    schedule = Select;
    encoding = Emission::instructions([Instruction::B(0,cond,Reg::X0,12), Instruction::Copy(dst,v1,32), Instruction::J(Reg::X0,8), Instruction::Copy(dst,v2,32)]);
    registers = { dst: GPR, cond: GPR, v1: GPR, v2: GPR };
    assembly = {
        lines: [{ mnemonic: "select32", operands: [] }]
    };
}

op RvSelect64(cond: Value<Type::BOOL>, v1: Value<Type::I64 | Type::PTR>, v2: Value<Type::I64 | Type::PTR>) -> (dst: Value<Type::I64 | Type::PTR>) {
    schedule = Select;
    encoding = Emission::instructions([Instruction::B(0,cond,Reg::X0,12), Instruction::Copy(dst,v1,64), Instruction::J(Reg::X0,8), Instruction::Copy(dst,v2,64)]);
    registers = { dst: GPR, cond: GPR, v1: GPR, v2: GPR };
    assembly = {
        lines: [{ mnemonic: "select64", operands: [] }]
    };
}

op RvJump(target: Successor) -> () {
    schedule = Branch;
    encoding = Emission::jump(target);
    registers = {  };
    flow = Jump;
    assembly = {
        lines: [{ mnemonic: "jump", operands: [] }]
    };
}

op RvBranch(cond: Value<Type::BOOL>, target: Successor) -> () {
    schedule = Branch;
    encoding = Emission::branch(1,cond,Reg::X0,target);
    registers = { cond: GPR };
    flow = Branch;
    assembly = {
        lines: [{ mnemonic: "branch", operands: [] }]
    };
}

op RvCall(sp: Value<GprValue>, target: Global, info: CallInfo) -> () {
    schedule = Call;
    encoding = Emission::call(target);
    registers = { sp: fixed(X2, GPR) };
    flow = Call;
    assembly = {
        lines: [{ mnemonic: "call", operands: [] }]
    };
}

op RvCallReg(sp: Value<GprValue>, target: Value<GprValue>, info: CallInfo) -> () {
    schedule = Call;
    encoding = Emission::instructions([Instruction::I(103,Reg::X1,0,target,0)]);
    registers = { sp: fixed(X2, GPR), target: GPR };
    flow = Call;
    assembly = {
        lines: [{ mnemonic: "callreg", operands: [] }]
    };
}

op RvRet() -> () {
    schedule = Return;
    encoding = Emission::instructions([Instruction::I(103,Reg::X0,0,Reg::X1,0)]);
    registers = {  };
    flow = Return;
    assembly = {
        lines: [{ mnemonic: "ret", operands: [] }]
    };
}

op RvTrap() -> () {
    schedule = Trap;
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
        schedule = IntAlu;
        movable = true;
        requires = [I];
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
        schedule = Branch;
        encoding = Emission::branch(F3,lhs,rhs,target);
        registers = { lhs: GPR, rhs: GPR };
        flow = Branch;
        requires = [I];
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
// A zero test of a sign-extended byte/half load observes only zero/nonzero.
// Inspect the load without moving it or duplicating its memory access.
template NarrowLoadBranch(Condition: expr, Target: ident) {
    select(n: lir::Brcond) {
        let cmp = def<lir::Icmp>(n.cond);
        require(matches(cmp.cc, Condition));
        let extended = def<lir::Sext>(cmp.lhs);
        require(type_is<Type::I8 | Type::I16>(extended.src));
        let loaded = def<lir::Load>(extended.src);
        let zero = def<lir::Constant>(cmp.rhs);
        require(matches(zero.imm, 0));
        replace(n, [build(Target(extended.src, reg(X0), n.then_blk)), build(RvJump(n.else_blk))]);
    }
}
expand NarrowLoadBranch(CC::E, RvBranchEq);
expand NarrowLoadBranch(CC::NE, RvBranchNe);

// RV64 pointers and i64 share their complete register representation. Looking
// through these casts avoids manufacturing temporary registers for pointer
// comparisons, especially null tests in pointer-chasing loops.
template PointerBranch(Condition: expr, Target: ident) {
    select(n: lir::Brcond) {
        choose {
            case {
                let cmp = def<lir::Icmp<Type::I64>>(n.cond);
                require(matches(cmp.cc, Condition));
                let left = def<lir::Ptrtoint>(cmp.lhs);
                let right = def<lir::Ptrtoint>(cmp.rhs);
                replace(n, [build(Target(left.src, right.src, n.then_blk)), build(RvJump(n.else_blk))]);
            }
            case {
                let cmp = def<lir::Icmp<Type::I64>>(n.cond);
                require(matches(cmp.cc, Condition));
                let pointer = def<lir::Ptrtoint>(cmp.lhs);
                let zero = def<lir::Constant>(cmp.rhs);
                require(matches(zero.imm, 0));
                replace(n, [build(Target(pointer.src, reg(X0), n.then_blk)), build(RvJump(n.else_blk))]);
            }
            case {
                let cmp = def<lir::Icmp<Type::I64>>(n.cond);
                require(matches(cmp.cc, Condition));
                let pointer = def<lir::Ptrtoint>(cmp.rhs);
                let zero = def<lir::Constant>(cmp.lhs);
                require(matches(zero.imm, 0));
                replace(n, [build(Target(reg(X0), pointer.src, n.then_blk)), build(RvJump(n.else_blk))]);
            }
        }
    }
}
expand PointerBranch(CC::E, RvBranchEq);
expand PointerBranch(CC::NE, RvBranchNe);
expand PointerBranch(CC::L, RvBranchLtS);
expand PointerBranch(CC::GE, RvBranchGeS);
expand PointerBranch(CC::B, RvBranchLtU);
expand PointerBranch(CC::AE, RvBranchGeU);

template IntBranch(Condition: expr, Target: ident) {
    select(n: lir::Brcond) {
        choose {
            case {
                let cmp = def<lir::Icmp<Type::I32 | Type::I64 | Type::PTR>>(n.cond);
                require(matches(cmp.cc, Condition));
                let zero = def<lir::Constant>(cmp.rhs);
                require(matches(zero.imm, 0));
                replace(n, [build(Target(cmp.lhs, reg(X0), n.then_blk)), build(RvJump(n.else_blk))]);
            }
            case {
                let cmp = def<lir::Icmp<Type::I32 | Type::I64 | Type::PTR>>(n.cond);
                require(matches(cmp.cc, Condition));
                let zero = def<lir::Constant>(cmp.lhs);
                require(matches(zero.imm, 0));
                replace(n, [build(Target(reg(X0), cmp.rhs, n.then_blk)), build(RvJump(n.else_blk))]);
            }
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
                let zero = def<lir::Constant>(cmp.rhs);
                require(matches(zero.imm, 0));
                replace(n, [build(Target(reg(X0), cmp.lhs, n.then_blk)), build(RvJump(n.else_blk))]);
            }
            case {
                let cmp = def<lir::Icmp<Type::I32 | Type::I64 | Type::PTR>>(n.cond);
                require(matches(cmp.cc, Condition));
                let zero = def<lir::Constant>(cmp.lhs);
                require(matches(zero.imm, 0));
                replace(n, [build(Target(cmp.rhs, reg(X0), n.then_blk)), build(RvJump(n.else_blk))]);
            }
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

// Expose fallback rotation temporaries to scheduling and allocation. No
// architectural scratch registers are reserved for these multi-op recipes.
template Rotate(Source: ident, Ty: type, First: ident, Second: ident, Or: ident) {
    select(n: Source<Ty>) {
        let reverse = temp(Ty);
        let left = temp(Ty);
        let right = temp(Ty);
        replace(n, [build(RvSub64(reverse, reg(X0), n.rhs)),
                    build(First(left, n.lhs, n.rhs)),
                    build(Second(right, n.lhs, reverse)),
                    build(Or(n.dst, left, right))]);
    }
}
expand Rotate(lir::Rotl, Type::I32, RvShl32, RvLshr32, RvOr32);
expand Rotate(lir::Rotr, Type::I32, RvLshr32, RvShl32, RvOr32);
expand Rotate(lir::Rotl, Type::I64, RvShl64, RvLshr64, RvOr64);
expand Rotate(lir::Rotr, Type::I64, RvLshr64, RvShl64, RvOr64);

select(n: lir::Inttoptr) {
    replace(n, build(RvMove64(n.src)));
}

select(n: lir::Ptrtoint) {
    replace(n, build(RvMove64(n.src)));
}

// Comparison instructions already produce 0/1 in the complete GPR.
template ExtendComparison(Source: ident) {
    select(n: lir::Zext) {
        require(type_is<Type::BOOL>(n.src));
        let comparison = def<Source>(n.src);
        replace(n, build(RvMove64(n.src)));
    }
}
expand ExtendComparison(lir::Icmp);
expand ExtendComparison(lir::Ieqz);
expand ExtendComparison(lir::Fcmp);

select(n: lir::Zext) {
    require(type_is<Type::I8 | Type::I16>(n.src));
    let load = def<lir::Load>(n.src);
    replace(n, build(RvMove64(n.src)));
}

template ZeroExtendTrunc(Ty: type, Target: ident) {
    select(n: lir::Zext) {
        require(type_is<Ty>(n.src));
        let truncated = def<lir::Trunc>(n.src);
        replace(n, build(Target(truncated.src)));
    }
}
expand ZeroExtendTrunc(Type::I8, RvZext8);
expand ZeroExtendTrunc(Type::I16, RvZext16);

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

op RvNez(src: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(51,dst,3,Reg::X0,src,0)]);
    registers = { dst: GPR, src: GPR };
    schedule = IntAlu;
    movable = true;
    requires = [I];
    assembly = { lines: [{ mnemonic: "snez", operands: [reg(dst,64),reg(src,64)] }] };
}

// A low-bit mask is already a canonical boolean. Avoid normalizing it again.
select(n: lir::Icmp) {
    require(matches(n.cc, CC::NE));
    let zero = def<lir::Constant>(n.rhs);
    require(matches(zero.imm, 0));
    let extended = def<lir::Zext>(n.lhs);
    require(type_is<Type::BOOL>(extended.src));
    replace(n, build(RvZext1(extended.src)));
}
select(n: lir::Icmp) {
    require(matches(n.cc, CC::E));
    let zero = def<lir::Constant>(n.rhs);
    require(matches(zero.imm, 0));
    let extended = def<lir::Zext>(n.lhs);
    require(type_is<Type::BOOL>(extended.src));
    let bit = temp(Type::BOOL);
    replace(n, [build(RvZext1(bit, extended.src)), build(RvXorImm(n.dst, bit, 1))]);
}

select(n: lir::Icmp) {
    require(matches(n.cc, CC::NE));
    let zero = def<lir::Constant>(n.rhs);
    require(matches(zero.imm, 0));
    let bit = def<lir::And>(n.lhs);
    let one = def<lir::Constant>(bit.rhs);
    require(matches(one.imm, 1));
    replace(n, build(RvAndImm(bit.lhs, 1)));
}
select(n: lir::Icmp) {
    require(matches(n.cc, CC::E));
    let zero = def<lir::Constant>(n.rhs);
    require(matches(zero.imm, 0));
    let bit = def<lir::And>(n.lhs);
    let one = def<lir::Constant>(bit.rhs);
    require(matches(one.imm, 1));
    let masked = temp(Type::BOOL);
    replace(n, [build(RvAndImm(masked, bit.lhs, 1)), build(RvXorImm(n.dst, masked, 1))]);
}

template CompareZero(Condition: expr, Target: ident) {
    select(n: lir::Icmp) {
        require(matches(n.cc, Condition));
        let zero = def<lir::Constant>(n.rhs);
        require(matches(zero.imm, 0));
        replace(n, build(Target(n.lhs)));
    }
    select(n: lir::Icmp) {
        require(matches(n.cc, Condition));
        let zero = def<lir::Constant>(n.lhs);
        require(matches(zero.imm, 0));
        replace(n, build(Target(n.rhs)));
    }
}
expand CompareZero(CC::E, RvEqz);
expand CompareZero(CC::NE, RvNez);

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
            replace(n, build(RvLoad8(n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::I16>(n.dst));
            replace(n, build(RvLoad16(n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(RvLoad32(n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::I64 | Type::PTR>(n.dst));
            replace(n, build(RvLoad64(n.base, n.offset, n.flags)));
        }
    }
}

// Stores observe only their low bits. Keep any other uses of the truncation,
// while avoiding its masking instructions on this edge of the selection DAG.
template StoreTruncated(Ty: type, Target: ident) {
    select(n: lir::Store) {
        require(type_is<Ty>(n.src));
        let truncated = def<lir::Trunc>(n.src);
        replace(n, build(Target(truncated.src, n.base, n.offset, n.flags)));
    }
}
expand StoreTruncated(Type::I8, RvStore8);
expand StoreTruncated(Type::I16, RvStore16);
expand StoreTruncated(Type::I32, RvStore32);

select(n: lir::Store) {
    choose {
        case {
            require(type_is<Type::BOOL | Type::I8>(n.src));
            replace(n, build(RvStore8(n.src, n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::I16>(n.src));
            replace(n, build(RvStore16(n.src, n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::I32>(n.src));
            replace(n, build(RvStore32(n.src, n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::I64 | Type::PTR>(n.src));
            replace(n, build(RvStore64(n.src, n.base, n.offset, n.flags)));
        }
    }
}

select(n: lir::StackAddr) {
    replace(n, build(RvStackAddr(n.slot)));
}

// A zero arm admits a two-instruction mask instead of a branch diamond.
// Build separate instructions so allocation sees the mask's dependencies.
template SelectZero(Ty: type, And: ident) {
    select(n: lir::Select<Ty>) {
        let zero = def<lir::Constant>(n.v2);
        require(matches(zero.imm, 0));
        let mask = temp(Ty);
        replace(n, [build(RvSub64(mask, reg(X0), n.cond)), build(And(n.dst, n.v1, mask))]);
    }
    select(n: lir::Select<Ty>) {
        let zero = def<lir::Constant>(n.v1);
        require(matches(zero.imm, 0));
        let mask = temp(Ty);
        replace(n, [build(RvAdd64Imm(mask, n.cond, -1)), build(And(n.dst, n.v2, mask))]);
    }
}
expand SelectZero(Type::I32, RvAnd32);
expand SelectZero(Type::I64, RvAnd64);

// Select between integer bit patterns without a control-flow diamond. The
// condition is a canonical BOOL, so -cond is either zero or an all-ones mask.
// A shared XOR operand cancels out of the difference: select(c, x^y, x)
// becomes x ^ (y & -c), without computing x^y twice.
template SelectXor(Ty: type, Xor: ident, And: ident) {
    select(n: lir::Select<Ty>) {
        let delta = def<lir::Xor>(n.v1);
        require(same_value(delta.lhs, n.v2));
        let mask = temp(Ty);
        let selected = temp(Ty);
        replace(n, [build(RvSub64(mask, reg(X0), n.cond)),
                    build(And(selected, delta.rhs, mask)),
                    build(Xor(n.dst, n.v2, selected))]);
    }
    select(n: lir::Select<Ty>) {
        let delta = def<lir::Xor>(n.v1);
        require(same_value(delta.rhs, n.v2));
        let mask = temp(Ty);
        let selected = temp(Ty);
        replace(n, [build(RvSub64(mask, reg(X0), n.cond)),
                    build(And(selected, delta.lhs, mask)),
                    build(Xor(n.dst, n.v2, selected))]);
    }
    select(n: lir::Select<Ty>) {
        let delta = def<lir::Xor>(n.v2);
        require(same_value(delta.lhs, n.v1));
        let mask = temp(Ty);
        let selected = temp(Ty);
        replace(n, [build(RvAdd64Imm(mask, n.cond, -1)),
                    build(And(selected, delta.rhs, mask)),
                    build(Xor(n.dst, n.v1, selected))]);
    }
    select(n: lir::Select<Ty>) {
        let delta = def<lir::Xor>(n.v2);
        require(same_value(delta.rhs, n.v1));
        let mask = temp(Ty);
        let selected = temp(Ty);
        replace(n, [build(RvAdd64Imm(mask, n.cond, -1)),
                    build(And(selected, delta.lhs, mask)),
                    build(Xor(n.dst, n.v1, selected))]);
    }
}
expand SelectXor(Type::I32, RvXor32, RvAnd32);
expand SelectXor(Type::I64, RvXor64, RvAnd64);

template SelectInteger(Ty: type, Xor: ident, And: ident) {
    select(n: lir::Select<Ty>) {
        let difference = temp(Ty);
        let mask = temp(Ty);
        let selected = temp(Ty);
        replace(n, [
            build(Xor(difference, n.v1, n.v2)),
            build(RvSub64(mask, reg(X0), n.cond)),
            build(And(selected, difference, mask)),
            build(Xor(n.dst, n.v2, selected))
        ]);
    }
}
expand SelectInteger(Type::I32, RvXor32, RvAnd32);
expand SelectInteger(Type::I64, RvXor64, RvAnd64);

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
    replace(n, build(RvCall(reg(X2), n.callee, n.info)));
}

select(n: lir::Callind) {
    replace(n, build(RvCallReg(reg(X2), n.callee, n.info)));
}

select(n: lir::Ret) {
    replace(n, build(RvRet()));
}

select(n: lir::Trap) {
    replace(n, build(RvTrap()));
}

// Targets end with the default edge, matching the generic branch table.
op RvBranchTable(index: Value<Type::I32>, targets: sequence(Successor)) -> () {
    registers = { index: GPR };
    clobbers = [X5, X6, X31];
    flow = Jump;
    schedule = Branch;
    requires = [I];
    encoding = Emission::table(index, targets);
    assembly = { lines: [{ mnemonic: "br_table", operands: [reg(index,32)] }] };
}
select(n: lir::Brjt) {
    replace(n, build(RvBranchTable(n.index, n.targets)));
}
