import "../../../defs/prelude.spec";
import "../../../encoder/defs/riscv64.spec";
import "registers.spec";

type CallInfo = rust("veloc_lir::CallInfo") { view = borrowed; }
type Block = rust("veloc_lir::BlockId");
type Successor = rust("veloc_lir::EdgeId");
type Global = rust("veloc_lir::SymbolId");
type StackSlot = rust("veloc_lir::StackSlot");
type Emission = rust("crate::target::riscv64::emitter::Emission") {
    trait = rust("crate::target::riscv64::emitter::host::Emission");
    fn instructions(code: sequence(Instruction)) -> Self;
    fn jump(target: Block) -> Self;
    fn branch(cond: Reg, target: Block) -> Self;
    fn call(target: Global) -> Self;
}
typeset GprValue = Type::BOOL | ScalarInteger | Type::PTR;
typeset ScalarValue = GprValue | ScalarFloat;

// All arithmetic families describe actual encoding fields here.
template Binary(Name: ident, Domain: expr, Class: ident, Major: expr, F3: expr, F7: expr, Mnemonic: expr) {
    op Name(lhs: Value<Domain>, rhs: Value<Domain>) -> (dst: Value<Domain>) {
        encoding = Emission::instructions([Instruction::R(Major,dst,F3,lhs,rhs,F7)]);
        registers = { dst: Class, lhs: Class, rhs: Class };
    }
    assembly Name { lines = [{ mnemonic: Mnemonic, operands: [reg(dst,64),reg(lhs,64),reg(rhs,64)] }]; }
}

expand Binary(RvAdd32, GprValue, GPR, 59, 0, 0, "addw");

expand Binary(RvAdd64, GprValue, GPR, 51, 0, 0, "add");

expand Binary(RvSub32, GprValue, GPR, 59, 0, 32, "subw");

expand Binary(RvSub64, GprValue, GPR, 51, 0, 32, "sub");

expand Binary(RvMul32, GprValue, GPR, 59, 0, 1, "mulw");

expand Binary(RvMul64, GprValue, GPR, 51, 0, 1, "mul");

expand Binary(RvSdiv32, GprValue, GPR, 59, 4, 1, "sdivw");

expand Binary(RvSdiv64, GprValue, GPR, 51, 4, 1, "sdiv");

expand Binary(RvUdiv32, GprValue, GPR, 59, 5, 1, "udivw");

expand Binary(RvUdiv64, GprValue, GPR, 51, 5, 1, "udiv");

expand Binary(RvSrem32, GprValue, GPR, 59, 6, 1, "sremw");

expand Binary(RvSrem64, GprValue, GPR, 51, 6, 1, "srem");

expand Binary(RvUrem32, GprValue, GPR, 59, 7, 1, "uremw");

expand Binary(RvUrem64, GprValue, GPR, 51, 7, 1, "urem");

expand Binary(RvShl32, GprValue, GPR, 59, 1, 0, "shlw");

expand Binary(RvShl64, GprValue, GPR, 51, 1, 0, "shl");

expand Binary(RvLshr32, GprValue, GPR, 59, 5, 0, "lshrw");

expand Binary(RvLshr64, GprValue, GPR, 51, 5, 0, "lshr");

expand Binary(RvAshr32, GprValue, GPR, 59, 5, 32, "ashrw");

expand Binary(RvAshr64, GprValue, GPR, 51, 5, 32, "ashr");

expand Binary(RvAnd32, GprValue, GPR, 51, 7, 0, "and");

expand Binary(RvAnd64, GprValue, GPR, 51, 7, 0, "and");

expand Binary(RvOr32, GprValue, GPR, 51, 6, 0, "or");

expand Binary(RvOr64, GprValue, GPR, 51, 6, 0, "or");

expand Binary(RvXor32, GprValue, GPR, 51, 4, 0, "xor");

expand Binary(RvXor64, GprValue, GPR, 51, 4, 0, "xor");

expand Binary(RvFadd32, Type::F32, FPR, 83, 0, 0, "fadd.s");

expand Binary(RvFadd64, Type::F64, FPR, 83, 0, 1, "fadd.d");

expand Binary(RvFsub32, Type::F32, FPR, 83, 0, 4, "fsub.s");

expand Binary(RvFsub64, Type::F64, FPR, 83, 0, 5, "fsub.d");

expand Binary(RvFmul32, Type::F32, FPR, 83, 0, 8, "fmul.s");

expand Binary(RvFmul64, Type::F64, FPR, 83, 0, 9, "fmul.d");

expand Binary(RvFdiv32, Type::F32, FPR, 83, 0, 12, "fdiv.s");

expand Binary(RvFdiv64, Type::F64, FPR, 83, 0, 13, "fdiv.d");

op RvMove32(src: Value<ScalarValue>) -> (dst: Value<ScalarValue>) {
    encoding = Emission::instructions([Instruction::Move(dst,src,32)]);
    registers = { dst: SCALAR, src: SCALAR };

}

assembly RvMove32 { lines = [{ mnemonic: "move32", operands: [reg(dst,32), reg(src,32)] }]; }

op RvLi32(imm: i64) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::Constant(dst,imm), Instruction::I(27,dst,0,dst,0)]);
    registers = { dst: GPR };

}

assembly RvLi32 { lines = [{ mnemonic: "li32", operands: [reg(dst,32), imm(imm)] }]; }

op RvRotl32(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::R(51,Reg::X28,0,Reg::X0,rhs,32), Instruction::R(59,Reg::X29,1,lhs,rhs,0), Instruction::R(59,Reg::X30,5,lhs,Reg::X28,0), Instruction::R(51,dst,6,Reg::X29,Reg::X30,0)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    implicit = { writes: [X28,X29,X30] };
}

assembly RvRotl32 { lines = [{ mnemonic: "rotl32", operands: [] }]; }

op RvRotr32(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::R(51,Reg::X28,0,Reg::X0,rhs,32), Instruction::R(59,Reg::X29,5,lhs,rhs,0), Instruction::R(59,Reg::X30,1,lhs,Reg::X28,0), Instruction::R(51,dst,6,Reg::X29,Reg::X30,0)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    implicit = { writes: [X28,X29,X30] };
}

assembly RvRotr32 { lines = [{ mnemonic: "rotr32", operands: [] }]; }

op RvMove64(src: Value<ScalarValue>) -> (dst: Value<ScalarValue>) {
    encoding = Emission::instructions([Instruction::Move(dst,src,64)]);
    registers = { dst: SCALAR, src: SCALAR };

}

assembly RvMove64 { lines = [{ mnemonic: "move64", operands: [reg(dst,64), reg(src,64)] }]; }

op RvLi64(imm: i64) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::Constant(dst,imm)]);
    registers = { dst: GPR };

}

assembly RvLi64 { lines = [{ mnemonic: "li64", operands: [reg(dst,64), imm(imm)] }]; }

op RvRotl64(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::R(51,Reg::X28,0,Reg::X0,rhs,32), Instruction::R(51,Reg::X29,1,lhs,rhs,0), Instruction::R(51,Reg::X30,5,lhs,Reg::X28,0), Instruction::R(51,dst,6,Reg::X29,Reg::X30,0)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    implicit = { writes: [X28,X29,X30] };
}

assembly RvRotl64 { lines = [{ mnemonic: "rotl64", operands: [] }]; }

op RvRotr64(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::R(51,Reg::X28,0,Reg::X0,rhs,32), Instruction::R(51,Reg::X29,5,lhs,rhs,0), Instruction::R(51,Reg::X30,1,lhs,Reg::X28,0), Instruction::R(51,dst,6,Reg::X29,Reg::X30,0)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };
    implicit = { writes: [X28,X29,X30] };
}

assembly RvRotr64 { lines = [{ mnemonic: "rotr64", operands: [] }]; }

op RvZext1(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::I(19,dst,1,src,63), Instruction::I(19,dst,5,dst,63)]);
    registers = { dst: GPR, src: GPR };

}

assembly RvZext1 { lines = [{ mnemonic: "zext1", operands: [] }]; }

op RvSext1(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::I(19,dst,1,src,63), Instruction::I(19,dst,5,dst,1087)]);
    registers = { dst: GPR, src: GPR };

}

assembly RvSext1 { lines = [{ mnemonic: "sext1", operands: [] }]; }

op RvZext8(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::I(19,dst,1,src,56), Instruction::I(19,dst,5,dst,56)]);
    registers = { dst: GPR, src: GPR };

}

assembly RvZext8 { lines = [{ mnemonic: "zext8", operands: [] }]; }

op RvSext8(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::I(19,dst,1,src,56), Instruction::I(19,dst,5,dst,1080)]);
    registers = { dst: GPR, src: GPR };

}

assembly RvSext8 { lines = [{ mnemonic: "sext8", operands: [] }]; }

op RvZext16(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::I(19,dst,1,src,48), Instruction::I(19,dst,5,dst,48)]);
    registers = { dst: GPR, src: GPR };

}

assembly RvZext16 { lines = [{ mnemonic: "zext16", operands: [] }]; }

op RvSext16(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::I(19,dst,1,src,48), Instruction::I(19,dst,5,dst,1072)]);
    registers = { dst: GPR, src: GPR };

}

assembly RvSext16 { lines = [{ mnemonic: "sext16", operands: [] }]; }

op RvZext32(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::I(19,dst,1,src,32), Instruction::I(19,dst,5,dst,32)]);
    registers = { dst: GPR, src: GPR };

}

assembly RvZext32 { lines = [{ mnemonic: "zext32", operands: [] }]; }

op RvSext32(src: Value<GprValue>) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::I(27,dst,0,src,0)]);
    registers = { dst: GPR, src: GPR };

}

assembly RvSext32 { lines = [{ mnemonic: "sext32", operands: [] }]; }

op RvCmpEq(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(51,dst,4,lhs,rhs,0), Instruction::I(19,dst,3,dst,1)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };

}

assembly RvCmpEq { lines = [{ mnemonic: "cmpeq", operands: [] }]; }

op RvCmpNe(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(51,dst,4,lhs,rhs,0), Instruction::I(19,dst,3,dst,1), Instruction::I(19,dst,4,dst,1)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };

}

assembly RvCmpNe { lines = [{ mnemonic: "cmpne", operands: [] }]; }

op RvCmpLtS(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(51,dst,2,lhs,rhs,0)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };

}

assembly RvCmpLtS { lines = [{ mnemonic: "cmplts", operands: [] }]; }

op RvCmpGtS(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(51,dst,2,rhs,lhs,0)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };

}

assembly RvCmpGtS { lines = [{ mnemonic: "cmpgts", operands: [] }]; }

op RvCmpLeS(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(51,dst,2,rhs,lhs,0), Instruction::I(19,dst,4,dst,1)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };

}

assembly RvCmpLeS { lines = [{ mnemonic: "cmples", operands: [] }]; }

op RvCmpGeS(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(51,dst,2,lhs,rhs,0), Instruction::I(19,dst,4,dst,1)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };

}

assembly RvCmpGeS { lines = [{ mnemonic: "cmpges", operands: [] }]; }

op RvCmpLtU(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(51,dst,3,lhs,rhs,0)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };

}

assembly RvCmpLtU { lines = [{ mnemonic: "cmpltu", operands: [] }]; }

op RvCmpGtU(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(51,dst,3,rhs,lhs,0)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };

}

assembly RvCmpGtU { lines = [{ mnemonic: "cmpgtu", operands: [] }]; }

op RvCmpLeU(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(51,dst,3,rhs,lhs,0), Instruction::I(19,dst,4,dst,1)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };

}

assembly RvCmpLeU { lines = [{ mnemonic: "cmpleu", operands: [] }]; }

op RvCmpGeU(lhs: Value<GprValue>, rhs: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(51,dst,3,lhs,rhs,0), Instruction::I(19,dst,4,dst,1)]);
    registers = { dst: GPR, lhs: GPR, rhs: GPR };

}

assembly RvCmpGeU { lines = [{ mnemonic: "cmpgeu", operands: [] }]; }

op RvEqz(src: Value<GprValue>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::I(19,dst,3,src,1)]);
    registers = { dst: GPR, src: GPR };

}

assembly RvEqz { lines = [{ mnemonic: "eqz", operands: [] }]; }

op RvLoad8(base: Value<Type::PTR>, offset: i64) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::Load(3,dst,Address { base: base, offset: offset },4)]);
    registers = { dst: GPR, base: GPR };
    memory = { kind: Read, bytes: 1 };
}

assembly RvLoad8 { lines = [{ mnemonic: "load8", operands: [] }]; }

op RvLoad8Stack(slot: StackSlot) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::Load(3,dst,slot,4)]);
    registers = { dst: GPR };
    memory = { kind: Read, bytes: 1 };
}

assembly RvLoad8Stack { lines = [{ mnemonic: "load8stack", operands: [] }]; }

op RvStore8(src: Value<GprValue>, base: Value<Type::PTR>, offset: i64) -> () {
    encoding = Emission::instructions([Instruction::Store(35,src,Address { base: base, offset: offset },0)]);
    registers = { src: GPR, base: GPR };
    memory = { kind: Write, bytes: 1 };
}

assembly RvStore8 { lines = [{ mnemonic: "store8", operands: [] }]; }

op RvStore8Stack(src: Value<GprValue>, slot: StackSlot) -> () {
    encoding = Emission::instructions([Instruction::Store(35,src,slot,0)]);
    registers = { src: GPR };
    memory = { kind: Write, bytes: 1 };
}

assembly RvStore8Stack { lines = [{ mnemonic: "store8stack", operands: [] }]; }

op RvLoad16(base: Value<Type::PTR>, offset: i64) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::Load(3,dst,Address { base: base, offset: offset },5)]);
    registers = { dst: GPR, base: GPR };
    memory = { kind: Read, bytes: 2 };
}

assembly RvLoad16 { lines = [{ mnemonic: "load16", operands: [] }]; }

op RvLoad16Stack(slot: StackSlot) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::Load(3,dst,slot,5)]);
    registers = { dst: GPR };
    memory = { kind: Read, bytes: 2 };
}

assembly RvLoad16Stack { lines = [{ mnemonic: "load16stack", operands: [] }]; }

op RvStore16(src: Value<GprValue>, base: Value<Type::PTR>, offset: i64) -> () {
    encoding = Emission::instructions([Instruction::Store(35,src,Address { base: base, offset: offset },1)]);
    registers = { src: GPR, base: GPR };
    memory = { kind: Write, bytes: 2 };
}

assembly RvStore16 { lines = [{ mnemonic: "store16", operands: [] }]; }

op RvStore16Stack(src: Value<GprValue>, slot: StackSlot) -> () {
    encoding = Emission::instructions([Instruction::Store(35,src,slot,1)]);
    registers = { src: GPR };
    memory = { kind: Write, bytes: 2 };
}

assembly RvStore16Stack { lines = [{ mnemonic: "store16stack", operands: [] }]; }

op RvLoad32(base: Value<Type::PTR>, offset: i64) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::Load(3,dst,Address { base: base, offset: offset },2)]);
    registers = { dst: GPR, base: GPR };
    memory = { kind: Read, bytes: 4 };
}

assembly RvLoad32 { lines = [{ mnemonic: "load32", operands: [] }]; }

op RvLoad32Stack(slot: StackSlot) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::Load(3,dst,slot,2)]);
    registers = { dst: GPR };
    memory = { kind: Read, bytes: 4 };
}

assembly RvLoad32Stack { lines = [{ mnemonic: "load32stack", operands: [] }]; }

op RvStore32(src: Value<GprValue>, base: Value<Type::PTR>, offset: i64) -> () {
    encoding = Emission::instructions([Instruction::Store(35,src,Address { base: base, offset: offset },2)]);
    registers = { src: GPR, base: GPR };
    memory = { kind: Write, bytes: 4 };
}

assembly RvStore32 { lines = [{ mnemonic: "store32", operands: [] }]; }

op RvStore32Stack(src: Value<GprValue>, slot: StackSlot) -> () {
    encoding = Emission::instructions([Instruction::Store(35,src,slot,2)]);
    registers = { src: GPR };
    memory = { kind: Write, bytes: 4 };
}

assembly RvStore32Stack { lines = [{ mnemonic: "store32stack", operands: [] }]; }

op RvLoad64(base: Value<Type::PTR>, offset: i64) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::Load(3,dst,Address { base: base, offset: offset },3)]);
    registers = { dst: GPR, base: GPR };
    memory = { kind: Read, bytes: 8 };
}

assembly RvLoad64 { lines = [{ mnemonic: "load64", operands: [] }]; }

op RvLoad64Stack(slot: StackSlot) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::Load(3,dst,slot,3)]);
    registers = { dst: GPR };
    memory = { kind: Read, bytes: 8 };
}

assembly RvLoad64Stack { lines = [{ mnemonic: "load64stack", operands: [] }]; }

op RvStore64(src: Value<GprValue>, base: Value<Type::PTR>, offset: i64) -> () {
    encoding = Emission::instructions([Instruction::Store(35,src,Address { base: base, offset: offset },3)]);
    registers = { src: GPR, base: GPR };
    memory = { kind: Write, bytes: 8 };
}

assembly RvStore64 { lines = [{ mnemonic: "store64", operands: [] }]; }

op RvStore64Stack(src: Value<GprValue>, slot: StackSlot) -> () {
    encoding = Emission::instructions([Instruction::Store(35,src,slot,3)]);
    registers = { src: GPR };
    memory = { kind: Write, bytes: 8 };
}

assembly RvStore64Stack { lines = [{ mnemonic: "store64stack", operands: [] }]; }

op RvLoadF32(base: Value<Type::PTR>, offset: i64) -> (dst: Value<Type::F32>) {
    encoding = Emission::instructions([Instruction::Load(7,dst,Address { base: base, offset: offset },2)]);
    registers = { dst: FPR, base: GPR };
    memory = { kind: Read, bytes: 4 };
}

assembly RvLoadF32 { lines = [{ mnemonic: "loadf32", operands: [] }]; }

op RvLoadF32Stack(slot: StackSlot) -> (dst: Value<Type::F32>) {
    encoding = Emission::instructions([Instruction::Load(7,dst,slot,2)]);
    registers = { dst: FPR };
    memory = { kind: Read, bytes: 4 };
}

assembly RvLoadF32Stack { lines = [{ mnemonic: "loadf32stack", operands: [] }]; }

op RvStoreF32(src: Value<Type::F32>, base: Value<Type::PTR>, offset: i64) -> () {
    encoding = Emission::instructions([Instruction::Store(39,src,Address { base: base, offset: offset },2)]);
    registers = { src: FPR, base: GPR };
    memory = { kind: Write, bytes: 4 };
}

assembly RvStoreF32 { lines = [{ mnemonic: "storef32", operands: [] }]; }

op RvStoreF32Stack(src: Value<Type::F32>, slot: StackSlot) -> () {
    encoding = Emission::instructions([Instruction::Store(39,src,slot,2)]);
    registers = { src: FPR };
    memory = { kind: Write, bytes: 4 };
}

assembly RvStoreF32Stack { lines = [{ mnemonic: "storef32stack", operands: [] }]; }

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

op RvStackAddr(slot: StackSlot) -> (dst: Value<Type::PTR>) {
    encoding = Emission::instructions([Instruction::Address(dst,slot)]);
    registers = { dst: GPR };

}

assembly RvStackAddr { lines = [{ mnemonic: "stackaddr", operands: [] }]; }

op RvAddOffset(base: Value<GprValue>, offset: i64) -> (dst: Value<GprValue>) {
    encoding = Emission::instructions([Instruction::Address(dst,Address {base:base,offset:offset})]);
    registers = { dst: GPR, base: GPR };

}

assembly RvAddOffset { lines = [{ mnemonic: "addoffset", operands: [] }]; }

op RvSelect32(cond: Value<Type::BOOL>, v1: Value<Type::I32 | Type::BOOL>, v2: Value<Type::I32 | Type::BOOL>) -> (dst: Value<Type::I32 | Type::BOOL>) {
    encoding = Emission::instructions([Instruction::B(0,cond,Reg::X0,12), Instruction::Move(dst,v1,32), Instruction::J(Reg::X0,8), Instruction::Move(dst,v2,32)]);
    registers = { dst: GPR, cond: GPR, v1: GPR, v2: GPR };

}

assembly RvSelect32 { lines = [{ mnemonic: "select32", operands: [] }]; }

op RvSelect64(cond: Value<Type::BOOL>, v1: Value<Type::I64 | Type::PTR>, v2: Value<Type::I64 | Type::PTR>) -> (dst: Value<Type::I64 | Type::PTR>) {
    encoding = Emission::instructions([Instruction::B(0,cond,Reg::X0,12), Instruction::Move(dst,v1,64), Instruction::J(Reg::X0,8), Instruction::Move(dst,v2,64)]);
    registers = { dst: GPR, cond: GPR, v1: GPR, v2: GPR };

}

assembly RvSelect64 { lines = [{ mnemonic: "select64", operands: [] }]; }

op RvSelectF32(cond: Value<Type::BOOL>, v1: Value<Type::F32>, v2: Value<Type::F32>) -> (dst: Value<Type::F32>) {
    encoding = Emission::instructions([Instruction::B(0,cond,Reg::X0,12), Instruction::Move(dst,v1,32), Instruction::J(Reg::X0,8), Instruction::Move(dst,v2,32)]);
    registers = { dst: FPR, cond: GPR, v1: FPR, v2: FPR };

}

assembly RvSelectF32 { lines = [{ mnemonic: "selectf32", operands: [] }]; }

op RvSelectF64(cond: Value<Type::BOOL>, v1: Value<Type::F64>, v2: Value<Type::F64>) -> (dst: Value<Type::F64>) {
    encoding = Emission::instructions([Instruction::B(0,cond,Reg::X0,12), Instruction::Move(dst,v1,64), Instruction::J(Reg::X0,8), Instruction::Move(dst,v2,64)]);
    registers = { dst: FPR, cond: GPR, v1: FPR, v2: FPR };

}

assembly RvSelectF64 { lines = [{ mnemonic: "selectf64", operands: [] }]; }

op RvFcmpEq32(lhs: Value<Type::F32>, rhs: Value<Type::F32>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(83,dst,2,lhs,rhs,80)]);
    registers = { dst: GPR, lhs: FPR, rhs: FPR };

}

assembly RvFcmpEq32 { lines = [{ mnemonic: "fcmpeq32", operands: [] }]; }

op RvFcmpNe32(lhs: Value<Type::F32>, rhs: Value<Type::F32>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(83,dst,2,lhs,rhs,80), Instruction::I(19,dst,4,dst,1)]);
    registers = { dst: GPR, lhs: FPR, rhs: FPR };

}

assembly RvFcmpNe32 { lines = [{ mnemonic: "fcmpne32", operands: [] }]; }

op RvFcmpLt32(lhs: Value<Type::F32>, rhs: Value<Type::F32>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(83,dst,1,lhs,rhs,80)]);
    registers = { dst: GPR, lhs: FPR, rhs: FPR };

}

assembly RvFcmpLt32 { lines = [{ mnemonic: "fcmplt32", operands: [] }]; }

op RvFcmpLe32(lhs: Value<Type::F32>, rhs: Value<Type::F32>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,lhs,rhs,80)]);
    registers = { dst: GPR, lhs: FPR, rhs: FPR };

}

assembly RvFcmpLe32 { lines = [{ mnemonic: "fcmple32", operands: [] }]; }

op RvFcmpGt32(lhs: Value<Type::F32>, rhs: Value<Type::F32>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(83,dst,1,rhs,lhs,80)]);
    registers = { dst: GPR, lhs: FPR, rhs: FPR };

}

assembly RvFcmpGt32 { lines = [{ mnemonic: "fcmpgt32", operands: [] }]; }

op RvFcmpGe32(lhs: Value<Type::F32>, rhs: Value<Type::F32>) -> (dst: Value<Type::BOOL>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,rhs,lhs,80)]);
    registers = { dst: GPR, lhs: FPR, rhs: FPR };

}

assembly RvFcmpGe32 { lines = [{ mnemonic: "fcmpge32", operands: [] }]; }

op RvFsqrt32(src: Value<Type::F32>) -> (dst: Value<Type::F32>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,src,Reg::X0,44)]);
    registers = { dst: FPR, src: FPR };

}

assembly RvFsqrt32 { lines = [{ mnemonic: "fsqrt32", operands: [] }]; }

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

op RvSitofp32F32(src: Value<Type::I32>) -> (dst: Value<Type::F32>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,src,Reg::X0,104)]);
    registers = { dst: FPR, src: GPR };

}

assembly RvSitofp32F32 { lines = [{ mnemonic: "sitofp32f32", operands: [] }]; }

op RvSitofp32F64(src: Value<Type::I32>) -> (dst: Value<Type::F64>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,src,Reg::X0,105)]);
    registers = { dst: FPR, src: GPR };

}

assembly RvSitofp32F64 { lines = [{ mnemonic: "sitofp32f64", operands: [] }]; }

op RvSitofp64F32(src: Value<Type::I64>) -> (dst: Value<Type::F32>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,src,Reg::X2,104)]);
    registers = { dst: FPR, src: GPR };

}

assembly RvSitofp64F32 { lines = [{ mnemonic: "sitofp64f32", operands: [] }]; }

op RvSitofp64F64(src: Value<Type::I64>) -> (dst: Value<Type::F64>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,src,Reg::X2,105)]);
    registers = { dst: FPR, src: GPR };

}

assembly RvSitofp64F64 { lines = [{ mnemonic: "sitofp64f64", operands: [] }]; }

op RvUitofp32F32(src: Value<Type::I32>) -> (dst: Value<Type::F32>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,src,Reg::X1,104)]);
    registers = { dst: FPR, src: GPR };

}

assembly RvUitofp32F32 { lines = [{ mnemonic: "uitofp32f32", operands: [] }]; }

op RvUitofp32F64(src: Value<Type::I32>) -> (dst: Value<Type::F64>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,src,Reg::X1,105)]);
    registers = { dst: FPR, src: GPR };

}

assembly RvUitofp32F64 { lines = [{ mnemonic: "uitofp32f64", operands: [] }]; }

op RvUitofp64F32(src: Value<Type::I64>) -> (dst: Value<Type::F32>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,src,Reg::X3,104)]);
    registers = { dst: FPR, src: GPR };

}

assembly RvUitofp64F32 { lines = [{ mnemonic: "uitofp64f32", operands: [] }]; }

op RvUitofp64F64(src: Value<Type::I64>) -> (dst: Value<Type::F64>) {
    encoding = Emission::instructions([Instruction::R(83,dst,0,src,Reg::X3,105)]);
    registers = { dst: FPR, src: GPR };

}

assembly RvUitofp64F64 { lines = [{ mnemonic: "uitofp64f64", operands: [] }]; }

op RvFptosi32F32(src: Value<Type::F32>) -> (dst: Value<Type::I32>) {
    encoding = Emission::instructions([Instruction::R(83,dst,1,src,Reg::X0,96)]);
    registers = { dst: GPR, src: FPR };

}

assembly RvFptosi32F32 { lines = [{ mnemonic: "fptosi32f32", operands: [] }]; }

op RvFptosi32F64(src: Value<Type::F64>) -> (dst: Value<Type::I32>) {
    encoding = Emission::instructions([Instruction::R(83,dst,1,src,Reg::X0,97)]);
    registers = { dst: GPR, src: FPR };

}

assembly RvFptosi32F64 { lines = [{ mnemonic: "fptosi32f64", operands: [] }]; }

op RvFptosi64F32(src: Value<Type::F32>) -> (dst: Value<Type::I64>) {
    encoding = Emission::instructions([Instruction::R(83,dst,1,src,Reg::X2,96)]);
    registers = { dst: GPR, src: FPR };

}

assembly RvFptosi64F32 { lines = [{ mnemonic: "fptosi64f32", operands: [] }]; }

op RvFptosi64F64(src: Value<Type::F64>) -> (dst: Value<Type::I64>) {
    encoding = Emission::instructions([Instruction::R(83,dst,1,src,Reg::X2,97)]);
    registers = { dst: GPR, src: FPR };

}

assembly RvFptosi64F64 { lines = [{ mnemonic: "fptosi64f64", operands: [] }]; }

op RvFptoui32F32(src: Value<Type::F32>) -> (dst: Value<Type::I32>) {
    encoding = Emission::instructions([Instruction::R(83,dst,1,src,Reg::X1,96)]);
    registers = { dst: GPR, src: FPR };

}

assembly RvFptoui32F32 { lines = [{ mnemonic: "fptoui32f32", operands: [] }]; }

op RvFptoui32F64(src: Value<Type::F64>) -> (dst: Value<Type::I32>) {
    encoding = Emission::instructions([Instruction::R(83,dst,1,src,Reg::X1,97)]);
    registers = { dst: GPR, src: FPR };

}

assembly RvFptoui32F64 { lines = [{ mnemonic: "fptoui32f64", operands: [] }]; }

op RvFptoui64F32(src: Value<Type::F32>) -> (dst: Value<Type::I64>) {
    encoding = Emission::instructions([Instruction::R(83,dst,1,src,Reg::X3,96)]);
    registers = { dst: GPR, src: FPR };

}

assembly RvFptoui64F32 { lines = [{ mnemonic: "fptoui64f32", operands: [] }]; }

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

op RvJump(target: Successor) -> () {
    encoding = Emission::jump(target);
    registers = {  };
    flow = Jump;
}

assembly RvJump { lines = [{ mnemonic: "jump", operands: [] }]; }

op RvBranch(cond: Value<Type::BOOL>, target: Successor) -> () {
    encoding = Emission::branch(cond,target);
    registers = { cond: GPR };
    flow = Branch;
}

assembly RvBranch { lines = [{ mnemonic: "branch", operands: [] }]; }

op RvCall(target: Global, info: CallInfo) -> () {
    encoding = Emission::call(target);
    registers = {  };
    flow = Call; implicit = {reads:[X2]};
}

assembly RvCall { lines = [{ mnemonic: "call", operands: [] }]; }

op RvCallReg(target: Value<GprValue>, info: CallInfo) -> () {
    encoding = Emission::instructions([Instruction::I(103,Reg::X1,0,target,0)]);
    registers = { target: GPR };
    flow = Call; implicit = {reads:[X2]};
}

assembly RvCallReg { lines = [{ mnemonic: "callreg", operands: [] }]; }

op RvRet() -> () {
    encoding = Emission::instructions([Instruction::I(103,Reg::X0,0,Reg::X1,0)]);
    registers = {  };
    flow = Return;
}

assembly RvRet { lines = [{ mnemonic: "ret", operands: [] }]; }

op RvTrap() -> () {
    encoding = Emission::instructions([Instruction::I(115,Reg::X0,0,Reg::X0,1)]);
    registers = {  };
    flow = Trap;
}

assembly RvTrap { lines = [{ mnemonic: "trap", operands: [] }]; }
