import "../../../defs/type_sets.spec";
select(n: lir::Add) { choose {
    case { require(type_is<Type::I32>(n.dst)); replace(n, build(RvAdd32(n.lhs,n.rhs))); }
    case { require(type_is<Type::I64>(n.dst)); replace(n, build(RvAdd64(n.lhs,n.rhs))); }
} }
select(n: lir::Sub) { choose {
    case { require(type_is<Type::I32>(n.dst)); replace(n, build(RvSub32(n.lhs,n.rhs))); }
    case { require(type_is<Type::I64>(n.dst)); replace(n, build(RvSub64(n.lhs,n.rhs))); }
} }
select(n: lir::Mul) { choose {
    case { require(type_is<Type::I32>(n.dst)); replace(n, build(RvMul32(n.lhs,n.rhs))); }
    case { require(type_is<Type::I64>(n.dst)); replace(n, build(RvMul64(n.lhs,n.rhs))); }
} }
select(n: lir::Sdiv) { choose {
    case { require(type_is<Type::I32>(n.dst)); replace(n, build(RvSdiv32(n.lhs,n.rhs))); }
    case { require(type_is<Type::I64>(n.dst)); replace(n, build(RvSdiv64(n.lhs,n.rhs))); }
} }
select(n: lir::Udiv) { choose {
    case { require(type_is<Type::I32>(n.dst)); replace(n, build(RvUdiv32(n.lhs,n.rhs))); }
    case { require(type_is<Type::I64>(n.dst)); replace(n, build(RvUdiv64(n.lhs,n.rhs))); }
} }
select(n: lir::Srem) { choose {
    case { require(type_is<Type::I32>(n.dst)); replace(n, build(RvSrem32(n.lhs,n.rhs))); }
    case { require(type_is<Type::I64>(n.dst)); replace(n, build(RvSrem64(n.lhs,n.rhs))); }
} }
select(n: lir::Urem) { choose {
    case { require(type_is<Type::I32>(n.dst)); replace(n, build(RvUrem32(n.lhs,n.rhs))); }
    case { require(type_is<Type::I64>(n.dst)); replace(n, build(RvUrem64(n.lhs,n.rhs))); }
} }
select(n: lir::Shl) { choose {
    case { require(type_is<Type::I32>(n.dst)); replace(n, build(RvShl32(n.lhs,n.rhs))); }
    case { require(type_is<Type::I64>(n.dst)); replace(n, build(RvShl64(n.lhs,n.rhs))); }
} }
select(n: lir::Lshr) { choose {
    case { require(type_is<Type::I32>(n.dst)); replace(n, build(RvLshr32(n.lhs,n.rhs))); }
    case { require(type_is<Type::I64>(n.dst)); replace(n, build(RvLshr64(n.lhs,n.rhs))); }
} }
select(n: lir::Ashr) { choose {
    case { require(type_is<Type::I32>(n.dst)); replace(n, build(RvAshr32(n.lhs,n.rhs))); }
    case { require(type_is<Type::I64>(n.dst)); replace(n, build(RvAshr64(n.lhs,n.rhs))); }
} }
select(n: lir::And) { choose {
    case { require(type_is<Type::I32 | Type::BOOL>(n.dst)); replace(n, build(RvAnd32(n.lhs,n.rhs))); }
    case { require(type_is<Type::I64>(n.dst)); replace(n, build(RvAnd64(n.lhs,n.rhs))); }
} }
select(n: lir::Or) { choose {
    case { require(type_is<Type::I32 | Type::BOOL>(n.dst)); replace(n, build(RvOr32(n.lhs,n.rhs))); }
    case { require(type_is<Type::I64>(n.dst)); replace(n, build(RvOr64(n.lhs,n.rhs))); }
} }
select(n: lir::Xor) { choose {
    case { require(type_is<Type::I32 | Type::BOOL>(n.dst)); replace(n, build(RvXor32(n.lhs,n.rhs))); }
    case { require(type_is<Type::I64>(n.dst)); replace(n, build(RvXor64(n.lhs,n.rhs))); }
} }
select(n: lir::Fadd) { choose {
    case { require(type_is<Type::F32>(n.dst)); replace(n, build(RvFadd32(n.lhs,n.rhs))); }
    case { require(type_is<Type::F64>(n.dst)); replace(n, build(RvFadd64(n.lhs,n.rhs))); }
} }
select(n: lir::Fsub) { choose {
    case { require(type_is<Type::F32>(n.dst)); replace(n, build(RvFsub32(n.lhs,n.rhs))); }
    case { require(type_is<Type::F64>(n.dst)); replace(n, build(RvFsub64(n.lhs,n.rhs))); }
} }
select(n: lir::Fmul) { choose {
    case { require(type_is<Type::F32>(n.dst)); replace(n, build(RvFmul32(n.lhs,n.rhs))); }
    case { require(type_is<Type::F64>(n.dst)); replace(n, build(RvFmul64(n.lhs,n.rhs))); }
} }
select(n: lir::Fdiv) { choose {
    case { require(type_is<Type::F32>(n.dst)); replace(n, build(RvFdiv32(n.lhs,n.rhs))); }
    case { require(type_is<Type::F64>(n.dst)); replace(n, build(RvFdiv64(n.lhs,n.rhs))); }
} }
select(n: lir::Copy) { choose {
    case { require(type_is<Type::BOOL | Type::I8 | Type::I16 | Type::I32 | Type::F32>(n.dst)); replace(n, build(RvMove32(n.src))); }
    case { require(type_is<Type::I64 | Type::PTR | Type::F64>(n.dst)); replace(n, build(RvMove64(n.src))); }
} }
select(n: lir::Bitcast) { choose {
    case { require(type_is<Type::BOOL | Type::I8 | Type::I16 | Type::I32 | Type::F32>(n.dst)); replace(n, build(RvMove32(n.src))); }
    case { require(type_is<Type::I64 | Type::PTR | Type::F64>(n.dst)); replace(n, build(RvMove64(n.src))); }
} }
select(n: lir::Constant) { choose {
    case { require(type_is<Type::BOOL | Type::I8 | Type::I16 | Type::I32>(n.dst)); replace(n, build(RvLi32(n.imm))); }
    case { require(type_is<Type::I64 | Type::PTR>(n.dst)); replace(n, build(RvLi64(n.imm))); }
} }
select(n: lir::Rotl) { choose {
    case { require(type_is<Type::I32>(n.dst)); replace(n, build(RvRotl32(n.lhs,n.rhs))); }
    case { require(type_is<Type::I64>(n.dst)); replace(n, build(RvRotl64(n.lhs,n.rhs))); }
} }
select(n: lir::Rotr) { choose {
    case { require(type_is<Type::I32>(n.dst)); replace(n, build(RvRotr32(n.lhs,n.rhs))); }
    case { require(type_is<Type::I64>(n.dst)); replace(n, build(RvRotr64(n.lhs,n.rhs))); }
} }
select(n: lir::Inttoptr) { choose {
    case {  replace(n, build(RvMove64(n.src))); }
} }
select(n: lir::Ptrtoint) { choose {
    case {  replace(n, build(RvMove64(n.src))); }
} }
select(n: lir::Zext) { choose {
    case { require(type_is<Type::BOOL>(n.src)); replace(n, build(RvZext1(n.src))); }
    case { require(type_is<Type::I8>(n.src)); replace(n, build(RvZext8(n.src))); }
    case { require(type_is<Type::I16>(n.src)); replace(n, build(RvZext16(n.src))); }
    case { require(type_is<Type::I32>(n.src)); replace(n, build(RvZext32(n.src))); }
} }
select(n: lir::Sext) { choose {
    case { require(type_is<Type::BOOL>(n.src)); replace(n, build(RvSext1(n.src))); }
    case { require(type_is<Type::I8>(n.src)); replace(n, build(RvSext8(n.src))); }
    case { require(type_is<Type::I16>(n.src)); replace(n, build(RvSext16(n.src))); }
    case { require(type_is<Type::I32>(n.src)); replace(n, build(RvSext32(n.src))); }
} }
select(n: lir::Trunc) { choose {
    case { require(type_is<Type::I8>(n.dst)); replace(n, build(RvZext8(n.src))); }
    case { require(type_is<Type::I16>(n.dst)); replace(n, build(RvZext16(n.src))); }
    case { require(type_is<Type::I32>(n.dst)); replace(n, build(RvSext32(n.src))); }
} }
select(n: lir::PtrAdd) { choose {
    case {  replace(n, build(RvAdd64(n.lhs,n.rhs))); }
} }
select(n: lir::Icmp) { choose {
    case { require(type_is<Type::BOOL>(n.dst)); require(matches(n.cc, CC::E)); replace(n, build(RvCmpEq(n.lhs,n.rhs))); }
    case { require(type_is<Type::BOOL>(n.dst)); require(matches(n.cc, CC::NE)); replace(n, build(RvCmpNe(n.lhs,n.rhs))); }
    case { require(type_is<Type::BOOL>(n.dst)); require(matches(n.cc, CC::L)); replace(n, build(RvCmpLtS(n.lhs,n.rhs))); }
    case { require(type_is<Type::BOOL>(n.dst)); require(matches(n.cc, CC::G)); replace(n, build(RvCmpGtS(n.lhs,n.rhs))); }
    case { require(type_is<Type::BOOL>(n.dst)); require(matches(n.cc, CC::LE)); replace(n, build(RvCmpLeS(n.lhs,n.rhs))); }
    case { require(type_is<Type::BOOL>(n.dst)); require(matches(n.cc, CC::GE)); replace(n, build(RvCmpGeS(n.lhs,n.rhs))); }
    case { require(type_is<Type::BOOL>(n.dst)); require(matches(n.cc, CC::B)); replace(n, build(RvCmpLtU(n.lhs,n.rhs))); }
    case { require(type_is<Type::BOOL>(n.dst)); require(matches(n.cc, CC::A)); replace(n, build(RvCmpGtU(n.lhs,n.rhs))); }
    case { require(type_is<Type::BOOL>(n.dst)); require(matches(n.cc, CC::BE)); replace(n, build(RvCmpLeU(n.lhs,n.rhs))); }
    case { require(type_is<Type::BOOL>(n.dst)); require(matches(n.cc, CC::AE)); replace(n, build(RvCmpGeU(n.lhs,n.rhs))); }
} }
select(n: lir::Ieqz) { choose {
    case { require(type_is<Type::BOOL>(n.dst));  replace(n, build(RvEqz(n.src))); }
} }
select(n: lir::Load) { choose {
    case { require(type_is<Type::BOOL | Type::I8>(n.dst)); replace(n, build(RvLoad8(n.base,n.offset))); }
    case { require(type_is<Type::I16>(n.dst)); replace(n, build(RvLoad16(n.base,n.offset))); }
    case { require(type_is<Type::I32>(n.dst)); replace(n, build(RvLoad32(n.base,n.offset))); }
    case { require(type_is<Type::I64 | Type::PTR>(n.dst)); replace(n, build(RvLoad64(n.base,n.offset))); }
    case { require(type_is<Type::F32>(n.dst)); replace(n, build(RvLoadF32(n.base,n.offset))); }
    case { require(type_is<Type::F64>(n.dst)); replace(n, build(RvLoadF64(n.base,n.offset))); }
} }
select(n: lir::Store) { choose {
    case { require(type_is<Type::BOOL | Type::I8>(n.src)); replace(n, build(RvStore8(n.src,n.base,n.offset))); }
    case { require(type_is<Type::I16>(n.src)); replace(n, build(RvStore16(n.src,n.base,n.offset))); }
    case { require(type_is<Type::I32>(n.src)); replace(n, build(RvStore32(n.src,n.base,n.offset))); }
    case { require(type_is<Type::I64 | Type::PTR>(n.src)); replace(n, build(RvStore64(n.src,n.base,n.offset))); }
    case { require(type_is<Type::F32>(n.src)); replace(n, build(RvStoreF32(n.src,n.base,n.offset))); }
    case { require(type_is<Type::F64>(n.src)); replace(n, build(RvStoreF64(n.src,n.base,n.offset))); }
} }
select(n: lir::StackAddr) { choose {
    case {  replace(n, build(RvStackAddr(n.slot))); }
} }
select(n: lir::Select) { choose {
    case { require(type_is<Type::I32 | Type::BOOL>(n.dst)); replace(n, build(RvSelect32(n.cond,n.v1,n.v2))); }
    case { require(type_is<Type::I64 | Type::PTR>(n.dst)); replace(n, build(RvSelect64(n.cond,n.v1,n.v2))); }
    case { require(type_is<Type::F32>(n.dst)); replace(n, build(RvSelectF32(n.cond,n.v1,n.v2))); }
    case { require(type_is<Type::F64>(n.dst)); replace(n, build(RvSelectF64(n.cond,n.v1,n.v2))); }
} }
select(n: lir::Fcmp) { choose {
    case { require(type_is<Type::BOOL>(n.dst)); require(type_is<Type::F32>(n.lhs));require(matches(n.cc, CC::E)); replace(n, build(RvFcmpEq32(n.lhs,n.rhs))); }
    case { require(type_is<Type::BOOL>(n.dst)); require(type_is<Type::F32>(n.lhs));require(matches(n.cc, CC::NE)); replace(n, build(RvFcmpNe32(n.lhs,n.rhs))); }
    case { require(type_is<Type::BOOL>(n.dst)); require(type_is<Type::F32>(n.lhs));require(matches(n.cc, CC::B)); replace(n, build(RvFcmpLt32(n.lhs,n.rhs))); }
    case { require(type_is<Type::BOOL>(n.dst)); require(type_is<Type::F32>(n.lhs));require(matches(n.cc, CC::BE)); replace(n, build(RvFcmpLe32(n.lhs,n.rhs))); }
    case { require(type_is<Type::BOOL>(n.dst)); require(type_is<Type::F32>(n.lhs));require(matches(n.cc, CC::A)); replace(n, build(RvFcmpGt32(n.lhs,n.rhs))); }
    case { require(type_is<Type::BOOL>(n.dst)); require(type_is<Type::F32>(n.lhs));require(matches(n.cc, CC::AE)); replace(n, build(RvFcmpGe32(n.lhs,n.rhs))); }
    case { require(type_is<Type::BOOL>(n.dst)); require(type_is<Type::F64>(n.lhs));require(matches(n.cc, CC::E)); replace(n, build(RvFcmpEq64(n.lhs,n.rhs))); }
    case { require(type_is<Type::BOOL>(n.dst)); require(type_is<Type::F64>(n.lhs));require(matches(n.cc, CC::NE)); replace(n, build(RvFcmpNe64(n.lhs,n.rhs))); }
    case { require(type_is<Type::BOOL>(n.dst)); require(type_is<Type::F64>(n.lhs));require(matches(n.cc, CC::B)); replace(n, build(RvFcmpLt64(n.lhs,n.rhs))); }
    case { require(type_is<Type::BOOL>(n.dst)); require(type_is<Type::F64>(n.lhs));require(matches(n.cc, CC::BE)); replace(n, build(RvFcmpLe64(n.lhs,n.rhs))); }
    case { require(type_is<Type::BOOL>(n.dst)); require(type_is<Type::F64>(n.lhs));require(matches(n.cc, CC::A)); replace(n, build(RvFcmpGt64(n.lhs,n.rhs))); }
    case { require(type_is<Type::BOOL>(n.dst)); require(type_is<Type::F64>(n.lhs));require(matches(n.cc, CC::AE)); replace(n, build(RvFcmpGe64(n.lhs,n.rhs))); }
} }
select(n: lir::Fsqrt) { choose {
    case { require(type_is<Type::F32>(n.dst)); replace(n, build(RvFsqrt32(n.src))); }
    case { require(type_is<Type::F64>(n.dst)); replace(n, build(RvFsqrt64(n.src))); }
} }
select(n: lir::Sitofp) { choose {
    case { require(type_is<Type::F32>(n.dst));require(type_is<Type::I32>(n.src)); replace(n, build(RvSitofp32F32(n.src))); }
    case { require(type_is<Type::F64>(n.dst));require(type_is<Type::I32>(n.src)); replace(n, build(RvSitofp32F64(n.src))); }
    case { require(type_is<Type::F32>(n.dst));require(type_is<Type::I64>(n.src)); replace(n, build(RvSitofp64F32(n.src))); }
    case { require(type_is<Type::F64>(n.dst));require(type_is<Type::I64>(n.src)); replace(n, build(RvSitofp64F64(n.src))); }
} }
select(n: lir::Uitofp) { choose {
    case { require(type_is<Type::F32>(n.dst));require(type_is<Type::I32>(n.src)); replace(n, build(RvUitofp32F32(n.src))); }
    case { require(type_is<Type::F64>(n.dst));require(type_is<Type::I32>(n.src)); replace(n, build(RvUitofp32F64(n.src))); }
    case { require(type_is<Type::F32>(n.dst));require(type_is<Type::I64>(n.src)); replace(n, build(RvUitofp64F32(n.src))); }
    case { require(type_is<Type::F64>(n.dst));require(type_is<Type::I64>(n.src)); replace(n, build(RvUitofp64F64(n.src))); }
} }
select(n: lir::Fptosi) { choose {
    case { require(type_is<Type::I32>(n.dst));require(type_is<Type::F32>(n.src)); replace(n, build(RvFptosi32F32(n.src))); }
    case { require(type_is<Type::I32>(n.dst));require(type_is<Type::F64>(n.src)); replace(n, build(RvFptosi32F64(n.src))); }
    case { require(type_is<Type::I64>(n.dst));require(type_is<Type::F32>(n.src)); replace(n, build(RvFptosi64F32(n.src))); }
    case { require(type_is<Type::I64>(n.dst));require(type_is<Type::F64>(n.src)); replace(n, build(RvFptosi64F64(n.src))); }
} }
select(n: lir::Fptoui) { choose {
    case { require(type_is<Type::I32>(n.dst));require(type_is<Type::F32>(n.src)); replace(n, build(RvFptoui32F32(n.src))); }
    case { require(type_is<Type::I32>(n.dst));require(type_is<Type::F64>(n.src)); replace(n, build(RvFptoui32F64(n.src))); }
    case { require(type_is<Type::I64>(n.dst));require(type_is<Type::F32>(n.src)); replace(n, build(RvFptoui64F32(n.src))); }
    case { require(type_is<Type::I64>(n.dst));require(type_is<Type::F64>(n.src)); replace(n, build(RvFptoui64F64(n.src))); }
} }
select(n: lir::Fpext) { choose {
    case {  replace(n, build(RvFpext(n.src))); }
} }
select(n: lir::Fptrunc) { choose {
    case {  replace(n, build(RvFptrunc(n.src))); }
} }
select(n: lir::Br) { choose {
    case {  replace(n, build(RvJump(n.target))); }
} }
select(n: lir::Brcond) { choose {
    case {  replace(n, [build(RvBranch(n.cond,n.then_blk)), build(RvJump(n.else_blk))]); }
} }
select(n: lir::Call) { choose {
    case {  replace(n, build(RvCall(n.callee,n.info))); }
} }
select(n: lir::Callind) { choose {
    case {  replace(n, build(RvCallReg(n.callee,n.info))); }
} }
select(n: lir::Ret) { choose {
    case {  replace(n, build(RvRet())); }
} }
select(n: lir::Trap) { choose {
    case {  replace(n, build(RvTrap())); }
} }
