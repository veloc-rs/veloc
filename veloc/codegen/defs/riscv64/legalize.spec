import "../legalize.spec";

select(inst: lir::Ret) {
    legal(inst);
}

select(inst: lir::Trap) {
    legal(inst);
}

select(inst: lir::Br) {
    legal(inst);
}

select(inst: lir::Brcond) {
    legal(inst);
}

select(inst: lir::Call) {
    legal(inst);
}

select(inst: lir::Callind) {
    legal(inst);
}

select<T: Word>(inst: lir::Add<T> | lir::Sub<T> | lir::Mul<T>) {
    legal(inst);
}

select<T: Narrow>(inst: lir::Add<T>) {
    replace(inst, build(lir::Trunc<T>(lir::Add<Type::I32>(lir::Zext<Type::I32>(inst.lhs), lir::Zext<Type::I32>(inst.rhs)))));
}

select<T: Narrow>(inst: lir::Sub<T>) {
    replace(inst, build(lir::Trunc<T>(lir::Sub<Type::I32>(lir::Zext<Type::I32>(inst.lhs), lir::Zext<Type::I32>(inst.rhs)))));
}

select<T: Narrow>(inst: lir::Mul<T>) {
    replace(inst, build(lir::Trunc<T>(lir::Mul<Type::I32>(lir::Zext<Type::I32>(inst.lhs), lir::Zext<Type::I32>(inst.rhs)))));
}

select(inst: lir::And<Type::BOOL> | lir::Or<Type::BOOL> | lir::Xor<Type::BOOL>) {
    legal(inst);
}

select<T: Word>(inst: lir::And<T> | lir::Or<T> | lir::Xor<T>) {
    legal(inst);
}

select<T: Narrow>(inst: lir::And<T>) {
    replace(inst, build(lir::Trunc<T>(lir::And<Type::I32>(lir::Zext<Type::I32>(inst.lhs), lir::Zext<Type::I32>(inst.rhs)))));
}

select<T: Narrow>(inst: lir::Or<T>) {
    replace(inst, build(lir::Trunc<T>(lir::Or<Type::I32>(lir::Zext<Type::I32>(inst.lhs), lir::Zext<Type::I32>(inst.rhs)))));
}

select<T: Narrow>(inst: lir::Xor<T>) {
    replace(inst, build(lir::Trunc<T>(lir::Xor<Type::I32>(lir::Zext<Type::I32>(inst.lhs), lir::Zext<Type::I32>(inst.rhs)))));
}

select<T: Word>(inst: lir::Shl<T> | lir::Lshr<T> | lir::Ashr<T> | lir::Rotl<T> | lir::Rotr<T> | lir::Sdiv<T> | lir::Udiv<T> | lir::Srem<T> | lir::Urem<T>) {
    legal(inst);
}

select<T: WordOrPtr>(inst: lir::Icmp<T>) {
    legal(inst);
}

select<T: ScalarFloat>(inst: lir::Fcmp<T>) {
    legal(inst);
}

select(inst: lir::Select<Type::BOOL>) {
    legal(inst);
}

select<T: WordValue>(inst: lir::Select<T>) {
    legal(inst);
}

select(inst: lir::StackAddr) {
    legal(inst);
}

select(inst: lir::SymbolAddr) { legal(inst); }

select<T: Scalar | Type::PTR>(inst: lir::Load<T>) {
    choose {
        case {
            require(!fits_signed(inst.offset, 32));
            replace(inst, build(inst {
                base: lir::PtrAdd<Type::PTR>(inst.base, lir::Constant<Type::I64>(inst.offset)),
                offset: 0,
            }));
        }
        case {
            legal(inst);
        }
    }
}

select<T: Scalar | Type::PTR>(inst: lir::Store<T>) {
    choose {
        case {
            require(!fits_signed(inst.offset, 32));
            replace(inst, build(inst {
                base: lir::PtrAdd<Type::PTR>(inst.base, lir::Constant<Type::I64>(inst.offset)),
                offset: 0,
            }));
        }
        case {
            legal(inst);
        }
    }
}

select(inst: lir::Constant<Type::BOOL>) {
    legal(inst);
}

select<T: IntOrPtr>(inst: lir::Constant<T>) {
    legal(inst);
}

select<T: Word>(inst: lir::Ieqz<T>) {
    legal(inst);
}

select<T: Narrow>(inst: lir::Ieqz<T>) {
    replace(inst, build(inst { src: lir::Zext<Type::I32>(inst.src) }));
}

select(inst: lir::Fneg<Type::F32>) {
    fneg_bits32(inst);
}

select(inst: lir::Fabs<Type::F32>) {
    fabs_bits32(inst);
}

select(inst: lir::Fneg<Type::F64>) {
    fneg_bits64(inst);
}

select(inst: lir::Fabs<Type::F64>) {
    fabs_bits64(inst);
}

select<T: ScalarFloat, U: Word>(inst: lir::Sitofp<T, U>) {
    legal(inst);
}

select<T: Word, U: ScalarFloat>(inst: lir::Fptosi<T, U>) {
    legal(inst);
}

select<T: ScalarFloat>(inst: lir::Uitofp<T, Type::I32>) {
    replace(inst, build(unsigned32_to_float<T>(inst.src)));
}
select<T: ScalarFloat>(inst: lir::Uitofp<T, Type::I64>) {
    replace(inst, build(unsigned64_to_float<T>(inst.src)));
}
select<T: ScalarFloat>(inst: lir::Fptoui<Type::I32, T>) {
    replace(inst, build(float_to_unsigned32<T>(inst.src)));
}
select(inst: lir::Fptoui<Type::I64, Type::F32>) {
    replace(inst, build(float_to_unsigned64<Type::F32>(
        inst.src, lir::Bitcast<Type::F32>(lir::Constant<Type::I32>(0x5f000000)))));
}
select(inst: lir::Fptoui<Type::I64, Type::F64>) {
    replace(inst, build(float_to_unsigned64<Type::F64>(
        inst.src, lir::Bitcast<Type::F64>(lir::Constant<Type::I64>(0x43e0000000000000)))));
}

select<T: ScalarFloat>(inst: lir::Fsqrt<T>) {
    legal(inst);
}

select(inst: lir::Fpext<Type::F64, Type::F32>) {
    legal(inst);
}

select(inst: lir::Fptrunc<Type::F32, Type::F64>) {
    legal(inst);
}

select<T: Word>(inst: lir::Zext<T, Type::BOOL>) {
    legal(inst);
}

select<T: Word, U: Narrow>(inst: lir::Zext<T, U>) {
    legal(inst);
}

select(inst: lir::Zext<Type::I64, Type::I32>) {
    legal(inst);
}

select<T: Word, U: Narrow>(inst: lir::Sext<T, U>) {
    legal(inst);
}

select(inst: lir::Sext<Type::I64, Type::I32>) {
    legal(inst);
}

select<T: SmallInt, U: Word>(inst: lir::Trunc<T, U>) {
    legal(inst);
}

select<T: Narrow, U: Narrow>(inst: lir::Trunc<T, U>) { legal(inst); }
select<T: Narrow, U: Narrow>(inst: lir::Zext<T, U> | lir::Sext<T, U>) { legal(inst); }
select<T: Narrow>(inst: lir::Zext<T, Type::BOOL>) { legal(inst); }

select(inst: lir::Inttoptr<Type::I64>) {
    legal(inst);
}

select(inst: lir::Ptrtoint<Type::I64>) {
    legal(inst);
}

select(inst: lir::PtrAdd<Type::I64>) {
    legal(inst);
}

select<T: Scalar | Type::PTR>(inst: lir::Copy<T>) {
    replace(inst, build(inst.src));
}

select(inst: lir::Bitcast<Type::F32, Type::I32>) {
    legal(inst);
}

select(inst: lir::Bitcast<Type::I32, Type::F32>) {
    legal(inst);
}

select(inst: lir::Bitcast<Type::F64, Type::I64>) {
    legal(inst);
}

select(inst: lir::Bitcast<Type::I64, Type::F64>) {
    legal(inst);
}

select<T: ScalarFloat>(inst: lir::Fconstant<T>) {
    legal(inst);
}

select<T: ScalarFloat>(inst: lir::Fadd<T> | lir::Fsub<T> | lir::Fmul<T> | lir::Fdiv<T>) {
    legal(inst);
}

select(inst: lir::Brjt<Type::I32>) {
    legal(inst);
}

select(inst: lir::Ctlz<Type::I32>) {
    leading_zeros32(inst);
}
select(inst: lir::Ctlz<Type::I64>) {
    leading_zeros64(inst);
}
select<T: Word>(inst: lir::Cttz<T>) {
    trailing_zeros(inst);
}

// RV64GC does not require Zbb.
select(inst: lir::Ctpop<Type::I32>) { popcount32(inst); }
select(inst: lir::Ctpop<Type::I64>) { popcount64(inst); }
