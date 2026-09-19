import "../legalize.spec";

// Instruction-local legality and concrete rewrite plans. Order is priority.
type Action = rust("crate::passes::lowering::LegalizeAction");
type Instruction = rust("crate::target::x86_64::inst::TargetInst") {
    const POPCNT32: Self;
    const POPCNT64: Self;
}
type Target = rust("crate::target::x86_64::inst::FeatureSet") {
    fn supports(&self, instruction: Instruction) -> bool;
}

rule arg_0<T: Scalar | Type::PTR>(inst: lir::Arg<T>) {
    action = legal;
}

rule ret_0(inst: lir::Ret) {
    action = legal;
}

rule trap_0(inst: lir::Trap) {
    action = legal;
}

rule br_0(inst: lir::Br) {
    action = legal;
}

rule brcond_0(inst: lir::Brcond) {
    action = legal;
}

rule call_0(inst: lir::Call) {
    action = legal;
}

rule callind_0(inst: lir::Callind, query: &Query) {
    when = query.input_is(0, Type::PTR);
    action = legal;
}

rule add_0<T: Word>(inst: lir::Add<T> | lir::Sub<T> | lir::Mul<T>) {
    action = legal;
}

rule widen_add<T: Narrow>(inst: lir::Add<T>) {
    replace = lir::Trunc<T>(lir::Add<Type::I32>(lir::Zext<Type::I32>(inst.lhs), lir::Zext<Type::I32>(inst.rhs)));
}

rule widen_sub<T: Narrow>(inst: lir::Sub<T>) {
    replace = lir::Trunc<T>(lir::Sub<Type::I32>(lir::Zext<Type::I32>(inst.lhs), lir::Zext<Type::I32>(inst.rhs)));
}

rule widen_mul<T: Narrow>(inst: lir::Mul<T>) {
    replace = lir::Trunc<T>(lir::Mul<Type::I32>(lir::Zext<Type::I32>(inst.lhs), lir::Zext<Type::I32>(inst.rhs)));
}

rule and_0(inst: lir::And<Type::BOOL> | lir::Or<Type::BOOL> | lir::Xor<Type::BOOL>) {
    action = legal;
}

rule and_1<T: Word>(inst: lir::And<T> | lir::Or<T> | lir::Xor<T>) {
    action = legal;
}

rule widen_and<T: Narrow>(inst: lir::And<T>) {
    replace = lir::Trunc<T>(lir::And<Type::I32>(lir::Zext<Type::I32>(inst.lhs), lir::Zext<Type::I32>(inst.rhs)));
}

rule widen_or<T: Narrow>(inst: lir::Or<T>) {
    replace = lir::Trunc<T>(lir::Or<Type::I32>(lir::Zext<Type::I32>(inst.lhs), lir::Zext<Type::I32>(inst.rhs)));
}

rule widen_xor<T: Narrow>(inst: lir::Xor<T>) {
    replace = lir::Trunc<T>(lir::Xor<Type::I32>(lir::Zext<Type::I32>(inst.lhs), lir::Zext<Type::I32>(inst.rhs)));
}

rule shl_0<T: Word>(inst: lir::Shl<T> | lir::Lshr<T> | lir::Ashr<T> | lir::Rotl<T> | lir::Rotr<T> | lir::Sdiv<T> | lir::Udiv<T> | lir::Srem<T> | lir::Urem<T>) {
    action = legal;
}

rule icmp_0<T: WordOrPtr>(inst: lir::Icmp<T>) {
    action = legal;
}

rule fcmp_0<T: ScalarFloat>(inst: lir::Fcmp<T>) {
    action = legal;
}

rule select_0(inst: lir::Select<Type::BOOL>) {
    action = legal;
}

rule select_1<T: WordValue>(inst: lir::Select<T>) {
    action = legal;
}

rule stackaddr_0(inst: lir::StackAddr) {
    action = legal;
}

rule load_0_large<T: Scalar | Type::PTR>(inst: lir::Load<T>, query: &Query) {
    when = !query.signed_offset(32);
    action = expand(load_displacement, inst);
}

rule load_0<T: Scalar | Type::PTR>(inst: lir::Load<T>) {
    action = legal;
}

rule store_0_large<T: Scalar | Type::PTR>(inst: lir::Store<T>, query: &Query) {
    when = !query.signed_offset(32);
    action = expand(store_displacement, inst);
}

rule store_0<T: Scalar | Type::PTR>(inst: lir::Store<T>) {
    action = legal;
}

rule constant_0(inst: lir::Constant<Type::BOOL>) {
    action = legal;
}

rule constant_1<T: IntOrPtr>(inst: lir::Constant<T>) {
    action = legal;
}

rule ieqz_0<T: Word>(inst: lir::Ieqz<T>) {
    action = legal;
}

rule fneg32(inst: lir::Fneg<Type::F32>) {
    action = expand(fneg_bits32, inst);
}

rule fabs32(inst: lir::Fabs<Type::F32>) {
    action = expand(fabs_bits32, inst);
}

rule fneg64(inst: lir::Fneg<Type::F64>) {
    action = expand(fneg_bits64, inst);
}

rule fabs64(inst: lir::Fabs<Type::F64>) {
    action = expand(fabs_bits64, inst);
}

rule sitofp_0<T: ScalarFloat, U: Word>(inst: lir::Sitofp<T, U>) {
    action = legal;
}

rule fptosi_0<T: Word, U: ScalarFloat>(inst: lir::Fptosi<T, U>) {
    action = legal;
}

rule uitofp32<T: ScalarFloat>(inst: lir::Uitofp<T, Type::I32>) {
    replace = unsigned32_to_float<T>(inst.src);
}
rule uitofp64<T: ScalarFloat>(inst: lir::Uitofp<T, Type::I64>) {
    replace = unsigned64_to_float<T>(inst.src);
}
rule fptoui32<T: ScalarFloat>(inst: lir::Fptoui<Type::I32, T>) {
    replace = float_to_unsigned32<T>(inst.src);
}
rule f32_to_u64(inst: lir::Fptoui<Type::I64, Type::F32>) {
    replace = float_to_unsigned64<Type::F32>(
        inst.src, lir::Bitcast<Type::F32>(lir::Constant<Type::I32>(0x5f000000)));
}
rule f64_to_u64(inst: lir::Fptoui<Type::I64, Type::F64>) {
    replace = float_to_unsigned64<Type::F64>(
        inst.src, lir::Bitcast<Type::F64>(lir::Constant<Type::I64>(0x43e0000000000000)));
}

rule fsqrt_0<T: ScalarFloat>(inst: lir::Fsqrt<T>) {
    action = legal;
}

rule fpext_0(inst: lir::Fpext<Type::F64, Type::F32>) {
    action = legal;
}

rule fptrunc_0(inst: lir::Fptrunc<Type::F32, Type::F64>) {
    action = legal;
}

rule zext_0<T: Word>(inst: lir::Zext<T, Type::BOOL>) {
    action = legal;
}

rule zext_1<T: Word, U: Narrow>(inst: lir::Zext<T, U>) {
    action = legal;
}

rule zext_2(inst: lir::Zext<Type::I64, Type::I32>) {
    action = legal;
}

rule sext_0<T: Word, U: Narrow>(inst: lir::Sext<T, U>) {
    action = legal;
}

rule sext_1(inst: lir::Sext<Type::I64, Type::I32>) {
    action = legal;
}

rule trunc_0<T: SmallInt, U: Word>(inst: lir::Trunc<T, U>) {
    action = legal;
}

rule inttoptr_0(inst: lir::Inttoptr<Type::I64>) {
    action = legal;
}

rule ptrtoint_0(inst: lir::Ptrtoint<Type::I64>) {
    action = legal;
}

rule ptradd_0(inst: lir::PtrAdd<Type::I64>) {
    action = legal;
}

rule copy_0<T: Scalar | Type::PTR>(inst: lir::Copy<T>) {
    action = legal;
}

rule bitcast_0(inst: lir::Bitcast<Type::F32, Type::I32>) {
    action = legal;
}

rule bitcast_1(inst: lir::Bitcast<Type::I32, Type::F32>) {
    action = legal;
}

rule bitcast_2(inst: lir::Bitcast<Type::F64, Type::I64>) {
    action = legal;
}

rule bitcast_3(inst: lir::Bitcast<Type::I64, Type::F64>) {
    action = legal;
}

rule fconstant_0<T: ScalarFloat>(inst: lir::Fconstant<T>) {
    action = legal;
}

rule fadd_0<T: ScalarFloat>(inst: lir::Fadd<T> | lir::Fsub<T> | lir::Fmul<T> | lir::Fdiv<T>) {
    action = legal;
}

rule brjt_0(inst: lir::Brjt<Type::I32>) {
    action = legal;
}

// Selection policy belongs to the target; shared rewrites never match implicitly.
rule ctpop32(inst: lir::Ctpop<Type::I32>, target: &Target) {
    action = match target.supports(Instruction::POPCNT32) {
        true => legal,
        _ => expand(popcount32, inst),
    };
}
rule ctpop64(inst: lir::Ctpop<Type::I64>, target: &Target) {
    action = match target.supports(Instruction::POPCNT64) {
        true => legal,
        _ => expand(popcount64, inst),
    };
}
rule ctlz32(inst: lir::Ctlz<Type::I32>) {
    action = expand(leading_zeros32, inst);
}
rule ctlz64(inst: lir::Ctlz<Type::I64>) {
    action = expand(leading_zeros64, inst);
}
rule cttz<T: Word>(inst: lir::Cttz<T>) {
    action = expand(trailing_zeros, inst);
}

rewrite load_displacement<T: Scalar | Type::PTR>(inst: lir::Load<T>)
    = rust("crate::target::x86_64::lowering::legalize::displacement");
rewrite store_displacement<T: Scalar | Type::PTR>(inst: lir::Store<T>)
    = rust("crate::target::x86_64::lowering::legalize::displacement");
