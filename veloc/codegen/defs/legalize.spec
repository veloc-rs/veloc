import "../../defs/type_sets.spec";

// Legalization policy domains, not alternative definitions of common types.
typeset Narrow = Type::I8 | Type::I16;
typeset Word = Type::I32 | Type::I64;
typeset IntOrPtr = ScalarInteger | Type::PTR;
typeset WordValue = Type::BOOL | Word | ScalarFloat | Type::PTR;
typeset WordOrPtr = Word | Type::PTR;
typeset SmallInt = Narrow | Type::I32;

// Instruction-local query contract shared by target policies.
type Query = rust("crate::passes::lowering::legalize::Query") {
    trait = rust("crate::passes::lowering::legalize::contracts::Query");
    fn signature(&self, results: sequence(sequence(Type)), inputs: sequence(sequence(Type))) -> bool;
    fn same(&self, indices: sequence(u32)) -> bool;
    fn value_type(&self, result: bool, index: u32) -> Type;
    fn input_is(&self, index: u32, ty: Type) -> bool;
    fn signed_offset(&self, bits: u32) -> bool;
}

type RewriteValue = rust("veloc_lir::Reg");
type RewriteOpcode = rust("veloc_lir::GenericOpcode");
type RewriteField = rust("veloc_lir::InstField");
type RewriteContext = rust("crate::passes::lowering::legalize::RewriteContext") {
    trait = rust("crate::passes::lowering::legalize::contracts::ValueRewrite");
    fn emit(&mut self, opcode: RewriteOpcode, ty: Type,
        inputs: sequence(RewriteValue), fields: sequence(RewriteField),
        result: optional(RewriteValue)) -> RewriteValue;
}

rewrite_interface ValueRules {
    contract = RewriteContext;
    emit = emit;
}

// Reusable value construction and explicit node rewrites. Functions do not replace roots.
// Fixed-width plans are straight-line: no runtime graph construction loop.

fn shift_fill<T: Word>(x: T, shift: T) -> T {
    lir::Or<T>(x, lir::Lshr<T>(x, shift))
}

fn low_bit<T: Word>(x: T) -> T {
    lir::And<T>(x, lir::Sub<T>(lir::Constant<T>(0), x))
}

fn low_mask<T: Word>(x: T) -> T {
    lir::Sub<T>(low_bit<T>(x), lir::Constant<T>(1))
}

fn count_bits32(x: Type::I32) -> Type::I32 {
    let pairs_mask = lir::Constant<Type::I32>(0x55555555);
    let nibbles_mask = lir::Constant<Type::I32>(0x33333333);
    let bytes_mask = lir::Constant<Type::I32>(0xf0f0f0f);
    let shifted = lir::Lshr<Type::I32>(x, lir::Constant<Type::I32>(1));
    let pairs = lir::Sub<Type::I32>(x, lir::And<Type::I32>(shifted, pairs_mask));
    let high = lir::Lshr<Type::I32>(pairs, lir::Constant<Type::I32>(2));
    let nibbles = lir::Add<Type::I32>(
        lir::And<Type::I32>(pairs, nibbles_mask),
        lir::And<Type::I32>(high, nibbles_mask)
    );
    let upper = lir::Lshr<Type::I32>(nibbles, lir::Constant<Type::I32>(4));
    let bytes = lir::And<Type::I32>(lir::Add<Type::I32>(nibbles, upper), bytes_mask);
    let sum8 = lir::Add<Type::I32>(bytes, lir::Lshr<Type::I32>(bytes, lir::Constant<Type::I32>(8)));
    let sum16 = lir::Add<Type::I32>(sum8, lir::Lshr<Type::I32>(sum8, lir::Constant<Type::I32>(16)));
    lir::And<Type::I32>(sum16, lir::Constant<Type::I32>(63));
}

rewrite popcount32(inst: lir::Ctpop<Type::I32>) {
    replace = count_bits32(inst.src);
}

rewrite leading_zeros32(inst: lir::Ctlz<Type::I32>) {
    replace {
        let fill1 = shift_fill<Type::I32>(inst.src, lir::Constant<Type::I32>(1));
        let fill2 = shift_fill<Type::I32>(fill1, lir::Constant<Type::I32>(2));
        let fill4 = shift_fill<Type::I32>(fill2, lir::Constant<Type::I32>(4));
        let fill8 = shift_fill<Type::I32>(fill4, lir::Constant<Type::I32>(8));
        let fill16 = shift_fill<Type::I32>(fill8, lir::Constant<Type::I32>(16));
        lir::Sub<Type::I32>(lir::Constant<Type::I32>(32), lir::Ctpop<Type::I32>(fill16));
    }
}

fn count_bits64(x: Type::I64) -> Type::I64 {
    let pairs_mask = lir::Constant<Type::I64>(0x5555555555555555);
    let nibbles_mask = lir::Constant<Type::I64>(0x3333333333333333);
    let bytes_mask = lir::Constant<Type::I64>(0xf0f0f0f0f0f0f0f);
    let shifted = lir::Lshr<Type::I64>(x, lir::Constant<Type::I64>(1));
    let pairs = lir::Sub<Type::I64>(x, lir::And<Type::I64>(shifted, pairs_mask));
    let high = lir::Lshr<Type::I64>(pairs, lir::Constant<Type::I64>(2));
    let nibbles = lir::Add<Type::I64>(
        lir::And<Type::I64>(pairs, nibbles_mask),
        lir::And<Type::I64>(high, nibbles_mask)
    );
    let upper = lir::Lshr<Type::I64>(nibbles, lir::Constant<Type::I64>(4));
    let bytes = lir::And<Type::I64>(lir::Add<Type::I64>(nibbles, upper), bytes_mask);
    let sum8 = lir::Add<Type::I64>(bytes, lir::Lshr<Type::I64>(bytes, lir::Constant<Type::I64>(8)));
    let sum16 = lir::Add<Type::I64>(sum8, lir::Lshr<Type::I64>(sum8, lir::Constant<Type::I64>(16)));
    let sum32 = lir::Add<Type::I64>(sum16, lir::Lshr<Type::I64>(sum16, lir::Constant<Type::I64>(32)));
    lir::And<Type::I64>(sum32, lir::Constant<Type::I64>(127));
}

rewrite popcount64(inst: lir::Ctpop<Type::I64>) {
    replace = count_bits64(inst.src);
}

rewrite leading_zeros64(inst: lir::Ctlz<Type::I64>) {
    replace {
        let fill1 = shift_fill<Type::I64>(inst.src, lir::Constant<Type::I64>(1));
        let fill2 = shift_fill<Type::I64>(fill1, lir::Constant<Type::I64>(2));
        let fill4 = shift_fill<Type::I64>(fill2, lir::Constant<Type::I64>(4));
        let fill8 = shift_fill<Type::I64>(fill4, lir::Constant<Type::I64>(8));
        let fill16 = shift_fill<Type::I64>(fill8, lir::Constant<Type::I64>(16));
        let fill32 = shift_fill<Type::I64>(fill16, lir::Constant<Type::I64>(32));
        lir::Sub<Type::I64>(lir::Constant<Type::I64>(64), lir::Ctpop<Type::I64>(fill32));
    }
}

// Wrapping subtraction yields all ones for zero, so ctpop returns the width.
rewrite trailing_zeros<T: Word>(inst: lir::Cttz<T>) {
    replace {
        lir::Ctpop<T>(low_mask<T>(inst.src));
    }
}

// IEEE floating-point sign operations preserve every other bit, including NaN payloads.

rewrite fneg_bits32(inst: lir::Fneg<Type::F32>) {
    replace {
        let bits = lir::Bitcast<Type::I32>(inst.src);
        let mask = lir::Constant<Type::I32>(2147483648);
        let changed = lir::Xor<Type::I32>(bits, mask);
        lir::Bitcast<Type::F32>(changed);
    }
}

rewrite fabs_bits32(inst: lir::Fabs<Type::F32>) {
    replace {
        let bits = lir::Bitcast<Type::I32>(inst.src);
        let mask = lir::Constant<Type::I32>(2147483647);
        let changed = lir::And<Type::I32>(bits, mask);
        lir::Bitcast<Type::F32>(changed);
    }
}

rewrite fneg_bits64(inst: lir::Fneg<Type::F64>) {
    replace {
        let bits = lir::Bitcast<Type::I64>(inst.src);
        let mask = lir::Constant<Type::I64>(-9223372036854775808);
        let changed = lir::Xor<Type::I64>(bits, mask);
        lir::Bitcast<Type::F64>(changed);
    }
}

rewrite fabs_bits64(inst: lir::Fabs<Type::F64>) {
    replace {
        let bits = lir::Bitcast<Type::I64>(inst.src);
        let mask = lir::Constant<Type::I64>(9223372036854775807);
        let changed = lir::And<Type::I64>(bits, mask);
        lir::Bitcast<Type::F64>(changed);
    }
}

// Signed-conversion fallback algorithms. Targets choose when to use them.
fn unsigned32_to_float<T: ScalarFloat>(x: Type::I32) -> T {
    lir::Sitofp<T>(lir::Zext<Type::I64>(x))
}

fn unsigned64_to_float<T: ScalarFloat>(x: Type::I64) -> T {
    // Preserve a sticky low bit before rounding, then restore the factor of two.
    let half = lir::Lshr<Type::I64>(x, lir::Constant<Type::I64>(1));
    let low = lir::And<Type::I64>(x, lir::Constant<Type::I64>(1));
    let rounded = lir::Or<Type::I64>(half, low);
    let converted = lir::Sitofp<T>(rounded);
    let doubled = lir::Fadd<T>(converted, converted);
    let direct = lir::Sitofp<T>(x);
    let high = lir::Icmp<Type::BOOL>(x, lir::Constant<Type::I64>(0), IntCC::LtS);
    lir::Select<T>(high, doubled, direct)
}

fn float_to_unsigned32<T: ScalarFloat>(x: T) -> Type::I32 {
    lir::Trunc<Type::I32>(lir::Fptosi<Type::I64>(x))
}

fn float_to_unsigned64<T: ScalarFloat>(x: T, threshold: T) -> Type::I64 {
    // For the upper half, subtract 2^63 and restore its bit after conversion.
    // These fallbacks retain the existing non-trapping conversion contract;
    // strict floating-point exception behavior needs a separate legalization.
    let high = lir::Fcmp<Type::BOOL>(x, threshold, FloatCC::Ge);
    let reduced = lir::Fsub<T>(x, threshold);
    let converted = lir::Fptosi<Type::I64>(reduced);
    let restored = lir::Xor<Type::I64>(converted, lir::Constant<Type::I64>(-9223372036854775808));
    let direct = lir::Fptosi<Type::I64>(x);
    lir::Select<Type::I64>(high, restored, direct)
}
