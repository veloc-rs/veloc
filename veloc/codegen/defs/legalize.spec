import "legalize_types.spec";

// Reusable value construction and explicit node rewrites. Functions do not replace roots.
// Fixed-width plans are straight-line: no runtime graph construction loop.
typeset BitWord = Type::I32 | Type::I64;

fn shift_fill<T: BitWord>(x: T, shift: T) -> T {
    lir::Or<T>(x, lir::Lshr<T>(x, shift))
}

fn low_bit<T: BitWord>(x: T) -> T {
    lir::And<T>(x, lir::Sub<T>(lir::Constant<T>(0), x))
}

fn low_mask<T: BitWord>(x: T) -> T {
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
rewrite trailing_zeros<T: BitWord>(inst: lir::Cttz<T>) {
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
