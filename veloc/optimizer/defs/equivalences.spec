import "../../defs/type_sets.spec";

// Directed search rules over modular integers. These are reviewed equalities,
// not automatically proven by the pattern/type checker.
rule<T: ScalarInteger>(x: T) {
    mir::ISub(x, x) => 0;
}
rule<T: ScalarInteger>(x: T) {
    mir::IXor(x, x) => 0;
}
rule<T: ScalarInteger>(x: T) {
    mir::ISub(x, 0) => x;
}
rule<T: ScalarInteger>(x: T, y: T) {
    mir::ISub(mir::IAdd(x, y), x) => y;
}

// Expansion reuses the ordinary template mechanism, not e-graph-specific syntax.
template associate(operation: ident) {
    rule<T: ScalarInteger>(x: T, y: T, z: T) {
        operation(operation(x, y), z) => operation(x, operation(y, z));
    }
}
expand associate(mir::IAdd);
expand associate(mir::IMul);
expand associate(mir::IAnd);
expand associate(mir::IOr);
expand associate(mir::IXor);

// Cancellation is valid for wrapping integers, including signed overflow.
rule<T: ScalarInteger>(x: T, y: T) {
    mir::IAdd(mir::ISub(x, y), y) => x;
}
rule<T: ScalarInteger>(x: T, y: T) {
    mir::IXor(mir::IXor(x, y), y) => x;
}
rule<T: ScalarInteger>(x: T, y: T) {
    mir::ISub(x, mir::ISub(x, y)) => y;
}

template absorb(outer: ident, inner: ident) {
    rule<T: ScalarInteger>(x: T, y: T) {
        outer(x, inner(x, y)) => x;
    }
}
expand absorb(mir::IAnd, mir::IOr);
expand absorb(mir::IOr, mir::IAnd);

// Factor instead of eagerly distributing: the e-graph retains the original
// expression, while this direction avoids multiplying intermediate candidates.
template factor(outer: ident, inner: ident) {
    rule<T: ScalarInteger>(x: T, y: T, z: T) {
        outer(inner(x, y), inner(x, z)) => inner(x, outer(y, z));
    }
}
expand factor(mir::IAdd, mir::IMul);
expand factor(mir::ISub, mir::IMul);
expand factor(mir::IOr, mir::IAnd);
expand factor(mir::IXor, mir::IAnd);
expand factor(mir::IAnd, mir::IOr);

template shift_identity(operation: ident) {
    rule<T: ScalarInteger>(x: T) {
        operation(x, 0) => x;
    }
}
expand shift_identity(mir::IShl);
expand shift_identity(mir::IShrU);
expand shift_identity(mir::IShrS);
expand shift_identity(mir::IRotl);
expand shift_identity(mir::IRotr);

template shift_zero(operation: ident) {
    rule<T: ScalarInteger>(x: T) {
        operation(0, x) => 0;
    }
}
expand shift_zero(mir::IShl);
expand shift_zero(mir::IShrU);
expand shift_zero(mir::IShrS);
expand shift_zero(mir::IRotl);
expand shift_zero(mir::IRotr);

rule<T: ScalarInteger>(x: T, amount: T) {
    mir::IRotr(mir::IRotl(x, amount), amount) => x;
}
rule<T: ScalarInteger>(x: T, amount: T) {
    mir::IRotl(mir::IRotr(x, amount), amount) => x;
}

// -1 denotes all bits set at the instantiated integer width.
rule<T: ScalarInteger>(x: T) {
    mir::IAnd(x, mir::IXor(x, -1)) => 0;
}
rule<T: ScalarInteger>(x: T) {
    mir::IOr(x, mir::IXor(x, -1)) => -1;
}
rule<T: ScalarInteger>(x: T) {
    mir::IXor(x, mir::IXor(x, -1)) => -1;
}
rule<T: ScalarInteger>(x: T, y: T) {
    mir::IAnd(x, mir::IXor(x, y)) => mir::IAnd(x, mir::IXor(y, -1));
}
rule<T: ScalarInteger>(x: T, y: T) {
    mir::IOr(x, mir::IXor(x, y)) => mir::IOr(x, y);
}
rule<T: ScalarInteger>(x: T) {
    mir::IAdd(x, mir::IXor(x, -1)) => -1;
}

// Each shift below is a homomorphism for bitwise AND/OR/XOR. Keeping the
// amount identical preserves MIR's modulo-width shift semantics.
template factor_shift(bits: ident, shift: ident) {
    rule<T: ScalarInteger>(x: T, y: T, amount: T) {
        bits(shift(x, amount), shift(y, amount)) => shift(bits(x, y), amount);
    }
}
expand factor_shift(mir::IAnd, mir::IShl);
expand factor_shift(mir::IOr, mir::IShl);
expand factor_shift(mir::IXor, mir::IShl);
expand factor_shift(mir::IAnd, mir::IShrU);
expand factor_shift(mir::IOr, mir::IShrU);
expand factor_shift(mir::IXor, mir::IShrU);
expand factor_shift(mir::IAnd, mir::IShrS);
expand factor_shift(mir::IOr, mir::IShrS);
expand factor_shift(mir::IXor, mir::IShrS);
expand factor_shift(mir::IAnd, mir::IRotl);
expand factor_shift(mir::IOr, mir::IRotl);
expand factor_shift(mir::IXor, mir::IRotl);
expand factor_shift(mir::IAnd, mir::IRotr);
expand factor_shift(mir::IOr, mir::IRotr);
expand factor_shift(mir::IXor, mir::IRotr);
