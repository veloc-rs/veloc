import "../../defs/type_sets.spec";

// Directed search rules over modular integers. Every matching case contributes
// an equality; cases are not ordered alternatives. These equalities are reviewed,
// not automatically proven by the pattern/type checker.
rule<T: ScalarInteger>(root: mir::ISub<T>) {
    case (x, x) => 0;
    case (x, 0) => x;
    case (mir::IAdd(x, y), x) => y;
    case (x, mir::ISub(x, y)) => y;
}

// Cancellation is valid for wrapping integers, including signed overflow.
rule<T: ScalarInteger>(root: mir::IAdd<T>) {
    case (mir::ISub(x, y), y) => x;
    case (x, mir::IXor(x, -1)) => -1;
}

// -1 denotes all bits set at the instantiated integer width.
rule<T: ScalarInteger>(root: mir::IXor<T>) {
    case (x, x) => 0;
    case (mir::IXor(x, y), y) => x;
    case (x, mir::IXor(x, -1)) => -1;
}

rule<T: ScalarInteger>(root: mir::IAnd<T>) {
    case (x, mir::IXor(x, -1)) => 0;
    case (x, mir::IXor(x, y)) => mir::IAnd(x, mir::IXor(y, -1));
}

rule<T: ScalarInteger>(root: mir::IOr<T>) {
    case (x, mir::IXor(x, -1)) => -1;
    case (x, mir::IXor(x, y)) => mir::IOr(x, y);
}

rule<T: ScalarInteger>(root: mir::IRotr<T>) {
    case (mir::IRotl(x, amount), amount) => x;
}

rule<T: ScalarInteger>(root: mir::IRotl<T>) {
    case (mir::IRotr(x, amount), amount) => x;
}

// Templates and separate groups contribute to the same opcode query program.
template associate(operation: ident) {
    rule<T: ScalarInteger>(root: operation<T>) {
        case (operation(x, y), z) => operation(x, operation(y, z));
    }
}
expand associate(mir::IAdd);
expand associate(mir::IMul);
expand associate(mir::IAnd);
expand associate(mir::IOr);
expand associate(mir::IXor);

template absorb(outer: ident, inner: ident) {
    rule<T: ScalarInteger>(root: outer<T>) {
        case (x, inner(x, y)) => x;
    }
}
expand absorb(mir::IAnd, mir::IOr);
expand absorb(mir::IOr, mir::IAnd);

// Factor instead of eagerly distributing: the e-graph retains the original
// expression, while this direction avoids multiplying intermediate candidates.
template factor(outer: ident, inner: ident) {
    rule<T: ScalarInteger>(root: outer<T>) {
        case (inner(x, y), inner(x, z)) => inner(x, outer(y, z));
    }
}
expand factor(mir::IAdd, mir::IMul);
expand factor(mir::ISub, mir::IMul);
expand factor(mir::IOr, mir::IAnd);
expand factor(mir::IXor, mir::IAnd);
expand factor(mir::IAnd, mir::IOr);

template shift(operation: ident) {
    rule<T: ScalarInteger>(root: operation<T>) {
        case (x, 0) => x;
        case (0, amount) => 0;
    }
}
expand shift(mir::IShl);
expand shift(mir::IShrU);
expand shift(mir::IShrS);
expand shift(mir::IRotl);
expand shift(mir::IRotr);

// Each shift below is a homomorphism for bitwise AND/OR/XOR. Keeping the
// amount identical preserves MIR's modulo-width shift semantics.
template factor_shift(bits: ident, shift: ident) {
    rule<T: ScalarInteger>(root: bits<T>) {
        case (shift(x, amount), shift(y, amount)) => shift(bits(x, y), amount);
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
