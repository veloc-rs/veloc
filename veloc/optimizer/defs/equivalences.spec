import "../../defs/type_sets.spec";

// Conditions are boolean; branches and the result share T. These folds do not
// need constant branches and also apply to pointers, floats and vectors.
rule<T: Any>(root: mir::Select<T>) {
    case (true, x, y) => x;
    case (false, x, y) => y;
    case (condition, x, x) => x;
}

// Rules are grouped by root opcode. Root-local identities fold without
// constructing an operation; other cases add alternatives to the e-graph.
// These equalities are reviewed, not proven by the pattern/type checker.
// Arithmetic is modular; -1 denotes all bits set at the integer width.
// Factor instead of distributing to avoid multiplying intermediate candidates.
// Shift factoring keeps the amount identical under modulo-width semantics.

rule<T: ScalarInteger>(root: mir::IAdd<T>) {
    case (mir::ISub(x, y), y) => x;
    case (x, mir::IXor(x, -1)) => -1;
    case (mir::IAdd(x, y), z) => mir::IAdd(x, mir::IAdd(y, z));
    case (mir::IMul(x, y), mir::IMul(x, z)) => mir::IMul(x, mir::IAdd(y, z));
}

rule<T: ScalarInteger>(root: mir::ISub<T>) {
    case (x, x) => 0;
    case (x, 0) => x;
    case (mir::IAdd(x, y), x) => y;
    case (x, mir::ISub(x, y)) => y;
    case (mir::IMul(x, y), mir::IMul(x, z)) => mir::IMul(x, mir::ISub(y, z));
}

rule<T: ScalarInteger>(root: mir::IMul<T>) {
    case (mir::IMul(x, y), z) => mir::IMul(x, mir::IMul(y, z));
}

rule<T: ScalarInteger>(root: mir::IAnd<T>) {
    case (x, mir::IXor(x, -1)) => 0;
    case (x, mir::IXor(x, y)) => mir::IAnd(x, mir::IXor(y, -1));
    case (mir::IAnd(x, y), z) => mir::IAnd(x, mir::IAnd(y, z));
    case (x, mir::IOr(x, y)) => x;
    case (mir::IOr(x, y), mir::IOr(x, z)) => mir::IOr(x, mir::IAnd(y, z));
    case (mir::IShl(x, amount), mir::IShl(y, amount)) => mir::IShl(mir::IAnd(x, y), amount);
    case (mir::IShrU(x, amount), mir::IShrU(y, amount)) => mir::IShrU(mir::IAnd(x, y), amount);
    case (mir::IShrS(x, amount), mir::IShrS(y, amount)) => mir::IShrS(mir::IAnd(x, y), amount);
    case (mir::IRotl(x, amount), mir::IRotl(y, amount)) => mir::IRotl(mir::IAnd(x, y), amount);
    case (mir::IRotr(x, amount), mir::IRotr(y, amount)) => mir::IRotr(mir::IAnd(x, y), amount);
}

rule<T: ScalarInteger>(root: mir::IOr<T>) {
    case (x, mir::IXor(x, -1)) => -1;
    case (x, mir::IXor(x, y)) => mir::IOr(x, y);
    case (mir::IOr(x, y), z) => mir::IOr(x, mir::IOr(y, z));
    case (x, mir::IAnd(x, y)) => x;
    case (mir::IAnd(x, y), mir::IAnd(x, z)) => mir::IAnd(x, mir::IOr(y, z));
    case (mir::IShl(x, amount), mir::IShl(y, amount)) => mir::IShl(mir::IOr(x, y), amount);
    case (mir::IShrU(x, amount), mir::IShrU(y, amount)) => mir::IShrU(mir::IOr(x, y), amount);
    case (mir::IShrS(x, amount), mir::IShrS(y, amount)) => mir::IShrS(mir::IOr(x, y), amount);
    case (mir::IRotl(x, amount), mir::IRotl(y, amount)) => mir::IRotl(mir::IOr(x, y), amount);
    case (mir::IRotr(x, amount), mir::IRotr(y, amount)) => mir::IRotr(mir::IOr(x, y), amount);
}

rule<T: ScalarInteger>(root: mir::IXor<T>) {
    case (x, x) => 0;
    case (mir::IXor(x, y), y) => x;
    case (x, mir::IXor(x, -1)) => -1;
    case (mir::IXor(x, y), z) => mir::IXor(x, mir::IXor(y, z));
    case (mir::IAnd(x, y), mir::IAnd(x, z)) => mir::IAnd(x, mir::IXor(y, z));
    case (mir::IShl(x, amount), mir::IShl(y, amount)) => mir::IShl(mir::IXor(x, y), amount);
    case (mir::IShrU(x, amount), mir::IShrU(y, amount)) => mir::IShrU(mir::IXor(x, y), amount);
    case (mir::IShrS(x, amount), mir::IShrS(y, amount)) => mir::IShrS(mir::IXor(x, y), amount);
    case (mir::IRotl(x, amount), mir::IRotl(y, amount)) => mir::IRotl(mir::IXor(x, y), amount);
    case (mir::IRotr(x, amount), mir::IRotr(y, amount)) => mir::IRotr(mir::IXor(x, y), amount);
}

rule<T: ScalarInteger>(root: mir::IShl<T>) {
    case (x, 0) => x;
    case (0, amount) => 0;
}

rule<T: ScalarInteger>(root: mir::IShrU<T>) {
    case (x, 0) => x;
    case (0, amount) => 0;
}

rule<T: ScalarInteger>(root: mir::IShrS<T>) {
    case (x, 0) => x;
    case (0, amount) => 0;
}

rule<T: ScalarInteger>(root: mir::IRotl<T>) {
    case (mir::IRotr(x, amount), amount) => x;
    case (x, 0) => x;
    case (0, amount) => 0;
}

rule<T: ScalarInteger>(root: mir::IRotr<T>) {
    case (mir::IRotl(x, amount), amount) => x;
    case (x, 0) => x;
    case (0, amount) => 0;
}
