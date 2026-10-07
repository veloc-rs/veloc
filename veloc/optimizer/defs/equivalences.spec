import "../../defs/type_sets.spec";
import "conversions.spec";
import "comparisons.spec";

// Conditions are boolean; branches and the result share T. These folds do not
// need constant branches and also apply to pointers, floats and vectors.
rule<T: Any>(root: mir::Select<T>) {
    case (true, x, y) => x;
    case (false, x, y) => y;
    case (condition, x, x) => x;
}

// Direct operand/constant reductions feed the shared instruction evaluator.
// Other rules run in the e-graph, which retains alternatives for extraction.
// These equalities are reviewed, not proven by the pattern/type checker.
// Arithmetic is modular; -1 denotes all bits set at the integer width.
// Factor instead of distributing to avoid multiplying intermediate candidates.
// Shift factoring keeps the amount identical under modulo-width semantics.
// Reassociate around known constants to expose folding without enumerating
// arbitrary permutations of variable-only associative chains.

rule<T: ScalarInteger>(root: mir::IAdd<T>) {
    case (mir::ISub(x, y), y) => x;
    case (x, mir::IXor(x, -1)) => -1;
    case (mir::IAdd(x, y), z) if is_const(y) => mir::IAdd(x, mir::IAdd(y, z));
    case (mir::IMul(x, y), mir::IMul(x, z)) => mir::IMul(x, mir::IAdd(y, z));
}

rule<T: ScalarInteger>(root: mir::ISub<T>) {
    case (x, c) if is_const(c) => mir::IAdd<T>(x, mir::INeg<T>(c));
    case (x, x) => 0;
    case (x, 0) => x;
    case (mir::IAdd(x, y), x) => y;
    case (x, mir::ISub(x, y)) => y;
    case (mir::IMul(x, y), mir::IMul(x, z)) => mir::IMul(x, mir::ISub(y, z));
}

rule<T: ScalarInteger>(root: mir::IMul<T>) {
    case (mir::IMul(x, y), z) if is_const(y) => mir::IMul(x, mir::IMul(y, z));
}

rule<T: ScalarInteger>(root: mir::IAnd<T>) {
    case (x, mir::IXor(x, -1)) => 0;
    case (x, mir::IXor(x, y)) => mir::IAnd(x, mir::IXor(y, -1));
    case (mir::IAnd(x, y), z) if is_const(y) => mir::IAnd(x, mir::IAnd(y, z));
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
    case (mir::IOr(x, y), z) if is_const(y) => mir::IOr(x, mir::IOr(y, z));
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
    case (mir::IXor(x, y), z) if is_const(y) => mir::IXor(x, mir::IXor(y, z));
    case (x, mir::IXor(x, -1)) => -1;
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
