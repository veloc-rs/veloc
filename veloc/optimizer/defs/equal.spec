import "../../defs/type_sets.spec";

// Equivalence rules shared by local folding and e-graph search, grouped by root.
// Rule order within each root is preserved. These equalities are reviewed,
// not proven by the pattern/type checker.
// Arithmetic is modular; -1 denotes all bits set at the integer width.
// Factoring and constant reassociation avoid enumerating arbitrary permutations.
// Direct operand/constant reductions also feed the shared local evaluator.

// Conversion and boolean equalities explored by the e-graph.
// Canonical cases also permit equal-cost changes toward the chosen normal form.
// Type arguments name result types; nested nodes bind independent type facts.
rule<T: ScalarInteger>(root: mir::Select<T>) {
    case (condition, 1, 0) => mir::ExtendU<T>(condition);
    case (condition, mir::Wrap<T>(x), mir::Wrap<T>(y)) if type_of(x) == type_of(y)
        => mir::Wrap<T>(mir::Select<type_of(x)>(condition, x, y));
    case (condition, mir::Wrap<T>(x), c) if is_const(c)
        => mir::Wrap<T>(mir::Select<type_of(x)>(condition, x, mir::ExtendU<type_of(x)>(c)));
    case (condition, c, mir::Wrap<T>(x)) if is_const(c)
        => mir::Wrap<T>(mir::Select<type_of(x)>(condition, mir::ExtendU<type_of(x)>(c), x));
}

rule(root: mir::Select<Type::BOOL>) {
    case (condition, true, false) => condition;
}

// Conditions are boolean; branches and the result share T. These folds do not
// need constant branches and also apply to pointers, floats and vectors.
rule<T: Any>(root: mir::Select<T>) {
    case (true, x, y) => x;
    case (false, x, y) => y;
    case (condition, x, x) => x;
}

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

// A contiguous interval [low, high] becomes an unsigned distance check.
rule<T: ScalarInteger>(root: mir::IAnd<T>) {
    case (mir::ExtendU<T>(mir::Icmp<Type::BOOL>(IntCC::GeS, x, low)),
          mir::ExtendU<T>(mir::Icmp<Type::BOOL>(IntCC::LeS, x, high)))
        if signed(low) <= signed(high)
        => mir::ExtendU<T>(mir::Icmp<Type::BOOL>(IntCC::LeU,
             mir::ISub<type_of(x)>(x, low), mir::ISub<type_of(x)>(high, low)));
    case (mir::ExtendU<T>(mir::Icmp<Type::BOOL>(IntCC::GeU, x, low)),
          mir::ExtendU<T>(mir::Icmp<Type::BOOL>(IntCC::LeU, x, high)))
        if unsigned(low) <= unsigned(high)
        => mir::ExtendU<T>(mir::Icmp<Type::BOOL>(IntCC::LeU,
             mir::ISub<type_of(x)>(x, low), mir::ISub<type_of(x)>(high, low)));
}

rule(root: mir::IAnd<Type::BOOL>) {
    case (mir::Icmp<Type::BOOL>(IntCC::GeS, x, low), mir::Icmp<Type::BOOL>(IntCC::LeS, x, high))
        if signed(low) <= signed(high)
        => mir::Icmp<Type::BOOL>(IntCC::LeU, mir::ISub<type_of(x)>(x, low), mir::ISub<type_of(x)>(high, low));
    case (mir::Icmp<Type::BOOL>(IntCC::GeU, x, low), mir::Icmp<Type::BOOL>(IntCC::LeU, x, high))
        if unsigned(low) <= unsigned(high)
        => mir::Icmp<Type::BOOL>(IntCC::LeU, mir::ISub<type_of(x)>(x, low), mir::ISub<type_of(x)>(high, low));
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

// Attributes occupy their declared operand positions. Captured attributes keep
// their declared type, and construction checks the complete instruction contract.
rule<T: ScalarInteger>(root: mir::Icmp<Type::BOOL>) {
    case (kind, mir::ISub<T>(x, y), 0)
        if kind == IntCC::Eq || kind == IntCC::Ne
        => mir::Icmp<Type::BOOL>(kind, x, y);
}

rule<W: ScalarInteger>(root: mir::Icmp<Type::BOOL>) {
    case (IntCC::Ne, mir::ExtendU<W>(x), 0) if type_of(x) == Type::BOOL => x;
    case (IntCC::Eq, mir::ExtendU<W>(x), 0) if type_of(x) == Type::BOOL
        => mir::Select<Type::BOOL>(x, false, true);
    case (kind, mir::ExtendU<W>(x), 0) if kind == IntCC::Eq || kind == IntCC::Ne
        => mir::Icmp<Type::BOOL>(kind, x, 0);
    case (kind, mir::ExtendS<W>(x), 0) if kind == IntCC::Eq || kind == IntCC::Ne
        => mir::Icmp<Type::BOOL>(kind, x, 0);
    // Both signed inputs are narrower than W, so their difference cannot overflow W.
    case (kind, mir::ISub<W>(mir::ExtendS<W>(x), mir::ExtendS<W>(y)), 0)
        if kind == IntCC::LtS || kind == IntCC::LeS || kind == IntCC::GtS || kind == IntCC::GeS
        => mir::Icmp<Type::BOOL>(kind, mir::ExtendS<W>(x), mir::ExtendS<W>(y));
}

// Shift a constant mask to test the original bits. Costing decides whether
// shared intermediate computations make the replacement worthwhile.
rule<T: ScalarInteger>(root: mir::Icmp<Type::BOOL>) {
    case (kind, mir::IAnd<T>(mir::IShrU<T>(x, amount), mask), 0)
        if (kind == IntCC::Eq || kind == IntCC::Ne) && unsigned(mask) <= low_mask(bits(T) - shift_amount(amount))
        => mir::Icmp<Type::BOOL>(kind, mir::IAnd<T>(x, mir::IShl<T>(mask, amount)), 0);
    case (kind, mir::IAnd<T>(mir::IShrS<T>(x, amount), mask), 0)
        if (kind == IntCC::Eq || kind == IntCC::Ne) && unsigned(mask) <= low_mask(bits(T) - shift_amount(amount))
        => mir::Icmp<Type::BOOL>(kind, mir::IAnd<T>(x, mir::IShl<T>(mask, amount)), 0);
}

// Widening preserves zero. Peel casts one at a time; booleans terminate the
// chain as an inverted selection because IEqz accepts integer operands.
rule<W: ScalarInteger>(root: mir::IEqz<Type::BOOL>) {
    case (mir::ExtendU<W>(x)) if type_of(x) == Type::BOOL => mir::Select<Type::BOOL>(x, false, true);
    case (mir::ExtendU<W>(x)) => mir::IEqz<Type::BOOL>(x);
    case (mir::ExtendS<W>(x)) => mir::IEqz<Type::BOOL>(x);
}

rule<T: ScalarInteger>(root: mir::IEqz<Type::BOOL>) {
    case (mir::IAnd<T>(mir::IShrU<T>(x, amount), mask))
        if unsigned(mask) <= low_mask(bits(T) - shift_amount(amount))
        => mir::IEqz<Type::BOOL>(mir::IAnd<T>(x, mir::IShl<T>(mask, amount)));
    case (mir::IAnd<T>(mir::IShrS<T>(x, amount), mask))
        if unsigned(mask) <= low_mask(bits(T) - shift_amount(amount))
        => mir::IEqz<Type::BOOL>(mir::IAnd<T>(x, mir::IShl<T>(mask, amount)));
}

// The matched casts already establish equal vector shapes and a wider middle
// type. Construction validates the resulting cast's complete type contract.
rule<T: Integer, W: Integer>(root: mir::Wrap<T>) {
    case (mir::ExtendU<W>(x)) if type_of(x) == T => x;
    case (mir::ExtendS<W>(x)) if type_of(x) == T => x;
    case (mir::ExtendU<W>(x)) if bits(type_of(x)) < bits(T) => mir::ExtendU<T>(x);
    case (mir::ExtendS<W>(x)) if bits(type_of(x)) < bits(T) => mir::ExtendS<T>(x);
    case (mir::ExtendU<W>(x)) if bits(type_of(x)) > bits(T) => mir::Wrap<T>(x);
    case (mir::ExtendS<W>(x)) if bits(type_of(x)) > bits(T) => mir::Wrap<T>(x);
}

// The layout determines whether the round trip loses any bits. Missing layout
// means the predicate cannot be established, so neither host applies the rule.
rule<I: ScalarInteger>(root: mir::IntToPtr<Type::PTR>) {
    case (mir::PtrToInt<I>(p)) if bits(I) >= pointer_bits() => p;
}

rule<I: ScalarInteger>(root: mir::PtrToInt<I>) {
    case (mir::IntToPtr<Type::PTR>(x)) if type_of(x) == I && bits(I) <= pointer_bits() => x;
}

rule(root: mir::PtrOffset<Type::PTR>) {
    case (ptr, 0) => ptr;
}
