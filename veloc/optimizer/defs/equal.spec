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

rule(root: mir::Select) {
    where T: ScalarInteger {
        case root<T>(condition, 1, 0) => mir::ExtendU<T>(condition);
        case root<T>(condition, mir::Wrap<T>(x), mir::Wrap<T>(y)) if type_of(x) == type_of(y)
            => mir::Wrap<T>(mir::Select<type_of(x)>(condition, x, y));
        case root<T>(condition, mir::Wrap<T>(x), c) if is_const(c)
            => mir::Wrap<T>(mir::Select<type_of(x)>(condition, x, mir::ExtendU<type_of(x)>(c)));
        case root<T>(condition, c, mir::Wrap<T>(x)) if is_const(c)
            => mir::Wrap<T>(mir::Select<type_of(x)>(condition, mir::ExtendU<type_of(x)>(c), x));
    }

    case root<Type::BOOL>(condition, true, false) => condition;

    where T: Any {
        // Conditions are boolean; branches and the result share T. These folds do not
        // need constant branches and also apply to pointers, floats and vectors.
        case root<T>(true, x, y) => x;
        case root<T>(false, x, y) => y;
        case root<T>(condition, x, x) => x;
    }
}

rule(root: mir::IAdd) {
    where T: ScalarInteger {
        case root<T>(mir::ISub(x, y), y) => x;
        case root<T>(x, mir::IXor(x, -1)) => -1;
        case root<T>(mir::IAdd(x, y), z) if is_const(y) => mir::IAdd(x, mir::IAdd(y, z));
        case root<T>(mir::IMul(x, y), mir::IMul(x, z)) => mir::IMul(x, mir::IAdd(y, z));
    }
}

rule(root: mir::ISub) {
    where T: ScalarInteger {
        case root<T>(x, c) if is_const(c) => mir::IAdd<T>(x, mir::INeg<T>(c));
        case root<T>(x, x) => 0;
        case root<T>(x, 0) => x;
        case root<T>(mir::IAdd(x, y), x) => y;
        case root<T>(x, mir::ISub(x, y)) => y;
        case root<T>(mir::IMul(x, y), mir::IMul(x, z)) => mir::IMul(x, mir::ISub(y, z));
    }
}

rule(root: mir::IMul) {
    where T: ScalarInteger {
        case root<T>(mir::IMul(x, y), z) if is_const(y) => mir::IMul(x, mir::IMul(y, z));
    }
}

rule(root: mir::IAnd) {
    where T: ScalarInteger {
        // A contiguous interval [low, high] becomes an unsigned distance check.
        case root<T>(mir::ExtendU<T>(mir::Icmp<Type::BOOL>(IntCC::GeS, x, low)),
              mir::ExtendU<T>(mir::Icmp<Type::BOOL>(IntCC::LeS, x, high)))
            if signed(low) <= signed(high)
            => mir::ExtendU<T>(mir::Icmp<Type::BOOL>(IntCC::LeU,
                 mir::ISub<type_of(x)>(x, low), mir::ISub<type_of(x)>(high, low)));
        case root<T>(mir::ExtendU<T>(mir::Icmp<Type::BOOL>(IntCC::GeU, x, low)),
              mir::ExtendU<T>(mir::Icmp<Type::BOOL>(IntCC::LeU, x, high)))
            if unsigned(low) <= unsigned(high)
            => mir::ExtendU<T>(mir::Icmp<Type::BOOL>(IntCC::LeU,
                 mir::ISub<type_of(x)>(x, low), mir::ISub<type_of(x)>(high, low)));
    }

    case root<Type::BOOL>(mir::Icmp<Type::BOOL>(IntCC::GeS, x, low), mir::Icmp<Type::BOOL>(IntCC::LeS, x, high))
        if signed(low) <= signed(high)
        => mir::Icmp<Type::BOOL>(IntCC::LeU, mir::ISub<type_of(x)>(x, low), mir::ISub<type_of(x)>(high, low));
    case root<Type::BOOL>(mir::Icmp<Type::BOOL>(IntCC::GeU, x, low), mir::Icmp<Type::BOOL>(IntCC::LeU, x, high))
        if unsigned(low) <= unsigned(high)
        => mir::Icmp<Type::BOOL>(IntCC::LeU, mir::ISub<type_of(x)>(x, low), mir::ISub<type_of(x)>(high, low));

    where T: ScalarInteger {
        case root<T>(x, mir::IXor(x, -1)) => 0;
        case root<T>(x, mir::IXor(x, y)) => mir::IAnd(x, mir::IXor(y, -1));
        case root<T>(mir::IAnd(x, y), z) if is_const(y) => mir::IAnd(x, mir::IAnd(y, z));
        case root<T>(x, mir::IOr(x, y)) => x;
        case root<T>(mir::IOr(x, y), mir::IOr(x, z)) => mir::IOr(x, mir::IAnd(y, z));
        case root<T>(mir::IShl(x, amount), mir::IShl(y, amount)) => mir::IShl(mir::IAnd(x, y), amount);
        case root<T>(mir::IShrU(x, amount), mir::IShrU(y, amount)) => mir::IShrU(mir::IAnd(x, y), amount);
        case root<T>(mir::IShrS(x, amount), mir::IShrS(y, amount)) => mir::IShrS(mir::IAnd(x, y), amount);
        case root<T>(mir::IRotl(x, amount), mir::IRotl(y, amount)) => mir::IRotl(mir::IAnd(x, y), amount);
        case root<T>(mir::IRotr(x, amount), mir::IRotr(y, amount)) => mir::IRotr(mir::IAnd(x, y), amount);
    }
}

rule(root: mir::IOr) {
    where T: ScalarInteger {
        case root<T>(x, mir::IXor(x, -1)) => -1;
        case root<T>(x, mir::IXor(x, y)) => mir::IOr(x, y);
        case root<T>(mir::IOr(x, y), z) if is_const(y) => mir::IOr(x, mir::IOr(y, z));
        case root<T>(x, mir::IAnd(x, y)) => x;
        case root<T>(mir::IAnd(x, y), mir::IAnd(x, z)) => mir::IAnd(x, mir::IOr(y, z));
        case root<T>(mir::IShl(x, amount), mir::IShl(y, amount)) => mir::IShl(mir::IOr(x, y), amount);
        case root<T>(mir::IShrU(x, amount), mir::IShrU(y, amount)) => mir::IShrU(mir::IOr(x, y), amount);
        case root<T>(mir::IShrS(x, amount), mir::IShrS(y, amount)) => mir::IShrS(mir::IOr(x, y), amount);
        case root<T>(mir::IRotl(x, amount), mir::IRotl(y, amount)) => mir::IRotl(mir::IOr(x, y), amount);
        case root<T>(mir::IRotr(x, amount), mir::IRotr(y, amount)) => mir::IRotr(mir::IOr(x, y), amount);
    }
}

rule(root: mir::IXor) {
    where T: ScalarInteger {
        case root<T>(x, x) => 0;
        case root<T>(mir::IXor(x, y), y) => x;
        case root<T>(mir::IXor(x, y), z) if is_const(y) => mir::IXor(x, mir::IXor(y, z));
        case root<T>(x, mir::IXor(x, -1)) => -1;
        case root<T>(mir::IAnd(x, y), mir::IAnd(x, z)) => mir::IAnd(x, mir::IXor(y, z));
        case root<T>(mir::IShl(x, amount), mir::IShl(y, amount)) => mir::IShl(mir::IXor(x, y), amount);
        case root<T>(mir::IShrU(x, amount), mir::IShrU(y, amount)) => mir::IShrU(mir::IXor(x, y), amount);
        case root<T>(mir::IShrS(x, amount), mir::IShrS(y, amount)) => mir::IShrS(mir::IXor(x, y), amount);
        case root<T>(mir::IRotl(x, amount), mir::IRotl(y, amount)) => mir::IRotl(mir::IXor(x, y), amount);
        case root<T>(mir::IRotr(x, amount), mir::IRotr(y, amount)) => mir::IRotr(mir::IXor(x, y), amount);
    }
}

rule(root: mir::IShl) {
    where T: ScalarInteger {
        case root<T>(x, 0) => x;
        case root<T>(0, amount) => 0;
    }
}

rule(root: mir::IShrU) {
    where T: ScalarInteger {
        case root<T>(x, 0) => x;
        case root<T>(0, amount) => 0;
    }
}

rule(root: mir::IShrS) {
    where T: ScalarInteger {
        case root<T>(x, 0) => x;
        case root<T>(0, amount) => 0;
    }
}

rule(root: mir::IRotl) {
    where T: ScalarInteger {
        case root<T>(mir::IRotr(x, amount), amount) => x;
        case root<T>(x, 0) => x;
        case root<T>(0, amount) => 0;
    }
}

rule(root: mir::IRotr) {
    where T: ScalarInteger {
        case root<T>(mir::IRotl(x, amount), amount) => x;
        case root<T>(x, 0) => x;
        case root<T>(0, amount) => 0;
    }
}

rule(root: mir::Icmp) {
    // Attributes occupy their declared operand positions. Captured attributes keep
    // their declared type, and construction checks the complete instruction contract.
    // Comparing an integer value with itself depends only on the predicate.
    case root<Type::BOOL>(IntCC::Eq, x, x) => true;
    case root<Type::BOOL>(IntCC::Ne, x, x) => false;
    case root<Type::BOOL>(IntCC::LtS, x, x) => false;
    case root<Type::BOOL>(IntCC::LtU, x, x) => false;
    case root<Type::BOOL>(IntCC::GtS, x, x) => false;
    case root<Type::BOOL>(IntCC::GtU, x, x) => false;
    case root<Type::BOOL>(IntCC::LeS, x, x) => true;
    case root<Type::BOOL>(IntCC::LeU, x, x) => true;
    case root<Type::BOOL>(IntCC::GeS, x, x) => true;
    case root<Type::BOOL>(IntCC::GeU, x, x) => true;

    where T: ScalarInteger {
        case root<Type::BOOL>(kind, mir::ISub<T>(x, y), 0)
            if kind == IntCC::Eq || kind == IntCC::Ne
            => mir::Icmp<Type::BOOL>(kind, x, y);
    }

    where W: ScalarInteger {
        case root<Type::BOOL>(IntCC::Ne, mir::ExtendU<W>(x), 0) if type_of(x) == Type::BOOL => x;
        case root<Type::BOOL>(IntCC::Eq, mir::ExtendU<W>(x), 0) if type_of(x) == Type::BOOL
            => mir::Select<Type::BOOL>(x, false, true);
        case root<Type::BOOL>(kind, mir::ExtendU<W>(x), 0) if kind == IntCC::Eq || kind == IntCC::Ne
            => mir::Icmp<Type::BOOL>(kind, x, 0);
        case root<Type::BOOL>(kind, mir::ExtendS<W>(x), 0) if kind == IntCC::Eq || kind == IntCC::Ne
            => mir::Icmp<Type::BOOL>(kind, x, 0);
        // Both signed inputs are narrower than W, so their difference cannot overflow W.
        case root<Type::BOOL>(kind, mir::ISub<W>(mir::ExtendS<W>(x), mir::ExtendS<W>(y)), 0)
            if kind == IntCC::LtS || kind == IntCC::LeS || kind == IntCC::GtS || kind == IntCC::GeS
            => mir::Icmp<Type::BOOL>(kind, mir::ExtendS<W>(x), mir::ExtendS<W>(y));
    }

    where T: ScalarInteger {
        // Shift a constant mask to test the original bits. Costing decides whether
        // shared intermediate computations make the replacement worthwhile.
        case root<Type::BOOL>(kind, mir::IAnd<T>(mir::IShrU<T>(x, amount), mask), 0)
            if (kind == IntCC::Eq || kind == IntCC::Ne) && unsigned(mask) <= low_mask(bits(T) - shift_amount(amount))
            => mir::Icmp<Type::BOOL>(kind, mir::IAnd<T>(x, mir::IShl<T>(mask, amount)), 0);
        case root<Type::BOOL>(kind, mir::IAnd<T>(mir::IShrS<T>(x, amount), mask), 0)
            if (kind == IntCC::Eq || kind == IntCC::Ne) && unsigned(mask) <= low_mask(bits(T) - shift_amount(amount))
            => mir::Icmp<Type::BOOL>(kind, mir::IAnd<T>(x, mir::IShl<T>(mask, amount)), 0);
    }
}

rule(root: mir::IEqz) {
    where W: ScalarInteger {
        // Widening preserves zero. Peel casts one at a time; booleans terminate the
        // chain as an inverted selection because IEqz accepts integer operands.
        case root<Type::BOOL>(mir::ExtendU<W>(x)) if type_of(x) == Type::BOOL => mir::Select<Type::BOOL>(x, false, true);
        case root<Type::BOOL>(mir::ExtendU<W>(x)) => mir::IEqz<Type::BOOL>(x);
        case root<Type::BOOL>(mir::ExtendS<W>(x)) => mir::IEqz<Type::BOOL>(x);
    }

    where T: ScalarInteger {
        case root<Type::BOOL>(mir::IAnd<T>(mir::IShrU<T>(x, amount), mask))
            if unsigned(mask) <= low_mask(bits(T) - shift_amount(amount))
            => mir::IEqz<Type::BOOL>(mir::IAnd<T>(x, mir::IShl<T>(mask, amount)));
        case root<Type::BOOL>(mir::IAnd<T>(mir::IShrS<T>(x, amount), mask))
            if unsigned(mask) <= low_mask(bits(T) - shift_amount(amount))
            => mir::IEqz<Type::BOOL>(mir::IAnd<T>(x, mir::IShl<T>(mask, amount)));
    }
}

rule(root: mir::Wrap) {
    where T: Integer, W: Integer {
        // The matched casts already establish equal vector shapes and a wider middle
        // type. Construction validates the resulting cast's complete type contract.
        case root<T>(mir::ExtendU<W>(x)) if type_of(x) == T => x;
        case root<T>(mir::ExtendS<W>(x)) if type_of(x) == T => x;
        case root<T>(mir::ExtendU<W>(x)) if bits(type_of(x)) < bits(T) => mir::ExtendU<T>(x);
        case root<T>(mir::ExtendS<W>(x)) if bits(type_of(x)) < bits(T) => mir::ExtendS<T>(x);
        case root<T>(mir::ExtendU<W>(x)) if bits(type_of(x)) > bits(T) => mir::Wrap<T>(x);
        case root<T>(mir::ExtendS<W>(x)) if bits(type_of(x)) > bits(T) => mir::Wrap<T>(x);
    }
}

rule(root: mir::IntToPtr) {
    where I: ScalarInteger {
        // The layout determines whether the round trip loses any bits. Missing layout
        // means the predicate cannot be established, so neither host applies the rule.
        case root<Type::PTR>(mir::PtrToInt<I>(p)) if bits(I) >= pointer_bits() => p;
    }
}

rule(root: mir::PtrToInt) {
    where I: ScalarInteger {
        case root<I>(mir::IntToPtr<Type::PTR>(x)) if type_of(x) == I && bits(I) <= pointer_bits() => x;
    }
}

rule(root: mir::PtrOffset) {
    case root<Type::PTR>(ptr, 0) => ptr;
}
