import "../../defs/type_sets.spec";

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
