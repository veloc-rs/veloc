import "../../defs/type_sets.spec";

rule Icmp {
    // Attributes occupy their declared operand positions. Captured attributes keep
    // their declared type, and construction checks the complete instruction contract.
    // Comparing an integer value with itself depends only on the predicate.
    case (IntCC::Eq, x, x) => true;
    case (IntCC::Ne, x, x) => false;
    case (IntCC::LtS, x, x) => false;
    case (IntCC::LtU, x, x) => false;
    case (IntCC::GtS, x, x) => false;
    case (IntCC::GtU, x, x) => false;
    case (IntCC::LeS, x, x) => true;
    case (IntCC::LeU, x, x) => true;
    case (IntCC::GeS, x, x) => true;
    case (IntCC::GeU, x, x) => true;

    where T: ScalarInteger, (_, T, T) -> Type::BOOL {
        case (kind, ISub(x, y), 0)
            if kind == IntCC::Eq || kind == IntCC::Ne
            => Icmp(kind, x, y);
    }

    where W: ScalarInteger, (_, W, W) -> Type::BOOL {
        case (IntCC::Ne, ExtendU(x: Type::BOOL), 0) => x;
        case (IntCC::Eq, ExtendU(x: Type::BOOL), 0)
            => Select(x, false, true);
        case (kind, ExtendU(x), 0) if kind == IntCC::Eq || kind == IntCC::Ne
            => Icmp(kind, x, 0);
        case (kind, ExtendS(x), 0) if kind == IntCC::Eq || kind == IntCC::Ne
            => Icmp(kind, x, 0);
        // Both signed inputs are narrower than W, so their difference cannot overflow W.
        case (kind, ISub(ExtendS(x), ExtendS(y)), 0)
            if kind == IntCC::LtS || kind == IntCC::LeS || kind == IntCC::GtS || kind == IntCC::GeS
            => Icmp(kind, ExtendS(x) -> W, ExtendS(y));
    }

    where T: ScalarInteger, (_, T, T) -> Type::BOOL {
        // Shift a constant mask to test the original bits. Costing decides whether
        // shared intermediate computations make the replacement worthwhile.
        case (kind, IAnd(IShrU(x, amount), mask), 0)
            if (kind == IntCC::Eq || kind == IntCC::Ne) && unsigned(mask) <= low_mask(bits(T) - shift_amount(amount))
            => Icmp(kind, IAnd(x, IShl(mask, amount)), 0);
        case (kind, IAnd(IShrS(x, amount), mask), 0)
            if (kind == IntCC::Eq || kind == IntCC::Ne) && unsigned(mask) <= low_mask(bits(T) - shift_amount(amount))
            => Icmp(kind, IAnd(x, IShl(mask, amount)), 0);
    }
}

rule IEqz {
    where W: ScalarInteger, (W) -> Type::BOOL {
        // Widening preserves zero. Peel casts one at a time; booleans terminate the
        // chain as an inverted selection because IEqz accepts integer operands.
        case (ExtendU(x: Type::BOOL)) => Select(x, false, true);
        case (ExtendU(x)) => IEqz(x);
        case (ExtendS(x)) => IEqz(x);
    }

    where T: ScalarInteger, (T) -> Type::BOOL {
        case (IAnd(IShrU(x, amount), mask))
            if unsigned(mask) <= low_mask(bits(T) - shift_amount(amount))
            => IEqz(IAnd(x, IShl(mask, amount)));
        case (IAnd(IShrS(x, amount), mask))
            if unsigned(mask) <= low_mask(bits(T) - shift_amount(amount))
            => IEqz(IAnd(x, IShl(mask, amount)));
    }
}
