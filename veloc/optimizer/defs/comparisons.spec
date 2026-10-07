import "../../defs/type_sets.spec";

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

rule(root: mir::PtrOffset<Type::PTR>) {
    case (ptr, 0) => ptr;
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
rule<T: ScalarInteger>(root: mir::IEqz<Type::BOOL>) {
    case (mir::IAnd<T>(mir::IShrU<T>(x, amount), mask))
        if unsigned(mask) <= low_mask(bits(T) - shift_amount(amount))
        => mir::IEqz<Type::BOOL>(mir::IAnd<T>(x, mir::IShl<T>(mask, amount)));
    case (mir::IAnd<T>(mir::IShrS<T>(x, amount), mask))
        if unsigned(mask) <= low_mask(bits(T) - shift_amount(amount))
        => mir::IEqz<Type::BOOL>(mir::IAnd<T>(x, mir::IShl<T>(mask, amount)));
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
