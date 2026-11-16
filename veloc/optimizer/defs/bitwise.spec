import "../../defs/type_sets.spec";

rule IAnd {
    where (Type::BOOL, Type::BOOL) -> Type::BOOL {
        case (x, true) => x;
        case (x, false) => false;
        case (x, x) => x;
    }

    where T: ScalarInteger {
        // Excluding two constants is the complement of the corresponding IOr.
        // With a clear differing bit, masking alone identifies both constants.
        case (Icmp(IntCC::Ne, x: T, a), Icmp(IntCC::Ne, x, b))
            if unsigned(a) < unsigned(b)
                && ((unsigned(b) - unsigned(a)) & (unsigned(b) - unsigned(a) - 1)) == 0
                && (unsigned(a) & (unsigned(b) - unsigned(a))) == 0
            => Icmp(IntCC::Ne, IAnd(x, IXor(ISub(b, a), -1)), a);
        // Otherwise translate a to zero before ignoring the differing bit.
        case (Icmp(IntCC::Ne, x: T, a), Icmp(IntCC::Ne, x, b))
            if unsigned(a) < unsigned(b)
                && ((unsigned(b) - unsigned(a)) & (unsigned(b) - unsigned(a) - 1)) == 0
            => Icmp(IntCC::Ne, IAnd(ISub(x, a), IXor(ISub(b, a), -1)), 0);
    }

    where T: ScalarInteger, (T, T) -> T {
        case (x, -1) => x;
        case (x, 0) => 0;
        case (x, x) => x;

        // A contiguous interval [low, high] becomes an unsigned distance check.
        case (ExtendU(Icmp(IntCC::GeS, x, low)),
              ExtendU(Icmp(IntCC::LeS, x, high)))
            if signed(low) <= signed(high)
            => ExtendU(Icmp(IntCC::LeU,
                 ISub(x, low), ISub(high, low)));
        case (ExtendU(Icmp(IntCC::GeU, x, low)),
              ExtendU(Icmp(IntCC::LeU, x, high)))
            if unsigned(low) <= unsigned(high)
            => ExtendU(Icmp(IntCC::LeU,
                 ISub(x, low), ISub(high, low)));
    }

    case (Icmp(IntCC::GeS, x, low), Icmp(IntCC::LeS, x, high))
        if signed(low) <= signed(high)
        => Icmp(IntCC::LeU, ISub(x, low), ISub(high, low));
    case (Icmp(IntCC::GeU, x, low), Icmp(IntCC::LeU, x, high))
        if unsigned(low) <= unsigned(high)
        => Icmp(IntCC::LeU, ISub(x, low), ISub(high, low));

    where T: ScalarInteger, (T, T) -> T {
        case (x, IXor(x, -1)) => 0;
        case (x, IXor(x, y)) => IAnd(x, IXor(y, -1));
        case (IAnd(x, y), z) if is_const(y) => IAnd(x, IAnd(y, z));
        case (x, IOr(x, y)) => x;
        case (IOr(x, y), IOr(x, z)) => IOr(x, IAnd(y, z));
        case (IShl(x, amount), IShl(y, amount)) => IShl(IAnd(x, y), amount);
        case (IShrU(x, amount), IShrU(y, amount)) => IShrU(IAnd(x, y), amount);
        case (IShrS(x, amount), IShrS(y, amount)) => IShrS(IAnd(x, y), amount);
        case (IRotl(x, amount), IRotl(y, amount)) => IRotl(IAnd(x, y), amount);
        case (IRotr(x, amount), IRotr(y, amount)) => IRotr(IAnd(x, y), amount);
    }
}

rule IOr {
    where (Type::BOOL, Type::BOOL) -> Type::BOOL {
        case (x, false) => x;
        case (x, true) => true;
        case (x, x) => x;
    }

    where T: ScalarInteger {
        // a < b and (b - a) & (b - a - 1) == 0 establish a power-of-two gap.
        // If a has that bit clear, a and b already differ in exactly one bit.
        case (Icmp(IntCC::Eq, x: T, a), Icmp(IntCC::Eq, x, b))
            if unsigned(a) < unsigned(b)
                && ((unsigned(b) - unsigned(a)) & (unsigned(b) - unsigned(a) - 1)) == 0
                && (unsigned(a) & (unsigned(b) - unsigned(a))) == 0
            => Icmp(IntCC::Eq, IAnd(x, IXor(ISub(b, a), -1)), a);
        // (x - a) is either zero or b - a; subtraction uses the operand width.
        case (Icmp(IntCC::Eq, x: T, a), Icmp(IntCC::Eq, x, b))
            if unsigned(a) < unsigned(b)
                && ((unsigned(b) - unsigned(a)) & (unsigned(b) - unsigned(a) - 1)) == 0
            => Icmp(IntCC::Eq, IAnd(ISub(x, a), IXor(ISub(b, a), -1)), 0);
    }

    where T: ScalarInteger, (T, T) -> T {
        case (x, 0) => x;
        case (x, -1) => -1;
        case (x, x) => x;
        case (x, IXor(x, -1)) => -1;
        case (x, IXor(x, y)) => IOr(x, y);
        case (IOr(x, y), z) if is_const(y) => IOr(x, IOr(y, z));
        case (x, IAnd(x, y)) => x;
        case (IAnd(x, y), IAnd(x, z)) => IAnd(x, IOr(y, z));
        case (IShl(x, amount), IShl(y, amount)) => IShl(IOr(x, y), amount);
        case (IShrU(x, amount), IShrU(y, amount)) => IShrU(IOr(x, y), amount);
        case (IShrS(x, amount), IShrS(y, amount)) => IShrS(IOr(x, y), amount);
        case (IRotl(x, amount), IRotl(y, amount)) => IRotl(IOr(x, y), amount);
        case (IRotr(x, amount), IRotr(y, amount)) => IRotr(IOr(x, y), amount);
    }
}

rule IXor {
    case (x: Type::BOOL, false) => x;
    case (Icmp(IntCC::Eq, x, y), true) => Icmp(IntCC::Ne, x, y);
    case (Icmp(IntCC::Ne, x, y), true) => Icmp(IntCC::Eq, x, y);

    where T: ScalarInteger, (T, T) -> T {
        case (x, 0) => x;
        case (x, x) => 0;
        case (IXor(x, y), y) => x;
        case (IXor(x, y), z) if is_const(y) => IXor(x, IXor(y, z));
        case (x, IXor(x, -1)) => -1;
        case (IAnd(x, y), IAnd(x, z)) => IAnd(x, IXor(y, z));
        case (IShl(x, amount), IShl(y, amount)) => IShl(IXor(x, y), amount);
        case (IShrU(x, amount), IShrU(y, amount)) => IShrU(IXor(x, y), amount);
        case (IShrS(x, amount), IShrS(y, amount)) => IShrS(IXor(x, y), amount);
        case (IRotl(x, amount), IRotl(y, amount)) => IRotl(IXor(x, y), amount);
        case (IRotr(x, amount), IRotr(y, amount)) => IRotr(IXor(x, y), amount);
    }
}

rule IShl {
    where T: ScalarInteger, (T, T) -> T {
        case (x, 0) => x;
        case (0, amount) => 0;
    }
}

rule IShrU {
    where T: ScalarInteger, (T, T) -> T {
        case (x, 0) => x;
        case (0, amount) => 0;
    }
}

rule IShrS {
    where T: ScalarInteger, (T, T) -> T {
        case (x, 0) => x;
        case (0, amount) => 0;
    }
}

rule IRotl {
    where T: ScalarInteger, (T, T) -> T {
        case (IRotr(x, amount), amount) => x;
        case (x, 0) => x;
        case (0, amount) => 0;
    }
}

rule IRotr {
    where T: ScalarInteger, (T, T) -> T {
        case (IRotl(x, amount), amount) => x;
        case (x, 0) => x;
        case (0, amount) => 0;
    }
}
