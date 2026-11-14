import "../../defs/type_sets.spec";

rule(root: mir::IAnd) {
    case root<Type::BOOL>(x, true) => x;
    case root<Type::BOOL>(x, false) => false;
    case root<Type::BOOL>(x, x) => x;

    where T: ScalarInteger {
        case root<T>(x, -1) => x;
        case root<T>(x, 0) => 0;
        case root<T>(x, x) => x;

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
    case root<Type::BOOL>(x, false) => x;
    case root<Type::BOOL>(x, true) => true;
    case root<Type::BOOL>(x, x) => x;

    where T: ScalarInteger {
        case root<T>(x, 0) => x;
        case root<T>(x, -1) => -1;
        case root<T>(x, x) => x;
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
    case root<Type::BOOL>(x, false) => x;

    where T: ScalarInteger {
        case root<T>(x, 0) => x;
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
