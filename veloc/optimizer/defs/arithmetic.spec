import "../../defs/type_sets.spec";

rule IAdd {
    where T: ScalarInteger, (T, T) -> T {
        case (x, 0) => x;
        case (ISub(x, y), y) => x;
        case (x, IXor(x, -1)) => -1;
        case (IAdd(x, y), z) if is_const(y) => IAdd(x, IAdd(y, z));
        case (IMul(x, y), IMul(x, z)) => IMul(x, IAdd(y, z));
    }
}

rule ISub {
    where T: ScalarInteger, (T, T) -> T {
        case (x, c) if is_const(c) => IAdd(x, INeg(c));
        case (x, x) => 0;
        case (x, 0) => x;
        case (IAdd(x, y), x) => y;
        case (x, ISub(x, y)) => y;
        case (IMul(x, y), IMul(x, z)) => IMul(x, ISub(y, z));
    }
}

rule IMul {
    where T: ScalarInteger, (T, T) -> T {
        case (x, 1) => x;
        case (x, 0) => 0;
        case (IMul(x, y), z) if is_const(y) => IMul(x, IMul(y, z));
    }
}
