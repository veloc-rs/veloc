import "../../defs/type_sets.spec";

rule(root: mir::IAdd) {
    where T: ScalarInteger {
        case root<T>(x, 0) => x;
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
        case root<T>(x, 1) => x;
        case root<T>(x, 0) => 0;
        case root<T>(mir::IMul(x, y), z) if is_const(y) => mir::IMul(x, mir::IMul(y, z));
    }
}
