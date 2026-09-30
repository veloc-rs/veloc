import "../common.spec";

expand Binary(RvMul32, GprValue, GPR, 59, 0, 1, "mulw", "M");

expand Binary(RvMul64, GprValue, GPR, 51, 0, 1, "mul", "M");

expand Binary(RvSdiv32, GprValue, GPR, 59, 4, 1, "divw", "M");

expand Binary(RvSdiv64, GprValue, GPR, 51, 4, 1, "div", "M");

expand Binary(RvUdiv32, GprValue, GPR, 59, 5, 1, "divuw", "M");

expand Binary(RvUdiv64, GprValue, GPR, 51, 5, 1, "divu", "M");

expand Binary(RvSrem32, GprValue, GPR, 59, 6, 1, "remw", "M");

expand Binary(RvSrem64, GprValue, GPR, 51, 6, 1, "rem", "M");

expand Binary(RvUrem32, GprValue, GPR, 59, 7, 1, "remuw", "M");

expand Binary(RvUrem64, GprValue, GPR, 51, 7, 1, "remu", "M");

select(n: lir::Mul) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(RvMul32(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(RvMul64(n.lhs, n.rhs)));
        }
    }
}

select(n: lir::Sdiv) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(RvSdiv32(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(RvSdiv64(n.lhs, n.rhs)));
        }
    }
}

select(n: lir::Udiv) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(RvUdiv32(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(RvUdiv64(n.lhs, n.rhs)));
        }
    }
}

select(n: lir::Srem) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(RvSrem32(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(RvSrem64(n.lhs, n.rhs)));
        }
    }
}

select(n: lir::Urem) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(RvUrem32(n.lhs, n.rhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(RvUrem64(n.lhs, n.rhs)));
        }
    }
}
