import "../../defs/type_sets.spec";

rule Wrap {
    where T: Integer, S: Any, (_) -> T {
        // The intermediate extension type comes from the matched node. Only
        // the input and final types matter; the existing casts verify shapes.
        case (ExtendU(x: T)) => x;
        case (ExtendS(x: T)) => x;
        case (ExtendU(x: S)) if bits(S) < bits(T) => ExtendU(x);
        case (ExtendS(x: S)) if bits(S) < bits(T) => ExtendS(x);
        case (ExtendU(x: S)) if bits(S) > bits(T) => Wrap(x);
        case (ExtendS(x: S)) if bits(S) > bits(T) => Wrap(x);
    }
}

rule IntToPtr {
    where I: ScalarInteger, (I) -> Type::PTR {
        // Missing target layout leaves the round-trip width check unknown.
        case (PtrToInt(p)) if bits(I) >= pointer_bits() => p;
    }
}

rule PtrToInt {
    where I: ScalarInteger, (Type::PTR) -> I {
        case (IntToPtr(x: I)) if bits(I) <= pointer_bits() => x;
    }
}
