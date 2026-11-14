import "../../defs/type_sets.spec";

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
