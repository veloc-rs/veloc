import "../../defs/type_sets.spec";

// Conversion and boolean equalities explored by the e-graph.
// Canonical cases also permit equal-cost changes toward the chosen normal form.
// Type arguments name result types; nested nodes bind independent type facts.
rule<T: ScalarInteger>(root: mir::Select<T>) {
    case (condition, 1, 0) => mir::ExtendU<T>(condition);
    case (condition, mir::Wrap<T>(x), mir::Wrap<T>(y)) if type_of(x) == type_of(y)
        => mir::Wrap<T>(mir::Select<type_of(x)>(condition, x, y));
    case (condition, mir::Wrap<T>(x), c) if is_const(c)
        => mir::Wrap<T>(mir::Select<type_of(x)>(condition, x, mir::ExtendU<type_of(x)>(c)));
    case (condition, c, mir::Wrap<T>(x)) if is_const(c)
        => mir::Wrap<T>(mir::Select<type_of(x)>(condition, mir::ExtendU<type_of(x)>(c), x));
}
rule(root: mir::Select<Type::BOOL>) {
    case (condition, true, false) => condition;
}

// Widening preserves zero. Peel casts one at a time; booleans terminate the
// chain as an inverted selection because IEqz accepts integer operands.
rule<W: ScalarInteger>(root: mir::IEqz<Type::BOOL>) {
    case (mir::ExtendU<W>(x)) if type_of(x) == Type::BOOL => mir::Select<Type::BOOL>(x, false, true);
    case (mir::ExtendU<W>(x)) => mir::IEqz<Type::BOOL>(x);
    case (mir::ExtendS<W>(x)) => mir::IEqz<Type::BOOL>(x);
}

// The layout determines whether the round trip loses any bits. Missing layout
// means the predicate cannot be established, so neither host applies the rule.
rule<I: ScalarInteger>(root: mir::IntToPtr<Type::PTR>) {
    case (mir::PtrToInt<I>(p)) if bits(I) >= pointer_bits() => p;
}
rule<I: ScalarInteger>(root: mir::PtrToInt<I>) {
    case (mir::IntToPtr<Type::PTR>(x)) if type_of(x) == I && bits(I) <= pointer_bits() => x;
}

// The matched casts already establish equal vector shapes and a wider middle
// type. Construction validates the resulting cast's complete type contract.
rule<T: Integer, W: Integer>(root: mir::Wrap<T>) {
    case (mir::ExtendU<W>(x)) if type_of(x) == T => x;
    case (mir::ExtendS<W>(x)) if type_of(x) == T => x;
    case (mir::ExtendU<W>(x)) if bits(type_of(x)) < bits(T) => mir::ExtendU<T>(x);
    case (mir::ExtendS<W>(x)) if bits(type_of(x)) < bits(T) => mir::ExtendS<T>(x);
    case (mir::ExtendU<W>(x)) if bits(type_of(x)) > bits(T) => mir::Wrap<T>(x);
    case (mir::ExtendS<W>(x)) if bits(type_of(x)) > bits(T) => mir::Wrap<T>(x);
}
