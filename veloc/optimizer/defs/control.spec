import "../../defs/type_sets.spec";
import "../../mir/defs/formats.spec";

rule(root: mir::Select) {
    where T: ScalarInteger {
        case root<T>(condition, 1, 0) => mir::ExtendU<T>(condition);
        case root<T>(condition, mir::Wrap<T>(x), mir::Wrap<T>(y)) if type_of(x) == type_of(y)
            => mir::Wrap<T>(mir::Select<type_of(x)>(condition, x, y));
        case root<T>(condition, mir::Wrap<T>(x), c) if is_const(c)
            => mir::Wrap<T>(mir::Select<type_of(x)>(condition, x, mir::ExtendU<type_of(x)>(c)));
        case root<T>(condition, c, mir::Wrap<T>(x)) if is_const(c)
            => mir::Wrap<T>(mir::Select<type_of(x)>(condition, mir::ExtendU<type_of(x)>(c), x));
    }

    case root<Type::BOOL>(condition, true, false) => condition;

    where T: Any {
        // Conditions are boolean; branches and the result share T. These folds do not
        // need constant branches and also apply to pointers, floats and vectors.
        case root<T>(true, x, y) => x;
        case root<T>(false, x, y) => y;
        case root<T>(condition, x, x) => x;
    }
}

// Captured successors include their complete argument lists. Replacements may
// select one original edge or permute both; they cannot construct new edges.
rule(root: mir::Br) {
    case root(mir::Select(c, yes, no), a, b)
        if constant_bits(yes) == 1 && constant_bits(no) == 0
        => mir::Br(c, a, b);
    case root(mir::Select(c, yes, no), a, b)
        if constant_bits(yes) == 0 && constant_bits(no) == 1
        => mir::Br(c, b, a);
    case root(_, a, a) => mir::Jump(a);
}
