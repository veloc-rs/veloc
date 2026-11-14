import "../../defs/type_sets.spec";
import "../../mir/defs/formats.spec";

rule Select {
    where T: ScalarInteger, (Type::BOOL, T, T) -> T {
        case (condition, 1, 0) => ExtendU(condition);
        case (condition, Wrap(x), Wrap(y)) if type_of(x) == type_of(y)
            => Wrap(Select(condition, x, y));
        case (condition, Wrap(x), c) if is_const(c)
            => Wrap(Select(condition, x, ExtendU(c)));
        case (condition, c, Wrap(x)) if is_const(c)
            => Wrap(Select(condition, ExtendU(c), x));
    }

    case (condition, true, false) -> Type::BOOL => condition;

    // These identities also apply to pointers, floats and vectors. Select's
    // signature supplies the shared branch/result type without an annotation.
    case (true, x, y) => x;
    case (false, x, y) => y;
    case (condition, x, x) => x;
}

// Captured successors include their complete argument lists. Replacements may
// select one original edge or permute both; they cannot construct new edges.
rule Br {
    case (Select(c, yes, no), a, b)
        if constant_bits(yes) == 1 && constant_bits(no) == 0
        => Br(c, a, b);
    case (Select(c, yes, no), a, b)
        if constant_bits(yes) == 0 && constant_bits(no) == 1
        => Br(c, b, a);
    case (_, a, a) => Jump(a);
}
