import "../types/defs/types.spec";

// Scalar and named vector constants come from the shared logical type declarations.
// Members are exact types, other sets, or vectors(set). The latter includes
// every legal fixed and scalable shape over a set of non-pointer scalar types.
typeset Any = Scalar | Type::PTR | Vector;
typeset Scalar = ScalarInteger | ScalarFloat | Type::BOOL;
typeset ScalarInteger = Type::I8 | Type::I16 | Type::I32 | Type::I64;
typeset ScalarFloat = Type::F32 | Type::F64;
typeset Integer = ScalarInteger | vectors(ScalarInteger);
typeset Float = ScalarFloat | vectors(ScalarFloat);
typeset Number = Integer | Float;
typeset Vector = vectors(Scalar);
