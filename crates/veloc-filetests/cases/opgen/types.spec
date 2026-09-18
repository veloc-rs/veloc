// ----- type-catalog/rust-types-are-available-in-signatures-and-constraints
// run: opgen
// check: pub enum Opcode
struct Unary { arg: Value }
op Scalar(arg: Value<Type::I32>) -> Value<Type::I32> {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "scalar";
    storage = Unary { arg };
    verify { require(arg.ty() == Type::I32, "expected i32"); }
}
op Vector(arg: Value<Type::I32X4>) -> Value<Type::I32X4> {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "vector";
    storage = Unary { arg };
    verify { require(arg.ty().lanes()? == 4, "expected four lanes"); }
}

// ----- type-catalog/arbitrary-vector-aliases-still-compose
// run: opgen
// check: pub enum Opcode
type WIDE = vector(Type::I16, 32);
type SCALABLE = vector(Type::I32, scalable(8));
typeset Shapes = WIDE | SCALABLE | Type::I32X4;
struct Unary { arg: Value }
op Copy<T: Shapes>(arg: Value<T>) -> Value<T> {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "copy";
    storage = Unary { arg };
}

// ----- type-catalog/unknown-associated-type
// run: opgen-error
// check: unknown type
type UNSUPPORTED = Type::I128;

// ----- type-catalog/imported-scalar-declarations-can-be-aliased
// run: opgen
// check: pub enum Opcode
type WORD = I32;

// ----- type-catalog/duplicate-scalar-domain-requires-an-alias
// run: opgen-error
// check: duplicate scalar domain
type WORD = int(32);

// ----- type-sets/exact-set-members-drive-codegen-and-bitvector-semantics-1
// run: opgen-error
// check: floating-point execution semantics are not modeled
typeset Wide = Type::I32 | Type::F64;
struct Pair { args: values(2) }
op Add<T: Wide>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    meta = OpInfo {};
    mnemonic = "add"; storage = Pair { args: [lhs, rhs] }; semantics = bv.add(lhs, rhs)
; }

// ----- type-sets/exact-shapes-detect-impossible-relations-at-definition-time-1
// run: opgen-const-error
// check: no admissible type signature
typeset V4 = Type::I32X4;
typeset V2 = Type::I64X2;
struct Unary { arg: Value }
op Convert<T: V4, U: V2>(arg: Value<T>) -> Value<U> {
    verify { require(U.same_shape(T), "input and result must have the same shape"); }
    meta = OpInfo { memory: MemoryEffect::NONE };
mnemonic = "convert"; storage = Unary { arg: arg }; }

// ----- type-sets/exact-shapes-detect-impossible-relations-at-definition-time-2
// run: opgen-const-error
// check: no admissible type signature
typeset V4 = Type::I32X4;
typeset V2 = Type::I64;
struct Unary { arg: Value }
op Convert<T: V4, U: V2>(arg: Value<T>) -> Value<U> {
    verify { require(U.same_shape(T), "input and result must have the same shape"); }
    meta = OpInfo { memory: MemoryEffect::NONE };
mnemonic = "convert"; storage = Unary { arg: arg }; }

// ----- type-sets/vector-constants-reject-unrepresentable-or-invalid-types-1
// run: opgen-error
// check: invalid vector lanes or element type
type V = vector(Type::PTR, 4);

// ----- type-sets/vector-constants-reject-unrepresentable-or-invalid-types-2
// run: opgen-error
// check: vector element must be a scalar type
type V = vector(Type::I32X4, 4);

// ----- type-sets/vector-constants-reject-unrepresentable-or-invalid-types-3
// run: opgen-error
// check: unknown type `Missing`
type V = vector(Missing, 4);

// ----- type-sets/vector-constants-reject-unrepresentable-or-invalid-types-4
// run: opgen-error
// check: invalid vector lanes or element type
type V = vector(Type::I32, 0);

// ----- type-sets/vector-constants-reject-unrepresentable-or-invalid-types-5
// run: opgen-error
// check: invalid vector lanes or element type
type V = vector(Type::I32, 1);

// ----- type-sets/vector-constants-reject-unrepresentable-or-invalid-types-6
// run: opgen-error
// check: invalid vector lanes or element type
type V = vector(Type::I32, 3);

// ----- type-sets/vector-constants-reject-unrepresentable-or-invalid-types-7
// run: opgen-error
// check: vector lanes exceed the supported type domain
type V = vector(Type::I32, 65536);

// ----- type-sets/vector-constants-reject-unrepresentable-or-invalid-types-8
// run: opgen-error
// check: vector shape must be a lane count or scalable(lanes)
type V = vector(Type::I32, maybe);

// ----- type-sets/vector-constants-reject-unrepresentable-or-invalid-types-9
// run: opgen-error
// check: duplicate type `V`
type V = Type::I32;
type V = vector(Type::I32, 4);

// ----- type-sets/vector-constants-reject-unrepresentable-or-invalid-types-10
// run: opgen-error
// check: type name must be uppercase and not INVALID
type INVALID = vector(Type::I32, 4);

// ----- type-sets/vector-constants-reject-unrepresentable-or-invalid-types-11
// run: opgen-error
// check: duplicate type `V`
type V = Type::I32X4;
type V = vector(Type::I32, 4);

// ----- type-sets/vector-families-require-scalar-sets-and-preserve-definition-checks-1
// run: opgen-error
// check: vectors() requires a set of non-pointer scalar types
typeset Bad = vectors(Type::PTR);

// ----- type-sets/vector-families-require-scalar-sets-and-preserve-definition-checks-2
// run: opgen-error
// check: vectors() requires a set of non-pointer scalar types
typeset Bad = vectors(Type::I32X4);

// ----- type-sets/vector-families-require-scalar-sets-and-preserve-definition-checks-3
// run: opgen-error
// check: vectors() requires a set of non-pointer scalar types
typeset Bad = vectors(Any);

// ----- type-sets/vector-families-require-scalar-sets-and-preserve-definition-checks-4
// run: opgen-error
// check: vectors() requires a set of non-pointer scalar types
typeset Bad = vectors(vectors(Type::I32));

// ----- type-sets/vector-families-require-scalar-sets-and-preserve-definition-checks-5
// run: opgen-error
// check: unknown type or typeset `Missing`
typeset Bad = vectors(Missing);

// ----- type-sets/vector-families-require-scalar-sets-and-preserve-definition-checks-6
// run: opgen-error
// check: expected a type, typeset or vectors(set)
typeset Bad = vectors();

// ----- type-sets/vector-families-require-scalar-sets-and-preserve-definition-checks-7
// run: opgen-error
// check: expected a type, typeset or vectors(set)
typeset Bad = vectors(Type::I32, Type::I64);

// ----- type-sets/vector-families-require-scalar-sets-and-preserve-definition-checks-8
// run: opgen
// check: pub enum Opcode
typeset Repeated = vectors(Type::I32) | vectors(Type::I32);

// ----- type-sets/vector-families-require-scalar-sets-and-preserve-definition-checks-9
// run: opgen-error
// check: cyclic type set `A`
typeset A = vectors(B); typeset B = A;

// ----- type-sets/vector-families-require-scalar-sets-and-preserve-definition-checks-10
// run: opgen-error
// check: typeset name shadows an exact type
type V = Type::I32X4;
typeset V = Type::I32;

// ----- type-sets/vector-families-require-scalar-sets-and-preserve-definition-checks-11
// run: opgen-error
// check: unknown type or typeset `vector_integer`
typeset Bad = vector_integer;

// ----- type-sets/typesets-use-shared-set-expressions
// run: opgen
// check: pub enum Opcode
// check: Add
typeset Wide = (Later | Type::I32) & Scalar;
typeset Later = Type::I64 | Type::I32;
typeset Lanes = vectors(Wide) & Vector;
struct Pair { args: values(2) }
op Add<T: Lanes>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    meta = OpInfo {};
    mnemonic = "add";
    storage = Pair { args: [lhs, rhs] };
    semantics = bv.add(lhs, rhs);
}

// ----- type-sets/old-class-declarations-are-rejected
// run: opgen-error
// check: unknown definition kind `class`
class Legacy { members = [Type::I32]; }

// ----- type-expressions/direct-patterns-and-shape-constraints-accept-expressions-1
// run: opgen-const-error
// check: no admissible type signature
struct Unary { arg: Value }
op Copy(arg: Value<Type::I32 | Type::I64>) -> Value<Type::I32 | Type::I64> {
    meta = OpInfo { memory: MemoryEffect::NONE };
mnemonic = "copy"; storage = Unary { arg: arg }; }
op Lane<T: vectors(Type::I32 | Type::I64)>(arg: Value<T>) -> Value<element(T)> {
    meta = OpInfo { memory: MemoryEffect::NONE };
mnemonic = "lane"; storage = Unary { arg: arg }; }
op Convert<T: Type::I32 | Type::I64, U: Float & Vector>(arg: Value<T>) -> Value<U> {
    verify { require(U.same_shape(T), "input and result must have the same shape"); }
    meta = OpInfo { memory: MemoryEffect::NONE };
mnemonic = "convert"; storage = Unary { arg: arg }; }

// ----- type-expressions/direct-patterns-and-shape-constraints-accept-expressions-2
// run: opgen-error
// check: unknown type or typeset `T`
struct Unary { arg: Value }
op Copy(arg: Value<Type::I32 | Type::I64>) -> Value<Type::I32 | Type::I64> {
    meta = OpInfo { memory: MemoryEffect::NONE };
mnemonic = "copy"; storage = Unary { arg: arg }; }
op Lane<T: vectors(T)>(arg: Value<T>) -> Value<element(T)> {
    meta = OpInfo { memory: MemoryEffect::NONE };
mnemonic = "lane"; storage = Unary { arg: arg }; }
op Convert<T: Type::I32 | Type::I64, U: Float & Scalar>(arg: Value<T>) -> Value<U> {
    verify { require(U.same_shape(T), "input and result must have the same shape"); }
    meta = OpInfo { memory: MemoryEffect::NONE };
mnemonic = "convert"; storage = Unary { arg: arg }; }

// ----- type-expressions/expression-nesting-is-bounded-but-flat-unions-are-not-recursive-1
// run: opgen-error
// check: definition nesting exceeds 64 levels
struct Pair { args: values(2) }
op Add<T: ((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((Type::I32))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    meta = OpInfo {};
    mnemonic = "add"; storage = Pair { args: [lhs, rhs] }; semantics = bv.add(lhs, rhs)
; }

// ----- type-expressions/invalid-expressions-empty-constraints-and-nested-cycles-are-diagnosed-1
// run: opgen-error
// check: type constraint must not be empty
struct Pair { args: values(2) }
op Add<T: Integer & Float>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    meta = OpInfo {};
    mnemonic = "add"; storage = Pair { args: [lhs, rhs] }; semantics = bv.add(lhs, rhs)
; }

// ----- type-expressions/invalid-expressions-empty-constraints-and-nested-cycles-are-diagnosed-2
// run: opgen-error
// check: type constraint must not be empty
struct Pair { args: values(2) }
op Add<T: vectors(Type::I32) & Type::I32>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    meta = OpInfo {};
    mnemonic = "add"; storage = Pair { args: [lhs, rhs] }; semantics = bv.add(lhs, rhs)
; }

// ----- type-expressions/invalid-expressions-empty-constraints-and-nested-cycles-are-diagnosed-3
// run: opgen-error
// check: unknown type or typeset `Missing`
struct Pair { args: values(2) }
op Add<T: Integer | Missing>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    meta = OpInfo {};
    mnemonic = "add"; storage = Pair { args: [lhs, rhs] }; semantics = bv.add(lhs, rhs)
; }

// ----- type-expressions/invalid-expressions-empty-constraints-and-nested-cycles-are-diagnosed-4
// run: opgen-error
// check: unknown type or typeset `Missing`
struct Pair { args: values(2) }
op Add<T: (Type::I32 & Type::F32) & Missing>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    meta = OpInfo {};
    mnemonic = "add"; storage = Pair { args: [lhs, rhs] }; semantics = bv.add(lhs, rhs)
; }

// ----- type-expressions/invalid-expressions-empty-constraints-and-nested-cycles-are-diagnosed-5
// run: opgen-error
// check: vectors() requires a set of non-pointer scalar types
struct Pair { args: values(2) }
op Add<T: vectors(Type::I32 | Type::PTR)>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    meta = OpInfo {};
    mnemonic = "add"; storage = Pair { args: [lhs, rhs] }; semantics = bv.add(lhs, rhs)
; }

// ----- type-expressions/invalid-expressions-empty-constraints-and-nested-cycles-are-diagnosed-6
// run: opgen-error
// check: expected a name
struct Pair { args: values(2) }
op Add<T: Integer |>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    meta = OpInfo {};
    mnemonic = "add"; storage = Pair { args: [lhs, rhs] }; semantics = bv.add(lhs, rhs)
; }

// ----- type-expressions/invalid-expressions-empty-constraints-and-nested-cycles-are-diagnosed-7
// run: opgen-error
// check: expected `,`
struct Pair { args: values(2) }
op Add<T: Integer || Type::BOOL>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    meta = OpInfo {};
    mnemonic = "add"; storage = Pair { args: [lhs, rhs] }; semantics = bv.add(lhs, rhs)
; }

// ----- type-expressions/invalid-expressions-empty-constraints-and-nested-cycles-are-diagnosed-8
// run: opgen-error
// check: expected `,`
struct Pair { args: values(2) }
op Add<T: Integer && Vector>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    meta = OpInfo {};
    mnemonic = "add"; storage = Pair { args: [lhs, rhs] }; semantics = bv.add(lhs, rhs)
; }

// ----- type-expressions/invalid-expressions-empty-constraints-and-nested-cycles-are-diagnosed-9
// run: opgen-error
// check: expected `,`
struct Pair { args: values(2) }
op Add<T: Integer + Type::BOOL>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    meta = OpInfo {};
    mnemonic = "add"; storage = Pair { args: [lhs, rhs] }; semantics = bv.add(lhs, rhs)
; }

// ----- type-expressions/invalid-expressions-empty-constraints-and-nested-cycles-are-diagnosed-10
// run: opgen-error
// check: expected `)`
struct Pair { args: values(2) }
op Add<T: (Integer | Type::BOOL>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    meta = OpInfo {};
    mnemonic = "add"; storage = Pair { args: [lhs, rhs] }; semantics = bv.add(lhs, rhs)
; }

// ----- type-expressions/invalid-expressions-empty-constraints-and-nested-cycles-are-diagnosed-11
// run: opgen-error
// check: expected a name
struct Pair { args: values(2), }
op Add<T: ()>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    meta = OpInfo {};
    mnemonic = "add"; storage = Pair { args: [lhs, rhs] }; semantics = bv.add(lhs, rhs)
; }

// ----- type-expressions/invalid-expressions-empty-constraints-and-nested-cycles-are-diagnosed-12
// run: opgen-error
// check: type set must not be empty
typeset Empty = Integer & Float;

// ----- type-expressions/invalid-expressions-empty-constraints-and-nested-cycles-are-diagnosed-13
// run: opgen-error
// check: cyclic type set `A`
typeset A = Type::I32 | vectors(B); typeset B = A & Scalar;

// ----- type-expressions/semantics-and-text-codecs-use-resolved-inline-constraints-1
// run: opgen-error
// check: invalid semantic types [Bool, Bool] -> [Bool]: expected a bitvector, got bool
struct Pair { args: values(2), }
op Add<T: Type::BOOL | vectors(Type::BOOL)>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    meta = OpInfo {};
    mnemonic = "add"; storage = Pair { args: [lhs, rhs] }; semantics = bv.add(lhs, rhs)
; }

// ----- type-expressions/semantics-and-text-codecs-use-resolved-inline-constraints-2
// run: opgen-error
// check: floating-point execution semantics are not modeled
struct Pair { args: values(2), }
op Add<T: Type::I32 | Type::F64>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    meta = OpInfo {};
    mnemonic = "add"; storage = Pair { args: [lhs, rhs] }; semantics = bv.add(lhs, rhs)
; }

// ----- type-expressions/semantics-and-text-codecs-use-resolved-inline-constraints-3
// run: opgen-error
// check: float text atoms require a scalar float first result
struct Literal { value: Float, }
op Literal<T: Float & Vector>(value: Float) -> Value<T> {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "literal"; storage = Literal { value: value };
}
