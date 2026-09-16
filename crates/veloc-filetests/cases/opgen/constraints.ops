// ----- constraints/constant-folding-preserves-precedence-and-short-circuiting-1
// run: opgen-error
// check: constraint is always false: ((1 + (2 * 3)) == 9)
struct Custom { bits: u64, yes: bool }
op Example(number: u64, flag: bool) -> Value<ScalarInteger> {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Custom { bits: number, yes: flag };
    verify {
        1 + 2 * 3 == 9;
    }
}

// ----- constraints/ill-typed-or-unbounded-expressions-are-definition-errors-1
// run: opgen-error
// check: binary expression operand types differ
struct Custom { bits: u64, yes: bool }
op Example(number: u64, flag: bool) -> Value<ScalarInteger> {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Custom { bits: number, yes: flag };
    verify {
        number && flag;
    }
}

// ----- constraints/ill-typed-or-unbounded-expressions-are-definition-errors-2
// run: opgen-error
// check: binary expression operand types differ
struct Custom { bits: u64, yes: bool }
op Example(number: u64, flag: bool) -> Value<ScalarInteger> {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Custom { bits: number, yes: flag };
    verify {
        number == flag;
    }
}

// ----- constraints/ill-typed-or-unbounded-expressions-are-definition-errors-3
// run: opgen-error
// check: unknown expression name or operation
struct Custom { bits: u64, yes: bool }
op Example(number: u64, flag: bool) -> Value<ScalarInteger> {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Custom { bits: number, yes: flag };
    verify {
        unknown != 0;
    }
}

// ----- constraints/ill-typed-or-unbounded-expressions-are-definition-errors-4
// run: opgen-error
// check: field access requires a struct
struct Custom { bits: u64, yes: bool }
op Example(number: u64, flag: bool) -> Value<ScalarInteger> {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Custom { bits: number, yes: flag };
    verify {
        number.field != 0;
    }
}

// ----- constraints/ill-typed-or-unbounded-expressions-are-definition-errors-5
// run: opgen-error
// check: all expects finite sequences or optional values
struct Custom { bits: u64, yes: bool }
op Example(number: u64, flag: bool) -> Value<ScalarInteger> {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Custom { bits: number, yes: flag };
    verify {
        all(number, |x| true);
    }
}

// ----- constraints/ill-typed-or-unbounded-expressions-are-definition-errors-6
// run: opgen-error
// check: all expects sequences followed by a predicate
struct Custom { bits: u64, yes: bool }
op Example(number: u64, flag: bool) -> Value<ScalarInteger> {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Custom { bits: number, yes: flag };
    verify {
        all(number, flag);
    }
}

// ----- constraints/ill-typed-or-unbounded-expressions-are-definition-errors-7
// run: opgen-error
// check: len expects a sequence
struct Custom { bits: u64, yes: bool }
op Example(number: u64, flag: bool) -> Value<ScalarInteger> {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Custom { bits: number, yes: flag };
    verify {
        len(flag) == 0;
    }
}

// ----- constraints/ill-typed-or-unbounded-expressions-are-definition-errors-8
// run: opgen-error
// check: unknown projection function `u64::ty`
struct Custom { bits: u64, yes: bool }
op Example(number: u64, flag: bool) -> (result: Value<ScalarInteger>) {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Custom { bits: number, yes: flag };
    verify {
        number.ty() == result;
    }
}

// ----- constraints/ill-typed-or-unbounded-expressions-are-definition-errors-9
// run: opgen-error
// check: unknown expression name or operation: missing
struct Custom { bits: u64, yes: bool }
op Example(number: u64, flag: bool) -> (result: Value<ScalarInteger>) {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Custom { bits: number, yes: flag };
    verify {
        missing == result;
    }
}

// ----- constraints/ill-typed-or-unbounded-expressions-are-definition-errors-10
// run: opgen-error
// check: unknown projection function `IntCC::Unknown`
struct Custom { bits: u64, yes: bool }
op Example(number: u64, flag: bool) -> Value<ScalarInteger> {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Custom { bits: number, yes: flag };
    verify {
        IntCC::Unknown == IntCC::Eq;
    }
}

// ----- constraints/ill-typed-or-unbounded-expressions-are-definition-errors-11
// run: opgen-error
// check: projection type mismatch: expected Named("IntCC"), found Named("FloatCC")
struct Custom { bits: u64, yes: bool }
op Example(number: u64, flag: bool) -> Value<ScalarInteger> {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Custom { bits: number, yes: flag };
    verify {
        IntCC::Eq == FloatCC::Eq;
    }
}

// ----- constraints/ill-typed-or-unbounded-expressions-are-definition-errors-12
// run: opgen-error
// check: expected value of type bool
struct Custom { bits: u64, yes: bool }
op Example(number: u64, flag: bool) -> Value<ScalarInteger> {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Custom { bits: number, yes: flag };
    verify {
        recurse(number);
    }
}

// ----- constraints/ill-typed-or-unbounded-expressions-are-definition-errors-13
// run: opgen-error
// check: unsupported expression operator or operand type
struct Custom { bits: u64, yes: bool }
op Example(number: u64, flag: bool) -> Value<ScalarInteger> {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Custom { bits: number, yes: flag };
    verify {
        flag + flag;
    }
}

// ----- constraints/ill-typed-or-unbounded-expressions-are-definition-errors-14
// run: opgen-error
// check: require expects a predicate and a diagnostic string
struct Custom { bits: u64, yes: bool }
op Example(number: u64, flag: bool) -> Value<ScalarInteger> {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Custom { bits: number, yes: flag };
    verify {
        require(flag);
    }
}

// ----- constraints/ill-typed-or-unbounded-expressions-are-definition-errors-15
// run: opgen-error
// check: expected value of type bool
struct Custom { bits: u64, yes: bool }
op Example(number: u64, flag: bool) -> Value<ScalarInteger> {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Custom { bits: number, yes: flag };
    verify {
        VectorConstant;
    }
}

// ----- constraints/ill-typed-or-unbounded-expressions-are-definition-errors-16
// run: opgen-error
// check: expected value of type bool
struct Custom { bits: u64, yes: bool }
op Example(number: u64, flag: bool) -> Value<ScalarInteger> {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Custom { bits: number, yes: flag };
    verify {
        nonzero(number);
    }
}

// ----- constraints/ill-typed-or-unbounded-expressions-are-definition-errors-17
// run: opgen-error
// check: expression arithmetic overflow
struct Custom { bits: u64, yes: bool }
op Example(number: u64, flag: bool) -> Value<ScalarInteger> {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Custom { bits: number, yes: flag };
    verify {
        170141183460469231731687303715884105727 + 1 == 0;
    }
}

// ----- constraints/ill-typed-or-unbounded-expressions-are-definition-errors-18
// run: opgen-error
// check: unknown expression name or operation
struct Custom { bits: u64, yes: bool }
op Example(number: u64, flag: bool) -> Value<ScalarInteger> {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Custom { bits: number, yes: flag };
    verify {
        true || missing > 0;
    }
}

// ----- constraints/pool-properties-and-lexical-binders-are-generic-1
// run: opgen-error
// check: projection type mismatch: expected Named("bool"), found Named("i128")
struct Custom { pool_id: ConstantPoolId }
op Example(data: Bytes) -> (result: Value<Vector>) {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Custom { pool_id: pool(data) };
    text = "{data:bytes}";
    verify {
        len(data) == result.lanes()?; all(data, |i| i);
    }
}

// ----- constraints/pool-properties-and-lexical-binders-are-generic-2
// run: opgen-error
// check: unknown expression name or operation: result
struct Custom { pool_id: ConstantPoolId }
op Example(data: Bytes) -> () {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Custom { pool_id: pool(data) };
    text = "{data:bytes}";
    verify {
        len(data) == result.lanes()?; all(data, |i| i < 2 * result.lanes()?);
    }
}

// ----- constraints/struct-access-uses-logical-names-not-rule-or-storage-names-1
// run: opgen-error
// check: unknown field PtrIndexImm.unknown
struct PtrIndexImm {
    offset: i32,
    scale: u32,
}

struct PtrIndex {
    ptr: Value,
    index: Value,
    imm_id: PtrIndexImm,
}

op PtrIndex(ptr: Value<Type::PTR>, index: Value<ScalarInteger>, imm: PtrIndexImm) -> Value<Type::PTR> {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "ptr-index";
    storage = PtrIndex { ptr: ptr, index: index, imm_id: imm };
    text = "{ptr}, {index}, scale={imm.scale}, offset={imm.offset}";

    verify {
        require(imm.unknown != 0, "ptr-index scale must be non-zero");
    }
}

// ----- constraints/type-queries-use-static-signatures-and-operand-positions-1
// run: opgen-const-error
// check: no admissible type signature
struct Unary { arg: Value }
op Example<T: Integer>(input: Value<T>) -> Value<T> {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Unary { arg: input };
    verify {
        input.ty().is_ptr(); input.ty().lanes()? > 0;
    }
}

// ----- type-constraints/lir-type-constraints-use-the-shared-expression-language
// run: opgen
// check: TypeError::Constraint("double width")
// check: checked_mul
type Reg = rust("crate::Reg");
enum InstField { variants = [Imm(i64)]; }
storage Operands { opcode = GenericOpcode; view = InstView; reader = InstRead; writer = InstBuild; register = Reg; attributes = InstField; }
struct Unary { dst: Reg, src: Reg }
fn double_bits(from: Type, to: Type) -> bool {
    value = to.element_bits()? == from.element_bits()? * 2
; }
op Widen<T: Integer, U: Integer>(src: Value<T>) -> (dst: Value<U>) {
    meta = OpInfo {};
    storage = Unary { dst, src };
    verify {
        require(U.same_shape(T), "input and result must have the same shape");
        require(double_bits(T, dst), "double width");
    }
    semantics = bv.sext(src, result(0))
; }

// ----- type-constraints/old-where-mechanism-is-removed
// run: opgen-error
// check: unknown field
struct Unary { arg: Value }
op Old(arg: Value<Type::I8>) -> (result: Value<Type::I16>) {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "old"; storage = Unary { arg };
    where = [wider(arg, result)];
}

// ----- type-constraints/impossible-type-predicates-reject-semantic-instantiation
// run: opgen-const-error
// check: no admissible semantic signature
struct Unary { arg: Value }
op Impossible<T: Integer>(arg: Value<T>) -> Value<T> {
    meta = OpInfo {};
    mnemonic = "impossible"; storage = Unary { arg };
    verify { require(arg.ty().element_bits()? > 128, "too wide"); }
    semantics = bv.neg(arg)
; }

// ----- type-constraints/scalar-type-filter-runs-before-semantic-sort-check
// run: opgen
// check: Opcode::IntegerOnly
struct Unary { arg: Value }
op IntegerOnly<T: ScalarInteger | Type::PTR>(arg: Value<T>) -> Value<T> {
    meta = OpInfo {};
    mnemonic = "integer-only"; storage = Unary { arg };
    verify { require(!arg.ty().is_ptr(), "no pointers"); }
    semantics = bv.neg(arg)
; }

// ----- type-constraints/partial-type-queries-preserve-short-circuiting
// run: opgen
// check: short circuit
struct Unary { arg: Value }
op Guarded<T: Integer>(arg: Value<T>) -> Value<T> {
    meta = OpInfo {};
    mnemonic = "guarded"; storage = Unary { arg };
    verify { require(arg.ty().is_scalar() || arg.ty().shape()? == arg.ty().shape()?, "short circuit"); }
    semantics = bv.neg(arg)
; }

// ----- constraints/all-checks-predicate-arity
// run: opgen-error
// check: all requires one predicate parameter per sequence
fn invalid(a: sequence(u32), b: sequence(u32)) -> bool {
    value = all(a, b, |x| x > 0);
}

// ----- constraints/all-rejects-duplicate-binders
// run: opgen-error
// check: duplicate predicate parameter
fn invalid(a: sequence(u32), b: sequence(u32)) -> bool {
    value = all(a, b, |x, x| x > 0);
}

// ----- constraints/all-is-not-specific-to-ssa
// run: opgen
// check: while
// not: value_type
// not: collect
struct Summary { equal: bool }
struct Data { left: u32, right: u32 }
fn same(a: optional(u32), b: optional(u32)) -> bool {
    value = all(a, b, |x, y| x == y);
}
op Example(left: u32, right: u32) -> () {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Data { left, right };
    query summary -> Summary { equal: same(some(left), some(right)) }
}
