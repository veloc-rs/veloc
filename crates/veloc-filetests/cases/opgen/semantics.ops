// ----- typed-semantics/comparison-properties-are-bound-by-name-not-ssa-position-1
// run: opgen-error
// check: unknown integer comparison property `missing`
struct Compare { cc: IntCC, args: values(2) }
op Test<T: ScalarInteger>(lhs: Value<T>, condition: IntCC, rhs: Value<T>) -> Value<Type::BOOL> {
    meta = OpInfo {};
    mnemonic = "test"; storage = Compare { cc: condition, args: [lhs, rhs] };
    semantics = bv.cmp(missing, lhs, rhs)
; }

// ----- typed-semantics/comparison-properties-are-bound-by-name-not-ssa-position-2
// run: opgen-error
// check: unknown semantic value `condition`
struct Compare { cc: IntCC, args: values(2) }
op Test<T: ScalarInteger>(lhs: Value<T>, condition: IntCC, rhs: Value<T>) -> Value<Type::BOOL> {
    meta = OpInfo {};
    mnemonic = "test"; storage = Compare { cc: condition, args: [lhs, rhs] };
    semantics = bv.add(condition, lhs)
; }

// ----- typed-semantics/multiple-results-require-matching-sorts-and-counts-1
// run: opgen-error
// check: invalid semantic types [Bv(Width(8)), Bv(Width(8))] -> [Bv(Width(8)), Bool]: expected bool, got bv8
struct Binary { args: values(2) }
op Pair<T: ScalarInteger>(lhs: Value<T>, rhs: Value<T>) -> (Value<T>, Value<Type::BOOL>) {
    meta = OpInfo {};
    mnemonic = "pair"; storage = Binary { args: [lhs, rhs] };
    semantics = [bv.add(lhs, rhs), bv.add(lhs, rhs)]
; }

// ----- typed-semantics/multiple-results-require-matching-sorts-and-counts-2
// run: opgen-error
// check: semantic result count does not match signature
struct Binary { args: values(2) }
op Pair<T: ScalarInteger>(lhs: Value<T>, rhs: Value<T>) -> (Value<T>, Value<Type::BOOL>) {
    meta = OpInfo {};
    mnemonic = "pair"; storage = Binary { args: [lhs, rhs] };
    semantics = bv.add(lhs, rhs)
; }

// ----- typed-semantics/traps-are-typed-behavior-not-input-assumptions-1
// run: opgen-error
// check: invalid semantic types [Bv(Width(8)), Bv(Width(8))] -> [Bv(Width(8))]: expected bool, got bv8
struct Binary { args: values(2) }
op Divide<T: ScalarInteger>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    meta = OpInfo {};
    mnemonic = "divide"; storage = Binary { args: [lhs, rhs] };
    semantics = bv.udiv(lhs, rhs);
    traps = [DivisionByZero(rhs)]
; }

// ----- typed-semantics/traps-are-typed-behavior-not-input-assumptions-2
// run: opgen-error
// check: unknown trap `UnknownTrap`
struct Binary { args: values(2) }
op Divide<T: ScalarInteger>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    meta = OpInfo {};
    mnemonic = "divide"; storage = Binary { args: [lhs, rhs] };
    semantics = bv.udiv(lhs, rhs);
    traps = [UnknownTrap(bv.eq(rhs, bv.zero()))]
; }

// ----- typed-semantics/traps-are-typed-behavior-not-input-assumptions-3
// run: opgen-error
// check: trap guards require executable semantics
struct Binary { args: values(2) }
op Divide<T: ScalarInteger>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    meta = OpInfo {};
    mnemonic = "divide"; storage = Binary { args: [lhs, rhs] };

    traps = [DivisionByZero(bv.eq(rhs, bv.zero()))]
; }

// ----- typed-semantics/widths-are-checked-against-every-signature-instance-1
// run: opgen-const-error
// check: invalid semantic types [Bv(Width(16))] -> [Bv(Width(8))]: cannot extend bv16 to bv8
struct Unary { arg: Value }
op Convert<T: Integer, U: Integer>(arg: Value<T>) -> (result: Value<U>) {
    meta = OpInfo {};
    mnemonic = "convert"; storage = Unary { arg: arg };
    verify {
        require(U.same_shape(T), "input and result must have the same shape");
        require(result.element_bits()? < arg.ty().element_bits()?, "result must have fewer bits per lane than arg");
    } semantics = bv.sext(arg, result(0))
; }

// ----- typed-semantics/widths-are-checked-against-every-signature-instance-2
// run: opgen-error
// check: expected 2 results, got 1
struct Unary { arg: Value }
op Convert<T: Integer, U: Integer>(arg: Value<T>) -> (result: Value<U>) {
    meta = OpInfo {};
    mnemonic = "convert"; storage = Unary { arg: arg };
    verify {
        require(U.same_shape(T), "input and result must have the same shape");
        require(result.element_bits()? > arg.ty().element_bits()?, "result must have more bits per lane than arg");
    } semantics = bv.sext(arg, result(1))
; }

// ----- typed-semantics/widths-are-checked-against-every-signature-instance-3
// run: opgen-error
// check: expected type(operand) or result(index)
struct Unary { arg: Value }
op Convert<T: Integer, U: Integer>(arg: Value<T>) -> (result: Value<U>) {
    meta = OpInfo {};
    mnemonic = "convert"; storage = Unary { arg: arg };
    verify {
        require(U.same_shape(T), "input and result must have the same shape");
        require(result.element_bits()? > arg.ty().element_bits()?, "result must have more bits per lane than arg");
    } semantics = bv.sext(arg, type(missing))
; }

// ----- typed-semantics/widths-are-checked-against-every-signature-instance-4
// run: opgen-const-error
// check: invalid semantic types [Bv(Width(8))] -> [Bv(Width(16))]: expected bv16, got bv8
struct Unary { arg: Value }
op Convert<T: Integer, U: Integer>(arg: Value<T>) -> (result: Value<U>) {
    meta = OpInfo {};
    mnemonic = "convert"; storage = Unary { arg: arg };
    verify {
        require(U.same_shape(T), "input and result must have the same shape");
        require(result.element_bits()? > arg.ty().element_bits()?, "result must have more bits per lane than arg");
    } semantics = arg
; }

// ----- typed-semantics/widths-are-checked-against-every-signature-instance-5
// run: opgen-const-error
// check: semantic recipes require a shared lane shape
struct Unary { arg: Value }
op Convert<T: Integer>(arg: Value<T>) -> (result: Value<Type::I32X4>) {
    meta = OpInfo {};
    mnemonic = "convert"; storage = Unary { arg: arg };
    verify {
        require(result.element_bits()? > arg.ty().element_bits()?, "result must have more bits per lane than arg");
    } semantics = bv.sext(arg, result(0))
; }

// ----- typed-semantics/independent-generics-do-not-imply-equal-shapes
// run: opgen-const-error
// check: semantic recipes require a shared lane shape
struct Unary { arg: Value }
op Convert<T: Integer, U: Integer>(arg: Value<T>) -> Value<U> {
    meta = OpInfo {};
    mnemonic = "convert"; storage = Unary { arg };
    verify { require(U.wider_than(T), "wider"); }
    semantics = bv.sext(arg, result(0))
; }

// ----- typed-semantics/equal-lane-counts-do-not-imply-equal-scalability
// run: opgen-const-error
// check: semantic recipes require a shared lane shape
struct Unary { arg: Value }
op Convert<T: Integer & Vector, U: Integer & Vector>(arg: Value<T>) -> Value<U> {
    meta = OpInfo {};
    mnemonic = "convert"; storage = Unary { arg };
    verify {
        require(U.lanes()? == T.lanes()?, "equal lane counts");
        require(U.wider_than(T), "wider");
    }
    semantics = bv.sext(arg, result(0))
; }

// ----- typed-semantics/shape-is-an-ordinary-predicate-not-a-type-pattern
// run: opgen-error
// check: unknown type pattern `shape`
struct Unary { arg: Value }
op Convert<T: Integer>(arg: Value<T>) -> Value<shape(T, Integer)> {
    meta = OpInfo {}; mnemonic = "convert"; storage = Unary { arg };
}

// ----- semantic-facts/algebraic-constants-are-typed-and-names-round-trip-1
// run: opgen-error
// check: unknown algebraic constant `Two`
struct Args { args: values(2) }
op Test<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> (result: Value<T>) {
    meta = OpInfo { memory: MemoryEffect::NONE };
             mnemonic = "test"; storage = Args { args: [lhs, rhs] };
             semantics = bv.add(lhs, rhs); identity = Two
; }

// ----- semantic-facts/rejects-incorrect-identity-and-absorbing-constants-1
// run: opgen-error
// check: bv.add does not support identity `One` at every supported width
struct Args { args: values(2) }
op Test<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> (result: Value<T>) {
    meta = OpInfo { traits: OpTraits::COMMUTATIVE.union(OpTraits::ASSOCIATIVE), memory: MemoryEffect::NONE };
             mnemonic = "test"; storage = Args { args: [lhs, rhs] };
             semantics = bv.add(lhs, rhs);  identity = One
; }

// ----- semantic-facts/rejects-incorrect-identity-and-absorbing-constants-2
// run: opgen-error
// check: bv.and does not support identity `Zero` at every supported width
struct Args { args: values(2) }
op Test<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> (result: Value<T>) {
    meta = OpInfo { traits: OpTraits::COMMUTATIVE.union(OpTraits::ASSOCIATIVE), memory: MemoryEffect::NONE };
             mnemonic = "test"; storage = Args { args: [lhs, rhs] };
             semantics = bv.and(lhs, rhs);  identity = Zero
; }

// ----- semantic-facts/rejects-incorrect-identity-and-absorbing-constants-3
// run: opgen-error
// check: bv.mul does not support absorbing `One` at every supported width
struct Args { args: values(2) }
op Test<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> (result: Value<T>) {
    meta = OpInfo { traits: OpTraits::COMMUTATIVE.union(OpTraits::ASSOCIATIVE), memory: MemoryEffect::NONE };
             mnemonic = "test"; storage = Args { args: [lhs, rhs] };
             semantics = bv.mul(lhs, rhs);  absorbing = One
; }

// ----- semantic-facts/rejects-incorrect-identity-and-absorbing-constants-4
// run: opgen-error
// check: bv.or does not support absorbing `Zero` at every supported width
struct Args { args: values(2) }
op Test<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> (result: Value<T>) {
    meta = OpInfo { traits: OpTraits::COMMUTATIVE.union(OpTraits::ASSOCIATIVE), memory: MemoryEffect::NONE };
             mnemonic = "test"; storage = Args { args: [lhs, rhs] };
             semantics = bv.or(lhs, rhs);  absorbing = Zero
; }

// ----- semantic-facts/rejects-incorrect-identity-and-absorbing-constants-5
// run: opgen-error
// check: bv.xor does not support absorbing `Zero` at every supported width
struct Args { args: values(2) }
op Test<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> (result: Value<T>) {
    meta = OpInfo { traits: OpTraits::COMMUTATIVE.union(OpTraits::ASSOCIATIVE), memory: MemoryEffect::NONE };
             mnemonic = "test"; storage = Args { args: [lhs, rhs] };
             semantics = bv.xor(lhs, rhs);  absorbing = Zero
; }

// ----- semantic-facts/rejects-traits-not-supported-by-the-semantic-primitive-1
// run: opgen-const-error
// check: semantics do not justify COMMUTATIVE
struct Args { args: values(2) }
op Test<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> (result: Value<T>) {
    meta = OpInfo { traits: OpTraits::COMMUTATIVE, memory: MemoryEffect::NONE };
             mnemonic = "test"; storage = Args { args: [lhs, rhs] };
semantics = bv.sub(lhs, rhs); }

// ----- semantic-facts/rejects-traits-not-supported-by-the-semantic-primitive-2
// run: opgen-const-error
// check: semantics do not justify ASSOCIATIVE
struct Args { args: values(2) }
op Test<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> (result: Value<T>) {
    meta = OpInfo { traits: OpTraits::ASSOCIATIVE, memory: MemoryEffect::NONE };
             mnemonic = "test"; storage = Args { args: [lhs, rhs] };
semantics = bv.sub(lhs, rhs); }

// ----- semantic-facts/rejects-traits-not-supported-by-the-semantic-primitive-3
// run: opgen-const-error
// check: semantics do not justify COMMUTATIVE
struct Args { args: values(1) }
op Test<T: Integer>(arg: Value<T>) -> (result: Value<T>) {
    meta = OpInfo { traits: OpTraits::COMMUTATIVE, memory: MemoryEffect::NONE };
             mnemonic = "test"; storage = Args { args: [arg] };
semantics = bv.neg(arg); }

// ----- semantic-facts/rejects-traits-not-supported-by-the-semantic-primitive-4
// run: opgen-const-error
// check: semantics do not justify IDEMPOTENT
struct Args { args: values(2) }
op Test<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> (result: Value<T>) {
    meta = OpInfo { traits: OpTraits::COMMUTATIVE.union(OpTraits::ASSOCIATIVE).union(OpTraits::IDEMPOTENT), memory: MemoryEffect::NONE };
             mnemonic = "test"; storage = Args { args: [lhs, rhs] };
semantics = bv.add(lhs, rhs); }

// ----- semantic-facts/rejects-traits-not-supported-by-the-semantic-primitive-5
// run: opgen-const-error
// check: semantics do not justify IDEMPOTENT
struct Args { args: values(2) }
op Test<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> (result: Value<T>) {
    meta = OpInfo { traits: OpTraits::COMMUTATIVE.union(OpTraits::ASSOCIATIVE).union(OpTraits::IDEMPOTENT), memory: MemoryEffect::NONE };
             mnemonic = "test"; storage = Args { args: [lhs, rhs] };
semantics = bv.mul(lhs, rhs); }

// ----- semantic-facts/rejects-traits-not-supported-by-the-semantic-primitive-6
// run: opgen-const-error
// check: semantics do not justify IDEMPOTENT
struct Args { args: values(2) }
op Test<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> (result: Value<T>) {
    meta = OpInfo { traits: OpTraits::COMMUTATIVE.union(OpTraits::ASSOCIATIVE).union(OpTraits::IDEMPOTENT), memory: MemoryEffect::NONE };
             mnemonic = "test"; storage = Args { args: [lhs, rhs] };
semantics = bv.xor(lhs, rhs); }

// ----- effects/unknown-is-not-a-member
// run: opgen-error
// check: unknown projection function `MemoryEffects::unknown`
struct Effectful {  }
op Effectful() -> () {
    meta = OpInfo { memory: MemoryEffect::known(MemoryEffects::unknown) };
mnemonic = "effectful"; storage = Effectful {}; }

// ----- effects/duplicate-behavior
// run: opgen-const
// check: constant contracts verified
struct Effectful {  }
op Effectful() -> () {
    meta = OpInfo { memory: MemoryEffect::known(MemoryEffects::READ.union(MemoryEffects::READ)) };
mnemonic = "effectful"; storage = Effectful {}; }

// ----- effects/old-alias-is-not-supported
// run: opgen-error
// check: unknown field
struct Effectful {  }
op Effectful() -> () {
    meta = OpInfo { memory: MemoryEffect::UNKNOWN };
    mnemonic = "effectful"; storage = Effectful {}; memory = READ
; }

// ----- effects/old-effect-declaration-is-not-supported
// run: opgen-error
// check: unknown definition kind
effect CUSTOM { reads = true; }

// ----- metadata/arbitrary-typed-metadata
// run: opgen
// check: pub struct Tuning
// check: pub enum Policy
// check: pub meta: crate::inst::Tuning
// check: const fn meta(self)
// check: MemoryEffects>::union
enum Policy { variants = [Automatic, Prefer(MemoryEffects)]; }
struct Tuning { policy: Policy, budget: u32 }
struct Work {  }
op Work() -> () {
    meta = Tuning { policy: Prefer(MemoryEffects::READ.union(MemoryEffects::WRITE)), budget: 8 };
    mnemonic = "work"; storage = Work {}
; }

// ----- metadata/nested-records-and-explicit-values
// run: opgen
// check: crate::inst::Settings { enabled: true, mask: <veloc_types::MemoryEffects as veloc_types::traits::MemoryEffects>::READ }
struct Settings { enabled: bool, mask: MemoryEffects }
enum Policy { variants = [Automatic, Configured(Settings)]; }
struct Tuning { policy: Policy, fallback: optional(Policy) }
struct Work {  }
op Work() -> () {
    meta = Tuning { policy: Configured(Settings { enabled: true, mask: MemoryEffects::READ }), fallback: none }; mnemonic = "work"; storage = Work {}
; }

// ----- metadata/unknown-metadata-field
// run: opgen-error
// check: unknown field `budegt`
struct Tuning { budget: u8 }
struct Work {  }
op Work() -> () { meta = Tuning { budegt: 8 }; mnemonic = "work"; storage = Work {}; }

// ----- metadata/missing-metadata-field
// run: opgen-error
// check: missing field `budget`
struct Tuning { budget: u8 }
struct Work {  }
op Work() -> () { meta = Tuning {}; mnemonic = "work"; storage = Work {}; }

// ----- metadata/wrong-enum-argument
// run: opgen-error
// check: integer literal is out of range for its type
enum Policy { variants = [Fixed(u8)]; }
struct Tuning { policy: Policy }
struct Work {  }
op Work() -> () { meta = Tuning { policy: Fixed(256) }; mnemonic = "work"; storage = Work {}; }

// ----- metadata/recursive-inline-data
// run: opgen-error
// check: recursive inline data type
struct Cycle { next: optional(Choice) }
enum Choice { variants = [End, Link(Cycle)]; }

// ----- metadata/metadata-cannot-contain-ssa-values
// run: opgen-error
// check: metadata cannot contain SSA Value fields
struct Tuning { input: Value }
struct Work {  }
op Work() -> () { meta = Tuning { input: 0 }; mnemonic = "work"; storage = Work {}; }

// ----- metadata/old-traits-entry-is-rejected
// run: opgen-error
// check: unknown field `traits`
struct Work {  }
op Work() -> () { meta = OpInfo { memory: MemoryEffect::UNKNOWN }; traits = OpTraits::empty(); mnemonic = "work"; storage = Work {}; }

// ----- metadata/analysis-fields-can-have-different-names
// run: opgen
// check: pub const fn traits(&self) -> veloc_types::OpTraits
// check: self.meta.attributes
// check: self.meta.accesses
struct Info { attributes: OpTraits, accesses: MemoryEffect, extra: MemoryEffects }
struct Pair { args: values(2) }
op Sum<T: ScalarInteger>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    meta = Info { extra: MemoryEffects::WRITE.union(MemoryEffects::READ) };
    mnemonic = "sum";
    storage = Pair { args: [lhs, rhs] };
    semantics = bv.add(lhs, rhs)
; }

// ----- comparisons/legacy-declaration-is-rejected
// run: opgen-error
// check: unknown definition kind `comparison`
comparison TestCC { domain = float; predicates = [Eq([equal])]; }
