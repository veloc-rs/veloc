// ----- type-members/cannot-overwrite-binding
// run: opgen-error
// check: unknown field `expr`
type Example = rust("crate::Example") {
    expr = Type::I32;
}

// ----- interfaces/recursive-helpers-even-if-unused
// run: opgen-error
// check: recursive projection functions are not supported
fn First(n: u32) -> u32 { value = Second(n); }
fn Second(n: u32) -> u32 { value = First(n); }

// ----- interfaces/unused-function-body-is-checked
// run: opgen-error
// check: projection type mismatch
fn Bad(n: bool) -> u32 { value = n; }

// ----- interfaces/recursive-data
// run: opgen-error
// check: recursive inline data type
struct First { other: optional(Second) }
struct Second { other: First }

// ----- interfaces/explicit-lossless-conversion
// run: opgen-error
// check: projection conversion must be lossless
fn Narrow(n: u64) -> u32 { value = u32(n); }

// ----- interfaces/old-record-keyword-is-rejected
// run: opgen-error
// check: unknown definition kind `record`
record Legacy {}

// ----- host/arbitrary-host-methods-are-declaration-driven
// run: opgen
// check: pub trait Numbers
// check: fn next(&self, n: u32) -> Option<u32>
// check: <crate::host::Numbers as crate::type_methods::Numbers>::next(_context, *_f0)
type Numbers = rust("crate::host::Numbers") {
    fn next(&self, n: u32) -> optional(u32);
}
fn Count(ctx: &Numbers, n: u32) -> u32 { value = ctx.next(n)?; }
struct Data { n: u32 }
op Example(n: u32) -> () {
    meta = OpInfo { memory: MemoryEffect::NONE }; mnemonic = "example";
    storage = Data { n: n };
    verify(ctx: Numbers) { require(Count(ctx, n) > n, "no next value"); }
}

// ----- host/helper-requirements-reach-the-caller
// run: opgen
// check: <crate::host::VerifyContext as crate::type_methods::VerifyContextInfo>::signature(_context, *_f0)
fn Count(ctx: &VerifyContext, sig: SigId) -> i128 { value = len(ctx.signature(sig)?.params()); }
struct Data { sig: SigId }
op Example(sig: SigId) -> () {
    meta = OpInfo { memory: MemoryEffect::NONE }; mnemonic = "example";
    storage = Data { sig: sig };
    verify(ctx: VerifyContext) { require(Count(ctx, sig) > 0, "expected parameters"); }
}

// ----- host/optional-host-results-must-be-handled
// run: opgen-error
// check: projection type mismatch
fn Count(ctx: &VerifyContext, sig: SigId) -> &Signature { value = ctx.signature(sig); }

// ----- host/try-only-applies-to-optionals
// run: opgen-error
// check: ? requires an optional value
fn Next(n: u32) -> u32 { value = n?; }

// ----- host/host-method-signatures-are-checked
// run: opgen-error
// check: projection type mismatch
fn Bad(ctx: &VerifyContext, n: u32) -> optional(&Signature) { value = ctx.function_signature(n); }

// ----- host/legacy-module-query-is-not-a-hidden-hook
// run: opgen-error
// check: unknown expression name or operation
fn Bad(sig: SigId) -> i128 { value = len(params(sig)); }

// ----- host/interface-in-clause-is-removed
// run: opgen-error
// check: expected `{`
extern struct Legacy in [query] { fn next(n: u32) -> u32; }

// ----- host/context-declarations-are-removed
// run: opgen-error
// check: unknown definition kind `context`
context query {}

// ----- host/unused-host-needs-no-adapter
// run: opgen
// check: pub trait Unused
// not: let _host
type Unused = rust("crate::host::Unused") { fn number(&self) -> u32; }
struct Data { n: u32 }
op Example(n: u32) -> () {
    meta = OpInfo { memory: MemoryEffect::NONE }; mnemonic = "example";
    storage = Data { n: n };
}

// ----- rust-types/declared-paths-are-shared-by-fields-and-host-signatures
// run: opgen
// check: pub value: crate::tokens::Token
// check: Some(crate::tokens::Token)
// check: fn read(&self, value: crate::tokens::Token) -> Option<crate::tokens::Token>
type Token = rust("crate::tokens::Token");
type Tokens = rust("crate::host::Tokens") { fn read(&self, value: Token) -> optional(Token); }
struct Entry { value: Token }
enum EntryResult { variants = [Some(Token), None]; }

// ----- rust-types/foreign-types-are-not-ssa-types
// run: opgen-error
// check: Token
type Token = rust("crate::tokens::Token");
struct Empty {}
op Example() -> Value<Token> {
    mnemonic = "example"; meta = OpInfo { memory: MemoryEffect::NONE }; storage = Empty {};
}

// ----- rust-types/foreign-types-are-not-structural-type-aliases
// run: opgen-error
// check: TOKEN
type TOKEN = rust("crate::tokens::Token");
type BAD = vector(TOKEN, 4);

// ----- rust-types/foreign-types-are-nominal
// run: opgen-error
// check: projection type mismatch
type First = rust("crate::tokens::Token");
type Second = rust("crate::tokens::Token");
type Tokens = rust("crate::host::Tokens") { fn first(&self) -> First; }
fn wrong(ctx: &Tokens) -> Second { value = ctx.first(); }

// ----- rust-types/foreign-types-have-no-implicit-fields
// run: opgen-error
// check: field
type Token = rust("crate::tokens::Token");
fn wrong(token: Token) -> u32 { value = token.number; }

// ----- rust-types/foreign-types-have-no-implicit-arithmetic
// run: opgen-error
// check: unsupported expression operator or operand type
type Token = rust("crate::tokens::Token");
fn wrong(token: Token) -> Token { value = token + token; }

// ----- rust-types/binding-needs-one-path
// run: opgen-error
// check: rust requires one type path string
type Token = rust();

// ----- rust-types/path-is-not-rust-code
// run: opgen-error
// check: Rust type path must be nonempty and qualified
type Token = rust("crate::Token; panic!()");

// ----- rust-types/binding-cannot-shadow-a-primitive
// run: opgen-error
// check: conflicts with a built-in type
type u32 = rust("crate::Token");

// ----- rust-types/binding-cannot-duplicate-generated-data
// run: opgen-error
// check: duplicate data type
type Token = rust("crate::tokens::Token");
struct Token { number: u32 }

// ----- rust-types/binding-does-not-silently-declare-other-types
// run: opgen-error
// check: unknown interface type
type Tokens = rust("crate::host::Tokens") { fn read(&self, value: Undeclared) -> u32; }

// ----- type-methods/methods-share-the-type-declaration-and-expand-defs-bodies
// run: opgen
// check: pub trait Token: Sized
// not: impl Token for
// check: crate::type_methods::Token>::number
// check: checked_mul
type Token = rust("crate::tokens::Token") {
    fn number(self) -> u32;
    fn twice(self) -> u32 { value = self.number() * 2; }
}
type Tokens = rust("crate::host::Tokens") { fn read(&self, number: u32) -> Token; }
struct Input { number: u32 }
op Check(number: u32) -> () {
    meta = OpInfo { memory: MemoryEffect::NONE }; mnemonic = "check"; storage = Input { number };
    verify(ctx: Tokens) { require(ctx.read(number).twice() > 0, "positive"); }
}

// ----- type-methods/external-trait-binding-selects-the-method-contract
// run: opgen
// check: crate::tokens::Read>::number
type Token = rust("crate::tokens::Token") {
    trait = rust("crate::tokens::Read");
    fn number(self) -> u32;
}
type Tokens = rust("crate::host::Tokens") { fn read(&self, number: u32) -> Token; }
struct Input { number: u32 }
op Check(number: u32) -> () {
    meta = OpInfo { memory: MemoryEffect::NONE }; mnemonic = "check"; storage = Input { number };
    verify(ctx: Tokens) { require(ctx.read(number).number() > 0, "positive"); }
}

// ----- type-methods/methods-must-be-declared
// run: opgen-error
// check: unknown projection function
fn wrong(ty: Type) -> bool { value = ty.undeclared(); }

// ----- type-methods/results-are-types-not-fake-ssa-values
// run: opgen-error
// check: unknown projection function `Type::ty`
struct Unary { arg: Value }
op Check(arg: Value<Type::I8>) -> (result: Value<Type::I16>) {
    meta = OpInfo { memory: MemoryEffect::NONE }; mnemonic = "check"; storage = Unary { arg };
    verify { require(result.ty().is_scalar(), "scalar"); }
}

// ----- type-methods/result-only-generic-has-a-concrete-slot
// run: opgen
// check: veloc_types::traits::TypeInfo>::element_bits(results[0])
struct Empty {}
op Check<T: Integer>() -> (result: Value<T>) {
    meta = OpInfo { memory: MemoryEffect::NONE }; mnemonic = "check"; storage = Empty {};
    verify { require(T.element_bits()? > 8, "wide"); }
}

// ----- type-methods/generic-and-value-names-cannot-collide
// run: opgen-error
// check: duplicate parameter
struct Unary { arg: Value }
op Check<T: Integer>(T: Value<T>) -> Value<T> {
    meta = OpInfo { memory: MemoryEffect::NONE }; mnemonic = "check"; storage = Unary { arg: T };
}

// ----- type-methods/rust-query-signature-is-checked
// run: opgen-rust-error
// check: mismatched types
fn wrong(ty: Type) -> bool = rust("crate::Type::element_bits");

// ----- type-methods/optional-query-needs-an-explicit-unwrapping-policy
// run: opgen-error
// check: optional operands
fn wrong(ty: Type) -> bool { value = ty.element_bits() > 8; }

// ----- type-methods/old-intrinsic-binding-is-not-kept-as-an-alias
// run: opgen-error
// check: expected rust binding
fn old(ty: Type) -> bool = intrinsic("type.is_owned");

// ----- type-methods/missing-offline-support-is-not-silently-ignored
// run: opgen-error
// check: semantic type constraints require const functions
fn runtime_only(ty: Type) -> bool = rust("crate::Type::is_valid");
struct Unary { arg: Value }
op Check<T: Integer>(arg: Value<T>) -> Value<T> {
    meta = OpInfo { memory: MemoryEffect::NONE }; mnemonic = "check"; storage = Unary { arg };
    verify { require(runtime_only(T), "valid"); }
    semantics = bv.neg(arg)
; }

// ----- type-methods/runtime-only-query-is-allowed-without-offline-semantics
// run: opgen
// check: crate::Type::is_scalable(operands[0])
fn runtime_only(ty: Type) -> bool = rust("crate::Type::is_scalable");
struct Unary { arg: Value }
op Check<T: Integer>(arg: Value<T>) -> Value<T> {
    meta = OpInfo { memory: MemoryEffect::NONE }; mnemonic = "check"; storage = Unary { arg };
    verify { require(runtime_only(T), "scalable"); }
}

// ----- type-methods/type-methods-can-be-used-in-constant-metadata
// run: opgen
// check: veloc_types::traits::TypeInfo>::element_bits(crate::types::I64)
struct Info { count: u32, memory: MemoryEffect }
struct Empty {}
op Check() -> () {
    meta = Info { count: Type::I64.element_bits()?, memory: MemoryEffect::NONE };
    mnemonic = "check"; storage = Empty {};
}

// ----- type-methods/declared-methods-cannot-recurse
// run: opgen-error
// check: recursive projection
type Token = rust("crate::tokens::Token") {
    fn first(self) -> bool { value = self.second(); }
    fn second(self) -> bool { value = self.first(); }
}

// ----- type-methods/short-circuit-and-absence-use-the-shared-type-kernel
// run: opgen
// check: Opcode::Check
struct Unary { arg: Value }
op Check<T: ScalarInteger | Type::PTR>(arg: Value<T>) -> Value<T> {
    meta = OpInfo { memory: MemoryEffect::NONE }; mnemonic = "check"; storage = Unary { arg };
    verify { require(!T.is_ptr() && T.element_bits()? > 0, "sized integer"); }
    semantics = bv.neg(arg)
; }

// ----- type-methods/malformed-default-binding-reports-an-error
// run: opgen-error
// check: unknown function field `rust`
fn wrong() -> bool { rust = default; }

// ----- type-methods/postfix-fields-and-methods-use-the-same-grammar
// run: opgen
// check: crate::type_methods::Token>::number
type Token = rust("crate::tokens::Token") {
    fn number(self) -> u32;
}
type Tokens = rust("crate::host::Tokens") { fn read(&self, number: u32) -> Token; }
struct Input { number: u32 }
struct Boxed { token: Token }
fn unwrap(boxed: Boxed) -> u32 { value = (boxed) . token . number(); }
op Check(number: u32) -> () {
    meta = OpInfo { memory: MemoryEffect::NONE }; mnemonic = "check"; storage = Input { number };
    verify(ctx: Tokens) { require(unwrap(Boxed { token: ctx . read(number) }) > 0, "positive"); }
}

// ----- type-methods/constant-projections-share-the-typed-evaluator
// run: opgen
// check: MemoryEffect>::NONE
struct Pair { lhs: u32, rhs: u32 }
fn same(pair: Pair) -> bool { value = (pair).lhs == pair.rhs; }
struct Empty {}
op Check() -> () {
    meta = OpInfo { memory: MemoryEffect::NONE }; mnemonic = "check"; storage = Empty {};
    verify { require(same(Pair { lhs: 3 * 7, rhs: 21 }), "same"); }
}

// ----- type-methods/methods-respect-the-declared-return-type
// run: opgen-error
// check: projection type mismatch
type Token = rust("crate::tokens::Token") {
    fn erase(self, value: Value(Type::I32)) -> Value { value = value; }
}
fn narrow(token: Token, value: Value(Type::I32)) -> Value(Type::I32) {
    value = token.erase(value)
; }

// ----- type-methods/unused-rust-methods-still-have-a-contract
// run: opgen
// check: pub trait Handle: Sized
// check: fn count(self) -> u32
// not: impl Handle for
type Handle = rust("crate::handles::Handle") { fn count(self) -> u32; }

// ----- type-methods/non-type-methods-must-also-be-declared
// run: opgen-error
// check: unknown projection function
type Handle = rust("crate::handles::Handle");
fn count(handle: Handle) -> u32 { value = handle.count(); }

// ----- type-methods/foreign-const-metadata-needs-no-offline-registration
// run: opgen
// check: pub const trait Token: Sized
// check: crate::type_methods::Token>::number(crate::tokens::new(7u32))
type Token = rust("crate::tokens::Token") {
    const fn number(self) -> u32;
}
const fn token(number: u32) -> Token = rust("crate::tokens::new");
struct Info { count: u32, memory: MemoryEffect }
struct Empty {}
op Check() -> () {
    meta = Info { count: token(7).number(), memory: MemoryEffect::NONE };
    mnemonic = "check"; storage = Empty {};
}

// ----- type-methods/runtime-method-is-not-constant-metadata
// run: opgen-error
// check: metadata projection must be compile-time constant
type Token = rust("crate::tokens::Token") {
    fn number(self) -> u32;
}
const fn token(number: u32) -> Token = rust("crate::tokens::new");
struct Info { count: u32, memory: MemoryEffect }
struct Empty {}
op Check() -> () {
    meta = Info { count: token(7).number(), memory: MemoryEffect::NONE };
    mnemonic = "check"; storage = Empty {};
}

// ----- type-methods/const-helpers-cannot-call-runtime-functions
// run: opgen-error
// check: const function calls a runtime-only operation
fn runtime(number: u32) -> u32 = rust("crate::runtime");
const fn twice(number: u32) -> u32 { value = runtime(number) * 2; }

// ----- type-methods/associated-constants-are-not-function-calls
// run: opgen-error
// check: associated constants are values, not functions
struct Empty {}
op Check() -> () {
    meta = OpInfo { memory: MemoryEffect::NONE() }; mnemonic = "check"; storage = Empty {};
}

// ----- type-methods/associated-constants-do-not-use-dot
// run: opgen-error
// check: unknown expression name or operation: MemoryEffect
struct Empty {}
op Check() -> () {
    meta = OpInfo { memory: MemoryEffect.NONE }; mnemonic = "check"; storage = Empty {};
}

// ----- type-methods/associated-path-requires-a-namespace
// run: opgen-error
// check: :: requires a type or namespace path
struct Empty {}
op Check() -> () {
    meta = OpInfo { memory: MemoryEffect::known(MemoryEffects::empty())::NONE };
    mnemonic = "check"; storage = Empty {};
}

// ----- expressions/helpers-share-verification-and-metadata-expressions
// run: opgen
// check: checked_mul(2u64)
// check: count: 14
// check: valid: true
struct Info { count: u64, valid: bool, memory: MemoryEffect }
fn Twice(n: u64) -> u64 { value = n * 2; }
fn Positive(n: u64) -> bool { value = 0 < n; }
struct Data { n: u64 }
op Example(n: u64) -> () {
    meta = Info { count: Twice(7), valid: Positive(Twice(7)), memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Data { n: n };

    verify {
        require(Positive(Twice(n)), "expected a positive doubled value");
    }
}

// ----- expressions/metadata-preserves-full-width-signed-integers
// run: opgen
// check: high: 18446744073709551615
// check: low: -7
struct Info { high: u64, low: i64, memory: MemoryEffect }
fn High() -> u64 { value = 18446744073709551615; }
fn Low() -> i64 { value = -7; }
struct Empty {}
op Example() -> () {
    meta = Info { high: High(), low: Low(), memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Empty {};
}

// ----- expressions/invalid-helper-result-is-still-type-checked
// run: opgen-error
// check: projection type mismatch
fn Nonzero(n: u32) -> bool { value = n; }

// ----- expressions/helper-does-not-silently-narrow-arguments
// run: opgen-error
// check: projection type mismatch
struct Data { n: u64 }
op Example(n: u64) -> () {
    meta = OpInfo { memory: MemoryEffect::NONE };
    mnemonic = "example"; storage = Data { n: n };
    verify {
        require(is_power_of_two(n), "bad alignment");
    }
}

// ----- expressions/helpers-cannot-shadow-primitives
// run: opgen-error
// check: function conflicts with a projection primitive
fn len(n: u32) -> u32 { value = n; }

// ----- expressions/legacy-constraints-have-no-compatibility-layer
// run: opgen-error
// check: use a verify block instead of constraints
struct Data {}
op Example() -> Value<Type::I32> { constraints = [true]; }

// ----- borrowed-types/records-cannot-retain-borrowed-views
// run: opgen-error
// check: expected data type name
struct Escaped { signature: &Signature }

// ----- borrowed-types/receiver-kind-is-checked
// run: opgen-error
// check: method receiver type mismatch
type Token = rust("crate::tokens::Token") { fn consume(self) -> u32; }
fn count(token: &Token) -> u32 { value = token.consume(); }

// ----- borrowed-types/context-must-bind-a-rust-type
// run: opgen-error
// check: context requires a Rust-bound type
struct Empty {}
op Example() -> () {
    meta = OpInfo { memory: MemoryEffect::NONE }; mnemonic = "example"; storage = Empty {};
    verify(ctx: Empty) { true; }
}

// ----- borrowed-types/legacy-extern-interface-is-rejected
// run: opgen-error
// check: expected `{`
extern struct Legacy { fn count() -> u32; }

// ----- interfaces/legacy-interface-is-rejected
// run: opgen-error
// check: unknown definition kind `interface`
interface Old { count = u32; }

// ----- interfaces/legacy-implements-is-rejected
// run: opgen-error
// check: unknown field `implements`
struct Info { count: u32 }
struct Empty {}
op A() -> () {
    meta = OpInfo { memory: MemoryEffect::NONE }; mnemonic = "a"; storage = Empty {};
    implements = [Info { count: 1 }];
}
