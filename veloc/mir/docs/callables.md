# Typed callable values in SSA MIR

MIR remains an SSA control-flow IR. Functions contain blocks, block arguments and
ordinary instructions whose results feed later instructions. Typed callable values
add closures to that model: a function and an explicit capture list become a value
that can be passed, returned, and called with the remaining arguments.

`call-value` is an ordinary call. It produces zero, one, or multiple SSA results,
and execution continues at the next instruction when the callee returns. Its
signature comes from the callee value's type; the instruction does not repeat it.
The enclosing function can return different types from those of the callable.

```text
local function main() -> i32
block0():
  v0: i32 = iconst 20
  v1: shared<(i32) -> i32> = closure-shared add(v0) : (i32, i32) -> i32
  v2: i32 = iconst 1
  v3: i32 = call-value v1(v2)
  v4: i32 = call-value v1(v2)
  v5: i32 = iadd v3, v4
  return v5

local function add(i32, i32) -> i32
block0(v0: i32, v1: i32):
  v2: i32 = iadd v0, v1
  return v2
```

The example returns 42. Both calls return to `main`, which combines their results.
Calls with multiple results use the existing `(v0: i32, v1: i64) = ...` syntax;
calls returning void have no result assignment.

## Signatures and ownership

A capture list binds a prefix of the target function's parameters. The remaining
parameters and the function's result types form the callable signature. Callable
types live in the same `Type`, DFG and use-def chains as scalar values; they are not
raw function pointers or registers outside SSA.

| Type | Captures and lifetime | Call discipline |
| --- | --- | --- |
| `local<(I...) -> R>` | Borrows its activation delimiter; may capture raw pointers and other local/shared callables | Repeatable while that delimiter is alive |
| `owned<(I...) -> R>` | Owns its captures, including other owned callables; has an explicit cleanup function | One-shot; consumed by a call, tail call, or drop |
| `shared<(I...) -> R>` | Immutable, duplicable captures, including other shared callables | Repeatable |

The shared environment is immutable, but its function can perform effects.
Repeated calls execute the body again against current memory; they do not roll
back previous effects or restore a snapshot.

Owned/shared environments cannot capture local callables or raw pointers. Local
environments cannot capture owned callables, since repeated calls would duplicate
ownership. These are the existing conservative capture restrictions, not a proof
of arbitrary pointer provenance or a new general memory ownership system.
Integer-to-pointer conversions and foreign memory retain their low-level contract.

Local values may flow through block arguments, function arguments and other local
environments. They cannot escape through function results, escaping environments,
globals or ordinary data memory operations. Owned/shared values can be returned
through typed function results. Host imports accepting or returning callable types
remain rejected until an ownership-aware ABI exists. Raw-pointer indirect calls
and intrinsic calls cannot erase a callable's ownership contract either.

The current three kinds combine lifetime and call multiplicity. In particular,
there is no separate scoped one-shot kind that both borrows stack data and moves
owned captures. A future expansion must specify those lifetime and duplication
rules explicitly.

## Operations

| Operation | Behavior |
| --- | --- |
| `closure-new` | Create an owned closure by binding a target's parameter prefix; its cleanup consumes those same captured parameters and returns void |
| `closure-local` | Create a borrowed, repeatable closure |
| `closure-shared` | Create a closure with immutable, duplicable captures |
| `call-value` | Call a typed callable, produce its SSA results, then continue |
| `tail-call` | Transfer to a named function without a return to the current instruction stream |
| `tail-call-value` | Transfer to a typed callable without a return to the current instruction stream |
| `closure-drop` | Consume an owned closure and call its cleanup, then continue |

`tail-call` and `tail-call-value` are terminators. Their result types must match
the enclosing function's returns. `call-value` has no such answer-type constraint:
its results are checked against the callable signature and can be used normally.
Both call forms consume an owned callee; local and shared callees remain reusable.
Ordinary `call` and `return` retain their direct-call behavior.

Closure creation does not execute its body. Calling an owned closure transfers
its captures into its body; dropping it instead transfers those captures into its
cleanup. Cleanup can explicitly drop captured owned closures and perform resource
effects. Successful calls do not subsequently invoke the closure's cleanup.

## Explicit CPS remains possible

A frontend can pass a callable representing subsequent work and enter it with
`tail-call-value`. A handler can accept a payload and an owned callable for the
rest of a computation, then call it, tail-call it, transfer it, or drop it.
Representing that rest requires explicit capture of every dependency it needs.
This is a possible encoding in the SSA IR; ordinary closure use does not require
a frontend to convert its program to CPS.

A typed closure is a target plus saved arguments. Creating one does not capture
the current instruction position, arbitrary callers, a handler stack or a native
stack. Returning a closure can preserve explicitly packaged work for a later
invocation, but it is not a coroutine suspension operation. Automatic CPS
conversion, suspended coroutine state, `shift/reset` and opaque stack capture
are not implemented by these operations.

## Definitions and validation

`mir.spec` specifies callable operations, signatures, capture constraints, ownership
transfers, effects and text syntax. Ordinary signatures and `verify` predicates
describe validity; `MAY_TRAP` and `TERMINATOR` are explicit metadata rather than
facts inferred from a separate control declaration.
The interpreter lowers instruction views directly into compiled control sites.
It records live callable roots at those sites and ordinary calls; this execution
requirement is independent of trap metadata. Native lowering rejects callable
types and unsupported tail calls explicitly.
The reserved `Callable` type pattern denotes structural callable values.

The validator checks signature references and types, instruction and operand
structure, generated contracts, SSA dominance, and ownership along reachable CFG
edges. Construction itself remains unchecked. Recursive signature cycles require
an explicit recursive-type design and are rejected.

Ownership uses forward dataflow, not a global SSA use count. Each successor starts
with its own copy of the incoming state and moves arguments into block parameters.
Alternative branch edges may transfer the same owned value; one edge cannot move
it twice. States must agree at joins. Every normal exit must call, transfer or drop
all outstanding owned values, and loops can carry ownership without duplication.
An owned result from `call-value` enters that same ownership analysis.

Unreachable blocks still receive structural/type checks. Reachable dominance and
ownership start at the entry block; changes to reachability require revalidation.

Traps abort the invocation. They do not run implicit guest cleanup or unwind owned
closures. Inputs already moved into an invocation remain consumed after a trap;
unreachable environments are reclaimed without cleanup calls. Recoverable errors
and cleanup must have explicit control-flow paths.

## Interpreter and optimization boundary

The bytecode interpreter executes closures and typed calls alongside existing SSA
arithmetic, memory and direct calls. Environments currently occupy a traced table
with non-reused handles and typed captures. Safepoint maps identify live callable
registers, including suspended callers of ordinary calls; collection never treats
pointer-looking integer bits as roots.

An owned call or drop removes its handle before entering the body or cleanup.
Returned owned/shared closures are held as external roots and can be passed into
later invocations of the same interpreter/program. Host copies do not duplicate
ownership. `release_shared` releases an externally held shared handle; owned
handles must be moved back into MIR and called or dropped.

Ordinary calls retain their caller and result destinations. Tail calls reuse the
VM register frame while retaining the existing activation delimiter. Borrowed
stack slots stay alive until that delimiter returns; unbounded tail loops that
allocate stack slots can therefore exhaust stack memory. Scalar tail loops do not
grow the call stack.

The interpreter's environment table is an implementation choice. Allocation
elimination, known-target specialization, contification and native placement are
future optimization work. Creation and ownership changes currently carry
conservative effects and are not ordinary arithmetic CSE/DCE candidates. Removing
an allocation requires deciding its failure and ownership semantics, as well as
proving that captures and observable cleanup behavior are preserved.

Native code generation explicitly rejects typed callables and tail calls until
closure/environment and calling-convention lowering exists. This work introduces
no automatic performance claim and no coroutine suspension implementation.

## CFG infrastructure

The final instruction's successor occurrences remain authoritative; adjacency is
a deduplicated analysis index. `EdgeRef { inst, index }` edits an individual edge,
including parallel edges to the same block. Installation updates SSA uses and
predecessor relations. SSA sealing remains builder-local, and dominance analysis
is independent of source block order. Callable operations use this existing CFG
and SSA infrastructure.
