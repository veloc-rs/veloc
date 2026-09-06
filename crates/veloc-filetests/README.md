# Compiler file tests

Run the file regressions and the generated-API execution tests:

```sh
cargo test -p veloc-filetests -p veloc-test-mir
cargo test -p veloc-filetests --test files opgen/
cargo test -p veloc-filetests --test files optimizer/simplify
cargo test --workspace
```

Each `.mir` or `.ops` file under `cases/` is discovered automatically. Use
`// ----- name` to separate independent cases in one file. Every case needs
exactly one `// run:` and at least one positive output check. Names and filters
appear in Cargo's test output; failures include the actual output and matching
diagnostics. There are no shell commands or output snapshots to update blindly.

## Checks

Checks use the Rust [filecheck](https://docs.rs/filecheck/0.5.0/filecheck/) library,
whose syntax differs from LLVM FileCheck:

```text
// run: simplify
// check: $(sum=v\d+) = iconst.i32 7
// not: iadd
// check: return $sum
local function add() -> i32
block0():
  v0 = iconst.i32 3
  v1 = iconst.i32 4
  v2 = iadd v0, v1
  return v2
```

`check:` matches in order; `nextln:` checks the next line; `unordered:` checks a
group without imposing an order. `not:` checks absence **between surrounding
matches**, not across the entire output. Capture SSA/register names instead of
depending on their numbering. Check operands, result types and uses, not just
the presence of an opcode. Assertions belong in `//` comments.

## Stages

| `run:` | Observable behavior |
| --- | --- |
| `roundtrip` | Parse, validate, print, reparse, validate, and check canonical text stability |
| `parse-error` | Require a parser diagnostic |
| `validate-error` | Parsing must succeed; require a validator diagnostic |
| `simplify` | Run simplify, compare cached/rebuilt use-def, check a fixed point, validate and round-trip |
| `o1` | Run the production O1 pipeline, validate and round-trip |
| `lower` | Translate MIR to LIR; print instructions, operands and register types |
| `execute` | Compare interpreter results/traps before and after O1, then check the expected outcome |
| `opgen-error` | Require a definition compiler diagnostic |
| `fixture` | Round-trip through MIR compiled with test-only operation definitions |
| `fixture-error` | Require a parser diagnostic from that test MIR |
| `fixture-validate-error` | Parsing succeeds, but test MIR validation fails |

The `.ops` driver prepends MIR's `types.ops`, `builtins.ops` and
`comparisons.ops`. Input files explicitly supply their operation/storage records.
This driver is for definitions using the standard type encoding; low-level
encoding changes and definition-order metamorphic tests stay in `veloc/spec`.

`execute` currently calls a parameterless `main`, compares scalar return bits
using their declared widths, and allows stack memory but no external memory.
It is not a JIT differential test. Rust panics fail the test; they are never
accepted as modeled traps. In particular, interpreter integer division by zero
currently panics, so division-trap coverage remains in semantic execution and
optimization-preservation tests rather than being treated as a passing runtime
trap test here.

## Generator coverage

`fixture/extra.ops` extends production MIR with unusual contracts: custom field
names, reversed text order, multiple results, result-only bindings, alternate
storage, predicates and composite semantics. `fixture` compiles the **real MIR
sources** against those definitions; it does not copy the runtime or mock types.
Its integration tests execute generated builders, predicates, comparison
transforms, evaluators and direct-lowering mappings. No generated handler IDs
are inspected. Ordinary builds of `veloc-mir` do not generate this fixture.

Keep small Rust tests for API-only invariants, exhaustive/property-style checks,
malformed in-memory IR that text cannot construct, build determinism, and
short-circuit/pool-access instrumentation. Existing semantic differential tests
and frontend program suites remain complementary coverage.
