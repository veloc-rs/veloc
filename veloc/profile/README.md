# Compiler profiling

`veloc-profile` is a std-only observation library with no compiler IR, target,
logging framework or serialization dependency. It does not control pass ordering,
analysis invalidation or optimization decisions.

## Collection

Create one `Profile` per compilation and pass clones to the components involved.
The default handle is disabled: it allocates nothing, reads no clock and does not
evaluate lazy metric/entity/detail producers. Enabled collection has overhead;
it is not intended as an always-on benchmark configuration.

- `Summary`: aggregate calls, failures, inclusive/self time and typed metrics.
- `Trace`: also retain real session-relative timestamps and function identities.
- `Config::details`: additionally retain lazy remarks and textual IR artifacts.
  Events and detail bytes are bounded independently; truncation is reported.

A stage is identified by its parent, static name and pipeline position. The
position distinguishes repeated occurrences of a pass, not function invocations.
Entity names belong to trace invocations, so summaries do not grow per function.
Analysis computations are nested under their requesting pass; a cache hit records
a counter without inventing another analysis duration.

```rust
use veloc_profile::{Config, Metric, Mode, Profile};

let profile = Profile::new(Config { mode: Mode::Summary, ..Config::default() });
let result: Result<(), &'static str> = profile.measure("compile", 0, || {
    let pass = profile.scope("simplify", 2);
    profile.count("rewrites", 4);
    profile.record_lazy(Metric::bytes("output"), || 128);
    pass.success();
    Ok(())
});
let report = profile.report(); // All scopes must be closed first.
println!("{report}");
assert!(result.is_ok());
```

Use `measure` for Result-returning phases: normal errors record `Failed`, while
unwinding records `Interrupted`. Manually created scopes must be finished with
`success`, `result` or `finish`; dropping an unfinished scope is an interruption.
Counters are owned by the active stage, or by `session` outside any scope.
Use `metric` once and `record` with its recorder-local ID for hot counters.
Snapshots should explicitly select Sum/Max/Last; wall time is never the sum of
nested inclusive times. Worker totals may exceed wall time when work overlaps.

`Report::summaries()` exposes structured data for regression tools.
`Report::chrome_trace()` exports a Chrome/Perfetto timeline, aggregate metrics,
metadata and truncation count. Export and formatting happen after compilation.
The caller retains the profile even if compilation returns an error.

## Parallel compilation

A profile is worker-local and deliberately not Send/Sync. Clones share a nested,
synchronous stack, not independent async tasks. Send `profile.fork(track)` to a
worker, call `start()` there and merge its completed report into the parent
report. Tracks must be unique; forks share the same monotonic clock. The receiving
report's event/detail bounds also apply during merge. Last-value metrics cannot
be merged across workers because their ordering is ambiguous. This is collection
infrastructure, not an implementation of parallel compilation itself.

## Wasm CLI

`--print-stats` now covers frontend, MIR optimization, backend, emission and JIT
linking. `--trace-file compile.json` exports the same session with real timing.
Add `--trace-details` for bounded optimization remarks and LIR snapshots. Detailed
IR formatting is diagnostic work and should be disabled for timing comparisons.

For example:

```sh
cargo run --release -p veloc-wasm --bin veloc-wasm -- \
  crates/veloc-wasm/tests/wasm/coremark.wasm --strategy jit \
  --print-stats --trace-file compile.json
```

Execution of the compiled Wasm is outside this compilation session. rustc/Cargo
build time is also outside it; compiler-build profiling remains a separate tool.
