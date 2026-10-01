//! Optional, worker-local compiler observation. No IR or target dependencies.
//!
//! Clones share one synchronous scope stack; do not interleave independent
//! tasks on it. Parallel workers use `fork` and merge completed reports.
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Write;
use std::rc::Rc;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    #[default]
    Off,
    Summary,
    Trace,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub mode: Mode,
    /// Per-worker trace bound; merged reports also retain at most this many events.
    pub max_events: usize,
    /// Remarks and textual artifacts are opt-in even in trace mode.
    pub details: bool,
    pub max_detail_bytes: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mode: Mode::Off,
            max_events: 100_000,
            details: false,
            max_detail_bytes: 1 << 20,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Aggregate {
    Sum,
    Max,
    Last,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Unit {
    Count,
    Bytes,
}

/// Definitions belong to the component that owns the measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Metric {
    pub name: &'static str,
    pub unit: Unit,
    pub aggregate: Aggregate,
}

impl Metric {
    pub const fn count(name: &'static str) -> Self {
        Self {
            name,
            unit: Unit::Count,
            aggregate: Aggregate::Sum,
        }
    }
    pub const fn bytes(name: &'static str) -> Self {
        Self {
            name,
            unit: Unit::Bytes,
            aggregate: Aggregate::Sum,
        }
    }
    pub const fn max(self) -> Self {
        Self {
            aggregate: Aggregate::Max,
            ..self
        }
    }
    pub const fn last(self) -> Self {
        Self {
            aggregate: Aggregate::Last,
            ..self
        }
    }
}

/// Recorder-local handle. Do not reuse in a forked worker or another session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetricId(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Success,
    Failed,
    Interrupted,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Stage {
    parent: Option<usize>,
    name: &'static str,
    position: u32,
}

#[derive(Clone, Debug, Default)]
struct Totals {
    calls: u64,
    failed: u64,
    interrupted: u64,
    elapsed: Duration,
    own: Duration,
    metrics: HashMap<usize, u64>,
}

#[derive(Clone, Debug)]
struct Event {
    stage: usize,
    start: Duration,
    duration: Duration,
    track: u32,
    outcome: Outcome,
    entity: Option<String>,
    detail: Option<(&'static str, String)>,
}

struct Active {
    stage: usize,
    start: Instant,
    children: Duration,
    entity: Option<String>,
}

struct State {
    config: Config,
    origin: Instant,
    track: u32,
    stages: Vec<Stage>,
    stage_ids: HashMap<Stage, usize>,
    metrics: Vec<Metric>,
    metric_ids: HashMap<Metric, usize>,
    totals: Vec<Totals>,
    active: Vec<Active>,
    events: Vec<Event>,
    dropped: u64,
    detail_bytes: usize,
    metadata: HashMap<String, String>,
    last: Duration,
}

/// Explicit session handle. The disabled form allocates nothing and reads no clock.
#[derive(Clone, Default)]
pub struct Profile(Option<Rc<RefCell<State>>>);

impl std::fmt::Debug for Profile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Profile")
            .field("enabled", &self.enabled())
            .finish()
    }
}

impl Profile {
    pub fn new(config: Config) -> Self {
        if config.mode == Mode::Off {
            return Self::default();
        }
        Self::worker(config, Instant::now(), 0)
    }

    fn worker(config: Config, origin: Instant, track: u32) -> Self {
        Self(Some(Rc::new(RefCell::new(State {
            config,
            origin,
            track,
            stages: Vec::new(),
            stage_ids: HashMap::new(),
            metrics: Vec::new(),
            metric_ids: HashMap::new(),
            totals: Vec::new(),
            active: Vec::new(),
            events: Vec::new(),
            dropped: 0,
            detail_bytes: 0,
            metadata: HashMap::new(),
            last: Duration::ZERO,
        }))))
    }

    /// Records ordinary errors as failures, and unwinding as interruption.
    pub fn measure<T, E>(
        &self,
        name: &'static str,
        position: u32,
        run: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        let scope = self.scope(name, position);
        let result = run();
        scope.result(&result);
        result
    }

    pub fn enabled(&self) -> bool {
        self.0.is_some()
    }

    /// A Send worker configuration; instantiate on the destination thread.
    /// Track IDs must be unique within the merged report.
    pub fn fork(&self, track: u32) -> Option<Worker> {
        self.0.as_ref().map(|s| {
            let s = s.borrow();
            Worker {
                config: s.config.clone(),
                origin: s.origin,
                track,
            }
        })
    }

    pub fn metadata(&self, key: &str, value: impl FnOnce() -> String) {
        if let Some(s) = &self.0 {
            s.borrow_mut().metadata.insert(key.into(), value());
        }
    }

    pub fn scope(&self, name: &'static str, position: u32) -> Scope {
        self.entity_scope(name, position, || String::new())
    }

    /// Entity formatting is lazy and only performed for detailed traces.
    pub fn entity_scope(
        &self,
        name: &'static str,
        position: u32,
        entity: impl FnOnce() -> String,
    ) -> Scope {
        let Some(state) = &self.0 else {
            return Scope::default();
        };
        let mut s = state.borrow_mut();
        let key = Stage {
            parent: s.active.last().map(|a| a.stage),
            name,
            position,
        };
        let stage = if let Some(&id) = s.stage_ids.get(&key) {
            id
        } else {
            let id = s.stages.len();
            s.stages.push(key.clone());
            s.stage_ids.insert(key, id);
            s.totals.push(Totals::default());
            id
        };
        let entity = if s.config.mode == Mode::Trace && s.events.len() < s.config.max_events {
            let value = entity();
            if value.is_empty() {
                s.active.last().and_then(|a| a.entity.clone())
            } else {
                Some(value)
            }
        } else {
            None
        };
        let depth = s.active.len();
        s.active.push(Active {
            stage,
            start: Instant::now(),
            children: Duration::ZERO,
            entity,
        });
        Scope {
            profile: self.clone(),
            depth,
            outcome: Outcome::Interrupted,
        }
    }

    /// Intern once for frequently updated metrics, then use `record`.
    pub fn metric(&self, metric: Metric) -> Option<MetricId> {
        let state = self.0.as_ref()?;
        let mut s = state.borrow_mut();
        if let Some(&id) = s.metric_ids.get(&metric) {
            return Some(MetricId(id));
        }
        let id = s.metrics.len();
        s.metrics.push(metric);
        s.metric_ids.insert(metric, id);
        Some(MetricId(id))
    }

    pub fn record(&self, id: MetricId, value: u64) {
        let Some(state) = &self.0 else {
            return;
        };
        let mut s = state.borrow_mut();
        let stage = if let Some(active) = s.active.last() {
            active.stage
        } else {
            // Metrics published outside a timed phase belong to the session.
            let key = Stage {
                parent: None,
                name: "session",
                position: 0,
            };
            if let Some(&id) = s.stage_ids.get(&key) {
                id
            } else {
                let id = s.stages.len();
                s.stages.push(key.clone());
                s.stage_ids.insert(key, id);
                s.totals.push(Totals::default());
                id
            }
        };
        let aggregate = s.metrics[id.0].aggregate;
        combine(
            s.totals[stage].metrics.entry(id.0).or_default(),
            value,
            aggregate,
        );
    }

    /// Convenience for metrics published once at phase boundaries.
    pub fn record_lazy(&self, metric: Metric, value: impl FnOnce() -> u64) {
        if let Some(id) = self.metric(metric) {
            self.record(id, value());
        }
    }

    pub fn count(&self, name: &'static str, value: u64) {
        self.record_lazy(Metric::count(name), || value);
    }

    pub fn remark(&self, text: impl FnOnce() -> String) {
        self.detail("remark", text);
    }
    pub fn artifact(&self, text: impl FnOnce() -> String) {
        self.detail("artifact", text);
    }

    fn detail(&self, kind: &'static str, text: impl FnOnce() -> String) {
        let Some(state) = &self.0 else {
            return;
        };
        let mut s = state.borrow_mut();
        let Some(active) = s.active.last() else {
            return;
        };
        let stage = active.stage;
        let entity = active.entity.clone();
        record_detail(&mut s, stage, entity, kind, text);
    }

    /// Snapshot after scopes finish. Reporting is not on the compilation hot path.
    pub fn report(&self) -> Report {
        let Some(state) = &self.0 else {
            return Report::default();
        };
        let s = state.borrow();
        assert!(
            s.active.is_empty(),
            "finish profile scopes before reporting"
        );
        Report {
            max_events: s.config.max_events,
            max_detail_bytes: s.config.max_detail_bytes,
            detail_bytes: s.detail_bytes,
            origin: Some(s.origin),
            stages: s.stages.clone(),
            totals: s.totals.clone(),
            metrics: s.metrics.clone(),
            events: s.events.clone(),
            dropped: s.dropped,
            elapsed: s.last,
            metadata: s.metadata.clone(),
        }
    }
}

/// Associates diagnostics with their original scope after its timer has ended.
/// Retaining this handle does not keep the scope active or extend its duration.
#[derive(Default)]
pub struct Observation {
    profile: Profile,
    identity: Option<(usize, Option<String>)>,
}
impl Observation {
    pub fn artifact(&self, text: impl FnOnce() -> String) {
        self.detail("artifact", text);
    }
    pub fn remark(&self, text: impl FnOnce() -> String) {
        self.detail("remark", text);
    }
    fn detail(&self, kind: &'static str, text: impl FnOnce() -> String) {
        if let (Some(state), Some((stage, entity))) = (&self.profile.0, &self.identity) {
            record_detail(&mut state.borrow_mut(), *stage, entity.clone(), kind, text);
        }
    }
}
fn record_detail(
    s: &mut State,
    stage: usize,
    entity: Option<String>,
    kind: &'static str,
    text: impl FnOnce() -> String,
) {
    if s.config.mode != Mode::Trace || !s.config.details {
        return;
    }
    if s.events.len() >= s.config.max_events || s.detail_bytes >= s.config.max_detail_bytes {
        s.dropped += 1;
        return;
    }
    let mut text = text();
    let mut available = s.config.max_detail_bytes - s.detail_bytes;
    if text.len() > available {
        while !text.is_char_boundary(available) {
            available -= 1;
        }
        text.truncate(available);
        s.dropped += 1;
    }
    s.detail_bytes += text.len();
    let event = Event {
        stage,
        start: s.origin.elapsed(),
        duration: Duration::ZERO,
        track: s.track,
        outcome: Outcome::Success,
        entity,
        detail: Some((kind, text)),
    };
    s.events.push(event);
}

pub struct Worker {
    config: Config,
    origin: Instant,
    track: u32,
}
impl Worker {
    pub fn start(self) -> Profile {
        Profile::worker(self.config, self.origin, self.track)
    }
}

#[must_use = "keep the scope alive until the observed operation completes"]
#[derive(Default)]
pub struct Scope {
    profile: Profile,
    depth: usize,
    outcome: Outcome,
}

impl Default for Outcome {
    fn default() -> Self {
        Self::Interrupted
    }
}

impl Scope {
    pub fn observation(&self) -> Observation {
        let Some(state) = &self.profile.0 else {
            return Observation::default();
        };
        let s = state.borrow();
        if s.config.mode != Mode::Trace || !s.config.details {
            return Observation::default();
        }
        let active = &s.active[self.depth];
        Observation {
            profile: self.profile.clone(),
            identity: Some((active.stage, active.entity.clone())),
        }
    }
    pub fn success(mut self) {
        self.outcome = Outcome::Success;
    }
    pub fn finish(mut self, outcome: Outcome) {
        self.outcome = outcome;
    }
    pub fn result<T, E>(mut self, result: &Result<T, E>) {
        self.outcome = if result.is_ok() {
            Outcome::Success
        } else {
            Outcome::Failed
        };
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        let Some(state) = &self.profile.0 else {
            return;
        };
        let end = Instant::now();
        let mut s = state.borrow_mut();
        assert_eq!(
            s.active.len(),
            self.depth + 1,
            "profile scopes must be nested"
        );
        let active = s.active.pop().unwrap();
        let duration = end.duration_since(active.start);
        let total = &mut s.totals[active.stage];
        total.calls += 1;
        total.elapsed += duration;
        total.own += duration.saturating_sub(active.children);
        total.failed += u64::from(self.outcome == Outcome::Failed);
        total.interrupted += u64::from(self.outcome == Outcome::Interrupted);
        if let Some(parent) = s.active.last_mut() {
            parent.children += duration;
        }
        s.last = end.duration_since(s.origin);
        if s.config.mode == Mode::Trace {
            if s.events.len() < s.config.max_events {
                let event = Event {
                    stage: active.stage,
                    start: active.start.duration_since(s.origin),
                    duration,
                    track: s.track,
                    outcome: self.outcome,
                    entity: active.entity,
                    detail: None,
                };
                s.events.push(event);
            } else {
                s.dropped += 1;
            }
        }
    }
}

fn combine(old: &mut u64, value: u64, aggregate: Aggregate) {
    *old = match aggregate {
        Aggregate::Sum => old.saturating_add(value),
        Aggregate::Max => (*old).max(value),
        Aggregate::Last => value,
    };
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    max_events: usize,
    max_detail_bytes: usize,
    detail_bytes: usize,
    origin: Option<Instant>,
    stages: Vec<Stage>,
    totals: Vec<Totals>,
    metrics: Vec<Metric>,
    events: Vec<Event>,
    pub dropped: u64,
    pub elapsed: Duration,
    pub metadata: HashMap<String, String>,
}

/// Owned, backend-independent data for dashboards and regression tools.
/// Inclusive durations overlap; do not sum them to obtain wall time.
#[derive(Debug)]
pub struct Summary {
    pub path: String,
    pub calls: u64,
    pub failed: u64,
    pub interrupted: u64,
    pub inclusive: Duration,
    pub exclusive: Duration,
    pub metrics: Vec<(Metric, u64)>,
}

impl Report {
    pub fn summaries(&self) -> impl Iterator<Item = Summary> + '_ {
        self.totals.iter().enumerate().map(|(id, t)| {
            let mut metrics: Vec<_> = t
                .metrics
                .iter()
                .map(|(&id, &value)| (self.metrics[id], value))
                .collect();
            metrics.sort_by_key(|(m, _)| m.name);
            Summary {
                path: self.path(id),
                calls: t.calls,
                failed: t.failed,
                interrupted: t.interrupted,
                inclusive: t.elapsed,
                exclusive: t.own,
                metrics,
            }
        })
    }
    /// Merge reports from workers forked from one session, with unique tracks.
    /// Last-value gauges have no cross-worker ordering and cannot be merged.
    pub fn merge(&mut self, other: Self) {
        if other.origin.is_none() {
            return;
        }
        if self.origin.is_none() {
            *self = other;
            return;
        }
        assert_eq!(
            self.origin, other.origin,
            "reports must share a session clock"
        );
        assert!(
            self.metrics
                .iter()
                .chain(&other.metrics)
                .all(|m| m.aggregate != Aggregate::Last),
            "last-value gauges cannot be merged across workers"
        );
        let mut metric_ids = Vec::new();
        for metric in other.metrics {
            let id = self
                .metrics
                .iter()
                .position(|m| *m == metric)
                .unwrap_or_else(|| {
                    self.metrics.push(metric);
                    self.metrics.len() - 1
                });
            metric_ids.push(id);
        }
        let mut stages = Vec::new();
        for (mut stage, total) in other.stages.into_iter().zip(other.totals) {
            stage.parent = stage.parent.map(|p| stages[p]);
            let id = self
                .stages
                .iter()
                .position(|s| *s == stage)
                .unwrap_or_else(|| {
                    self.stages.push(stage);
                    self.totals.push(Totals::default());
                    self.stages.len() - 1
                });
            stages.push(id);
            let into = &mut self.totals[id];
            into.calls += total.calls;
            into.failed += total.failed;
            into.interrupted += total.interrupted;
            into.elapsed += total.elapsed;
            into.own += total.own;
            for (metric, value) in total.metrics {
                let id = metric_ids[metric];
                combine(
                    into.metrics.entry(id).or_default(),
                    value,
                    self.metrics[id].aggregate,
                );
            }
        }
        for mut event in other.events {
            let bytes = event.detail.as_ref().map_or(0, |(_, text)| text.len());
            if self.events.len() >= self.max_events
                || bytes > self.max_detail_bytes.saturating_sub(self.detail_bytes)
            {
                self.dropped += 1;
                continue;
            }
            self.detail_bytes += bytes;
            event.stage = stages[event.stage];
            self.events.push(event);
        }
        self.dropped += other.dropped;
        self.elapsed = self.elapsed.max(other.elapsed);
        self.metadata.extend(other.metadata);
    }

    fn path(&self, mut stage: usize) -> String {
        let mut parts = Vec::new();
        loop {
            let s = &self.stages[stage];
            parts.push(format!("{}[{}]", s.name, s.position));
            if let Some(parent) = s.parent {
                stage = parent;
            } else {
                break;
            }
        }
        parts.reverse();
        parts.join("/")
    }

    /// Chrome trace JSON with actual session-relative timestamps and track IDs.
    pub fn chrome_trace(&self) -> String {
        let mut out = String::from("{\"displayTimeUnit\":\"ms\",\"traceEvents\":[");
        for (i, e) in self.events.iter().enumerate() {
            if i != 0 {
                out.push(',');
            }
            let name = quote(self.stages[e.stage].name);
            let args = format!(
                "\"stage\":{},\"entity\":{},\"outcome\":{}",
                quote(&self.path(e.stage)),
                quote(e.entity.as_deref().unwrap_or("")),
                quote(&format!("{:?}", e.outcome))
            );
            if let Some((kind, text)) = &e.detail {
                write!(out, "{{\"name\":{name},\"ph\":\"i\",\"s\":\"t\",\"pid\":1,\"tid\":{},\"ts\":{},\"args\":{{{args},\"kind\":{},\"text\":{}}}}}", e.track, micros(e.start), quote(kind), quote(text)).unwrap();
            } else {
                write!(out, "{{\"name\":{name},\"ph\":\"X\",\"pid\":1,\"tid\":{},\"ts\":{},\"dur\":{},\"args\":{{{args}}}}}", e.track, micros(e.start), micros(e.duration)).unwrap();
            }
        }
        write!(out, "],\"dropped_events\":{},\"metadata\":{{", self.dropped).unwrap();
        let mut metadata: Vec<_> = self.metadata.iter().collect();
        metadata.sort();
        for (i, (key, value)) in metadata.into_iter().enumerate() {
            if i != 0 {
                out.push(',');
            }
            write!(out, "{}:{}", quote(key), quote(value)).unwrap();
        }
        out.push_str("},\"summary\":[");
        for (i, s) in self.summaries().enumerate() {
            if i != 0 {
                out.push(',');
            }
            write!(out, "{{\"stage\":{},\"calls\":{},\"failed\":{},\"interrupted\":{},\"inclusive_us\":{},\"exclusive_us\":{},\"metrics\":[",
                quote(&s.path), s.calls, s.failed, s.interrupted, micros(s.inclusive), micros(s.exclusive)).unwrap();
            for (j, (m, value)) in s.metrics.iter().enumerate() {
                if j != 0 {
                    out.push(',');
                }
                write!(
                    out,
                    "{{\"name\":{},\"value\":{},\"unit\":{},\"aggregate\":{}}}",
                    quote(m.name),
                    value,
                    quote(&format!("{:?}", m.unit)),
                    quote(&format!("{:?}", m.aggregate))
                )
                .unwrap();
            }
            out.push_str("]}");
        }
        out.push_str("]}");
        out
    }
}

fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000_000.0
}
fn quote(text: &str) -> String {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if c <= '\u{1f}' => write!(out, "\\u{:04x}", c as u32).unwrap(),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Compilation wall time: {:.3} ms",
            self.elapsed.as_secs_f64() * 1000.0
        )?;
        writeln!(
            f,
            "Stage | calls | inclusive ms | self ms | failed/interrupted"
        )?;
        for (id, t) in self.totals.iter().enumerate() {
            writeln!(
                f,
                "{} | {} | {:.3} | {:.3} | {}/{}",
                self.path(id),
                t.calls,
                t.elapsed.as_secs_f64() * 1000.0,
                t.own.as_secs_f64() * 1000.0,
                t.failed,
                t.interrupted
            )?;
            let mut metrics: Vec<_> = t.metrics.iter().collect();
            metrics.sort_by_key(|(id, _)| self.metrics[**id].name);
            for (&id, value) in metrics {
                let m = self.metrics[id];
                writeln!(
                    f,
                    "  {}: {} ({:?}, {:?})",
                    m.name, value, m.unit, m.aggregate
                )?;
            }
        }
        if self.dropped != 0 {
            writeln!(f, "Truncated records: {}", self.dropped)?;
        }
        Ok(())
    }
}
