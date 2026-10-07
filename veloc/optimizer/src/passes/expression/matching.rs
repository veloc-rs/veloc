//! Equality queries over a stable graph. Captured matches construct checked
//! replacement plans after enumeration finishes.
use super::storage::{Expressions, Value};
use crate::evaluate::matches_constant;
use hashbrown::{HashMap, HashSet};
use smallvec::SmallVec;
use std::collections::BTreeMap;
use veloc_bytecode::{Reader, equivalence::Instruction as Op};
use veloc_mir::{Opcode, Type};

use super::{
    Limit,
    graph::{Graph, Root},
};

struct Group {
    slots: usize,
    cursors: usize,
    entry: usize,
    work: usize,
}

pub(super) fn supports(opcode: Opcode) -> bool {
    group(opcode).is_some()
}

/// One input to a generated query. Its entry contains only branches compatible
/// with this binding; every reachable Capture can record a successful rule.
pub(super) struct Trigger {
    pub root: Opcode,
    entry: usize, // Query bytecode shared by all rules using this input.
    work: usize,  // Static relation scans in this specialized query.
    slot: usize,
    scan: Option<usize>, // Generated plan identity, never a bytecode address.
    pub path: PathId,
    types: usize,
    requires_constant: bool,
    constants: &'static [(usize, bool)],
}

impl Trigger {
    /// Filter each input using its own pattern domain before walking parents.
    pub(super) fn accepts(&self, body: &Expressions, class: Root) -> bool {
        if !(PROGRAM.types[self.types])(body.value_type(class.value())) {
            return false;
        }
        if !self.requires_constant && self.constants.is_empty() {
            return true;
        }
        let Some(value) = body.as_scalar_const(class.value()) else {
            return false;
        };
        self.constants.iter().all(|&(constant, equal)| {
            matches_constant(Some(value), PROGRAM.constants[constant], equal)
        })
    }
}

pub(super) struct Edge {
    pub opcode: Opcode,
    pub columns: &'static [usize],
}

/// Interned reverse path shared by generated triggers.
#[derive(Clone, Copy)]
pub(super) struct PathId(usize);

impl PathId {
    pub(super) fn index(self) -> usize {
        self.0
    }
}

/// Index into the generated trigger table, distinct from rule IDs, slots and
/// bytecode offsets. Only this module and its generated tables construct IDs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub(super) struct TriggerId(usize);

pub(super) fn trigger(id: TriggerId) -> &'static Trigger {
    &TRIGGERS[id.0]
}

/// Added expressions retain concrete node identity; class changes only carry
/// equivalence identity. Both are grouped by canonical class during scheduling.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Seed {
    Added(Value),
    Class(Root),
}

impl Seed {
    pub(super) fn root(self, graph: &Graph) -> Root {
        match self {
            Self::Added(value) => graph.find(value),
            Self::Class(root) => graph.canonicalize(root),
        }
    }
}

/// One root/opcode query, with the union of all affected input positions.
pub(super) struct Query {
    pub root: Root,
    pub opcode: Opcode,
    /// Trigger ID -> seeds sorted by (class, node). Added-node seeds keep
    /// their concrete identity; class-change seeds are class representatives.
    pub inputs: BTreeMap<TriggerId, Vec<Seed>>,
}

struct Program {
    code: &'static [u8],
    opcodes: &'static [Opcode],
    constants: &'static [u64],
    bindings: &'static [Bindings],
    types: &'static [fn(Type) -> bool],
    properties: &'static [fn(crate::evaluate::Properties, crate::evaluate::Properties) -> bool],
    rules: &'static [Rule],
}

struct Rule {
    plan: usize,
    name: &'static str,
}
include!(concat!(env!("OUT_DIR"), "/equal.rs"));

/// The rule compiler resolves binding and backtracking. The executor reuses
/// registers and retains replacement, guard and type inputs of complete matches.
pub(super) struct Machine {
    slots: Vec<Root>,      // Class bindings used only while querying a stable graph.
    values: Vec<Value>,    // Search values used to construct candidate replacements.
    witnesses: Vec<Value>, // Concrete nodes supplying immutable attributes.
    nodes: Vec<Value>,
    cursors: Vec<Option<Cursor>>,
    relations: Vec<Relation>,
    index: HashMap<Source, usize>,
    matches: Matches,
}

/// Alternative column-to-slot mappings, e.g. `&[&[x, y], &[y, x]]`.
/// Every alternative permutes the same slots and has the operation's arity.
type Bindings = &'static [&'static [usize]];

/// Position within cached rows and their alternative variable bindings.
struct Cursor {
    relation: usize,
    source_slot: usize,
    row: usize,
    binding: usize,
    bindings: Bindings,
    input: Option<TriggerId>,
}

/// Relation data can be shared independently of bindings and input constraints.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Source {
    Empty,
    Relation {
        class: Root,
        opcode: Opcode,
    },
    // A class-filtered slice of one input's seeds. Indices are valid only
    // within the current query; relation caches are reset between queries.
    Candidates {
        input: TriggerId,
        start: usize,
        end: usize,
    },
}

/// Lazily populated rows shared by scans of the same class/opcode pair.
struct Relation {
    cursor: RuleCursor,
    rows: Vec<Row>,
    seen: HashSet<(SmallVec<[Root; 3]>, crate::evaluate::Properties)>,
    exhausted: bool,
}

struct Row {
    node: Value,
    args: SmallVec<[Root; 3]>,
}

/// All queries in a round share this set. Complete matches own their captures;
/// application order is unspecified and no graph update occurs during capture.
#[derive(Default)]
struct Matches {
    entries: HashSet<Match>,
    remaining: usize,
}

#[derive(PartialEq, Eq, Hash)]
struct Match {
    root: Root,
    rule: usize,
    captures: SmallVec<[Root; 3]>,
    nodes: SmallVec<[Value; 2]>,
}

impl Matches {
    fn begin(&mut self, limit: usize) {
        self.entries.clear();
        self.remaining = limit;
    }

    /// Save complete, unique matches up to the per-round storage limit.
    /// The match that fills the buffer is still applied before search stops.
    fn capture(
        &mut self,
        root: Root,
        rule: usize,
        slots: &[Root],
        values: veloc_bytecode::Words<'_>,
        nodes: &[Value],
    ) -> Result<(), Limit> {
        if self.remaining == 0 {
            return Err(Limit::Matches);
        }
        let captures: SmallVec<[Root; 3]> = values.iter().map(|slot| slots[slot]).collect();
        if !self.entries.insert(Match {
            root,
            rule,
            captures,
            nodes: nodes.into(),
        }) {
            return Ok(());
        }
        self.remaining -= 1;
        if self.remaining == 0 {
            Err(Limit::Matches)
        } else {
            Ok(())
        }
    }
}

impl Machine {
    pub(super) fn new() -> Self {
        Self {
            slots: Vec::new(),
            values: Vec::new(),
            witnesses: Vec::new(),
            nodes: Vec::new(),
            cursors: Vec::new(),
            relations: Vec::new(),
            index: HashMap::new(),
            matches: Matches::default(),
        }
    }

    /// Enumerate a whole round against one immutable graph. Relation buffers
    /// are shared across input positions; captures are deduplicated across the
    /// whole round. No query cursor survives a graph update.
    pub(super) fn search(
        &mut self,
        graph: &Graph,
        body: &Expressions,
        queries: &[Query],
        matches: usize,
        fuel: &mut usize,
    ) -> Result<(), Limit> {
        let scope = graph.profile.scope("egraph.search", 0);
        self.index.clear();
        self.matches.begin(matches);
        let result = queries.iter().try_for_each(|query| {
            let root = query.root;
            if body.as_const(root.value()).is_some() {
                return Ok(());
            }
            self.index.clear();
            // Multiple changed positions often activate overlapping queries.
            // Compare their static scan work with one complete enumeration;
            // this choice affects search work, never the valid equalities.
            let group = group(query.opcode).expect("generated query group");
            let work: usize = query.inputs.keys().map(|&id| trigger(id).work).sum();
            if work >= group.work {
                graph
                    .profile
                    .count("queries.coalesced", query.inputs.len() as u64);
                return self.query(graph, body, query, None, fuel);
            }
            // Inputs are alternatives (OR), never cumulative constraints.
            // Distinct entry plans retain their own order while sharing rows.
            for &input in query.inputs.keys() {
                self.query(graph, body, query, Some(input), fuel)?;
            }
            Ok(())
        });
        // No live scan or cache entry is carried into the mutation phase.
        // Keep backing allocations for the next round.
        self.cursors.clear();
        self.index.clear();
        graph
            .profile
            .count("matches", self.matches.entries.len() as u64);
        if graph.profile.enabled() {
            let mut rules = vec![0u64; PROGRAM.rules.len()];
            for matched in &self.matches.entries {
                rules[matched.rule] += 1;
            }
            for (rule, count) in PROGRAM.rules.iter().zip(rules) {
                if count != 0 {
                    graph.profile.count(rule.name, count);
                }
            }
        }
        scope.success();
        result
    }

    fn query(
        &mut self,
        graph: &Graph,
        body: &Expressions,
        query: &Query,
        input: Option<TriggerId>,
        fuel: &mut usize,
    ) -> Result<(), Limit> {
        *fuel = fuel.checked_sub(1).ok_or(Limit::MatchWork)?;
        let entry = input.map(trigger);
        let root = query.root;
        let group = group(query.opcode).expect("generated query group");
        self.slots.resize(group.slots, root);
        self.slots[0] = root;
        self.witnesses.resize(group.slots, root.value());
        self.cursors.clear();
        self.cursors.resize_with(group.cursors, || None);
        let mut reader = Reader {
            bytes: PROGRAM.code,
            pc: entry.map_or(group.entry, |entry| entry.entry),
        };
        loop {
            match Op::read(&mut reader) {
                Op::CheckTypeIn {
                    value,
                    types,
                    otherwise,
                } => {
                    let ty = body.value_type(self.slots[value].value());
                    if !(PROGRAM.types[types])(ty) {
                        reader.pc = otherwise;
                    }
                }
                Op::CheckSameType {
                    lhs,
                    rhs,
                    otherwise,
                } => {
                    let dfg = body;
                    if dfg.value_type(self.slots[lhs].value())
                        != dfg.value_type(self.slots[rhs].value())
                    {
                        reader.pc = otherwise;
                    }
                }
                Op::OpenScan {
                    scan,
                    cursor,
                    source,
                    opcode,
                    bindings,
                } => {
                    let source_slot = source;
                    let class = self.slots[source];
                    let source =
                        if let Some(input) = input.filter(|_| entry.unwrap().scan == Some(scan)) {
                            let seeds = &query.inputs[&input];
                            // Seeds are sorted by (class, node). Resolve the class
                            // slice once, then enumerate only affected nodes.
                            let start = seeds.partition_point(|&seed| seed.root(graph) < class);
                            let end = seeds.partition_point(|&seed| seed.root(graph) <= class);
                            if start != end {
                                Source::Candidates { input, start, end }
                            } else {
                                Source::Empty
                            }
                        } else {
                            Source::Relation {
                                class,
                                opcode: PROGRAM.opcodes[opcode],
                            }
                        };
                    self.open(
                        cursor,
                        source_slot,
                        source,
                        PROGRAM.bindings[bindings],
                        input,
                    );
                }
                Op::ScanNext { cursor, exhausted } => {
                    // Scans charge each attempted binding. A budget stop returns
                    // to the host, which still applies the saved matches.
                    if *fuel == 0 {
                        return Err(Limit::MatchWork);
                    }
                    if !self.scan_next(graph, body, query, cursor, fuel) {
                        if *fuel == 0 {
                            return Err(Limit::MatchWork);
                        }
                        reader.pc = exhausted;
                    }
                }
                Op::CheckEqual {
                    lhs,
                    rhs,
                    otherwise,
                } => {
                    if self.slots[lhs] != self.slots[rhs] {
                        reader.pc = otherwise;
                    }
                }
                Op::CheckIsConstant { value, otherwise } => {
                    if body.as_scalar_const(self.slots[value].value()).is_none() {
                        reader.pc = otherwise;
                    }
                }
                Op::CheckConstantEq {
                    value,
                    constant,
                    otherwise,
                } => {
                    if !matches_constant(
                        body.as_scalar_const(self.slots[value].value()),
                        PROGRAM.constants[constant],
                        true,
                    ) {
                        reader.pc = otherwise;
                    }
                }
                Op::CheckConstantNe {
                    value,
                    constant,
                    otherwise,
                } => {
                    if !matches_constant(
                        body.as_scalar_const(self.slots[value].value()),
                        PROGRAM.constants[constant],
                        false,
                    ) {
                        reader.pc = otherwise;
                    }
                }
                Op::CheckProperties {
                    lhs,
                    rhs,
                    predicate,
                    otherwise,
                } => {
                    let properties = |slot| {
                        body.properties(
                            body.value_inst(self.witnesses[slot])
                                .expect("scanned instruction"),
                        )
                    };
                    if !(PROGRAM.properties[predicate])(properties(lhs), properties(rhs)) {
                        reader.pc = otherwise;
                    }
                }
                // Retain the values needed by the shared plan and its guards.
                Op::Capture {
                    rule,
                    values,
                    nodes,
                } => {
                    // Query specialization establishes the input's scope. There
                    // is no current-rule filter: all matching branches contribute.
                    self.values.clear();
                    self.values
                        .extend(values.iter().map(|slot| self.slots[slot].value()));
                    self.nodes.clear();
                    self.nodes
                        .extend(nodes.iter().map(|slot| self.witnesses[slot]));
                    let cx = crate::rewrite::Context {
                        body,
                        layout: graph.layout,
                    };
                    if crate::evaluate::accepts(
                        PROGRAM.rules[rule].plan,
                        &cx,
                        &self.values,
                        &self.nodes,
                    ) {
                        self.matches
                            .capture(root, rule, &self.slots, values, &self.nodes)?;
                    }
                }
                Op::Jump { target } => reader.pc = target,
                Op::Return {} => return Ok(()),
            }
        }
    }

    /// Apply the saved matches after all queries have released their cursors.
    /// A full node budget can reject construction while still allowing later
    /// matches that only merge existing values or establish constant facts.
    pub(super) fn apply(&mut self, graph: &mut Graph, ir: &mut Expressions) -> Result<(), Limit> {
        let scope = graph.profile.scope("egraph.apply", 0);
        let mut status = Ok(());
        // Move the set out while rewriting so its keys remain immutable. Drain
        // every saved match, including after a node-budget stop, then reuse its
        // allocation next round. No discovery order is required.
        let mut pending = core::mem::take(&mut self.matches.entries);
        for matched in pending.drain() {
            let root = graph.canonicalize(matched.root);
            if ir.as_const(root.value()).is_some() {
                continue;
            }
            let rule = &PROGRAM.rules[matched.rule];
            self.values.clear();
            self.values.extend(
                matched
                    .captures
                    .iter()
                    .map(|&v| graph.canonicalize(v).value()),
            );
            let cx = crate::rewrite::Context {
                body: ir,
                layout: graph.layout,
            };
            graph.profile.count("egraph.replacement_plans", 1);
            let Some(plan) = crate::evaluate::plan(rule.plan, &cx, &self.values, &matched.nodes)
            else {
                continue;
            };
            let replacement = plan.materialize(|step, args| match *step {
                crate::rewrite::Step::Constant(c) => Ok(graph.literal(ir, c)),
                crate::rewrite::Step::Build {
                    opcode,
                    ty,
                    properties,
                    ..
                } => graph.build(ir, opcode, args, ty, properties),
            });
            match replacement {
                Ok(value) => {
                    graph.union(ir, root.value(), value);
                    log::trace!("egraph rule {}", rule.name);
                }
                Err(limit) => status = Err(limit),
            }
        }
        self.matches.entries = pending;
        scope.success();
        status
    }

    /// Open a source with static bindings and an input checked when bound.
    fn open(
        &mut self,
        cursor: usize,
        source_slot: usize,
        source: Source,
        bindings: Bindings,
        input: Option<TriggerId>,
    ) {
        let id = self.index.len();
        let relation = *self.index.entry(source).or_insert_with(|| {
            let cursor = RuleCursor { source, row: 0 };
            if let Some(relation) = self.relations.get_mut(id) {
                relation.cursor = cursor;
                relation.rows.clear();
                relation.seen.clear();
                relation.exhausted = false;
            } else {
                self.relations.push(Relation {
                    cursor,
                    rows: Vec::new(),
                    seen: HashSet::new(),
                    exhausted: false,
                });
            }
            id
        });
        self.cursors[cursor] = Some(Cursor {
            relation,
            source_slot,
            row: 0,
            binding: 0,
            bindings,
            input,
        });
    }

    /// Yield the next valid binding without exposing rejected candidates to
    /// the bytecode loop. Each attempted row/binding pair consumes search fuel.
    fn scan_next(
        &mut self,
        graph: &Graph,
        body: &Expressions,
        query: &Query,
        cursor: usize,
        fuel: &mut usize,
    ) -> bool {
        let cursor = self.cursors[cursor].as_mut().expect("opened query cursor");
        let relation = &mut self.relations[cursor.relation];
        while *fuel > 0 {
            *fuel -= 1;
            // Fetch at most one new row; other scans reuse the same cache.
            if cursor.row == relation.rows.len() && !relation.exhausted {
                match relation.cursor.next(graph, body, query) {
                    Some(row) => {
                        let inst = body.value_inst(row.node).expect("relation result");
                        let properties = body.properties(inst);
                        // Concrete SSA definitions must remain available to
                        // extraction, but equal query rows carry the same
                        // bindings and attributes. Enumerate them only once.
                        if !relation.seen.insert((row.args.clone(), properties)) {
                            continue;
                        }
                        relation.rows.push(row);
                    }
                    None => relation.exhausted = true,
                }
            }
            let Some(row) = relation.rows.get(cursor.row) else {
                return false;
            };
            let binding = cursor.bindings[cursor.binding];
            debug_assert_eq!(binding.len(), row.args.len(), "pattern operand count");
            cursor.binding += 1;
            // All alternatives permute the same slots. Equal binary operands
            // make both orientations identical, so visit this row only once.
            if cursor.binding == cursor.bindings.len()
                || matches!(row.args.as_slice(), [a, b] if a == b)
            {
                cursor.binding = 0;
                cursor.row += 1;
            }

            if cursor.input.is_some_and(|input| {
                binding.iter().zip(&row.args).any(|(&slot, &arg)| {
                    slot == trigger(input).slot
                        && query.inputs[&input]
                            .binary_search_by_key(&arg, |&seed| seed.root(graph))
                            .is_err()
                })
            }) {
                continue;
            }
            self.witnesses[cursor.source_slot] = row.node;
            // Do not overwrite any slots until the entire binding is accepted.
            for (&slot, &value) in binding.iter().zip(&row.args) {
                self.slots[slot] = value;
            }
            return true;
        }
        false
    }
}

impl Group {
    /// Offsets are actual byte positions in the shared program.
    #[allow(dead_code)]
    fn disassemble(&self, out: &mut dyn std::fmt::Write) -> std::fmt::Result {
        writeln!(out, "entry @{}", self.entry)?;
        writeln!(
            out,
            "opcodes {:?}, constants {:?}",
            PROGRAM.opcodes, PROGRAM.constants
        )?;
        let mut reader = Reader {
            bytes: PROGRAM.code,
            pc: self.entry,
        };
        loop {
            let pc = reader.pc;
            let op = Op::read(&mut reader);
            writeln!(out, "{pc:04x}: {op:?}")?;
            if matches!(op, Op::Return {}) {
                return Ok(());
            }
        }
    }
}

/// Valid only while querying a stable graph, before applying any updates.
struct RuleCursor {
    source: Source,
    row: usize,
}

impl RuleCursor {
    fn next(&mut self, graph: &Graph, body: &Expressions, query: &Query) -> Option<Row> {
        // Borrow only for this read; the reusable cursor owns no graph borrow.
        // Empty, changed-node and relation sources share the exhaustion rule.
        let value = match &self.source {
            Source::Empty => return None,
            Source::Candidates { input, start, end } => {
                let Seed::Added(value) = query.inputs[input][*start..*end].get(self.row)? else {
                    unreachable!("only added-node triggers constrain a concrete scan");
                };
                *value
            }
            Source::Relation { class, opcode } => {
                *graph.alternatives(*class, *opcode).get(self.row)?
            }
        };
        self.row += 1;
        let inst = body.value_inst(value).expect("relation result");
        // Rebuilding filters folded nodes from both relations and added seeds.
        // The compiler accepts only single-result pattern operations.
        Some(Row {
            node: value,
            args: graph.canonical_args(body, inst),
        })
    }
}
