//! Equality-saturation bytecode. All rules share one program and constant pool.
//! Queries run against a stable graph; actions run after enumeration finishes.
use hashbrown::HashSet;
use smallvec::SmallVec;
use std::collections::{BTreeMap, HashMap};
use veloc_bytecode::{Reader, equivalence::Instruction as Op};
use veloc_mir::{FuncBody, Opcode, Type, Value, constant::ScalarConst, function::Expressions};
use veloc_types::TypeInfo;

use super::{
    Limit,
    graph::{Graph, Root},
};

struct Group {
    slots: usize,
    cursors: usize,
    entry: usize,
}

/// One input to a generated query. Its entry contains only branches compatible
/// with this binding; every reachable Capture can record a successful rule.
pub(super) struct Trigger {
    pub root: Opcode,
    entry: usize, // Query bytecode shared by all rules using this input.
    slot: usize,
    scan: Option<usize>, // Generated plan identity, never a bytecode address.
    pub path: PathId,
    types: usize,
    constants: &'static [(usize, bool)],
}

impl Trigger {
    /// Reject impossible inputs before walking their reverse paths. Pattern
    /// typing ensures every bound input shares the root's type and bit width.
    pub(super) fn accepts(&self, body: &FuncBody, class: Root) -> bool {
        if !PROGRAM.types[self.types].contains(&body.dfg().value_type(class.value())) {
            return false;
        }
        if self.constants.is_empty() {
            return true;
        }
        let Some(value) = body.dfg().as_scalar_const(class.value()) else {
            return false;
        };
        let mask = u64::MAX >> (64 - value.ty().element_bits().unwrap());
        self.constants.iter().all(|&(constant, equal)| {
            matches_constant(Some(value), PROGRAM.constants[constant], mask, equal)
        })
    }
}

/// Pattern literals are masked to the matched type. Unknown values establish
/// neither equality nor inequality; scheduling and the VM use the same check.
fn matches_constant(value: Option<ScalarConst>, bits: u64, mask: u64, equal: bool) -> bool {
    value.is_some_and(|value| (value.to_bits() == (bits & mask)) == equal)
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
    types: &'static [&'static [Type]],
    rules: &'static [Rule],
}

struct Rule {
    entry: usize,
    slots: usize,
    name: &'static str,
    captures: &'static [usize],
}
include!(concat!(env!("OUT_DIR"), "/equivalences.rs"));

/// The rule compiler resolves binding and backtracking. The executor reuses
/// registers and retains only the RHS inputs of each complete match.
pub(super) struct Machine {
    slots: Vec<Root>,   // Class bindings used only while querying a stable graph.
    values: Vec<Value>, // MIR values used to construct detached replacements.
    cursors: Vec<Option<Cursor>>,
    relations: Vec<Relation>,
    index: HashMap<Source, usize>,
    matches: Matches,
    args: SmallVec<[Value; 3]>,
}

/// Alternative column-to-slot mappings, e.g. `&[&[x, y], &[y, x]]`.
/// Every alternative permutes the same slots and has the operation's arity.
type Bindings = &'static [&'static [usize]];

/// Position within cached rows and their alternative variable bindings.
struct Cursor {
    relation: usize,
    row: usize,
    binding: usize,
    bindings: Bindings,
    input: TriggerId,
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
    exhausted: bool,
}

type Row = SmallVec<[Root; 3]>;

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
}

impl Matches {
    fn begin(&mut self, limit: usize) {
        self.entries.clear();
        self.remaining = limit;
    }

    /// Save complete, unique matches up to the per-round storage limit.
    /// The match that fills the buffer is still applied before search stops.
    fn capture(&mut self, root: Root, rule: usize, slots: &[Root]) -> Result<(), Limit> {
        if self.remaining == 0 {
            return Err(Limit::Matches);
        }
        let captures: SmallVec<[Root; 3]> = PROGRAM.rules[rule]
            .captures
            .iter()
            .map(|&slot| slots[slot])
            .collect();
        if !self.entries.insert(Match {
            root,
            rule,
            captures,
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
            cursors: Vec::new(),
            relations: Vec::new(),
            index: HashMap::new(),
            matches: Matches::default(),
            args: SmallVec::new(),
        }
    }

    /// Enumerate a whole round against one immutable graph. Relation buffers
    /// are shared across input positions; captures are deduplicated across the
    /// whole round. No query cursor survives a graph update.
    pub(super) fn search(
        &mut self,
        graph: &Graph,
        body: &FuncBody,
        queries: &[Query],
        matches: usize,
        fuel: &mut usize,
    ) -> Result<(), Limit> {
        let scope = graph.profile.scope("egraph.search", 0);
        self.index.clear();
        self.matches.begin(matches);
        let result = queries.iter().try_for_each(|query| {
            let root = query.root;
            if body.dfg().as_const(root.value()).is_some() {
                return Ok(());
            }
            self.index.clear();
            // Inputs are alternatives (OR), never cumulative constraints.
            // Distinct entry plans retain their own order while sharing rows.
            for &input in query.inputs.keys() {
                if *fuel == 0 {
                    return Err(Limit::MatchWork);
                }
                *fuel -= 1;
                self.query(graph, body, query, input, fuel)?;
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
        body: &FuncBody,
        query: &Query,
        input: TriggerId,
        fuel: &mut usize,
    ) -> Result<(), Limit> {
        let entry = trigger(input);
        let seeds = &query.inputs[&input];
        let root = query.root;
        let ty = body.dfg().value_type(root.value());
        let mask = u64::MAX >> (64 - ty.element_bits().unwrap());
        let group = group(query.opcode).expect("generated query group");
        self.slots.resize(group.slots, root);
        self.slots[0] = root;
        self.cursors.clear();
        self.cursors.resize_with(group.cursors, || None);
        let mut reader = Reader {
            bytes: PROGRAM.code,
            pc: entry.entry,
        };
        loop {
            match Op::read(&mut reader) {
                Op::CheckTypeIn {
                    value,
                    types,
                    otherwise,
                } => {
                    let ty = body.dfg().value_type(self.slots[value].value());
                    if !PROGRAM.types[types].contains(&ty) {
                        reader.pc = otherwise;
                    }
                }
                Op::CheckSameType {
                    lhs,
                    rhs,
                    otherwise,
                } => {
                    let dfg = body.dfg();
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
                    let class = self.slots[source];
                    let source = if entry.scan == Some(scan) {
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
                    self.open(cursor, source, PROGRAM.bindings[bindings], input);
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
                Op::CheckConstantEq {
                    value,
                    constant,
                    otherwise,
                } => {
                    if !matches_constant(
                        body.dfg().as_scalar_const(self.slots[value].value()),
                        PROGRAM.constants[constant],
                        mask,
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
                        body.dfg().as_scalar_const(self.slots[value].value()),
                        PROGRAM.constants[constant],
                        mask,
                        false,
                    ) {
                        reader.pc = otherwise;
                    }
                }
                // Queries retain only the values needed by the replacement.
                Op::Capture { rule } => {
                    // Query specialization establishes the input's scope. There
                    // is no current-rule filter: all matching branches contribute.
                    self.matches.capture(root, rule, &self.slots)?;
                }
                Op::Jump { target } => reader.pc = target,
                Op::Return {} => return Ok(()),
                op => panic!("rewrite opcode in query: {op:?}"),
            }
        }
    }

    /// Apply the saved matches after all queries have released their cursors.
    /// A full node budget can reject construction while still allowing later
    /// matches that only merge existing values or establish constant facts.
    pub(super) fn apply(
        &mut self,
        graph: &mut Graph,
        ir: &mut Expressions<'_>,
    ) -> Result<(), Limit> {
        let scope = graph.profile.scope("egraph.apply", 0);
        let mut status = Ok(());
        // Move the set out while rewriting so its keys remain immutable. Drain
        // every saved match, including after a node-budget stop, then reuse its
        // allocation next round. No discovery order is required.
        let mut pending = core::mem::take(&mut self.matches.entries);
        for matched in pending.drain() {
            let root = graph.canonicalize(matched.root);
            if ir.body().dfg().as_const(root.value()).is_some() {
                continue;
            }
            let rule = &PROGRAM.rules[matched.rule];
            self.values.resize(rule.slots, root.value());
            self.values[0] = root.value();
            for (slot, &value) in matched.captures.iter().enumerate() {
                self.values[slot + 1] = graph.canonicalize(value).value();
            }
            if let Err(limit) = self.rewrite(graph, ir, rule) {
                status = Err(limit);
            }
        }
        self.matches.entries = pending;
        scope.success();
        status
    }

    fn rewrite(
        &mut self,
        graph: &mut Graph,
        ir: &mut Expressions<'_>,
        rule: &Rule,
    ) -> Result<(), Limit> {
        let root = self.values[0];
        let ty = ir.body().dfg().value_type(root);
        let mask = u64::MAX >> (64 - ty.element_bits().expect("integer rewrite type"));
        let constant = |index| {
            ScalarConst::from_bits(ty, PROGRAM.constants[index] & mask)
                .expect("typed rule constant")
        };
        let mut reader = Reader {
            bytes: PROGRAM.code,
            pc: rule.entry,
        };
        loop {
            match Op::read(&mut reader) {
                Op::Constant {
                    dst,
                    constant: index,
                } => {
                    let value = graph.literal(ir, constant(index));
                    self.values[dst] = value;
                }
                Op::Build { dst, opcode, args } => {
                    self.args.clear();
                    self.args.extend(
                        args.iter()
                            .map(|slot| graph.find(self.values[slot]).value()),
                    );
                    let value = graph.build(ir, PROGRAM.opcodes[opcode], &self.args, ty)?;
                    self.values[dst] = value;
                }
                Op::Union { value } => {
                    graph.union(ir.body(), root, self.values[value]);
                    log::trace!("egraph rule {}", rule.name);
                }
                Op::SetConstant { constant: index } => {
                    graph.fold_to(ir, root, constant(index));
                    log::trace!("egraph rule {}", rule.name);
                }
                Op::Return {} => return Ok(()),
                op => panic!("query opcode in rewrite: {op:?}"),
            }
        }
    }

    /// Open a source with static bindings and an input checked when bound.
    fn open(&mut self, cursor: usize, source: Source, bindings: Bindings, input: TriggerId) {
        let id = self.index.len();
        let relation = *self.index.entry(source).or_insert_with(|| {
            let cursor = RuleCursor { source, row: 0 };
            if let Some(relation) = self.relations.get_mut(id) {
                relation.cursor = cursor;
                relation.rows.clear();
                relation.exhausted = false;
            } else {
                self.relations.push(Relation {
                    cursor,
                    rows: Vec::new(),
                    exhausted: false,
                });
            }
            id
        });
        self.cursors[cursor] = Some(Cursor {
            relation,
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
        body: &FuncBody,
        query: &Query,
        cursor: usize,
        fuel: &mut usize,
    ) -> bool {
        let cursor = self.cursors[cursor].as_mut().expect("opened query cursor");
        let relation = &mut self.relations[cursor.relation];
        let input_slot = trigger(cursor.input).slot;
        let seeds = &query.inputs[&cursor.input];
        while *fuel > 0 {
            *fuel -= 1;
            // Fetch at most one new row; other scans reuse the same cache.
            if cursor.row == relation.rows.len() && !relation.exhausted {
                match relation.cursor.next(graph, body, query) {
                    Some(row) => relation.rows.push(row),
                    None => relation.exhausted = true,
                }
            }
            let Some(row) = relation.rows.get(cursor.row) else {
                return false;
            };
            let binding = cursor.bindings[cursor.binding];
            debug_assert_eq!(binding.len(), row.len(), "pattern operand count");
            cursor.binding += 1;
            // All alternatives permute the same slots. Equal binary operands
            // make both orientations identical, so visit this row only once.
            if cursor.binding == cursor.bindings.len() || matches!(row.as_slice(), [a, b] if a == b)
            {
                cursor.binding = 0;
                cursor.row += 1;
            }

            if binding.iter().zip(row).any(|(&slot, &arg)| {
                slot == input_slot
                    && seeds
                        .binary_search_by_key(&arg, |&seed| seed.root(graph))
                        .is_err()
            }) {
                continue;
            }
            // Do not overwrite any slots until the entire binding is accepted.
            for (&slot, &value) in binding.iter().zip(row) {
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
    fn next(&mut self, graph: &Graph, body: &FuncBody, query: &Query) -> Option<Row> {
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
        let inst = body.dfg().value_inst(value).expect("relation result");
        // Rebuilding filters folded nodes from both relations and added seeds.
        // The compiler accepts only single-result pattern operations.
        Some(graph.canonical_args(body, inst))
    }
}
