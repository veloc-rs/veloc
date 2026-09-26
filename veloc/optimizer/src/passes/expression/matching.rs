//! Equality-saturation bytecode. All rules share one program and constant pool.
//! Queries run against a stable graph; actions run after enumeration finishes.
use hashbrown::{HashTable, hash_map::DefaultHashBuilder};
use smallvec::SmallVec;
use std::collections::{BTreeMap, HashMap};
use std::hash::BuildHasher;
use veloc_bytecode::{Reader, equivalence::Instruction as Op};
use veloc_mir::{FuncBody, Opcode, Type, Value, constant::ScalarConst, function::Expressions};
use veloc_types::TypeInfo;

use super::graph::Graph;

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
    pub path: &'static [Edge],
}

pub(super) struct Edge {
    pub opcode: Opcode,
    pub columns: &'static [usize],
}

pub(super) fn trigger(id: usize) -> &'static Trigger {
    &TRIGGERS[id]
}

/// One root/opcode query, with the union of all affected input positions.
pub(super) struct Query {
    pub root: Value,
    pub opcode: Opcode,
    /// Trigger ID -> seeds sorted by (class, node). Added-node seeds keep
    /// their concrete identity; constant/merge seeds are class representatives.
    pub inputs: BTreeMap<usize, Vec<Value>>,
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
    slots: Vec<Value>,
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
    input: usize,
}

/// Relation data can be shared independently of bindings and input constraints.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Source {
    Empty,
    Relation {
        class: Value,
        opcode: Opcode,
    },
    // A class-filtered slice of one input's seeds. Indices are valid only
    // within the current query; relation caches are reset between queries.
    Candidates {
        input: usize,
        start: usize,
        end: usize,
    },
}

/// Lazily populated rows shared by scans of the same class/opcode pair.
struct Relation {
    cursor: RuleCursor,
    rows: Vec<SmallVec<[Value; 3]>>,
    exhausted: bool,
}

/// All queries in a round share this buffer. Roots are explicit because a
/// round can match several classes before any graph updates are applied.
#[derive(Default)]
struct Matches {
    values: Vec<Value>,
    rows: Vec<Match>,
    seen: HashTable<usize>, // Row indices, without copying captured values.
    hasher: DefaultHashBuilder,
    limit: usize,
}

#[derive(Clone, Copy)]
struct Match {
    root: Value,
    rule: usize,
    start: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum QueryLimit {
    Work,
    Matches,
}

impl Matches {
    fn begin(&mut self, limit: usize) {
        self.values.clear();
        self.rows.clear();
        self.seen.clear();
        self.limit = limit;
    }

    /// Save a complete match; return whether querying may continue.
    /// The soft budget may be exceeded by at most one match's charge.
    fn capture(&mut self, root: Value, rule: usize, slots: &[Value]) -> bool {
        let captures = PROGRAM.rules[rule].captures;
        let values: SmallVec<[Value; 3]> = captures.iter().map(|&slot| slots[slot]).collect();
        let hash = self.hasher.hash_one((root, rule, values.as_slice()));
        if self
            .seen
            .find(hash, |&row| {
                let old = self.rows[row];
                old.root == root
                    && old.rule == rule
                    && self.values[old.start..old.start + captures.len()] == values[..]
            })
            .is_some()
        {
            return true;
        }
        let row = self.rows.len();
        self.rows.push(Match {
            root,
            rule,
            start: self.values.len(),
        });
        self.values.extend_from_slice(&values);
        self.seen.insert_unique(hash, row, |&row| {
            let old = self.rows[row];
            let end = old.start + PROGRAM.rules[old.rule].captures.len();
            self.hasher
                .hash_one((old.root, old.rule, &self.values[old.start..end]))
        });
        // Charge the three header fields too, even for zero-capture rules.
        self.rows.len() * 3 + self.values.len() < self.limit
    }
}

impl Machine {
    pub(super) fn new() -> Self {
        Self {
            slots: Vec::new(),
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
        fuel: &mut usize,
    ) -> Result<(), QueryLimit> {
        self.index.clear();
        self.matches.begin(*fuel);
        let result = queries.iter().try_for_each(|query| {
            let root = query.root;
            if graph.constant(root).is_some() {
                return Ok(());
            }
            let ty = body.dfg().value_type(root);
            if !ty.is_integer() && ty != Type::BOOL {
                return Ok(());
            }
            self.index.clear();
            // Inputs are alternatives (OR), never cumulative constraints.
            // Distinct entry plans retain their own order while sharing rows.
            for &input in query.inputs.keys() {
                if *fuel == 0 {
                    return Err(QueryLimit::Work);
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
        result
    }

    fn query(
        &mut self,
        graph: &Graph,
        body: &FuncBody,
        query: &Query,
        input: usize,
        fuel: &mut usize,
    ) -> Result<(), QueryLimit> {
        let entry = trigger(input);
        let seeds = &query.inputs[&input];
        let root = query.root;
        let ty = body.dfg().value_type(root);
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
                    let ty = body.dfg().value_type(self.slots[value]);
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
                    if dfg.value_type(self.slots[lhs]) != dfg.value_type(self.slots[rhs]) {
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
                        let start = seeds.partition_point(|&v| graph.find(v) < class);
                        let end = seeds.partition_point(|&v| graph.find(v) <= class);
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
                        return Err(QueryLimit::Work);
                    }
                    if !self.scan_next(graph, body, query, cursor, fuel) {
                        if *fuel == 0 {
                            return Err(QueryLimit::Work);
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
                    let bits = PROGRAM.constants[constant] & mask;
                    if graph.constants[self.slots[value]].map(|c| c.to_bits()) != Some(bits) {
                        reader.pc = otherwise;
                    }
                }
                Op::CheckConstantNe {
                    value,
                    constant,
                    otherwise,
                } => {
                    let bits = PROGRAM.constants[constant] & mask;
                    // Unknown is not evidence of inequality.
                    if !graph.constants[self.slots[value]].is_some_and(|c| c.to_bits() != bits) {
                        reader.pc = otherwise;
                    }
                }
                // Queries retain only the values needed by the replacement.
                Op::Capture { rule } => {
                    // Query specialization establishes the input's scope. There
                    // is no current-rule filter: all matching branches contribute.
                    if !self.matches.capture(root, rule, &self.slots) {
                        return Err(QueryLimit::Matches);
                    }
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
    pub(super) fn apply(&mut self, graph: &mut Graph, ir: &mut Expressions<'_>) -> bool {
        let mut complete = true;
        for row in 0..self.matches.rows.len() {
            let matched = self.matches.rows[row];
            let root = graph.find(matched.root);
            if graph.constant(root).is_some() {
                continue;
            }
            let rule = &PROGRAM.rules[matched.rule];
            self.slots.resize(rule.slots, root);
            self.slots[0] = root;
            for (slot, &value) in self.matches.values
                [matched.start..matched.start + rule.captures.len()]
                .iter()
                .enumerate()
            {
                self.slots[slot + 1] = graph.find(value);
            }
            complete &= self.rewrite(graph, ir, rule);
        }
        complete
    }

    fn rewrite(&mut self, graph: &mut Graph, ir: &mut Expressions<'_>, rule: &Rule) -> bool {
        let root = self.slots[0];
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
                    let Some(value) = graph.literal(ir, constant(index)) else {
                        return false;
                    };
                    self.slots[dst] = value;
                }
                Op::Build { dst, opcode, args } => {
                    self.args.clear();
                    self.args
                        .extend(args.iter().map(|slot| graph.find(self.slots[slot])));
                    let Some(value) = graph.build(ir, PROGRAM.opcodes[opcode], &self.args, ty)
                    else {
                        return false;
                    };
                    self.slots[dst] = value;
                }
                Op::Union { value } => {
                    graph.union(ir.body(), root, self.slots[value]);
                    log::trace!("egraph rule {}", rule.name);
                }
                Op::SetConstant { constant: index } => {
                    graph.set_const(ir.body(), root, constant(index));
                    log::trace!("egraph rule {}", rule.name);
                }
                Op::Return {} => return true,
                op => panic!("query opcode in rewrite: {op:?}"),
            }
        }
    }

    /// Open a source with static bindings and an input checked when bound.
    fn open(&mut self, cursor: usize, source: Source, bindings: Bindings, input: usize) {
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
        let input = trigger(cursor.input).slot;
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
                slot == input
                    && seeds
                        .binary_search_by_key(&arg, |&v| graph.find(v))
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
    fn next(
        &mut self,
        graph: &Graph,
        body: &FuncBody,
        query: &Query,
    ) -> Option<SmallVec<[Value; 3]>> {
        // Borrow only for this read; the reusable cursor owns no graph borrow.
        // Empty, changed-node and relation sources share the exhaustion rule.
        let values: &[Value] = match &self.source {
            Source::Empty => &[],
            Source::Candidates { input, start, end } => &query.inputs[input][*start..*end],
            Source::Relation { class, opcode } => graph
                .relations
                .get(&(*class, *opcode))
                .map_or(&[], Vec::as_slice),
        };
        let value = *values.get(self.row)?;
        self.row += 1;
        let inst = body.dfg().value_inst(value).expect("relation result");
        // The compiler accepts only single-result pattern operations. Result
        // types are shared by all members of the queried equivalence class;
        // operand type requirements are emitted in the matching program.
        Some(graph.canonical_args(body, inst))
    }
}
