//! Equality-saturation bytecode. All rules share one program and constant pool.
//! Queries run against a stable graph; actions run after enumeration finishes.
use smallvec::SmallVec;
use std::collections::HashMap;
use veloc_mir::{Opcode, Type, Value};

use super::graph::Graph;
use veloc_mir::constant::ScalarConst;
use veloc_mir::function::Expressions;
use veloc_types::TypeInfo;

use veloc_bytecode::{Reader, equivalence::Instruction as Op};

struct Group {
    slots: usize,
    cursors: usize,
    entry: usize,
}

pub(super) struct Dependency {
    pub root: Opcode,
    pub path: &'static [Opcode],
}

pub(super) fn dependencies() -> &'static [Dependency] {
    DEPENDENCIES
}

struct Program {
    code: &'static [u8],
    opcodes: &'static [Opcode],
    constants: &'static [u64],
    captures: &'static [usize],
    types: &'static [&'static [Type]],
    names: &'static [&'static str],
}
include!(concat!(env!("OUT_DIR"), "/equivalences.rs"));

/// The rule compiler resolves binding and backtracking. The executor reuses
/// registers and retains only the RHS inputs of each complete match.
pub(super) struct Machine {
    slots: Vec<Value>,
    cursors: Vec<Option<Cursor>>,
    relations: Vec<Relation>,
    index: HashMap<(Value, Opcode), usize>,
    matches: Matches,
    args: SmallVec<[Value; 3]>,
}

struct Cursor {
    relation: usize,
    row: usize,
    plan: usize,
}

struct Relation {
    cursor: RuleCursor,
    rows: Vec<SmallVec<[Value; 3]>>,
    exhausted: bool,
}

/// Captured RHS inputs for one rule. Search fills the buffer before Apply
/// starts iteration; the allocation is reused by subsequent rules.
#[derive(Default)]
struct Matches {
    values: Vec<Value>,
    captures: &'static [usize],
    // A match with no captured inputs still counts as one match.
    count: usize,
    next: usize,
}

impl Matches {
    fn begin(&mut self, captures: &'static [usize]) {
        self.values.clear();
        self.captures = captures;
        self.count = 0;
    }

    fn capture(&mut self, slots: &[Value]) {
        self.values
            .extend(self.captures.iter().map(|&slot| slots[slot]));
        self.count += 1;
    }

    fn apply(&mut self) {
        self.next = 0;
    }

    fn next(&mut self) -> Option<&[Value]> {
        if self.next == self.count {
            return None;
        }
        let width = self.captures.len();
        let start = self.next * width;
        self.next += 1;
        Some(&self.values[start..start + width])
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

    pub(super) fn run(
        &mut self,
        ctx: &mut RuleContext<'_, '_>,
        opcode: Opcode,
        root: Value,
        mask: u64,
        fuel: &mut usize,
    ) {
        let Some(group) = group(opcode) else { return };
        // The compiler assigns slots before use. Query and update temporaries
        // share this array because updates begin only after enumeration ends.
        self.slots.resize(group.slots, root);
        self.slots[0] = ctx.canonical(root);
        self.cursors.clear();
        self.cursors.resize_with(group.cursors, || None);
        // Reuse relation row allocations. Clearing the index invalidates all
        // entries; Open resets each buffer when assigning it a new query.
        self.index.clear();
        let mut revision = ctx.revision();
        let mut name = "";
        let mut reader = Reader {
            bytes: PROGRAM.code,
            pc: group.entry,
        };
        loop {
            let inst = Op::read(&mut reader);
            match inst {
                Op::CheckTypeIn {
                    value,
                    types,
                    failure,
                } => {
                    if !PROGRAM.types[types]
                        .contains(&ctx.ir.body().dfg().value_type(self.slots[value]))
                    {
                        reader.pc = failure;
                    }
                }
                Op::CheckSameType { lhs, rhs, failure } => {
                    let dfg = ctx.ir.body().dfg();
                    if dfg.value_type(self.slots[lhs]) != dfg.value_type(self.slots[rhs]) {
                        reader.pc = failure;
                    }
                }
                Op::StopIfConstant {} => {
                    if ctx.constant(root).is_some() {
                        return;
                    }
                }
                Op::Begin {
                    name: rule,
                    captures: start,
                    len,
                } => {
                    if *fuel == 0 {
                        return;
                    }
                    name = PROGRAM.names[rule];
                    self.matches.begin(&PROGRAM.captures[start..start + len]);
                    self.slots[0] = ctx.canonical(root);
                    self.cursors.clear();
                    self.cursors.resize_with(group.cursors, || None);
                    // Cached rows are reusable across rules, but never across
                    // graph mutations. Populate them lazily under query fuel.
                    if revision != ctx.revision() {
                        self.index.clear();
                        revision = ctx.revision();
                    }
                }
                Op::Open {
                    cursor,
                    source,
                    opcode,
                } => {
                    let opcode = PROGRAM.opcodes[opcode];
                    // Query slots contain canonical values throughout the
                    // stable search phase, which ends at Apply.
                    let class = self.slots[source];
                    let id = self.index.len();
                    let relation = *self.index.entry((class, opcode)).or_insert_with(|| {
                        let cursor = ctx.open(class, opcode);
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
                        plan: 0,
                    });
                }
                Op::Next {
                    cursor,
                    plans,
                    failure,
                    bindings,
                } => {
                    // Budget exhaustion follows the ordinary exhausted-query
                    // branches, so already captured matches still get applied.
                    let found = *fuel > 0 && {
                        *fuel -= 1;
                        let cursor = self.cursors[cursor].as_mut().expect("opened query cursor");
                        let relation = &mut self.relations[cursor.relation];
                        if cursor.row == relation.rows.len() && !relation.exhausted {
                            if let Some(row) = ctx.next(&mut relation.cursor) {
                                relation.rows.push(row);
                            } else {
                                relation.exhausted = true;
                            }
                        }
                        if let Some(row) = relation.rows.get(cursor.row) {
                            assert!(plans > 0, "nonempty binding plans");
                            assert_eq!(row.len() * plans, bindings.len(), "pattern operand count");
                            // Plans map relation columns to slots. The VM does
                            // not need to know why a rule has several plans.
                            let start = cursor.plan * row.len();
                            for (slot, &value) in
                                bindings.iter().skip(start).take(row.len()).zip(row)
                            {
                                self.slots[slot] = value;
                            }
                            cursor.plan += 1;
                            if cursor.plan == plans {
                                cursor.plan = 0;
                                cursor.row += 1;
                            }
                            true
                        } else {
                            false
                        }
                    };
                    if !found {
                        reader.pc = failure;
                    }
                }
                Op::CheckEqual { lhs, rhs, failure } => {
                    if self.slots[lhs] != self.slots[rhs] {
                        reader.pc = failure;
                    }
                }
                Op::CheckConstantEq {
                    value,
                    constant,
                    failure,
                }
                | Op::CheckConstantNe {
                    value,
                    constant,
                    failure,
                } => {
                    let value = self.slots[value];
                    let bits = PROGRAM.constants[constant] & mask;
                    if !ctx.graph.constants[value].is_some_and(|c| {
                        (c.to_bits() == bits) == matches!(inst, Op::CheckConstantEq { .. })
                    }) {
                        reader.pc = failure;
                    }
                }
                Op::Capture {} => {
                    self.matches.capture(&self.slots);
                }
                Op::Apply {} => {
                    // No relation cursor survives a graph update. The captured
                    // IDs remain valid, and are canonicalized after prior unions.
                    self.cursors.clear();
                    self.matches.apply();
                }
                Op::NextMatch { failure } => {
                    if let Some(values) = self.matches.next() {
                        for (slot, &value) in values.iter().enumerate() {
                            self.slots[slot + 1] = ctx.canonical(value);
                        }
                    } else {
                        reader.pc = failure;
                    }
                }
                Op::Jump { target } => reader.pc = target,
                Op::Constant {
                    dst: slot,
                    constant,
                    failure,
                } => {
                    let bits = PROGRAM.constants[constant] & mask;
                    if let Some(value) = ctx.literal(bits) {
                        self.slots[slot] = value;
                    } else {
                        reader.pc = failure;
                    }
                }
                Op::Build {
                    dst: slot,
                    opcode,
                    args,
                    failure,
                } => {
                    let opcode = PROGRAM.opcodes[opcode];
                    self.args.clear();
                    self.args
                        .extend(args.iter().map(|slot| ctx.canonical(self.slots[slot])));
                    if let Some(value) = ctx.build(opcode, &self.args) {
                        self.slots[slot] = value;
                    } else {
                        reader.pc = failure;
                    }
                }
                Op::Union { value } => {
                    ctx.union(root, self.slots[value]);
                    log::trace!("egraph rule {}", name);
                }
                Op::SetConstant { constant } => {
                    ctx.set_constant(root, PROGRAM.constants[constant] & mask);
                    log::trace!("egraph rule {}", name);
                }
                Op::Return {} => return,
            }
        }
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

pub(super) struct RuleContext<'a, 'body> {
    pub(super) graph: &'a mut Graph,
    pub(super) ir: &'a mut Expressions<'body>,
    pub(super) ty: Type,
}

/// Valid only while querying a stable graph, before applying any updates.
struct RuleCursor {
    class: Value,
    opcode: Opcode,
    row: usize,
}

impl RuleContext<'_, '_> {
    fn revision(&self) -> usize {
        self.graph.revision
    }

    fn canonical(&self, value: Value) -> Value {
        self.graph.find(value)
    }
    fn constant(&self, value: Value) -> Option<u64> {
        self.graph.constant(value).map(|c| c.to_bits())
    }

    fn open(&self, value: Value, opcode: Opcode) -> RuleCursor {
        RuleCursor {
            class: value,
            opcode,
            row: 0,
        }
    }

    fn next(&self, cursor: &mut RuleCursor) -> Option<SmallVec<[Value; 3]>> {
        let values = self.graph.relations.get(&(cursor.class, cursor.opcode))?;
        let &value = values.get(cursor.row)?;
        cursor.row += 1;
        let f = self.ir.body();
        let inst = f.dfg().value_inst(value).expect("relation result");
        // The compiler accepts only single-result pattern operations. Result
        // types are shared by all members of the queried equivalence class;
        // operand type requirements are emitted in the matching program.
        Some(self.graph.canonical_args(f, inst))
    }
    fn union(&mut self, lhs: Value, rhs: Value) {
        self.graph.union(self.ir.body(), lhs, rhs);
    }

    fn set_constant(&mut self, value: Value, bits: u64) {
        let constant = ScalarConst::from_bits(self.ty, bits).expect("typed rule constant");
        self.graph.set_const(self.ir.body(), value, constant);
    }

    fn literal(&mut self, value: u64) -> Option<Value> {
        let mask = u64::MAX >> (64 - self.ty.element_bits()?);
        self.graph
            .literal(self.ir, ScalarConst::from_bits(self.ty, value & mask)?)
    }

    fn build(&mut self, opcode: Opcode, args: &[Value]) -> Option<Value> {
        self.graph.build(self.ir, opcode, args, self.ty)
    }
}
