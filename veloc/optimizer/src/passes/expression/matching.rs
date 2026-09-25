//! Equality-saturation bytecode. All rules share one program and constant pool.
//! Queries run against a stable graph; actions run after enumeration finishes.
use smallvec::SmallVec;
use std::collections::HashMap;
use veloc_mir::{Opcode, Type, Value};

use super::graph::Graph;
use veloc_mir::constant::ScalarConst;
use veloc_mir::function::Expressions;
use veloc_types::TypeInfo;

use veloc_bytecode::{Reader, opcodes};

// u32 operands; Build additionally carries a length-prefixed slot list.
opcodes! {
    enum Op {
        CheckType(1, true),
        Begin(3, false),
        Open(3, false),
        Next(1, true),
        Bind(2, false),
        CheckEqual(2, true),
        CheckConstant(3, true),
        Capture(0, false),
        Jump(0, true),
        Constant(2, false),
        Build(3, false),
        Union(1, false),
        SetConstant(1, false),
        NextMatch(0, true),
        Return(0, false),
    }
}

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
    row: SmallVec<[Value; 3]>,
    output: Vec<Value>,
    args: SmallVec<[Value; 3]>,
}

struct Cursor {
    relation: usize,
    row: usize,
}

struct Relation {
    cursor: RuleCursor,
    rows: Vec<SmallVec<[Value; 3]>>,
    exhausted: bool,
}

impl Machine {
    pub(super) fn new() -> Self {
        Self {
            slots: Vec::new(),
            cursors: Vec::new(),
            relations: Vec::new(),
            index: HashMap::new(),
            row: SmallVec::new(),
            output: Vec::new(),
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
        self.output.clear();
        self.relations.clear();
        self.index.clear();
        let mut revision = ctx.revision();
        let mut captures = &PROGRAM.captures[0..0];
        let mut name = "";
        let mut count = 0;
        let mut current = 0;
        let mut next_match = group.entry;
        let mut reader = Reader {
            bytes: PROGRAM.code,
            pc: group.entry,
        };
        loop {
            let pc = reader.pc;
            match Op::decode(reader.byte()) {
                Op::CheckType => {
                    let types = PROGRAM.types[reader.u32()];
                    reader.branch(types.contains(&ctx.ty()));
                }
                Op::Begin => {
                    if *fuel == 0 || ctx.constant(root).is_some() {
                        return;
                    }
                    name = PROGRAM.names[reader.u32()];
                    let start = reader.u32();
                    let len = reader.u32();
                    captures = &PROGRAM.captures[start..start + len];
                    self.slots[0] = ctx.canonical(root);
                    self.cursors.clear();
                    self.cursors.resize_with(group.cursors, || None);
                    self.output.clear();
                    count = 0;
                    current = 0;
                    // Cached rows are reusable across rules, but never across
                    // graph mutations. Populate them lazily under query fuel.
                    if revision != ctx.revision() {
                        self.relations.clear();
                        self.index.clear();
                        revision = ctx.revision();
                    }
                }
                Op::Open => {
                    let cursor = reader.u32();
                    let source = reader.u32();
                    let opcode = PROGRAM.opcodes[reader.u32()];
                    let class = ctx.canonical(self.slots[source]);
                    let relation = *self.index.entry((class, opcode)).or_insert_with(|| {
                        let id = self.relations.len();
                        self.relations.push(Relation {
                            cursor: ctx.open(class, opcode),
                            rows: Vec::new(),
                            exhausted: false,
                        });
                        id
                    });
                    self.cursors[cursor] = Some(Cursor { relation, row: 0 });
                }
                Op::Next => {
                    let cursor = reader.u32();
                    // Budget exhaustion follows the ordinary exhausted-query
                    // branches, so already captured matches still get applied.
                    let found = *fuel > 0 && {
                        *fuel -= 1;
                        let cursor = self.cursors[cursor].as_mut().expect("opened query cursor");
                        let relation = &mut self.relations[cursor.relation];
                        if cursor.row == relation.rows.len() && !relation.exhausted {
                            if ctx.next(&mut relation.cursor, &mut self.row) {
                                relation.rows.push(self.row.clone());
                            } else {
                                relation.exhausted = true;
                            }
                        }
                        if let Some(row) = relation.rows.get(cursor.row) {
                            self.row.clone_from(row);
                            cursor.row += 1;
                            true
                        } else {
                            false
                        }
                    };
                    reader.branch(found);
                }
                Op::Bind => {
                    let column = reader.u32();
                    let slot = reader.u32();
                    self.slots[slot] = ctx.canonical(self.row[column]);
                }
                Op::CheckEqual => {
                    let lhs = self.slots[reader.u32()];
                    let rhs = self.slots[reader.u32()];
                    reader.branch(ctx.canonical(lhs) == ctx.canonical(rhs));
                }
                Op::CheckConstant => {
                    let value = self.slots[reader.u32()];
                    let bits = PROGRAM.constants[reader.u32()] & mask;
                    let equal = reader.u32() != 0;
                    reader.branch(ctx.constant(value).is_some_and(|c| (c == bits) == equal));
                }
                Op::Capture => {
                    self.output
                        .extend(captures.iter().map(|&slot| self.slots[slot]));
                    count += 1;
                }
                Op::NextMatch => {
                    // No relation cursor survives a graph update. The captured
                    // IDs remain valid, and are canonicalized after prior unions.
                    self.cursors.clear();
                    next_match = pc;
                    let available = current < count && ctx.constant(root).is_none();
                    reader.branch(available);
                    if available {
                        let start = current * captures.len();
                        for (slot, &value) in self.output[start..start + captures.len()]
                            .iter()
                            .enumerate()
                        {
                            self.slots[slot] = ctx.canonical(value);
                        }
                        current += 1;
                    }
                }
                Op::Jump => reader.pc = reader.u32(),
                Op::Constant => {
                    let slot = reader.u32();
                    let bits = PROGRAM.constants[reader.u32()] & mask;
                    if let Some(value) = ctx.literal(bits) {
                        self.slots[slot] = value;
                    } else {
                        reader.pc = next_match;
                    }
                }
                Op::Build => {
                    let slot = reader.u32();
                    let opcode = PROGRAM.opcodes[reader.u32()];
                    self.args.clear();
                    for _ in 0..reader.u32() {
                        self.args.push(ctx.canonical(self.slots[reader.u32()]));
                    }
                    if let Some(value) = ctx.build(opcode, &self.args) {
                        self.slots[slot] = value;
                    } else {
                        reader.pc = next_match;
                    }
                }
                Op::Union => {
                    ctx.union(root, self.slots[reader.u32()]);
                    log::trace!("egraph rule {}", name);
                }
                Op::SetConstant => {
                    ctx.set_constant(root, PROGRAM.constants[reader.u32()] & mask);
                    log::trace!("egraph rule {}", name);
                }
                Op::Return => return,
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
            write!(out, "{:04x}: ", reader.pc)?;
            let op = Op::decode(reader.byte());
            write!(out, "{op:?}")?;
            let (arity, branch) = op.format();
            let mut last = 0;
            for _ in 0..arity {
                last = reader.u32();
                write!(out, " {last}")?;
            }
            if matches!(op, Op::Build) {
                write!(out, " [")?;
                for i in 0..last {
                    if i != 0 {
                        write!(out, ", ")?;
                    }
                    write!(out, "{}", reader.u32())?;
                }
                write!(out, "]")?;
            }
            if branch {
                write!(out, " -> {:04x}", reader.u32())?;
            }
            writeln!(out)?;
            if matches!(op, Op::Return) {
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
    reverse: bool,
}

impl RuleContext<'_, '_> {
    fn revision(&self) -> usize {
        self.graph.revision
    }

    fn ty(&self) -> Type {
        self.ty
    }

    fn canonical(&self, value: Value) -> Value {
        self.graph.find(value)
    }
    fn constant(&self, value: Value) -> Option<u64> {
        self.graph.constant(value).map(|c| c.to_bits())
    }

    fn open(&self, value: Value, opcode: Opcode) -> RuleCursor {
        RuleCursor {
            class: self.graph.find(value),
            opcode,
            row: 0,
            reverse: false,
        }
    }

    fn next(&self, cursor: &mut RuleCursor, row: &mut SmallVec<[Value; 3]>) -> bool {
        let Some(values) = self.graph.relations.get(&(cursor.class, cursor.opcode)) else {
            return false;
        };
        while let Some(&value) = values.get(cursor.row) {
            let f = self.ir.body();
            let inst = f.dfg().value_inst(value).expect("relation result");
            if f.dfg().inst_results(inst).len() != 1 || f.dfg().value_type(value) != self.ty {
                cursor.row += 1;
                continue;
            }
            *row = self.graph.canonical_args(f, inst);
            if row.iter().any(|&v| f.dfg().value_type(v) != self.ty) {
                cursor.row += 1;
                continue;
            }
            if cursor.reverse {
                row.swap(0, 1);
                cursor.reverse = false;
                cursor.row += 1;
            } else if cursor.opcode.spec().is_commutative() && row.len() == 2 && row[0] != row[1] {
                cursor.reverse = true;
            } else {
                cursor.row += 1;
            }
            return true;
        }
        false
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
