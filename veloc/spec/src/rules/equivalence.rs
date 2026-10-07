//! Incremental e-class query compilation over the shared checked rule model.
use super::expression::{self, AttributeValue, CheckedRule, Filter, PatternKind, TypeSource};
use crate::{Definitions, Error};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write,
};
use veloc_bytecode::{Reader, Words, equivalence::Instruction as Op};

pub(crate) struct Output {
    pub local_folds: String,
    pub equivalences: String,
}

pub(crate) fn generate(
    source: &crate::Source,
    defs: &Definitions,
    dialect: &str,
    opcode: &str,
    types: &str,
) -> Result<Output, Error> {
    if !super::identifier(dialect)
        || ![opcode, types]
            .iter()
            .all(|p| p.split("::").all(super::identifier))
    {
        return Err(Error::at(
            source.text(),
            0,
            "invalid expression Rust binding",
        ));
    }
    let rules = expression::compile(source, defs, dialect)?;
    let local_folds =
        expression::emit_attributes(defs) + &expression::emit_local(defs, &rules, opcode, types);
    let mut groups = BTreeMap::<&str, Vec<(usize, &CheckedRule)>>::new();
    for (id, rule) in rules.iter().enumerate().filter(|(_, r)| !r.is_flat()) {
        groups.entry(rule.opcode()).or_default().push((id, rule));
    }
    let mut program = Bytecode::new(defs);
    let mut output = format!("fn group(opcode: {opcode}) -> Option<Group> {{ match opcode {{\n");
    for (op, rules) in groups {
        let group = program.group(&rules);
        writeln!(output, "{opcode}::{op} => Some(Group {{ {group} }}),").unwrap();
    }
    output.push_str("_ => None,\n} }\n");
    output.push_str(&program.emit(opcode));
    Ok(Output {
        local_folds,
        equivalences: output,
    })
}

/// Build-time byte assembler. Opcode numbers belong to the runtime enum;
/// operands and branch destinations are little-endian u32 values.
#[derive(Default)]
struct Code {
    bytes: Vec<u8>,
}
impl Code {
    fn emit(&mut self, inst: Op<'_>) -> usize {
        let pc = self.bytes.len();
        inst.encode(&mut self.bytes);
        pc
    }

    fn patch(&mut self, pc: usize, field: &str, value: usize) {
        let inst = Op::read(&mut Reader {
            bytes: &self.bytes,
            pc,
        });
        let offset = pc + inst.field_offset(field).expect("bytecode relocation field");
        self.bytes[offset..offset + 4].copy_from_slice(&veloc_bytecode::encode_u32(value));
    }
}

struct Action {
    plan: usize,
    name: String,
    captures: Vec<usize>,
    nodes: Vec<usize>,
}

#[derive(Clone, PartialEq, Eq)]
enum Step {
    Type(usize, usize),
    Scan {
        source: usize,
        opcode: usize,
        cursor: usize,
        bindings: Vec<Vec<usize>>,
    },
    Same(usize, usize),
    Equal(usize, usize),
    IsConstant(usize),
    Constant(usize, usize, bool),
    Properties(usize, usize, usize),
    Capture(usize),
}

/// Check each available binding before opening another relation. In particular,
/// constants and repeated variables in a sibling operand can reject a row
/// before its other operand starts a nested scan. This is a dependency-based
/// query plan, independent of the source pattern's spelling order.
fn order_checks(steps: Vec<Step>) -> Vec<Step> {
    let mut pending = steps;
    let mut ordered = Vec::new();
    let mut bound = BTreeSet::from([0]);
    let mut witnesses = BTreeSet::new();
    while !pending.is_empty() {
        let check = pending.iter().position(|step| match step {
            Step::Type(v, _) | Step::IsConstant(v) | Step::Constant(v, _, _) => bound.contains(v),
            Step::Same(a, b) | Step::Equal(a, b) => bound.contains(a) && bound.contains(b),
            Step::Properties(a, b, _) => witnesses.contains(a) && witnesses.contains(b),
            _ => false,
        });
        let next = check.unwrap_or_else(|| {
            pending
                .iter()
                .position(|step| match step {
                    Step::Scan { source, .. } => bound.contains(source),
                    Step::Capture(_) => pending.len() == 1,
                    _ => false,
                })
                .expect("acyclic checked query dependencies")
        });
        let step = pending.remove(next);
        if let Step::Scan {
            source, bindings, ..
        } = &step
        {
            bound.extend(bindings[0].iter().copied());
            witnesses.insert(*source);
        }
        ordered.push(step);
    }
    ordered
}

/// Predicates on any occurrence of a repeated variable constrain all its
/// occurrences. Derive them from the same equality checks used by the matcher.
fn input_constants(prefix: &[(Step, usize)], slot: usize) -> (bool, Vec<(usize, bool)>) {
    let mut aliases = BTreeSet::new();
    let mut pending = vec![slot];
    while let Some(slot) = pending.pop() {
        if !aliases.insert(slot) {
            continue;
        }
        for (step, _) in prefix {
            if let Step::Equal(a, b) = *step {
                if a == slot {
                    pending.push(b);
                }
                if b == slot {
                    pending.push(a);
                }
            }
        }
    }
    let requires_constant = prefix
        .iter()
        .any(|(step, _)| matches!(step, Step::IsConstant(slot) if aliases.contains(slot)));
    let constants = prefix
        .iter()
        .filter_map(|(step, _)| match *step {
            Step::Constant(slot, constant, equal) if aliases.contains(&slot) => {
                Some((constant, equal))
            }
            _ => None,
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    (requires_constant, constants)
}

struct Search {
    step: Step,
    scan: Option<usize>,
    children: Vec<Search>,
}

/// Assign identities on the structured plan, before bytecode layout exists.
fn number_scans(nodes: &mut [Search], next: &mut usize) {
    for node in nodes {
        if matches!(node.step, Step::Scan { .. }) {
            node.scan = Some(*next);
            *next += 1;
        }
        number_scans(&mut node.children, next);
    }
}

/// Prune at generation time; the VM never dispatches on a requested rule.
fn serves(node: &Search, rules: &BTreeSet<usize>) -> bool {
    match node.step {
        Step::Capture(rule) => rules.contains(&rule),
        _ => node.children.iter().any(|child| serves(child, rules)),
    }
}

fn insert_path(nodes: &mut Vec<Search>, steps: &[Step]) {
    let Some((step, rest)) = steps.split_first() else {
        return;
    };
    let index = nodes
        .iter()
        .position(|node| node.step == *step)
        .unwrap_or_else(|| {
            nodes.push(Search {
                step: step.clone(),
                scan: None,
                children: Vec::new(),
            });
            nodes.len() - 1
        });
    insert_path(&mut nodes[index].children, rest);
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Edge {
    opcode: usize,
    columns: Vec<usize>,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Event {
    Added(usize),
    ClassChanged,
}

struct Trigger {
    root: usize,
    // Used only to select a query subtree during generation.
    rules: BTreeSet<usize>,
    entry: usize,
    slot: usize,
    event: Event,
    scan: Option<usize>,
    path: Vec<Edge>,
    types: usize,
    requires_constant: bool,
    constants: Vec<(usize, bool)>,
    work: usize,
}

struct Bytecode<'a> {
    definitions: &'a Definitions,
    scans: usize,
    code: Code,
    opcodes: Vec<String>,
    constants: Vec<u64>,
    bindings: Vec<Vec<Vec<usize>>>,
    actions: Vec<Action>,
    triggers: Vec<Trigger>,
    types: Vec<String>,
    properties: Vec<String>,
    cursors: usize,
    commutative: BTreeSet<String>,
}

impl<'a> Bytecode<'a> {
    fn new(defs: &'a Definitions) -> Self {
        Self {
            definitions: defs,
            scans: 0,
            code: Code::default(),
            opcodes: Vec::new(),
            constants: Vec::new(),
            bindings: Vec::new(),
            actions: Vec::new(),
            triggers: Vec::new(),
            types: Vec::new(),
            properties: Vec::new(),
            cursors: 0,
            commutative: defs
                .ops
                .iter()
                .filter(|op| op.traits.contains("COMMUTATIVE"))
                .map(|op| op.name.clone())
                .collect(),
        }
    }

    fn group(&mut self, rules: &[(usize, &CheckedRule)]) -> String {
        let entry = self.code.bytes.len();
        let first_action = self.actions.len();
        let mut tree = Vec::new();
        let (mut slots, mut cursors) = (1, 0);
        for &(id, rule) in rules {
            self.cursors = 0;
            let mut steps = Vec::new();
            self.pattern(rule, 0, &mut steps);
            for filter in &rule.filters() {
                steps.push(match *filter {
                    Filter::IsConstant(slot) => Step::IsConstant(slot),
                    Filter::Constant(slot, bits, equal) => {
                        let constant = self.constant(bits);
                        Step::Constant(slot, constant, equal)
                    }
                });
            }
            let action = self.actions.len();
            self.actions.push(Action {
                plan: id,
                name: rule.name.clone(),
                captures: rule.captures(),
                nodes: rule.nodes(),
            });
            steps.push(Step::Capture(action));
            insert_path(&mut tree, &order_checks(steps));
            slots = slots.max(rule.pattern.len());
            cursors = cursors.max(self.cursors);
        }
        number_scans(&mut tree, &mut self.scans);
        let first_trigger = self.triggers.len();
        self.collect_triggers(&tree, &mut Vec::new());

        // A shared input activates all relevant rules, not one rule task at a
        // time. Exact input identity includes the stable scan and reverse path.
        let mut queries = BTreeMap::new();
        for trigger in self.triggers.drain(first_trigger..) {
            let key = (
                trigger.root,
                trigger.slot,
                trigger.event.clone(),
                trigger.scan,
                trigger.path.clone(),
                trigger.types,
                trigger.requires_constant,
                trigger.constants.clone(),
            );
            queries
                .entry(key)
                .and_modify(|old: &mut Trigger| old.rules.extend(trigger.rules.iter().copied()))
                .or_insert(trigger);
        }
        let all_rules: BTreeSet<_> = (first_action..self.actions.len()).collect();
        fn work(nodes: &[Search], rules: &BTreeSet<usize>) -> usize {
            nodes
                .iter()
                .filter(|node| serves(node, rules))
                .map(|node| {
                    usize::from(matches!(node.step, Step::Scan { .. }))
                        + work(&node.children, rules)
                })
                .sum()
        }
        let full_work = work(&tree, &all_rules);
        let full_exits = self.search(&tree, &all_rules);
        self.patch_exits(full_exits, self.code.bytes.len());
        self.code.emit(Op::Return {});
        let mut entries = BTreeMap::from([(all_rules, entry)]);
        let mut exits = Vec::new();
        for (_, mut trigger) in queries {
            trigger.work = work(&tree, &trigger.rules);
            // Different inputs can use the same selected query. Emit it once.
            trigger.entry = if let Some(&entry) = entries.get(&trigger.rules) {
                entry
            } else {
                let entry = self.code.bytes.len();
                exits.extend(self.search(&tree, &trigger.rules));
                entries.insert(trigger.rules.clone(), entry);
                entry
            };
            self.triggers.push(trigger);
        }
        self.patch_exits(exits, self.code.bytes.len());
        self.code.emit(Op::Return {});
        format!("entry: {entry}, work: {full_work}, slots: {slots}, cursors: {cursors}")
    }

    fn patch_exits(&mut self, exits: Vec<(usize, &'static str)>, target: usize) {
        for (pc, field) in exits {
            self.code.patch(pc, field, target);
        }
    }

    /// Siblings are alternatives, not exclusive cases. Exhaust every child
    /// before advancing its parent scan; failed checks try the next sibling.
    fn search(&mut self, nodes: &[Search], rules: &BTreeSet<usize>) -> Vec<(usize, &'static str)> {
        let mut exits = Vec::new();
        for node in nodes {
            if !serves(node, rules) {
                continue;
            }
            self.patch_exits(exits, self.code.bytes.len());
            exits = Vec::new();
            match &node.step {
                Step::Scan {
                    source,
                    opcode,
                    cursor,
                    bindings,
                } => {
                    let scan = node.scan.expect("numbered scan");
                    let bindings = crate::bytecode::intern(&mut self.bindings, bindings.clone());
                    self.code.emit(Op::OpenScan {
                        scan,
                        cursor: *cursor,
                        source: *source,
                        opcode: *opcode,
                        bindings,
                    });
                    let retry = self.code.bytes.len();
                    let pc = self.code.emit(Op::ScanNext {
                        cursor: *cursor,
                        exhausted: 0,
                    });
                    let children = self.search(&node.children, rules);
                    self.patch_exits(children, retry);
                    exits.push((pc, "exhausted"));
                }
                Step::Capture(rule) => {
                    self.code.emit(Op::Capture {
                        rule: *rule,
                        values: Words::Values(&self.actions[*rule].captures),
                        nodes: Words::Values(&self.actions[*rule].nodes),
                    });
                    exits.push((self.code.emit(Op::Jump { target: 0 }), "target"));
                }
                step => {
                    let op = match *step {
                        Step::Type(value, types) => Op::CheckTypeIn {
                            value,
                            types,
                            otherwise: 0,
                        },
                        Step::Same(lhs, rhs) => Op::CheckSameType {
                            lhs,
                            rhs,
                            otherwise: 0,
                        },
                        Step::Equal(lhs, rhs) => Op::CheckEqual {
                            lhs,
                            rhs,
                            otherwise: 0,
                        },
                        Step::IsConstant(value) => Op::CheckIsConstant {
                            value,
                            otherwise: 0,
                        },
                        Step::Constant(value, constant, true) => Op::CheckConstantEq {
                            value,
                            constant,
                            otherwise: 0,
                        },
                        Step::Constant(value, constant, false) => Op::CheckConstantNe {
                            value,
                            constant,
                            otherwise: 0,
                        },
                        Step::Properties(lhs, rhs, predicate) => Op::CheckProperties {
                            lhs,
                            rhs,
                            predicate,
                            otherwise: 0,
                        },
                        _ => unreachable!(),
                    };
                    exits.push((self.code.emit(op), "otherwise"));
                    exits.extend(self.search(&node.children, rules));
                }
            }
        }
        exits
    }

    fn collect_triggers(&mut self, nodes: &[Search], prefix: &mut Vec<(Step, usize)>) {
        for node in nodes {
            if let Step::Capture(rule) = node.step {
                self.triggers_for(rule, prefix);
            } else {
                prefix.push((node.step.clone(), node.scan.unwrap_or(0)));
                self.collect_triggers(&node.children, prefix);
                prefix.pop();
            }
        }
    }

    /// Derive wake-up entries and operand indexes from the exact scan plan
    /// that emitted the matcher, not a second traversal of the source pattern.
    fn triggers_for(&mut self, rule: usize, prefix: &[(Step, usize)]) {
        let types: BTreeMap<_, _> = prefix
            .iter()
            .filter_map(|(step, _)| match step {
                Step::Type(slot, types) => Some((*slot, *types)),
                _ => None,
            })
            .collect();
        let mut paths: BTreeMap<usize, Vec<Edge>> = BTreeMap::from([(0, Vec::new())]);
        let mut root = None;
        for (step, scan) in prefix {
            if let Step::Scan {
                source,
                opcode,
                bindings,
                ..
            } = step
            {
                root.get_or_insert(*opcode);
                let parent = paths[source].clone();
                let (requires_constant, constants) = input_constants(prefix, *source);
                self.triggers.push(Trigger {
                    root: root.unwrap(),
                    rules: BTreeSet::from([rule]),
                    entry: 0,
                    slot: *source,
                    event: Event::Added(*opcode),
                    scan: Some(*scan),
                    path: parent.clone(),
                    types: types[source],
                    requires_constant,
                    constants,
                    work: 0,
                });
                let slots = &bindings[0];
                for &slot in slots {
                    // Canonical argument sorting may swap physical columns,
                    // even for a symmetric pattern that needs only one binding.
                    let columns =
                        if self.commutative.contains(&self.opcodes[*opcode]) && slots.len() == 2 {
                            vec![0, 1]
                        } else {
                            bindings
                                .iter()
                                .flat_map(|binding| binding.iter().enumerate())
                                .filter_map(|(i, &s)| (s == slot).then_some(i))
                                .collect::<BTreeSet<_>>()
                                .into_iter()
                                .collect()
                        };
                    let mut path = vec![Edge {
                        opcode: *opcode,
                        columns,
                    }];
                    path.extend(parent.clone());
                    paths.insert(slot, path);
                }
            }
        }
        let root = root.expect("rule root scan");
        for (slot, path) in paths {
            // Unions can create new joins and satisfy repeated-variable checks
            // or constant predicates without adding an expression. Every bound
            // class is a dependency; conditions filter its final state.
            let (requires_constant, constants) = input_constants(prefix, slot);
            self.triggers.push(Trigger {
                root,
                rules: BTreeSet::from([rule]),
                entry: 0,
                slot,
                event: Event::ClassChanged,
                scan: None,
                path,
                types: types[&slot],
                requires_constant,
                constants,
                work: 0,
            });
        }
    }

    fn emit(&self, opcode: &str) -> String {
        let opcodes = self
            .opcodes
            .iter()
            .map(|op| format!("{opcode}::{op}"))
            .collect::<Vec<_>>()
            .join(", ");
        let mut output = String::new();
        let mut code = String::new();
        let mut reader = Reader {
            bytes: &self.code.bytes,
            pc: 0,
        };
        while reader.pc < reader.bytes.len() {
            let start = reader.pc;
            let inst = Op::read(&mut reader);
            writeln!(code, "\n        // @{start:04x} {inst:?}").unwrap();
            for byte in &reader.bytes[start..reader.pc] {
                write!(code, "{byte}, ").unwrap();
            }
        }
        let types = self
            .types
            .iter()
            .map(|t| format!("|ty| {t}"))
            .collect::<Vec<_>>()
            .join(", ");
        let bindings = self
            .bindings
            .iter()
            .enumerate()
            .map(|(id, bindings)| {
                let rows = bindings
                    .iter()
                    .map(|slots| format!("&{slots:?}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("        /* {id} */ &[{rows}],")
            })
            .collect::<Vec<_>>()
            .join("\n");
        let actions = self
            .actions
            .iter()
            .map(|action| format!("Rule {{ plan: {}, name: {:?} }}", action.plan, action.name))
            .collect::<Vec<_>>()
            .join(",\n");
        let properties = self.properties.join(",\n");
        writeln!(output, "#[rustfmt::skip]\nstatic PROGRAM: Program = Program {{\n    types: &[{types}],\n    properties: &[{properties}],\n    rules: &[{actions}],\n    opcodes: &[{opcodes}],\n    constants: &{:?},\n    bindings: &[\n{bindings}\n    ],\n    code: &[{}\n],\n}};",
            self.constants, code).unwrap();
        writeln!(output, "static TRIGGERS: &[Trigger] = &[").unwrap();
        let mut added: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut changed = Vec::new();
        let mut indexes: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
        let mut paths = Vec::new();
        for (id, trigger) in self.triggers.iter().enumerate() {
            match trigger.event {
                Event::Added(op) => added.entry(op).or_default().push(id),
                Event::ClassChanged => changed.push(id),
            }
            let path = crate::bytecode::intern(&mut paths, trigger.path.clone());
            for edge in &trigger.path {
                indexes
                    .entry(edge.opcode)
                    .or_default()
                    .extend(&edge.columns);
            }
            writeln!(output, "Trigger {{ root: {opcode}::{}, entry: {}, work: {}, slot: {}, scan: {:?}, path: PathId({path}), types: {}, requires_constant: {}, constants: &{:?} }},",
                self.opcodes[trigger.root], trigger.entry, trigger.work, trigger.slot, trigger.scan, trigger.types, trigger.requires_constant, trigger.constants).unwrap();
        }
        writeln!(output, "];").unwrap();
        writeln!(output, "pub(super) static PATHS: &[&[Edge]] = &[").unwrap();
        for path in paths {
            let edges = path
                .iter()
                .map(|edge| {
                    format!(
                        "Edge {{ opcode: {opcode}::{}, columns: &{:?} }}",
                        self.opcodes[edge.opcode], edge.columns
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(output, "&[{edges}],").unwrap();
        }
        writeln!(output, "];").unwrap();
        let trigger_ids = |ids: &[usize]| {
            ids.iter()
                .map(|id| format!("TriggerId({id})"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        writeln!(
            output,
            "pub(super) fn added(op: {opcode}) -> &'static [TriggerId] {{ match op {{"
        )
        .unwrap();
        for (op, ids) in added {
            writeln!(
                output,
                "{opcode}::{} => &[{}],",
                self.opcodes[op],
                trigger_ids(&ids)
            )
            .unwrap();
        }
        writeln!(output, "_ => &[], }} }}").unwrap();
        writeln!(
            output,
            "pub(super) const CLASS_TRIGGERS: &[TriggerId] = &[{}];",
            trigger_ids(&changed)
        )
        .unwrap();
        writeln!(
            output,
            "pub(super) fn indexed_columns(op: {opcode}) -> &'static [usize] {{ match op {{"
        )
        .unwrap();
        for (op, columns) in indexes {
            let columns: Vec<_> = columns.into_iter().collect();
            writeln!(output, "{opcode}::{} => &{:?},", self.opcodes[op], columns).unwrap();
        }
        writeln!(output, "_ => &[], }} }}").unwrap();
        output
    }

    fn constant(&mut self, value: u64) -> usize {
        crate::bytecode::intern(&mut self.constants, value)
    }

    fn opcode(&mut self, name: &str) -> usize {
        crate::bytecode::intern(&mut self.opcodes, name.to_owned())
    }

    /// Each pattern slot keeps its own type domain. Reverse triggers therefore
    /// filter a cast input by its input type, rather than the root result type.
    fn pattern(&mut self, rule: &CheckedRule, source: usize, steps: &mut Vec<Step>) {
        let pattern = &rule.pattern[source];
        let types = crate::bytecode::intern(
            &mut self.types,
            crate::types::generate::accepts(&pattern.ty.domain, "ty"),
        );
        steps.push(Step::Type(source, types));
        if let TypeSource::Value(other) = pattern.ty.source
            && other != source
        {
            steps.push(Step::Same(source, other));
        }
        match &pattern.kind {
            PatternKind::Value(previous) => {
                if *previous != source {
                    steps.push(Step::Equal(*previous, source));
                }
            }
            PatternKind::Constant(bits) => {
                let constant = self.constant(*bits);
                steps.push(Step::Constant(source, constant, true));
            }
            PatternKind::Operation {
                opcode,
                args,
                attributes,
                ..
            } => {
                for (index, attr) in attributes.iter().enumerate() {
                    let operation = self
                        .definitions
                        .ops
                        .iter()
                        .find(|o| o.name == *opcode)
                        .unwrap();
                    let lhs = crate::storage::compact::bind_attributes(
                        operation,
                        &self.definitions.storage,
                        &BTreeMap::from([(attr.name.clone(), "a".into())]),
                        "lhs",
                        "return false;",
                    );
                    let (other, predicate) = match &attr.value {
                        AttributeValue::Literal(literal) => {
                            (source, format!("|lhs, _| {{ {lhs} a == {literal} }}"))
                        }
                        AttributeValue::Binding { node, index: field }
                            if *node != source || *field != index =>
                        {
                            let PatternKind::Operation {
                                opcode: other_op,
                                attributes,
                                ..
                            } = &rule.pattern[*node].kind
                            else {
                                unreachable!()
                            };
                            let operation = self
                                .definitions
                                .ops
                                .iter()
                                .find(|o| o.name == *other_op)
                                .unwrap();
                            let rhs = crate::storage::compact::bind_attributes(
                                operation,
                                &self.definitions.storage,
                                &BTreeMap::from([(attributes[*field].name.clone(), "b".into())]),
                                "rhs",
                                "return false;",
                            );
                            (*node, format!("|lhs, rhs| {{ {lhs} {rhs} a == b }}"))
                        }
                        _ => continue,
                    };
                    let predicate = crate::bytecode::intern(&mut self.properties, predicate);
                    steps.push(Step::Properties(source, other, predicate));
                }
                let cursor = self.cursors;
                self.cursors += 1;
                let opcode = self.opcode(opcode);
                let bindings = rule
                    .orders(source)
                    .into_iter()
                    .map(|order| order.into_iter().map(|i| args[i]).collect())
                    .collect();
                steps.push(Step::Scan {
                    source,
                    opcode,
                    cursor,
                    bindings,
                });
                for &slot in args {
                    self.pattern(rule, slot, steps);
                }
            }
        }
    }
}
