//! Equality indexes, congruence rebuilding and saturation over MIR values.
use super::matching::{self, RuleContext};
use core::hash::BuildHasher;
use cranelift_entity::{EntityRef, SecondaryMap, packed_option::PackedOption};
use hashbrown::{HashMap, HashSet, HashTable, hash_map::DefaultHashBuilder};
use smallvec::SmallVec;
use veloc_mir::constant::ScalarConst;
use veloc_mir::function::Expressions;
use veloc_mir::{FuncBody, Inst, IntCC, Opcode as Op, Value};
use veloc_types::{Type, TypeInfo};

/// Equivalence is an overlay on MIR value identities. MIR definitions are never
/// rewritten to union-find representatives during saturation.
#[derive(Default)]
struct UnionFind {
    parents: SecondaryMap<Value, PackedOption<Value>>,
    sizes: SecondaryMap<Value, usize>,
}

impl UnionFind {
    fn insert(&mut self, value: Value) -> bool {
        if self.parents[value].is_some() {
            return false;
        }
        self.parents[value] = value.into();
        self.sizes[value] = 1;
        true
    }

    fn find(&self, mut value: Value) -> Value {
        loop {
            let parent = self.parents[value].expect("registered MIR value");
            if parent == value {
                return value;
            }
            value = parent;
        }
    }

    fn find_mut(&mut self, mut value: Value) -> Value {
        loop {
            let parent = self.parents[value].expect("registered MIR value");
            if parent == value {
                return value;
            }
            let grandparent = self.parents[parent].expect("registered parent");
            self.parents[value] = grandparent.into();
            value = grandparent;
        }
    }

    fn union(&mut self, a: Value, b: Value) -> Option<(Value, Value)> {
        let (mut a, mut b) = (self.find_mut(a), self.find_mut(b));
        if a == b {
            return None;
        }
        if self.sizes[a] < self.sizes[b] {
            core::mem::swap(&mut a, &mut b);
        }
        self.parents[b] = a.into();
        self.sizes[a] += self.sizes[b];
        Some((a, b))
    }
}

/// Queue membership is released on pop, allowing later facts to wake a task.
struct Worklist<K: EntityRef, const BIT: u8> {
    pending: Vec<K>,
}

impl<K: EntityRef, const BIT: u8> Default for Worklist<K, BIT> {
    fn default() -> Self {
        Self {
            pending: Vec::new(),
        }
    }
}

impl<K: EntityRef, const BIT: u8> Worklist<K, BIT> {
    fn push(&mut self, queued: &mut SecondaryMap<K, u8>, id: K) {
        if queued[id] & BIT == 0 {
            queued[id] |= BIT;
            self.pending.push(id);
        }
    }

    fn pop(&mut self, queued: &mut SecondaryMap<K, u8>) -> Option<K> {
        let id = self.pending.pop()?;
        queued[id] &= !BIT;
        Some(id)
    }
}

/// A temporary lookup key, reconstructed from MIR. The hash table stores only
/// Inst IDs and cached hashes; it owns no second instruction representation.
/// Properties are precisely those exposed by the supported semantic recipes.
#[derive(PartialEq, Eq, Hash)]
struct Key {
    opcode: Op,
    args: SmallVec<[Value; 3]>,
    results: SmallVec<[Type; 2]>,
    properties: SmallVec<[IntCC; 1]>,
}

/// All expression storage belongs to MIR. This structure holds only equality,
/// analysis facts, and indexes over existing MIR values/instructions.
pub(super) struct Graph {
    pub(super) values: Vec<Value>,
    classes: UnionFind,
    pub(super) supported: SecondaryMap<Inst, bool>,
    pub(super) floating: SecondaryMap<Inst, bool>,
    // Equal constants identify the same class, without requiring a literal node.
    const_classes: HashMap<ScalarConst, Value>,
    // Constant analysis facts belong to classes and store their actual value.
    pub(super) constants: SecondaryMap<Value, Option<ScalarConst>>,
    pub(super) users: SecondaryMap<Value, Vec<Inst>>,
    value_queued: SecondaryMap<Value, u8>,
    inst_queued: SecondaryMap<Inst, u8>,
    dirty_users: Worklist<Value, 1>,
    changed_classes: Worklist<Value, 2>,
    rebuild_work: Worklist<Inst, 1>,
    rule_work: Worklist<Inst, 2>,
    analysis_work: Worklist<Inst, 4>,
    memo: HashTable<Inst>,
    hashes: SecondaryMap<Inst, u64>,
    hasher: DefaultHashBuilder,
    pub(super) relations: HashMap<(Value, Op), Vec<Value>>,
    class_ops: SecondaryMap<Value, Vec<Op>>,
    pub(super) revision: usize,
    pub(super) limit: usize,
}

impl Graph {
    pub(super) fn new() -> Self {
        Self {
            values: Vec::new(),
            classes: UnionFind::default(),
            supported: SecondaryMap::new(),
            floating: SecondaryMap::new(),
            const_classes: HashMap::new(),
            constants: SecondaryMap::new(),
            users: SecondaryMap::new(),
            value_queued: SecondaryMap::new(),
            inst_queued: SecondaryMap::new(),
            dirty_users: Worklist::default(),
            rebuild_work: Worklist::default(),
            rule_work: Worklist::default(),
            analysis_work: Worklist::default(),
            changed_classes: Worklist::default(),
            memo: HashTable::new(),
            hashes: SecondaryMap::new(),
            hasher: DefaultHashBuilder::default(),
            relations: HashMap::new(),
            class_ops: SecondaryMap::new(),
            revision: 0,
            limit: usize::MAX,
        }
    }

    pub(super) fn find(&self, value: Value) -> Value {
        self.classes.find(value)
    }

    pub(super) fn register_value(&mut self, value: Value) {
        if self.classes.insert(value) {
            self.values.push(value);
        }
    }

    pub(super) fn floating_inst(&self, f: &FuncBody, value: Value) -> Option<Inst> {
        f.dfg()
            .value_inst(value)
            .filter(|&inst| self.floating[inst])
    }

    pub(super) fn args<'a>(&self, f: &'a FuncBody, value: Value) -> &'a [Value] {
        if self.constant(value).is_some() {
            return &[];
        }
        self.floating_inst(f, value)
            .map_or(&[], |inst| f.dfg().operands(inst))
    }

    pub(super) fn canonical_args(&self, f: &FuncBody, inst: Inst) -> SmallVec<[Value; 3]> {
        let mut args: SmallVec<_> = f
            .dfg()
            .operands(inst)
            .iter()
            .map(|&v| self.find(v))
            .collect();
        if f.dfg().opcode(inst).spec().is_commutative() && args.len() == 2 && args[0] > args[1] {
            args.swap(0, 1);
        }
        args
    }

    fn key(&self, f: &FuncBody, inst: Inst) -> Key {
        let dfg = f.dfg();
        Key {
            opcode: dfg.opcode(inst),
            args: self.canonical_args(f, inst),
            results: dfg
                .inst_results(inst)
                .iter()
                .map(|&v| dfg.value_type(v))
                .collect(),
            properties: crate::evaluate::properties(&dfg.inst(inst)),
        }
    }

    fn lookup(&self, f: &FuncBody, key: &Key) -> Option<Inst> {
        self.memo
            .find(self.hasher.hash_one(key), |&inst| self.key(f, inst) == *key)
            .copied()
    }

    pub(super) fn register_inst(&mut self, f: &FuncBody, inst: Inst) {
        for &v in f
            .dfg()
            .operands(inst)
            .iter()
            .chain(f.dfg().inst_results(inst))
        {
            self.register_value(v);
        }
        if !candidate(f, inst) {
            return;
        }
        self.supported[inst] = true;
        self.floating[inst] = f.dfg().inst(inst).can_speculate();
        if let [value] = f.dfg().inst_results(inst)
            && let Some(literal) = f.dfg().as_scalar_const(*value)
        {
            self.set_const(f, *value, literal);
            return;
        }
        let mut args = self.canonical_args(f, inst);
        args.sort_unstable();
        args.dedup();
        for arg in args {
            self.users[arg].push(inst);
        }
        if self.floating[inst] {
            let result = f.dfg().first_result(inst).expect("expression result");
            let class = self.find(result);
            let opcode = f.dfg().opcode(inst);
            let row = self.relations.entry((class, opcode)).or_default();
            if row.is_empty() {
                self.class_ops[class].push(opcode);
            }
            row.push(result);
            // A new alternative can satisfy a nested pattern in an existing
            // parent even when no operand or constant fact changes.
            self.changed_classes.push(&mut self.value_queued, class);
            self.rebuild_work.push(&mut self.inst_queued, inst);
            self.rule_work.push(&mut self.inst_queued, inst);
        }
        self.analysis_work.push(&mut self.inst_queued, inst);
        self.revision += 1;
    }

    pub(super) fn union(&mut self, f: &FuncBody, a: Value, b: Value) {
        assert_eq!(
            f.dfg().value_type(a),
            f.dfg().value_type(b),
            "cannot equate different types"
        );
        let Some((a, b)) = self.classes.union(a, b) else {
            return;
        };
        if let (Some(x), Some(y)) = (self.constants[a], self.constants[b]) {
            assert_eq!(x, y, "rewrite equated distinct constants");
        }
        let a_const = self.constants[a];
        let b_const = self.constants[b];
        // Only users on the side that just learned the constant need another
        // analysis pass. Keep the two complementary cases explicit here.
        let retry_a = a_const.is_none() && b_const.is_some();
        let retry_b = a_const.is_some() && b_const.is_none();
        self.constants[a] = a_const.or(b_const);
        if retry_a {
            for &user in &self.users[a] {
                self.analysis_work.push(&mut self.inst_queued, user);
            }
        }
        for user in core::mem::take(&mut self.users[b]) {
            if self.floating[user] {
                self.rebuild_work.push(&mut self.inst_queued, user);
            }
            if retry_b {
                self.analysis_work.push(&mut self.inst_queued, user);
            }
            self.users[a].push(user);
        }
        for opcode in core::mem::take(&mut self.class_ops[b]) {
            let source = self.relations.remove(&(b, opcode)).expect("indexed opcode");
            let target = self.relations.entry((a, opcode)).or_default();
            if target.is_empty() {
                self.class_ops[a].push(opcode);
            }
            target.extend(source);
        }
        self.dirty_users.push(&mut self.value_queued, a);
        self.changed_classes.push(&mut self.value_queued, a);
        self.revision += 1;
    }

    pub(super) fn constant(&self, value: Value) -> Option<ScalarConst> {
        self.constants[self.find(value)]
    }

    pub(super) fn set_const(&mut self, f: &FuncBody, value: Value, constant: ScalarConst) {
        assert_eq!(
            f.dfg().value_type(value),
            constant.ty(),
            "constant fact type"
        );
        let class = self.find(value);
        if let Some(old) = self.constants[class] {
            assert_eq!(old, constant, "inconsistent constant class");
            return;
        }
        self.constants[class] = Some(constant);
        for &user in &self.users[class] {
            self.analysis_work.push(&mut self.inst_queued, user);
        }
        self.changed_classes.push(&mut self.value_queued, class);
        self.revision += 1;
        if let Some(&other) = self.const_classes.get(&constant) {
            self.union(f, class, other);
        } else {
            self.const_classes.insert(constant, class);
        }
    }

    /// Repair canonical hashes without rewriting MIR operand edges.
    pub(super) fn rebuild(&mut self, f: &FuncBody) {
        while let Some(inst) = self.rebuild_work.pop(&mut self.inst_queued) {
            if let Ok(entry) = self.memo.find_entry(self.hashes[inst], |&old| old == inst) {
                entry.remove();
            }
            let key = self.key(f, inst);
            let hash = self.hasher.hash_one(&key);
            self.hashes[inst] = hash;
            if let Some(other) = self.lookup(f, &key) {
                for (&a, &b) in f
                    .dfg()
                    .inst_results(inst)
                    .iter()
                    .zip(f.dfg().inst_results(other))
                {
                    self.union(f, a, b);
                }
            } else {
                self.memo.insert_unique(hash, inst, |&i| self.hashes[i]);
            }
            self.rule_work.push(&mut self.inst_queued, inst);
        }
        // Consolidate once after a wave of unions, rather than sorting the
        // growing winner list after every individual merge.
        while let Some(class) = self.dirty_users.pop(&mut self.value_queued) {
            if self.find(class) == class {
                self.users[class].sort_unstable();
                self.users[class].dedup();
            }
        }
    }

    /// Follow only ancestor paths present in rule patterns. Class changes are
    /// conservative (union, new alternative, or constant fact), but unrelated
    /// user chains never enter the rule worklist.
    fn wake_rules(&mut self, f: &FuncBody) {
        let mut seen = HashSet::new();
        let mut frontier = HashSet::new();
        let mut next = HashSet::new();
        while let Some(value) = self.changed_classes.pop(&mut self.value_queued) {
            let class = self.find(value);
            if !seen.insert(class) {
                continue;
            }
            for dependency in matching::dependencies() {
                frontier.clear();
                frontier.insert(class);
                for &opcode in dependency.path {
                    next.clear();
                    for &value in &frontier {
                        for &user in &self.users[value] {
                            if self.floating[user] && f.dfg().opcode(user) == opcode {
                                for &result in f.dfg().inst_results(user) {
                                    next.insert(self.find(result));
                                }
                            }
                        }
                    }
                    std::mem::swap(&mut frontier, &mut next);
                    if frontier.is_empty() {
                        break;
                    }
                }
                for &class in &frontier {
                    if let Some(rows) = self.relations.get(&(class, dependency.root)) {
                        for &value in rows {
                            self.rule_work.push(
                                &mut self.inst_queued,
                                f.dfg().value_inst(value).expect("relation result"),
                            );
                        }
                    }
                }
            }
        }
    }

    pub(super) fn literal(
        &mut self,
        ir: &mut Expressions<'_>,
        value: ScalarConst,
    ) -> Option<Value> {
        if let Some(&class) = self.const_classes.get(&value) {
            return Some(self.find(class));
        }
        if self.values.len() >= self.limit {
            return None;
        }
        let inst = ir.create(|w| w.scalar_const(value), &[value.ty()]);
        self.register_inst(ir.body(), inst);
        ir.body().dfg().first_result(inst)
    }

    pub(super) fn build(
        &mut self,
        ir: &mut Expressions<'_>,
        opcode: Op,
        args: &[Value],
        ty: Type,
    ) -> Option<Value> {
        let mut args: SmallVec<[Value; 3]> = args.iter().map(|&v| self.find(v)).collect();
        if opcode.spec().is_commutative() && args.len() == 2 && args[0] > args[1] {
            args.swap(0, 1);
        }
        let key = Key {
            opcode,
            args,
            results: smallvec::smallvec![ty],
            properties: SmallVec::new(),
        };
        if let Some(inst) = self.lookup(ir.body(), &key) {
            return ir.body().dfg().first_result(inst);
        }
        if self.values.len() >= self.limit {
            return None;
        }
        let inst = ir.create(
            |w| {
                w.from_values(opcode, &key.args)
                    .expect("value-only rule operation")
            },
            &[ty],
        );
        assert!(
            candidate(ir.body(), inst),
            "rule operation lacks a semantic recipe"
        );
        self.register_inst(ir.body(), inst);
        self.rebuild(ir.body());
        ir.body().dfg().first_result(inst)
    }

    pub(super) fn saturate(&mut self, ir: &mut Expressions<'_>, rounds: usize, fuel: &mut usize) {
        let mut matcher = matching::Machine::new();
        let mut searched = HashSet::new();
        for _ in 0..rounds {
            let before = self.revision;
            self.fold_constants(ir, fuel);
            self.rebuild(ir.body());
            self.wake_rules(ir.body());
            searched.clear();
            while *fuel > 0 {
                let Some(inst) = self.rule_work.pop(&mut self.inst_queued) else {
                    break;
                };
                *fuel -= 1;
                if !self.floating[inst] {
                    continue;
                }
                let f = ir.body();
                let [result] = f.dfg().inst_results(inst) else {
                    continue;
                };
                let result = *result;
                // Constant classes are settled for MIR simplification. Retain
                // their expressions for nested matches, but stop expanding them.
                if self.constants[self.find(result)].is_some() {
                    continue;
                }
                let ty = f.dfg().value_type(result);
                let opcode = f.dfg().opcode(inst);
                if !ty.is_integer() && ty != Type::BOOL {
                    continue;
                }
                // Each search scans the entire class/opcode relation. Multiple
                // concrete instructions must not repeat an unchanged query.
                let query = (self.find(result), opcode);
                if !searched.insert(query) {
                    continue;
                }
                let mask = u64::MAX >> (64 - ty.element_bits().unwrap());
                let mut context = RuleContext {
                    graph: self,
                    ir,
                    ty,
                };
                matcher.run(&mut context, opcode, result, mask, fuel);
            }
            self.fold_constants(ir, fuel);
            if self.revision == before || *fuel == 0 {
                break;
            }
        }
        self.rebuild(ir.body());
    }

    fn fold_constants(&mut self, ir: &mut Expressions<'_>, fuel: &mut usize) {
        while *fuel > 0 {
            let Some(inst) = self.analysis_work.pop(&mut self.inst_queued) else {
                break;
            };
            *fuel -= 1;
            let f = ir.body();
            if f.dfg()
                .inst_results(inst)
                .iter()
                .all(|&v| self.constants[self.find(v)].is_some())
            {
                continue;
            }
            let folded = crate::evaluate::fold(f.dfg(), inst, |v| self.constant(v));
            if let Some(constants) = folded {
                for (&value, constant) in f.dfg().inst_results(inst).iter().zip(constants) {
                    self.set_const(f, value, constant);
                }
            }
        }
    }
}

fn candidate(f: &FuncBody, inst: Inst) -> bool {
    let view = f.dfg().inst(inst);
    let results = f.dfg().inst_results(inst);
    !results.is_empty()
        && results
            .iter()
            .all(|&v| ScalarConst::from_bits(f.dfg().value_type(v), 0).is_some())
        && (results.len() == 1 && f.dfg().as_scalar_const(results[0]).is_some()
            || crate::evaluate::can_fold(view.opcode()))
        && view.memory_effect().is_none()
        && !view.is_terminator()
        && !view.opcode().transfers_ownership()
}
