//! Equality indexes, congruence rebuilding and saturation over MIR values.
use super::{Limit, matching};
use crate::evaluate::Fold;
use core::hash::BuildHasher;
use cranelift_entity::{EntityRef, SecondaryMap, packed_option::PackedOption};
use hashbrown::{HashMap, HashSet, HashTable, hash_map::DefaultHashBuilder};
use smallvec::SmallVec;
use veloc_mir::constant::ScalarConst;
use veloc_mir::function::Expressions;
use veloc_mir::{FuncBody, Inst, IntCC, Opcode as Op, Value, ValueDef};
use veloc_types::Type;

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

    /// The graph chooses the root; union-find only maintains the forest.
    fn link(&mut self, root: Value, other: Value) {
        debug_assert_ne!(root, other);
        debug_assert_eq!(self.parents[root], root.into());
        debug_assert_eq!(self.parents[other], other.into());
        self.parents[other] = root.into();
        self.sizes[root] += self.sizes[other];
    }
}

/// Queue membership is released on pop, allowing later facts to wake a task.
struct Worklist<K: EntityRef> {
    pending: Vec<K>,
    queued: SecondaryMap<K, bool>,
}

impl<K: EntityRef> Default for Worklist<K> {
    fn default() -> Self {
        Self {
            pending: Vec::new(),
            queued: SecondaryMap::new(),
        }
    }
}

impl<K: EntityRef> Worklist<K> {
    fn push(&mut self, id: K) {
        if !self.queued[id] {
            self.queued[id] = true;
            self.pending.push(id);
        }
    }

    fn pop(&mut self) -> Option<K> {
        let id = self.pending.pop()?;
        self.queued[id] = false;
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

/// Mutations are batched until congruence indexes have been repaired.
/// Query batches order events by kind (added, constant, merged), then by ID.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Change {
    Added(Inst),
    Constant(Value),
    Merged(Value),
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum InstKind {
    #[default]
    Unsupported,
    /// Can be analyzed, but cannot be moved or speculated.
    Pinned,
    Floating,
    /// Completely replaced; execution may be erased after candidate references
    /// are released. No longer participates in search or extraction.
    Folded,
}

/// All expression storage belongs to MIR. This structure holds equality and
/// dependency indexes over existing values/instructions. A class containing a
/// constant is rooted at that unique MIR literal; no separate fact table exists.
pub(super) struct Graph {
    pub(super) values: Vec<Value>,
    classes: UnionFind,
    pub(super) kinds: SecondaryMap<Inst, InstKind>,
    pub(super) users: SecondaryMap<Value, Vec<Inst>>,
    dirty_classes: Worklist<Value>,
    changes: Vec<Change>,
    // Only opcode/column pairs requested by the generated query plans.
    parents: SecondaryMap<Value, HashMap<(Op, usize), Vec<Value>>>,
    rebuild_work: Worklist<Inst>,
    // Direct MIR operand witnesses, never arbitrary union-find representatives.
    aliases: HashMap<Value, Value>,
    memo: HashTable<Inst>,
    hashes: SecondaryMap<Inst, u64>,
    hasher: DefaultHashBuilder,
    pub(super) relations: HashMap<(Value, Op), Vec<Value>>,
    class_ops: SecondaryMap<Value, Vec<Op>>,
    pub(super) limit: usize,
}

impl Graph {
    pub(super) fn new() -> Self {
        Self {
            values: Vec::new(),
            classes: UnionFind::default(),
            kinds: SecondaryMap::new(),
            users: SecondaryMap::new(),
            dirty_classes: Worklist::default(),
            rebuild_work: Worklist::default(),
            aliases: HashMap::new(),
            changes: Vec::new(),
            parents: SecondaryMap::new(),
            memo: HashTable::new(),
            hashes: SecondaryMap::new(),
            hasher: DefaultHashBuilder::default(),
            relations: HashMap::new(),
            class_ops: SecondaryMap::new(),
            limit: usize::MAX,
        }
    }

    pub(super) fn find(&self, value: Value) -> Value {
        self.classes.find(value)
    }

    pub(super) fn register_value(&mut self, f: &FuncBody, value: Value) {
        if self.classes.insert(value) {
            self.values.push(value);
            if f.dfg().as_const(value).is_some() {
                self.changes.push(Change::Constant(value));
            }
        }
    }

    pub(super) fn floating_inst(&self, f: &FuncBody, value: Value) -> Option<Inst> {
        f.dfg()
            .value_inst(value)
            .filter(|&inst| self.kinds[inst] == InstKind::Floating)
    }

    pub(super) fn args<'a>(&self, f: &'a FuncBody, value: Value) -> &'a [Value] {
        if f.dfg().as_const(self.find(value)).is_some() {
            return &[];
        }
        self.floating_inst(f, value)
            .map_or(&[], |inst| f.dfg().operands(inst))
    }

    pub(super) fn canonical_args(&self, f: &FuncBody, inst: Inst) -> SmallVec<[Value; 3]> {
        self.normalize_args(f.dfg().opcode(inst), f.dfg().operands(inst))
    }

    fn normalize_args(&self, opcode: Op, args: &[Value]) -> SmallVec<[Value; 3]> {
        let mut args: SmallVec<_> = args.iter().map(|&v| self.find(v)).collect();
        if opcode.spec().is_commutative() && args.len() == 2 && args[0] > args[1] {
            args.swap(0, 1);
        }
        args
    }

    fn key(&self, f: &FuncBody, inst: Inst) -> Key {
        let dfg = f.dfg();
        Key {
            opcode: dfg.opcode(inst),
            args: dfg.operands(inst).iter().map(|&v| self.find(v)).collect(),
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
            .find(self.hasher.hash_one(key), |&inst| {
                let mut stored = self.key(f, inst);
                stored.args = self.normalize_args(stored.opcode, &stored.args);
                stored == *key
            })
            .copied()
    }

    pub(super) fn register_inst(&mut self, f: &FuncBody, inst: Inst) {
        for &v in f
            .dfg()
            .operands(inst)
            .iter()
            .chain(f.dfg().inst_results(inst))
        {
            self.register_value(f, v);
        }
        if !can_analyze(f, inst) {
            return;
        }
        self.kinds[inst] = if f.dfg().inst(inst).can_speculate() {
            InstKind::Floating
        } else {
            InstKind::Pinned
        };
        let mut args: SmallVec<[Value; 3]> = f
            .dfg()
            .operands(inst)
            .iter()
            .map(|&v| self.find(v))
            .collect();
        args.sort_unstable();
        args.dedup();
        for arg in args {
            self.users[arg].push(inst);
        }
        if self.kinds[inst] == InstKind::Floating {
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
            self.changes.push(Change::Added(inst));
            for &column in matching::indexed_columns(opcode) {
                let arg = self.find(f.dfg().operands(inst)[column]);
                self.parents[arg]
                    .entry((opcode, column))
                    .or_default()
                    .push(result);
            }
        }
        // Import, construction and later input changes use the same reducer.
        self.rebuild_work.push(inst);
    }

    pub(super) fn union(&mut self, f: &FuncBody, a: Value, b: Value) {
        assert_eq!(
            f.dfg().value_type(a),
            f.dfg().value_type(b),
            "cannot equate different types"
        );
        let (mut a, mut b) = (self.classes.find_mut(a), self.classes.find_mut(b));
        if a == b {
            return;
        }
        let a_const = matches!(f.dfg().value_def(a), ValueDef::Const(_));
        let b_const = matches!(f.dfg().value_def(b), ValueDef::Const(_));
        assert!(!(a_const && b_const), "rewrite equated distinct constants");
        // Canonical literals are terminal roots. Other classes use union by
        // size; attaching one to a literal adds at most one final parent edge.
        if b_const || (!a_const && self.classes.sizes[a] < self.classes.sizes[b]) {
            core::mem::swap(&mut a, &mut b);
        }
        self.classes.link(a, b);
        let constant = a_const || b_const;
        for user in core::mem::take(&mut self.users[b]) {
            if self.kinds[user] == InstKind::Folded {
                continue;
            }
            self.rebuild_work.push(user);
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
        self.dirty_classes.push(a);
        for (key, values) in core::mem::take(&mut self.parents[b]) {
            self.parents[a].entry(key).or_default().extend(values);
        }
        self.changes.push(Change::Merged(a));
        if constant {
            // A known result also makes its pure producers unnecessary.
            for &opcode in &self.class_ops[a] {
                for &value in &self.relations[&(a, opcode)] {
                    let inst = f.dfg().value_inst(value).expect("indexed expression");
                    self.rebuild_work.push(inst);
                }
            }
            // The losing class's parents can now match constant predicates,
            // even when this literal was already known in another class.
            self.changes.push(Change::Constant(a));
        }
    }

    /// Drop a reduced expression from all search/hash-consing indexes. Its MIR
    /// storage stays immutable until the session ends, but is not a memo tombstone.
    fn discard(&mut self, f: &FuncBody, inst: Inst) {
        assert!(self.kinds[inst] == InstKind::Floating);
        self.kinds[inst] = InstKind::Folded;
        if let Ok(entry) = self.memo.find_entry(self.hashes[inst], |&old| old == inst) {
            entry.remove();
        }
        let node = f.dfg().inst_results(inst)[0];
        let class = self.find(node);
        let opcode = f.dfg().opcode(inst);
        let rows = self
            .relations
            .get_mut(&(class, opcode))
            .expect("indexed expression");
        rows.retain(|&v| v != node);
        if rows.is_empty() {
            self.relations.remove(&(class, opcode));
            self.class_ops[class].retain(|&op| op != opcode);
        }
        for &column in matching::indexed_columns(opcode) {
            let arg = self.find(f.dfg().operands(inst)[column]);
            let index = &mut self.parents[arg];
            if let Some(rows) = index.get_mut(&(opcode, column)) {
                rows.retain(|&v| v != node);
                if rows.is_empty() {
                    index.remove(&(opcode, column));
                }
            }
        }
        // Dependency lists are compacted in batches; queued stale work is skipped.
    }

    /// An executable replacement must follow actual operand edges, not the
    /// arbitrary representative chosen by union-by-size.
    pub(super) fn replacement(&self, f: &FuncBody, mut value: Value) -> Value {
        loop {
            let class = self.find(value);
            if f.dfg().as_const(class).is_some() {
                return class;
            }
            match self.aliases.get(&value) {
                Some(&next) => value = next,
                None => return value,
            }
        }
    }

    /// Resolve one replacement and compress its path while committing folds.
    pub(super) fn resolve_alias(&mut self, f: &FuncBody, mut value: Value) -> Value {
        let result = self.replacement(f, value);
        while let Some(next) = self.aliases.get_mut(&value) {
            if *next == result {
                break;
            }
            value = core::mem::replace(next, result);
        }
        result
    }

    /// Bounded local reasoning over a not-yet-allocated operation. Argument
    /// order is preserved so Operand(i) also identifies the original MIR input.
    fn reduce(&self, f: &FuncBody, key: &Key) -> Option<SmallVec<[Fold; 2]>> {
        if let [ty] = key.results.as_slice()
            && key.properties.is_empty()
            && let Some(fold) = matching::fold(key.opcode, *ty, &key.args, |value| {
                f.dfg().as_scalar_const(value)
            })
        {
            return Some(smallvec::smallvec![fold]);
        }
        let constants: SmallVec<[ScalarConst; 3]> = key
            .args
            .iter()
            .map(|&value| f.dfg().as_scalar_const(value))
            .collect::<Option<_>>()?;
        crate::evaluate::evaluate(key.opcode, &constants, &key.results, &key.properties)
            .map(|values| values.into_iter().map(Fold::Constant).collect())
    }

    pub(super) fn fold_to(
        &mut self,
        ir: &mut Expressions<'_>,
        value: Value,
        constant: ScalarConst,
    ) {
        // Facts are not speculative alternatives: finish publishing a fold even
        // at the node limit. Each reduction publishes only its fixed results.
        let literal = ir.constant(constant.into());
        self.register_value(ir.body(), literal);
        self.union(ir.body(), value, literal);
    }

    /// Normalize dirty expressions before exposing them to the matcher. This
    /// bounded reduction never creates operations, only literals or equalities;
    /// unlike exploratory rules it also completes when search fuel is exhausted.
    pub(super) fn rebuild(&mut self, ir: &mut Expressions<'_>) {
        while let Some(inst) = self.rebuild_work.pop() {
            if self.kinds[inst] == InstKind::Folded {
                continue;
            }
            if let Ok(entry) = self.memo.find_entry(self.hashes[inst], |&old| old == inst) {
                entry.remove();
            }
            let f = ir.body();
            let results: SmallVec<[Value; 2]> = f.dfg().inst_results(inst).into();
            let known = results
                .iter()
                .all(|&v| f.dfg().as_const(self.find(v)).is_some());
            let mut key = self.key(f, inst);
            // A known value alone cannot discharge a pinned operation's trap.
            // Evaluate its actual inputs before granting permission to erase it.
            let reduced = if known && self.kinds[inst] == InstKind::Floating {
                None
            } else {
                self.reduce(f, &key)
            };
            if (known && self.kinds[inst] == InstKind::Floating) || reduced.is_some() {
                if let Some(reduced) = reduced {
                    assert_eq!(reduced.len(), results.len(), "fold result arity");
                    for (&value, fold) in results.iter().zip(reduced) {
                        match fold {
                            Fold::Operand(index) => {
                                let operand = ir.body().dfg().operands(inst)[index];
                                let replacement = self.replacement(ir.body(), operand);
                                self.aliases.insert(value, replacement);
                                self.union(ir.body(), value, operand);
                            }
                            Fold::Constant(c) => self.fold_to(ir, value, c),
                        }
                    }
                }
                if self.kinds[inst] == InstKind::Floating {
                    self.discard(ir.body(), inst);
                } else {
                    self.kinds[inst] = InstKind::Folded;
                }
                continue;
            }
            if self.kinds[inst] != InstKind::Floating {
                continue;
            }
            // No reduction occurred, so these operand classes are still current.
            key.args = self.normalize_args(key.opcode, &key.args);
            let hash = self.hasher.hash_one(&key);
            self.hashes[inst] = hash;
            if let Some(other) = self.lookup(ir.body(), &key) {
                let outputs: SmallVec<[Value; 2]> = ir.body().dfg().inst_results(other).into();
                for (&a, &b) in results.iter().zip(&outputs) {
                    self.union(ir.body(), a, b);
                }
            } else {
                self.memo.insert_unique(hash, inst, |&i| self.hashes[i]);
            }
        }
        // Consolidate once after a wave of unions, rather than sorting the
        // growing winner list after every individual merge.
        while let Some(class) = self.dirty_classes.pop() {
            if self.find(class) == class {
                self.users[class].retain(|&inst| self.kinds[inst] != InstKind::Folded);
                self.users[class].sort_unstable();
                self.users[class].dedup();
                for rows in self.parents[class].values_mut() {
                    rows.sort_unstable();
                    rows.dedup();
                }
            }
        }
    }

    /// Resolve delta entries through the operand indexes requested by the
    /// matcher plan. Each root/opcode batch keeps all affected input positions;
    /// seeds at the same position are searched together, not as separate tasks.
    pub(super) fn schedule(&mut self, f: &FuncBody, queries: &mut Vec<matching::Query>) {
        let mut changes = core::mem::take(&mut self.changes);
        // Distinct events may become duplicates after their classes merge.
        // Normalize once at the stable query boundary, before ordering them.
        for change in &mut changes {
            match change {
                Change::Added(_) => {}
                Change::Constant(value) | Change::Merged(value) => {
                    *value = self.find(*value);
                }
            }
        }
        changes.sort_unstable();
        changes.dedup();

        let mut batches = HashMap::new();
        let mut frontier = HashSet::new();
        let mut next = HashSet::new();
        for change in changes.drain(..) {
            let (seed, entries) = match change {
                Change::Added(inst) if self.kinds[inst] == InstKind::Folded => continue,
                Change::Added(inst) => (
                    f.dfg().first_result(inst).expect("expression result"),
                    matching::added(f.dfg().opcode(inst)),
                ),
                Change::Constant(value) => (value, matching::CONSTANT_TRIGGERS),
                Change::Merged(value) => (value, matching::MERGE_TRIGGERS),
            };
            for &entry in entries {
                let trigger = matching::trigger(entry);
                self.trace_parents(seed, trigger.path, &mut frontier, &mut next);
                for &root in &frontier {
                    if !self.relations.contains_key(&(root, trigger.root)) {
                        continue;
                    }
                    let batch = *batches.entry((root, trigger.root)).or_insert_with(|| {
                        let id = queries.len();
                        queries.push(matching::Query {
                            root,
                            opcode: trigger.root,
                            inputs: Default::default(),
                        });
                        id
                    });
                    queries[batch].inputs.entry(entry).or_default().push(seed);
                }
            }
        }
        self.changes = changes;
        // Sort once after aggregation. Added seeds retain node identity, while
        // grouping by class makes constrained candidate slices cheap to locate.
        for query in queries.iter_mut() {
            for seeds in query.inputs.values_mut() {
                seeds.sort_unstable_by_key(|&v| (self.find(v), v));
                seeds.dedup();
            }
        }
        queries.sort_unstable_by_key(|query| (query.root, query.opcode as usize));
    }

    /// Follow a nested pattern outward from its changed input to query roots.
    /// Reuse both sets across triggers; each step deduplicates merged classes.
    fn trace_parents(
        &self,
        seed: Value,
        path: &[matching::Edge],
        frontier: &mut HashSet<Value>,
        next: &mut HashSet<Value>,
    ) {
        frontier.clear();
        frontier.insert(self.find(seed));
        for edge in path {
            next.clear();
            for &class in frontier.iter() {
                for &column in edge.columns {
                    if let Some(rows) = self.parents[class].get(&(edge.opcode, column)) {
                        next.extend(rows.iter().map(|&v| self.find(v)));
                    }
                }
            }
            core::mem::swap(frontier, next);
            if frontier.is_empty() {
                break;
            }
        }
    }

    pub(super) fn literal(
        &mut self,
        ir: &mut Expressions<'_>,
        value: ScalarConst,
    ) -> Result<Value, Limit> {
        let result = ir.constant(value.into());
        if self.classes.parents[result].is_none() && self.values.len() >= self.limit {
            return Err(Limit::Nodes);
        }
        self.register_value(ir.body(), result);
        Ok(result)
    }

    pub(super) fn build(
        &mut self,
        ir: &mut Expressions<'_>,
        opcode: Op,
        args: &[Value],
        ty: Type,
    ) -> Result<Value, Limit> {
        let mut key = Key {
            opcode,
            args: args.iter().map(|&v| self.find(v)).collect(),
            results: smallvec::smallvec![ty],
            properties: SmallVec::new(),
        };
        if let Some(reduced) = self.reduce(ir.body(), &key) {
            assert_eq!(reduced.len(), 1, "rule operation result arity");
            return Ok(match reduced.into_iter().next().unwrap() {
                Fold::Operand(index) => self.replacement(ir.body(), args[index]),
                Fold::Constant(c) => {
                    // A literal is a terminal answer, not a speculative node.
                    let value = ir.constant(c.into());
                    self.register_value(ir.body(), value);
                    value
                }
            });
        }
        key.args = self.normalize_args(opcode, &key.args);
        if let Some(inst) = self.lookup(ir.body(), &key) {
            return Ok(ir
                .body()
                .dfg()
                .first_result(inst)
                .expect("expression result"));
        }
        if self.values.len() >= self.limit {
            return Err(Limit::Nodes);
        }
        let inst = ir.create(
            |w| {
                w.from_values(opcode, &key.args)
                    .expect("value-only rule operation")
            },
            &[ty],
        );
        assert!(
            can_analyze(ir.body(), inst),
            "rule operation lacks a semantic recipe"
        );
        self.register_inst(ir.body(), inst);
        // Make new nodes reusable within this update batch. Existing keys made
        // stale by unions are repaired together at the next query boundary.
        let hash = self.hasher.hash_one(&key);
        self.hashes[inst] = hash;
        self.memo.insert_unique(hash, inst, |&i| self.hashes[i]);
        Ok(ir
            .body()
            .dfg()
            .first_result(inst)
            .expect("expression result"))
    }

    pub(super) fn is_idle(&self) -> bool {
        self.changes.is_empty() && self.rebuild_work.pending.is_empty()
    }
}

fn can_analyze(f: &FuncBody, inst: Inst) -> bool {
    let view = f.dfg().inst(inst);
    let results = f.dfg().inst_results(inst);
    !results.is_empty()
        && (matching::can_fold(view.opcode())
            || (results
                .iter()
                .all(|&v| ScalarConst::from_bits(f.dfg().value_type(v), 0).is_some())
                && crate::evaluate::can_fold(view.opcode())))
        && view.memory_effect().is_none()
        && !view.is_terminator()
        && !view.opcode().transfers_ownership()
}
