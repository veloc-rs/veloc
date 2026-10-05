//! Equality indexes, congruence rebuilding and saturation over search values.
use super::storage::{Expressions, Inst, Value};
use super::{Limit, matching};
use crate::evaluate::{Fold, Properties};
use core::hash::BuildHasher;
use cranelift_entity::{EntityRef, PrimaryMap, SecondaryMap, packed_option::PackedOption};
use hashbrown::{HashMap, HashTable, hash_map::DefaultHashBuilder};
use smallvec::SmallVec;
use veloc_mir::Opcode as Op;
use veloc_mir::constant::ScalarConst;
use veloc_types::Type;

/// An equivalence-class identity, not an executable operand witness. A saved
/// root may cease to be canonical after a union; normalize it before reuse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub(super) struct Root(Value);

impl Root {
    /// Expose search identity explicitly for metadata, constants or candidates. This does not establish dominance at an executable use.
    pub(super) fn value(self) -> Value {
        self.0
    }
}

impl EntityRef for Root {
    fn new(index: usize) -> Self {
        Self(Value::new(index))
    }

    fn index(self) -> usize {
        self.0.index()
    }
}

/// Equivalence is an overlay on search value identities. Definitions are never
/// rewritten to union-find representatives during saturation.
#[derive(Default)]
struct UnionFind {
    parents: SecondaryMap<Value, PackedOption<Value>>,
    sizes: SecondaryMap<Root, usize>,
}

impl UnionFind {
    fn insert(&mut self, value: Value) -> bool {
        if self.parents[value].is_some() {
            return false;
        }
        self.parents[value] = value.into();
        self.sizes[Root(value)] = 1;
        true
    }

    fn find(&self, mut value: Value) -> Root {
        loop {
            let parent = self.parents[value].expect("registered search value");
            if parent == value {
                return Root(value);
            }
            value = parent;
        }
    }

    fn find_mut(&mut self, mut value: Value) -> Root {
        loop {
            let parent = self.parents[value].expect("registered search value");
            if parent == value {
                return Root(value);
            }
            let grandparent = self.parents[parent].expect("registered parent");
            self.parents[value] = grandparent.into();
            value = grandparent;
        }
    }

    /// The graph chooses the root; union-find only maintains the forest.
    fn link(&mut self, root: Root, other: Root) {
        debug_assert_ne!(root, other);
        debug_assert_eq!(self.parents[root.0], root.0.into());
        debug_assert_eq!(self.parents[other.0], other.0.into());
        self.parents[other.0] = root.0.into();
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

/// A temporary lookup key, reconstructed from search storage. The hash table stores only
/// Inst IDs and cached hashes; it owns no second instruction representation.
/// Properties are precisely those exposed by the supported semantic recipes.
#[derive(PartialEq, Eq, Hash)]
struct Key {
    opcode: Op,
    args: SmallVec<[Root; 3]>,
    results: SmallVec<[Type; 2]>,
    properties: Properties,
}

/// Mutations are batched until congruence indexes have been repaired.
/// Query batches order events by kind (added, class changed), then by ID.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Change {
    Added(Inst),
    ClassChanged(Root),
    /// Consumed by index compaction before scheduling queries.
    Folded(Inst),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum InstKind {
    /// Can be analyzed, but cannot be moved or speculated.
    Pinned,
    Floating,
    /// Completely replaced; execution may be erased after candidate references
    /// are released. No longer participates in search or extraction.
    Folded,
}

/// Indexes owned by one canonical class. During rebuilding, lists may contain
/// folded instructions; compaction restores live, unique entries before queries.
#[derive(Clone, Default)]
struct ClassIndex {
    users: Vec<Inst>,
    expressions: HashMap<Op, Vec<Value>>,
    // Only opcode/column pairs requested by the generated query plans.
    parents: HashMap<(Op, usize), Vec<Value>>,
}

impl ClassIndex {
    fn merge(&mut self, other: Self) {
        self.users.extend(other.users);
        for (opcode, values) in other.expressions {
            self.expressions.entry(opcode).or_default().extend(values);
        }
        for (key, values) in other.parents {
            self.parents.entry(key).or_default().extend(values);
        }
    }

    fn compact(&mut self, f: &Expressions, kinds: &PrimaryMap<Inst, InstKind>) {
        self.users.retain(|&inst| kinds[inst] != InstKind::Folded);
        self.users.sort_unstable();
        self.users.dedup();
        let live = |&value: &Value| {
            let inst = f.value_inst(value).expect("indexed expression");
            kinds[inst] != InstKind::Folded
        };
        // Each expression is registered once and moves with its class. Keep
        // its insertion order; unlike parent rows, these rows cannot overlap.
        self.expressions.retain(|_, rows| {
            rows.retain(live);
            !rows.is_empty()
        });
        self.parents.retain(|_, rows| {
            rows.retain(live);
            rows.sort_unstable();
            rows.dedup();
            !rows.is_empty()
        });
    }
}

/// Compact operations share one representation for imports and candidates. This
/// structure holds equality and dependency indexes over them. A class
/// containing a constant is rooted at its interned literal.
pub(super) struct Graph {
    pub(super) profile: crate::Profile,
    pub(super) layout: Option<veloc_types::DataLayout>,
    pub(super) values: Vec<Value>,
    classes: UnionFind,
    pub(super) kinds: PrimaryMap<Inst, InstKind>,
    indexes: SecondaryMap<Root, ClassIndex>,
    changes: Vec<Change>,
    rebuild_work: Worklist<Inst>,
    // Direct operand witnesses, never arbitrary union-find representatives.
    aliases: HashMap<Value, Value>,
    memo: HashTable<Inst>,
    hashes: SecondaryMap<Inst, u64>,
    hasher: DefaultHashBuilder,
    /// Candidate instructions still allowed; literals and import never consume it.
    pub(super) remaining_nodes: usize,
}

impl Graph {
    pub(super) fn new() -> Self {
        Self {
            profile: crate::Profile::default(),
            layout: None,
            values: Vec::new(),
            classes: UnionFind::default(),
            kinds: PrimaryMap::new(),
            indexes: SecondaryMap::new(),
            rebuild_work: Worklist::default(),
            aliases: HashMap::new(),
            changes: Vec::new(),
            memo: HashTable::new(),
            hashes: SecondaryMap::new(),
            hasher: DefaultHashBuilder::default(),
            remaining_nodes: usize::MAX,
        }
    }

    pub(super) fn find(&self, value: Value) -> Root {
        self.classes.find(value)
    }

    pub(super) fn canonicalize(&self, root: Root) -> Root {
        self.classes.find(root.0)
    }

    pub(super) fn alternatives(&self, class: Root, opcode: Op) -> &[Value] {
        self.indexes[class]
            .expressions
            .get(&opcode)
            .map_or(&[], Vec::as_slice)
    }

    pub(super) fn users(&self, class: Root) -> &[Inst] {
        &self.indexes[class].users
    }

    pub(super) fn register_value(&mut self, f: &Expressions, value: Value) {
        if self.classes.insert(value) {
            self.values.push(value);
            if f.as_const(value).is_some() {
                self.changes.push(Change::ClassChanged(Root(value)));
            }
        }
    }

    pub(super) fn floating_inst(&self, f: &Expressions, value: Value) -> Option<Inst> {
        f.value_inst(value)
            .filter(|&inst| self.kinds[inst] == InstKind::Floating)
    }

    pub(super) fn args<'a>(&self, f: &'a Expressions<'_>, value: Value) -> &'a [Value] {
        if f.as_const(self.find(value).value()).is_some() {
            return &[];
        }
        self.floating_inst(f, value)
            .map_or(&[], |inst| f.operands(inst))
    }

    pub(super) fn canonical_args(&self, f: &Expressions, inst: Inst) -> SmallVec<[Root; 3]> {
        let mut args: SmallVec<[Root; 3]> =
            f.operands(inst).iter().map(|&v| self.find(v)).collect();
        Self::order_args(f.opcode(inst), &mut args);
        args
    }

    /// The caller has resolved every root against this immutable graph.
    fn order_args(opcode: Op, args: &mut [Root]) {
        if opcode.spec().is_commutative() && args.len() == 2 && args[0] > args[1] {
            args.swap(0, 1);
        }
    }

    fn key(&self, f: &Expressions, inst: Inst) -> Key {
        Key {
            opcode: f.opcode(inst),
            args: f.operands(inst).iter().map(|&v| self.find(v)).collect(),
            results: f
                .inst_results(inst)
                .iter()
                .map(|&v| f.value_type(v))
                .collect(),
            properties: f.properties(inst),
        }
    }

    fn lookup(&self, f: &Expressions, key: &Key) -> Option<Inst> {
        self.memo
            .find(self.hasher.hash_one(key), |&inst| {
                let mut stored = self.key(f, inst);
                Self::order_args(stored.opcode, &mut stored.args);
                stored == *key
            })
            .copied()
    }

    pub(super) fn register_inst(&mut self, f: &Expressions, inst: Inst) {
        for &v in f.operands(inst).iter().chain(f.inst_results(inst).iter()) {
            self.register_value(f, v);
        }
        let kind = if f.can_speculate(inst) {
            InstKind::Floating
        } else {
            InstKind::Pinned
        };
        assert_eq!(
            self.kinds.push(kind),
            inst,
            "register expressions in storage order"
        );
        let mut args: SmallVec<[Root; 3]> =
            f.operands(inst).iter().map(|&v| self.find(v)).collect();
        args.sort_unstable();
        args.dedup();
        for arg in args {
            self.indexes[arg].users.push(inst);
        }
        if self.kinds[inst] == InstKind::Floating {
            let result = f.first_result(inst).expect("expression result");
            let class = self.find(result);
            let opcode = f.opcode(inst);
            self.indexes[class]
                .expressions
                .entry(opcode)
                .or_default()
                .push(result);
            // A new alternative can satisfy a nested pattern in an existing
            // parent even when no operand or constant fact changes.
            self.changes.push(Change::Added(inst));
            for &column in matching::indexed_columns(opcode) {
                let arg = self.find(f.operands(inst)[column]);
                self.indexes[arg]
                    .parents
                    .entry((opcode, column))
                    .or_default()
                    .push(result);
            }
        }
        // Import, construction and later input changes use the same reducer.
        self.rebuild_work.push(inst);
    }

    pub(super) fn union(&mut self, f: &Expressions, a: Value, b: Value) {
        assert_eq!(
            f.value_type(a),
            f.value_type(b),
            "cannot equate different types"
        );
        let (mut a, mut b) = (self.classes.find_mut(a), self.classes.find_mut(b));
        if a == b {
            return;
        }
        let a_const = f.as_const(a.value()).is_some();
        let b_const = f.as_const(b.value()).is_some();
        assert!(!(a_const && b_const), "rewrite equated distinct constants");
        // Canonical literals are terminal roots. Other classes use union by
        // size; attaching one to a literal adds at most one final parent edge.
        if b_const || (!a_const && self.classes.sizes[a] < self.classes.sizes[b]) {
            core::mem::swap(&mut a, &mut b);
        }
        self.classes.link(a, b);
        let constant = a_const || b_const;
        let source = core::mem::take(&mut self.indexes[b]);
        for &user in &source.users {
            if self.kinds[user] == InstKind::Folded {
                continue;
            }
            self.rebuild_work.push(user);
        }
        self.indexes[a].merge(source);
        // One event covers both new equalities and any resulting constant fact.
        self.changes.push(Change::ClassChanged(a));
        if constant {
            // A known result also makes its pure producers unnecessary.
            for values in self.indexes[a].expressions.values() {
                for &value in values {
                    let inst = f.value_inst(value).expect("indexed expression");
                    if self.kinds[inst] != InstKind::Folded {
                        self.rebuild_work.push(inst);
                    }
                }
            }
        }
    }

    /// An executable replacement must follow actual operand edges, not the
    /// arbitrary representative chosen by union-by-size.
    pub(super) fn replacement(&self, f: &Expressions, mut value: Value) -> Value {
        loop {
            let class = self.find(value);
            if f.as_const(class.value()).is_some() {
                return class.value();
            }
            match self.aliases.get(&value) {
                Some(&next) => value = next,
                None => return value,
            }
        }
    }

    /// Resolve one replacement and compress its path while planning folds.
    pub(super) fn resolve_alias(&mut self, f: &Expressions, mut value: Value) -> Value {
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
    /// order is preserved so Operand(i) also identifies the source input.
    fn reduce(&self, f: &Expressions, key: &Key) -> Option<SmallVec<[Fold; 2]>> {
        let args: SmallVec<[Value; 3]> = key.args.iter().map(|root| root.value()).collect();
        crate::evaluate::reduce(key.opcode, &args, &key.results, &key.properties, |value| {
            f.as_scalar_const(value)
        })
    }

    pub(super) fn fold_to(&mut self, ir: &mut Expressions, value: Value, constant: ScalarConst) {
        // Facts are not speculative alternatives: finish publishing a fold even
        // at the node limit. Each reduction publishes only its fixed results.
        let literal = ir.constant(constant.into());
        self.register_value(ir, literal);
        self.union(ir, value, literal);
    }

    fn try_fold(&mut self, ir: &mut Expressions, inst: Inst, key: &Key) -> bool {
        let results: SmallVec<[Value; 2]> = ir.inst_results(inst).iter().copied().collect();
        // A known value alone cannot discharge a pinned operation's trap.
        // Evaluate its actual inputs before granting permission to erase it.
        let known = self.kinds[inst] == InstKind::Floating
            && results
                .iter()
                .all(|&v| ir.as_const(self.find(v).value()).is_some());
        if !known {
            let Some(reduced) = self.reduce(ir, key) else {
                return false;
            };
            assert_eq!(reduced.len(), results.len(), "fold result arity");
            for (&value, fold) in results.iter().zip(reduced) {
                match fold {
                    Fold::Operand(index) => {
                        let operand = ir.operands(inst)[index];
                        let replacement = self.replacement(ir, operand);
                        self.aliases.insert(value, replacement);
                        self.union(ir, value, operand);
                    }
                    Fold::Constant(c) => self.fold_to(ir, value, c),
                }
            }
        }
        // Rebuilding already removed the memo entry. Lists are compacted once
        // the entire wave has settled; search storage stays alive through extraction.
        self.kinds[inst] = InstKind::Folded;
        self.changes.push(Change::Folded(inst));
        true
    }

    /// Reduce dirty expressions before matching. Allocation-free folds finish
    /// independently of the query and node budgets. Nested rules run in queries.
    pub(super) fn rebuild(&mut self, ir: &mut Expressions) {
        let scope = self.profile.scope("egraph.rebuild", 0);
        while let Some(inst) = self.rebuild_work.pop() {
            if self.kinds[inst] == InstKind::Folded {
                continue;
            }
            if let Ok(entry) = self.memo.find_entry(self.hashes[inst], |&old| old == inst) {
                entry.remove();
            }
            let mut key = self.key(ir, inst);
            if self.try_fold(ir, inst, &key) {
                continue;
            }
            if self.kinds[inst] != InstKind::Floating {
                continue;
            }
            Self::order_args(key.opcode, &mut key.args);
            let hash = self.hasher.hash_one(&key);
            self.hashes[inst] = hash;
            if let Some(other) = self.lookup(ir, &key) {
                let results: SmallVec<[Value; 2]> = ir.inst_results(inst).iter().copied().collect();
                let outputs: SmallVec<[Value; 2]> =
                    ir.inst_results(other).iter().copied().collect();
                for (&a, &b) in results.iter().zip(&outputs) {
                    self.union(ir, a, b);
                }
            } else {
                self.memo.insert_unique(hash, inst, |&i| self.hashes[i]);
            }
        }
        self.compact_indexes(ir);
        scope.success();
    }

    /// Derive cleanup work from mutations at the stable rebuild boundary. Only
    /// matching events survive, including when the caller stops at a budget limit.
    fn compact_indexes(&mut self, f: &Expressions) {
        let mut affected = Vec::new();
        self.changes.retain_mut(|change| match change {
            Change::Added(inst) => self.kinds[*inst] != InstKind::Folded,
            Change::ClassChanged(class) => {
                *class = self.classes.find(class.value());
                affected.push(*class);
                true
            }
            Change::Folded(inst) => {
                // Operand classes own users/parents; result classes own the
                // expressions. Either may have merged since the fold occurred.
                affected.extend(
                    f.operands(*inst)
                        .iter()
                        .chain(f.inst_results(*inst).iter())
                        .map(|&value| self.classes.find(value)),
                );
                false
            }
        });
        self.changes.sort_unstable();
        self.changes.dedup();
        affected.sort_unstable();
        affected.dedup();
        for class in affected {
            self.indexes[class].compact(f, &self.kinds);
        }
    }

    /// Resolve delta entries through the operand indexes requested by the
    /// matcher plan. Each root/opcode batch keeps all affected input positions;
    /// seeds at the same position are searched together, not as separate tasks.
    pub(super) fn schedule(&mut self, f: &Expressions, queries: &mut Vec<matching::Query>) {
        let scope = self.profile.scope("egraph.schedule", 0);
        let mut changes = core::mem::take(&mut self.changes);
        let mut batches = HashMap::new();
        let mut frontier = SmallVec::<[Root; 8]>::new();
        let mut next = SmallVec::<[Root; 8]>::new();
        // One entry per generated path, reused across seeds in this immutable
        // scheduling phase. No runtime hashing of path descriptions is needed.
        let mut paths = vec![(None, SmallVec::<[Root; 8]>::new()); matching::PATHS.len()];
        for change in changes.drain(..) {
            let (seed, entries) = match change {
                Change::Added(inst) => (
                    matching::Seed::Added(f.first_result(inst).expect("expression result")),
                    matching::added(f.opcode(inst)),
                ),
                Change::ClassChanged(root) => {
                    (matching::Seed::Class(root), matching::CLASS_TRIGGERS)
                }
                Change::Folded(_) => unreachable!("folded events are consumed by rebuild"),
            };
            let class = seed.root(self);
            for &entry in entries {
                let trigger = matching::trigger(entry);
                if !trigger.accepts(f, class) {
                    continue;
                }
                // Different rule inputs often walk the same reverse path from
                // this seed. Cache the roots independently of their predicates.
                let (seed_class, roots) = &mut paths[trigger.path.index()];
                if *seed_class != Some(class) {
                    self.trace_parents(
                        class,
                        matching::PATHS[trigger.path.index()],
                        &mut frontier,
                        &mut next,
                    );
                    roots.clear();
                    roots.extend_from_slice(&frontier);
                    *seed_class = Some(class);
                }
                for &root in roots.iter() {
                    if self.alternatives(root, trigger.root).is_empty() {
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
                seeds.sort_unstable_by_key(|&seed| (seed.root(self), seed));
                seeds.dedup();
            }
        }
        queries.sort_unstable_by_key(|query| (query.root, query.opcode as usize));
        self.profile.count("queries", queries.len() as u64);
        scope.success();
    }

    /// Follow a nested pattern outward from its changed input to query roots.
    /// Most reverse paths have few parents. Reuse compact buffers and deduplicate
    /// after each edge instead of hashing every intermediate class.
    fn trace_parents(
        &self,
        seed: Root,
        path: &[matching::Edge],
        frontier: &mut SmallVec<[Root; 8]>,
        next: &mut SmallVec<[Root; 8]>,
    ) {
        frontier.clear();
        frontier.push(seed);
        for edge in path {
            next.clear();
            for &class in frontier.iter() {
                for &column in edge.columns {
                    if let Some(rows) = self.indexes[class].parents.get(&(edge.opcode, column)) {
                        next.extend(rows.iter().map(|&v| self.find(v)));
                    }
                }
            }
            next.sort_unstable();
            next.dedup();
            core::mem::swap(frontier, next);
            if frontier.is_empty() {
                break;
            }
        }
    }

    pub(super) fn literal(&mut self, ir: &mut Expressions, value: ScalarConst) -> Value {
        let result = ir.constant(value.into());
        self.register_value(ir, result);
        result
    }

    pub(super) fn build(
        &mut self,
        ir: &mut Expressions,
        opcode: Op,
        args: &[Value],
        ty: Type,
        properties: Properties,
    ) -> Result<Value, Limit> {
        let mut key = Key {
            opcode,
            args: args.iter().map(|&v| self.find(v)).collect(),
            results: smallvec::smallvec![ty],
            properties,
        };
        if let Some(reduced) = self.reduce(ir, &key) {
            assert_eq!(reduced.len(), 1, "rule operation result arity");
            return Ok(match reduced.into_iter().next().unwrap() {
                Fold::Operand(index) => self.replacement(ir, args[index]),
                Fold::Constant(c) => {
                    // A literal is a terminal answer, not a speculative node.
                    self.literal(ir, c)
                }
            });
        }
        Self::order_args(opcode, &mut key.args);
        if let Some(inst) = self.lookup(ir, &key) {
            return Ok(ir.first_result(inst).expect("expression result"));
        }
        if self.remaining_nodes == 0 {
            return Err(Limit::Nodes);
        }
        // Detached candidates may use representatives; executable operands are
        // selected separately under dominance checks during extraction.
        let args: SmallVec<[Value; 3]> = key.args.iter().map(|root| root.value()).collect();
        let inst = ir.create(opcode, &args, ty, properties);
        self.remaining_nodes -= 1;
        self.register_inst(ir, inst);
        // Make new nodes reusable within this update batch. Existing keys made
        // stale by unions are repaired together at the next query boundary.
        let hash = self.hasher.hash_one(&key);
        self.hashes[inst] = hash;
        self.memo.insert_unique(hash, inst, |&i| self.hashes[i]);
        Ok(ir.first_result(inst).expect("expression result"))
    }

    pub(super) fn is_idle(&self) -> bool {
        self.changes.is_empty() && self.rebuild_work.pending.is_empty()
    }
}
