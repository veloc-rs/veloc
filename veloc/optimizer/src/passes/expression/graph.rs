//! Equality indexes, congruence rebuilding and saturation over MIR values.
use super::{Limit, matching};
use crate::evaluate::Fold;
use core::hash::BuildHasher;
use cranelift_entity::{EntityRef, SecondaryMap, packed_option::PackedOption};
use hashbrown::{HashMap, HashTable, hash_map::DefaultHashBuilder};
use smallvec::SmallVec;
use veloc_mir::constant::ScalarConst;
use veloc_mir::function::Expressions;
use veloc_mir::{FuncBody, Inst, IntCC, Opcode as Op, Value, ValueDef};
use veloc_types::Type;

/// An equivalence-class identity, not an executable operand witness. A saved
/// root may cease to be canonical after a union; normalize it before reuse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub(super) struct Root(Value);

impl Root {
    /// Expose MIR identity explicitly for metadata, constants or detached
    /// candidates. This does not establish dominance at an executable use.
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

/// Equivalence is an overlay on MIR value identities. MIR definitions are never
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
            let parent = self.parents[value].expect("registered MIR value");
            if parent == value {
                return Root(value);
            }
            value = parent;
        }
    }

    fn find_mut(&mut self, mut value: Value) -> Root {
        loop {
            let parent = self.parents[value].expect("registered MIR value");
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

/// A temporary lookup key, reconstructed from MIR. The hash table stores only
/// Inst IDs and cached hashes; it owns no second instruction representation.
/// Properties are precisely those exposed by the supported semantic recipes.
#[derive(PartialEq, Eq, Hash)]
struct Key {
    opcode: Op,
    args: SmallVec<[Root; 3]>,
    results: SmallVec<[Type; 2]>,
    properties: SmallVec<[IntCC; 1]>,
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

    fn compact(&mut self, f: &FuncBody, kinds: &SecondaryMap<Inst, InstKind>) {
        self.users.retain(|&inst| kinds[inst] != InstKind::Folded);
        self.users.sort_unstable();
        self.users.dedup();
        let live = |&value: &Value| {
            let inst = f.dfg().value_inst(value).expect("indexed expression");
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

/// All expression storage belongs to MIR. This structure holds equality and
/// dependency indexes over existing values/instructions. A class containing a
/// constant is rooted at that unique MIR literal; no separate fact table exists.
pub(super) struct Graph {
    pub(super) profile: crate::Profile,
    pub(super) values: Vec<Value>,
    classes: UnionFind,
    pub(super) kinds: SecondaryMap<Inst, InstKind>,
    indexes: SecondaryMap<Root, ClassIndex>,
    changes: Vec<Change>,
    rebuild_work: Worklist<Inst>,
    // Direct MIR operand witnesses, never arbitrary union-find representatives.
    aliases: HashMap<Value, Value>,
    memo: HashTable<Inst>,
    hashes: SecondaryMap<Inst, u64>,
    hasher: DefaultHashBuilder,
    /// Detached instructions still allowed; literals and import never consume it.
    pub(super) remaining_nodes: usize,
}

impl Graph {
    pub(super) fn new() -> Self {
        Self {
            profile: crate::Profile::default(),
            values: Vec::new(),
            classes: UnionFind::default(),
            kinds: SecondaryMap::new(),
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

    pub(super) fn register_value(&mut self, f: &FuncBody, value: Value) {
        if self.classes.insert(value) {
            self.values.push(value);
            if f.dfg().as_const(value).is_some() {
                self.changes.push(Change::ClassChanged(Root(value)));
            }
        }
    }

    pub(super) fn floating_inst(&self, f: &FuncBody, value: Value) -> Option<Inst> {
        f.dfg()
            .value_inst(value)
            .filter(|&inst| self.kinds[inst] == InstKind::Floating)
    }

    pub(super) fn args<'a>(&self, f: &'a FuncBody, value: Value) -> &'a [Value] {
        if f.dfg().as_const(self.find(value).value()).is_some() {
            return &[];
        }
        self.floating_inst(f, value)
            .map_or(&[], |inst| f.dfg().operands(inst))
    }

    pub(super) fn canonical_args(&self, f: &FuncBody, inst: Inst) -> SmallVec<[Root; 3]> {
        let args: SmallVec<[Root; 3]> = f
            .dfg()
            .operands(inst)
            .iter()
            .map(|&v| self.find(v))
            .collect();
        self.normalize_args(f.dfg().opcode(inst), &args)
    }

    fn normalize_args(&self, opcode: Op, args: &[Root]) -> SmallVec<[Root; 3]> {
        let mut args: SmallVec<_> = args.iter().map(|&root| self.canonicalize(root)).collect();
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
        let mut args: SmallVec<[Root; 3]> = f
            .dfg()
            .operands(inst)
            .iter()
            .map(|&v| self.find(v))
            .collect();
        args.sort_unstable();
        args.dedup();
        for arg in args {
            self.indexes[arg].users.push(inst);
        }
        if self.kinds[inst] == InstKind::Floating {
            let result = f.dfg().first_result(inst).expect("expression result");
            let class = self.find(result);
            let opcode = f.dfg().opcode(inst);
            self.indexes[class]
                .expressions
                .entry(opcode)
                .or_default()
                .push(result);
            // A new alternative can satisfy a nested pattern in an existing
            // parent even when no operand or constant fact changes.
            self.changes.push(Change::Added(inst));
            for &column in matching::indexed_columns(opcode) {
                let arg = self.find(f.dfg().operands(inst)[column]);
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
        let a_const = matches!(f.dfg().value_def(a.value()), ValueDef::Const(_));
        let b_const = matches!(f.dfg().value_def(b.value()), ValueDef::Const(_));
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
                    let inst = f.dfg().value_inst(value).expect("indexed expression");
                    if self.kinds[inst] != InstKind::Folded {
                        self.rebuild_work.push(inst);
                    }
                }
            }
        }
    }

    /// An executable replacement must follow actual operand edges, not the
    /// arbitrary representative chosen by union-by-size.
    pub(super) fn replacement(&self, f: &FuncBody, mut value: Value) -> Value {
        loop {
            let class = self.find(value);
            if f.dfg().as_const(class.value()).is_some() {
                return class.value();
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
        let args: SmallVec<[Value; 3]> = key.args.iter().map(|root| root.value()).collect();
        crate::evaluate::reduce(key.opcode, &args, &key.results, &key.properties, |value| {
            f.dfg().as_scalar_const(value)
        })
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

    fn try_fold(&mut self, ir: &mut Expressions<'_>, inst: Inst, key: &Key) -> bool {
        let results: SmallVec<[Value; 2]> = ir.body().dfg().inst_results(inst).into();
        // A known value alone cannot discharge a pinned operation's trap.
        // Evaluate its actual inputs before granting permission to erase it.
        let known = self.kinds[inst] == InstKind::Floating
            && results
                .iter()
                .all(|&v| ir.body().dfg().as_const(self.find(v).value()).is_some());
        if !known {
            let Some(reduced) = self.reduce(ir.body(), key) else {
                return false;
            };
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
        // Rebuilding already removed the memo entry. Lists are compacted once
        // the entire wave has settled; MIR storage remains alive until commit.
        self.kinds[inst] = InstKind::Folded;
        self.changes.push(Change::Folded(inst));
        true
    }

    /// Normalize dirty expressions before exposing them to the matcher. This
    /// bounded reduction never creates operations, only literals or equalities;
    /// unlike exploratory rules it also completes when search fuel is exhausted.
    pub(super) fn rebuild(&mut self, ir: &mut Expressions<'_>) {
        let scope = self.profile.scope("egraph.rebuild", 0);
        while let Some(inst) = self.rebuild_work.pop() {
            if self.kinds[inst] == InstKind::Folded {
                continue;
            }
            if let Ok(entry) = self.memo.find_entry(self.hashes[inst], |&old| old == inst) {
                entry.remove();
            }
            let mut key = self.key(ir.body(), inst);
            if self.try_fold(ir, inst, &key) {
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
                let results: SmallVec<[Value; 2]> = ir.body().dfg().inst_results(inst).into();
                let outputs: SmallVec<[Value; 2]> = ir.body().dfg().inst_results(other).into();
                for (&a, &b) in results.iter().zip(&outputs) {
                    self.union(ir.body(), a, b);
                }
            } else {
                self.memo.insert_unique(hash, inst, |&i| self.hashes[i]);
            }
        }
        self.compact_indexes(ir.body());
        scope.success();
    }

    /// Derive cleanup work from mutations at the stable rebuild boundary. Only
    /// matching events survive, including when the caller stops at a budget limit.
    fn compact_indexes(&mut self, f: &FuncBody) {
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
                    f.dfg()
                        .operands(*inst)
                        .iter()
                        .chain(f.dfg().inst_results(*inst))
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
    pub(super) fn schedule(&mut self, f: &FuncBody, queries: &mut Vec<matching::Query>) {
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
                    matching::Seed::Added(f.dfg().first_result(inst).expect("expression result")),
                    matching::added(f.dfg().opcode(inst)),
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

    pub(super) fn literal(&mut self, ir: &mut Expressions<'_>, value: ScalarConst) -> Value {
        let result = ir.constant(value.into());
        self.register_value(ir.body(), result);
        result
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
                    self.literal(ir, c)
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
        if self.remaining_nodes == 0 {
            return Err(Limit::Nodes);
        }
        // Detached candidates may use representatives; executable operands are
        // selected separately under dominance checks during extraction.
        let args: SmallVec<[Value; 3]> = key.args.iter().map(|root| root.value()).collect();
        let inst = ir.create(
            |w| {
                w.from_values(opcode, &args)
                    .expect("value-only rule operation")
            },
            &[ty],
        );
        self.remaining_nodes -= 1;
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
    crate::evaluate::can_reduce(f.dfg(), inst)
}
