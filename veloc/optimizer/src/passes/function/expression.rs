//! One equality-graph pipeline; fast mode changes budgets, not semantics.
use crate::{FunctionPass, Metrics, OptConfig, PreservedAnalyses};
use alloc::{collections::VecDeque, vec, vec::Vec};
use core::hash::BuildHasher;
use cranelift_entity::{PrimaryMap, SecondaryMap};
use hashbrown::{HashMap, HashSet, HashTable, hash_map::DefaultHashBuilder};
use smallvec::SmallVec;
use veloc_analyzer::AnalysisManager;
use veloc_mir::ValueDef;
use veloc_mir::constant::ScalarConst;
use veloc_mir::function::Dominators;
use veloc_mir::{FuncBody, Inst, IntCC, Opcode as Op, Type, Value};
use veloc_types::TypeInfo;

/// Estimates execution cost, not the effort spent searching a rewrite rule.
/// Costs are clamped to at least one so cyclic e-classes cannot win extraction.
pub trait CostModel {
    fn operation(&self, opcode: veloc_mir::Opcode, ty: veloc_mir::Type) -> usize;
    fn constant(&self, value: veloc_mir::constant::ScalarConst) -> usize;
}

/// Target-independent baseline. Targets may provide their own estimates without
/// changing equality rules or treating machine costs as MIR semantic facts.
pub struct GenericCost;
impl CostModel for GenericCost {
    fn operation(&self, _: veloc_mir::Opcode, _: veloc_mir::Type) -> usize {
        1
    }
    fn constant(&self, _: veloc_mir::constant::ScalarConst) -> usize {
        1
    }
}

pub struct ExpressionPass {
    pub budget: Budget,
}

impl FunctionPass for ExpressionPass {
    fn name(&self) -> &str {
        "ExpressionPass"
    }

    fn run(
        &self,
        am: &mut AnalysisManager<'_>,
        config: &OptConfig,
        metrics: &mut Metrics,
    ) -> PreservedAnalyses {
        if run(
            am.function_mut(),
            self.budget,
            config.is_debug_enabled("simplify"),
            metrics,
        ) {
            PreservedAnalyses::none()
        } else {
            PreservedAnalyses::all()
        }
    }
}

pub fn run(func: &mut FuncBody, budget: Budget, debug: bool, metrics: &mut Metrics) -> bool {
    run_with_cost(func, budget, &GenericCost, debug, metrics)
}

pub fn run_with_cost(
    func: &mut FuncBody,
    budget: Budget,
    cost: &dyn CostModel,
    debug: bool,
    metrics: &mut Metrics,
) -> bool {
    let changed = optimize_function(func, budget, cost, metrics);
    if changed && debug {
        log::info!("Optimized expression graph");
    }
    changed
}

/// Deterministic search limits for one function. Both profiles
/// run the same graph optimizer; neither is a separate greedy rewrite engine.
#[derive(Clone, Copy)]
pub struct Budget {
    /// Additional nodes allowed beyond the imported function.
    pub graph_nodes: usize,
    pub rounds: usize,
    pub match_steps: usize,
}

impl Budget {
    pub const FAST: Self = Self {
        graph_nodes: 160,
        rounds: 2,
        match_steps: 16_384,
    };
    pub const DEFAULT: Self = Self {
        graph_nodes: 512,
        rounds: 6,
        match_steps: 262_144,
    };
}

/// An equivalence class; use Graph::find to resolve its current representative.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
struct ClassId(u32);
cranelift_entity::entity_impl!(ClassId, "class");

/// A stable expression node, independent of equivalence-class merges.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
struct NodeId(u32);
cranelift_entity::entity_impl!(NodeId, "node");

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
struct OpId(u32);
cranelift_entity::entity_impl!(OpId, "op");

/// Kept separate from class payloads so representative lookups touch only IDs.
#[derive(Default)]
struct UnionFind {
    parents: PrimaryMap<ClassId, ClassId>,
    sizes: SecondaryMap<ClassId, usize>,
}

impl UnionFind {
    fn insert(&mut self) -> ClassId {
        let id = self.parents.next_key();
        self.parents.push(id);
        self.sizes[id] = 1;
        id
    }

    fn find(&self, mut id: ClassId) -> ClassId {
        while self.parents[id] != id {
            id = self.parents[id];
        }
        id
    }

    fn find_mut(&mut self, mut id: ClassId) -> ClassId {
        // Path halving needs neither a temporary path nor a second traversal.
        while self.parents[id] != id {
            let grandparent = self.parents[self.parents[id]];
            self.parents[id] = grandparent;
            id = grandparent;
        }
        id
    }

    /// Returns (winner, loser); sizes count class IDs, not expression nodes.
    fn union(&mut self, a: ClassId, b: ClassId) -> Option<(ClassId, ClassId)> {
        let (mut a, mut b) = (self.find_mut(a), self.find_mut(b));
        if a == b {
            return None;
        }
        if self.sizes[a] < self.sizes[b] {
            core::mem::swap(&mut a, &mut b);
        }
        self.parents[b] = a;
        self.sizes[a] += self.sizes[b];
        Some((a, b))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Operation {
    opcode: Op,
    args: Vec<ClassId>,
    results: Vec<Type>,
    // All properties exposed by the supported scalar semantic recipes (e.g.
    // comparison predicates). A source instruction ID is not a property.
    properties: Vec<IntCC>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Node {
    Input(Value, Type),
    Constant(ScalarConst),
    Result(OpId, usize),
    // A fixed occurrence: evaluate its inputs, but never move or CSE the node.
    Anchor(OpId, Inst, usize),
}
impl Node {
    fn ty(&self, graph: &Graph) -> Type {
        match self {
            Self::Input(_, ty) => *ty,
            Self::Constant(c) => c.ty(),
            Self::Result(op, index) | Self::Anchor(op, _, index) => {
                graph.operations[*op].results[*index]
            }
        }
    }
    /// Inputs needed to materialize this node. Fixed occurrences are leaves.
    fn args<'a>(&self, graph: &'a Graph) -> &'a [ClassId] {
        match self {
            Self::Result(op, _) => &graph.operations[*op].args,
            _ => &[],
        }
    }
    /// Inputs whose facts can change this node, including fixed occurrences.
    fn dependencies<'a>(&self, graph: &'a Graph) -> &'a [ClassId] {
        match self {
            Self::Result(op, _) | Self::Anchor(op, _, _) => &graph.operations[*op].args,
            _ => &[],
        }
    }
    fn price(&self, graph: &Graph, model: &dyn CostModel) -> usize {
        match self {
            Self::Input(..) | Self::Anchor(..) => 0,
            Self::Constant(c) => model.constant(*c).max(1),
            Self::Result(op, _) => {
                let op = &graph.operations[*op];
                model.operation(op.opcode, op.results[0]).max(1)
            }
        }
    }
}

struct Graph {
    nodes: PrimaryMap<NodeId, Node>,
    // Membership is explicit: IDs from these two arenas are not interchangeable.
    types: PrimaryMap<ClassId, Type>,
    class_has_node: SecondaryMap<ClassId, bool>,
    node_classes: SecondaryMap<NodeId, ClassId>,
    // Union-find is separate from expression-node storage.
    classes: UnionFind,
    constants: SecondaryMap<ClassId, Option<ScalarConst>>,
    users: SecondaryMap<ClassId, Vec<NodeId>>,
    dirty: Vec<NodeId>,
    queued: SecondaryMap<NodeId, bool>,
    rule_work: Vec<NodeId>,
    rule_queued: SecondaryMap<NodeId, bool>,
    analysis: Vec<NodeId>,
    analyze: SecondaryMap<NodeId, bool>,
    memo: HashMap<Node, NodeId>,
    operations: PrimaryMap<OpId, Operation>,
    // Keys live only in the arena; the hash table stores IDs, not cloned keys.
    op_memo: HashTable<OpId>,
    op_hasher: DefaultHashBuilder,
    // Relational fact index: (canonical e-class, opcode) -> matching nodes.
    // The key is maintained when classes merge, so rule matching never scans
    // unrelated members of an e-class.
    nodes_by_class_op: HashMap<(ClassId, Op), Vec<NodeId>>,
    class_ops: SecondaryMap<ClassId, Vec<Op>>,
    // Reconstruction provenance is deliberately not part of semantic identity.
    templates: SecondaryMap<OpId, Option<Inst>>,
    revision: usize,
    limit: usize,
}
impl Graph {
    fn new(limit: usize) -> Self {
        Self {
            nodes: PrimaryMap::new(),
            types: PrimaryMap::new(),
            class_has_node: SecondaryMap::new(),
            node_classes: SecondaryMap::new(),
            classes: UnionFind::default(),
            constants: SecondaryMap::new(),
            users: SecondaryMap::new(),
            dirty: Vec::new(),
            queued: SecondaryMap::new(),
            rule_work: Vec::new(),
            rule_queued: SecondaryMap::new(),
            analysis: Vec::new(),
            analyze: SecondaryMap::new(),
            memo: HashMap::new(),
            operations: PrimaryMap::new(),
            op_memo: HashTable::new(),
            op_hasher: DefaultHashBuilder::default(),
            nodes_by_class_op: HashMap::new(),
            class_ops: SecondaryMap::new(),
            templates: SecondaryMap::new(),
            revision: 0,
            limit,
        }
    }
    fn find(&self, id: ClassId) -> ClassId {
        self.classes.find(id)
    }
    fn find_mut(&mut self, id: ClassId) -> ClassId {
        self.classes.find_mut(id)
    }
    fn class(&self, node: NodeId) -> ClassId {
        self.find(self.node_classes[node])
    }
    fn ty(&self, id: ClassId) -> Type {
        self.types[id]
    }
    fn queue_rule(&mut self, node: NodeId) {
        if !self.rule_queued[node] {
            self.rule_queued[node] = true;
            self.rule_work.push(node);
        }
    }
    fn operation(&mut self, op: Operation, template: Option<Inst>) -> OpId {
        let hash = self.op_hasher.hash_one(&op);
        let id = if let Some(&id) = self.op_memo.find(hash, |&id| self.operations[id] == op) {
            id
        } else {
            let id = self.operations.push(op);
            self.op_memo.insert_unique(hash, id, |&id| {
                self.op_hasher.hash_one(&self.operations[id])
            });
            id
        };
        if self.templates[id].is_none() {
            self.templates[id] = template;
        }
        id
    }
    fn normalize(&mut self, mut node: Node) -> Node {
        if let Node::Result(id, _) | Node::Anchor(id, _, _) = &mut node {
            let op = &self.operations[*id];
            let canonical = op.args.iter().all(|&arg| self.classes.find_mut(arg) == arg);
            let commutative = op.opcode.spec().is_commutative() && op.args.len() == 2;
            if !canonical || (commutative && op.args[0] > op.args[1]) {
                let template = self.templates[*id];
                let mut normalized = op.clone();
                for arg in &mut normalized.args {
                    *arg = self.find_mut(*arg);
                }
                if commutative && normalized.args[0] > normalized.args[1] {
                    normalized.args.swap(0, 1);
                }
                *id = self.operation(normalized, template);
            }
        }
        node
    }
    /// Forward references allocate only a class, not a fake expression node.
    fn create_class(&mut self, ty: Type) -> ClassId {
        let class = self.types.push(ty);
        let id = self.classes.insert();
        debug_assert_eq!(class, id);
        class
    }
    fn add(&mut self, node: Node) -> Option<ClassId> {
        let node = self.normalize(node);
        if let Some(&id) = self.memo.get(&node) {
            return Some(self.class(id));
        }
        if self.nodes.len() >= self.limit {
            return None;
        }
        let class = self.create_class(node.ty(self));
        self.insert_node(class, node);
        Some(class)
    }
    fn add_to_class(&mut self, class: ClassId, node: Node) -> Option<ClassId> {
        let class = self.find_mut(class);
        let node = self.normalize(node);
        assert_eq!(
            self.ty(class),
            node.ty(self),
            "node type does not match its class"
        );
        if let Some(&id) = self.memo.get(&node) {
            self.union(class, self.class(id));
            return Some(self.find_mut(class));
        }
        if self.nodes.len() >= self.limit {
            return None;
        }
        self.insert_node(class, node);
        Some(class)
    }
    /// Insert a canonical, absent node into a representative class. Callers
    /// perform type, memo and budget checks before changing graph storage.
    fn insert_node(&mut self, class: ClassId, node: Node) {
        let id = self.nodes.push(node);
        self.node_classes[id] = class;
        self.class_has_node[class] = true;
        if let Node::Result(op, 0) = node {
            let opcode = self.operations[op].opcode;
            let key = (class, opcode);
            if !self.nodes_by_class_op.contains_key(&key) {
                self.class_ops[class].push(opcode);
            }
            self.nodes_by_class_op.entry(key).or_default().push(id);
        }
        if let Node::Constant(c) = node {
            if let Some(old) = self.constants[class] {
                assert_eq!(old, c, "rewrite equated distinct constants");
            } else {
                self.constants[class] = Some(c);
                for &user in &self.users[class] {
                    if !self.analyze[user] {
                        self.analyze[user] = true;
                        self.analysis.push(user);
                    }
                }
            }
        }
        self.analyze[id] = true;
        self.analysis.push(id);
        self.queue_rule(id);
        let mut dependencies: SmallVec<[ClassId; 3]> = node
            .dependencies(self)
            .iter()
            .map(|&arg| self.find(arg))
            .collect();
        // x + x is one dependent node, not two distinct rebuild obligations.
        dependencies.sort_unstable();
        dependencies.dedup();
        for arg in dependencies {
            self.users[arg].push(id);
        }
        self.memo.insert(node, id);
        self.revision += 1;
    }
    fn union(&mut self, a: ClassId, b: ClassId) {
        let (a, b) = (self.find_mut(a), self.find_mut(b));
        assert_eq!(self.ty(a), self.ty(b), "cannot equate different types");
        if let Some((a, b)) = self.classes.union(a, b) {
            if let (Some(x), Some(y)) = (self.constants[a], self.constants[b]) {
                assert_eq!(x, y, "rewrite equated distinct constants");
            }
            let winner_constant = self.constants[a];
            let loser_constant = self.constants[b];
            self.constants[a] = winner_constant.or(loser_constant);
            // Existing winner users need analysis only if their class learned
            // a new fact. Otherwise only loser users may have learned one.
            if winner_constant.is_none() && loser_constant.is_some() {
                for &user in &self.users[a] {
                    if !self.analyze[user] {
                        self.analyze[user] = true;
                        self.analysis.push(user);
                    }
                }
            }
            // Move the losing class's relation rows to the new representative.
            // Keeping one row per (class, opcode) makes rule scans proportional
            // to relevant alternatives instead of all class members.
            for opcode in core::mem::take(&mut self.class_ops[b]) {
                let source = self
                    .nodes_by_class_op
                    .remove(&(b, opcode))
                    .unwrap_or_default();
                let mut target = self
                    .nodes_by_class_op
                    .remove(&(a, opcode))
                    .unwrap_or_default();
                if target.is_empty() {
                    self.class_ops[a].push(opcode);
                }
                for &node in target.iter().chain(source.iter()) {
                    self.queue_rule(node);
                }
                target.extend(source);
                self.nodes_by_class_op.insert((a, opcode), target);
            }
            self.class_has_node[a] |= self.class_has_node[b];
            // Only users of the losing representative have non-canonical keys.
            for user in core::mem::take(&mut self.users[b]) {
                if !self.queued[user] {
                    self.queued[user] = true;
                    self.dirty.push(user);
                }
                self.queue_rule(user);
                if loser_constant.is_none() && winner_constant.is_some() && !self.analyze[user] {
                    self.analyze[user] = true;
                    self.analysis.push(user);
                }
                self.users[a].push(user);
            }
            // A node may have depended on both classes before the merge. Keep
            // the relation set-like after moving the two adjacency lists.
            self.users[a].sort_unstable();
            self.users[a].dedup();
            self.revision += 1;
        }
    }
    fn scan(&self, class: ClassId, opcode: Op) -> impl Iterator<Item = &Node> {
        let class = self.find(class);
        self.nodes_by_class_op
            .get(&(class, opcode))
            .into_iter()
            .flat_map(|ids| ids.iter().map(|&id| &self.nodes[id]))
    }
    fn constant(&self, class: ClassId) -> Option<ScalarConst> {
        self.constants[self.find(class)]
    }
    fn rebuild(&mut self) {
        while let Some(id) = self.dirty.pop() {
            self.queued[id] = false;
            let old = self.nodes[id];
            if self.memo.get(&old) == Some(&id) {
                self.memo.remove(&old);
            }
            let node = self.normalize(old);
            self.nodes[id] = node;
            self.queue_rule(id);
            if let Some(&other) = self.memo.get(&node) {
                self.union(self.class(id), self.class(other));
            } else {
                self.memo.insert(node, id);
            }
        }
    }
    fn saturate(&mut self, rounds: usize, fuel: &mut usize) {
        // Once all inputs are constant, their facts cannot change. Cache both
        // successful evaluation and a refusal (for example, division by zero).
        let mut evaluated = HashMap::new();
        let mut matcher = crate::equivalence::RelationalMatcher::new();
        for _ in 0..rounds {
            self.rebuild();
            let before = self.revision;
            self.fold_constants(fuel, &mut evaluated);
            while let Some(id) = self.rule_work.pop() {
                self.rule_queued[id] = false;
                if *fuel == 0 {
                    break;
                }
                *fuel -= 1;
                let Node::Result(op, _) = self.nodes[id] else {
                    continue;
                };
                let op = &self.operations[op];
                if op.results.len() != 1 {
                    continue;
                }
                let ty = op.results[0];
                let opcode = op.opcode;
                if op.args.iter().any(|&a| self.ty(a) != ty) {
                    continue;
                }
                if let [a, b] = op.args.as_slice() {
                    let constants = [self.constant(*a), self.constant(*b)];
                    let args = [self.find(*a), self.find(*b)];
                    if let Some(replacement) = crate::rewrite::algebraic(opcode, &args, &constants)
                    {
                        match replacement {
                            crate::rewrite::Replacement::Value(value) => {
                                self.union(self.class(id), value)
                            }
                            crate::rewrite::Replacement::Constants(values) => {
                                if let Some(value) = self.add(Node::Constant(values[0])) {
                                    self.union(self.class(id), value);
                                }
                            }
                        }
                    }
                }
                if !ty.is_integer() {
                    continue;
                }
                let mask = u64::MAX >> (64 - ty.element_bits().unwrap());
                let mut context = RuleContext { graph: self, ty };
                for rule in crate::equivalence::rules(opcode) {
                    if !rule.types.contains(&ty) {
                        continue;
                    }
                    matcher.search(&context, rule, context.graph.class(id), mask, fuel);
                    for env in matcher.matches() {
                        if let Some(value) =
                            crate::equivalence::emit(&mut context, &rule.replacement, env)
                        {
                            context.graph.union(context.graph.class(id), value);
                            log::trace!("egraph rule {}", rule.name);
                        }
                    }
                }
            }
            self.fold_constants(fuel, &mut evaluated);
            if self.revision == before || *fuel == 0 {
                break;
            }
        }
        self.rebuild();
    }

    fn fold_constants(
        &mut self,
        fuel: &mut usize,
        evaluated: &mut HashMap<OpId, Option<Vec<ScalarConst>>>,
    ) {
        while *fuel > 0 {
            let Some(id) = self.analysis.pop() else { break };
            self.analyze[id] = false;
            *fuel -= 1;
            let (op_id, index) = match self.nodes[id] {
                Node::Result(op, index) | Node::Anchor(op, _, index) => (op, index),
                _ => continue,
            };
            let op = &self.operations[op_id];
            // Unknown inputs are not cached: a later union wakes their users.
            let Some(args) = op
                .args
                .iter()
                .map(|&arg| self.constant(arg))
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            let folded = evaluated.entry(op_id).or_insert_with(|| {
                crate::rewrite::evaluate(op.opcode, &args, &op.results, &op.properties)
            });
            if let Some(values) = folded
                && let Some(literal) = self.add(Node::Constant(values[index]))
            {
                self.union(self.class(id), literal);
            }
        }
    }
    fn extract(
        &self,
        roots: &[ClassId],
        model: &dyn CostModel,
    ) -> Option<SecondaryMap<ClassId, Option<NodeId>>> {
        let mut costs = SecondaryMap::<ClassId, _>::with_default((usize::MAX, usize::MAX));
        let mut best = SecondaryMap::<ClassId, Option<NodeId>>::new();
        let mut pending: VecDeque<_> = self.nodes.keys().collect();
        let mut queued = SecondaryMap::<NodeId, bool>::with_default(true);
        while let Some(id) = pending.pop_front() {
            queued[id] = false;
            let node = &self.nodes[id];
            let class = self.class(id);
            // A proven literal is a canonical result, including when its
            // operands were zero-cost SSA boundary inputs.
            if !matches!(node, Node::Constant(..)) && self.constant(class).is_some() {
                continue;
            }
            let mut price = node.price(self, model);
            let mut depth = 0;
            for &arg in node.args(self) {
                let arg = self.find(arg);
                price = price.saturating_add(costs[arg].0);
                depth = depth.max(costs[arg].1);
            }
            let depth = depth.saturating_add(usize::from(!matches!(node, Node::Input(..))));
            if price != usize::MAX && (price, depth) < costs[class] {
                costs[class] = (price, depth);
                best[class] = Some(id);
                for &user in &self.users[class] {
                    if !queued[user] {
                        queued[user] = true;
                        pending.push_back(user);
                    }
                }
            }
        }
        // Tree costs give an inexpensive, acyclic starting point. Then compare
        // alternatives using the *whole* reachable DAG, counting an operation
        // once even if several roots or result projections use it. This bounded
        // local search is not an optimal DAG extractor.
        let mut work = usize::MAX;
        let mut plan = self.plan(roots, &best, &mut work)?;
        let mut price = self.plan_price(&plan, model);
        let mut work = self.nodes.len().saturating_mul(16);
        loop {
            let mut improved = false;
            let reachable: HashSet<_> = plan.iter().map(|&(class, _)| class).collect();
            for (id, node) in self.nodes.iter() {
                let class = self.class(id);
                if work == 0 {
                    break;
                }
                work -= 1;
                if !reachable.contains(&class)
                    || best[class] == Some(id)
                    || (!matches!(node, Node::Constant(..)) && self.constant(class).is_some())
                {
                    continue;
                }
                let old = best[class].replace(id);
                if let Some(candidate) = self.plan(roots, &best, &mut work) {
                    let candidate_price = self.plan_price(&candidate, model);
                    if candidate_price < price {
                        plan = candidate;
                        price = candidate_price;
                        improved = true;
                        continue;
                    }
                }
                best[class] = old;
            }
            if !improved || work == 0 {
                break;
            }
        }
        Some(best)
    }

    /// A topological materialization plan, or None for a cyclic/unavailable
    /// choice. The budget also bounds exploration of rejected alternatives.
    fn plan(
        &self,
        roots: &[ClassId],
        best: &SecondaryMap<ClassId, Option<NodeId>>,
        work: &mut usize,
    ) -> Option<Vec<(ClassId, NodeId)>> {
        let mut state = SecondaryMap::<ClassId, u8>::new();
        let mut output = Vec::new();
        for &root in roots {
            let mut pending = vec![(self.find(root), false)];
            while let Some((class, ready)) = pending.pop() {
                if *work == 0 {
                    return None;
                }
                *work -= 1;
                if state[class] == 2 {
                    continue;
                }
                let id = best[class]?;
                let node = &self.nodes[id];
                if !ready && !node.args(self).is_empty() {
                    if state[class] == 1 {
                        return None;
                    }
                    state[class] = 1;
                    pending.push((class, true));
                    pending.extend(
                        node.args(self)
                            .iter()
                            .rev()
                            .map(|&arg| (self.find(arg), false)),
                    );
                } else {
                    state[class] = 2;
                    output.push((class, id));
                }
            }
        }
        Some(output)
    }

    fn plan_price(&self, plan: &[(ClassId, NodeId)], model: &dyn CostModel) -> usize {
        let mut operations = HashSet::new();
        plan.iter().fold(0usize, |price, &(_, id)| {
            let node = &self.nodes[id];
            if let Node::Result(op, _) = node
                && !operations.insert(op)
            {
                return price;
            }
            price.saturating_add(node.price(self, model))
        })
    }
}

struct RuleContext<'a> {
    graph: &'a mut Graph,
    ty: Type,
}
impl crate::equivalence::Context for RuleContext<'_> {
    type Value = ClassId;
    fn canonical(&self, value: ClassId) -> ClassId {
        self.graph.find(value)
    }
    fn constant(&self, value: ClassId) -> Option<u64> {
        self.graph.constant(value).map(|c| c.to_bits())
    }
    fn scan(&self, value: ClassId, opcode: Op, mut visit: impl FnMut(&[ClassId])) {
        for node in self.graph.scan(value, opcode) {
            if let Node::Result(op, 0) = node
                && let op = &self.graph.operations[*op]
                && op.results == [self.ty]
                && op.args.iter().all(|&a| self.graph.ty(a) == self.ty)
            {
                let args: SmallVec<[ClassId; 3]> =
                    op.args.iter().map(|&a| self.graph.find(a)).collect();
                visit(&args);
                if opcode.spec().is_commutative()
                    && let [a, b] = args.as_slice()
                    && a != b
                {
                    visit(&[*b, *a]);
                }
            }
        }
    }
    fn literal(&mut self, value: u64) -> Option<ClassId> {
        let mask = u64::MAX >> (64 - self.ty.element_bits()?);
        self.graph.add(Node::Constant(ScalarConst::from_bits(
            self.ty,
            value & mask,
        )?))
    }
    fn build(&mut self, opcode: Op, args: &[ClassId]) -> Option<ClassId> {
        if self.graph.nodes.len() >= self.graph.limit {
            return None;
        }
        let op = self.graph.operation(
            Operation {
                opcode,
                args: args.to_vec(),
                results: vec![self.ty],
                properties: Vec::new(),
            },
            None,
        );
        self.graph.add(Node::Result(op, 0))
    }
}

fn candidate(f: &FuncBody, id: Inst) -> bool {
    if f.layout().inst_block(id).is_none() {
        return false;
    }
    let inst = f.dfg().inst(id);
    let results = f.dfg().inst_results(id);
    !results.is_empty()
        && results
            .iter()
            .all(|&v| ScalarConst::from_bits(f.dfg().value_type(v), 0).is_some())
        && (results.len() == 1 && f.dfg().as_scalar_const(results[0]).is_some()
            || crate::rewrite::can_fold(inst.opcode()))
        && inst.memory_effect().is_none()
        && !inst.opcode().spec().is_terminator()
        && !inst.opcode().transfers_ownership()
}

/// Allocate SSA classes first, then populate their definitions. Only the
/// ordered candidate list is needed here; all remaining values become leaves.
fn import(
    f: &FuncBody,
    candidates: &[Inst],
    budget: Budget,
) -> (Graph, PrimaryMap<Value, ClassId>) {
    let mut graph = Graph::new(f.dfg().values().len().saturating_add(budget.graph_nodes));
    let mut values = PrimaryMap::new();
    for (_, data) in f.dfg().values().iter() {
        values.push(graph.create_class(data.ty));
    }
    for &id in candidates {
        let results = f.dfg().inst_results(id);
        if let [dst] = results
            && let Some(c) = f.dfg().as_scalar_const(*dst)
        {
            graph
                .add_to_class(values[*dst], Node::Constant(c))
                .expect("reserved definition");
            continue;
        }
        let op = graph.operation(
            Operation {
                opcode: f.dfg().inst(id).opcode(),
                args: f.dfg().operands(id).iter().map(|v| values[*v]).collect(),
                results: results.iter().map(|&v| f.dfg().value_type(v)).collect(),
                properties: crate::rewrite::properties(&f.dfg().inst(id)).into_vec(),
            },
            Some(id),
        );
        let pure = f.dfg().inst(id).can_speculate();
        for (index, &dst) in results.iter().enumerate() {
            let node = if pure {
                Node::Result(op, index)
            } else {
                Node::Anchor(op, id, index)
            };
            graph
                .add_to_class(values[dst], node)
                .expect("reserved definition");
        }
    }
    for (value, data) in f.dfg().values().iter() {
        let class = graph.find(values[value]);
        if !graph.class_has_node[class] {
            graph
                .add_to_class(class, Node::Input(value, data.ty))
                .expect("reserved input");
        }
    }
    graph.rebuild();
    (graph, values)
}

/// Places selected pure nodes into the existing CFG. Generated values are
/// cached per e-class, but a cached value is usable only where it dominates.
/// There is no speculative hoisting across branches or out of loops.
struct Placement {
    dom: Dominators,
    positions: HashMap<Inst, usize>,
    available: SecondaryMap<ClassId, Vec<Value>>,
}

impl Placement {
    fn new(f: &FuncBody, ids: &[Inst]) -> Self {
        Self {
            dom: Dominators::compute(f.cfg(), f.entry_block(), f.dfg().block_count()),
            // Even positions belong to generated instructions immediately
            // before the original instruction at the following odd position.
            positions: ids
                .iter()
                .enumerate()
                .map(|(i, &id)| (id, i * 2 + 1))
                .collect(),
            available: SecondaryMap::new(),
        }
    }

    fn dominates(&self, f: &FuncBody, value: Value, anchor: Inst) -> bool {
        let use_block = f.layout().inst_block(anchor).unwrap();
        match f.dfg().values()[value].def {
            ValueDef::Param(block) => block == use_block || self.dom.dominates(block, use_block),
            ValueDef::Inst(def) => {
                let Some(block) = f.layout().inst_block(def) else {
                    return false;
                };
                if block == use_block {
                    self.positions[&def] < self.positions[&anchor]
                } else {
                    self.dom.dominates(block, use_block)
                }
            }
        }
    }

    fn existing(&self, f: &FuncBody, class: ClassId, anchor: Inst) -> Option<Value> {
        self.available[class]
            .iter()
            .rev()
            .copied()
            .find(|&v| self.dominates(f, v, anchor))
    }

    fn materialize(
        &mut self,
        f: &mut FuncBody,
        anchor: Inst,
        root: ClassId,
        graph: &Graph,
        choices: &SecondaryMap<ClassId, Option<NodeId>>,
    ) -> Option<Value> {
        // Explicit stack avoids recursive traversal of long SSA expression chains.
        let mut pending = vec![(root, false)];
        let mut local = HashMap::new();
        while let Some((class, ready)) = pending.pop() {
            if local.contains_key(&class) {
                continue;
            }
            if let Some(value) = self.existing(f, class, anchor) {
                local.insert(class, value);
                continue;
            }
            let node = &graph.nodes[choices[class]?];
            if !ready && !node.args(graph).is_empty() {
                pending.push((class, true));
                pending.extend(
                    node.args(graph)
                        .iter()
                        .rev()
                        .map(|&arg| (graph.find(arg), false)),
                );
                continue;
            }
            let (value, emitted) = match node {
                Node::Input(value, _) => {
                    if !self.dominates(f, *value, anchor) {
                        return None;
                    }
                    (*value, None)
                }
                Node::Anchor(_, inst, index) => (f.dfg().inst_results(*inst)[*index], None),
                Node::Constant(c) => {
                    if let Some(value) = f.dfg().first_result(anchor)
                        && f.dfg().as_scalar_const(value) == Some(*c)
                    {
                        local.insert(class, value);
                        self.available[class].push(value);
                        continue;
                    }
                    let inst = f
                        .edit()
                        .insert_before(anchor, |w| w.scalar_const(*c), &[c.ty()]);
                    (f.dfg().first_result(inst).unwrap(), Some(inst))
                }
                Node::Result(op_id, index) => {
                    let op = &graph.operations[*op_id];
                    let args: Vec<_> = op.args.iter().map(|&a| local[&graph.find(a)]).collect();
                    // Keep an instruction whose selected expression is already
                    // present. Merely importing and exporting must not clone
                    // every instruction or report a change on a fixed point.
                    let reuse = class == root
                        && f.dfg().inst(anchor).opcode() == op.opcode
                        && f.dfg().operands(anchor) == args
                        && f.dfg()
                            .inst_results(anchor)
                            .iter()
                            .map(|&v| f.dfg().value_type(v))
                            .eq(op.results.iter().copied())
                        && crate::rewrite::properties(&f.dfg().inst(anchor)).as_slice()
                            == op.properties;
                    let mut edit = f.edit();
                    let inst = if reuse {
                        anchor
                    } else if let Some(template) = graph.templates[*op_id] {
                        let inst = edit.insert_before(anchor, |w| w.copy(template), &op.results);
                        for (i, &arg) in args.iter().enumerate() {
                            edit.set_operand(inst, i as u32, arg);
                        }
                        inst
                    } else {
                        edit.insert_before(
                            anchor,
                            |w| {
                                w.from_values(op.opcode, &args)
                                    .expect("value-only rule operation")
                            },
                            &op.results,
                        )
                    };
                    // An operation is emitted once for all its result projections.
                    for (i, &value) in edit.body().dfg().inst_results(inst).iter().enumerate() {
                        if let Some(&result) = graph.memo.get(&Node::Result(*op_id, i)) {
                            let result = graph.class(result);
                            local.insert(result, value);
                            if result != class {
                                self.available[result].push(value);
                            }
                        }
                    }
                    (
                        edit.body().dfg().inst_results(inst)[*index],
                        (!reuse).then_some(inst),
                    )
                }
            };
            if let Some(inst) = emitted {
                self.positions.insert(inst, self.positions[&anchor] - 1);
            }
            local.insert(class, value);
            self.available[class].push(value);
        }
        local.get(&root).copied()
    }
}

fn optimize_function(
    f: &mut FuncBody,
    budget: Budget,
    model: &dyn CostModel,
    metrics: &mut Metrics,
) -> bool {
    let entry = f.entry_block();
    // Dominating blocks precede their users. Unreachable blocks are processed
    // separately; no cross-block reuse is allowed without proven dominance.
    let mut blocks = f.cfg().compute_rpo(entry);
    let reachable: HashSet<_> = blocks.iter().copied().collect();
    blocks.extend(f.layout().block_order().filter(|b| !reachable.contains(b)));
    let ids: Vec<_> = blocks
        .into_iter()
        .flat_map(|b| f.layout().block_insts(b))
        .collect();
    let candidates: Vec<_> = ids.iter().copied().filter(|&id| candidate(f, id)).collect();
    if candidates.is_empty() {
        return false;
    }
    let (mut graph, values) = import(f, &candidates, budget);
    let mut fuel = budget.match_steps;
    // Pure and pinned computations share dependency-driven constant propagation.
    // Only pure Result nodes participate in algebraic rewriting and placement.
    graph.saturate(budget.rounds, &mut fuel);
    let roots: Vec<_> = candidates
        .iter()
        .flat_map(|&id| f.dfg().inst_results(id))
        .map(|v| values[*v])
        .collect();
    let Some(choices) = graph.extract(&roots, model) else {
        return false;
    };
    let mut placement = Placement::new(f, &ids);
    let mut changed = 0u64;
    let mut removable = Vec::new();
    for id in candidates {
        let results = f.dfg().inst_results(id).to_vec();
        let pinned = !f.dfg().inst(id).can_speculate();
        // A fixed occurrence becomes removable only after every result has a
        // proven literal. Evaluator refusals (including traps) remain pinned.
        if pinned && !results.iter().all(|&v| graph.constant(values[v]).is_some()) {
            continue;
        }
        for old in results {
            if f.dfg().uses(old).next().is_none() {
                continue;
            }
            let class = graph.find(values[old]);
            if let Some(new) = placement.materialize(f, id, class, &graph, &choices)
                && old != new
            {
                f.edit().replace_all_uses(old, new);
                changed += 1;
            }
        }
        if pinned {
            removable.push(id);
        }
    }
    removable.retain(|&id| {
        f.dfg()
            .inst_results(id)
            .iter()
            .all(|&v| f.dfg().uses(v).next().is_none())
    });
    if !removable.is_empty() {
        f.edit().erase_insts(&removable);
        changed += removable.len() as u64;
    }
    let cleaned = super::dce::run_dce(f, false, metrics);
    metrics.add("egraph.nodes", graph.nodes.len() as u64);
    metrics.add("egraph.rewritten_values", changed);
    metrics.add("egraph.match_steps", (budget.match_steps - fuel) as u64);
    changed != 0 || cleaned
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Metrics;

    #[test]
    fn cross_block_graph_preserves_effects_and_ssa() {
        let parsed = veloc_mir::ModuleParser::new()
            .parse(
                r#"
local function cross(i64, ptr) -> i64
block0(v0: i64, v1: ptr):
  v2: i64 = iconst 3
  v3: i64 = iadd v0, v2
  jump block1()
block1():
  v4: i64 = load.volatile v1, offset=0
  v5: i64 = iconst 4
  v6: i64 = iadd v3, v5
  v7: i64 = iadd v6, v4
  return v7
"#,
            )
            .unwrap();
        let mut module = (*parsed).clone();
        module.validate().unwrap();
        let f = module.bodies[veloc_mir::FuncId(0)].as_deref_mut().unwrap();
        let load = f
            .layout()
            .block_order()
            .flat_map(|b| f.layout().block_insts(b))
            .find(|&i| f.dfg().inst(i).opcode() == Op::Load)
            .unwrap();
        let load_block = f.layout().inst_block(load);
        assert!(super::run(
            f,
            Budget::DEFAULT,
            false,
            &mut Metrics::default()
        ));
        assert_eq!(f.layout().inst_block(load), load_block);
        assert_eq!(f.dfg().inst(load).opcode(), Op::Load);
        let constants: Vec<_> = f
            .layout()
            .block_order()
            .flat_map(|b| f.layout().block_insts(b))
            .filter_map(|i| f.dfg().first_result(i))
            .filter_map(|v| f.dfg().as_scalar_const(v))
            .map(|c| c.to_bits())
            .collect();
        assert_eq!(constants, [7]);
        module.validate().unwrap();
    }

    #[test]
    fn generated_rules_combine_with_wrapping_evaluation() {
        for ty in [Type::I8, Type::I16, Type::I32, Type::I64] {
            let mut graph = Graph::new(Budget::DEFAULT.graph_nodes);
            let binary = |graph: &mut Graph, opcode, a, b| {
                let op = graph.operation(
                    Operation {
                        opcode,
                        args: vec![a, b],
                        results: vec![ty],
                        properties: vec![],
                    },
                    None,
                );
                graph.add(Node::Result(op, 0)).unwrap()
            };
            let x = graph.add(Node::Input(Value(0), ty)).unwrap();
            let y = graph.add(Node::Input(Value(1), ty)).unwrap();
            let sum = binary(&mut graph, Op::IAdd, x, y);
            let cancel = binary(&mut graph, Op::ISub, sum, x);
            let max = graph
                .add(Node::Constant(
                    ScalarConst::from_bits(ty, u64::MAX >> (64 - ty.element_bits().unwrap()))
                        .unwrap(),
                ))
                .unwrap();
            let one = graph
                .add(Node::Constant(ScalarConst::from_bits(ty, 1).unwrap()))
                .unwrap();
            let left = binary(&mut graph, Op::IAdd, x, max);
            let wrapped = binary(&mut graph, Op::IAdd, left, one);
            let mut fuel = Budget::DEFAULT.match_steps;
            graph.saturate(Budget::DEFAULT.rounds, &mut fuel);
            assert_eq!(graph.find(cancel), graph.find(y), "{ty:?}");
            assert_eq!(graph.find(wrapped), graph.find(x), "{ty:?}");
            let extracted = graph.extract(&[wrapped], &GenericCost).unwrap();
            let selected = extracted[graph.find(wrapped)].unwrap();
            assert_eq!(graph.nodes[selected], Node::Input(Value(0), ty));
        }
    }
}
