//! One equality-graph pipeline; fast mode changes budgets, not semantics.
use crate::{FunctionPass, Metrics, OptConfig, PreservedAnalyses};
use alloc::{vec, vec::Vec};
use hashbrown::{HashMap, HashSet};
use veloc_analyzer::AnalysisManager;
use veloc_mir::constant::ScalarConst;
use veloc_mir::{Function, Inst, IntCC, Opcode as Op, Type, Value};
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

pub fn run(func: &mut Function, budget: Budget, debug: bool, metrics: &mut Metrics) -> bool {
    run_with_cost(func, budget, &GenericCost, debug, metrics)
}

pub fn run_with_cost(
    func: &mut Function,
    budget: Budget,
    cost: &dyn CostModel,
    debug: bool,
    metrics: &mut Metrics,
) -> bool {
    let changed = optimize_regions(func, budget, cost, metrics);
    if changed && debug {
        log::info!("Optimized expression regions in {}", func.name);
    }
    changed
}

/// Deterministic limits shared by all regions in one function. Both profiles
/// run the same graph optimizer; neither is a separate greedy rewrite engine.
#[derive(Clone, Copy)]
pub struct Budget {
    pub region_nodes: usize,
    pub graph_nodes: usize,
    pub rounds: usize,
    pub match_steps: usize,
}

impl Budget {
    pub const FAST: Self = Self {
        region_nodes: 32,
        graph_nodes: 160,
        rounds: 2,
        match_steps: 16_384,
    };
    pub const DEFAULT: Self = Self {
        region_nodes: 96,
        graph_nodes: 512,
        rounds: 6,
        match_steps: 262_144,
    };
}

type Class = usize;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Operation {
    opcode: Op,
    args: Vec<Class>,
    results: Vec<Type>,
    properties: Vec<IntCC>,
    // Original instructions preserve arbitrary storage attributes when rebuilt.
    template: Option<Inst>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Node {
    Input(Value, Type),
    Constant(ScalarConst),
    Result(Operation, usize),
}
impl Node {
    fn ty(&self) -> Type {
        match self {
            Self::Input(_, ty) => *ty,
            Self::Constant(c) => c.ty(),
            Self::Result(op, index) => op.results[*index],
        }
    }
    fn args(&self) -> &[Class] {
        match self {
            Self::Result(op, _) => &op.args,
            _ => &[],
        }
    }
    fn price(&self, model: &dyn CostModel) -> usize {
        match self {
            Self::Input(..) => 0,
            Self::Constant(c) => model.constant(*c).max(1),
            Self::Result(op, _) => model.operation(op.opcode, op.results[0]).max(1),
        }
    }
}

struct Graph {
    nodes: Vec<Node>,
    parents: Vec<Class>,
    next: Vec<Class>,
    memo: HashMap<Node, Class>,
    revision: usize,
    limit: usize,
}
impl Graph {
    fn new(limit: usize) -> Self {
        Self {
            nodes: Vec::new(),
            parents: Vec::new(),
            next: Vec::new(),
            memo: HashMap::new(),
            revision: 0,
            limit,
        }
    }
    fn find(&self, mut id: Class) -> Class {
        while self.parents[id] != id {
            id = self.parents[id];
        }
        id
    }
    fn ty(&self, id: Class) -> Type {
        self.nodes[self.find(id)].ty()
    }
    fn normalize(&self, mut node: Node) -> Node {
        if let Node::Result(op, _) = &mut node {
            for arg in &mut op.args {
                *arg = self.find(*arg);
            }
            if op.opcode.spec().is_commutative() && op.args.len() == 2 && op.args[0] > op.args[1] {
                op.args.swap(0, 1);
            }
        }
        node
    }
    fn add(&mut self, node: Node) -> Option<Class> {
        let node = self.normalize(node);
        if let Some(&id) = self.memo.get(&node) {
            return Some(self.find(id));
        }
        if self.nodes.len() >= self.limit {
            return None;
        }
        let id = self.nodes.len();
        self.nodes.push(node.clone());
        self.parents.push(id);
        self.next.push(id);
        self.memo.insert(node, id);
        self.revision += 1;
        Some(id)
    }
    fn union(&mut self, a: Class, b: Class) {
        let (a, b) = (self.find(a), self.find(b));
        assert_eq!(self.ty(a), self.ty(b), "cannot equate different types");
        if a != b {
            self.next.swap(a, b);
            self.parents[a.max(b)] = a.min(b);
            self.revision += 1;
        }
    }
    fn members(&self, class: Class) -> impl Iterator<Item = &Node> {
        let start = self.find(class);
        let mut cursor = Some(start);
        core::iter::from_fn(move || {
            let id = cursor?;
            cursor = (self.next[id] != start).then_some(self.next[id]);
            Some(&self.nodes[id])
        })
    }
    fn constant(&self, class: Class) -> Option<ScalarConst> {
        self.members(class).find_map(|n| match n {
            Node::Constant(c) => Some(*c),
            _ => None,
        })
    }
    fn rebuild(&mut self) {
        loop {
            let before = self.revision;
            for id in 0..self.parents.len() {
                self.parents[id] = self.find(id);
            }
            self.memo.clear();
            for id in 0..self.nodes.len() {
                let node = self.normalize(self.nodes[id].clone());
                self.nodes[id] = node.clone();
                if let Some(other) = self.memo.insert(node, id) {
                    self.union(id, other);
                }
            }
            if before == self.revision {
                break;
            }
        }
    }
    fn saturate(&mut self, rounds: usize, fuel: &mut usize) {
        for _ in 0..rounds {
            self.rebuild();
            let before = self.revision;
            // Evaluate each multi-result operation once per round, not once per projection.
            let mut evaluated = HashMap::<Operation, Option<Vec<ScalarConst>>>::new();
            for id in 0..self.nodes.len() {
                if *fuel == 0 {
                    break;
                }
                *fuel -= 1;
                let Node::Result(op, index) = self.nodes[id].clone() else {
                    continue;
                };
                let folded = evaluated.entry(op.clone()).or_insert_with(|| {
                    let args = op
                        .args
                        .iter()
                        .map(|&a| self.constant(a))
                        .collect::<Option<Vec<_>>>()?;
                    crate::rewrite::evaluate(op.opcode, &args, &op.results, &op.properties)
                });
                if let Some(values) = folded {
                    if let Some(literal) = self.add(Node::Constant(values[index])) {
                        self.union(id, literal);
                    }
                }
                if op.results.len() != 1 {
                    continue;
                }
                let ty = op.results[0];
                if op.args.iter().any(|&a| self.ty(a) != ty) {
                    continue;
                }
                if let [a, b] = op.args.as_slice() {
                    let constants = [self.constant(*a), self.constant(*b)];
                    if let Some(replacement) =
                        crate::rewrite::algebraic(op.opcode, &[*a, *b], &constants)
                    {
                        match replacement {
                            crate::rewrite::Replacement::Value(value) => self.union(id, value),
                            crate::rewrite::Replacement::Constants(values) => {
                                if let Some(value) = self.add(Node::Constant(values[0])) {
                                    self.union(id, value);
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
                for rule in crate::equivalence::rules(op.opcode) {
                    if !rule.types.contains(&ty) {
                        continue;
                    }
                    for env in crate::equivalence::matches(&context, rule, id, mask, fuel) {
                        if let Some(value) =
                            crate::equivalence::emit(&mut context, &rule.replacement, &env)
                        {
                            context.graph.union(id, value);
                            log::trace!("egraph rule {}", rule.name);
                        }
                    }
                }
            }
            if self.revision == before || *fuel == 0 {
                break;
            }
        }
        self.rebuild();
    }
    fn extract(&self, roots: &[Class], model: &dyn CostModel) -> Option<Vec<(Class, Node)>> {
        let mut costs = vec![(usize::MAX, usize::MAX); self.nodes.len()];
        let mut best = vec![None; self.nodes.len()];
        for _ in 0..self.nodes.len() {
            let mut changed = false;
            for (id, node) in self.nodes.iter().enumerate() {
                let class = self.find(id);
                // A proven literal is a canonical result, including when its
                // operands were zero-cost SSA boundary inputs.
                if matches!(node, Node::Result(..)) && self.constant(class).is_some() {
                    continue;
                }
                let mut price = node.price(model);
                let mut depth = 0;
                for &arg in node.args() {
                    price = price.saturating_add(costs[arg].0);
                    depth = depth.max(costs[arg].1);
                }
                let depth = depth.saturating_add(usize::from(!matches!(node, Node::Input(..))));
                if price != usize::MAX && (price, depth) < costs[class] {
                    costs[class] = (price, depth);
                    best[class] = Some(id);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let mut seen = HashSet::new();
        let mut output = Vec::new();
        for &root in roots {
            let mut pending = vec![(self.find(root), false)];
            while let Some((class, ready)) = pending.pop() {
                if seen.contains(&class) {
                    continue;
                }
                let node = &self.nodes[best[class]?];
                if !ready && !node.args().is_empty() {
                    pending.push((class, true));
                    pending.extend(node.args().iter().rev().map(|&arg| (arg, false)));
                } else {
                    seen.insert(class);
                    output.push((class, node.clone()));
                }
            }
        }
        Some(output)
    }
}

struct RuleContext<'a> {
    graph: &'a mut Graph,
    ty: Type,
}
impl crate::equivalence::Context for RuleContext<'_> {
    type Value = Class;
    fn canonical(&self, value: Class) -> Class {
        self.graph.find(value)
    }
    fn constant(&self, value: Class) -> Option<u64> {
        self.graph.constant(value).map(|c| c.to_bits())
    }
    fn alternatives(&self, value: Class, opcode: Op, mut visit: impl FnMut(&[Class])) {
        for node in self.graph.members(value) {
            if let Node::Result(op, 0) = node
                && op.opcode == opcode
                && op.results == [self.ty]
                && op.args.iter().all(|&a| self.graph.ty(a) == self.ty)
            {
                let args: Vec<_> = op.args.iter().map(|&a| self.graph.find(a)).collect();
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
    fn literal(&mut self, value: u64) -> Option<Class> {
        let mask = u64::MAX >> (64 - self.ty.element_bits()?);
        self.graph.add(Node::Constant(ScalarConst::from_bits(
            self.ty,
            value & mask,
        )?))
    }
    fn build(&mut self, opcode: Op, args: &[Class]) -> Option<Class> {
        self.graph.add(Node::Result(
            Operation {
                opcode,
                args: args.to_vec(),
                results: vec![self.ty],
                properties: Vec::new(),
                template: None,
            },
            0,
        ))
    }
}

fn candidate(f: &Function, id: Inst) -> bool {
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

fn collect(f: &Function, id: Inst, ids: &mut Vec<Inst>, fuel: &mut usize, visited: &HashSet<Inst>) {
    if *fuel == 0 || ids.contains(&id) || visited.contains(&id) {
        return;
    }
    *fuel -= 1;
    for &input in f.dfg().operands(id) {
        if let Some(def) = f.dfg().value_inst(input)
            && candidate(f, def)
            && f.dfg().inst(def).can_speculate()
            && f.dfg()
                .inst_results(def)
                .iter()
                .all(|&v| f.dfg().uses(v).all(|site| site.inst() == id))
        {
            collect(f, def, ids, fuel, visited);
        }
    }
    ids.push(id);
}

fn optimize(
    f: &mut Function,
    root: Inst,
    ids: &[Inst],
    budget: Budget,
    fuel: &mut usize,
    model: &dyn CostModel,
) -> bool {
    if ids.len() == 1 && f.dfg().operands(root).is_empty() {
        return false;
    }
    let reserve: usize = ids
        .iter()
        .map(|&id| f.dfg().inst_results(id).len() + f.dfg().operands(id).len() * 2)
        .sum();
    let mut graph = Graph::new(budget.graph_nodes.max(reserve));
    let mut values = HashMap::new();
    for &id in ids {
        let results = f.dfg().inst_results(id);
        if let [dst] = results
            && let Some(c) = f.dfg().as_scalar_const(*dst)
        {
            values.insert(*dst, graph.add(Node::Constant(c)).unwrap());
            continue;
        }
        let args = f
            .dfg()
            .operands(id)
            .iter()
            .map(|&input| {
                *values.entry(input).or_insert_with(|| {
                    let class = graph
                        .add(Node::Input(input, f.dfg().value_type(input)))
                        .unwrap();
                    if let Some(c) = f.dfg().as_scalar_const(input) {
                        let literal = graph.add(Node::Constant(c)).unwrap();
                        graph.union(class, literal);
                    }
                    class
                })
            })
            .collect();
        let op = Operation {
            opcode: f.dfg().inst(id).opcode(),
            args,
            results: results.iter().map(|&v| f.dfg().value_type(v)).collect(),
            properties: crate::rewrite::properties(&f.dfg().inst(id)).into_vec(),
            template: Some(id),
        };
        for (index, &dst) in results.iter().enumerate() {
            values.insert(dst, graph.add(Node::Result(op.clone(), index)).unwrap());
        }
    }
    let results = f.dfg().inst_results(root).to_vec();
    let roots: Vec<_> = results.iter().map(|v| values[v]).collect();
    graph.saturate(budget.rounds, fuel);
    let Some(plan) = graph.extract(&roots, model) else {
        return false;
    };
    let mut operations = HashSet::new();
    let after = plan.iter().fold(0usize, |sum, (_, node)| {
        if let Node::Result(op, _) = node
            && !operations.insert(op.clone())
        {
            return sum;
        }
        sum.saturating_add(node.price(model))
    });
    let before = ids.iter().fold(0usize, |sum, &id| {
        let value = f.dfg().first_result(id).unwrap();
        sum.saturating_add(if let Some(c) = f.dfg().as_scalar_const(value) {
            model.constant(c).max(1)
        } else {
            model
                .operation(f.dfg().inst(id).opcode(), f.dfg().value_type(value))
                .max(1)
        })
    });
    // Fully evaluated roots may replace a single operation by one or several
    // literals even at equal/higher materialization cost: no execution remains.
    let all_constants = roots.iter().all(|&c| graph.constant(c).is_some());
    if after >= before && !all_constants {
        return false;
    }
    let mut materialized = vec![None; graph.nodes.len()];
    let mut emitted = HashMap::<Operation, Vec<Value>>::new();
    let mut edit = f.edit();
    for (class, node) in plan {
        let value = match node {
            Node::Input(value, _) => value,
            Node::Constant(c) => {
                let inst = edit.insert_before(root, |w| w.scalar_const(c), &[c.ty()]);
                edit.body().dfg().first_result(inst).unwrap()
            }
            Node::Result(op, index) => {
                if !emitted.contains_key(&op) {
                    let args: Vec<_> = op.args.iter().map(|&a| materialized[a].unwrap()).collect();
                    let inst = if let Some(template) = op.template {
                        let inst = edit.insert_before(root, |w| w.copy(template), &op.results);
                        for (i, &value) in args.iter().enumerate() {
                            edit.set_operand(inst, i as u32, value);
                        }
                        inst
                    } else {
                        edit.insert_before(
                            root,
                            |w| {
                                w.from_values(op.opcode, &args)
                                    .expect("value-only rule operation")
                            },
                            &op.results,
                        )
                    };
                    emitted.insert(op.clone(), edit.body().dfg().inst_results(inst).to_vec());
                }
                emitted[&op][index]
            }
        };
        materialized[class] = Some(value);
    }
    for (dst, class) in results.into_iter().zip(roots) {
        edit.replace_all_uses(dst, materialized[graph.find(class)].unwrap());
    }
    edit.erase_insts(ids);
    true
}

fn optimize_regions(
    f: &mut Function,
    budget: Budget,
    model: &dyn CostModel,
    metrics: &mut crate::Metrics,
) -> bool {
    let roots: Vec<_> = f
        .layout()
        .block_order()
        .flat_map(|b| f.layout().block_insts(b))
        .filter(|&id| candidate(f, id))
        .collect();
    let mut visited = HashSet::new();
    let mut ids = Vec::new();
    let mut fuel = budget.match_steps;
    let mut changed = 0;
    let mut regions = 0;
    for root in roots.into_iter().rev() {
        if fuel == 0 {
            break;
        }
        if visited.contains(&root) || !candidate(f, root) {
            continue;
        }
        ids.clear();
        let mut remaining = budget.region_nodes.max(1);
        collect(f, root, &mut ids, &mut remaining, &visited);
        visited.extend(ids.iter().copied());
        regions += 1;
        if optimize(f, root, &ids, budget, &mut fuel, model) {
            changed += 1;
        }
    }
    metrics.add("egraph.regions", regions);
    metrics.add("egraph.rewritten_regions", changed);
    metrics.add("egraph.match_steps", (budget.match_steps - fuel) as u64);
    changed != 0
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Metrics;

    #[test]
    fn cross_block_cones_preserve_effects_and_ssa() {
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
        let f = &mut module.functions[veloc_mir::FuncId(0)];
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
            let binary = |opcode, a, b| {
                Node::Result(
                    Operation {
                        opcode,
                        args: vec![a, b],
                        results: vec![ty],
                        properties: vec![],
                        template: None,
                    },
                    0,
                )
            };
            let x = graph.add(Node::Input(Value(0), ty)).unwrap();
            let y = graph.add(Node::Input(Value(1), ty)).unwrap();
            let sum = graph.add(binary(Op::IAdd, x, y)).unwrap();
            let cancel = graph.add(binary(Op::ISub, sum, x)).unwrap();
            let max = graph
                .add(Node::Constant(
                    ScalarConst::from_bits(ty, u64::MAX >> (64 - ty.element_bits().unwrap()))
                        .unwrap(),
                ))
                .unwrap();
            let one = graph
                .add(Node::Constant(ScalarConst::from_bits(ty, 1).unwrap()))
                .unwrap();
            let left = graph.add(binary(Op::IAdd, x, max)).unwrap();
            let wrapped = graph.add(binary(Op::IAdd, left, one)).unwrap();
            let mut fuel = Budget::DEFAULT.match_steps;
            graph.saturate(Budget::DEFAULT.rounds, &mut fuel);
            assert_eq!(graph.find(cancel), graph.find(y), "{ty:?}");
            assert_eq!(graph.find(wrapped), graph.find(x), "{ty:?}");
            let extracted = graph.extract(&[wrapped], &GenericCost).unwrap();
            assert_eq!(extracted.len(), 1);
            assert_eq!(extracted[0].1, Node::Input(Value(0), ty));
        }
    }
}
