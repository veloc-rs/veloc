//! Choose shared, acyclic expressions and place them at dominance-valid uses.
use super::{CostModel, graph::Graph};
use cranelift_entity::SecondaryMap;
use hashbrown::{HashMap, HashSet};
use smallvec::SmallVec;
use std::collections::VecDeque;
use veloc_mir::function::{Dominators, FrozenExpressions, InstOrder};
use veloc_mir::{FuncBody, Inst, Value, ValueDef};

/// Reused across extraction trials. Only visited entries need resetting.
#[derive(Default)]
struct PlanWorkspace {
    state: SecondaryMap<Value, u8>,
    touched: Vec<Value>,
    pending: Vec<(Value, bool)>,
    output: Vec<(Value, Value)>,
    operations: HashSet<Inst>,
}

impl Graph {
    fn price(&self, f: &FuncBody, value: Value, model: &dyn CostModel) -> usize {
        if let Some(c) = self.constant(value) {
            model.constant(c).max(1)
        } else if let Some(inst) = self.floating_inst(f, value) {
            model
                .operation(f.dfg().opcode(inst), f.dfg().value_type(value))
                .max(1)
        } else {
            0
        }
    }

    pub(super) fn extract(
        &self,
        f: &FuncBody,
        roots: &[Value],
        model: &dyn CostModel,
    ) -> Option<SecondaryMap<Value, Option<Value>>> {
        let mut costs = SecondaryMap::<Value, _>::with_default((usize::MAX, usize::MAX));
        let mut best = SecondaryMap::<Value, Option<Value>>::new();
        let mut pending = VecDeque::new();
        let mut queued = SecondaryMap::<Value, bool>::new();
        // A known constant is a terminal choice. Placement creates its literal
        // only if an executable use actually needs it.
        for &value in &self.values {
            let class = self.find(value);
            if let Some(literal) = self.constants[class] {
                if best[class].is_none() {
                    costs[class] = (model.constant(literal).max(1), 1);
                    best[class] = Some(class);
                }
            } else {
                pending.push_back(value);
                queued[value] = true;
            }
        }
        while let Some(value) = pending.pop_front() {
            queued[value] = false;
            let class = self.find(value);
            let mut price = self.price(f, value, model);
            let mut depth = 0usize;
            for &arg in self.args(f, value) {
                let arg = self.find(arg);
                price = price.saturating_add(costs[arg].0);
                depth = depth.max(costs[arg].1);
            }
            let depth = depth.saturating_add(usize::from(self.floating_inst(f, value).is_some()));
            if price != usize::MAX && (price, depth) < costs[class] {
                costs[class] = (price, depth);
                best[class] = Some(value);
                for &user in &self.users[class] {
                    for &result in f.dfg().inst_results(user) {
                        if self.constants[self.find(result)].is_none() && !queued[result] {
                            queued[result] = true;
                            pending.push_back(result);
                        }
                    }
                }
            }
        }
        let mut work = usize::MAX;
        let mut scratch = PlanWorkspace::default();
        if !self.plan(f, roots, &best, &mut work, &mut scratch) {
            return None;
        }
        let mut plan = std::mem::take(&mut scratch.output);
        let mut price = self.plan_price(f, &plan, model, &mut scratch.operations);
        let mut work = self.values.len().saturating_mul(16);
        let mut reachable = HashSet::new();
        loop {
            let mut improved = false;
            reachable.clear();
            reachable.extend(plan.iter().map(|&(class, _)| class));
            for &value in &self.values {
                let class = self.find(value);
                if work == 0 {
                    break;
                }
                work -= 1;
                if !reachable.contains(&class)
                    || best[class] == Some(value)
                    || self.constants[class].is_some()
                {
                    continue;
                }
                let old = best[class].replace(value);
                if self.plan(f, roots, &best, &mut work, &mut scratch) {
                    let candidate_price =
                        self.plan_price(f, &scratch.output, model, &mut scratch.operations);
                    if candidate_price < price {
                        std::mem::swap(&mut plan, &mut scratch.output);
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

    fn plan(
        &self,
        f: &FuncBody,
        roots: &[Value],
        best: &SecondaryMap<Value, Option<Value>>,
        work: &mut usize,
        scratch: &mut PlanWorkspace,
    ) -> bool {
        let PlanWorkspace {
            state,
            touched,
            pending,
            output,
            ..
        } = scratch;
        for class in touched.drain(..) {
            state[class] = 0;
        }
        pending.clear();
        output.clear();
        for &root in roots {
            pending.push((self.find(root), false));
            while let Some((class, ready)) = pending.pop() {
                if *work == 0 {
                    return false;
                }
                *work -= 1;
                if state[class] == 2 {
                    continue;
                }
                let Some(value) = best[class] else {
                    return false;
                };
                if state[class] == 0 {
                    touched.push(class);
                }
                let args = self.args(f, value);
                if !ready && !args.is_empty() {
                    if state[class] == 1 {
                        return false;
                    }
                    state[class] = 1;
                    pending.push((class, true));
                    pending.extend(args.iter().rev().map(|&arg| (self.find(arg), false)));
                } else {
                    state[class] = 2;
                    output.push((class, value));
                }
            }
        }
        true
    }

    fn plan_price(
        &self,
        f: &FuncBody,
        plan: &[(Value, Value)],
        model: &dyn CostModel,
        operations: &mut HashSet<Inst>,
    ) -> usize {
        operations.clear();
        plan.iter().fold(0usize, |price, &(_, value)| {
            if self.constant(value).is_none()
                && let Some(inst) = self.floating_inst(f, value)
                && !operations.insert(inst)
            {
                return price;
            }
            price.saturating_add(self.price(f, value, model))
        })
    }
}

/// Materialize the selected graph at actual uses. Reuse is constrained by SSA
/// dominance; different branches can receive separate copies of one candidate.
pub(super) struct Placement {
    dom: Dominators,
    order: InstOrder,
    available: SecondaryMap<Value, Vec<Value>>,
    boundaries: SecondaryMap<Value, Vec<Value>>,
    pending: Vec<(Value, bool)>,
    pub(super) local: HashMap<Value, Value>,
}

impl Placement {
    pub(super) fn new(f: &FuncBody, graph: &Graph) -> Self {
        let mut boundaries = SecondaryMap::<Value, Vec<Value>>::new();
        let mut available = SecondaryMap::<Value, Vec<Value>>::new();
        for &value in &graph.values {
            if f.dfg().as_scalar_const(value).is_some() {
                available[graph.find(value)].push(value);
            }
            if graph.floating_inst(f, value).is_none() {
                boundaries[graph.find(value)].push(value);
            }
        }
        Self {
            dom: Dominators::compute(f.cfg(), f.entry_block(), f.dfg().block_count()),
            order: InstOrder::default(),
            available,
            boundaries,
            pending: Vec::new(),
            local: HashMap::new(),
        }
    }

    fn dominates(&mut self, f: &FuncBody, value: Value, anchor: Inst) -> bool {
        let use_block = f.layout().inst_block(anchor).expect("placed use");
        match f.dfg().value_def(value) {
            ValueDef::Param(block) => block == use_block || self.dom.dominates(block, use_block),
            ValueDef::Inst(def) => {
                let Some(block) = f.layout().inst_block(def) else {
                    return false;
                };
                if block == use_block {
                    self.order.comes_before(f.layout(), def, anchor)
                } else {
                    self.dom.dominates(block, use_block)
                }
            }
        }
    }

    pub(super) fn materialize(
        &mut self,
        ir: &mut FrozenExpressions<'_>,
        anchor: Inst,
        root: Value,
        graph: &Graph,
        choices: &SecondaryMap<Value, Option<Value>>,
    ) -> Option<Value> {
        self.pending.clear();
        self.pending.push((root, false));
        while let Some((class, ready)) = self.pending.pop() {
            if self.local.contains_key(&class) {
                continue;
            }
            if let Some(value) = (0..self.available[class].len()).rev().find_map(|index| {
                let value = self.available[class][index];
                self.dominates(ir.body(), value, anchor).then_some(value)
            }) {
                self.local.insert(class, value);
                continue;
            }
            if let Some(constant) = graph.constant(class) {
                let value = ir.constant(anchor, constant);
                self.local.insert(class, value);
                self.available[class].push(value);
                continue;
            }
            let selected = choices[class]?;
            let Some(source) = graph.floating_inst(ir.body(), selected) else {
                // A single global choice may be unavailable here, although an
                // equivalent pinned value dominates this particular use.
                let value = if self.dominates(ir.body(), selected, anchor) {
                    selected
                } else {
                    (0..self.boundaries[class].len()).find_map(|index| {
                        let value = self.boundaries[class][index];
                        self.dominates(ir.body(), value, anchor).then_some(value)
                    })?
                };
                self.local.insert(class, value);
                self.available[class].push(value);
                continue;
            };
            let args = graph.args(ir.body(), selected);
            if !ready && !args.is_empty() {
                self.pending.push((class, true));
                self.pending
                    .extend(args.iter().rev().map(|&v| (graph.find(v), false)));
                continue;
            }
            let args: SmallVec<[Value; 3]> =
                args.iter().map(|&v| self.local[&graph.find(v)]).collect();
            let reuse = self.dominates(ir.body(), selected, anchor)
                && ir.body().dfg().operands(source) == args.as_slice();
            let original: SmallVec<[Value; 2]> = ir.body().dfg().inst_results(source).into();
            let inst = if reuse {
                source
            } else {
                ir.place(anchor, source, &args)
            };
            let actual = ir.body().dfg().inst_results(inst);
            let selected_index = original
                .iter()
                .position(|&v| v == selected)
                .expect("result membership");
            for (&old, &new) in original.iter().zip(actual) {
                let result_class = graph.find(old);
                // Another projection may already have a cheaper selected value.
                self.local.entry(result_class).or_insert(new);
                if !self.available[result_class].contains(&new) {
                    self.available[result_class].push(new);
                }
            }
            self.local.insert(class, actual[selected_index]);
        }
        self.local.get(&root).copied()
    }
}
