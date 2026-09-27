//! Select expressions at executable uses, then commit a dominance-valid plan.
use super::{CostModel, graph::Graph};
use cranelift_entity::SecondaryMap;
use hashbrown::{HashMap, HashSet};
use smallvec::SmallVec;
use std::collections::VecDeque;
use veloc_analyzer::Dominators;
use veloc_mir::constant::ScalarConst;
use veloc_mir::function::{FrozenExpressions, InstOrder};
use veloc_mir::{Block, FuncBody, Inst, Value, ValueDef};

/// References either executable MIR or one result of a planned instruction.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Input {
    Existing(Value),
    Result { step: usize, index: usize },
}

enum Recipe {
    Constant(ScalarConst),
    Operation {
        source: Inst,
        args: SmallVec<[Input; 3]>,
    },
}

struct Step {
    before: Inst,
    recipe: Recipe,
    refs: usize,
}

struct Rewrite {
    inst: Inst,
    operand: u32,
    input: Input,
}

/// A complete plan, not a global class-to-value preference. Steps are in
/// dependency order, and each use names an occurrence available at that use.
/// Planning never edits MIR; failed trials leave no partially emitted code.
#[derive(Default)]
pub(super) struct Extraction {
    steps: Vec<Step>,
    rewrites: Vec<Rewrite>,
}

impl Extraction {
    pub(super) fn apply(self, ir: &mut FrozenExpressions<'_>) -> u64 {
        if self.rewrites.is_empty() {
            return 0;
        }
        let mut results = Vec::<SmallVec<[Value; 2]>>::with_capacity(self.steps.len());
        for step in self.steps {
            if step.refs == 0 {
                results.push(SmallVec::new());
                continue;
            }
            let values = match step.recipe {
                Recipe::Constant(value) => smallvec::smallvec![ir.constant(step.before, value)],
                Recipe::Operation { source, args } => {
                    let args: SmallVec<[Value; 3]> =
                        args.iter().map(|arg| arg.value(&results)).collect();
                    let inst = ir.place(step.before, source, &args);
                    ir.body().dfg().inst_results(inst).into()
                }
            };
            results.push(values);
        }
        let mut changed = 0;
        for rewrite in self.rewrites {
            let value = rewrite.input.value(&results);
            if ir.body().dfg().operands(rewrite.inst)[rewrite.operand as usize] != value {
                ir.replace_input(rewrite.inst, rewrite.operand, value);
                changed += 1;
            }
        }
        changed
    }
}

impl Input {
    fn value(self, results: &[SmallVec<[Value; 2]>]) -> Value {
        match self {
            Self::Existing(value) => value,
            Self::Result { step, index } => results[step][index],
        }
    }
}

/// A checkpoint only covers the current use's new dependencies. Restoring it
/// never walks the previously committed plan or changes existing MIR.
#[derive(Clone, Copy, Default)]
struct Mark {
    steps: usize,
    bound: usize,
    selected: usize,
}

#[derive(Default)]
struct Plan {
    steps: Vec<Recipe>,
    inputs: SecondaryMap<Value, Option<Input>>,
    bound: Vec<Value>,
    selected: Vec<(Value, Value)>,
    operations: HashMap<Inst, usize>,
}

impl Plan {
    fn mark(&self) -> Mark {
        Mark {
            steps: self.steps.len(),
            bound: self.bound.len(),
            selected: self.selected.len(),
        }
    }

    fn restore(&mut self, mark: Mark) {
        for recipe in self.steps.drain(mark.steps..) {
            if let Recipe::Operation { source, .. } = recipe {
                self.operations.remove(&source);
            }
        }
        for class in self.bound.drain(mark.bound..) {
            self.inputs[class] = None;
        }
        self.selected.truncate(mark.selected);
    }

    fn bind(&mut self, class: Value, input: Input) {
        debug_assert!(self.inputs[class].is_none());
        self.inputs[class] = Some(input);
        self.bound.push(class);
    }
}

/// Explicit DFS frames avoid native-stack growth for deep expressions. Each
/// frame tries alternatives until all dependencies can be placed at this use.
struct Frame {
    class: Value,
    next: usize,
    preferred: Option<Value>,
    selected: Option<Value>,
    arg: usize,
    mark: Mark,
}

impl Frame {
    fn new(class: Value, mark: Mark) -> Self {
        Self {
            class,
            next: 0,
            preferred: None,
            selected: None,
            arg: 0,
            mark,
        }
    }
}

/// A preference belongs to a use site, not globally to an equivalence class.
/// Sibling blocks may need different representatives of the same class.
type Choices = HashMap<(Inst, Value), Value>;

#[derive(Clone, Copy)]
struct Choice {
    anchor: Inst,
    class: Value,
    value: Value,
}

struct Selection {
    extraction: Extraction,
    choices: Vec<Choice>,
    cost: usize,
}

/// Enter/leave events let selection use a single binding per class. Leaving a
/// dominator subtree rolls bindings back, so siblings never see each other's
/// newly materialized values.
enum Visit {
    Enter,
    Anchor(Inst),
    Leave,
}

#[derive(Default)]
struct Scope {
    inputs: SecondaryMap<Value, Option<Input>>,
    undo: Vec<(Value, Option<Input>)>,
    marks: Vec<usize>,
}

impl Scope {
    fn bind(&mut self, class: Value, input: Input) {
        self.undo.push((class, self.inputs[class]));
        self.inputs[class] = Some(input);
    }

    fn leave(&mut self) {
        let mark = self.marks.pop().expect("dominator scope");
        for (class, old) in self.undo.drain(mark..).rev() {
            self.inputs[class] = old;
        }
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    value: Value,
    rank: (usize, usize),
}

struct Planner<'a> {
    graph: &'a Graph,
    body: &'a FuncBody,
    work: &'a mut usize,
    candidates: SecondaryMap<Value, SmallVec<[Candidate; 2]>>,
    dom: &'a Dominators,
    order: InstOrder,
    available: Scope,
    active: SecondaryMap<Value, bool>,
}

impl Graph {
    fn price(&self, f: &FuncBody, value: Value, model: &dyn CostModel) -> usize {
        if let Some(c) = self.constant(value) {
            model.constant(c).max(1)
        } else if let Some(inst) = self.floating_inst(f, value) {
            let result = f.dfg().inst_results(inst)[0];
            model
                .operation(f.dfg().opcode(inst), f.dfg().value_type(result))
                .max(1)
        } else {
            // A pinned definition is an existing input to extraction. Its own
            // execution cost is outside the movable expression being rebuilt.
            0
        }
    }

    /// Rank candidates globally, but do not commit to one expression per class.
    /// These tree costs are only search hints; placement checks availability and
    /// charges each newly planned instruction once, including multi-result ops.
    /// Repeated dependencies are intentionally counted repeatedly here. Exact
    /// sharing depends on the selected occurrences, not just class membership.
    fn candidates(
        &self,
        f: &FuncBody,
        model: &dyn CostModel,
        work: &mut usize,
    ) -> SecondaryMap<Value, SmallVec<[Candidate; 2]>> {
        let mut costs = SecondaryMap::with_default((usize::MAX, usize::MAX));
        let mut ranks = SecondaryMap::with_default((usize::MAX, usize::MAX));
        let mut pending = VecDeque::new();
        let mut queued = SecondaryMap::<Value, bool>::new();
        let mut candidates = SecondaryMap::<Value, SmallVec<[Candidate; 2]>>::new();
        for &value in &self.values {
            let class = self.find(value);
            candidates[class].push(Candidate {
                value,
                rank: (usize::MAX, usize::MAX),
            });
            if let Some(c) = self.constants[class] {
                costs[class] = (model.constant(c).max(1), 0);
            } else {
                pending.push_back(value);
                queued[value] = true;
            }
        }
        while *work != 0 {
            let Some(value) = pending.pop_front() else {
                break;
            };
            *work -= 1;
            queued[value] = false;
            let class = self.find(value);
            let mut price = self.price(f, value, model);
            let mut depth = 0usize;
            for &arg in self.args(f, value) {
                let arg = self.find(arg);
                price = price.saturating_add(costs[arg].0);
                depth = depth.max(costs[arg].1);
            }
            depth = depth.saturating_add(usize::from(self.floating_inst(f, value).is_some()));
            ranks[value] = (price, depth);
            if price != usize::MAX && (price, depth) < costs[class] {
                costs[class] = (price, depth);
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
        for values in candidates.values_mut() {
            for candidate in values.iter_mut() {
                candidate.rank = ranks[candidate.value];
            }
            // A budget stop leaves partial estimates intact. Unestimated
            // candidates retain MAX and sort last, but remain eligible to plan.
            values.sort_by_key(|candidate| (candidate.rank, candidate.value));
        }
        candidates
    }

    pub(super) fn extract(
        &self,
        body: &FuncBody,
        anchors: &[Inst],
        model: &dyn CostModel,
        dom: &Dominators,
        rank: &mut usize,
        work: &mut usize,
    ) -> Extraction {
        if *work == 0 {
            return Extraction::default();
        }
        // Ranking is optional guidance, not a prerequisite for placement.
        // Even zero ranking fuel must leave all candidates available to search.
        let candidates = self.candidates(body, model, rank);
        let mut planner = Planner {
            graph: self,
            body,
            work,
            candidates,
            dom,
            order: InstOrder::default(),
            available: Scope::default(),
            active: SecondaryMap::new(),
        };
        let visits = planner.visits(anchors);
        let mut original = Extraction::default();
        for &inst in anchors {
            for (operand, &value) in body.dfg().operands(inst).iter().enumerate() {
                original.rewrites.push(Rewrite {
                    inst,
                    operand: operand as u32,
                    input: Input::Existing(value),
                });
            }
        }
        let Some(original_cost) = planner.cost(&mut original, model) else {
            return Extraction::default();
        };
        let mut choices = Choices::new();
        let Some(mut best) = planner.select(&visits, &choices, model) else {
            return Extraction::default();
        };

        // Search complete multi-root selections, not independently priced
        // trees. A paired move can expose sharing even when changing either
        // root alone would be more expensive. This is bounded local search,
        // not an exact minimum-cost DAG solver.
        while *planner.work != 0 {
            let moves = planner.alternatives(&best);
            let mut improved = None;
            'search: for (index, &a) in moves.iter().enumerate() {
                let old_a = choices.insert((a.anchor, a.class), a.value);
                if let Some(trial) = planner.select(&visits, &choices, model) {
                    if trial.cost < best.cost {
                        improved = Some(trial);
                        break;
                    }
                }
                for &b in &moves[index + 1..] {
                    if *planner.work == 0 {
                        break 'search;
                    }
                    *planner.work -= 1;
                    if (a.anchor, a.class) == (b.anchor, b.class)
                        || !planner.shares_dependency(a.value, b.value)
                    {
                        continue;
                    }
                    let old = choices.insert((b.anchor, b.class), b.value);
                    if let Some(trial) = planner.select(&visits, &choices, model) {
                        if trial.cost < best.cost {
                            improved = Some(trial);
                            break 'search;
                        }
                    }
                    restore_choice(&mut choices, b, old);
                }
                restore_choice(&mut choices, a, old_a);
            }
            let Some(trial) = improved else { break };
            best = trial;
            choices.clear();
            choices.extend(best.choices.iter().map(|c| ((c.anchor, c.class), c.value)));
        }
        if best.cost <= original_cost {
            best.extraction
        } else {
            Extraction::default()
        }
    }
}

fn restore_choice(choices: &mut Choices, choice: Choice, old: Option<Value>) {
    let key = (choice.anchor, choice.class);
    if let Some(value) = old {
        choices.insert(key, value);
    } else {
        choices.remove(&key);
    }
}

impl Planner<'_> {
    fn visits(&self, anchors: &[Inst]) -> Vec<Visit> {
        let mut uses = SecondaryMap::<Block, Vec<Inst>>::new();
        for &inst in anchors {
            uses[self.body.layout().inst_block(inst).expect("executable use")].push(inst);
        }
        let mut visits = Vec::new();
        let mut stack = vec![(self.body.entry_block(), false)];
        while let Some((block, leave)) = stack.pop() {
            if leave {
                visits.push(Visit::Leave);
                continue;
            }
            visits.push(Visit::Enter);
            visits.extend(uses[block].iter().copied().map(Visit::Anchor));
            stack.push((block, true));
            stack.extend(self.dom.children(block).map(|child| (child, false)));
        }
        visits
    }

    /// Replaying a trial changes only planning state. MIR remains frozen until
    /// a complete selection wins; budget exhaustion cannot leave partial edits.
    fn select(
        &mut self,
        visits: &[Visit],
        choices: &Choices,
        model: &dyn CostModel,
    ) -> Option<Selection> {
        self.available = Scope::default();
        let mut extraction = Extraction::default();
        let mut selected = Vec::new();
        let mut plan = Plan::default();
        let mut stack = Vec::new();
        for visit in visits {
            let anchor = match *visit {
                Visit::Enter => {
                    self.available.marks.push(self.available.undo.len());
                    continue;
                }
                Visit::Leave => {
                    self.available.leave();
                    continue;
                }
                Visit::Anchor(anchor) => anchor,
            };
            if *self.work == 0 {
                return None;
            }
            *self.work -= 1;
            plan.restore(Mark::default());
            let base = extraction.steps.len();
            for (operand, &root) in self.body.dfg().operands(anchor).iter().enumerate() {
                let class = self.graph.find(root);
                let mark = plan.mark();
                if !self.resolve(anchor, base, class, choices, &mut plan, &mut stack) {
                    if *self.work == 0 {
                        return None;
                    }
                    plan.restore(mark);
                    plan.bind(class, Input::Existing(root));
                }
                extraction.rewrites.push(Rewrite {
                    inst: anchor,
                    operand: operand as u32,
                    input: plan.inputs[class].expect("planned executable use"),
                });
            }
            selected.extend(plan.selected.iter().map(|&(class, value)| Choice {
                anchor,
                class,
                value,
            }));
            for &class in &plan.bound {
                self.available
                    .bind(class, plan.inputs[class].expect("bound class"));
            }
            // A multi-result instruction is one computation, not one per class.
            for (step, recipe) in plan.steps.iter().enumerate() {
                if let Recipe::Operation { source, .. } = recipe {
                    for (index, &value) in self.body.dfg().inst_results(*source).iter().enumerate()
                    {
                        let class = self.graph.find(value);
                        if self.available.inputs[class].is_none() {
                            self.available.bind(
                                class,
                                Input::Result {
                                    step: base + step,
                                    index,
                                },
                            );
                        }
                    }
                }
            }
            extraction
                .steps
                .extend(plan.steps.drain(..).map(|recipe| Step {
                    before: anchor,
                    recipe,
                    refs: 0,
                }));
            plan.operations.clear();
        }
        let cost = self.cost(&mut extraction, model)?;
        Some(Selection {
            extraction,
            choices: selected,
            cost,
        })
    }

    /// Charge the selected computation DAG, including original definitions kept
    /// by any root. Counting only new recipes would incorrectly make a reused
    /// instruction free, or miss an old computation still needed by another use.
    /// Reference counts belong to occurrences, never to equivalence classes.
    fn cost(&mut self, extraction: &mut Extraction, model: &dyn CostModel) -> Option<usize> {
        let mut seen = HashSet::new();
        let mut pending: Vec<_> = extraction.rewrites.iter().map(|r| r.input).collect();
        let mut cost = 0usize;
        while let Some(input) = pending.pop() {
            if *self.work == 0 {
                return None;
            }
            *self.work -= 1;
            let price = match input {
                Input::Existing(value) => {
                    let ValueDef::Inst(inst) = self.body.dfg().value_def(value) else {
                        continue;
                    };
                    if !seen.insert(inst) {
                        continue;
                    }
                    if let Some(c) = self.body.dfg().as_scalar_const(value) {
                        model.constant(c).max(1)
                    } else if self.graph.floating[inst] {
                        pending.extend(
                            self.body
                                .dfg()
                                .operands(inst)
                                .iter()
                                .copied()
                                .map(Input::Existing),
                        );
                        let result = self.body.dfg().inst_results(inst)[0];
                        model
                            .operation(
                                self.body.dfg().opcode(inst),
                                self.body.dfg().value_type(result),
                            )
                            .max(1)
                    } else {
                        continue;
                    }
                }
                Input::Result { step, .. } => {
                    let step = &mut extraction.steps[step];
                    step.refs += 1;
                    if step.refs != 1 {
                        continue;
                    }
                    match &step.recipe {
                        Recipe::Constant(c) => model.constant(*c).max(1),
                        Recipe::Operation { source, args } => {
                            pending.extend(args.iter().copied());
                            let value = self.body.dfg().inst_results(*source)[0];
                            model
                                .operation(
                                    self.body.dfg().opcode(*source),
                                    self.body.dfg().value_type(value),
                                )
                                .max(1)
                        }
                    }
                }
            };
            cost = cost.saturating_add(price);
        }
        Some(cost)
    }

    fn alternatives(&mut self, selection: &Selection) -> Vec<Choice> {
        let mut moves = Vec::new();
        for &choice in &selection.choices {
            for candidate in &self.candidates[choice.class] {
                if *self.work == 0 {
                    return moves;
                }
                *self.work -= 1;
                if candidate.value != choice.value {
                    moves.push(Choice {
                        value: candidate.value,
                        ..choice
                    });
                }
            }
        }
        moves
    }

    fn shares_dependency(&self, a: Value, b: Value) -> bool {
        self.graph.args(self.body, a).iter().any(|&x| {
            // Sharing a pinned input such as a block parameter has no saving
            // in this model; it should not trigger expensive paired trials.
            (self.graph.constant(x).is_some() || self.graph.floating_inst(self.body, x).is_some())
                && self
                    .graph
                    .args(self.body, b)
                    .iter()
                    .any(|&y| self.graph.find(x) == self.graph.find(y))
        })
    }

    fn inst_dominates(&mut self, def: Inst, anchor: Inst) -> bool {
        let Some(block) = self.body.layout().inst_block(def) else {
            return false;
        };
        let use_block = self
            .body
            .layout()
            .inst_block(anchor)
            .expect("executable use");
        if block == use_block {
            self.order.comes_before(self.body.layout(), def, anchor)
        } else {
            self.dom.dominates(block, use_block)
        }
    }

    fn dominates(&mut self, value: Value, anchor: Inst) -> bool {
        match self.body.dfg().value_def(value) {
            ValueDef::Param(block) => {
                let use_block = self
                    .body
                    .layout()
                    .inst_block(anchor)
                    .expect("executable use");
                block == use_block || self.dom.dominates(block, use_block)
            }
            ValueDef::Inst(def) => self.inst_dominates(def, anchor),
        }
    }

    /// On a cost tie prefer a definition already available here. Otherwise two
    /// sibling blocks could repeatedly copy each other's equivalent expression
    /// on every optimizer invocation instead of keeping their own definition.
    fn preferred(&mut self, class: Value, anchor: Inst) -> Option<Value> {
        let mut best = None;
        let mut rank = (usize::MAX, usize::MAX);
        for index in 0..self.candidates[class].len() {
            let candidate = self.candidates[class][index];
            if candidate.rank > rank {
                break;
            }
            let available = self.dominates(candidate.value, anchor);
            if !available
                && self
                    .graph
                    .floating_inst(self.body, candidate.value)
                    .is_none()
            {
                continue;
            }
            if best.is_none() {
                best = Some(candidate.value);
                rank = candidate.rank;
            }
            if available {
                return Some(candidate.value);
            }
        }
        best
    }

    fn resolve(
        &mut self,
        anchor: Inst,
        base: usize,
        root: Value,
        choices: &Choices,
        plan: &mut Plan,
        stack: &mut Vec<Frame>,
    ) -> bool {
        if plan.inputs[root].is_some() {
            return true;
        }
        debug_assert!(stack.is_empty());
        stack.push(Frame::new(root, plan.mark()));
        self.active[root] = true;
        while let Some(frame) = stack.last_mut() {
            if *self.work == 0 {
                for frame in stack.drain(..) {
                    self.active[frame.class] = false;
                }
                return false;
            }
            *self.work -= 1;
            let class = frame.class;
            if frame.selected.is_none() {
                if frame.next == 0 {
                    if let Some(input) = self.available.inputs[class] {
                        plan.bind(class, input);
                        self.active[class] = false;
                        stack.pop();
                        continue;
                    }
                    if let Some(c) = self.graph.constant(class) {
                        let existing = (0..self.candidates[class].len()).find_map(|index| {
                            let value = self.candidates[class][index].value;
                            (self.body.dfg().as_scalar_const(value).is_some()
                                && self.dominates(value, anchor))
                            .then_some(value)
                        });
                        let input = if let Some(value) = existing {
                            Input::Existing(value)
                        } else {
                            let step = base + plan.steps.len();
                            plan.steps.push(Recipe::Constant(c));
                            Input::Result { step, index: 0 }
                        };
                        plan.bind(class, input);
                        self.active[class] = false;
                        stack.pop();
                        continue;
                    }
                    frame.preferred = choices
                        .get(&(anchor, class))
                        .copied()
                        .or_else(|| self.preferred(class, anchor));
                }
                // An explicit preference comes first, followed by cost-ranked
                // alternatives. Unavailable leaves and cycles reject only this
                // candidate, not the entire extraction.
                let preferred = frame.preferred;
                let value = if frame.next == 0 && preferred.is_some() {
                    preferred
                } else {
                    self.candidates[class]
                        .get(frame.next - usize::from(preferred.is_some()))
                        .map(|candidate| candidate.value)
                };
                frame.next += 1;
                let Some(value) = value else {
                    plan.restore(frame.mark);
                    self.active[class] = false;
                    stack.pop();
                    let Some(parent) = stack.last_mut() else {
                        return false;
                    };
                    plan.restore(parent.mark);
                    parent.selected = None;
                    parent.arg = 0;
                    continue;
                };
                if frame.next > 1 && preferred == Some(value) {
                    continue;
                }
                if self.graph.floating_inst(self.body, value).is_none() {
                    if self.dominates(value, anchor) {
                        plan.bind(class, Input::Existing(value));
                        self.active[class] = false;
                        stack.pop();
                    }
                    continue;
                }
                frame.selected = Some(value);
            }
            let value = frame.selected.expect("selected floating expression");
            let args = self.graph.args(self.body, value);
            if let Some(&arg) = args.get(frame.arg) {
                let arg = self.graph.find(arg);
                if plan.inputs[arg].is_some() {
                    frame.arg += 1;
                } else if self.active[arg] {
                    plan.restore(frame.mark);
                    frame.selected = None;
                    frame.arg = 0;
                } else {
                    stack.push(Frame::new(arg, plan.mark()));
                    self.active[arg] = true;
                }
                continue;
            }
            let source = self
                .graph
                .floating_inst(self.body, value)
                .expect("floating expression");
            let args: SmallVec<[Input; 3]> = args
                .iter()
                .map(|&arg| plan.inputs[self.graph.find(arg)].expect("planned operand"))
                .collect();
            let results = self.body.dfg().inst_results(source);
            let index = results
                .iter()
                .position(|&v| v == value)
                .expect("result membership");
            let reuse = self.dominates(value, anchor)
                && self
                    .body
                    .dfg()
                    .operands(source)
                    .iter()
                    .zip(&args)
                    .all(|(&old, &new)| new == Input::Existing(old));
            let input = if reuse {
                Input::Existing(value)
            } else {
                let step = if let Some(&step) = plan.operations.get(&source) {
                    step
                } else {
                    let step = base + plan.steps.len();
                    plan.steps.push(Recipe::Operation { source, args });
                    plan.operations.insert(source, step);
                    step
                };
                Input::Result { step, index }
            };
            plan.bind(class, input);
            plan.selected.push((class, value));
            self.active[class] = false;
            stack.pop();
        }
        true
    }
}
