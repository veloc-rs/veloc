//! Select expressions at executable uses, then commit a dominance-valid plan.
use super::{
    CostModel, Limit,
    graph::{Graph, InstKind, Root},
};
use cranelift_entity::SecondaryMap;
use hashbrown::HashSet;
use smallvec::SmallVec;
use std::collections::VecDeque;
use veloc_analyzer::Dominators;
use veloc_mir::function::{FrozenExpressions, InstOrder};
use veloc_mir::{Block, FuncBody, Inst, Value, ValueDef};

/// References either executable MIR or one result of a planned instruction.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Input {
    Existing(Value),
    Result { step: StepId, index: usize },
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct StepId(usize);

struct PlannedInst {
    before: Inst,
    source: Inst,
    args: SmallVec<[Input; 3]>,
}

struct Rewrite {
    inst: Inst,
    operand: u32,
    input: Input,
}

/// An accepted plan with fixed instruction IDs, analyzed liveness and changed
/// uses. Live instructions include all their dependencies, which precede them
/// in the plan. Only applying this plan edits executable MIR.
#[derive(Default)]
pub(super) struct Extraction {
    steps: Vec<PlannedInst>,
    live: Vec<bool>,
    rewrites: Vec<Rewrite>,
}

impl Extraction {
    pub(super) fn apply(self, ir: &mut FrozenExpressions<'_>) -> u64 {
        // Keep result slots indexed by the original StepId, including dead slots.
        let mut results = vec![SmallVec::<[Value; 2]>::new(); self.steps.len()];
        for (id, step) in self.steps.into_iter().enumerate() {
            if !self.live[id] {
                continue;
            }
            let args: SmallVec<[Value; 3]> =
                step.args.iter().map(|arg| arg.value(&results)).collect();
            let inst = ir.place(step.before, step.source, &args);
            results[id] = ir.body().dfg().inst_results(inst).into();
        }
        let changed = self.rewrites.len() as u64;
        for rewrite in self.rewrites {
            let value = rewrite.input.value(&results);
            ir.replace_input(rewrite.inst, rewrite.operand, value);
        }
        changed
    }
}

impl Input {
    fn value(self, results: &[SmallVec<[Value; 2]>]) -> Value {
        match self {
            Self::Existing(value) => value,
            Self::Result { step, index } => results[step.0][index],
        }
    }
}

/// Search checkpoints cover only tentative instructions and bindings. Already
/// selected uses are never rolled back by a later operand's search.
#[derive(Clone, Copy)]
struct Mark {
    steps: usize,
    bound: usize,
}

#[derive(Default)]
struct Draft {
    // Instruction IDs are shared by all anchors. Successful searches can leave
    // unused instructions here; later uses may still reuse their result bindings.
    steps: Vec<PlannedInst>,
    // Keep unchanged uses until final DAG analysis: they can retain old work.
    uses: Vec<Rewrite>,
    bindings: Bindings,
}

impl Draft {
    fn mark(&self) -> Mark {
        Mark {
            steps: self.steps.len(),
            bound: self.bindings.bound.len(),
        }
    }

    fn restore(&mut self, mark: Mark) {
        self.steps.truncate(mark.steps);
        self.bindings.restore(mark.bound);
    }
}

/// One available occurrence per class. Bindings never shadow each other;
/// both candidate rollback and dominator scopes remove an appended suffix.
#[derive(Default)]
struct Bindings {
    inputs: SecondaryMap<Root, Option<Input>>,
    bound: Vec<Root>,
    scopes: Vec<usize>,
}

impl Bindings {
    fn enter(&mut self) {
        self.scopes.push(self.bound.len());
    }

    fn leave(&mut self) {
        let mark = self.scopes.pop().expect("dominator scope");
        self.restore(mark);
    }

    fn restore(&mut self, mark: usize) {
        for class in self.bound.drain(mark..) {
            self.inputs[class] = None;
        }
    }

    fn bind(&mut self, class: Root, input: Input) {
        debug_assert!(self.inputs[class].is_none());
        self.inputs[class] = Some(input);
        self.bound.push(class);
    }
}

struct Analysis {
    cost: usize,
    live: Vec<bool>,
}

/// Explicit DFS frames avoid native-stack growth for deep expressions. Each
/// frame tries alternatives until all dependencies can be placed at this use.
struct Frame {
    class: Root,
    candidates: Candidates,
    state: State,
    mark: Mark,
}

/// Frame lifetime and cycle detection must advance together on every exit path.
#[derive(Default)]
struct SearchStack {
    frames: Vec<Frame>,
    active: SecondaryMap<Root, bool>,
}

impl SearchStack {
    fn push(&mut self, class: Root, mark: Mark) {
        debug_assert!(!self.active[class]);
        self.frames.push(Frame::new(class, mark));
        self.active[class] = true;
    }

    fn pop(&mut self) {
        let frame = self.frames.pop().expect("active search frame");
        self.active[frame.class] = false;
    }

    fn clear(&mut self) {
        for frame in self.frames.drain(..) {
            self.active[frame.class] = false;
        }
    }
}

#[derive(Clone, Copy)]
enum State {
    Start,
    Choose,
    Inputs { value: Value, next: usize },
}

enum Resolve {
    Ready(Input),
    Unavailable,
}

/// Try the preferred value once, then the ranked alternatives without repeats.
#[derive(Default)]
struct Candidates {
    preferred: Option<Value>,
    next: usize,
}

impl Candidates {
    fn next(&mut self, ranked: &[Candidate]) -> Option<Value> {
        if self.next == 0 {
            self.next = 1;
            if let Some(value) = self.preferred {
                return Some(value);
            }
        }
        while let Some(candidate) = ranked.get(self.next - 1) {
            self.next += 1;
            if Some(candidate.value) != self.preferred {
                return Some(candidate.value);
            }
        }
        None
    }
}

impl Frame {
    fn new(class: Root, mark: Mark) -> Self {
        Self {
            class,
            candidates: Candidates::default(),
            state: State::Start,
            mark,
        }
    }

    fn retry(&mut self, draft: &mut Draft) {
        draft.restore(self.mark);
        self.state = State::Choose;
    }
}

/// Enter/leave events let selection use a single binding per class. Leaving a
/// dominator subtree rolls bindings back, so siblings never see each other's
/// newly materialized values.
enum Visit {
    Enter,
    Anchor(Inst),
    Leave,
}

/// Tree cost first, then depth. Unknown estimates sort after finite ones.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Rank {
    cost: usize,
    depth: usize,
}

impl Rank {
    const ZERO: Self = Self { cost: 0, depth: 0 };
    const UNKNOWN: Self = Self {
        cost: usize::MAX,
        depth: usize::MAX,
    };

    fn operation(cost: usize, inputs: impl Iterator<Item = Self>) -> Self {
        let mut rank = Self {
            cost: cost.max(1),
            depth: 0,
        };
        for input in inputs {
            rank.cost = rank.cost.saturating_add(input.cost);
            rank.depth = rank.depth.max(input.depth);
        }
        rank.depth = rank.depth.saturating_add(1);
        rank
    }
}

impl Default for Rank {
    fn default() -> Self {
        Self::UNKNOWN
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    value: Value,
    rank: Rank,
}

struct Planner<'a> {
    graph: &'a Graph,
    body: &'a FuncBody,
    work: &'a mut usize,
    candidates: SecondaryMap<Root, SmallVec<[Candidate; 2]>>,
    dom: &'a Dominators,
    order: InstOrder,
}

impl Graph {
    fn estimate_rank(
        &self,
        f: &FuncBody,
        value: Value,
        model: &dyn CostModel,
        best: &SecondaryMap<Root, Rank>,
    ) -> Rank {
        if let Some(inst) = self.floating_inst(f, value) {
            let result = f.dfg().inst_results(inst)[0];
            Rank::operation(
                model.operation(f.dfg().opcode(inst), f.dfg().value_type(result)),
                self.args(f, value).iter().map(|&arg| best[self.find(arg)]),
            )
        } else {
            // A pinned definition is an existing input to extraction. Its own
            // execution cost is outside the movable expression being rebuilt.
            Rank::ZERO
        }
    }

    /// Rank candidates globally, but do not commit to one expression per class.
    /// These tree costs are only search hints; placement checks availability and
    /// charges each newly planned instruction once, including multi-result ops.
    /// Repeated dependencies are intentionally counted repeatedly here. Exact
    /// sharing depends on the selected occurrences, not just class membership.
    /// The budget bounds propagation; initialization and final candidate ranking
    /// still scan the graph once each, even when propagation runs out of fuel.
    fn candidates(
        &self,
        f: &FuncBody,
        model: &dyn CostModel,
        work: &mut usize,
    ) -> SecondaryMap<Root, SmallVec<[Candidate; 2]>> {
        let scope = self.profile.scope("egraph.rank", 0);
        let mut best = SecondaryMap::<Root, Rank>::new();
        let mut queued = SecondaryMap::<Value, bool>::new();
        let mut pending = VecDeque::new();
        let values = self.values.iter().copied().filter(|&value| {
            !f.dfg()
                .value_inst(value)
                .is_some_and(|inst| self.kinds[inst] == InstKind::Folded)
        });
        for value in values.clone() {
            let class = self.find(value);
            // A literal is the answer, not an alternative to rank or place.
            if f.dfg().as_const(class.value()).is_some() {
                best[class] = Rank::ZERO;
                continue;
            }
            pending.push_back(value);
            queued[value] = true;
        }
        while *work != 0 {
            let Some(value) = pending.pop_front() else {
                break;
            };
            *work -= 1;
            queued[value] = false;
            let class = self.find(value);
            let rank = self.estimate_rank(f, value, model, &best);
            if rank.cost != usize::MAX && rank < best[class] {
                best[class] = rank;
                for &user in self.users(class) {
                    for &result in f.dfg().inst_results(user) {
                        if f.dfg().as_const(self.find(result).value()).is_none() && !queued[result]
                        {
                            queued[result] = true;
                            pending.push_back(result);
                        }
                    }
                }
            }
        }
        if !pending.is_empty() {
            self.profile.count("budget_stops", 1);
            log::debug!("egraph ranking limited: {:?}", Limit::RankWork);
        }
        // Use the same final class estimates for every candidate, including
        // candidates still pending when propagation exhausted its budget.
        let mut candidates = SecondaryMap::<Root, SmallVec<[Candidate; 2]>>::new();
        for value in values {
            let class = self.find(value);
            if f.dfg().as_const(class.value()).is_none() {
                candidates[class].push(Candidate {
                    value,
                    rank: self.estimate_rank(f, value, model, &best),
                });
            }
        }
        for values in candidates.values_mut() {
            // Dependencies may still have unknown costs after a budget stop.
            // Their candidates sort last, but remain eligible to plan.
            values.sort_by_key(|candidate| (candidate.rank, candidate.value));
        }
        scope.success();
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
            log::debug!("egraph extraction limited: {:?}", Limit::ExtractWork);
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
        };
        let visits = planner.visits(anchors);
        let draft = planner.select(&visits);
        if *planner.work == 0 {
            self.profile.count("budget_stops", 1);
            log::debug!("egraph extraction limited: {:?}", Limit::ExtractWork);
        }
        planner.finish(draft, anchors, model)
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

    /// Build a draft in dominance order. MIR stays frozen until the complete
    /// draft passes the cost check; candidates can still backtrack locally.
    /// When search runs out of fuel, retain completed operand plans and fill
    /// remaining uses with their original values so DAG pricing stays complete.
    fn select(&mut self, visits: &[Visit]) -> Draft {
        let scope = self.graph.profile.scope("egraph.select", 0);
        let mut draft = Draft::default();
        let mut stack = SearchStack::default();
        for visit in visits {
            let anchor = match *visit {
                Visit::Enter => {
                    draft.bindings.enter();
                    continue;
                }
                Visit::Leave => {
                    // Completed uses keep their instructions. Only availability
                    // ends here, so siblings cannot reuse this subtree's values.
                    draft.bindings.leave();
                    continue;
                }
                Visit::Anchor(anchor) => anchor,
            };
            for (operand, &root) in self.body.dfg().operands(anchor).iter().enumerate() {
                let class = self.graph.find(root);
                let input = if *self.work == 0 {
                    Input::Existing(root)
                } else {
                    match self.resolve(anchor, class, &mut draft, &mut stack) {
                        Ok(Resolve::Ready(input)) => input,
                        Ok(Resolve::Unavailable) => {
                            draft.bindings.bind(class, Input::Existing(root));
                            Input::Existing(root)
                        }
                        Err(limit) => {
                            debug_assert_eq!(limit, Limit::ExtractWork);
                            Input::Existing(root)
                        }
                    }
                };
                draft.uses.push(Rewrite {
                    inst: anchor,
                    operand: operand as u32,
                    input,
                });
            }
        }
        scope.success();
        draft
    }

    /// Accept the full computation DAG before filtering unchanged uses. Preserve
    /// instruction IDs and transfer analyzed liveness directly to the final plan;
    /// search bindings are discarded, and emission only reads the liveness mask.
    fn finish(&self, draft: Draft, anchors: &[Inst], model: &dyn CostModel) -> Extraction {
        let original = anchors
            .iter()
            .flat_map(|&inst| self.body.dfg().operands(inst).iter().copied())
            .map(Input::Existing);
        let original_cost = self.analyze(original, &[], model).cost;
        let analysis = self.analyze(
            draft.uses.iter().map(|usage| usage.input),
            &draft.steps,
            model,
        );
        if analysis.cost > original_cost {
            return Extraction::default();
        }

        let rewrites = draft
            .uses
            .into_iter()
            .filter(|usage| {
                usage.input
                    != Input::Existing(self.body.dfg().operands(usage.inst)[usage.operand as usize])
            })
            .collect();
        Extraction {
            steps: draft.steps,
            live: analysis.live,
            rewrites,
        }
    }

    /// Charge the selected computation DAG, including original definitions kept
    /// by any root. Counting only new instructions would incorrectly make a reused
    /// instruction free, or miss an old computation still needed by another use.
    /// Liveness belongs to occurrences, never to equivalence classes.
    /// Analysis is a complete traversal, independent of candidate-search fuel.
    /// Liveness is returned separately; the draft itself stays unchanged.
    fn analyze(
        &self,
        inputs: impl Iterator<Item = Input>,
        steps: &[PlannedInst],
        model: &dyn CostModel,
    ) -> Analysis {
        let scope = self.graph.profile.scope("egraph.cost", 0);
        let mut seen = HashSet::new();
        let mut pending: Vec<_> = inputs.collect();
        let mut analysis = Analysis {
            cost: 0,
            live: vec![false; steps.len()],
        };
        while let Some(input) = pending.pop() {
            let inst = match input {
                Input::Existing(value) => {
                    let ValueDef::Inst(inst) = self.body.dfg().value_def(value) else {
                        continue;
                    };
                    if !seen.insert(inst) {
                        continue;
                    }
                    if matches!(
                        self.graph.kinds[inst],
                        InstKind::Floating | InstKind::Folded
                    ) {
                        pending.extend(
                            self.body
                                .dfg()
                                .operands(inst)
                                .iter()
                                .copied()
                                .map(Input::Existing),
                        );
                        inst
                    } else {
                        continue;
                    }
                }
                Input::Result { step, .. } => {
                    if analysis.live[step.0] {
                        continue;
                    }
                    analysis.live[step.0] = true;
                    let step = &steps[step.0];
                    pending.extend(step.args.iter().copied());
                    step.source
                }
            };
            let result = self.body.dfg().inst_results(inst)[0];
            let price = model
                .operation(
                    self.body.dfg().opcode(inst),
                    self.body.dfg().value_type(result),
                )
                .max(1);
            analysis.cost = analysis.cost.saturating_add(price);
        }
        scope.success();
        analysis
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
            ValueDef::Const(_) => true,
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
    fn preferred(&mut self, class: Root, anchor: Inst) -> Option<Value> {
        let mut best = None;
        let mut rank = Rank::UNKNOWN;
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

    /// Record a floating instruction after all of its operand classes are bound.
    /// Reuse the original occurrence when possible, otherwise append one instruction;
    /// bind every result so later dependencies can share the same computation.
    fn plan_instruction(&mut self, anchor: Inst, value: Value, draft: &mut Draft) {
        debug_assert!(draft.bindings.inputs[self.graph.find(value)].is_none());
        let args = self.graph.args(self.body, value);
        let source = self
            .graph
            .floating_inst(self.body, value)
            .expect("floating expression");
        let args: SmallVec<[Input; 3]> = args
            .iter()
            .map(|&arg| draft.bindings.inputs[self.graph.find(arg)].expect("planned operand"))
            .collect();
        let reuse = self.dominates(value, anchor)
            && self
                .body
                .dfg()
                .operands(source)
                .iter()
                .zip(&args)
                .all(|(&old, &new)| new == Input::Existing(old));
        let step = if reuse {
            None
        } else {
            let step = StepId(draft.steps.len());
            draft.steps.push(PlannedInst {
                before: anchor,
                source,
                args,
            });
            Some(step)
        };
        // Record the instruction and all its result bindings before the next
        // checkpoint. Rollback removes them together; success makes every result
        // available to later operands and dominated anchors without publishing.
        for (index, &result) in self.body.dfg().inst_results(source).iter().enumerate() {
            let class = self.graph.find(result);
            if draft.bindings.inputs[class].is_some() {
                continue;
            }
            let input = if self.body.dfg().as_const(class.value()).is_some() {
                Input::Existing(class.value())
            } else {
                step.map_or(Input::Existing(result), |step| Input::Result {
                    step,
                    index,
                })
            };
            draft.bindings.bind(class, input);
        }
    }

    /// Resolve one operand transactionally. Success retains its dependencies;
    /// failure restores the incoming draft. Every exit leaves the search empty.
    fn resolve(
        &mut self,
        anchor: Inst,
        root: Root,
        draft: &mut Draft,
        stack: &mut SearchStack,
    ) -> Result<Resolve, Limit> {
        debug_assert!(stack.frames.is_empty());
        if let Some(input) = draft.bindings.inputs[root] {
            return Ok(Resolve::Ready(input));
        }
        let mark = draft.mark();
        stack.push(root, mark);
        let outcome = loop {
            let Some(frame) = stack.frames.last_mut() else {
                break Ok(Resolve::Ready(
                    draft.bindings.inputs[root].expect("resolved root"),
                ));
            };
            if *self.work == 0 {
                break Err(Limit::ExtractWork);
            }
            *self.work -= 1;
            let class = frame.class;
            // A dependency can also produce another result that satisfies
            // this suspended frame. No further candidate search is needed.
            if draft.bindings.inputs[class].is_some() {
                stack.pop();
                continue;
            }
            match frame.state {
                State::Start => {
                    if self.body.dfg().as_const(class.value()).is_some() {
                        draft.bindings.bind(class, Input::Existing(class.value()));
                        stack.pop();
                        continue;
                    }

                    frame.candidates.preferred = self.preferred(class, anchor);
                    frame.state = State::Choose;
                }
                State::Choose => {
                    let Some(value) = frame.candidates.next(&self.candidates[class]) else {
                        // This dependency has no viable representative. Reject
                        // the parent's candidate, restoring its checkpoint.
                        stack.pop();
                        let Some(parent) = stack.frames.last_mut() else {
                            break Ok(Resolve::Unavailable);
                        };
                        parent.retry(draft);
                        continue;
                    };
                    if self.graph.floating_inst(self.body, value).is_none() {
                        if self.dominates(value, anchor) {
                            draft.bindings.bind(class, Input::Existing(value));
                            stack.pop();
                        }
                        continue;
                    }
                    frame.state = State::Inputs { value, next: 0 };
                }
                State::Inputs { value, next } => {
                    let args = self.graph.args(self.body, value);
                    if let Some(&arg) = args.get(next) {
                        let arg = self.graph.find(arg);
                        if draft.bindings.inputs[arg].is_some() {
                            frame.state = State::Inputs {
                                value,
                                next: next + 1,
                            };
                        } else if stack.active[arg] {
                            // A cyclic candidate cannot produce a finite plan.
                            // Keep the candidate cursor and try the next one.
                            frame.retry(draft);
                        } else {
                            stack.push(arg, draft.mark());
                        }
                        continue;
                    }
                    self.plan_instruction(anchor, value, draft);
                    stack.pop();
                }
            }
        };
        stack.clear();
        if !matches!(outcome, Ok(Resolve::Ready(_))) {
            draft.restore(mark);
        }
        outcome
    }
}
