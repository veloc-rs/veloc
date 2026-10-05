//! Eliminate redundant block parameters through their incoming dependencies.
//! A component with one external input aliases that value. Components with
//! multiple external inputs are searched for redundant inner components.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use cranelift_entity::SecondaryMap;
use veloc_analyzer::AnalysisManager;
use veloc_mir::{FuncBody, Value};

struct Parameters {
    values: Vec<Value>,
    incoming: SecondaryMap<Value, Vec<Value>>,
}

impl Parameters {
    fn new(f: &FuncBody) -> Self {
        let mut graph = Self {
            values: Vec::new(),
            incoming: SecondaryMap::new(),
        };
        for block in f.layout().block_order() {
            graph.values.extend_from_slice(f.dfg().block_params(block));
            let Some(terminator) = f.layout().last_inst(block) else {
                continue;
            };
            f.dfg().inst(terminator).visit_successors(|edge| {
                for (&param, &arg) in f.dfg().block_params(edge.block).iter().zip(edge.args) {
                    graph.incoming[param].push(arg);
                }
            });
        }
        graph
    }
}

pub struct SimplifyParamsPass;
impl FunctionPass for SimplifyParamsPass {
    fn reuse_key(&self) -> Option<core::any::TypeId> {
        Some(core::any::TypeId::of::<Self>())
    }
    fn name(&self) -> &'static str {
        "SimplifyParamsPass"
    }
    fn run(&self, am: &mut AnalysisManager<'_>, _: &OptConfig, metrics: &Profile) -> PassOutcome {
        let f = am.function_mut();
        let graph = Parameters::new(f);
        let mut replacements = Solver::new(&graph).solve();
        let changed = replacements.apply(f, &graph.values);
        metrics.count("params.forwarded", changed);
        metrics.count("params.removed", changed);
        if changed == 0 {
            PassOutcome::Unchanged
        } else {
            PassOutcome::Changed
        }
    }
}

/// Directed replacements into proven external SSA values.
#[derive(Default)]
struct Replacements {
    aliases: SecondaryMap<Value, Option<Value>>,
}

impl Replacements {
    fn resolve(&mut self, mut value: Value) -> Value {
        while let Some(next) = self.aliases[value] {
            // Shorten chains without changing which SSA value represents them.
            self.aliases[value] = Some(self.aliases[next].unwrap_or(next));
            value = next;
        }
        value
    }

    fn apply(&mut self, f: &mut FuncBody, params: &[Value]) -> u64 {
        let mut changed = 0;
        for &param in params {
            let value = self.resolve(param);
            if value != param {
                f.edit().replace_all_uses(param, value);
                changed += 1;
            }
        }
        if changed == 0 {
            return 0;
        }
        // All uses are replaced before any parameter positions are removed.
        // General dead-parameter elimination remains the responsibility of DCE.
        let blocks: Vec<_> = f.layout().block_order().collect();
        for block in blocks {
            let keep: Vec<_> = f
                .dfg()
                .block_params(block)
                .iter()
                .map(|&p| self.aliases[p].is_none())
                .collect();
            if keep.contains(&false) {
                f.edit().retain_block_params(block, &keep);
            }
        }
        changed
    }
}

struct Solver<'a> {
    graph: &'a Parameters,
    replacements: Replacements,
    components: Components,
    inside: SecondaryMap<Value, bool>,
}

impl<'a> Solver<'a> {
    fn new(graph: &'a Parameters) -> Self {
        Self {
            graph,
            replacements: Replacements::default(),
            components: Components::default(),
            inside: SecondaryMap::new(),
        }
    }

    fn solve(mut self) -> Replacements {
        let mut pending =
            self.components
                .compute(self.graph, &self.graph.values, &mut self.replacements);
        pending.reverse();
        while let Some(members) = pending.pop() {
            let inner = self.simplify(&members);
            if !inner.is_empty() {
                // Finish dependencies inside this component before its users.
                let components =
                    self.components
                        .compute(self.graph, &inner, &mut self.replacements);
                pending.extend(components.into_iter().rev());
            }
        }
        self.replacements
    }

    /// Braun et al., CC 2013, section 3.2. Singletons use the same rule as
    /// cycles; when a component varies, search parameters with only internal inputs.
    fn simplify(&mut self, members: &[Value]) -> Vec<Value> {
        for &member in members {
            self.inside[member] = true;
        }
        let mut outside = None;
        let mut different = false;
        let mut inner = Vec::new();
        for &member in members {
            let mut is_inner = true;
            for &arg in &self.graph.incoming[member] {
                let arg = self.replacements.resolve(arg);
                if !self.inside[arg] {
                    is_inner = false;
                    different |= outside.is_some_and(|value| value != arg);
                    outside = Some(arg);
                }
            }
            if is_inner {
                inner.push(member);
            }
        }
        for &member in members {
            self.inside[member] = false;
        }
        if different {
            // At least one member has external inputs, so this subset shrinks.
            return inner;
        }
        if let Some(value) = outside {
            for &member in members {
                self.replacements.aliases[member] = Some(value);
            }
        }
        // Without an external input there is no justified replacement value.
        Vec::new()
    }
}

#[derive(Clone, Copy, Default)]
enum Visit {
    #[default]
    Outside,
    Pending,
    Active {
        index: usize,
        low: usize,
    },
    Complete,
}

struct Frame {
    value: Value,
    next_input: usize,
}

/// Iterative Tarjan traversal of a parameter subset. Components are emitted
/// before their users. Scratch storage is reused for nested induced subgraphs.
#[derive(Default)]
struct Components {
    visits: SecondaryMap<Value, Visit>,
    active: Vec<Value>,
    frames: Vec<Frame>,
    next_index: usize,
}

impl Components {
    fn compute(
        &mut self,
        graph: &Parameters,
        members: &[Value],
        replacements: &mut Replacements,
    ) -> Vec<Vec<Value>> {
        self.next_index = 0;
        for &member in members {
            self.visits[member] = Visit::Pending;
        }
        let mut result = Vec::new();
        for &member in members {
            if !matches!(self.visits[member], Visit::Pending) {
                continue;
            }
            self.enter(member);
            while let Some(frame) = self.frames.last_mut() {
                let value = frame.value;
                if let Some(&arg) = graph.incoming[value].get(frame.next_input) {
                    frame.next_input += 1;
                    let arg = replacements.resolve(arg);
                    match self.visits[arg] {
                        Visit::Pending => self.enter(arg),
                        Visit::Active { index, .. } => self.update_low_link(value, index),
                        Visit::Outside | Visit::Complete => {}
                    }
                    continue;
                }
                self.frames.pop();
                let Visit::Active { index, low } = self.visits[value] else {
                    unreachable!("DFS frame is active");
                };
                if low == index {
                    let mut component = Vec::new();
                    loop {
                        let member = self.active.pop().expect("component root is active");
                        self.visits[member] = Visit::Complete;
                        component.push(member);
                        if member == value {
                            break;
                        }
                    }
                    result.push(component);
                } else if let Some(parent) = self.frames.last() {
                    self.update_low_link(parent.value, low);
                }
            }
        }
        for &member in members {
            self.visits[member] = Visit::Outside;
        }
        result
    }

    fn enter(&mut self, value: Value) {
        let index = self.next_index;
        self.next_index += 1;
        self.visits[value] = Visit::Active { index, low: index };
        self.active.push(value);
        self.frames.push(Frame {
            value,
            next_input: 0,
        });
    }

    fn update_low_link(&mut self, value: Value, reachable: usize) {
        let Visit::Active { low, .. } = &mut self.visits[value] else {
            unreachable!("only active nodes have a low-link");
        };
        *low = (*low).min(reachable);
    }
}
