//! Propagate identical incoming SSA values through cycles of block parameters.
//! Each strongly connected component is resolved against its external inputs.
//! A varying parameter remains an opaque SSA value for downstream components.
use crate::{FunctionPass, OptConfig, PreservedAnalyses, Profile};
use cranelift_entity::SecondaryMap;
use veloc_analyzer::AnalysisManager;
use veloc_mir::{FuncBody, Value};

struct Parameters {
    values: Vec<Value>,
    is_param: SecondaryMap<Value, bool>,
    incoming: SecondaryMap<Value, Vec<Value>>,
}

impl Parameters {
    fn new(f: &FuncBody) -> Self {
        let mut graph = Self {
            values: Vec::new(),
            is_param: SecondaryMap::new(),
            incoming: SecondaryMap::new(),
        };
        for block in f.layout().block_order().filter(|&b| b != f.entry_block()) {
            for &param in f.dfg().block_params(block) {
                graph.values.push(param);
                graph.is_param[param] = true;
            }
        }
        for block in f.layout().block_order() {
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
    fn name(&self) -> &'static str {
        "SimplifyParamsPass"
    }
    fn run(
        &self,
        am: &mut AnalysisManager<'_>,
        _: &OptConfig,
        metrics: &Profile,
    ) -> PreservedAnalyses {
        let f = am.function_mut();
        let mut changed = forward_trivial(f);
        let graph = Parameters::new(f);
        let mut components = Components::new(&graph);
        for &param in &graph.values {
            if components.index[param].is_none() {
                components.visit(param);
            }
        }
        let mut aliases = SecondaryMap::<Value, Option<Value>>::new();
        for members in components.result {
            let mut outside = None;
            let mut different = false;
            let mut inside = SecondaryMap::<Value, bool>::new();
            for &member in &members {
                inside[member] = true;
            }
            for &member in &members {
                for &arg in &graph.incoming[member] {
                    if inside[arg] {
                        continue;
                    }
                    let arg = aliases[arg].unwrap_or(arg);
                    if outside.is_some_and(|value| value != arg) {
                        different = true;
                    }
                    outside = Some(arg);
                }
            }
            if !different && let Some(value) = outside {
                for member in members {
                    aliases[member] = Some(value);
                }
            }
        }
        for &param in &graph.values {
            if let Some(value) = aliases[param] {
                f.edit().replace_all_uses(param, value);
                changed += 1;
            }
        }
        // Every forwarded parameter now has no uses. Remove its position and
        // incoming arguments; general dead-parameter elimination belongs to DCE.
        if changed != 0 {
            let blocks: Vec<_> = f
                .layout()
                .block_order()
                .filter(|&block| block != f.entry_block())
                .collect();
            for block in blocks {
                let keep: Vec<_> = f
                    .dfg()
                    .block_params(block)
                    .iter()
                    .map(|&param| aliases[param].is_none())
                    .collect();
                if keep.contains(&false) {
                    f.edit().retain_block_params(block, &keep);
                }
            }
        }
        metrics.count("params.forwarded", changed);
        metrics.count("params.removed", changed);
        if changed == 0 {
            PreservedAnalyses::all()
        } else {
            PreservedAnalyses::none()
        }
    }
}

// A varying loop SCC can contain forwarding parameters alongside induction
// parameters. Peel those identities first; rejecting the whole SCC would keep
// copies around every branch and obscure the actual data dependencies.
fn forward_trivial(f: &mut FuncBody) -> u64 {
    use std::collections::VecDeque;
    let graph = Parameters::new(f);
    let mut users = SecondaryMap::<Value, Vec<Value>>::new();
    let mut aliases = SecondaryMap::<Value, Option<Value>>::new();
    let mut last_root = SecondaryMap::<Value, Option<Value>>::new();
    for &param in &graph.values {
        last_root[param] = Some(param);
        for &arg in &graph.incoming[param] {
            users[arg].push(param);
        }
    }
    fn root(mut value: Value, aliases: &SecondaryMap<Value, Option<Value>>) -> Value {
        while let Some(next) = aliases[value] {
            value = next;
        }
        value
    }
    let mut work: VecDeque<_> = graph.values.iter().copied().collect();
    while let Some(param) = work.pop_front() {
        if aliases[param].is_none() {
            let mut inputs = graph.incoming[param]
                .iter()
                .map(|&v| root(v, &aliases))
                .filter(|&v| v != param);
            if let Some(value) = inputs.next()
                && inputs.all(|v| v == value)
            {
                aliases[param] = Some(value);
            }
        }
        let value = root(param, &aliases);
        if last_root[param] != Some(value) {
            last_root[param] = Some(value);
            work.extend(users[param].iter().copied());
        }
    }
    let mut changed = 0;
    for &param in &graph.values {
        let value = root(param, &aliases);
        if value != param {
            f.edit().replace_all_uses(param, value);
            changed += 1;
        }
    }
    if changed != 0 {
        let blocks: Vec<_> = f
            .layout()
            .block_order()
            .filter(|&b| b != f.entry_block())
            .collect();
        for block in blocks {
            let keep: Vec<_> = f
                .dfg()
                .block_params(block)
                .iter()
                .map(|&p| aliases[p].is_none())
                .collect();
            if keep.contains(&false) {
                f.edit().retain_block_params(block, &keep);
            }
        }
    }
    changed
}

/// Tarjan visits incoming dependencies, so components are emitted before their
/// users. Only parameters participate; instruction results and entry arguments
/// are already stable identities, regardless of whether their runtime value varies.
struct Components<'a> {
    graph: &'a Parameters,
    index: SecondaryMap<Value, Option<usize>>,
    low: SecondaryMap<Value, usize>,
    active: SecondaryMap<Value, bool>,
    stack: Vec<Value>,
    next: usize,
    result: Vec<Vec<Value>>,
}
impl<'a> Components<'a> {
    fn new(graph: &'a Parameters) -> Self {
        Self {
            graph,
            index: SecondaryMap::new(),
            low: SecondaryMap::new(),
            active: SecondaryMap::new(),
            stack: vec![],
            next: 0,
            result: vec![],
        }
    }
    fn visit(&mut self, value: Value) {
        let index = self.next;
        self.next += 1;
        self.index[value] = Some(index);
        self.low[value] = index;
        self.stack.push(value);
        self.active[value] = true;
        for &arg in &self.graph.incoming[value] {
            if !self.graph.is_param[arg] {
                continue;
            }
            if self.index[arg].is_none() {
                self.visit(arg);
                self.low[value] = self.low[value].min(self.low[arg]);
            } else if self.active[arg] {
                self.low[value] = self.low[value].min(self.index[arg].unwrap());
            }
        }
        if self.low[value] == index {
            let mut members = Vec::new();
            loop {
                let member = self.stack.pop().unwrap();
                self.active[member] = false;
                members.push(member);
                if member == value {
                    break;
                }
            }
            self.result.push(members);
        }
    }
}
