//! One equality-graph pipeline; fast mode changes budgets, not semantics.
//!
//! Search creates ordinary, detached MIR instructions. A Value denotes one
//! concrete result; union-find adds equivalence without changing its definition.
//! Rebuilding repairs indexes, not MIR edges. Extraction then copies the chosen
//! expressions into dominance-valid positions and commits only executable uses.
//! The candidate session releases its use-def links before dead-code removal.
mod extract;
mod graph;
mod matching;

use crate::{FunctionPass, Metrics, OptConfig, PreservedAnalyses};
use extract::Placement;
use graph::Graph;
use smallvec::SmallVec;
use veloc_analyzer::AnalysisManager;
use veloc_mir::function::Expressions;
use veloc_mir::{FuncBody, Inst, Value};

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

/// One owner of the MIR candidate arena and its equality indexes.
struct EqualitySession<'a> {
    ir: Expressions<'a>,
    graph: Graph,
    anchors: Vec<Inst>,
}

impl<'a> EqualitySession<'a> {
    fn new(f: &'a mut FuncBody, budget: Budget) -> Self {
        let mut graph = Graph::new();
        let mut anchors = Vec::new();
        for block in f.cfg().compute_rpo(f.entry_block()) {
            for &param in f.dfg().block_params(block) {
                graph.register_value(param);
            }
            for inst in f.layout().block_insts(block) {
                graph.register_inst(f, inst);
                if !graph.floating[inst] {
                    anchors.push(inst);
                }
            }
        }
        graph.limit = graph.values.len().saturating_add(budget.graph_nodes);
        graph.rebuild(f);
        Self {
            ir: f.expressions(),
            graph,
            anchors,
        }
    }

    fn saturate(&mut self, rounds: usize, fuel: &mut usize) {
        self.graph.saturate(&mut self.ir, rounds, fuel);
    }

    fn finish(self, model: &dyn CostModel) -> (u64, Vec<Inst>) {
        let Self { ir, graph, anchors } = self;
        // Only executable consumers are roots. Candidate uses do not make
        // expressions live, and pure alternatives never become roots themselves.
        let roots: Vec<_> = anchors
            .iter()
            .flat_map(|&inst| ir.body().dfg().operands(inst).iter().copied())
            .collect();
        let Some(choices) = graph.extract(ir.body(), &roots, model) else {
            return (0, Vec::new());
        };
        let removable = anchors
            .iter()
            .copied()
            .filter(|&inst| {
                graph.supported[inst]
                    && ir
                        .body()
                        .dfg()
                        .inst_results(inst)
                        .iter()
                        .all(|&v| graph.constants[graph.find(v)].is_some())
            })
            .collect();
        let mut placement = Placement::new(ir.body(), &graph);
        let mut ir = ir.freeze();
        let mut changed = 0;
        for inst in anchors {
            placement.local.clear();
            let args: SmallVec<[Value; 4]> = ir.body().dfg().operands(inst).into();
            for (index, old) in args.into_iter().enumerate() {
                let class = graph.find(old);
                if let Some(new) = placement.materialize(&mut ir, inst, class, &graph, &choices)
                    && old != new
                {
                    ir.replace_input(inst, index as u32, new);
                    changed += 1;
                }
            }
        }
        (changed, removable)
    }
}

fn optimize_function(
    f: &mut FuncBody,
    budget: Budget,
    model: &dyn CostModel,
    metrics: &mut Metrics,
) -> bool {
    let mut session = EqualitySession::new(f, budget);
    if !session.graph.supported.values().any(|&supported| supported) {
        return false;
    }
    let mut fuel = budget.match_steps;
    session.saturate(budget.rounds, &mut fuel);
    metrics.add("egraph.nodes", session.graph.values.len() as u64);
    let (mut changed, mut removable) = session.finish(model);
    // Candidate use-def links have been released before checking actual uses.
    removable.retain(|&inst| {
        f.dfg()
            .inst_results(inst)
            .iter()
            .all(|&v| f.dfg().uses(v).next().is_none())
    });
    if !removable.is_empty() {
        f.edit().erase_insts(&removable);
        changed += removable.len() as u64;
    }
    let cleaned = super::dce::run_dce(f, false, metrics);
    metrics.add("egraph.rewritten_values", changed);
    metrics.add("egraph.match_steps", (budget.match_steps - fuel) as u64);
    changed != 0 || cleaned
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Metrics;
    use veloc_mir::constant::ScalarConst;
    use veloc_mir::{Opcode as Op, Type, Value};
    use veloc_types::TypeInfo;

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
        let mut module = parsed;
        module.validate().unwrap();
        let f = module.body_mut(veloc_mir::FuncId(0)).unwrap();
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
            let mut body = FuncBody::new(&[ty, ty]);
            let x = Value(0);
            let y = Value(1);
            let mut graph = Graph::new();
            graph.limit = Budget::DEFAULT.graph_nodes;
            graph.register_value(x);
            graph.register_value(y);
            let mut ir = body.expressions();
            let sum = graph.build(&mut ir, Op::IAdd, &[x, y], ty).unwrap();
            let cancel = graph.build(&mut ir, Op::ISub, &[sum, x], ty).unwrap();
            let max = graph
                .literal(
                    &mut ir,
                    ScalarConst::from_bits(ty, u64::MAX >> (64 - ty.element_bits().unwrap()))
                        .unwrap(),
                )
                .unwrap();
            let one = graph
                .literal(&mut ir, ScalarConst::from_bits(ty, 1).unwrap())
                .unwrap();
            let left = graph.build(&mut ir, Op::IAdd, &[x, max], ty).unwrap();
            let wrapped = graph.build(&mut ir, Op::IAdd, &[left, one], ty).unwrap();
            let mut fuel = Budget::DEFAULT.match_steps;
            graph.saturate(&mut ir, Budget::DEFAULT.rounds, &mut fuel);
            assert_eq!(graph.find(cancel), graph.find(y), "{ty:?}");
            assert_eq!(graph.find(wrapped), graph.find(x), "{ty:?}");
            let extracted = graph.extract(ir.body(), &[wrapped], &GenericCost).unwrap();
            assert_eq!(extracted[graph.find(wrapped)], Some(x));
        }
    }
}
