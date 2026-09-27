//! One equality-graph pipeline; fast mode changes budgets, not semantics.
//!
//! Rewrites create ordinary, detached MIR instructions. A Value denotes one
//! concrete result; union-find adds equivalence without changing its definition.
//! Rebuilding repairs indexes, not MIR edges. Extraction then copies the chosen
//! expressions into dominance-valid positions and commits only executable uses.
//! The candidate session releases its use-def links before dead-code removal.
mod extract;
mod graph;
mod matching;

use crate::{FunctionPass, Metrics, OptConfig, PreservedAnalyses};
use graph::{Graph, InstKind};
use veloc_analyzer::{AnalysisManager, Dominators};
use veloc_mir::function::Expressions;
use veloc_mir::{FuncBody, Inst};

/// Relative costs of materializing constants and movable operations, not total
/// function runtime or compiler work. Ranking adds these as tree costs; planning
/// accounts for shared occurrences. Existing pinned values are zero-cost inputs,
/// not claims that their producing instructions are free to execute.
/// Materialization costs are clamped to at least one.
pub trait CostModel {
    /// Cost of one operation, using its first result type even when another
    /// projection is the requested value. Multi-result ops are charged once.
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
        if run_with_analyses(
            am,
            self.budget,
            &GenericCost,
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
    run_with_analyses(
        &mut AnalysisManager::new(func),
        budget,
        cost,
        debug,
        metrics,
    )
}

fn run_with_analyses(
    am: &mut AnalysisManager<'_>,
    budget: Budget,
    cost: &dyn CostModel,
    debug: bool,
    metrics: &mut Metrics,
) -> bool {
    // Expression selection and DCE preserve CFG topology. Own the snapshot
    // while editing instructions, without holding a borrow of the manager.
    let dom = am.take_dominators();
    let changed = optimize_function(am.function_mut(), budget, cost, &dom, metrics);
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
    /// Cost-propagation steps for candidate ranking. Partial estimates remain
    /// usable; exhaustion must not consume the placement budget.
    pub rank_steps: usize,
    /// Placement and joint DAG-selection work, independently of ranking.
    /// Exhaustion keeps the last complete selection, or the original function
    /// if the initial selection has not finished.
    pub extract_steps: usize,
}

impl Budget {
    pub const FAST: Self = Self {
        graph_nodes: 160,
        rounds: 2,
        match_steps: 16_384,
        rank_steps: 4_096,
        extract_steps: 12_288,
    };
    pub const DEFAULT: Self = Self {
        graph_nodes: 512,
        rounds: 6,
        match_steps: 262_144,
        rank_steps: 32_768,
        extract_steps: 98_304,
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
                graph.register_value(f, param);
            }
            for inst in f.layout().block_insts(block) {
                graph.register_inst(f, inst);
                if graph.kinds[inst] != InstKind::Floating {
                    anchors.push(inst);
                }
            }
        }
        graph.limit = graph.values.len().saturating_add(budget.graph_nodes);
        Self {
            ir: f.expressions(),
            graph,
            anchors,
        }
    }

    fn saturate(&mut self, rounds: usize, fuel: &mut usize) {
        let stop = saturate(&mut self.graph, &mut self.ir, rounds, fuel);
        log::debug!("egraph stopped: {stop:?}");
    }

    fn finish(
        self,
        model: &dyn CostModel,
        dom: &Dominators,
        rank: &mut usize,
        work: &mut usize,
    ) -> (u64, Vec<Inst>) {
        let Self { ir, graph, anchors } = self;
        // Preserve use locations through selection. Candidate uses do not make
        // expressions live; only executable anchors request occurrences.
        let extraction = graph.extract(ir.body(), &anchors, model, dom, rank, work);
        let removable = anchors
            .iter()
            .copied()
            .filter(|&inst| {
                graph.kinds[inst] != InstKind::Unsupported
                    && ir
                        .body()
                        .dfg()
                        .inst_results(inst)
                        .iter()
                        .all(|&v| ir.body().dfg().as_const(graph.find(v)).is_some())
            })
            .collect();
        let mut ir = ir.freeze();
        let changed = extraction.apply(&mut ir);
        (changed, removable)
    }
}

/// Budget stops are not saturation: unsearched inputs or pending facts may
/// still yield useful rewrites. Every exit leaves structural indexes repaired.
#[derive(Clone, Copy, Debug)]
enum Stop {
    Saturated,
    Rounds,
    Work,
    Matches,
    Nodes,
}

fn saturate(graph: &mut Graph, ir: &mut Expressions<'_>, rounds: usize, fuel: &mut usize) -> Stop {
    let mut machine = matching::Machine::new();
    let mut queries = Vec::new();
    graph.prepare(ir, fuel);
    for _ in 0..rounds {
        if graph.is_idle() {
            return Stop::Saturated;
        }
        if *fuel == 0 {
            return Stop::Work;
        }
        queries.clear();
        graph.schedule(ir.body(), &mut queries);
        // Nothing relevant to a rule changed. Analysis is already drained by
        // prepare, so an empty query set really is a fixed point.
        if queries.is_empty() {
            return Stop::Saturated;
        }

        // All roots see the same immutable graph. Even a partial search has
        // sound matches worth applying before returning a budget stop.
        let searched = machine.search(graph, ir.body(), &queries, fuel);
        let applied = machine.apply(graph, ir);
        graph.prepare(ir, fuel);
        if let Err(limit) = searched {
            return match limit {
                matching::QueryLimit::Work => Stop::Work,
                matching::QueryLimit::Matches => Stop::Matches,
            };
        }
        if !applied {
            return Stop::Nodes;
        }
    }
    if graph.is_idle() {
        Stop::Saturated
    } else if *fuel == 0 {
        Stop::Work
    } else {
        Stop::Rounds
    }
}

fn optimize_function(
    f: &mut FuncBody,
    budget: Budget,
    model: &dyn CostModel,
    dom: &Dominators,
    metrics: &mut Metrics,
) -> bool {
    let mut session = EqualitySession::new(f, budget);
    if !session
        .graph
        .kinds
        .values()
        .any(|&kind| kind != InstKind::Unsupported)
    {
        return false;
    }
    let mut fuel = budget.match_steps;
    session.saturate(budget.rounds, &mut fuel);
    metrics.add("egraph.nodes", session.graph.values.len() as u64);
    let mut rank = budget.rank_steps;
    let mut work = budget.extract_steps;
    let (mut changed, mut removable) = session.finish(model, dom, &mut rank, &mut work);
    metrics.add("egraph.rank_steps", (budget.rank_steps - rank) as u64);
    metrics.add("egraph.extract_steps", (budget.extract_steps - work) as u64);
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
    fn constants_remain_roots_and_wake_new_users() {
        let mut body = FuncBody::new(&[Type::I32; 7]);
        let mut graph = Graph::new();
        for index in 0..7 {
            graph.register_value(&body, Value(index));
        }
        // Form a larger nonconstant class before attaching it to a literal.
        graph.union(&body, Value(0), Value(1));
        graph.union(&body, Value(2), Value(0));
        let mut ir = body.expressions();
        let sum = graph
            .build(&mut ir, Op::IAdd, &[Value(2), Value(4)], Type::I32)
            .unwrap();
        let one = graph.literal(&mut ir, ScalarConst::from(1i32)).unwrap();

        graph.union(ir.body(), Value(2), one);
        for index in 0..3 {
            assert_eq!(graph.find(Value(index)), one);
        }
        // Both argument orders preserve the literal root. The last merge must
        // wake the addition even though this constant root was already known.
        graph.union(ir.body(), one, Value(3));
        graph.union(ir.body(), Value(4), one);
        graph.union(ir.body(), Value(0), Value(4));
        for index in 0..5 {
            assert_eq!(graph.find(Value(index)), one);
        }
        let mut fuel = 100;
        graph.prepare(&mut ir, &mut fuel);
        let two = ir.constant(ScalarConst::from(2i32).into());
        assert_eq!(graph.find(sum), two);
        assert_eq!(graph.find(one), one);

        // Finish one matching round with a known zero and an unrelated parent.
        // Only the subsequent merge can reveal the parent's x + 0 identity.
        let zero = graph.literal(&mut ir, ScalarConst::from(0i32)).unwrap();
        let parent = graph
            .build(&mut ir, Op::IAdd, &[Value(5), Value(6)], Type::I32)
            .unwrap();
        let mut fuel = Budget::DEFAULT.match_steps;
        super::saturate(&mut graph, &mut ir, Budget::DEFAULT.rounds, &mut fuel);
        graph.union(ir.body(), Value(6), zero);
        super::saturate(&mut graph, &mut ir, Budget::DEFAULT.rounds, &mut fuel);
        assert_eq!(graph.find(parent), graph.find(Value(5)));
    }

    #[test]
    #[should_panic(expected = "rewrite equated distinct constants")]
    fn distinct_constants_cannot_be_equated() {
        let mut body = FuncBody::new(&[]);
        let mut ir = body.expressions();
        let mut graph = Graph::new();
        let one = graph.literal(&mut ir, ScalarConst::from(1i32)).unwrap();
        let two = graph.literal(&mut ir, ScalarConst::from(2i32)).unwrap();
        graph.union(ir.body(), one, two);
    }

    #[test]
    fn cross_block_graph_preserves_effects_and_ssa() {
        let parsed = veloc_mir::ModuleParser::new()
            .parse(
                r#"
local function cross(i64, ptr) -> i64
block0(v0: i64, v1: ptr):
  v3: i64 = iadd v0, i64(3)
  jump block1()
block1():
  v4: i64 = load.volatile v1, offset=0
  v6: i64 = iadd v3, i64(4)
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
            .dfg()
            .values()
            .keys()
            .filter(|&v| f.dfg().uses(v).next().is_some())
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
            graph.register_value(&body, x);
            graph.register_value(&body, y);
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
            super::saturate(&mut graph, &mut ir, Budget::DEFAULT.rounds, &mut fuel);
            assert_eq!(graph.find(cancel), graph.find(y), "{ty:?}");
            assert_eq!(graph.find(wrapped), graph.find(x), "{ty:?}");
        }
    }

    #[test]
    fn extraction_preserves_sibling_uses_and_handles_budget_stops() {
        // The two additions share an e-class, but neither definition dominates
        // the other return. Keep both occurrences instead of copying them anew.
        for extract_steps in [0, 1, 32, Budget::DEFAULT.extract_steps] {
            let mut module = veloc_mir::ModuleParser::new()
                .parse(
                    r#"
local function siblings(bool, i64, i64) -> i64
block0(v0: bool, v1: i64, v2: i64):
  br v0, block1(), block2()
block1():
  v3: i64 = iadd v1, v2
  return v3
block2():
  v4: i64 = iadd v2, v1
  return v4
"#,
                )
                .unwrap();
            module.validate().unwrap();
            let body = module.body_mut(veloc_mir::FuncId(0)).unwrap();
            let budget = Budget {
                extract_steps,
                ..Budget::DEFAULT
            };
            assert!(!run(body, budget, false, &mut Metrics::default()));
            assert!(!run(body, budget, false, &mut Metrics::default()));
            module.validate().unwrap();
        }
    }

    #[test]
    fn ranking_exhaustion_does_not_disable_extraction() {
        for rank_steps in [0, 1] {
            let mut module = veloc_mir::ModuleParser::new()
                .parse(
                    r#"
local function identity(i64) -> i64
block0(v0: i64):
  v2: i64 = iadd v0, i64(0)
  return v2
"#,
                )
                .unwrap();
            module.validate().unwrap();
            let body = module.body_mut(veloc_mir::FuncId(0)).unwrap();
            let budget = Budget {
                rank_steps,
                ..Budget::DEFAULT
            };
            assert!(run(body, budget, false, &mut Metrics::default()));
            let ret = body
                .layout()
                .block_insts(body.entry_block())
                .last()
                .unwrap();
            assert_eq!(body.dfg().operands(ret), &[Value(0)]);
            module.validate().unwrap();
        }
    }

    #[test]
    fn extraction_coordinates_shared_computations_across_uses() {
        struct Cost;
        impl CostModel for Cost {
            fn operation(&self, op: Op, _: Type) -> usize {
                if op == Op::IMul { 5 } else { 1 }
            }
            fn constant(&self, _: ScalarConst) -> usize {
                1
            }
        }

        let mut module = veloc_mir::ModuleParser::new()
            .parse(
                r#"
local function shared(i64, ptr) -> i64
block0(v0: i64, v1: ptr):
  v4: i64 = imul v0, i64(6)
  store v4, v1, offset=0
  v5: i64 = imul v0, i64(9)
  return v5
"#,
            )
            .unwrap();
        module.validate().unwrap();
        let body = module.body_mut(veloc_mir::FuncId(0)).unwrap();
        let roots: Vec<_> = body
            .layout()
            .block_insts(body.entry_block())
            .filter(|&inst| body.dfg().opcode(inst) == Op::IMul)
            .map(|inst| body.dfg().inst_results(inst)[0])
            .collect();
        let mut session = EqualitySession::new(body, Budget::DEFAULT);
        let graph = &mut session.graph;
        let ir = &mut session.ir;
        let three = graph
            .literal(ir, ScalarConst::from_bits(Type::I64, 3).unwrap())
            .unwrap();
        let shared = graph
            .build(ir, Op::IMul, &[Value(0), three], Type::I64)
            .unwrap();
        let six = graph
            .build(ir, Op::IAdd, &[shared, shared], Type::I64)
            .unwrap();
        let nine = graph
            .build(ir, Op::IAdd, &[six, shared], Type::I64)
            .unwrap();
        graph.union(ir.body(), roots[0], six);
        graph.union(ir.body(), roots[1], nine);
        graph.rebuild(ir.body());

        // Each original multiply costs 6 with its literal: total 12. Changing
        // just one root is worse. Changing both costs 6 + 1 + 1 = 8, with the
        // shared multiply placed before the store and reused by the return.
        let mut rank = Budget::DEFAULT.rank_steps;
        let mut work = Budget::DEFAULT.extract_steps;
        let dom = Dominators::compute(session.ir.body().cfg(), session.ir.body().entry_block());
        let (changed, _) = session.finish(&Cost, &dom, &mut rank, &mut work);
        assert_eq!(changed, 2);
        super::super::dce::run_dce(body, false, &mut Metrics::default());
        let mut multiplies = 0;
        let mut additions = 0;
        for inst in body.layout().block_insts(body.entry_block()) {
            match body.dfg().opcode(inst) {
                Op::IMul => multiplies += 1,
                Op::IAdd => additions += 1,
                _ => {}
            }
        }
        assert_eq!((multiplies, additions), (1, 2));
        module.validate().unwrap();
    }
}
