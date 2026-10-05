//! Equality search over compact expressions, with original MIR kept intact.
//! Spec supplies shared operation semantics. Extraction builds an owned plan;
//! only plan application allocates MIR values and updates executable uses.
mod extract;
mod graph;
mod matching;
mod storage;

use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use graph::{Graph, InstKind};
use storage::{Expressions, Storage, Value as ExprValue};
use veloc_analyzer::{AnalysisManager, Dominators};
use veloc_mir::{FuncBody, Inst};

/// Relative costs of movable operations, not total
/// function runtime or compiler work. Ranking adds these as tree costs; planning
/// accounts for shared occurrences. Existing pinned values are zero-cost inputs,
/// not claims that their producing instructions are free to execute.
/// Known constants are terminal MIR values, never costed alternatives. Machine
/// materialization belongs to lowering. Operation costs are at least one.
pub trait CostModel {
    /// Cost of one operation, using its first result type even when another
    /// projection is the requested value. Multi-result ops are charged once.
    fn operation(&self, opcode: veloc_mir::Opcode, ty: veloc_mir::Type) -> usize;
}

/// Target-independent baseline. Targets may provide their own estimates without
/// changing equality rules or treating machine costs as MIR semantic facts.
pub struct GenericCost;
impl CostModel for GenericCost {
    fn operation(&self, _: veloc_mir::Opcode, _: veloc_mir::Type) -> usize {
        1
    }
}

pub struct ExpressionPass {
    pub budget: Budget,
}

impl FunctionPass for ExpressionPass {
    fn name(&self) -> &'static str {
        "ExpressionPass"
    }

    fn run(
        &self,
        am: &mut AnalysisManager<'_>,
        config: &OptConfig,
        metrics: &Profile,
    ) -> PassOutcome {
        if run_with_analyses(
            am,
            self.budget,
            &GenericCost,
            config.data_layout,
            config.is_debug_enabled("simplify"),
            metrics,
        ) {
            PassOutcome::Changed
        } else {
            PassOutcome::Unchanged
        }
    }
}

pub fn run(func: &mut FuncBody, budget: Budget, debug: bool, metrics: &Profile) -> bool {
    run_with_cost(func, budget, &GenericCost, debug, metrics)
}

pub fn run_with_cost(
    func: &mut FuncBody,
    budget: Budget,
    cost: &dyn CostModel,
    debug: bool,
    metrics: &Profile,
) -> bool {
    run_with_analyses(
        &mut AnalysisManager::new(func).with_profile(metrics.clone()),
        budget,
        cost,
        None,
        debug,
        metrics,
    )
}

fn run_with_analyses(
    am: &mut AnalysisManager<'_>,
    budget: Budget,
    cost: &dyn CostModel,
    layout: Option<veloc_types::DataLayout>,
    debug: bool,
    metrics: &Profile,
) -> bool {
    // Expression selection preserves CFG topology. Own the snapshot
    // while editing instructions, without holding a borrow of the manager.
    let dom = am.take_dominators();
    let changed = optimize_function(am.function_mut(), budget, cost, &dom, layout, metrics);
    if changed && debug {
        log::info!("Optimized expression graph");
    }
    changed
}

/// Search and extraction limits for one function. Both profiles
/// run the same graph optimizer; neither is a separate greedy rewrite engine.
#[derive(Clone, Copy)]
pub struct Budget {
    /// New candidate instructions allowed across all rewrite rounds. Imports,
    /// literals, reuse and operations folded during construction do not count.
    /// Later folding does not refund an instruction's allocation.
    pub graph_nodes: usize,
    /// Maximum rounds of matching, rewriting and rebuilding.
    pub rounds: usize,
    /// Maximum unique matches retained per round, independent of search fuel.
    pub matches: usize,
    /// Query entries and attempted row bindings across all rounds.
    pub match_steps: usize,
    /// Cost-propagation steps for candidate ranking. Initialization, final
    /// candidate estimates and sorting are outside this budget. On exhaustion,
    /// candidates use the partial class estimates without spending placement fuel.
    pub rank_steps: usize,
    /// Candidate-search state transitions, independently of ranking. Exhaustion
    /// rolls back the unfinished operand; remaining uses keep their originals.
    /// Completed choices still undergo the full DAG cost check, which is unmetered.
    pub extract_steps: usize,
}

impl Budget {
    pub const FAST: Self = Self {
        graph_nodes: 160,
        rounds: 2,
        matches: 4_096,
        match_steps: 16_384,
        rank_steps: 4_096,
        extract_steps: 12_288,
    };
    pub const DEFAULT: Self = Self {
        graph_nodes: 512,
        rounds: 6,
        matches: 65_536,
        match_steps: 262_144,
        rank_steps: 32_768,
        extract_steps: 98_304,
    };
}

/// One owner of search expressions and their equality indexes.
struct EqualitySession<'a> {
    ir: Expressions<'a>,
    graph: Graph,
    anchors: Vec<Inst>,
}

impl<'a> EqualitySession<'a> {
    fn new(f: &'a FuncBody, budget: Budget) -> Self {
        let mut ir = Expressions::new(f);
        let mut graph = Graph::new();
        let mut anchors = Vec::new();
        for block in f.cfg().compute_rpo(f.entry_block()) {
            for original in f.layout().block_insts(block) {
                let inst = ir.import_inst(original);
                if inst.is_none_or(|inst| !ir.can_speculate(inst)) {
                    anchors.push(original);
                }
            }
        }
        // Import all definitions before indexing: an input may be defined in a
        // later RPO block. Only referenced leaves and supported operations exist.
        for value in ir.values() {
            graph.register_value(&ir, value);
        }
        for inst in ir.insts() {
            graph.register_inst(&ir, inst);
        }
        graph.remaining_nodes = budget.graph_nodes;
        Self { ir, graph, anchors }
    }

    fn saturate(&mut self, rounds: usize, matches: usize, fuel: &mut usize) -> Stop {
        let stop = saturate(&mut self.graph, &mut self.ir, rounds, matches, fuel);
        match stop {
            Stop::Saturated => log::debug!("egraph saturated"),
            Stop::Limited(limit) => log::debug!("egraph limited: {limit:?}"),
        }
        stop
    }

    fn finish(
        self,
        model: &dyn CostModel,
        dom: &Dominators,
        rank: &mut usize,
        work: &mut usize,
    ) -> OptimizationPlan {
        let Self {
            mut ir,
            mut graph,
            mut anchors,
        } = self;
        let mut aliases = hashbrown::HashMap::new();
        let mut folds = Vec::new();
        for index in 0..graph.values.len() {
            let value = graph.values[index];
            let replacement = graph.resolve_alias(&ir, value);
            if value != replacement
                && let Some(original) = ir.original_value(value)
            {
                aliases.insert(value, replacement);
                folds.push((original, replacement));
            }
        }
        // Price and select through the proven fold mapping. Formal uses stay intact
        // until the complete plan is applied, including when extraction stops.
        ir.set_replacements(aliases);
        anchors.retain(|&inst| {
            ir.imported_inst(inst)
                .is_none_or(|i| graph.kinds[i] != InstKind::Folded)
        });
        let extraction = graph.extract(&ir, &anchors, model, dom, rank, work);
        let replaced = ir
            .insts()
            .filter(|&inst| graph.kinds[inst] == InstKind::Folded)
            .filter_map(|inst| ir.original_inst(inst))
            .collect();
        OptimizationPlan {
            ir: ir.into_storage(),
            extraction,
            folds,
            replaced,
            metrics: graph.profile,
        }
    }
}

/// Own every candidate needed for emission. Dropping this plan leaves MIR
/// unchanged; applying it is the only operation that publishes search results.
struct OptimizationPlan {
    ir: Storage,
    extraction: extract::Extraction,
    folds: Vec<(veloc_mir::Value, ExprValue)>,
    replaced: Vec<Inst>,
    metrics: Profile,
}

impl OptimizationPlan {
    fn apply(self, body: &mut FuncBody) -> u64 {
        let mut changed = 0;
        for (old, replacement) in self.folds {
            let value = self.ir.materialize(body, replacement);
            changed += body.dfg().uses(old).count() as u64;
            body.edit().replace_all_uses(old, value);
        }
        changed += self.extraction.apply(body, &self.ir, &self.metrics);
        body.edit().erase_insts(&self.replaced);
        changed + self.replaced.len() as u64
    }
}

/// Budget stops are not saturation: unsearched inputs or pending facts may
/// still yield useful rewrites. Every exit leaves structural indexes repaired.
#[derive(Clone, Copy, Debug)]
enum Stop {
    Saturated,
    Limited(Limit),
}

/// Shared resource-limit reasons; each phase owns its counters and recovery.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Limit {
    Rounds,
    Nodes,
    Matches,
    MatchWork,
    RankWork,
    ExtractWork,
}

fn saturate(
    graph: &mut Graph,
    ir: &mut Expressions,
    rounds: usize,
    matches: usize,
    fuel: &mut usize,
) -> Stop {
    let mut machine = matching::Machine::new();
    let mut queries = Vec::new();
    graph.rebuild(ir);
    for _ in 0..rounds {
        if graph.is_idle() {
            return Stop::Saturated;
        }
        if *fuel == 0 {
            return Stop::Limited(Limit::MatchWork);
        }
        queries.clear();
        graph.schedule(ir, &mut queries);
        // Nothing relevant to a rule changed. Analysis is already drained by
        // rebuilding, so an empty query set really is a fixed point.
        if queries.is_empty() {
            return Stop::Saturated;
        }

        // All roots see the same immutable graph. Even a partial search has
        // sound matches worth applying before returning a budget stop.
        let searched = machine.search(graph, ir, &queries, matches, fuel);
        let applied = machine.apply(graph, ir);
        graph.rebuild(ir);
        if let Err(limit) = searched {
            return Stop::Limited(limit);
        }
        if let Err(limit) = applied {
            return Stop::Limited(limit);
        }
    }
    if graph.is_idle() {
        Stop::Saturated
    } else if *fuel == 0 {
        Stop::Limited(Limit::MatchWork)
    } else {
        Stop::Limited(Limit::Rounds)
    }
}

fn optimize_function(
    f: &mut FuncBody,
    budget: Budget,
    model: &dyn CostModel,
    dom: &Dominators,
    layout: Option<veloc_types::DataLayout>,
    metrics: &Profile,
) -> bool {
    let scope = metrics.scope("egraph.import", 0);
    let mut session = EqualitySession::new(f, budget);
    session.graph.profile = metrics.clone();
    session.graph.layout = layout;
    scope.success();
    if session.ir.insts().next().is_none() {
        return false;
    }
    let mut fuel = budget.match_steps;
    let scope = metrics.scope("egraph.saturate", 0);
    let stop = session.saturate(budget.rounds, budget.matches, &mut fuel);
    if let Stop::Limited(limit) = stop {
        metrics.count("budget_stops", 1);
        metrics.remark(|| format!("egraph saturation stopped: {limit:?}"));
    }
    scope.success();
    metrics.count("egraph.nodes", session.graph.values.len() as u64);
    metrics.count(
        "egraph.created_nodes",
        (budget.graph_nodes - session.graph.remaining_nodes) as u64,
    );
    let mut rank = budget.rank_steps;
    let mut work = budget.extract_steps;
    let scope = metrics.scope("egraph.extract", 0);
    let plan = session.finish(model, dom, &mut rank, &mut work);
    let changed = plan.apply(f);
    scope.success();
    metrics.count("egraph.rank_steps", (budget.rank_steps - rank) as u64);
    metrics.count("egraph.extract_steps", (budget.extract_steps - work) as u64);
    metrics.count("egraph.rewritten_values", changed);
    metrics.count("egraph.match_steps", (budget.match_steps - fuel) as u64);
    changed != 0
}
#[cfg(test)]
mod tests {
    use super::storage::Value;
    use super::*;
    use crate::Profile;
    use veloc_mir::constant::ScalarConst;
    use veloc_mir::{Opcode as Op, Type, Value as MirValue};
    use veloc_types::TypeInfo;

    #[test]
    fn local_folds_need_neither_memo_tombstones_nor_extraction_budget() {
        let mut module = veloc_mir::ModuleParser::new()
            .parse(
                r#"
local function identities(v0: i64) -> (i64, i64, i64)
block0():
  v2: i64 = isub v0, i64(0)
  v3: i64 = isub v0, i64(0)
  v4: i64 = iadd v0, i64(1)
  v5: i64 = isub v4, i64(1)
  return v2, v3, v5
"#,
            )
            .unwrap();
        let body = module.body_mut(veloc_mir::FuncId(0)).unwrap();
        let insts: Vec<_> = body.layout().block_insts(body.entry_block()).collect();
        let mut session = EqualitySession::new(body, Budget::DEFAULT);
        let mut fuel = Budget::DEFAULT.match_steps;
        session.saturate(Budget::DEFAULT.rounds, Budget::DEFAULT.matches, &mut fuel);
        let graph = &mut session.graph;
        let ir = &mut session.ir;
        // Import folding removes both concrete identities from search.
        for &inst in &insts[..2] {
            assert!(graph.kinds[ir.imported_inst(inst).unwrap()] == InstKind::Folded);
        }
        // Cancellation is still an ordinary equality: keep this alternative.
        assert!(graph.kinds[ir.imported_inst(insts[3]).unwrap()] == InstKind::Floating);
        let other = ir.inst_results(ir.imported_inst(insts[3]).unwrap())[0];
        assert_eq!(graph.find(other), graph.find(Value(0)));
        assert_eq!(graph.alternatives(graph.find(other), Op::ISub), [other]);

        // Construction returns the input, without allocating or consulting a tombstone.
        let count = graph.values.len();
        let inst_count = ir.inst_count();
        graph.remaining_nodes = 0;
        let zero = ir.constant(ScalarConst::from(0i64).into());
        let rebuilt = graph
            .build(
                ir,
                Op::ISub,
                &[Value(0), zero],
                Type::I64,
                crate::evaluate::Properties::None,
            )
            .unwrap();
        assert_eq!(graph.values.len(), count);
        assert_eq!(rebuilt, Value(0));
        assert!(ir.value_inst(rebuilt).is_none());
        let one = ir.constant(ScalarConst::from(1i64).into());
        let folded = graph
            .build(
                ir,
                Op::IAdd,
                &[one, one],
                Type::I64,
                crate::evaluate::Properties::None,
            )
            .unwrap();
        assert_eq!(ir.as_scalar_const(folded), Some(ScalarConst::from(2i64)));
        assert_eq!(ir.inst_count(), inst_count);

        let dom = Dominators::compute(body.cfg(), body.entry_block());
        // Even without extraction fuel, executable uses get the original operand.
        session
            .finish(&GenericCost, &dom, &mut 0, &mut 0)
            .apply(body);
        assert_eq!(
            body.layout()
                .block_insts(body.entry_block())
                .collect::<Vec<_>>(),
            insts[2..]
        );
        assert_eq!(
            &body.dfg().operands(*insts.last().unwrap())[..2],
            &[MirValue(0), MirValue(0)]
        );
        module.validate().unwrap();
    }

    #[test]
    fn known_constants_do_not_require_extraction_budget() {
        for mut work in [0, Budget::DEFAULT.extract_steps] {
            let mut module = veloc_mir::ModuleParser::new()
                .parse(
                    r#"
local function folded() -> i64
block0():
  v0: i64 = iadd i64(3), i64(4)
  return v0
"#,
                )
                .unwrap();
            let body = module.body_mut(veloc_mir::FuncId(0)).unwrap();
            let dom = Dominators::compute(body.cfg(), body.entry_block());
            let mut session = EqualitySession::new(body, Budget::DEFAULT);
            // Import reduction is independent of exploratory matching fuel.
            session.saturate(0, Budget::DEFAULT.matches, &mut 0);
            let mut rank = 0;
            session
                .finish(&GenericCost, &dom, &mut rank, &mut work)
                .apply(body);
            let insts: Vec<_> = body.layout().block_insts(body.entry_block()).collect();
            assert_eq!(insts.len(), 1, "folded arithmetic must be dead");
            let value = body.dfg().operands(insts[0])[0];
            assert_eq!(
                body.dfg().as_scalar_const(value),
                Some(ScalarConst::from(7i64))
            );
            module.validate().unwrap();
        }
    }

    #[test]
    fn constants_remain_roots_and_wake_new_users() {
        let body = FuncBody::new(&[Type::I32; 7]);
        let mut ir = Expressions::new(&body);
        let mut graph = Graph::new();
        for index in 0..7 {
            let value = ir.import_value(MirValue(index));
            graph.register_value(&ir, value);
        }
        // Form a larger nonconstant class before attaching it to a literal.
        graph.union(&ir, Value(0), Value(1));
        graph.union(&ir, Value(2), Value(0));
        let sum = graph
            .build(
                &mut ir,
                Op::IAdd,
                &[Value(2), Value(4)],
                Type::I32,
                crate::evaluate::Properties::None,
            )
            .unwrap();
        let one = graph.literal(&mut ir, ScalarConst::from(1i32));

        graph.union(&ir, Value(2), one);
        for index in 0..3 {
            assert_eq!(graph.find(Value(index)).value(), one);
        }
        // Both argument orders preserve the literal root. The last merge must
        // wake the addition even though this constant root was already known.
        graph.union(&ir, one, Value(3));
        graph.union(&ir, Value(4), one);
        graph.union(&ir, Value(0), Value(4));
        for index in 0..5 {
            assert_eq!(graph.find(Value(index)).value(), one);
        }
        graph.rebuild(&mut ir);
        let two = ir.constant(ScalarConst::from(2i32).into());
        assert_eq!(graph.find(sum).value(), two);
        assert_eq!(graph.find(one).value(), one);

        // Finish one matching round with a known zero and an unrelated parent.
        // Only the subsequent merge can reveal the parent's x + 0 identity.
        let zero = graph.literal(&mut ir, ScalarConst::from(0i32));
        let parent = graph
            .build(
                &mut ir,
                Op::IAdd,
                &[Value(5), Value(6)],
                Type::I32,
                crate::evaluate::Properties::None,
            )
            .unwrap();
        let mut fuel = Budget::DEFAULT.match_steps;
        super::saturate(
            &mut graph,
            &mut ir,
            Budget::DEFAULT.rounds,
            Budget::DEFAULT.matches,
            &mut fuel,
        );
        graph.union(&ir, Value(6), zero);
        super::saturate(
            &mut graph,
            &mut ir,
            Budget::DEFAULT.rounds,
            Budget::DEFAULT.matches,
            &mut fuel,
        );
        assert_eq!(graph.find(parent), graph.find(Value(5)));
    }

    #[test]
    #[should_panic(expected = "rewrite equated distinct constants")]
    fn distinct_constants_cannot_be_equated() {
        let body = FuncBody::new(&[]);
        let mut ir = Expressions::new(&body);
        let mut graph = Graph::new();
        let one = graph.literal(&mut ir, ScalarConst::from(1i32));
        let two = graph.literal(&mut ir, ScalarConst::from(2i32));
        graph.union(&ir, one, two);
    }

    #[test]
    fn cross_block_graph_preserves_effects_and_ssa() {
        let parsed = veloc_mir::ModuleParser::new()
            .parse(
                r#"
local function cross(v0: i64, v1: ptr) -> i64
block0():
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
        assert!(super::run(f, Budget::DEFAULT, false, &Profile::default()));
        crate::DcePass.run(
            &mut AnalysisManager::new(f),
            &OptConfig::default(),
            &Profile::default(),
        );
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
            let body = FuncBody::new(&[ty, ty]);
            let mut ir = Expressions::new(&body);
            let x = ir.import_value(MirValue(0));
            let y = ir.import_value(MirValue(1));
            let mut graph = Graph::new();
            graph.remaining_nodes = Budget::DEFAULT.graph_nodes;
            graph.register_value(&ir, x);
            graph.register_value(&ir, y);
            let sum = graph
                .build(
                    &mut ir,
                    Op::IAdd,
                    &[x, y],
                    ty,
                    crate::evaluate::Properties::None,
                )
                .unwrap();
            let cancel = graph
                .build(
                    &mut ir,
                    Op::ISub,
                    &[sum, x],
                    ty,
                    crate::evaluate::Properties::None,
                )
                .unwrap();
            let max = graph.literal(
                &mut ir,
                ScalarConst::from_bits(ty, u64::MAX >> (64 - ty.element_bits().unwrap())).unwrap(),
            );
            let one = graph.literal(&mut ir, ScalarConst::from_bits(ty, 1).unwrap());
            let left = graph
                .build(
                    &mut ir,
                    Op::IAdd,
                    &[x, max],
                    ty,
                    crate::evaluate::Properties::None,
                )
                .unwrap();
            let wrapped = graph
                .build(
                    &mut ir,
                    Op::IAdd,
                    &[left, one],
                    ty,
                    crate::evaluate::Properties::None,
                )
                .unwrap();
            let mut fuel = Budget::DEFAULT.match_steps;
            super::saturate(
                &mut graph,
                &mut ir,
                Budget::DEFAULT.rounds,
                Budget::DEFAULT.matches,
                &mut fuel,
            );
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
local function siblings(v0: bool, v1: i64, v2: i64) -> i64
block0():
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
            assert!(!run(body, budget, false, &Profile::default()));
            assert!(!run(body, budget, false, &Profile::default()));
            module.validate().unwrap();
        }
    }

    #[test]
    fn ranking_exhaustion_does_not_disable_extraction() {
        for rank_steps in [0, 1] {
            let mut module = veloc_mir::ModuleParser::new()
                .parse(
                    r#"
local function identity(v0: i64) -> i64
block0():
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
            assert!(run(body, budget, false, &Profile::default()));
            let ret = body
                .layout()
                .block_insts(body.entry_block())
                .last()
                .unwrap();
            assert_eq!(body.dfg().operands(ret), &[MirValue(0)]);
            module.validate().unwrap();
        }
    }

    #[test]
    fn extraction_reuses_computations_across_uses() {
        let mut module = veloc_mir::ModuleParser::new()
            .parse(
                r#"
local function shared(v0: i64, v1: ptr) -> i64
block0():
  v4: i64 = imul v0, i64(3)
  store v4, v1, offset=0
  v5: i64 = imul v0, i64(3)
  return v5
"#,
            )
            .unwrap();
        module.validate().unwrap();
        let body = module.body_mut(veloc_mir::FuncId(0)).unwrap();
        let mut session = EqualitySession::new(body, Budget::DEFAULT);
        session.graph.rebuild(&mut session.ir);

        // One dominance-ordered plan reuses the multiply at both anchors;
        // sharing does not require retrying complete selections.
        let mut rank = Budget::DEFAULT.rank_steps;
        let mut work = Budget::DEFAULT.extract_steps;
        let dom = Dominators::compute(body.cfg(), body.entry_block());
        let changed = session
            .finish(&GenericCost, &dom, &mut rank, &mut work)
            .apply(body);
        assert_eq!(changed, 1);
        crate::DcePass.run(
            &mut AnalysisManager::new(body),
            &OptConfig::default(),
            &Profile::default(),
        );
        let mut multiply = None;
        let mut store = None;
        let mut ret = None;
        for inst in body.layout().block_insts(body.entry_block()) {
            match body.dfg().opcode(inst) {
                Op::IMul => assert!(multiply.replace(inst).is_none()),
                Op::Store => store = Some(inst),
                Op::Return => ret = Some(inst),
                _ => {}
            }
        }
        let result = body.dfg().inst_results(multiply.unwrap())[0];
        assert_eq!(body.dfg().operands(store.unwrap())[1], result);
        assert_eq!(body.dfg().operands(ret.unwrap()), &[result]);
        module.validate().unwrap();
    }
}
