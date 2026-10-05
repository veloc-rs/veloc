//! Sparse conditional constant propagation. Values and executable edges reach
//! a joint fixed point before any IR is edited; unreachable predecessors never
//! contribute to a block parameter. Work follows SSA uses, not whole-CFG rounds.
use crate::evaluate::Fact;
use crate::{FunctionPass, OptConfig, PassOutcome, Profile, evaluate};
use cranelift_entity::SecondaryMap;
use std::collections::VecDeque;
use veloc_analyzer::AnalysisManager;
use veloc_mir::function::EdgeRef;
use veloc_mir::{Block, FuncBody, Inst, InstView, Value, ValueDef};

pub struct SccpPass;

enum Successors {
    Pending,
    All,
    One(usize),
}

#[derive(Default)]
struct Solver {
    facts: SecondaryMap<Value, Fact>,
    executable: SecondaryMap<Block, bool>,
    queued: SecondaryMap<Inst, bool>,
    work: VecDeque<Inst>,
    evaluations: u64,
    updates: u64,
}

impl Solver {
    fn fact(&self, f: &FuncBody, value: Value) -> Fact {
        match f.dfg().value_def(value) {
            ValueDef::Const(_) => f
                .dfg()
                .as_scalar_const(value)
                .map_or(Fact::Varying, Fact::Constant),
            ValueDef::FunctionParam(_) => Fact::Varying,
            _ => self.facts[value],
        }
    }
    fn enqueue(&mut self, f: &FuncBody, inst: Inst) {
        if !self.queued[inst]
            && f.layout()
                .inst_block(inst)
                .is_some_and(|b| self.executable[b])
        {
            self.queued[inst] = true;
            self.work.push_back(inst);
        }
    }
    fn update(&mut self, f: &FuncBody, value: Value, fact: Fact) {
        let next = self.facts[value].join(fact);
        if next != self.facts[value] {
            self.facts[value] = next;
            self.updates += 1;
            for site in f.dfg().uses(value) {
                self.enqueue(f, site.inst());
            }
        }
    }
    fn activate(&mut self, f: &FuncBody, block: Block) {
        if !self.executable[block] {
            self.executable[block] = true;
            for inst in f.layout().block_insts(block) {
                self.enqueue(f, inst);
            }
        }
    }
    fn successors(&self, f: &FuncBody, inst: Inst) -> Successors {
        let (selector, count) = match f.dfg().inst(inst) {
            InstView::Br { condition, .. } => (condition, None),
            InstView::BrTable { index, table } => (index, Some(table.len())),
            _ => return Successors::All,
        };
        match self.fact(f, selector) {
            Fact::Unknown => Successors::Pending,
            Fact::Varying => Successors::All,
            Fact::Constant(c) => Successors::One(match count {
                Some(count) => (c.to_bits() as usize).min(count - 1),
                None => usize::from(c.to_bits() == 0),
            }),
        }
    }
    fn transfer(&mut self, f: &FuncBody, inst: Inst) {
        let dfg = f.dfg();
        let view = dfg.inst(inst);
        if view.is_terminator() {
            let choice = self.successors(f, inst);
            let mut index = 0;
            view.visit_successors(|edge| {
                if matches!(choice, Successors::All)
                    || matches!(choice, Successors::One(i) if i == index)
                {
                    self.activate(f, edge.block);
                    for (&param, &arg) in dfg.block_params(edge.block).iter().zip(edge.args) {
                        self.update(f, param, self.fact(f, arg));
                    }
                }
                index += 1;
            });
            return;
        }
        let results = dfg.inst_results(inst);
        if results.iter().all(|&v| self.facts[v] == Fact::Varying) {
            return;
        }
        self.evaluations += 1;
        let args = dfg.operands(inst);
        let evaluated = evaluate::evaluate_inst(dfg, inst, |v| self.fact(f, v));
        for (&value, result) in results.iter().zip(evaluated) {
            let fact = result.fact(|i| self.fact(f, args[i]));
            self.update(f, value, fact);
        }
    }
    fn solve(f: &FuncBody) -> Self {
        let mut solver = Self::default();
        solver.activate(f, f.entry_block());
        while let Some(inst) = solver.work.pop_front() {
            solver.queued[inst] = false;
            solver.transfer(f, inst);
        }
        solver
    }
}

impl FunctionPass for SccpPass {
    fn reuse_key(&self) -> Option<core::any::TypeId> {
        Some(core::any::TypeId::of::<Self>())
    }
    fn name(&self) -> &'static str {
        "SccpPass"
    }
    fn run(&self, am: &mut AnalysisManager<'_>, _: &OptConfig, metrics: &Profile) -> PassOutcome {
        let solver = Solver::solve(am.function());
        let f = am.function();
        let mut replacements = Vec::new();
        let mut branches = Vec::new();
        let mut folded = Vec::new();
        for block in f.layout().block_order().filter(|&b| solver.executable[b]) {
            for &param in f.dfg().block_params(block) {
                if let Fact::Constant(c) = solver.facts[param]
                    && f.dfg().uses(param).next().is_some()
                {
                    replacements.push((param, c));
                }
            }
            for inst in f.layout().block_insts(block) {
                if let Successors::One(chosen) = solver.successors(f, inst) {
                    branches.push(EdgeRef {
                        inst,
                        index: chosen.try_into().expect("too many successors"),
                    });
                }
                let results = f.dfg().inst_results(inst);
                for &value in results {
                    if let Fact::Constant(c) = solver.facts[value] {
                        replacements.push((value, c));
                    }
                }
                if !results.is_empty()
                    && results
                        .iter()
                        .all(|&v| matches!(solver.facts[v], Fact::Constant(_)))
                {
                    // The shared evaluator only produces these facts for
                    // supported operations, proving absence of traps.
                    folded.push(inst);
                }
            }
        }
        metrics.count("sccp.evaluations", solver.evaluations);
        metrics.count("sccp.updates", solver.updates);
        metrics.count("sccp.constants", replacements.len() as u64);
        metrics.count("sccp.branches", branches.len() as u64);
        if replacements.is_empty() && branches.is_empty() {
            return PassOutcome::Unchanged;
        }
        let f = am.function_mut();
        // Fold branches first so subsequent replacement visits only retained uses.
        for edge in branches {
            f.edit().fold_to_edge(edge);
        }
        for (value, constant) in replacements {
            let literal = f.edit().constant(constant.into());
            f.edit().replace_all_uses(value, literal);
        }
        f.edit().erase_insts(&folded);
        f.edit().remove_unreachable();
        PassOutcome::Changed
    }
}
