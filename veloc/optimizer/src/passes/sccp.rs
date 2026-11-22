//! Sparse conditional constant propagation. Values and executable edges reach
//! a joint fixed point before any IR is edited; unreachable predecessors never
//! contribute to a block parameter. Work follows SSA uses, not whole-CFG rounds.
use crate::rewrite::{Fact, Fold, Successors};
use crate::{FunctionPass, OptConfig, PassOutcome, Profile, rewrite};
use cranelift_entity::SecondaryMap;
use std::collections::{HashSet, VecDeque};
use veloc_analyzer::AnalysisManager;
use veloc_mir::function::EdgeRef;
use veloc_mir::{Block, FuncBody, Inst, Opcode, Value, ValueDef};

pub struct SccpPass;

#[derive(Default)]
struct Solver {
    facts: SecondaryMap<Value, Fact>,
    parameter_extensions: SecondaryMap<Value, usize>,
    edges: HashSet<EdgeRef>,
    incoming: SecondaryMap<Block, usize>,
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
            ValueDef::StackSlot { .. } | ValueDef::FunctionParam(_) => Fact::Varying,
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
        let old = self.facts[value];
        let mut next = old.join(fact);
        if let ValueDef::BlockParam(block) = f.dfg().value_def(value) {
            // Loop feedback passes through block parameters. Allow one extension
            // per executable incoming edge, plus one repeat; new predecessors
            // therefore get room to contribute without a fixed global cap.
            let incoming = self.incoming[block];
            let extensions = &mut self.parameter_extensions[value];
            if next != old && matches!(next, Fact::Range(_)) {
                if *extensions >= incoming + 1 {
                    next = Fact::Varying;
                } else {
                    *extensions += 1;
                }
            }
        }
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
        rewrite::successors(f.dfg().inst(inst), |value| self.fact(f, value))
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
                    if self.edges.insert(EdgeRef {
                        inst,
                        index: index as u32,
                    }) {
                        self.incoming[edge.block] += 1;
                    }
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
        self.evaluations += 1;
        let args = dfg.operands(inst);
        if let Some(folds) = rewrite::fold_inst(dfg, inst, |v| self.fact(f, v).constant()) {
            for (&value, fold) in results.iter().zip(folds) {
                let fact = match fold {
                    Fold::Operand(i) => self.fact(f, args[i]),
                    Fold::Constant(c) => Fact::Constant(c),
                };
                self.update(f, value, fact);
            }
            return;
        }
        let fact = if !rewrite::can_fold(view.opcode()) {
            // Unsupported/effectful operations cannot wait on unknown inputs.
            Fact::Varying
        } else if view.opcode() == Opcode::Select {
            // Join the arms before the general Unknown check: one pending arm
            // must not prevent a varying condition from propagating the other.
            self.fact(f, args[0])
                .select(self.fact(f, args[1]), self.fact(f, args[2]))
        } else if args.iter().any(|&v| self.fact(f, v) == Fact::Unknown) {
            Fact::Unknown
        } else {
            crate::range::evaluate(dfg, inst, |v| self.fact(f, v)).unwrap_or(Fact::Varying)
        };
        for &value in results {
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
                    // Concrete and range evaluation only establish constants
                    // when the operation is free of effects and traps.
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
