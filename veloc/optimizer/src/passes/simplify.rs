//! Worklist constant propagation and single-instruction reductions. Structural
//! expression rules run in ExpressionPass; directed rewrites come from spec.
//! Every CFG edge is considered; this pass does not prune branches.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile, rewrite};
use cranelift_entity::SecondaryMap;
use smallvec::SmallVec;
use std::collections::VecDeque;
use veloc_analyzer::AnalysisManager;
use veloc_mir::{Block, FuncBody, Inst, Value, function::EdgeRef};

pub struct SimplifyPass;

impl FunctionPass for SimplifyPass {
    fn reuse_key(&self) -> Option<core::any::TypeId> {
        Some(core::any::TypeId::of::<Self>())
    }
    fn name(&self) -> &'static str {
        "SimplifyPass"
    }

    fn run(
        &self,
        am: &mut AnalysisManager<'_>,
        config: &OptConfig,
        metrics: &Profile,
    ) -> PassOutcome {
        let mut work = Worklist::new(am.function());
        let func = am.function_mut();
        let mut folded = 0;
        let mut parameters = 0;
        while let Some(item) = work.pop() {
            match item {
                Item::Instruction(inst) => {
                    if func.layout().inst_block(inst).is_none() {
                        continue;
                    }
                    if fold_instruction(func, inst, &mut work) {
                        folded += 1;
                    }
                }
                Item::Parameters(block) => {
                    if work.incoming[block].is_empty() {
                        continue;
                    }
                    let params = func.dfg().block_params(block).to_vec();
                    for (index, param) in params.into_iter().enumerate() {
                        if func.dfg().uses(param).next().is_none() {
                            continue;
                        }
                        let mut literal = None;
                        let mut agrees = true;
                        for &edge in &work.incoming[block] {
                            let argument = func
                                .dfg()
                                .inst(edge.inst)
                                .successor(edge.index as usize)
                                .unwrap()
                                .args[index];
                            // A loop carrying the same parameter preserves the
                            // constant supplied by its other incoming edges.
                            if argument == param {
                                continue;
                            }
                            if func.dfg().as_const(argument).is_none()
                                || literal.is_some_and(|value| value != argument)
                            {
                                agrees = false;
                                break;
                            }
                            literal = Some(argument);
                        }
                        if agrees && let Some(value) = literal {
                            work.replace(func, param, value);
                            parameters += 1;
                        }
                    }
                }
            }
        }
        metrics.count("simplify.folded_insts", folded);
        metrics.count("simplify.constant_params", parameters);
        if folded + parameters == 0 {
            PassOutcome::Unchanged
        } else {
            if config.is_debug_enabled("simplify") {
                log::info!("Simplified {folded} instructions and {parameters} block parameters");
            }
            PassOutcome::Changed
        }
    }
}

fn fold_instruction(func: &mut FuncBody, inst: Inst, work: &mut Worklist) -> bool {
    if let Some(rewrite) = crate::rewrite::find_rewrite(func, inst) {
        rewrite.apply(func, inst);
        work.changed(func, inst);
        return true;
    }
    let dfg = func.dfg();
    let Some(folds) = rewrite::fold_inst(dfg, inst, |value| dfg.as_scalar_const(value)) else {
        return false;
    };
    let args: SmallVec<[Value; 3]> = dfg.operands(inst).into();
    let results: SmallVec<[Value; 2]> = dfg.inst_results(inst).into();
    assert_eq!(folds.len(), results.len(), "fold result arity");
    for (value, fold) in results.into_iter().zip(folds) {
        let replacement = match fold {
            rewrite::Fold::Operand(index) => args[index],
            rewrite::Fold::Constant(value) => func.edit().constant(value.into()),
        };
        work.replace(func, value, replacement);
    }
    // Successful evaluation proves the operation cannot trap for these inputs.
    func.edit().erase_inst(inst);
    true
}

enum Item {
    Instruction(Inst),
    Parameters(Block),
}

struct Worklist {
    pending: VecDeque<Item>,
    instructions: SecondaryMap<Inst, bool>,
    blocks: SecondaryMap<Block, bool>,
    incoming: SecondaryMap<Block, Vec<EdgeRef>>,
}

impl Worklist {
    fn new(func: &FuncBody) -> Self {
        let mut work = Self {
            pending: VecDeque::new(),
            instructions: SecondaryMap::new(),
            blocks: SecondaryMap::new(),
            incoming: SecondaryMap::new(),
        };
        for block in func.layout().block_order() {
            work.block(block);
            for inst in func.layout().block_insts(block) {
                let view = func.dfg().inst(inst);
                if !view.is_terminator() {
                    work.instruction(inst);
                }
                let mut index = 0;
                view.visit_successors(|edge| {
                    work.incoming[edge.block].push(EdgeRef { inst, index });
                    index += 1;
                });
            }
        }
        work
    }

    fn instruction(&mut self, inst: Inst) {
        if !self.instructions[inst] {
            self.instructions[inst] = true;
            self.pending.push_back(Item::Instruction(inst));
        }
    }

    fn block(&mut self, block: Block) {
        if !self.blocks[block] {
            self.blocks[block] = true;
            self.pending.push_back(Item::Parameters(block));
        }
    }

    fn pop(&mut self) -> Option<Item> {
        let item = self.pending.pop_front()?;
        match item {
            Item::Instruction(inst) => self.instructions[inst] = false,
            Item::Parameters(block) => self.blocks[block] = false,
        }
        Some(item)
    }

    fn users(&mut self, func: &FuncBody, value: Value) {
        for site in func.dfg().uses(value) {
            let inst = site.inst();
            let view = func.dfg().inst(inst);
            if view.is_terminator() {
                // Only edge arguments affect destination parameters.
                view.visit_successors(|edge| {
                    if edge.args.contains(&value) {
                        self.block(edge.block);
                    }
                });
            } else {
                self.instruction(inst);
            }
        }
    }

    fn changed(&mut self, func: &FuncBody, inst: Inst) {
        self.instruction(inst);
        for &value in func.dfg().inst_results(inst) {
            self.users(func, value);
        }
    }

    fn replace(&mut self, func: &mut FuncBody, old: Value, new: Value) {
        // Read old uses before replacement detaches them; queued work runs later.
        self.users(func, old);
        func.edit().replace_all_uses(old, new);
    }
}
