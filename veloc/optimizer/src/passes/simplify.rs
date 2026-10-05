//! Worklist constant propagation and root-local simplification before memory
//! optimization. Every CFG edge is considered; this pass does not prune branches.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile, evaluate};
use cranelift_entity::SecondaryMap;
use smallvec::SmallVec;
use std::collections::VecDeque;
use veloc_analyzer::AnalysisManager;
use veloc_mir::{Block, FuncBody, Inst, InstView, Value, function::EdgeRef};

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
                    if func.dfg().inst(inst).is_terminator() {
                        func.dfg()
                            .inst(inst)
                            .visit_successors(|edge| work.block(edge.block));
                    } else if fold_rewrite(func, inst, config, &mut work)
                        || fold_instruction(func, inst, &mut work)
                    {
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
                            let mut argument = None;
                            let mut position = 0;
                            func.dfg().inst(edge.inst).visit_successors(|successor| {
                                if position == edge.index {
                                    argument = successor.args.get(index).copied();
                                }
                                position += 1;
                            });
                            let Some(argument) = argument else {
                                agrees = false;
                                break;
                            };
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

/// Apply the same directed rules that equality rebuilding consumes.
fn fold_rewrite(f: &mut FuncBody, inst: Inst, config: &OptConfig, work: &mut Worklist) -> bool {
    if !evaluate::can_rewrite(f.dfg().inst(inst).opcode()) {
        return false;
    }
    let Some(result) = f.dfg().first_result(inst) else {
        return false;
    };
    let cx = crate::rewrite::Context {
        body: f,
        layout: config.data_layout,
    };
    let Some(plan) = evaluate::rewrite(&cx, result) else {
        return false;
    };
    let (replacement, created) = crate::rewrite::apply(f, inst, &plan);
    for instruction in created {
        work.instruction(instruction);
    }
    work.replace(f, result, replacement);
    f.edit().erase_inst(inst);
    true
}

fn fold_instruction(func: &mut FuncBody, inst: Inst, work: &mut Worklist) -> bool {
    if let Some(access) = inst.memory_access(func.dfg())
        && let Some(address) = func.dfg().value_inst(access.ptr)
        && let InstView::PtrOffset { ptr, offset } = func.dfg().inst(address)
        && let Ok(combined) = u32::try_from(access.offset + i64::from(offset))
    {
        match func.dfg().inst(inst) {
            InstView::Load { flags, .. } => {
                func.edit()
                    .replace_inst(inst, |w| w.load(ptr, combined, flags));
            }
            InstView::Store { value, flags, .. } => {
                func.edit()
                    .replace_inst(inst, |w| w.store(ptr, value, combined, flags));
            }
            _ => return false,
        }
        work.instruction(inst);
        return true;
    }
    if let InstView::PtrIndex { ptr, index, imm_id } = func.dfg().inst(inst)
        && let Some(index) = func.dfg().as_scalar_const(index)
    {
        let offset = index
            .to_bits()
            .wrapping_mul(imm_id.scale as u64)
            .wrapping_add(imm_id.offset as u64) as i64;
        if let Ok(offset) = i32::try_from(offset) {
            func.edit()
                .replace_inst(inst, |w| w.ptr_offset(ptr, offset));
            work.instruction(inst);
            return true;
        }
    }
    let dfg = func.dfg();
    let Some(folds) = evaluate::reduce_inst(dfg, inst, |value| dfg.as_scalar_const(value)) else {
        return false;
    };
    let args: SmallVec<[Value; 3]> = dfg.operands(inst).into();
    let results: SmallVec<[Value; 2]> = dfg.inst_results(inst).into();
    assert_eq!(folds.len(), results.len(), "fold result arity");
    for (value, fold) in results.into_iter().zip(folds) {
        let replacement = match fold {
            evaluate::Fold::Operand(index) => args[index],
            evaluate::Fold::Constant(value) => func.edit().constant(value.into()),
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
                work.instruction(inst);
                let mut index = 0;
                func.dfg().inst(inst).visit_successors(|edge| {
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

    fn replace(&mut self, func: &mut FuncBody, old: Value, new: Value) {
        for site in func.dfg().uses(old) {
            self.instruction(site.inst());
        }
        func.edit().replace_all_uses(old, new);
    }
}
