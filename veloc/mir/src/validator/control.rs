//! Structural and SSA validation, independent of opcode-specific type rules.
//! Check handles before any type projection dereferences them.

use super::ValidationError;
use crate::function::Dominators;
use crate::{Block, FunctionRef, Module, Result, Value, ValueDef};
use alloc::vec::Vec;

// Positions are assigned only to instructions attached to a block.
const UNSEEN: usize = usize::MAX;

/// Information retained for SSA validation after instruction contracts pass.
pub(super) struct Structure {
    positions: Vec<usize>,
    successors: Vec<Vec<Block>>,
    predecessors: Vec<Vec<Block>>,
}

/// Temporary membership sets are needed only while checking the structure.
struct Checker<'a> {
    func: &'a FunctionRef<'a>,
    module: &'a Module,
    blocks: Vec<bool>,
    defined: Vec<bool>,
    structure: Structure,
}

impl Structure {
    pub(super) fn check(func: &FunctionRef, module: &Module) -> Result<Self> {
        let body = func.body.expect("defined function");
        let blocks = body.dfg().blocks.len();
        let mut checker = Checker {
            func,
            module,
            blocks: vec![false; blocks],
            defined: vec![false; body.dfg().values().len()],
            structure: Self {
                positions: vec![UNSEEN; body.dfg().instructions().len()],
                successors: vec![Vec::new(); blocks],
                predecessors: vec![Vec::new(); blocks],
            },
        };

        checker.check_blocks()?;
        checker.check_definitions()?;
        // Definitions in later layout blocks may be referenced by earlier ones.
        // Resolve operand handles only after collecting every definition.
        checker.check_operands()?;
        checker.check_entry_params()?;
        Ok(checker.structure)
    }

    pub(super) fn check_ssa(mut self, func: &FunctionRef) -> Result<()> {
        let body = func.body.expect("defined function");
        for block in body.layout().block_order() {
            let data = &body.cfg().blocks[block];
            let mut succs = data.succs.clone();
            let mut preds = data.preds.clone();
            succs.sort_unstable();
            preds.sort_unstable();
            self.predecessors[block.0 as usize].sort_unstable();
            if succs != self.successors[block.0 as usize]
                || preds != self.predecessors[block.0 as usize]
            {
                return func.fail(format!("CFG index disagrees with terminators at {block}"));
            }
        }
        let Some(entry) = func.entry_block() else {
            return Ok(());
        };
        let dom = Dominators::compute(&body.cfg(), entry, body.dfg().blocks.len());
        for block in body.layout().block_order() {
            for inst in body.layout().block_insts(block) {
                for &value in body.dfg().operands(inst) {
                    let definition = body.dfg().value_def(value);
                    let (owner, source) = match definition {
                        ValueDef::Param(owner) => (owner, None),
                        ValueDef::Inst(source) => {
                            (body.layout().inst_block(source).unwrap(), Some(source))
                        }
                    };
                    if owner == block {
                        if let Some(source) = source
                            && self.positions[source.0 as usize] >= self.positions[inst.0 as usize]
                        {
                            return func.fail(format!(
                                "definition of {value} does not precede its use at {inst}"
                            ));
                        }
                    } else if dom.is_reachable(block) && !dom.dominates(owner, block) {
                        return func.fail(format!(
                            "definition of {value} in {owner} does not dominate its use in {block}"
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

impl Checker<'_> {
    /// Establish block membership before inspecting any instructions or targets.
    fn check_blocks(&mut self) -> Result<()> {
        let func = self.func;
        let body = func.body.expect("defined function");
        let layout = &body.layout();
        let Some(signature) = self.module.signatures().get(func.decl.signature) else {
            return func.fail("unknown function signature".into());
        };
        for block in layout.block_order() {
            let Some(present) = self.blocks.get_mut(block.0 as usize) else {
                return func.fail(format!("unknown block {block} in layout"));
            };
            if *present {
                return func.fail(format!("duplicate block {block} in layout"));
            }
            *present = true;
        }
        if let Some(entry) = func.entry_block() {
            if !self.blocks.get(entry.0 as usize).copied().unwrap_or(false) {
                return func.fail("entry block is not in layout".into());
            }
            let params = &body.dfg().blocks[entry].params;
            // Types are checked later, once all parameter handles are valid.
            if params.len() != signature.params().len() {
                return func.fail(format!(
                    "entry parameter count mismatch: expected {}, got {}",
                    signature.params().len(),
                    params.len()
                ));
            }
        } else if !layout.block_order().next().is_none() {
            return func.fail("function with blocks has no entry".into());
        }
        Ok(())
    }

    /// Check instruction placement and definitions, and reconstruct the CFG.
    fn check_definitions(&mut self) -> Result<()> {
        let func = self.func;
        let body = func.body.expect("defined function");
        let dfg = &body.dfg();
        for block in body.layout().block_order() {
            let data = &body.dfg().blocks[block];
            for &param in &data.params {
                self.define(param, ValueDef::Param(block))?;
            }
            let Some(last) = body.layout().last_inst(block) else {
                return Err(ValidationError::EmptyBlock(block).into());
            };
            for (position, inst) in body.layout().block_insts(block).enumerate() {
                let Some(slot) = self.structure.positions.get_mut(inst.0 as usize) else {
                    return func.fail(format!("unknown instruction {inst} in {block}"));
                };
                if *slot != UNSEEN {
                    return func.fail(format!("instruction {inst} appears more than once"));
                }
                *slot = position;
                if body.layout().inst_block(inst) != Some(block) {
                    return func.fail(format!(
                        "instruction {inst} has inconsistent block ownership"
                    ));
                }
                let view = dfg.inst(inst);
                if view.is_terminator() && inst != last {
                    return func.fail(format!("terminator {inst} is not last in {block}"));
                }
                for &value in dfg.inst_results(inst) {
                    self.define(value, ValueDef::Inst(inst))?;
                }
                view.try_visit_successors(|edge| {
                    if !self
                        .blocks
                        .get(edge.block.0 as usize)
                        .copied()
                        .unwrap_or(false)
                    {
                        return func.fail(format!(
                            "successor {} is not in function layout",
                            edge.block
                        ));
                    }
                    if !view.is_terminator() {
                        return func.fail(format!("non-terminator {inst} has a successor"));
                    }
                    self.structure.successors[block.0 as usize].push(edge.block);
                    Ok(())
                })?;
            }
            if !dfg.inst(last).is_terminator() {
                return Err(ValidationError::NoTerminator(block).into());
            }
            // CFG adjacency is a set of blocks; ownership still visits each
            // successor occurrence separately, including parallel edges.
            let outgoing = &mut self.structure.successors[block.0 as usize];
            outgoing.sort_unstable();
            outgoing.dedup();
            for &to in outgoing.iter() {
                self.structure.predecessors[to.0 as usize].push(block);
            }
        }
        Ok(())
    }

    fn define(&mut self, value: Value, owner: ValueDef) -> Result<()> {
        let func = self.func;
        let body = func.body.expect("defined function");
        let Some(data) = body.dfg().values().get(value) else {
            return func.fail(format!("unknown definition {value}"));
        };
        if data.def != owner {
            return func.fail(format!("definition {value} has inconsistent ownership"));
        }
        if self.defined[value.0 as usize] {
            return func.fail(format!("duplicate definition {value}"));
        }
        self.defined[value.0 as usize] = true;
        super::types::check_type(self.module, data.ty)
            .map_err(|error| crate::Error::Message(format!("value {value}: {error}")))?;
        Ok(())
    }

    fn check_operands(&self) -> Result<()> {
        let func = self.func;
        let body = func.body.expect("defined function");
        for block in body.layout().block_order() {
            for inst in body.layout().block_insts(block) {
                for &value in body.dfg().operands(inst) {
                    if !self.defined.get(value.0 as usize).copied().unwrap_or(false) {
                        return func.fail(format!(
                            "operand {value} of {inst} has no attached definition"
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn check_entry_params(&self) -> Result<()> {
        let func = self.func;
        let body = func.body.expect("defined function");
        let Some(entry) = func.entry_block() else {
            return Ok(());
        };
        let signature = &self.module.signatures()[func.decl.signature];
        for (&param, &expected) in body.dfg().blocks[entry]
            .params
            .iter()
            .zip(signature.params())
        {
            if body.dfg().value_type(param) != expected {
                return func.fail(format!(
                    "entry parameter {param} type mismatch: expected {expected}, got {}",
                    body.dfg().value_type(param)
                ));
            }
        }
        Ok(())
    }
}
