//! Linear resource flow. Ownership follows CFG edges,
//! not use counts: two mutually exclusive branches can move the same value.
use crate::{Block, Function, Inst, Result, Successor, Value};
use alloc::{collections::BTreeSet, vec::Vec};
use veloc_types::TypeInfo;

/// Owned values that remain available at a program point.
type Owned = BTreeSet<Value>;

struct Ownership<'a> {
    func: &'a Function,
    // None means unreached; Some(empty) means reached with no owned values.
    entries: Vec<Option<Owned>>,
    pending: Vec<Block>,
}

pub(super) fn validate(func: &Function) -> Result<()> {
    let Some(entry) = func.entry_block else {
        return Ok(());
    };
    if !func.dfg.values().iter().any(|(_, v)| v.ty.is_owned()) {
        return Ok(());
    }
    let mut ownership = Ownership {
        func,
        entries: vec![None; func.layout.blocks.len()],
        pending: vec![entry],
    };
    ownership.entries[entry.0 as usize] = Some(
        func.layout.blocks[entry]
            .params
            .iter()
            .copied()
            .filter(|&v| func.dfg.value_type(v).is_owned())
            .collect(),
    );
    while let Some(block) = ownership.pending.pop() {
        let mut available = ownership.entries[block.0 as usize]
            .clone()
            .expect("pending blocks have an entry state");
        for &inst in &func.layout.blocks[block].insts {
            let view = func.dfg.inst(inst);
            // Non-edge inputs execute once, before choosing any successor.
            view.try_visit_ownership(|value, moves| {
                consume(func, inst, &mut available, value, moves)
            })?;

            let mut has_successors = false;
            view.try_visit_successors(|edge| {
                has_successors = true;
                let next = ownership.transfer_edge(inst, edge, &available)?;
                ownership.merge_entry(inst, edge.block, next)
            })?;
            if has_successors {
                continue;
            }

            for &result in func.dfg.inst_results(inst) {
                if func.dfg.value_type(result).is_owned() && !available.insert(result) {
                    return Err(func
                        .constraint_error(inst, "owned callable overwritten before consumption"));
                }
            }
            // Traps abort the invocation; only normal exits must transfer every
            // remaining owned value. No implicit guest cleanup runs on abort.
            let spec = view.opcode().spec();
            if spec.is_terminator()
                && !spec.traits().contains(crate::inst::OpTraits::ABORT)
                && !available.is_empty()
            {
                return Err(func.constraint_error(
                    inst,
                    "owned callable must be called, transferred or dropped on every exit",
                ));
            }
        }
    }
    Ok(())
}

impl Ownership<'_> {
    /// Each edge starts from the same pre-branch state. Consume its arguments,
    /// then make the destination parameters available under their own SSA names.
    fn transfer_edge(&self, inst: Inst, edge: Successor<'_>, available: &Owned) -> Result<Owned> {
        let mut next = available.clone();
        for &value in edge.args {
            consume(self.func, inst, &mut next, value, true)?;
        }
        for &param in &self.func.layout.blocks[edge.block].params {
            if self.func.dfg.value_type(param).is_owned() && !next.insert(param) {
                return Err(self
                    .func
                    .constraint_error(inst, "backedge overwrites an unconsumed callable"));
            }
        }
        Ok(next)
    }

    /// Every incoming path must agree exactly. Revisiting an identical state
    /// requires no work, so loops terminate without a separate visited set.
    fn merge_entry(&mut self, inst: Inst, block: Block, next: Owned) -> Result<()> {
        let entry = &mut self.entries[block.0 as usize];
        match entry {
            Some(previous) if *previous != next => {
                return Err(self.func.constraint_error(
                    inst,
                    "inconsistent callable ownership at control-flow join",
                ));
            }
            Some(_) => {}
            None => {
                *entry = Some(next);
                self.pending.push(block);
            }
        }
        Ok(())
    }
}

fn consume(
    func: &Function,
    inst: Inst,
    available: &mut Owned,
    value: Value,
    moves: bool,
) -> Result<()> {
    if !func.dfg.value_type(value).is_owned() {
        return Ok(());
    }
    if !moves {
        return Err(func.constraint_error(inst, "operation has no ownership transfer contract"));
    }
    if !available.remove(&value) {
        return Err(
            func.constraint_error(inst, "callable used after move or consumed more than once")
        );
    }
    Ok(())
}
