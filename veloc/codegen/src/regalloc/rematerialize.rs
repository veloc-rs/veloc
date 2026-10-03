//! Recreate cheap constants instead of reading their private spill slots.
//! The rewrite inspects the complete allocation plan, including edge copies;
//! a slot with any other store is never treated as a constant.
use super::{EdgeAllocation, InstAllocation, Transfer};
use crate::target::{SpillKind, TargetRegalloc};
use cranelift_entity::SecondaryMap;
use std::collections::HashMap;
use veloc_lir::{InstId, MachineFunction, MachineOpcode, StackSlot, Type};

pub(super) fn rewrite(
    function: &MachineFunction,
    target: &dyn TargetRegalloc,
    plans: &mut SecondaryMap<InstId, InstAllocation>,
    entry: &mut Vec<Transfer>,
    edges: &mut [EdgeAllocation],
) {
    let mut constants = HashMap::<StackSlot, (u32, i64)>::new();
    for block in function.blocks() {
        for id in function.block_insts(block) {
            let source = function.inst(id);
            let Some(immediate) = target.rematerializable_constant(source) else {
                continue;
            };
            let MachineOpcode::Target(opcode) = source.opcode() else {
                unreachable!()
            };
            let [result] = plans[id].results.as_slice() else {
                continue;
            };
            for &transfer in &plans[id].after {
                if let Transfer::Spill {
                    kind: SpillKind::Store,
                    reg,
                    slot,
                    ty,
                } = transfer
                    && reg == (*result).into()
                    && matches!(ty, Type::I32 | Type::I64 | Type::PTR)
                {
                    constants.insert(slot, (opcode, immediate));
                }
            }
        }
    }
    if constants.is_empty() {
        return;
    }
    let mut stores = HashMap::<StackSlot, usize>::new();
    let transfers = plans
        .iter()
        .flat_map(|(_, p)| p.before.iter().chain(&p.after))
        .chain(entry.iter())
        .chain(edges.iter().flat_map(|e| e.instructions.iter()));
    for transfer in transfers {
        if let Transfer::Spill {
            kind: SpillKind::Store,
            slot,
            ..
        } = transfer
        {
            *stores.entry(*slot).or_default() += 1;
        }
    }
    constants.retain(|slot, _| stores.get(slot) == Some(&1));
    let rewrite = |transfers: &mut Vec<Transfer>| {
        transfers.retain_mut(|transfer| {
            if let Transfer::Spill {
                kind, reg, slot, ..
            } = *transfer
                && let Some(&(opcode, immediate)) = constants.get(&slot)
            {
                if kind == SpillKind::Store {
                    return false;
                }
                *transfer = Transfer::Rematerialize {
                    reg,
                    opcode,
                    immediate,
                };
            }
            true
        })
    };
    for (_, plan) in plans.iter_mut() {
        rewrite(&mut plan.before);
        rewrite(&mut plan.after);
    }
    rewrite(entry);
    for edge in edges {
        rewrite(&mut edge.instructions);
    }
}
