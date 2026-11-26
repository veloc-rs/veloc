//! Block-local forwarding and elimination of redundant loads and stores.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use hashbrown::HashSet;
use std::collections::VecDeque;
use veloc_analyzer::AnalysisManager;
use veloc_mir::{FuncBody, FuncDecls, Inst, InstView, Opcode, Type, Value, memory::Access};
use veloc_types::DataLayout;

const MAX_ACCESSES: usize = 64;

pub struct MemoryPass;

impl FunctionPass for MemoryPass {
    fn name(&self) -> &'static str {
        "MemoryPass"
    }

    fn reuse_key(&self) -> Option<core::any::TypeId> {
        Some(core::any::TypeId::of::<Self>())
    }

    fn run(
        &self,
        am: &mut AnalysisManager<'_>,
        config: &OptConfig,
        metrics: &Profile,
    ) -> PassOutcome {
        let layout = &config.data_layout;
        let decls = am.decls();
        if run_memory(am.function_mut(), decls, layout, metrics) {
            PassOutcome::Changed
        } else {
            PassOutcome::Unchanged
        }
    }
}

struct Cell {
    access: Access,
    ty: Type,
    value: Value,
    // A nontrapping store that no intervening read or trap can observe.
    pending_store: Option<Inst>,
}

fn private_slots(func: &FuncBody) -> HashSet<Value> {
    let mut slots: HashSet<_> = func.stack_slots().iter().copied().collect();
    if slots.is_empty() {
        return slots;
    }
    let mut visited = HashSet::new();
    let mut mark_escape = |mut value| {
        // Follow derived pointers to their allocation regardless of chain length.
        while visited.insert(value) {
            match func.dfg().value_inst_view(value) {
                Some(InstView::PtrOffset { ptr, .. }) => value = ptr,
                _ => {
                    slots.remove(&value);
                    break;
                }
            }
        }
    };
    for block in func.layout().block_order() {
        for inst in func.layout().block_insts(block) {
            match func.dfg().inst(inst) {
                InstView::PtrOffset { .. } | InstView::Load { .. } => {}
                // The stored value may escape, even if it is also the address.
                InstView::Store { value, .. } => mark_escape(value),
                _ => {
                    for &value in func.dfg().operands(inst) {
                        mark_escape(value);
                    }
                }
            }
        }
    }
    slots
}

pub fn run_memory(
    func: &mut FuncBody,
    decls: &FuncDecls,
    layout: &DataLayout,
    metrics: &Profile,
) -> bool {
    let private = private_slots(func);
    let mut dead = HashSet::new();
    let mut next = func.layout().block_order().next();
    while let Some(block) = next {
        let mut cells: VecDeque<Cell> = VecDeque::new();
        let mut cursor = func.layout().first_inst(block);
        while let Some(inst) = cursor {
            cursor = func.layout().next_inst(inst);
            let view = func.dfg().inst(inst);
            if view.opcode() == Opcode::Return {
                dead.extend(
                    cells
                        .iter()
                        .filter(|c| private.contains(&c.access.base))
                        .filter_map(|c| c.pending_store),
                );
                break;
            }

            let Some(access) = func.access(inst, layout) else {
                let effect = view.memory_effect(decls);
                if view.has_volatile_access() || view.may_free(decls) || effect.may_write() {
                    cells.clear();
                } else if effect.may_read()
                    || view.may_trap(decls)
                    || matches!(view, InstView::Call { .. })
                {
                    // A read, trap, or call that never returns can expose the prior store.
                    // On normal continuation the recorded bytes still have their old value.
                    for cell in &mut cells {
                        cell.pending_store = None;
                    }
                }
                continue;
            };

            if access.flags.is_volatile() {
                // A fault can expose pending writes even to disjoint locations.
                for cell in &mut cells {
                    cell.pending_store = None;
                }
                // Even a volatile read can change the addressed device register.
                cells.retain(|cell| !func.may_alias(access, cell.access, layout));
                continue;
            }

            let (value, pending_store) = match view {
                InstView::Store { value, .. } => {
                    cells.retain(|cell| {
                        if !func.may_alias(access, cell.access, layout) {
                            return true;
                        }
                        if access.covers(cell.access)
                            && let Some(store) = cell.pending_store
                        {
                            dead.insert(store);
                        }
                        false
                    });
                    (value, Some(inst))
                }
                InstView::Load { .. } => {
                    let result = func.dfg().first_result(inst).expect("load result");
                    let ty = func.dfg().value_type(result);
                    let previous = cells.iter().find(|cell| {
                        cell.access.base == access.base
                            && cell.access.offset == access.offset
                            && cell.ty == ty
                    });
                    if let Some(previous) = previous {
                        func.edit().replace_all_uses(result, previous.value);
                    }
                    if func.dfg().uses(result).next().is_none() {
                        dead.insert(inst);
                        continue;
                    }
                    // A retained read observes any overlapping store.
                    for cell in &mut cells {
                        if func.may_alias(access, cell.access, layout) {
                            cell.pending_store = None;
                        }
                    }
                    (result, None)
                }
                _ => unreachable!("access requires a load or store"),
            };
            if cells.len() == MAX_ACCESSES {
                cells.pop_front();
            }
            cells.push_back(Cell {
                access,
                ty: func.dfg().value_type(value),
                value,
                pending_store,
            });
        }
        next = func.layout().next_block(block);
    }
    if dead.is_empty() {
        return false;
    }
    metrics.count("memory.removed_insts", dead.len() as u64);
    func.edit().erase_insts(&dead);
    true
}
