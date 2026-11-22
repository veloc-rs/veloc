//! Block-local forwarding and dead-store elimination for fixed-width memory accesses.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use hashbrown::HashSet;
use veloc_analyzer::AnalysisManager;
use veloc_mir::{FuncBody, FuncDecls, Inst, InstView, Opcode, Type, Value, memory::Access};
use veloc_types::DataLayout;

const MAX_CELLS: usize = 64;

pub struct MemoryPass;

impl FunctionPass for MemoryPass {
    fn name(&self) -> &'static str {
        "MemoryPass"
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
    let mut forwarded = false;
    let mut dead = HashSet::new();
    let mut next = func.layout().block_order().next();
    while let Some(block) = next {
        let mut cells: Vec<Cell> = Vec::new();
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

            if view.has_volatile_access() {
                cells.clear();
                continue;
            }
            let Some(access) = func.access(inst, layout) else {
                if !view.memory_effect().is_none() || view.may_trap(decls) {
                    cells.clear();
                }
                continue;
            };

            let may_trap = view.may_trap(decls);
            if may_trap {
                // A trap can expose earlier writes before a later overwrite executes.
                for cell in &mut cells {
                    cell.pending_store = None;
                }
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
                    (value, (!may_trap).then_some(inst))
                }
                InstView::Load { .. } => {
                    let result = func.dfg().first_result(inst).expect("load result");
                    let ty = func.dfg().value_type(result);
                    if let Some(previous) = cells.iter().find(|cell| {
                        cell.access.base == access.base
                            && cell.access.offset == access.offset
                            && cell.ty == ty
                    }) {
                        if func.dfg().uses(result).next().is_some() {
                            func.edit().replace_all_uses(result, previous.value);
                            forwarded = true;
                        }
                        if !may_trap {
                            dead.insert(inst);
                        }
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
            cells.push(Cell {
                access,
                ty: func.dfg().value_type(value),
                value,
                pending_store,
            });
            if cells.len() > MAX_CELLS {
                cells.clear();
            }
        }
        next = func.layout().next_block(block);
    }
    if dead.is_empty() {
        return forwarded;
    }
    metrics.count("memory.removed_insts", dead.len() as u64);
    func.edit()
        .erase_insts(&dead.into_iter().collect::<Vec<_>>());
    true
}
