//! Bounded block-local forwarding and dead-store elimination for stack slots.
//! Unknown aliases, lifetime effects and volatile accesses are barriers.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use hashbrown::HashSet;
use veloc_analyzer::AnalysisManager;
use veloc_mir::{FuncBody, Inst, InstView, Opcode, Type, Value, memory::Access};
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
        if run_memory(am.function_mut(), layout, metrics) {
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
    // Deletable until a read observes it.
    pending_store: Option<Inst>,
}

impl Cell {
    fn overlaps(&self, access: Access) -> bool {
        self.access.base == access.base
            && i128::from(self.access.offset) < access.end()
            && i128::from(access.offset) < self.access.end()
    }
}

fn escaped_slots(func: &FuncBody) -> HashSet<Value> {
    let mut escaped = HashSet::new();
    let mut visited = HashSet::new();
    let mut mark_escape = |mut value| {
        // Follow derived pointers to their allocation regardless of chain length.
        while visited.insert(value) {
            if matches!(
                func.dfg().value_def(value),
                veloc_mir::ValueDef::StackSlot { .. }
            ) {
                escaped.insert(value);
                break;
            }
            match func.dfg().value_inst_view(value) {
                Some(InstView::PtrOffset { ptr, .. }) => value = ptr,
                _ => break,
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
    escaped
}

pub fn run_memory(func: &mut FuncBody, layout: &DataLayout, metrics: &Profile) -> bool {
    let escaped = escaped_slots(func);
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
                        .filter(|c| !escaped.contains(&c.access.base))
                        .filter_map(|c| c.pending_store),
                );
                break;
            }

            if view.has_volatile_access() {
                cells.clear();
                continue;
            }
            let Some(access) = func
                .access(inst, layout)
                .filter(|a| func.dfg().stack_slot_layout(a.base).is_some())
            else {
                if !view.memory_effect().is_none() || view.opcode().spec().may_trap() {
                    cells.clear();
                }
                continue;
            };

            let (value, pending_store) = match view {
                InstView::Store { value, .. } => {
                    cells.retain(|cell| {
                        if !cell.overlaps(access) {
                            return true;
                        }
                        if access.offset <= cell.access.offset
                            && access.end() >= cell.access.end()
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
                    if let Some(previous) = cells.iter().find(|cell| {
                        cell.access.base == access.base
                            && cell.access.offset == access.offset
                            && cell.ty == ty
                    }) {
                        func.edit().replace_all_uses(result, previous.value);
                        dead.insert(inst);
                        continue;
                    }
                    // A retained read observes any overlapping store.
                    for cell in &mut cells {
                        if cell.overlaps(access) {
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
        return false;
    }
    metrics.count("memory.removed_insts", dead.len() as u64);
    func.edit()
        .erase_insts(&dead.into_iter().collect::<Vec<_>>());
    true
}
