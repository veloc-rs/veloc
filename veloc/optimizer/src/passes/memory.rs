//! Bounded block-local forwarding and dead-store elimination for entry objects.
//! Unknown aliases, lifetime effects and volatile accesses are barriers.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use hashbrown::HashSet;
use veloc_analyzer::AnalysisManager;
use veloc_mir::{FuncBody, Inst, InstView, Opcode, Type, Value};
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
        let Some(layout) = config.data_layout.as_ref() else {
            return PassOutcome::Unchanged;
        };
        if run_memory(am.function_mut(), layout, metrics) {
            PassOutcome::Changed
        } else {
            PassOutcome::Unchanged
        }
    }
}

struct Cell {
    object: Inst,
    offset: u32,
    bytes: u32,
    ty: Type,
    value: Value,
    // Deletable until a read observes it.
    pending_store: Option<Inst>,
}

impl Cell {
    fn overlaps(&self, object: Inst, offset: u32, bytes: u32) -> bool {
        self.object == object && self.offset < offset + bytes && offset < self.offset + self.bytes
    }
}

fn escaped_allocas(func: &FuncBody) -> HashSet<Inst> {
    let mut escaped = HashSet::new();
    let mut visited = HashSet::new();
    let mut mark_escape = |mut value| {
        // Follow derived pointers to their allocation regardless of chain length.
        while visited.insert(value) {
            let Some(inst) = func.dfg().value_inst(value) else {
                break;
            };
            match func.dfg().inst(inst) {
                InstView::PtrOffset { ptr, .. } => value = ptr,
                InstView::Alloca { .. } => {
                    escaped.insert(inst);
                    break;
                }
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

/// A nonvolatile load or store wholly inside an entry allocation.
fn stack_access(func: &FuncBody, inst: Inst, layout: &DataLayout) -> Option<(Inst, u32, u32)> {
    let flags = func.dfg().inst(inst).memory_flags()?;
    if flags.is_volatile() {
        return None;
    }
    let location = func.memory_location(inst, layout)?;
    let (object, offset) = func.stack_access(location, flags)?;
    Some((object, offset, location.bytes))
}

pub fn run_memory(func: &mut FuncBody, layout: &DataLayout, metrics: &Profile) -> bool {
    let escaped = escaped_allocas(func);
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
                        .filter(|c| !escaped.contains(&c.object))
                        .filter_map(|c| c.pending_store),
                );
                break;
            }

            let Some((object, offset, bytes)) = stack_access(func, inst, layout) else {
                if !view.memory_effect().is_none()
                    || view.has_volatile_access()
                    || view.opcode().spec().may_trap()
                {
                    cells.clear();
                }
                continue;
            };

            let (value, pending_store) = match view {
                InstView::Store { value, .. } => {
                    cells.retain(|cell| {
                        if !cell.overlaps(object, offset, bytes) {
                            return true;
                        }
                        if offset <= cell.offset
                            && offset + bytes >= cell.offset + cell.bytes
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
                        cell.object == object && cell.offset == offset && cell.ty == ty
                    }) {
                        func.edit().replace_all_uses(result, previous.value);
                        dead.insert(inst);
                        continue;
                    }
                    // A retained read observes any overlapping store.
                    for cell in &mut cells {
                        if cell.overlaps(object, offset, bytes) {
                            cell.pending_store = None;
                        }
                    }
                    (result, None)
                }
                _ => unreachable!("memory_location requires a load or store"),
            };
            cells.push(Cell {
                object,
                offset,
                bytes,
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
