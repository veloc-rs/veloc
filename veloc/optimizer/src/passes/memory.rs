//! Bounded block-local forwarding and dead-store elimination for entry objects.
//! Unknown aliases, lifetime effects and volatile accesses are barriers.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use hashbrown::HashSet;
use veloc_analyzer::AnalysisManager;
use veloc_mir::{FuncBody, Inst, InstView, Opcode, Type, Value};

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
    store: Option<Inst>,
}

impl Cell {
    fn overlaps(&self, object: Inst, offset: u32, bytes: u32) -> bool {
        self.object == object && self.offset < offset + bytes && offset < self.offset + self.bytes
    }
}

pub fn run_memory(
    func: &mut FuncBody,
    layout: &veloc_types::DataLayout,
    metrics: &Profile,
) -> bool {
    // A stack pointer passed anywhere except through a derived address or a
    // memory address may escape. Storing a pointer also escapes its object.
    let mut escaped = HashSet::new();
    let mut escape_roots = Vec::new();
    let blocks = func.layout().block_order().collect::<Vec<_>>();
    for &block in &blocks {
        for inst in func.layout().block_insts(block) {
            let view = func.dfg().inst(inst);
            for &value in func.dfg().operands(inst) {
                let address_only = match view {
                    InstView::PtrOffset { .. } => true,
                    InstView::Load { ptr, .. } => value == ptr,
                    InstView::Store {
                        ptr, value: stored, ..
                    } => value == ptr && value != stored,
                    _ => false,
                };
                if !address_only {
                    escape_roots.push(value);
                }
            }
        }
    }

    // Unlike a bounded positive bounds proof, failure to follow an escape must
    // never mean "does not escape". Traverse the whole derived-address chain,
    // once per value, including long chains and malformed cycles.
    let mut visited = HashSet::new();
    while let Some(value) = escape_roots.pop() {
        if !visited.insert(value) {
            continue;
        }
        let Some(inst) = func.dfg().value_inst(value) else {
            continue;
        };
        match func.dfg().inst(inst) {
            InstView::Alloca { .. } => {
                escaped.insert(inst);
            }
            InstView::PtrOffset { ptr, .. } => escape_roots.push(ptr),
            _ => {}
        }
    }

    let mut dead = HashSet::new();
    for block in blocks {
        let mut cells: Vec<Cell> = Vec::new();
        let insts = func.layout().block_insts(block).collect::<Vec<_>>();
        for inst in insts {
            let view = func.dfg().inst(inst);
            if view.opcode() == Opcode::Return {
                for cell in &cells {
                    if !escaped.contains(&cell.object)
                        && let Some(store) = cell.store
                    {
                        dead.insert(store);
                    }
                }
            }
            let effect = view.memory_effect();
            let known = func.memory_location(inst, layout).and_then(|location| {
                let flags = match view {
                    InstView::Load { flags, .. } | InstView::Store { flags, .. } => flags,
                    _ => return None,
                };
                if flags.is_volatile() {
                    return None;
                }
                let (object, offset) = func.stack_access(location, flags)?;
                Some((object, offset, location.bytes))
            });
            let Some((object, offset, bytes)) = known else {
                if !effect.is_none()
                    || view.has_volatile_access()
                    || view.opcode().spec().may_trap()
                {
                    cells.clear();
                }
                continue;
            };
            if let InstView::Store { value, .. } = view {
                cells.retain(|cell| {
                    if !cell.overlaps(object, offset, bytes) {
                        return true;
                    }
                    if offset <= cell.offset
                        && offset + bytes >= cell.offset + cell.bytes
                        && let Some(store) = cell.store
                    {
                        dead.insert(store);
                    }
                    false
                });
                cells.push(Cell {
                    object,
                    offset,
                    bytes,
                    ty: func.dfg().value_type(value),
                    value,
                    store: Some(inst),
                });
            } else if let InstView::Load { .. } = view {
                let result = func.dfg().first_result(inst).expect("load result");
                let ty = func.dfg().value_type(result);
                let previous = cells
                    .iter()
                    .find(|cell| cell.object == object && cell.offset == offset && cell.ty == ty);
                if let Some(previous) = previous {
                    let value = previous.value;
                    func.edit().replace_all_uses(result, value);
                    dead.insert(inst);
                } else {
                    // An actual read observes prior stores, preventing their deletion.
                    for cell in &mut cells {
                        if cell.overlaps(object, offset, bytes) {
                            cell.store = None;
                        }
                    }
                    cells.push(Cell {
                        object,
                        offset,
                        bytes,
                        ty,
                        value: result,
                        store: None,
                    });
                }
            }
            // Bound both work and memory on large generated basic blocks.
            if cells.len() > 64 {
                cells.clear();
            }
        }
    }
    if dead.is_empty() {
        return false;
    }
    metrics.count("memory.removed_insts", dead.len() as u64);
    func.edit()
        .erase_insts(&dead.into_iter().collect::<Vec<_>>());
    true
}
