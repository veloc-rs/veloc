//! Bounded block-local forwarding and dead-store elimination for entry objects.
//! Unknown aliases, lifetime effects and volatile accesses are barriers.
use crate::{FunctionPass, Metrics, OptConfig, PreservedAnalyses};
use hashbrown::HashSet;
use veloc_analyzer::AnalysisManager;
use veloc_mir::{Function, Inst, InstView, Opcode, Type, Value};

pub struct MemoryPass;

impl FunctionPass for MemoryPass {
    fn name(&self) -> &str {
        "MemoryPass"
    }

    fn run(
        &self,
        am: &mut AnalysisManager<'_>,
        _: &OptConfig,
        metrics: &mut Metrics,
    ) -> PreservedAnalyses {
        if run_memory(am.function_mut(), metrics) {
            PreservedAnalyses::none()
        } else {
            PreservedAnalyses::all()
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

pub fn run_memory(func: &mut Function, metrics: &mut Metrics) -> bool {
    // A stack pointer passed anywhere except through a derived address or a
    // memory address may escape. Storing a pointer also escapes its object.
    let mut escaped = HashSet::new();
    let mut escape_roots = Vec::new();
    let blocks = func.layout().block_order().to_vec();
    for &block in &blocks {
        for &inst in &func.layout().blocks()[block].insts {
            let view = func.dfg().inst(inst);
            let access = func.memory_access(inst);
            for &value in func.dfg().operands(inst) {
                let address_only = match view {
                    InstView::PtrOffset { .. } => true,
                    _ => access.is_some_and(|a| value == a.ptr && a.stored != Some(value)),
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
        let insts = func.layout().blocks()[block].insts.clone();
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
            let known = func.memory_access(inst).and_then(|access| {
                if view.has_volatile_access() {
                    return None;
                }
                let (object, offset) = func.stack_access(access, None)?;
                Some((access, object, offset, access.bytes(None)?))
            });
            let Some((access, object, offset, bytes)) = known else {
                if !effect.is_none()
                    || view.has_volatile_access()
                    || view.opcode().spec().may_trap()
                {
                    cells.clear();
                }
                continue;
            };
            if let Some(value) = access.stored {
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
                    ty: access.ty,
                    value,
                    store: Some(inst),
                });
            } else {
                let previous = cells.iter().find(|cell| {
                    cell.object == object && cell.offset == offset && cell.ty == access.ty
                });
                if let Some(previous) = previous {
                    let value = previous.value;
                    let result = func.dfg().first_result(inst).expect("read result");
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
                        ty: access.ty,
                        value: func.dfg().first_result(inst).expect("read result"),
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
    metrics.add("memory.removed_insts", dead.len() as u64);
    func.edit()
        .erase_insts(&dead.into_iter().collect::<Vec<_>>());
    true
}
