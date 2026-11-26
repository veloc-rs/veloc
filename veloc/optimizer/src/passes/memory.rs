//! Block-local forwarding and elimination of redundant loads and stores.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use hashbrown::HashSet;
use std::collections::VecDeque;
use veloc_analyzer::AnalysisManager;
use veloc_mir::{
    FuncBody, FuncDecls, Inst, InstView, Opcode, Type, Value,
    memory::{Access, ByteRange, Location},
};
use veloc_types::{DataLayout, Region};

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
    range: ByteRange,
    region: Region,
    ty: Type,
    value: Value,
    // A nontrapping store that no intervening read or trap can observe.
    pending_store: Option<Inst>,
}

impl Cell {
    fn location(&self) -> Location {
        Location {
            region: self.region,
            range: Some(self.range),
        }
    }
}

fn private_slots(func: &FuncBody) -> HashSet<Value> {
    let dfg = func.dfg();
    let mut private = HashSet::new();
    let mut pointers = Vec::new();
    'slot: for &slot in func.stack_slots() {
        pointers.clear();
        pointers.push(slot);
        while let Some(ptr) = pointers.pop() {
            for site in dfg.uses(ptr) {
                let inst = site.inst();
                if func.layout().inst_block(inst).is_none() {
                    continue;
                }
                match dfg.inst(inst) {
                    InstView::PtrOffset { .. } => {
                        pointers.extend_from_slice(dfg.inst_results(inst))
                    }
                    InstView::Load { .. } => {}
                    InstView::Store { value, .. } if value != ptr => {}
                    _ => continue 'slot,
                }
            }
        }
        private.insert(slot);
    }
    private
}

pub fn run_memory(
    func: &mut FuncBody,
    decls: &FuncDecls,
    layout: &DataLayout,
    metrics: &Profile,
) -> bool {
    let private = private_slots(func);
    let mut dead = Vec::new();
    let mut cells = VecDeque::new();
    let mut next = func.layout().block_order().next();
    while let Some(block) = next {
        cells.clear();
        let mut cursor = func.layout().first_inst(block);
        while let Some(inst) = cursor {
            cursor = func.layout().next_inst(inst);
            if let Some(access) = func.access(inst) {
                visit_access(func, inst, access, layout, &mut cells, &mut dead);
                continue;
            }
            let view = func.dfg().inst(inst);
            if view.opcode() == Opcode::Return {
                dead.extend(
                    cells
                        .iter()
                        .filter(|c| private.contains(&c.range.base))
                        .filter_map(|c| c.pending_store),
                );
                break;
            }

            let effect = view.memory_effect(decls);
            if view.may_free(decls) || effect.may_write() {
                cells.clear();
            } else if effect.may_read()
                || view.may_trap(decls)
                || matches!(view, InstView::Call { .. })
            {
                // Reads and exits can expose a store even if its bytes stay unchanged.
                for cell in &mut cells {
                    cell.pending_store = None;
                }
            }
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

fn visit_access(
    func: &mut FuncBody,
    inst: Inst,
    access: Access,
    layout: &DataLayout,
    cells: &mut VecDeque<Cell>,
    dead: &mut Vec<Inst>,
) {
    let location = access.location(layout);
    let aliases = |cell: &Cell| func.may_alias(location, cell.location(), layout);
    if access.flags.is_volatile() {
        // A fault can expose any prior store. A device read can change the addressed bytes.
        cells.retain_mut(|cell| {
            cell.pending_store = None;
            !aliases(cell)
        });
        return;
    }

    let view = func.dfg().inst(inst);
    let (range, value, pending_store) = if view.opcode().memory_effect().may_write() {
        cells.retain(|cell| {
            if !aliases(cell) {
                return true;
            }
            if let (Some(range), Some(store)) = (location.range, cell.pending_store)
                && range.covers(&cell.range)
            {
                dead.push(store);
            }
            false
        });
        let (Some(range), InstView::Store { value, .. }) = (location.range, view) else {
            return;
        };
        (range, value, Some(inst))
    } else {
        let result = func.dfg().first_result(inst).expect("load result");
        if let Some(range) = location.range {
            if let Some(cell) = cells.iter().find(|cell| {
                cell.region == location.region && cell.range == range && cell.ty == access.ty
            }) {
                func.edit().replace_all_uses(result, cell.value);
                dead.push(inst);
                return;
            }
            if func.dfg().use_empty(result) {
                dead.push(inst);
                return;
            }
        }
        for cell in cells.iter_mut().filter(|cell| aliases(cell)) {
            cell.pending_store = None;
        }
        let Some(range) = location.range else { return };
        (range, result, None)
    };
    if cells.len() == MAX_ACCESSES {
        cells.pop_front();
    }
    cells.push_back(Cell {
        range,
        region: location.region,
        ty: access.ty,
        value,
        pending_store,
    });
}
