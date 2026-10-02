//! Values in fixed, non-renamable hardware state units (condition bits, etc.).
//!
//! Selection supplies ordinary SSA def-use edges. State placement is a separate
//! constraint: ordinary register allocation must never invent a copy or spill
//! for these values. This required lowering runs even without scheduling.
use crate::analysis::CfgInfo;
use crate::pipeline::{FunctionPass, FunctionSession, FunctionStage};
use crate::target::TargetInstructions;
use crate::{Error, Result};
use hashbrown::HashMap;
use std::collections::VecDeque;
use veloc_lir::{BlockId, InstId, MachineFunction, MachineOpcode, Reg};

type Available = crate::analysis::state::StateContents;
type Repairs = HashMap<InstId, Vec<InstId>>;

fn apply_repairs(f: &MachineFunction, repairs: &Repairs, id: InstId, available: &mut Available) {
    for &producer in repairs.get(&id).into_iter().flatten() {
        available.apply(f, f.inst(producer));
    }
}

/// Must-availability across the CFG. Absence means unknown, including function
/// entry. Equal incoming SSA identities can flow across any number of blocks.
fn availability(
    f: &MachineFunction,
    cfg: &CfgInfo,
    repairs: &Repairs,
) -> HashMap<BlockId, Available> {
    let mut incoming = HashMap::<BlockId, Available>::new();
    let mut outgoing = HashMap::<BlockId, Available>::new();
    let mut pending = VecDeque::from([f.entry_block()]);
    let mut queued = hashbrown::HashSet::<BlockId>::from([f.entry_block()]);
    while let Some(block) = pending.pop_front() {
        queued.remove(&block);
        let mut state = Available::default();
        if block != f.entry_block() {
            // An unvisited predecessor is lattice top, not an unknown
            // hardware value. This preserves values around loops that do
            // not overwrite them. Entry contents remain genuinely unknown.
            let mut preds = cfg.preds(block).iter().filter_map(|p| outgoing.get(p));
            if let Some(first) = preds.next() {
                state = first.clone();
                for predecessor in preds {
                    state.intersect(predecessor);
                }
            }
        }
        incoming.insert(block, state.clone());
        for id in f.block_insts(block) {
            apply_repairs(f, repairs, id, &mut state);
            state.apply(f, f.inst(id));
        }
        if outgoing.get(&block) != Some(&state) {
            outgoing.insert(block, state);
            for &successor in cfg.succs(block) {
                if queued.insert(successor) {
                    pending.push_back(successor);
                }
            }
        }
    }
    incoming
}

fn recipe(f: &MachineFunction, target: &dyn TargetInstructions, value: Reg) -> Result<InstId> {
    let mut defs = f.defs(value);
    let id = defs
        .next()
        .ok_or_else(|| Error::codegen("state value has no instruction definition"))?
        .inst();
    if defs.next().is_some() {
        return Err(Error::codegen("state value is not SSA"));
    }
    let inst = f.inst(id);
    let MachineOpcode::Target(opcode) = inst.opcode() else {
        return Err(Error::codegen(
            "state producer must be a target instruction",
        ));
    };
    if !target.instruction_metadata(opcode).rematerializable
        || inst.mem_flags().is_some()
        || inst
            .inputs()
            .iter()
            .any(|v| v.is_preg() || f.state_unit(*v).is_some())
    {
        return Err(Error::codegen(
            "overwritten state requires a safe rematerialization recipe",
        ));
    }
    Ok(id)
}

fn plan(f: &MachineFunction, target: &dyn TargetInstructions, cfg: &CfgInfo) -> Result<Repairs> {
    let mut repairs = Repairs::new();
    loop {
        let incoming = availability(f, cfg, &repairs);
        let mut changed = false;
        for block in f.blocks() {
            let mut available = incoming.get(&block).cloned().unwrap_or_default();
            for id in f.block_insts(block) {
                apply_repairs(f, &repairs, id, &mut available);
                let inst = f.inst(id);
                for &value in inst.inputs() {
                    let Some(location) = f.state_unit(value) else {
                        continue;
                    };
                    if available.get(location) == Some(&value) {
                        continue;
                    }
                    let producer = recipe(f, target, value)?;
                    let list = repairs.entry(id).or_default();
                    if list.contains(&producer) {
                        return Err(Error::codegen(
                            "simultaneous state inputs require incompatible hardware contents",
                        ));
                    }
                    list.push(producer);
                    available.apply(f, f.inst(producer));
                    changed = true;
                }
                // Recomputing one result may overwrite a different required bit.
                if inst.inputs().iter().any(|v| {
                    f.state_unit(*v)
                        .is_some_and(|p| available.get(p) != Some(v))
                }) {
                    return Err(Error::codegen(
                        "state inputs cannot coexist; materialize predicates before selection",
                    ));
                }
                available.apply(f, inst);
            }
        }
        // Each iteration permanently adds a producer at a use; no arbitrary
        // iteration budget, and no dependence on optional optimization passes.
        if !changed {
            return Ok(repairs);
        }
    }
}

pub struct ResolveStatePass;

impl FunctionPass for ResolveStatePass {
    fn name(&self) -> &'static str {
        "resolve-state"
    }
    fn input_stage(&self) -> FunctionStage {
        FunctionStage::Selected
    }
    fn run(&self, cx: &mut FunctionSession<'_>) -> Result<()> {
        let f = cx.function();
        if !f.vregs().values().any(|data| data.state_unit().is_some()) {
            return Ok(());
        }
        let cfg = cx.cfg().clone();
        let f = cx.function();
        let repairs = plan(f, cx.target, &cfg)?;
        cx.profile.count(
            "state_rematerializations",
            repairs.values().map(|r| r.len() as u64).sum(),
        );
        let ids: Vec<_> = f.blocks().flat_map(|b| f.block_insts(b)).collect();
        let mut f = cx.edit();
        // Insert all recipes while their SSA definitions are still intact.
        for &id in &ids {
            for &producer in repairs.get(&id).into_iter().flatten() {
                let source = f.inst(producer);
                let opcode = source.opcode();
                let inputs = source.inputs().to_vec();
                let original_results = source.results().to_vec();
                let clobbers: Vec<_> = source.clobbers().collect();
                let mut results = Vec::new();
                for value in original_results {
                    if let Some(location) = f.state_unit(value) {
                        results.push(location);
                    } else if value.is_preg() {
                        return Err(Error::codegen(
                            "cannot rematerialize an explicit physical data result",
                        ));
                    } else {
                        let data = f.vreg_data(value).clone();
                        results.push(f.editor().alloc_vreg_data(data));
                    }
                }
                let mut editor = f.editor();
                let mut cursor = editor.before(id);
                let mut writer = cursor.writer().with_clobbers(clobbers);
                let fields = writer.copy_fields(producer);
                writer.write(opcode, &results, &inputs, fields);
            }
        }
        for id in ids {
            let inst = f.inst(id);
            let inputs: Vec<_> = inst
                .inputs()
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(i, v)| f.state_unit(v).map(|p| (i, p)))
                .collect();
            let results: Vec<_> = inst
                .results()
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(i, v)| f.state_unit(v).map(|p| (i, p)))
                .collect();
            let mut edit = f.editor();
            for (i, p) in inputs {
                edit.set_inst_input(id, i, p);
            }
            for (i, p) in results {
                edit.set_inst_result(id, i, p);
            }
        }
        Ok(())
    }
}
