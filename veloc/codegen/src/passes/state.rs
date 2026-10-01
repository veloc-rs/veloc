//! Values in fixed, non-renamable hardware state units (condition bits, etc.).
//!
//! Selection supplies ordinary SSA def-use edges. State placement is a separate
//! constraint: ordinary register allocation must never invent a copy or spill
//! for these values. This required lowering runs even without scheduling.
use crate::analysis::FunctionAnalysisCtx;
use crate::pipeline::{FunctionPass, FunctionSession, FunctionStage};
use crate::target::TargetInstructions;
use crate::{Error, Result};
use hashbrown::HashMap;
use veloc_lir::{
    BlockId, InstId, InstRef, MachineFunction, MachineOpcode, OperandRef, Placement, Reg,
};

/// Placement inferred from the target's operand categories. SSA identities and
/// def-use edges remain in the IR; this map only identifies their storage roots.
#[derive(Default)]
pub(crate) struct StateValues {
    locations: HashMap<Reg, Reg>,
}

impl StateValues {
    pub fn collect(f: &MachineFunction, target: &dyn TargetInstructions) -> Result<Self> {
        let mut values = Self::default();
        for block in f.blocks() {
            for id in f.block_insts(block) {
                let inst = f.inst(id);
                for c in crate::regalloc::constraints::constraints(inst, target) {
                    let Placement::State(location) = c.placement else {
                        continue;
                    };
                    let value = *c
                        .operand
                        .get(inst.inputs(), inst.results())
                        .ok_or_else(|| Error::codegen("missing state operand"))?;
                    if value.is_preg() {
                        if value != location {
                            return Err(Error::codegen("incorrect physical state operand"));
                        }
                    } else if values
                        .locations
                        .insert(value, location)
                        .is_some_and(|old| old != location)
                    {
                        return Err(Error::codegen(
                            "state value has incompatible hardware locations",
                        ));
                    }
                }
            }
        }
        // Every occurrence must respect the same storage contract. In particular,
        // ordinary copies and block parameters cannot silently copy hardware state.
        for block in f.blocks() {
            if f.block_params(block)
                .unwrap()
                .iter()
                .any(|v| values.location(*v).is_some())
            {
                return Err(Error::codegen(
                    "state block parameters require explicit edge lowering",
                ));
            }
            for id in f.block_insts(block) {
                let inst = f.inst(id);
                let constraints: Vec<_> =
                    crate::regalloc::constraints::constraints(inst, target).collect();
                for (operand, value) in inst
                    .inputs()
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (OperandRef::Input(i), v))
                    .chain(
                        inst.results()
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| (OperandRef::Result(i), v)),
                    )
                {
                    if let Some(location) = values.location(value) {
                        if !constraints.iter().any(|c| {
                            c.operand == operand && c.placement == Placement::State(location)
                        }) {
                            return Err(Error::codegen(
                                "state value used without its state placement contract",
                            ));
                        }
                    }
                }
                if inst.edge_ids().any(|edge| {
                    inst.edge(edge)
                        .args
                        .iter()
                        .any(|v| values.location(*v).is_some())
                }) {
                    return Err(Error::codegen(
                        "state edge arguments require explicit edge lowering",
                    ));
                }
            }
        }
        Ok(values)
    }

    pub fn location(&self, value: Reg) -> Option<Reg> {
        self.locations.get(&value).copied()
    }

    pub fn physical(&self, value: Reg) -> Reg {
        self.location(value).unwrap_or(value)
    }
}

type Available = HashMap<Reg, Reg>;
type Repairs = HashMap<InstId, Vec<InstId>>;

/// All writes invalidate previous contents, including unused results and
/// clobbers. Explicit state results then install their new SSA identities.
/// Reads and untouched roots preserve the available value.
fn transfer(inst: InstRef<'_>, values: &StateValues, available: &mut Available) {
    for reg in inst.register_access().writes() {
        available.remove(&values.physical(reg));
    }
    for &value in inst.results() {
        if let Some(location) = values.location(value) {
            available.insert(location, value);
        }
    }
}

fn apply_repairs(
    f: &MachineFunction,
    values: &StateValues,
    repairs: &Repairs,
    id: InstId,
    available: &mut Available,
) {
    for &producer in repairs.get(&id).into_iter().flatten() {
        transfer(f.inst(producer), values, available);
    }
}

/// Must-availability across the CFG. Absence means unknown, including function
/// entry. Equal incoming SSA identities can flow across any number of blocks.
fn availability(
    f: &MachineFunction,
    target: &dyn TargetInstructions,
    values: &StateValues,
    repairs: &Repairs,
) -> HashMap<BlockId, Available> {
    let mut analyses = FunctionAnalysisCtx::default();
    let cfg = analyses.cfg(f, target);
    let mut incoming = HashMap::<BlockId, Available>::new();
    let mut outgoing = HashMap::<BlockId, Available>::new();
    loop {
        let mut changed = false;
        for block in f.blocks() {
            let mut state = Available::new();
            if block != f.entry_block() {
                let preds = cfg.preds(block);
                if let Some(first) = preds.first() {
                    state = outgoing.get(first).cloned().unwrap_or_default();
                    state.retain(|unit, value| {
                        preds
                            .iter()
                            .skip(1)
                            .all(|p| outgoing.get(p).and_then(|s| s.get(unit)) == Some(value))
                    });
                }
            }
            incoming.insert(block, state.clone());
            for id in f.block_insts(block) {
                apply_repairs(f, values, repairs, id, &mut state);
                transfer(f.inst(id), values, &mut state);
            }
            if outgoing.get(&block) != Some(&state) {
                outgoing.insert(block, state);
                changed = true;
            }
        }
        if !changed {
            return incoming;
        }
    }
}

fn recipe(
    f: &MachineFunction,
    target: &dyn TargetInstructions,
    values: &StateValues,
    value: Reg,
) -> Result<InstId> {
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
        || inst.memory().is_some()
        || inst
            .inputs()
            .iter()
            .any(|v| v.is_preg() || values.location(*v).is_some())
    {
        return Err(Error::codegen(
            "overwritten state requires a safe rematerialization recipe",
        ));
    }
    Ok(id)
}

fn plan(
    f: &MachineFunction,
    target: &dyn TargetInstructions,
    values: &StateValues,
) -> Result<Repairs> {
    let mut repairs = Repairs::new();
    loop {
        let incoming = availability(f, target, values, &repairs);
        let mut changed = false;
        for block in f.blocks() {
            let mut available = incoming[&block].clone();
            for id in f.block_insts(block) {
                apply_repairs(f, values, &repairs, id, &mut available);
                let inst = f.inst(id);
                for &value in inst.inputs() {
                    let Some(location) = values.location(value) else {
                        continue;
                    };
                    if available.get(&location) == Some(&value) {
                        continue;
                    }
                    let producer = recipe(f, target, values, value)?;
                    let list = repairs.entry(id).or_default();
                    if list.contains(&producer) {
                        return Err(Error::codegen(
                            "simultaneous state inputs require incompatible hardware contents",
                        ));
                    }
                    list.push(producer);
                    transfer(f.inst(producer), values, &mut available);
                    changed = true;
                }
                // Recomputing one result may overwrite a different required bit.
                if inst.inputs().iter().any(|v| {
                    values
                        .location(*v)
                        .is_some_and(|p| available.get(&p) != Some(v))
                }) {
                    return Err(Error::codegen(
                        "state inputs cannot coexist; materialize predicates before selection",
                    ));
                }
                transfer(inst, values, &mut available);
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
        let values = StateValues::collect(f, cx.target)?;
        if values.locations.is_empty() {
            return Ok(());
        }
        let repairs = plan(f, cx.target, &values)?;
        let ids: Vec<_> = f.blocks().flat_map(|b| f.block_insts(b)).collect();
        let mut f = cx.edit();
        // Insert all recipes while their SSA definitions are still intact.
        for &id in &ids {
            for &producer in repairs.get(&id).into_iter().flatten() {
                let source = f.inst(producer);
                let opcode = source.opcode();
                let inputs = source.inputs().to_vec();
                let original_results = source.results().to_vec();
                let fields: Vec<_> = (0..source.fields().len())
                    .map(|i| source.fields().at(i))
                    .collect();
                let clobbers: Vec<_> = source.clobbers().collect();
                let mut results = Vec::new();
                for value in original_results {
                    if let Some(location) = values.location(value) {
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
                f.editor()
                    .before(id)
                    .writer()
                    .with_clobbers(clobbers)
                    .write(opcode, &results, &inputs, fields);
            }
        }
        for id in ids {
            let inst = f.inst(id);
            let inputs: Vec<_> = inst
                .inputs()
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(i, v)| values.location(v).map(|p| (i, p)))
                .collect();
            let results: Vec<_> = inst
                .results()
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(i, v)| values.location(v).map(|p| (i, p)))
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
