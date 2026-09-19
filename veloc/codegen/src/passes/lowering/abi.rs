use crate::analysis::{ChangeSet, PassEffect};
use crate::error::{Error, Result};
use crate::pipeline::{FunctionPass, FunctionPassContext};
use crate::target::{AbiAssignment, AbiLocation, CallConv, CallConvPlan, TargetMachine};
use alloc::vec::Vec;
use smallvec::{SmallVec, smallvec};
use veloc_lir::{GenericOpcode, InstId, MachineFunction, MachineOpcode, Reg, StackSlot, Writable};
use veloc_lir::{InstBuild, InstRead};
use veloc_lir::{MemoryAccess, MemoryKind};

pub struct AbiLoweringPass;

impl AbiLoweringPass {
    pub fn new() -> Self {
        Self
    }
}

fn plan_signature(target: &dyn TargetMachine, sig: &veloc_mir::Signature) -> Result<CallConvPlan> {
    CallConv::from(sig.call_conv).plan_signature(target.desc().arch, sig)
}

fn plan_callsite(target: &dyn TargetMachine, sig: &veloc_mir::Signature) -> Result<CallConvPlan> {
    CallConv::from(sig.call_conv).plan_callsite(target.desc().arch, sig.params(), sig.returns())
}

// Plan all boundaries before editing: unsupported ABIs must not leave a
// partially lowered function. Validation of input IR remains the validator's job.
fn check_plan(plan: &CallConvPlan) -> Result<()> {
    for assignment in plan.args.iter().chain(&plan.returns) {
        if assignment.parts.len() != 1 {
            return Err(Error::codegen(
                "multi-part ABI lowering is not supported yet",
            ));
        }
    }
    Ok(())
}

fn registers(assignments: &[AbiAssignment]) -> Vec<Reg> {
    assignments
        .iter()
        .flat_map(|a| &a.parts)
        .filter_map(|p| match p.loc {
            AbiLocation::Reg(reg) => Some(reg),
            AbiLocation::Stack { .. } => None,
        })
        .collect()
}

fn call_effects(target: &dyn TargetMachine, plan: &CallConvPlan) -> veloc_lir::RegEffects {
    let file = &target.desc().registers;
    let preserved = plan.call_conv.preserved_regs(plan.arch);
    let mut defs: Vec<_> = file
        .regs
        .iter()
        .map(|info| info.preg)
        .filter(|reg| {
            !preserved.contains(reg)
                && *reg != file.special_regs.stack_pointer
                && Some(*reg) != file.special_regs.frame_pointer
        })
        .collect();
    defs.sort_unstable();
    defs.dedup();
    veloc_lir::RegEffects {
        uses: registers(&plan.args),
        defs,
    }
}

fn single_part_assignment<'a>(
    assignment: &'a AbiAssignment,
    kind: &'static str,
) -> &'a crate::target::AbiPart {
    match assignment.parts.as_slice() {
        [part] => part,
        _ => panic!("multi-part ABI {} lowering is not supported yet", kind),
    }
}

fn stack_slot_for_assignment(
    target: &dyn TargetMachine,
    mfunc: &mut MachineFunction,
    part: &crate::target::AbiPart,
) -> StackSlot {
    let stack_pointer = target.desc().registers.special_regs.stack_pointer;
    match part.loc {
        AbiLocation::Stack {
            base,
            base_reg,
            offset,
            size,
            align,
            ..
        } => {
            let base_reg = match base {
                crate::target::AbiStackBase::IncomingArgs => base_reg.unwrap_or(stack_pointer),
                crate::target::AbiStackBase::OutgoingArgs => stack_pointer,
            };
            mfunc
                .editor()
                .alloc_stack_slot_with_base(base_reg, offset, size, align)
        }
        AbiLocation::Reg(_) => unreachable!("stack slot requested for register assignment"),
    }
}

fn stack_access(
    target: &dyn TargetMachine,
    ty: veloc_lir::Type,
    align: u32,
    kind: MemoryKind,
) -> MemoryAccess {
    let bytes = target
        .desc()
        .data_layout
        .layout_of(ty)
        .and_then(|layout| layout.store_size.fixed_bytes())
        .expect("ABI stack value requires a fixed storage layout");
    let mut access = MemoryAccess::new(kind, bytes);
    access.alignment = align;
    access.may_trap = false;
    access
}

fn build_load_from_assignment(
    target: &dyn TargetMachine,
    mfunc: &mut MachineFunction,
    assignment: &AbiAssignment,
    dst: Reg,
    kind: &'static str,
) -> SmallVec<[InstId; 2]> {
    let part = single_part_assignment(assignment, kind);
    match part.loc {
        AbiLocation::Reg(reg) => smallvec![mfunc.editor().writer().copy(Writable(dst), reg)],
        AbiLocation::Stack { align, .. } => {
            let slot = stack_slot_for_assignment(target, mfunc, part);
            let address = mfunc.editor().alloc_vreg(veloc_lir::Type::PTR);
            let addr = mfunc.editor().writer().stack_addr(Writable(address), slot);
            let access = stack_access(target, part.ty, align, MemoryKind::Read);
            let load = mfunc
                .editor()
                .writer()
                .with_memory(access)
                .load(Writable(dst), address, 0);
            smallvec![addr, load]
        }
    }
}

fn build_store_to_assignment(
    target: &dyn TargetMachine,
    mfunc: &mut MachineFunction,
    src: Reg,
    assignment: &AbiAssignment,
    kind: &'static str,
) -> (SmallVec<[InstId; 2]>, Option<StackSlot>) {
    let part = single_part_assignment(assignment, kind);
    match part.loc {
        AbiLocation::Reg(reg) => (
            smallvec![mfunc.editor().writer().copy(Writable(reg), src)],
            None,
        ),
        AbiLocation::Stack { align, .. } => {
            let slot = stack_slot_for_assignment(target, mfunc, part);
            let address = mfunc.editor().alloc_vreg(veloc_lir::Type::PTR);
            let addr = mfunc.editor().writer().stack_addr(Writable(address), slot);
            let access = stack_access(target, part.ty, align, MemoryKind::Write);
            let store = mfunc
                .editor()
                .writer()
                .with_memory(access)
                .store(src, address, 0);
            (smallvec![addr, store], Some(slot))
        }
    }
}

fn lower_formal_arguments(
    target: &dyn TargetMachine,
    mfunc: &mut MachineFunction,
    plan: &CallConvPlan,
) {
    if mfunc.num_blocks() == 0 {
        return;
    }

    let entry = mfunc.entry_block().unwrap();
    let ids: Vec<_> = mfunc.block_insts(entry).collect();
    for id in ids {
        let inst = mfunc.inst(id);
        if !inst.is_generic() {
            continue;
        }
        if let veloc_lir::InstView::Arg(decoded) = inst.view() {
            let assignment = plan
                .args
                .get(usize::try_from(decoded.index).expect("negative argument index"))
                .expect("missing ABI argument assignment");
            let dst = decoded.dst;
            let replacement =
                build_load_from_assignment(target, mfunc, assignment, dst, "argument");
            mfunc.editor().replace_with(id, &replacement);
        }
    }
}

fn lower_callsite(
    target: &dyn TargetMachine,
    mfunc: &mut MachineFunction,
    id: InstId,
    plan: &CallConvPlan,
) {
    let inst = mfunc.inst(id);
    let (results, args) = match inst.view() {
        veloc_lir::InstView::Call(call) => (call.results, call.args),
        veloc_lir::InstView::CallIndirect(call) => (call.results, call.args),
        _ => unreachable!("callsite lowering"),
    };
    if args.len() != plan.args.len() {
        panic!(
            "call argument count mismatch: LIR has {}, ABI plan has {}",
            args.len(),
            plan.args.len()
        );
    }
    if results.len() != plan.returns.len() {
        panic!(
            "call result count mismatch: LIR has {}, ABI plan has {}",
            results.len(),
            plan.returns.len()
        );
    }

    let args: SmallVec<[Reg; 8]> = args.iter().copied().collect();
    let defs: SmallVec<[Reg; 4]> = results.iter().copied().collect();
    let returns = registers(&plan.returns);
    mfunc.editor().set_inst_results(id, &returns);
    let mut info = mfunc.call_info(id).clone();
    info.stack_args.clear();
    for (src, assignment) in args.into_iter().zip(plan.args.iter()) {
        let (insts, slot) =
            build_store_to_assignment(target, mfunc, src, assignment, "call argument");
        info.stack_args.extend(slot);
        for inst in insts {
            mfunc.editor().insert_before(id, inst);
        }
    }

    mfunc
        .editor()
        .set_inst_extra(id, veloc_lir::InstExtra::Call(info));
    // The call now uses ABI locations, not the original logical argument list.
    // Keep the indirect callee as its own explicit input.
    let callee = match mfunc.inst(id).view() {
        veloc_lir::InstView::CallIndirect(call) => Some(call.callee),
        _ => None,
    };
    let mut inputs = SmallVec::<[Reg; 8]>::new();
    inputs.extend(callee);
    inputs.extend(registers(&plan.args));
    mfunc.editor().set_inst_inputs(id, &inputs);
    let mut effects = call_effects(target, plan);
    if let Some(existing) = mfunc.inst(id).effects() {
        effects.uses.extend_from_slice(existing.uses);
        effects.defs.extend_from_slice(existing.defs);
    }
    effects.uses.sort_unstable();
    effects.uses.dedup();
    effects.defs.sort_unstable();
    effects.defs.dedup();
    mfunc.editor().set_inst_effects(id, effects);

    mfunc.stack_frame.arg_size = mfunc.stack_frame.arg_size.max(plan.stack_arg_bytes);
    let mut after = id;

    for (dst, assignment) in defs.into_iter().zip(plan.returns.iter()) {
        for inst in build_load_from_assignment(target, mfunc, assignment, dst, "call return") {
            mfunc.editor().insert_after(after, inst);
            after = inst;
        }
    }
}

fn lower_return(
    target: &dyn TargetMachine,
    mfunc: &mut MachineFunction,
    sig: &veloc_mir::Signature,
    plan: &CallConvPlan,
    values: &[Reg],
) -> Vec<InstId> {
    if values.len() != plan.returns.len() {
        panic!(
            "return value count mismatch: LIR has {}, ABI plan has {}",
            values.len(),
            plan.returns.len()
        );
    }
    if values.len() != sig.returns().len() {
        panic!(
            "return value count mismatch: LIR has {}, signature expects {}",
            values.len(),
            sig.returns().len()
        );
    }

    let mut pre = Vec::with_capacity(values.len());
    for (&src, assignment) in values.iter().zip(plan.returns.iter()) {
        pre.extend(build_store_to_assignment(target, mfunc, src, assignment, "return value").0);
    }

    pre
}

impl FunctionPass for AbiLoweringPass {
    fn name(&self) -> &'static str {
        "abi-lowered"
    }

    fn run(
        &self,
        mfunc: &mut MachineFunction,
        ctx: &mut FunctionPassContext<'_>,
    ) -> Result<PassEffect> {
        let plan = plan_signature(ctx.target, ctx.func_sig)?;
        check_plan(&plan)?;
        let ids: Vec<_> = mfunc.blocks().flat_map(|b| mfunc.block_insts(b)).collect();
        let calls = ids
            .iter()
            .copied()
            .filter(|&id| {
                matches!(
                    mfunc.inst(id).opcode(),
                    MachineOpcode::Generic(GenericOpcode::Call | GenericOpcode::Callind)
                )
            })
            .map(|id| {
                let plan = plan_callsite(ctx.target, &mfunc.call_info(id).sig)?;
                check_plan(&plan)?;
                Ok((id, plan))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut calls = calls.into_iter();
        let return_regs = registers(&plan.returns);
        mfunc.stack_frame.arg_size = 0;
        lower_formal_arguments(ctx.target, mfunc, &plan);

        for inst_id in ids {
            let inst = mfunc.inst(inst_id);
            match inst.opcode() {
                MachineOpcode::Generic(GenericOpcode::Call | GenericOpcode::Callind) => {
                    let (call_id, call_plan) = calls.next().expect("planned callsite");
                    debug_assert_eq!(call_id, inst_id);
                    lower_callsite(ctx.target, mfunc, inst_id, &call_plan);
                }
                MachineOpcode::Generic(GenericOpcode::Ret) => {
                    let veloc_lir::InstView::Return(ret) = inst.view() else {
                        unreachable!()
                    };
                    let values = ret.values.to_vec();
                    let pre = lower_return(ctx.target, mfunc, ctx.func_sig, &plan, &values);
                    for inst in pre {
                        mfunc.editor().insert_before(inst_id, inst);
                    }
                    let replacement = mfunc.editor().writer().ret(&return_regs);
                    mfunc.editor().replace_inst(inst_id, replacement);
                }
                _ => {}
            }
        }

        Ok(PassEffect::new(
            ChangeSet::INST_SEMANTICS | ChangeSet::PHYSICAL_REGS,
        ))
    }
}
