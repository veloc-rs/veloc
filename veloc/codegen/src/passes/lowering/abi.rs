use crate::error::Result;
use crate::pipeline::{ChangeSet, FunctionPass, FunctionPassContext, PassEffect};
use crate::target::arch::{AbiAssignment, AbiLocation, CallConv, CallConvPlan, TargetMachine};
use alloc::vec::Vec;
use veloc_lir::{GenericOpcode, InstId, MachineFunction, MachineOpcode, Reg, StackSlot, Writable};
use veloc_lir::{InstBuild, InstRead};

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

fn single_part_assignment<'a>(
    assignment: &'a AbiAssignment,
    kind: &'static str,
) -> &'a crate::target::arch::AbiPart {
    match assignment.parts.as_slice() {
        [part] => part,
        _ => panic!("multi-part ABI {} lowering is not supported yet", kind),
    }
}

fn stack_slot_for_assignment(
    target: &dyn TargetMachine,
    mfunc: &mut MachineFunction,
    part: &crate::target::arch::AbiPart,
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
                crate::target::arch::AbiStackBase::IncomingArgs => {
                    base_reg.unwrap_or(stack_pointer)
                }
                crate::target::arch::AbiStackBase::OutgoingArgs => stack_pointer,
            };
            mfunc
                .editor()
                .alloc_stack_slot_with_base(base_reg, offset, size, align)
        }
        AbiLocation::Reg(_) => unreachable!("stack slot requested for register assignment"),
    }
}

fn build_load_from_assignment(
    target: &dyn TargetMachine,
    mfunc: &mut MachineFunction,
    assignment: &AbiAssignment,
    dst: Reg,
    kind: &'static str,
) -> InstId {
    let part = single_part_assignment(assignment, kind);
    match part.loc {
        AbiLocation::Reg(reg) => mfunc.editor().writer().copy(Writable(dst), reg),
        AbiLocation::Stack { .. } => {
            let slot = stack_slot_for_assignment(target, mfunc, part);
            mfunc.editor().writer().stack_load(Writable(dst), slot)
        }
    }
}

fn build_store_to_assignment(
    target: &dyn TargetMachine,
    mfunc: &mut MachineFunction,
    src: Reg,
    assignment: &AbiAssignment,
    kind: &'static str,
) -> InstId {
    let part = single_part_assignment(assignment, kind);
    match part.loc {
        AbiLocation::Reg(reg) => mfunc.editor().writer().copy(Writable(reg), src),
        AbiLocation::Stack { .. } => {
            let slot = stack_slot_for_assignment(target, mfunc, part);
            mfunc.editor().writer().stack_store(src, slot)
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
            mfunc.editor().replace_inst(id, replacement);
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

    let args: Vec<_> = args.to_vec();
    let defs: Vec<_> = results.iter().copied().collect();
    let returns: Vec<_> = plan
        .returns
        .iter()
        .flat_map(|a| &a.parts)
        .filter_map(|p| {
            if let AbiLocation::Reg(reg) = p.loc {
                Some(reg)
            } else {
                None
            }
        })
        .collect();
    mfunc.editor().set_inst_results(id, &returns);
    for (src, assignment) in args.into_iter().zip(plan.args.iter()) {
        let inst = build_store_to_assignment(target, mfunc, src, assignment, "call argument");
        mfunc.editor().insert_before(id, inst);
    }

    mfunc.stack_frame.arg_size = mfunc.stack_frame.arg_size.max(plan.stack_arg_bytes);
    let mut after = id;

    for (dst, assignment) in defs.into_iter().zip(plan.returns.iter()) {
        let inst = build_load_from_assignment(target, mfunc, assignment, dst, "call return");
        mfunc.editor().insert_after(after, inst);
        after = inst;
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
        pre.push(build_store_to_assignment(
            target,
            mfunc,
            src,
            assignment,
            "return value",
        ));
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
        mfunc.stack_frame.arg_size = 0;
        lower_formal_arguments(ctx.target, mfunc, &plan);

        let ids: Vec<_> = mfunc.blocks().flat_map(|b| mfunc.block_insts(b)).collect();
        for inst_id in ids {
            let inst = mfunc.inst(inst_id);
            match inst.opcode() {
                MachineOpcode::Generic(GenericOpcode::Call | GenericOpcode::Callind) => {
                    let call_plan = plan_callsite(ctx.target, &mfunc.call_info(inst_id).sig)?;
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
                    let regs: Vec<_> = plan
                        .returns
                        .iter()
                        .flat_map(|assignment| {
                            assignment.parts.iter().filter_map(|part| match part.loc {
                                AbiLocation::Reg(reg) => Some(reg),
                                _ => None,
                            })
                        })
                        .collect();
                    let replacement = mfunc.editor().writer().ret(&regs);
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
