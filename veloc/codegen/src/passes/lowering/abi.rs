use crate::error::Result;
use crate::mir::{
    GenericOpcode, MachineFunction, MachineInst, MachineOpcode, Reg, StackSlot, Writable,
};
use crate::pipeline::stages::LegalizedMir;
use crate::pipeline::{ChangeSet, FunctionPassContext, PassEffect, StageTransformPass};
use crate::target::arch::{AbiAssignment, AbiLocation, CallConv, CallConvPlan, TargetMachine};
use alloc::vec::Vec;

pub struct AbiLoweringPass;

impl AbiLoweringPass {
    pub fn new() -> Self {
        Self
    }
}

fn plan_signature(target: &dyn TargetMachine, sig: &veloc_ir::Signature) -> Result<CallConvPlan> {
    CallConv::from(sig.call_conv).plan_signature(target.desc().arch, sig)
}

fn plan_callsite(target: &dyn TargetMachine, sig: &veloc_ir::Signature) -> Result<CallConvPlan> {
    CallConv::from(sig.call_conv).plan_callsite(target.desc().arch, &sig.params, &sig.returns)
}

fn reg_copy_inst(dst: Reg, src: Reg) -> MachineInst {
    MachineInst::build_copy(Writable(dst), src)
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

fn stack_slot_for_assignment<S>(
    target: &dyn TargetMachine,
    mfunc: &mut MachineFunction<S>,
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
            mfunc.alloc_stack_slot_with_base(base_reg, offset, size, align)
        }
        AbiLocation::Reg(_) => unreachable!("stack slot requested for register assignment"),
    }
}

fn build_load_from_assignment<S>(
    target: &dyn TargetMachine,
    mfunc: &mut MachineFunction<S>,
    assignment: &AbiAssignment,
    dst: Reg,
    kind: &'static str,
) -> MachineInst {
    let part = single_part_assignment(assignment, kind);
    match part.loc {
        AbiLocation::Reg(reg) => reg_copy_inst(dst, reg),
        AbiLocation::Stack { .. } => {
            let slot = stack_slot_for_assignment(target, mfunc, part);
            MachineInst::build_stack_load(Writable(dst), slot)
        }
    }
}

fn build_store_to_assignment<S>(
    target: &dyn TargetMachine,
    mfunc: &mut MachineFunction<S>,
    src: Reg,
    assignment: &AbiAssignment,
    kind: &'static str,
) -> MachineInst {
    let part = single_part_assignment(assignment, kind);
    match part.loc {
        AbiLocation::Reg(reg) => reg_copy_inst(reg, src),
        AbiLocation::Stack { .. } => {
            let slot = stack_slot_for_assignment(target, mfunc, part);
            MachineInst::build_stack_store(src, slot)
        }
    }
}

fn lower_formal_arguments(
    target: &dyn TargetMachine,
    mfunc: &mut MachineFunction<LegalizedMir>,
    plan: &CallConvPlan,
) {
    if mfunc.blocks.is_empty() {
        return;
    }

    let func_name = mfunc.name.clone();
    mfunc
        .rewrite_block::<(), _>(0, |cursor| {
            let inst = cursor.current_inst_clone();
            if inst.generic_opcode() == Some(GenericOpcode::G_ARG) {
                let decoded = inst.as_arg().unwrap_or_else(|err| {
                    panic!("invalid G_ARG while lowering `{}`: {}", func_name, err);
                });
                let assignment = match plan.args.get(decoded.index) {
                    Some(assignment) => assignment,
                    None => panic!(
                        "missing ABI assignment for argument {} in {}",
                        decoded.index, func_name
                    ),
                };
                let inst = build_load_from_assignment(
                    target,
                    cursor.mfunc_mut(),
                    assignment,
                    decoded.dst,
                    "argument",
                );
                cursor.replace_current(inst);
            } else {
                cursor.keep_current();
            }
            Ok(())
        })
        .unwrap_or_else(|_: ()| panic!("ABI argument lowering failed for `{}`", func_name));
}

fn lower_callsite<S>(
    target: &dyn TargetMachine,
    cursor: &mut crate::mir::BlockRewriteCursor<'_, S>,
    plan: &CallConvPlan,
    inst_id: crate::mir::InstId,
) {
    let shape = {
        let call = cursor.mfunc().as_call(inst_id);
        call.shape
    };
    if shape.args.len() != plan.args.len() {
        panic!(
            "call argument count mismatch: MIR has {}, ABI plan has {}",
            shape.args.len(),
            plan.args.len()
        );
    }
    if shape.defs.len() != plan.returns.len() {
        panic!(
            "call result count mismatch: MIR has {}, ABI plan has {}",
            shape.defs.len(),
            plan.returns.len()
        );
    }

    for (src, assignment) in shape.args.iter().copied().zip(plan.args.iter()) {
        let inst =
            build_store_to_assignment(target, cursor.mfunc_mut(), src, assignment, "call argument");
        cursor.emit_before(inst);
    }

    for (dst, assignment) in shape.defs.iter().copied().zip(plan.returns.iter()) {
        let inst =
            build_load_from_assignment(target, cursor.mfunc_mut(), assignment, dst, "call return");
        cursor.emit_before(inst);
    }

    cursor.emit_existing_before(inst_id);
    cursor.remove_current();
}

fn lower_return<S>(
    target: &dyn TargetMachine,
    mfunc: &mut MachineFunction<S>,
    sig: &veloc_ir::Signature,
    plan: &CallConvPlan,
    inst_id: crate::mir::InstId,
) -> Vec<MachineInst> {
    let values = mfunc.dfg[inst_id]
        .as_ret()
        .unwrap_or_else(|err| panic!("invalid G_RET during ABI lowering: {}", err))
        .values;
    if values.len() != plan.returns.len() {
        panic!(
            "return value count mismatch: MIR has {}, ABI plan has {}",
            values.len(),
            plan.returns.len()
        );
    }
    if values.len() != sig.returns.len() {
        panic!(
            "return value count mismatch: MIR has {}, signature expects {}",
            values.len(),
            sig.returns.len()
        );
    }

    let mut pre = Vec::with_capacity(values.len());
    for (src, assignment) in values.iter().copied().zip(plan.returns.iter()) {
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

impl StageTransformPass<LegalizedMir, LegalizedMir> for AbiLoweringPass {
    fn name(&self) -> &'static str {
        "abi-lowered"
    }

    fn run(
        &self,
        mut mfunc: MachineFunction<LegalizedMir>,
        ctx: &mut FunctionPassContext<'_, LegalizedMir>,
    ) -> Result<(MachineFunction<LegalizedMir>, PassEffect)> {
        let plan = plan_signature(ctx.target, ctx.func_sig)?;
        mfunc.stack_frame.arg_size = plan.stack_arg_bytes;
        lower_formal_arguments(ctx.target, &mut mfunc, &plan);

        let func_name = mfunc.name.clone();
        let num_blocks = mfunc.num_blocks();
        for block_idx in 0..num_blocks {
            mfunc
                .rewrite_block::<(), _>(block_idx, |cursor| {
                    let inst = cursor.current_inst_clone();
                    let inst_id = cursor.current_inst_id();

                    match inst.opcode {
                        MachineOpcode::Generic(GenericOpcode::G_CALL)
                        | MachineOpcode::Generic(GenericOpcode::G_CALLIND) => {
                            let call_plan = {
                                let call = cursor.mfunc().as_call(inst_id);
                                plan_callsite(ctx.target, &call.info.sig).unwrap_or_else(|err| {
                                    panic!(
                                    "failed to plan callsite for `{:?}` while lowering `{}`: {}",
                                    call.info.sig, func_name, err
                                );
                                })
                            };
                            lower_callsite(ctx.target, cursor, &call_plan, inst_id);
                        }
                        MachineOpcode::Generic(GenericOpcode::G_RET) => {
                            let ret_plan = &plan;
                            let pre = lower_return(
                                ctx.target,
                                cursor.mfunc_mut(),
                                ctx.func_sig,
                                ret_plan,
                                inst_id,
                            );
                            for inst in pre {
                                cursor.emit_before(inst);
                            }
                            cursor.emit_existing_before(inst_id);
                            cursor.remove_current();
                        }
                        _ => cursor.keep_current(),
                    }
                    Ok(())
                })
                .unwrap_or_else(|_: ()| {
                    panic!(
                        "ABI lowering failed while rewriting block {} in `{}`",
                        block_idx, func_name
                    )
                });
        }

        Ok((
            mfunc.into_stage(),
            PassEffect::new(ChangeSet::INST_SEMANTICS | ChangeSet::PHYSICAL_REGS),
        ))
    }
}
