use crate::error::Result;
use crate::pipeline::{ChangeSet, FunctionPassContext, PassEffect, StageTransformPass};
use crate::target::arch::{AbiAssignment, AbiLocation, CallConv, CallConvPlan, TargetMachine};
use alloc::vec::Vec;
use veloc_lir::stages::LegalizedLir;
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
) -> InstId {
    let part = single_part_assignment(assignment, kind);
    match part.loc {
        AbiLocation::Reg(reg) => mfunc.writer().copy(Writable(dst), reg),
        AbiLocation::Stack { .. } => {
            let slot = stack_slot_for_assignment(target, mfunc, part);
            mfunc.writer().stack_load(Writable(dst), slot)
        }
    }
}

fn build_store_to_assignment<S>(
    target: &dyn TargetMachine,
    mfunc: &mut MachineFunction<S>,
    src: Reg,
    assignment: &AbiAssignment,
    kind: &'static str,
) -> InstId {
    let part = single_part_assignment(assignment, kind);
    match part.loc {
        AbiLocation::Reg(reg) => mfunc.writer().copy(Writable(reg), src),
        AbiLocation::Stack { .. } => {
            let slot = stack_slot_for_assignment(target, mfunc, part);
            mfunc.writer().stack_store(src, slot)
        }
    }
}

fn lower_formal_arguments(
    target: &dyn TargetMachine,
    mfunc: &mut MachineFunction<LegalizedLir>,
    plan: &CallConvPlan,
) {
    if mfunc.blocks.is_empty() {
        return;
    }

    let func_name = mfunc.name.clone();
    mfunc
        .rewrite_block::<(), _>(0, |cursor| {
            let inst = cursor.current_inst();
            // Legalization may already have introduced target instructions.
            if !inst.is_generic() {
                cursor.keep_current();
                return Ok(());
            }
            if let veloc_lir::InstView::Arg(decoded) = inst.view() {
                let assignment = match plan
                    .args
                    .get(usize::try_from(decoded.index).expect("negative argument index"))
                {
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
    cursor: &mut veloc_lir::BlockRewriteCursor<'_, S>,
    plan: &CallConvPlan,
) {
    let inst = cursor.current_inst();
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
    let id = cursor.current_inst_id();
    cursor.mfunc_mut().set_inst_results(id, &returns);
    for (src, assignment) in args.into_iter().zip(plan.args.iter()) {
        let inst =
            build_store_to_assignment(target, cursor.mfunc_mut(), src, assignment, "call argument");
        cursor.emit(inst);
    }

    cursor.keep_current();
    cursor.mfunc_mut().stack_frame.arg_size = cursor
        .mfunc()
        .stack_frame
        .arg_size
        .max(plan.stack_arg_bytes);

    for (dst, assignment) in defs.into_iter().zip(plan.returns.iter()) {
        let inst =
            build_load_from_assignment(target, cursor.mfunc_mut(), assignment, dst, "call return");
        cursor.emit(inst);
    }
}

fn lower_return<S>(
    target: &dyn TargetMachine,
    mfunc: &mut MachineFunction<S>,
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

impl StageTransformPass<LegalizedLir, LegalizedLir> for AbiLoweringPass {
    fn name(&self) -> &'static str {
        "abi-lowered"
    }

    fn run(
        &self,
        mut mfunc: MachineFunction<LegalizedLir>,
        ctx: &mut FunctionPassContext<'_, LegalizedLir>,
    ) -> Result<(MachineFunction<LegalizedLir>, PassEffect)> {
        let plan = plan_signature(ctx.target, ctx.func_sig)?;
        mfunc.stack_frame.arg_size = 0;
        lower_formal_arguments(ctx.target, &mut mfunc, &plan);

        let func_name = mfunc.name.clone();
        let num_blocks = mfunc.num_blocks();
        for block_idx in 0..num_blocks {
            mfunc
                .rewrite_block::<(), _>(block_idx, |cursor| {
                    let inst = cursor.current_inst();
                    let inst_id = cursor.current_inst_id();

                    match inst.opcode() {
                        MachineOpcode::Generic(GenericOpcode::Call)
                        | MachineOpcode::Generic(GenericOpcode::Callind) => {
                            let call_plan = {
                                let call = cursor.mfunc().call_info(inst_id);
                                plan_callsite(ctx.target, &call.sig).unwrap_or_else(|err| {
                                    panic!(
                                    "failed to plan callsite for `{:?}` while lowering `{}`: {}",
                                    call.sig, func_name, err
                                );
                                })
                            };
                            lower_callsite(ctx.target, cursor, &call_plan);
                        }
                        MachineOpcode::Generic(GenericOpcode::Ret) => {
                            let veloc_lir::InstView::Return(ret) = inst.view() else {
                                unreachable!()
                            };
                            let values: Vec<_> = ret.values.to_vec();
                            let ret_plan = &plan;
                            let pre = lower_return(
                                ctx.target,
                                cursor.mfunc_mut(),
                                ctx.func_sig,
                                ret_plan,
                                &values,
                            );
                            for inst in pre {
                                cursor.emit(inst);
                            }
                            let regs: Vec<_> = ret_plan
                                .returns
                                .iter()
                                .flat_map(|assignment| {
                                    assignment.parts.iter().filter_map(|part| match part.loc {
                                        AbiLocation::Reg(reg) => Some(reg),
                                        _ => None,
                                    })
                                })
                                .collect();
                            let id = cursor.mfunc_mut().writer().ret(&regs);
                            cursor.replace_current(id);
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
