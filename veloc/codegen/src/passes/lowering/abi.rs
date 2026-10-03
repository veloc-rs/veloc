use crate::error::{Error, Result};
use crate::pipeline::{FunctionPass, FunctionSession, FunctionStage};
use crate::target::{AbiLocation, AbiPlan, CallConv, TargetMachine};
use smallvec::SmallVec;
use veloc_lir::MemFlags;
use veloc_lir::{GenericOpcode, InstId, MachineOpcode, Reg, StackObject, StackSlot, Type};
use veloc_lir::{InstBuild, InstRead, OperandConstraint, OperandRef};

pub struct AbiLoweringPass;

impl AbiLoweringPass {
    pub fn new() -> Self {
        Self
    }
}

// Reject transfer modes that this lowering does not implement yet.
fn plan_signature(
    target: &dyn TargetMachine,
    sig: &veloc_mir::Signature,
    params: &[Type],
) -> Result<AbiPlan> {
    if !params.starts_with(sig.params()) || (!sig.variadic && params.len() != sig.params().len()) {
        return Err(Error::codegen(
            "call arguments do not match the declared signature",
        ));
    }
    let desc = target.desc();
    let convention = CallConv::from(sig.call_conv);
    let plan = if sig.variadic {
        convention.plan_variadic(
            desc.arch,
            &desc.data_layout,
            params,
            sig.returns(),
            sig.params().len(),
        )?
    } else {
        convention.plan(desc.arch, &desc.data_layout, params, sig.returns())?
    };
    if plan
        .returns
        .iter()
        .any(|a| matches!(a.loc, AbiLocation::Stack { .. }))
    {
        return Err(Error::codegen("stack return lowering is not implemented"));
    }
    Ok(plan)
}

/// Prepare a stack slot and typed access. ABI slot size may exceed access width.
fn stack_access(
    insert: &mut veloc_lir::InstInserter<'_>,
    object: StackObject,
    size: u32,
    align: u32,
) -> (Reg, StackSlot, MemFlags) {
    let slot = insert.alloc_stack_object(object, size, align);
    let address = insert.alloc_vreg(Type::PTR);
    insert.stack_addr(address, slot);
    (address, slot, MemFlags::new().with_alignment(align))
}

fn return_constraints(
    plan: &AbiPlan,
    operand: fn(usize) -> OperandRef,
) -> impl Iterator<Item = OperandConstraint> + '_ {
    plan.returns
        .iter()
        .enumerate()
        .map(move |(index, assignment)| {
            let AbiLocation::Reg(reg) = assignment.loc else {
                unreachable!("checked register return")
            };
            OperandConstraint::fixed(operand(index), reg)
        })
}

fn lower_formal_arguments(mfunc: &mut veloc_lir::FuncEditor<'_>, plan: &AbiPlan) {
    let entry = mfunc.entry_block();
    assert_eq!(
        mfunc.params().len(),
        plan.args.len(),
        "ABI parameter count mismatch"
    );
    let params = mfunc.take_params();
    let mut bindings = Vec::new();
    for (dst, assignment) in params.into_iter().zip(&plan.args) {
        match assignment.loc {
            AbiLocation::Reg(reg) => {
                mfunc.append_param(dst);
                bindings.push(veloc_lir::EntryBinding {
                    value: dst,
                    location: reg.as_preg().expect("physical ABI location"),
                });
            }
            AbiLocation::Stack {
                offset,
                size,
                align,
            } => {
                let mut insert = mfunc.at_start(entry);
                let (address, _, flags) =
                    stack_access(&mut insert, StackObject::Incoming { offset }, size, align);
                insert.load(dst, address, 0, flags);
            }
        }
    }
    mfunc.set_entry_bindings(bindings);
}

/// Replace a checked value operation with a runtime call and establish its ABI
/// contract in the same tracked edit. The rule supplies the symbol; the
/// operation supplies the concrete signature and argument order.
pub(super) fn emit_libcall(
    target: &dyn TargetMachine,
    symbols: &mut veloc_lir::SymbolTable,
    mfunc: &mut veloc_lir::FuncEditor<'_>,
    id: InstId,
    symbol: &str,
) -> Result<()> {
    let inst = mfunc.inst(id);
    if inst.mem_flags().is_some() || inst.clobbers().next().is_some() {
        return Err(Error::codegen(
            "libcall replacement cannot discard memory facts or register clobbers",
        ));
    }
    let args = SmallVec::<[Reg; 4]>::from_slice(inst.inputs());
    let results = SmallVec::<[Reg; 2]>::from_slice(inst.results());
    let params: SmallVec<[veloc_mir::Type; 4]> =
        args.iter().map(|&reg| mfunc.vreg_data(reg).ty).collect();
    let returns: SmallVec<[veloc_mir::Type; 2]> =
        results.iter().map(|&reg| mfunc.vreg_data(reg).ty).collect();
    let sig = veloc_mir::Signature::new(params, returns, veloc_types::CallConv::SystemV);
    // Diagnose unsupported ABI representations before replacing the operation.
    let plan = plan_signature(target, &sig, sig.params())?;
    let callee = symbols.get_or_create_function(symbol, veloc_mir::Linkage::Import);
    mfunc.replace(id).call(
        &results,
        callee,
        &args,
        veloc_lir::CallInfo {
            sig,
            clobbers: Default::default(),
            frame: None,
            stack_args: Default::default(),
        },
    );
    apply_call_abi(mfunc, id, &plan);
    Ok(())
}

fn lower_call(
    target: &dyn TargetMachine,
    mfunc: &mut veloc_lir::FuncEditor<'_>,
    id: InstId,
) -> Result<()> {
    let info = mfunc.call_info(id);
    let args = match mfunc.inst(id).view() {
        veloc_lir::InstView::Call(call) => call.args,
        veloc_lir::InstView::CallIndirect(call) => call.args,
        _ => unreachable!("callsite lowering"),
    };
    let params: SmallVec<[Type; 8]> = args.iter().map(|&arg| mfunc.vreg_data(arg).ty).collect();
    let plan = plan_signature(target, &info.sig, &params)?;
    apply_call_abi(mfunc, id, &plan);
    Ok(())
}

fn apply_call_abi(mfunc: &mut veloc_lir::FuncEditor<'_>, id: InstId, plan: &AbiPlan) {
    let inst = mfunc.inst(id);
    let (results, args, callee) = match inst.view() {
        veloc_lir::InstView::Call(call) => (call.results, call.args, None),
        veloc_lir::InstView::CallIndirect(call) => (call.results, call.args, Some(call.callee)),
        _ => unreachable!("callsite lowering"),
    };
    assert_eq!(args.len(), plan.args.len(), "call argument count mismatch");
    assert_eq!(
        results.len(),
        plan.returns.len(),
        "call result count mismatch"
    );

    let mut logical_args = SmallVec::<[Reg; 8]>::from_slice(args);
    // Some ABIs transfer unnamed floats through integer registers. Keep the
    // language type intact until this boundary, then reinterpret its bits.
    for (src, assignment) in logical_args.iter_mut().zip(&plan.args) {
        if mfunc.vreg_data(*src).ty != assignment.ty {
            let converted = mfunc.alloc_vreg(assignment.ty);
            mfunc.before(id).bitcast(converted, *src);
            *src = converted;
        }
    }
    let frame = mfunc.alloc_call_frame(plan.stack);
    mfunc.before(id).call_frame_setup(frame);

    let mut inputs = SmallVec::<[Reg; 8]>::new();
    inputs.extend(callee);
    let mut constraints = Vec::new();
    let mut stack_args = SmallVec::new();
    {
        let mut insert = mfunc.before(id);
        for (&src, assignment) in logical_args.iter().zip(&plan.args) {
            match assignment.loc {
                AbiLocation::Reg(reg) => {
                    constraints.push(OperandConstraint::fixed(
                        OperandRef::Input(inputs.len()),
                        reg,
                    ));
                    inputs.push(src);
                }
                AbiLocation::Stack {
                    offset,
                    size,
                    align,
                } => {
                    let (address, slot, flags) = stack_access(
                        &mut insert,
                        StackObject::Outgoing { frame, offset },
                        size,
                        align,
                    );
                    insert.store(src, address, 0, flags);
                    stack_args.push(slot);
                }
            }
        }
    }
    constraints.extend(return_constraints(plan, OperandRef::Result));
    // Keep SSA definitions and uses; physical locations are requirements at this call.
    mfunc.set_call_abi(id, &inputs, frame, plan.abi.clobbers, stack_args);
    mfunc.set_inst_constraints(id, constraints);
    mfunc.after(id).call_frame_destroy(frame);
}

fn lower_return(mfunc: &mut veloc_lir::FuncEditor<'_>, id: InstId, plan: &AbiPlan) {
    let veloc_lir::InstView::Return(ret) = mfunc.inst(id).view() else {
        unreachable!("planned return")
    };
    assert_eq!(
        ret.values.len(),
        plan.returns.len(),
        "return value count mismatch"
    );
    mfunc.set_inst_constraints(id, return_constraints(plan, OperandRef::Input).collect());
}

impl FunctionPass for AbiLoweringPass {
    fn name(&self) -> &'static str {
        "abi-lowered"
    }

    fn input_stage(&self) -> FunctionStage {
        FunctionStage::Generic
    }
    fn run(&self, cx: &mut FunctionSession<'_>) -> crate::Result<()> {
        let target = cx.target;
        let plan = plan_signature(target, cx.signature, cx.signature.params())?;
        let mut mfunc = cx.edit();
        lower_formal_arguments(&mut mfunc.editor(), &plan);
        let mut cursor = veloc_lir::InstCursor::new(&mfunc);
        while let Some(id) = cursor.next(&mfunc) {
            match mfunc.inst(id).opcode() {
                MachineOpcode::Generic(GenericOpcode::Call | GenericOpcode::Callind) => {
                    lower_call(target, &mut mfunc.editor(), id)?;
                }
                MachineOpcode::Generic(GenericOpcode::Ret) => {
                    lower_return(&mut mfunc.editor(), id, &plan);
                }
                _ => {}
            }
        }
        Ok(())
    }
}
