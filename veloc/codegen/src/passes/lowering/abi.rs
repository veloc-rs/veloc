use crate::analysis::{ChangeSet, PassEffect};
use crate::error::{Error, Result};
use crate::pipeline::{FunctionPass, FunctionPassContext};
use crate::target::{AbiLocation, AbiPlan, CallConv, TargetMachine};
use smallvec::SmallVec;
use veloc_lir::{
    GenericOpcode, InstId, MachineFunction, MachineOpcode, Reg, StackObject, StackSlot, Type,
};
use veloc_lir::{InstBuild, InstRead, OperandConstraint, OperandRef};
use veloc_lir::{MemoryAccess, MemoryKind};

pub struct AbiLoweringPass;

impl AbiLoweringPass {
    pub fn new() -> Self {
        Self
    }
}

// Reject transfer modes that this lowering does not implement yet.
fn plan_signature(target: &dyn TargetMachine, sig: &veloc_mir::Signature) -> Result<AbiPlan> {
    let desc = target.desc();
    let plan = CallConv::from(sig.call_conv).plan(
        desc.arch,
        &desc.data_layout,
        sig.params(),
        sig.returns(),
    )?;
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
    target: &dyn TargetMachine,
    insert: &mut veloc_lir::InstInserter<'_>,
    object: StackObject,
    size: u32,
    align: u32,
    ty: Type,
    kind: MemoryKind,
) -> (Reg, StackSlot, MemoryAccess) {
    let slot = insert.alloc_stack_object(object, size, align);
    let address = insert.alloc_vreg(Type::PTR);
    insert.stack_addr(address, slot);
    let bytes = target
        .desc()
        .data_layout
        .layout_of(ty)
        .and_then(|layout| layout.store_size.fixed_bytes())
        .expect("checked ABI storage layout");
    let mut access = MemoryAccess::new(kind, bytes);
    access.alignment = align;
    access.may_trap = false;
    (address, slot, access)
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

fn lower_formal_arguments(
    target: &dyn TargetMachine,
    mfunc: &mut veloc_lir::FuncEditor<'_>,
    plan: &AbiPlan,
) {
    let entry = mfunc.entry_block();
    assert_eq!(
        mfunc.params().len(),
        plan.args.len(),
        "ABI parameter count mismatch"
    );
    let params = mfunc.take_params();
    let mut locations = Vec::new();
    for (dst, assignment) in params.into_iter().zip(&plan.args) {
        match assignment.loc {
            AbiLocation::Reg(reg) => {
                mfunc.append_param(dst);
                locations.push(reg.as_preg().expect("physical ABI location"));
            }
            AbiLocation::Stack {
                offset,
                size,
                align,
            } => {
                let mut insert = mfunc.at_start(entry);
                let (address, _, access) = stack_access(
                    target,
                    &mut insert,
                    StackObject::Incoming { offset },
                    size,
                    align,
                    assignment.ty,
                    MemoryKind::Read,
                );
                insert.with_memory(access).load(dst, address, 0);
            }
        }
    }
    mfunc.set_param_locations(locations);
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
    if inst.memory().is_some()
        || !inst.implicit_uses().is_empty()
        || !inst.implicit_defs().is_empty()
    {
        return Err(Error::codegen(
            "libcall replacement cannot discard memory facts or implicit register effects",
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
    let plan = plan_signature(target, &sig)?;
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
    apply_call_abi(target, mfunc, id, &plan);
    Ok(())
}

fn lower_call(
    target: &dyn TargetMachine,
    mfunc: &mut veloc_lir::FuncEditor<'_>,
    id: InstId,
) -> Result<()> {
    let plan = plan_signature(target, &mfunc.call_info(id).sig)?;
    apply_call_abi(target, mfunc, id, &plan);
    Ok(())
}

fn apply_call_abi(
    target: &dyn TargetMachine,
    mfunc: &mut veloc_lir::FuncEditor<'_>,
    id: InstId,
    plan: &AbiPlan,
) {
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

    let logical_args = SmallVec::<[Reg; 8]>::from_slice(args);
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
                    let (address, slot, access) = stack_access(
                        target,
                        &mut insert,
                        StackObject::Outgoing { frame, offset },
                        size,
                        align,
                        assignment.ty,
                        MemoryKind::Write,
                    );
                    insert.with_memory(access).store(src, address, 0);
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

    fn run(
        &self,
        mfunc: &mut MachineFunction,
        ctx: &mut FunctionPassContext<'_>,
    ) -> Result<PassEffect> {
        let plan = plan_signature(ctx.target, ctx.func_sig)?;

        lower_formal_arguments(ctx.target, &mut mfunc.editor(), &plan);
        // The cursor saves the next original instruction before each rewrite.
        // Planning errors abort compilation without rolling back earlier edits.
        let mut cursor = veloc_lir::InstCursor::new(mfunc);
        while let Some(id) = cursor.next(mfunc) {
            match mfunc.inst(id).opcode() {
                MachineOpcode::Generic(GenericOpcode::Call | GenericOpcode::Callind) => {
                    lower_call(ctx.target, &mut mfunc.editor(), id)?;
                }
                MachineOpcode::Generic(GenericOpcode::Ret) => {
                    lower_return(&mut mfunc.editor(), id, &plan);
                }
                _ => {}
            }
        }

        Ok(PassEffect::new(
            ChangeSet::INST_SEMANTICS | ChangeSet::PHYSICAL_REGS | ChangeSet::STACK_FRAME,
        ))
    }
}
