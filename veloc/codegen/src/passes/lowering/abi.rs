use crate::analysis::{ChangeSet, PassEffect};
use crate::error::{Error, Result};
use crate::pipeline::{FunctionPass, FunctionPassContext};
use crate::target::{AbiAssignment, AbiLocation, AbiPlan, CallConv, TargetMachine};
use smallvec::SmallVec;
use veloc_lir::{GenericOpcode, InstId, MachineFunction, MachineOpcode, Reg, StackSlot, Writable};
use veloc_lir::{InstBuild, InstRead};
use veloc_lir::{MemoryAccess, MemoryKind};

pub struct AbiLoweringPass;

impl AbiLoweringPass {
    pub fn new() -> Self {
        Self
    }
}

// Check each ABI plan before lowering its boundary.
fn plan_signature(target: &dyn TargetMachine, sig: &veloc_mir::Signature) -> Result<AbiPlan> {
    let plan =
        CallConv::from(sig.call_conv).plan(target.desc().arch, sig.params(), sig.returns())?;
    let frame = target.frame_lowering();
    if plan.stack.align > frame.stack_alignment() {
        return Err(Error::codegen("ABI requires unsupported stack realignment"));
    }
    for assignment in plan.args.iter().chain(&plan.returns) {
        if let AbiLocation::Stack { offset, size, .. } = assignment.loc {
            let fits = offset
                .checked_add(size)
                .and_then(|end| i32::try_from(end).ok());
            if fits.is_none() {
                return Err(Error::codegen(
                    "ABI stack area exceeds frame addressing range",
                ));
            }
            let bytes = target
                .desc()
                .data_layout
                .layout_of(assignment.ty)
                .and_then(|layout| layout.store_size.fixed_bytes());
            if bytes.is_none_or(|bytes| bytes > size) {
                return Err(Error::codegen(
                    "ABI stack slot cannot hold the transferred type",
                ));
            }
        }
    }
    if plan
        .returns
        .iter()
        .any(|a| matches!(a.loc, AbiLocation::Stack { .. }))
    {
        return Err(Error::codegen("stack return lowering is not implemented"));
    }
    Ok(plan)
}

fn registers(assignments: &[AbiAssignment]) -> impl Iterator<Item = Reg> + '_ {
    assignments.iter().filter_map(|p| match p.loc {
        AbiLocation::Reg(reg) => Some(reg),
        AbiLocation::Stack { .. } => None,
    })
}

/// The semantic argument area is independent of the physical frame strategy.
#[derive(Clone, Copy)]
enum ArgArea {
    Incoming,
    Outgoing(veloc_lir::CallFrameId),
}

/// Converts ABI locations into transfers; layout insertion is owned by LIR.
struct Transfer<'a> {
    target: &'a dyn TargetMachine,
    func: veloc_lir::InstInserter<'a>,
}

impl<'a> Transfer<'a> {
    fn new(target: &'a dyn TargetMachine, func: veloc_lir::InstInserter<'a>) -> Self {
        Self { target, func }
    }

    fn address(&mut self, area: ArgArea, offset: u32, size: u32, align: u32) -> (Reg, StackSlot) {
        let object = match area {
            ArgArea::Incoming => veloc_lir::StackObject::Incoming { offset },
            ArgArea::Outgoing(frame) => veloc_lir::StackObject::Outgoing { frame, offset },
        };
        let slot = self.func.alloc_stack_object(object, size, align);
        let address = self.func.alloc_vreg(veloc_lir::Type::PTR);
        self.func.writer().stack_addr(Writable(address), slot);
        (address, slot)
    }

    fn access(&self, assignment: &AbiAssignment, align: u32, kind: MemoryKind) -> MemoryAccess {
        let bytes = self
            .target
            .desc()
            .data_layout
            .layout_of(assignment.ty)
            .and_then(|layout| layout.store_size.fixed_bytes())
            .expect("checked ABI storage layout");
        let mut access = MemoryAccess::new(kind, bytes);
        access.alignment = align;
        access.may_trap = false;
        access
    }

    fn read(&mut self, dst: Reg, assignment: &AbiAssignment, area: ArgArea) {
        match assignment.loc {
            AbiLocation::Reg(reg) => self.func.writer().copy(Writable(dst), reg),
            AbiLocation::Stack {
                offset,
                size,
                align,
            } => {
                let (address, _) = self.address(area, offset, size, align);
                let access = self.access(assignment, align, MemoryKind::Read);
                self.func
                    .writer()
                    .with_memory(access)
                    .load(Writable(dst), address, 0)
            }
        };
    }

    fn write(&mut self, src: Reg, assignment: &AbiAssignment, area: ArgArea) -> Option<StackSlot> {
        match assignment.loc {
            AbiLocation::Reg(reg) => {
                self.func.writer().copy(Writable(reg), src);
                None
            }
            AbiLocation::Stack {
                offset,
                size,
                align,
            } => {
                let (address, slot) = self.address(area, offset, size, align);
                let access = self.access(assignment, align, MemoryKind::Write);
                self.func
                    .writer()
                    .with_memory(access)
                    .store(src, address, 0);
                Some(slot)
            }
        }
    }
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
    let mut transfer = Transfer::new(target, mfunc.at_start(entry));
    for (dst, assignment) in params.into_iter().zip(&plan.args) {
        transfer.read(dst, assignment, ArgArea::Incoming);
    }
}

pub(super) fn lower_call(
    target: &dyn TargetMachine,
    mfunc: &mut veloc_lir::FuncEditor<'_>,
    id: InstId,
) -> Result<()> {
    let plan = plan_signature(target, &mfunc.call_info(id).sig)?;
    lower_callsite(target, mfunc, id, &plan);
    Ok(())
}

fn lower_callsite(
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

    let results = SmallVec::<[Reg; 2]>::from_slice(results);
    let frame = mfunc.alloc_call_frame(plan.stack);
    mfunc.before(id).writer().call_frame_setup(frame);

    // Place logical arguments in their ABI locations before the call.
    let mut stack_args = SmallVec::new();
    {
        let mut transfer = Transfer::new(target, mfunc.before(id));
        for (index, assignment) in plan.args.iter().enumerate() {
            // Borrow only long enough to copy one ID; insertion may grow the store.
            let src = transfer.func.inst(id).inputs()[index + usize::from(callee.is_some())];
            stack_args.extend(transfer.write(src, assignment, ArgArea::Outgoing(frame)));
        }
    }

    // Commit the full ABI call before defining the original SSA results.
    let args: SmallVec<[Reg; 8]> = registers(&plan.args).collect();
    let returns: SmallVec<[Reg; 2]> = registers(&plan.returns).collect();
    mfunc.set_call_abi(id, &returns, &args, frame, plan.abi.clobbers, stack_args);

    // The old definitions are now released, so the copies can reuse their IDs.
    let mut insert = mfunc.after(id);
    for (&dst, assignment) in results.iter().zip(&plan.returns) {
        let AbiLocation::Reg(reg) = assignment.loc else {
            unreachable!("checked register return")
        };
        insert.writer().copy(Writable(dst), reg);
    }
    insert.writer().call_frame_destroy(frame);
}

fn lower_return(
    mfunc: &mut veloc_lir::FuncEditor<'_>,
    id: InstId,
    plan: &AbiPlan,
    return_regs: &[Reg],
) {
    let veloc_lir::InstView::Return(ret) = mfunc.inst(id).view() else {
        unreachable!("planned return")
    };
    let value_count = ret.values.len();
    assert_eq!(
        value_count,
        plan.returns.len(),
        "return value count mismatch"
    );

    {
        let mut insert = mfunc.before(id);
        for (index, assignment) in plan.returns.iter().enumerate() {
            let src = insert.inst(id).inputs()[index];
            let AbiLocation::Reg(reg) = assignment.loc else {
                unreachable!("checked register return")
            };
            insert.writer().copy(Writable(reg), src);
        }
    }
    mfunc.set_inst_inputs(id, return_regs);
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
        let return_regs: SmallVec<[Reg; 4]> = registers(&plan.returns).collect();

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
                    lower_return(&mut mfunc.editor(), id, &plan, &return_regs);
                }
                _ => {}
            }
        }

        Ok(PassEffect::new(
            ChangeSet::INST_SEMANTICS | ChangeSet::PHYSICAL_REGS | ChangeSet::STACK_FRAME,
        ))
    }
}
