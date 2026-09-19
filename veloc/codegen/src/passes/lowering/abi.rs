use crate::analysis::{ChangeSet, PassEffect};
use crate::error::{Error, Result};
use crate::pipeline::{FunctionPass, FunctionPassContext};
use crate::target::{AbiAssignment, AbiLocation, AbiPlan, CallConv, TargetMachine};
use alloc::vec::Vec;
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

// Check every boundary before editing the function. The ABI plan itself
// is independent of which side of the call is being lowered.
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

fn call_effects(target: &dyn TargetMachine, plan: &AbiPlan) -> veloc_lir::RegEffects {
    let file = &target.desc().registers;
    let preserved = plan.abi.preserved;
    let defs: Vec<_> = file
        .regs
        .iter()
        .map(|info| info.preg)
        .filter(|reg| {
            !preserved.contains(reg)
                && *reg != file.special_regs.stack_pointer
                && Some(*reg) != file.special_regs.frame_pointer
        })
        .collect();
    veloc_lir::RegEffects {
        uses: registers(&plan.args).collect(),
        defs,
    }
}

/// Directly emits moves at a boundary. The insertion point advances after a
/// call, while before an instruction successive inserts naturally keep order.
struct Transfer<'a> {
    target: &'a dyn TargetMachine,
    func: &'a mut MachineFunction,
    point: Insert,
    incoming: bool,
}

enum Insert {
    Before(InstId),
    After(InstId),
}

impl<'a> Transfer<'a> {
    fn new(target: &'a dyn TargetMachine, func: &'a mut MachineFunction, point: Insert) -> Self {
        Self {
            target,
            func,
            point,
            incoming: false,
        }
    }

    fn incoming(mut self) -> Self {
        self.incoming = true;
        self
    }

    fn emit(&mut self, inst: InstId) {
        match self.point {
            Insert::Before(at) => self.func.editor().insert_before(at, inst),
            Insert::After(at) => {
                self.func.editor().insert_after(at, inst);
                self.point = Insert::After(inst);
            }
        }
    }

    fn address(&mut self, offset: u32, size: u32, align: u32) -> (Reg, StackSlot) {
        let object = if self.incoming {
            veloc_lir::StackObject::Incoming { offset }
        } else {
            veloc_lir::StackObject::Outgoing { offset }
        };
        let slot = self.func.editor().alloc_stack_object(object, size, align);
        let address = self.func.editor().alloc_vreg(veloc_lir::Type::PTR);
        let inst = self
            .func
            .editor()
            .writer()
            .stack_addr(Writable(address), slot);
        self.emit(inst);
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

    fn read(&mut self, dst: Reg, assignment: &AbiAssignment) {
        let inst = match assignment.loc {
            AbiLocation::Reg(reg) => self.func.editor().writer().copy(Writable(dst), reg),
            AbiLocation::Stack {
                offset,
                size,
                align,
            } => {
                let (address, _) = self.address(offset, size, align);
                let access = self.access(assignment, align, MemoryKind::Read);
                self.func
                    .editor()
                    .writer()
                    .with_memory(access)
                    .load(Writable(dst), address, 0)
            }
        };
        self.emit(inst);
    }

    fn write(&mut self, src: Reg, assignment: &AbiAssignment) -> Option<StackSlot> {
        let (inst, slot) = match assignment.loc {
            AbiLocation::Reg(reg) => (self.func.editor().writer().copy(Writable(reg), src), None),
            AbiLocation::Stack {
                offset,
                size,
                align,
            } => {
                let (address, slot) = self.address(offset, size, align);
                let access = self.access(assignment, align, MemoryKind::Write);
                (
                    self.func
                        .editor()
                        .writer()
                        .with_memory(access)
                        .store(src, address, 0),
                    Some(slot),
                )
            }
        };
        self.emit(inst);
        slot
    }
}

fn lower_formal_arguments(target: &dyn TargetMachine, mfunc: &mut MachineFunction, plan: &AbiPlan) {
    let entry = mfunc.entry_block();
    let mut cursor = veloc_lir::InstCursor::block(mfunc, entry);
    while let Some(id) = cursor.next(mfunc) {
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
            Transfer::new(target, mfunc, Insert::Before(id))
                .incoming()
                .read(dst, assignment);
            mfunc.editor().replace_with(id, &[]);
        }
    }
}

fn lower_callsite(
    target: &dyn TargetMachine,
    mfunc: &mut MachineFunction,
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

    // Place logical arguments in their ABI locations before the call.
    let mut stack_args = SmallVec::new();
    {
        let mut transfer = Transfer::new(target, mfunc, Insert::Before(id));
        for (index, assignment) in plan.args.iter().enumerate() {
            // Borrow only long enough to copy one ID; insertion may grow the store.
            let src = transfer.func.inst(id).inputs()[index + usize::from(callee.is_some())];
            stack_args.extend(transfer.write(src, assignment));
        }
    }

    rewrite_call(target, mfunc, id, plan, callee, stack_args);

    // Read the ABI results back into the original SSA result registers.
    let mut transfer = Transfer::new(target, mfunc, Insert::After(id));
    for (index, assignment) in plan.returns.iter().enumerate() {
        let dst = transfer.func.inst(id).results()[index];
        let AbiLocation::Reg(reg) = assignment.loc else {
            unreachable!("checked register return");
        };
        // Release the old definition before creating its replacement copy.
        transfer.func.editor().set_inst_result(id, index, reg);
        transfer.read(dst, assignment);
    }
}

fn rewrite_call(
    target: &dyn TargetMachine,
    mfunc: &mut MachineFunction,
    id: InstId,
    plan: &AbiPlan,
    callee: Option<Reg>,
    stack_args: SmallVec<[StackSlot; 2]>,
) {
    // The call now uses ABI locations, not the original logical argument list.
    // Keep the indirect callee as its own explicit input.
    let mut inputs = SmallVec::<[Reg; 8]>::new();
    inputs.extend(callee);
    inputs.extend(registers(&plan.args));
    let mut effects = call_effects(target, plan);
    if let Some(existing) = mfunc.inst(id).effects() {
        effects.uses.extend_from_slice(existing.uses);
        effects.defs.extend_from_slice(existing.defs);
    }
    effects.uses.sort_unstable();
    effects.uses.dedup();
    effects.defs.sort_unstable();
    effects.defs.dedup();
    let mut editor = mfunc.editor();
    editor.set_inst_inputs(id, &inputs);
    editor.set_call_stack(id, stack_args, plan.stack);
    editor.set_inst_effects(id, effects);
}

fn lower_return(
    target: &dyn TargetMachine,
    mfunc: &mut MachineFunction,
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
        let mut transfer = Transfer::new(target, mfunc, Insert::Before(id));
        for (index, assignment) in plan.returns.iter().enumerate() {
            let src = transfer.func.inst(id).inputs()[index];
            transfer.write(src, assignment);
        }
    }
    let replacement = mfunc.editor().writer().ret(return_regs);
    mfunc.editor().replace_inst(id, replacement);
}

// Keep each plan attached to its instruction, rather than synchronizing
// a second iterator with a later full-function scan.
enum Boundary {
    Call(InstId, AbiPlan),
    Return(InstId),
}

fn plan_boundaries(target: &dyn TargetMachine, mfunc: &MachineFunction) -> Result<Vec<Boundary>> {
    let mut boundaries = Vec::new();
    for id in mfunc.blocks().flat_map(|block| mfunc.block_insts(block)) {
        match mfunc.inst(id).opcode() {
            MachineOpcode::Generic(GenericOpcode::Call | GenericOpcode::Callind) => {
                let plan = plan_signature(target, &mfunc.call_info(id).sig)?;
                boundaries.push(Boundary::Call(id, plan));
            }
            MachineOpcode::Generic(GenericOpcode::Ret) => boundaries.push(Boundary::Return(id)),
            _ => {}
        }
    }
    Ok(boundaries)
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
        let boundaries = plan_boundaries(ctx.target, mfunc)?;
        let return_regs: SmallVec<[Reg; 4]> = registers(&plan.returns).collect();

        lower_formal_arguments(ctx.target, mfunc, &plan);
        for boundary in boundaries {
            match boundary {
                Boundary::Call(id, call_plan) => {
                    lower_callsite(ctx.target, mfunc, id, &call_plan);
                }
                Boundary::Return(id) => {
                    lower_return(ctx.target, mfunc, id, &plan, &return_regs);
                }
            }
        }

        Ok(PassEffect::new(
            ChangeSet::INST_SEMANTICS | ChangeSet::PHYSICAL_REGS,
        ))
    }
}
