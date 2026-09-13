//! Allocation is a plan over an unchanged function, not a mutation of its values.
use alloc::vec::Vec;
use cranelift_entity::SecondaryMap;
use smallvec::SmallVec;
use veloc_lir::stages::{PostIselOptimized, RegAllocated};
use veloc_lir::{InstId, MachineFunction, MachineOperand, PReg, StackFrame, Writable};

/// Physical locations and insertions for one instruction. Locations are indexed
/// by operand occurrence, not by virtual register: split ranges may have different
/// locations at different instructions. Non-register operands have no location.
#[derive(Debug, Clone, Default)]
pub struct InstAllocation {
    pub(crate) locations: SmallVec<[Option<PReg>; 4]>,
    pub(crate) before: Vec<InstId>,
    pub(crate) after: Vec<InstId>,
}

impl InstAllocation {
    pub fn locations(&self) -> &[Option<PReg>] {
        &self.locations
    }

    pub fn before(&self) -> &[InstId] {
        &self.before
    }

    pub fn after(&self) -> &[InstId] {
        &self.after
    }
}

/// Owns the exact input of the allocation plan. It cannot be edited or replaced
/// behind the plan's back. Materialization consumes both without cloning the IR.
pub struct Allocation {
    pub(crate) source: MachineFunction<PostIselOptimized>,
    pub(crate) instructions: SecondaryMap<InstId, InstAllocation>,
    pub(crate) frame: StackFrame,
    pub(crate) edges: Vec<super::edges::EdgeAllocation>,
}

impl Allocation {
    pub fn source(&self) -> &MachineFunction<PostIselOptimized> {
        &self.source
    }

    pub fn inst(&self, id: InstId) -> &InstAllocation {
        &self.instructions[id]
    }

    pub fn edges(&self) -> &[super::edges::EdgeAllocation] {
        &self.edges
    }

    pub fn frame(&self) -> &StackFrame {
        &self.frame
    }

    /// Physical IR is produced only after all location and spill decisions have
    /// succeeded. This step neither consults a target nor runs allocation again.
    pub fn materialize(self) -> MachineFunction<RegAllocated> {
        let Self {
            mut source,
            mut instructions,
            frame,
            edges,
        } = self;
        source.stack_frame = frame;
        for block in 0..source.num_blocks() {
            source
                .rewrite_block(block, |cursor| {
                    let id = cursor.current_inst_id();
                    let plan = core::mem::take(&mut instructions[id]);
                    for inst in plan.before {
                        cursor.emit(inst);
                    }
                    // Preserve all non-register fields and payloads. No instruction
                    // replacement or extra-payload cloning is necessary.
                    let mut operands: SmallVec<[_; 4]> =
                        cursor.mfunc().inst(id).operands().iter().cloned().collect();
                    assert_eq!(operands.len(), plan.locations.len());
                    for (operand, location) in operands.iter_mut().zip(plan.locations) {
                        match (operand, location) {
                            (MachineOperand::Def(reg), Some(loc)) => {
                                *reg = Writable(loc.into());
                            }
                            (MachineOperand::Use(reg), Some(loc)) => {
                                *reg = loc.into();
                            }
                            (operand, None) => assert!(operand.as_reg().is_none()),
                            _ => unreachable!("allocation must match the source operand shape"),
                        }
                    }
                    cursor.mfunc_mut().set_inst_operands(id, operands);
                    cursor.keep_current();
                    // The cursor appends to the output sequence; after keep_current
                    // these insertions follow the instruction.
                    for inst in plan.after {
                        cursor.emit(inst);
                    }
                    Ok::<(), core::convert::Infallible>(())
                })
                .unwrap();
        }
        // Layout changes happen only now: each nonempty edge plan gets a block,
        // so conditional branches and critical edges execute only their own moves.
        for edge in edges {
            let block = source.create_synthetic_block();
            let index = source.find_block_index(block).unwrap();
            for id in edge.instructions {
                source.append_inst_id_to_block(index, id);
            }
            source.set_inst_operand(edge.branch, edge.operand, MachineOperand::Block(block));
        }
        let ids: Vec<_> = source
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter().copied())
            .collect();
        for id in ids {
            if matches!(source.inst_extra(id), Some(veloc_lir::InstExtra::Branch(_))) {
                source.clear_inst_extra(id);
            }
        }
        for block in &mut source.blocks {
            block.params.clear();
        }
        source.params.clear();
        source.is_regallocated = true;
        source.into_stage()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TargetConfig;
    use crate::pipeline::FunctionAnalysisCtx;
    use crate::regalloc::RegisterAllocator;
    use crate::target::x86_64::{
        X86_64TargetMachine,
        isle::{REG_RAX, TargetInst},
    };
    use veloc_lir::{MachineOpcode, Type, stages::RawLir};

    #[test]
    fn allocation_preserves_input_and_materializes_spills_in_order() {
        let target = X86_64TargetMachine::new(TargetConfig::default());
        let mut f = MachineFunction::<RawLir>::new("pressure".into());
        f.create_synthetic_block();
        let mut values = Vec::new();
        // All values are live together, forcing both assigned and spilled ranges.
        for n in 0..40 {
            let reg = f.alloc_vreg(Type::I64);
            values.push(reg);
            {
                let id = f.writer().generic(
                    MachineOpcode::Target(TargetInst::X86Mov64Imm64.as_u32()),
                    smallvec::smallvec![MachineOperand::Def(Writable(reg)), MachineOperand::Imm(n)],
                );
                f.append_inst_id_to_block(0, id);
                id
            };
        }
        for &reg in &values {
            {
                let id = f.writer().unary(
                    MachineOpcode::Target(TargetInst::X86Mov64.as_u32()),
                    Writable(REG_RAX),
                    reg,
                );
                f.append_inst_id_to_block(0, id);
                id
            };
        }
        let ids = f.block_insts(0).to_vec();
        let plan = RegisterAllocator::new(&target)
            .allocate(
                f.into_stage(),
                veloc_mir::CallConv::SystemV,
                &mut FunctionAnalysisCtx::default(),
            )
            .unwrap();
        assert_eq!(plan.source().block_insts(0), ids);
        assert!(plan.source().stack_frame.slots.is_empty());
        assert!(!plan.frame().slots.is_empty());
        for (&id, &reg) in ids.iter().zip(&values) {
            assert_eq!(plan.source().inst(id).defs().collect::<Vec<_>>(), [reg]);
        }
        let insertions: Vec<_> = ids
            .iter()
            .map(|&id| {
                let inst = plan.inst(id);
                assert_eq!(
                    inst.locations().len(),
                    plan.source().inst(id).operands().len()
                );
                (
                    inst.before()
                        .iter()
                        .map(|&i| plan.source().inst(i).opcode())
                        .collect::<Vec<_>>(),
                    inst.after()
                        .iter()
                        .map(|&i| plan.source().inst(i).opcode())
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        assert!(insertions.iter().any(|(before, _)| !before.is_empty()));
        assert!(insertions.iter().any(|(_, after)| !after.is_empty()));

        let physical = plan.materialize();
        let mut offset = 0;
        for (id, (before, after)) in ids.into_iter().zip(insertions) {
            for op in before {
                assert_eq!(physical.inst(physical.block_insts(0)[offset]).opcode(), op);
                offset += 1;
            }
            assert_eq!(physical.block_insts(0)[offset], id);
            offset += 1;
            for op in after {
                assert_eq!(physical.inst(physical.block_insts(0)[offset]).opcode(), op);
                offset += 1;
            }
        }
        assert_eq!(physical.block_insts(0).len(), offset);
        assert!(physical.is_regallocated);
        for &id in physical.block_insts(0) {
            assert!(
                physical
                    .inst(id)
                    .uses()
                    .chain(physical.inst(id).defs())
                    .all(|r| r.is_preg())
            );
        }
    }
}
