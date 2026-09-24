//! Allocation is a plan over an unchanged function, not a mutation of its values.
use crate::target::{SpillKind, TargetRegalloc};
use cranelift_entity::SecondaryMap;
use smallvec::SmallVec;
use std::vec::Vec;
use veloc_lir::{InstId, MachineFunction, PReg, Reg, StackBatch, StackSlot, Type};

/// A planned physical transfer, not an instruction in the source function.
#[derive(Debug, Clone, Copy)]
pub enum Transfer {
    Copy {
        dst: Reg,
        src: Reg,
        ty: Type,
    },
    Spill {
        kind: SpillKind,
        reg: Reg,
        slot: StackSlot,
        ty: Type,
    },
}
impl Transfer {
    fn emit(
        self,
        target: &dyn TargetRegalloc,
        writer: veloc_lir::InstWriter<'_>,
    ) -> crate::Result<InstId> {
        match self {
            Self::Copy { dst, src, ty } => target.copy_instruction(writer, dst, src, ty),
            Self::Spill {
                kind,
                reg,
                slot,
                ty,
            } => target.spill_instruction(writer, kind, reg, slot, ty),
        }
    }
}

/// Physical locations and insertions for one instruction. Locations are indexed
/// separately by result and input occurrence, not by virtual register: split ranges may have different
/// locations at different instructions. Only register inputs have locations; attributes are not visited.
#[derive(Debug, Clone, Default)]
pub struct InstAllocation {
    pub(crate) results: SmallVec<[PReg; 2]>,
    pub(crate) locations: SmallVec<[PReg; 4]>,
    pub(crate) before: Vec<Transfer>,
    pub(crate) after: Vec<Transfer>,
}

impl InstAllocation {
    pub fn results(&self) -> &[PReg] {
        &self.results
    }
    pub fn locations(&self) -> &[PReg] {
        &self.locations
    }

    pub fn before(&self) -> &[Transfer] {
        &self.before
    }

    pub fn after(&self) -> &[Transfer] {
        &self.after
    }
}

/// Owns the exact input of the allocation plan. It cannot be edited or replaced
/// behind the plan's back. Materialization consumes both without cloning the IR.
pub struct Allocation {
    pub(crate) source: MachineFunction,
    pub(crate) instructions: SecondaryMap<InstId, InstAllocation>,
    pub(crate) frame: StackBatch,
    pub(crate) edges: Vec<super::edges::EdgeAllocation>,
}

impl Allocation {
    pub fn source(&self) -> &MachineFunction {
        &self.source
    }

    pub fn inst(&self, id: InstId) -> &InstAllocation {
        &self.instructions[id]
    }

    pub fn edges(&self) -> &[super::edges::EdgeAllocation] {
        &self.edges
    }

    pub fn frame(&self) -> &StackBatch {
        &self.frame
    }

    /// Physical IR is produced only after all location and spill decisions have
    /// succeeded. Target hooks emit transfers at their final insertion points.
    pub fn materialize(self, target: &dyn TargetRegalloc) -> crate::Result<MachineFunction> {
        let Self {
            mut source,
            mut instructions,
            frame,
            edges,
        } = self;
        source.stack_frame.append(frame);
        let mut block = source.blocks().next();
        while let Some(current_block) = block {
            let next_block = source.layout().next_block(current_block);
            let mut cursor = source.layout().first_inst(current_block);
            while let Some(id) = cursor {
                let next_id = source.layout().next_inst(id);
                let plan = core::mem::take(&mut instructions[id]);
                let mut edit = source.editor();
                for inst in plan.before {
                    inst.emit(target, edit.before(id).writer())?;
                }
                assert_eq!(plan.results.len(), edit.inst(id).results().len());
                for (index, reg) in plan.results.into_iter().enumerate() {
                    edit.set_inst_result(id, index, reg.into());
                }
                assert_eq!(plan.locations.len(), edit.inst(id).inputs().len());
                for (index, reg) in plan.locations.into_iter().enumerate() {
                    edit.set_inst_input(id, index, reg.into());
                }
                let mut after = id;
                for inst in plan.after {
                    after = inst.emit(target, edit.after(after).writer())?;
                }
                cursor = next_id;
            }
            block = next_block;
        }
        // Layout changes happen only now: each nonempty edge plan gets a block,
        // so conditional branches and critical edges execute only their own moves.
        for edge in edges {
            let block = source.editor().create_block();
            for transfer in edge.instructions {
                transfer.emit(target, source.editor().at_end(block).writer())?;
            }
            target.jump_instruction(source.editor().at_end(block).writer(), edge.target)?;
            let successor = source
                .inst(edge.branch)
                .edge_ids()
                .next()
                .expect("branch edge");
            source.editor().redirect_edge(successor, block);
            source.editor().set_edge_args(successor, &[]);
        }
        let mut block = source.blocks().next();
        while let Some(current_block) = block {
            let next_block = source.layout().next_block(current_block);
            let mut cursor = source.layout().first_inst(current_block);
            while let Some(id) = cursor {
                let next_id = source.layout().next_inst(id);
                source.editor().clear_successor_args(id);
                cursor = next_id;
            }
            block = next_block;
        }
        source.editor().clear_block_params();
        assert!(
            source.params().is_empty(),
            "ABI lowering must consume function parameters"
        );
        Ok(source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TargetConfig;
    use crate::analysis::FunctionAnalysisCtx;
    use crate::regalloc::RegisterAllocator;
    use crate::target::x86_64::{
        X86_64TargetMachine,
        inst::{REG_RAX, TargetInst},
    };
    use veloc_lir::FieldValue;
    use veloc_lir::{MachineOpcode, Type};

    #[test]
    fn allocation_preserves_input_and_materializes_spills_in_order() {
        let target = X86_64TargetMachine::new(TargetConfig::default()).unwrap();
        let mut f = MachineFunction::new("pressure".into());
        let mut values = Vec::new();
        // All values are live together, forcing both assigned and spilled ranges.
        for n in 0..40 {
            let reg = f.editor().alloc_vreg(Type::I64);
            values.push(reg);
            {
                let id = f
                    .editor()
                    .at_end(veloc_lir::BlockId::from_u32(0))
                    .writer()
                    .write(
                        MachineOpcode::Target(TargetInst::X86Mov64Imm64.as_u32()),
                        &[reg],
                        &[],
                        [FieldValue::Imm(n)],
                    );

                id
            };
        }
        for &reg in &values {
            {
                let id = TargetInst::X86Mov64.write(
                    f.editor().at_end(veloc_lir::BlockId::from_u32(0)).writer(),
                    &[REG_RAX],
                    &[reg],
                    [],
                );

                id
            };
        }
        let ids = f
            .block_insts(veloc_lir::BlockId::from_u32(0))
            .collect::<Vec<_>>();
        let plan = RegisterAllocator::new(&target)
            .allocate(f, &mut FunctionAnalysisCtx::default())
            .unwrap();
        assert_eq!(
            plan.source()
                .block_insts(veloc_lir::BlockId::from_u32(0))
                .collect::<Vec<_>>(),
            ids
        );
        assert!(plan.source().stack_frame.slots().is_empty());
        assert!(!plan.frame().slots().is_empty());
        for (&id, &reg) in ids.iter().zip(&values) {
            assert_eq!(plan.source().inst(id).defs().collect::<Vec<_>>(), [reg]);
        }
        let insertions: Vec<_> = ids
            .iter()
            .map(|&id| {
                let inst = plan.inst(id);
                assert_eq!(
                    inst.locations().len(),
                    plan.source().inst(id).inputs().len()
                );
                (
                    inst.before()
                        .iter()
                        .map(|&transfer| {
                            let mut scratch = MachineFunction::new("transfer".into());
                            let block = scratch.entry_block();
                            let id = transfer
                                .emit(&target, scratch.editor().at_end(block).writer())
                                .unwrap();
                            scratch.inst(id).opcode()
                        })
                        .collect::<Vec<_>>(),
                    inst.after()
                        .iter()
                        .map(|&transfer| {
                            let mut scratch = MachineFunction::new("transfer".into());
                            let block = scratch.entry_block();
                            let id = transfer
                                .emit(&target, scratch.editor().at_end(block).writer())
                                .unwrap();
                            scratch.inst(id).opcode()
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        assert!(insertions.iter().any(|(before, _)| !before.is_empty()));
        assert!(insertions.iter().any(|(_, after)| !after.is_empty()));

        let physical = plan.materialize(&target).unwrap();
        let mut offset = 0;
        for (id, (before, after)) in ids.into_iter().zip(insertions) {
            for op in before {
                assert_eq!(
                    physical
                        .inst(
                            physical
                                .block_insts(veloc_lir::BlockId::from_u32(0))
                                .collect::<Vec<_>>()[offset]
                        )
                        .opcode(),
                    op
                );
                offset += 1;
            }
            assert_eq!(
                physical
                    .block_insts(veloc_lir::BlockId::from_u32(0))
                    .collect::<Vec<_>>()[offset],
                id
            );
            offset += 1;
            for op in after {
                assert_eq!(
                    physical
                        .inst(
                            physical
                                .block_insts(veloc_lir::BlockId::from_u32(0))
                                .collect::<Vec<_>>()[offset]
                        )
                        .opcode(),
                    op
                );
                offset += 1;
            }
        }
        assert_eq!(
            physical
                .block_insts(veloc_lir::BlockId::from_u32(0))
                .collect::<Vec<_>>()
                .len(),
            offset
        );
        for id in physical
            .block_insts(veloc_lir::BlockId::from_u32(0))
            .collect::<Vec<_>>()
        {
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
