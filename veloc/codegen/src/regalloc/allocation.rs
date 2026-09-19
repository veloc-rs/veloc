//! Allocation is a plan over an unchanged function, not a mutation of its values.
use alloc::vec::Vec;
use cranelift_entity::SecondaryMap;
use smallvec::SmallVec;
use veloc_lir::{InstId, MachineFunction, PReg, StackBatch};

/// Physical locations and insertions for one instruction. Locations are indexed
/// separately by result and input occurrence, not by virtual register: split ranges may have different
/// locations at different instructions. Only register inputs have locations; attributes are not visited.
#[derive(Debug, Clone, Default)]
pub struct InstAllocation {
    pub(crate) results: SmallVec<[PReg; 2]>,
    pub(crate) locations: SmallVec<[PReg; 4]>,
    pub(crate) before: Vec<InstId>,
    pub(crate) after: Vec<InstId>,
}

impl InstAllocation {
    pub fn results(&self) -> &[PReg] {
        &self.results
    }
    pub fn locations(&self) -> &[PReg] {
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
    /// succeeded. This step neither consults a target nor runs allocation again.
    pub fn materialize(self) -> MachineFunction {
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
                    edit.insert_before(id, inst);
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
                    edit.insert_after(after, inst);
                    after = inst;
                }
                cursor = next_id;
            }
            block = next_block;
        }
        // Layout changes happen only now: each nonempty edge plan gets a block,
        // so conditional branches and critical edges execute only their own moves.
        for edge in edges {
            let block = source.editor().create_block();
            for id in edge.instructions {
                source.editor().append_inst(block, id);
            }
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
        source.params.clear();
        source
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
                let id = f.editor().writer().write(
                    MachineOpcode::Target(TargetInst::X86Mov64Imm64.as_u32()),
                    &[reg],
                    &[],
                    [FieldValue::Imm(n)],
                );
                f.editor().append_inst(veloc_lir::BlockId::from_u32(0), id);
                id
            };
        }
        for &reg in &values {
            {
                let id = TargetInst::X86Mov64.write(f.editor().writer(), &[REG_RAX], &[reg], []);
                f.editor().append_inst(veloc_lir::BlockId::from_u32(0), id);
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
