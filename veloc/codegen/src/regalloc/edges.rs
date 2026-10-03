//! SSA edge arguments become physical parallel copies only after allocation.
use super::allocation::Transfer;
use super::linear_scan::RegisterAllocator;
use super::moves::{Location, Move, MoveResolver};
use crate::{Error, Result};
use std::vec::Vec;
use veloc_lir::{InstId, MachineFunction, Reg, StackBatch};

/// A physical move plan for one selected branch, emitted during materialization.
pub struct EdgeAllocation {
    pub(crate) branch: InstId,
    pub(crate) edge: veloc_lir::EdgeId,
    pub(crate) instructions: Vec<Transfer>,
    pub(crate) target: veloc_lir::BlockId,
}

impl EdgeAllocation {
    pub fn branch(&self) -> InstId {
        self.branch
    }
    pub fn instructions(&self) -> &[Transfer] {
        &self.instructions
    }
}

impl RegisterAllocator<'_> {
    pub(super) fn location(&self, reg: Reg) -> Result<Location> {
        if reg.is_preg() {
            return Ok(Location::Reg(reg));
        }
        if let Some(reg) = self.assigned(reg) {
            return Ok(Location::Reg(reg.into()));
        }
        self.spill_slot(reg)
            .map(Location::Stack)
            .ok_or_else(|| Error::codegen("value has no allocated location"))
    }

    pub(super) fn plan_edges(
        &self,
        f: &MachineFunction,
        frame: &mut StackBatch,
    ) -> Result<Vec<EdgeAllocation>> {
        let mut edges = Vec::new();
        let mut resolver = MoveResolver::default();
        for block in f.blocks() {
            for id in f.block_insts(block) {
                for edge_id in f.inst(id).edge_ids() {
                    let successor = f.inst(id).edge(edge_id);
                    let target_block = successor.block;
                    let args = successor.args;
                    let target = &target_block;
                    let params = f
                        .block_params(*target)
                        .ok_or_else(|| Error::codegen("unknown edge target"))?;
                    if params.len() != args.len() {
                        return Err(Error::codegen("edge argument count mismatch"));
                    }
                    let mut pending = Vec::new();
                    for (&dst, &src) in params.iter().zip(args) {
                        let ty = f.vreg_data(dst).ty;
                        if ty != f.vreg_data(src).ty {
                            return Err(Error::codegen("edge argument type mismatch"));
                        }
                        let dst = self.location(dst)?;
                        let src = self.location(src)?;
                        if dst != src {
                            pending.push(Move { dst, src, ty });
                        }
                    }
                    let instructions = resolver.resolve(self.target, frame, pending, &[])?;
                    if !instructions.is_empty() {
                        edges.push(EdgeAllocation {
                            branch: id,
                            edge: edge_id,
                            target: *target,
                            instructions,
                        });
                    }
                }
            }
        }
        Ok(edges)
    }
}
