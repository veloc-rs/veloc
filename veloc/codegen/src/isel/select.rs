//! Instruction Selector - 指令选择器
//!
//! 将通用 LIR 指令转换为目标架构特定指令。
//!
//! 目标提供静态规则和扩展，通用 VM 负责匹配与构建。

use super::matching::{self, SelectionPrograms};
use crate::analysis::CfgInfo;
use crate::target::FeatureSetRef;
use std::vec::Vec;
use veloc_lir::{GenericOpcode, InstCursor, InstId, MachineFunction, Reg};

/// Target data and the explicit host extension used by the selection VM.
#[derive(Clone, Copy)]
pub struct SelectPolicy<'a> {
    pub programs: &'static SelectionPrograms,
    pub features: FeatureSetRef<'a>,
    pub metadata: fn(u32) -> &'static crate::target::TargetInstMetadata,
    pub predicate: &'a dyn SelectHooks,
}

pub trait SelectHooks: Send + Sync {
    fn predicate(&self, id: u32, reg: Reg) -> bool;
}

fn format_select_failure_inst(mfunc: &MachineFunction, inst_id: InstId) -> std::string::String {
    use std::format;

    let inst = &mfunc.inst(inst_id);
    let operand_types = inst
        .results()
        .iter()
        .copied()
        .chain(inst.uses())
        .filter_map(|reg| {
            if reg.is_vreg() {
                Some(format!("{:?}:{:?}", reg, mfunc.vreg_data(reg).ty))
            } else {
                Some(format!("{:?}:preg", reg))
            }
        })
        .collect::<Vec<_>>();

    if operand_types.is_empty() {
        format!("{inst:?}")
    } else {
        format!("{inst:?}; operand_types=[{}]", operand_types.join(", "))
    }
}

/// Reused construction buffers. A successful commit leaves both empty.
struct SelectionScratch {
    selected: Vec<InstId>,
    edge_transfers: Vec<(veloc_lir::EdgeId, veloc_lir::EdgeId)>,
}

impl SelectionScratch {
    fn commit(&mut self, mfunc: &mut MachineFunction, root: InstId) {
        let mut edit = mfunc.editor();
        for &(original, replacement) in &self.edge_transfers {
            assert!(edit.inst(root).edge_ids().any(|edge| edge == original));
            assert!(
                self.selected
                    .iter()
                    .any(|&inst| edit.inst(inst).edge_ids().any(|edge| edge == replacement))
            );
        }
        for (original, replacement) in self.edge_transfers.drain(..) {
            edit.transfer_edge(original, replacement);
        }
        match self.selected.as_slice() {
            // Retain the root's identity for a single-instruction replacement.
            &[replacement] => edit.replace_inst(root, replacement),
            // Zero or multiple instructions: the emitted sequence is already
            // placed before the root, which can now be removed.
            _ => edit.invalidate_inst(root),
        }
        self.selected.clear();
    }
}

/// 指令选择器
///
/// 这是 GlobalISel 的核心组件之一，负责驱动指令选择过程。
/// Executes target-provided programs through the shared VM.
pub struct InstructionSelector<'a> {
    target: SelectPolicy<'a>,
}

impl<'a> InstructionSelector<'a> {
    /// 创建新的指令选择器
    pub fn new(target: SelectPolicy<'a>) -> Self {
        Self { target }
    }

    /// Build the selected sequence, preserve boundary metadata, then commit.
    fn select_inst(
        &self,
        mfunc: &mut MachineFunction,
        inst_id: InstId,
        generic: GenericOpcode,
        scratch: &mut SelectionScratch,
    ) -> Result<(), crate::error::Error> {
        debug_assert!(scratch.selected.is_empty() && scratch.edge_transfers.is_empty());
        let opcode = mfunc.inst(inst_id).opcode();
        let program = self
            .target
            .programs
            .get(generic)
            .ok_or_else(|| crate::error::Error::select(opcode, "No selection program"))?;
        let memory = mfunc.inst(inst_id).memory();
        let mut edit = mfunc.editor();
        let mut insert = edit.before(inst_id);
        matching::execute(
            program,
            self.target.features,
            &|id, reg| self.target.predicate.predicate(id, reg),
            &mut insert,
            inst_id,
            &mut scratch.selected,
            &mut scratch.edge_transfers,
        )
        .ok_or_else(|| crate::error::Error::select(opcode, "No matching selection rule"))?;
        let constraints = edit.inst(inst_id).constraints().to_vec();
        if !constraints.is_empty() {
            let [destination] = scratch.selected.as_slice() else {
                return Err(crate::Error::codegen(
                    "constrained boundary must select one instruction",
                ));
            };
            let old = edit.inst(inst_id);
            let new = edit.inst(*destination);
            if old.inputs() != new.inputs() || old.results() != new.results() {
                return Err(crate::Error::codegen(
                    "selection changed constrained operand identities",
                ));
            }
            edit.set_inst_constraints(*destination, constraints);
        }
        if let Some(access) = memory {
            let mut destination = None;
            for &id in scratch.selected.iter() {
                let veloc_lir::MachineOpcode::Target(op) = edit.inst(id).opcode() else {
                    continue;
                };
                if let Some(shape) = (self.target.metadata)(op).memory {
                    if shape != (access.kind, access.bytes) || destination.replace(id).is_some() {
                        return Err(crate::error::Error::codegen(
                            "selection changed the memory access direction, size or count",
                        ));
                    }
                }
            }
            let id = destination.ok_or_else(|| {
                crate::error::Error::codegen("selection dropped the source memory access")
            })?;
            edit.set_inst_memory(id, Some(access));
        }
        scratch.commit(mfunc, inst_id);
        Ok(())
    }

    fn select_one(
        &self,
        mfunc: &mut MachineFunction,
        inst: InstId,
        scratch: &mut SelectionScratch,
    ) -> Result<(), crate::error::Error> {
        let source = mfunc.inst(inst);
        if source.is_invalid() || source.is_call_frame() {
            return Ok(());
        }
        // Matching does not own a producer's other uses. Remove pure producers
        // only when traversal reaches them and all their results are unused.
        if source.is_pure_value()
            && source
                .results()
                .iter()
                .all(|&reg| mfunc.uses(reg).next().is_none())
        {
            mfunc.editor().invalidate_inst(inst);
            return Ok(());
        }
        let veloc_lir::MachineOpcode::Generic(opcode) = source.opcode() else {
            return Ok(());
        };
        self.select_inst(mfunc, inst, opcode, scratch)
            .map_err(|error| match error {
                crate::error::Error::Select(error) => crate::error::Error::select(
                    error.opcode,
                    std::format!(
                        "{}; inst_id={:?}, inst={}",
                        error.reason,
                        inst,
                        format_select_failure_inst(mfunc, inst),
                    ),
                ),
                error => error,
            })
    }

    /// Requires unreachable blocks to have been removed. Selection preserves
    /// CFG topology; snapshot its order before rewriting instructions.
    pub fn select(
        &self,
        mfunc: &mut MachineFunction,
        cfg: &CfgInfo,
    ) -> Result<(), crate::error::Error> {
        let blocks = cfg.compute_post_order(mfunc.entry_block());
        assert_eq!(
            blocks.len(),
            mfunc.num_blocks(),
            "unreachable blocks must be removed before instruction selection"
        );
        let mut scratch = SelectionScratch {
            selected: Vec::with_capacity(4),
            edge_transfers: Vec::new(),
        };
        for block in blocks {
            let mut cursor = InstCursor::reverse_block(mfunc, block);
            while let Some(inst) = cursor.next(mfunc) {
                self.select_one(mfunc, inst, &mut scratch)?;
            }
        }
        Ok(())
    }
}
