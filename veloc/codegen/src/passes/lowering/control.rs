//! Explicit control-flow lowering, independent of instruction legalization.
use crate::error::Error;
use crate::pipeline::{FunctionPass, FunctionSession, FunctionStage};
use std::vec::Vec;
use veloc_lir::{GenericOpcode, InstBuild, InstRead, InstView, Successor};
use veloc_mir::{IntCC, Type};

/// A target opts into comparison-chain lowering. Keeping this outside legality
/// leaves room for targets to retain tables or choose another implementation.
pub(crate) struct BranchTableLowering;

impl FunctionPass for BranchTableLowering {
    fn name(&self) -> &'static str {
        "lower-branch-tables"
    }

    fn input_stage(&self) -> FunctionStage {
        FunctionStage::Generic
    }
    fn run(&self, cx: &mut FunctionSession<'_>) -> crate::Result<()> {
        let f = cx.function();
        let tables: Vec<_> = f
            .blocks()
            .flat_map(|b| f.block_insts(b))
            .filter(|&id| f.inst(id).generic_opcode() == Some(GenericOpcode::Brjt))
            .collect();
        if tables.is_empty() {
            return Ok(());
        }
        let mut f = cx.edit();
        for id in tables {
            let InstView::BranchTable(table) = f.inst(id).view() else {
                unreachable!()
            };
            let index = table.index;
            let targets: Vec<Successor> = f
                .successors(id)
                .map(|edge| Successor {
                    block: edge.block,
                    args: edge.args.into(),
                })
                .collect();
            let Some((default, cases)) = targets.split_last() else {
                return Err(Error::codegen("branch table has no default target"));
            };
            let block = f.inst_block(id).expect("placed branch table");
            if f.block_insts(block).next_back() != Some(id) {
                return Err(Error::codegen("branch table must terminate its block"));
            }
            if cases.is_empty() {
                let edge = f.editor().create_edge(default.block, &default.args);
                f.editor().replace(id).br(edge);
                continue;
            }
            let ty = f.vreg_data(index).ty;
            f.editor().invalidate_inst(id);
            let mut current = block;
            for (case, target) in cases.iter().enumerate() {
                let last = case + 1 == cases.len();
                let next = if last {
                    default.block
                } else {
                    f.editor().create_block()
                };
                let value = f.editor().alloc_vreg(ty);
                let equal = f.editor().alloc_vreg(Type::BOOL);
                f.editor().at_end(current).constant(value, case as i64);
                f.editor()
                    .at_end(current)
                    .icmp(equal, index, value, IntCC::Eq);
                let yes = f.editor().create_edge(target.block, &target.args);
                let no = f
                    .editor()
                    .create_edge(next, if last { &default.args } else { &[] });
                f.editor().at_end(current).brcond(equal, yes, no);
                current = next;
            }
        }
        Ok(())
    }
}
