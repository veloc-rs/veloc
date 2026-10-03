//! Explicit control-flow lowering, independent of instruction legalization.
use crate::error::Error;
use crate::pipeline::{FunctionPass, FunctionSession, FunctionStage};
use std::vec::Vec;
use veloc_lir::{
    BlockId, FuncEditor, GenericOpcode, InstBuild, InstRead, InstView, Reg, Successor,
};
use veloc_mir::{IntCC, Type};

/// A target opts into comparison-tree lowering. Keeping this outside legality
/// leaves room for targets to retain tables or choose another implementation.
pub(crate) struct BranchTableLowering {
    pub max_cases: usize,
}

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
            .filter(|&id| {
                f.inst(id).generic_opcode() == Some(GenericOpcode::Brjt)
                    && f.inst(id).fields().successors().len().saturating_sub(1) <= self.max_cases
            })
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
            if targets.is_empty() {
                return Err(Error::codegen("branch table has no default target"));
            }
            let block = f.inst_block(id).expect("placed branch table");
            if f.block_insts(block).next_back() != Some(id) {
                return Err(Error::codegen("branch table must terminate its block"));
            }
            f.editor().invalidate_inst(id);
            emit_tree(&mut f, block, index, 0, &targets);
        }
        Ok(())
    }
}

// The final leaf represents [case_count, 2^32), including negative i32 bit
// patterns. Unsigned splits therefore include the default without a separate
// bounds test. Identical subranges collapse into one edge with its arguments.
fn emit_tree(
    f: &mut FuncEditor<'_>,
    block: BlockId,
    index: Reg,
    start: usize,
    targets: &[Successor],
) {
    let first = &targets[0];
    if targets
        .iter()
        .all(|t| t.block == first.block && t.args == first.args)
    {
        let edge = f.create_edge(first.block, &first.args);
        f.at_end(block).br(edge);
        return;
    }
    // Split between runs, never within a group of identical edges. Leaf arms
    // point directly to their destination, retaining block arguments; creating
    // forwarding blocks here would add jumps and obscure trace placement.
    let boundaries: Vec<_> = (1..targets.len())
        .filter(|&i| {
            targets[i - 1].block != targets[i].block || targets[i - 1].args != targets[i].args
        })
        .collect();
    let middle = boundaries[boundaries.len() / 2];
    let mut arm = |start, targets: &[Successor]| {
        let first = &targets[0];
        if targets
            .iter()
            .all(|t| t.block == first.block && t.args == first.args)
        {
            f.create_edge(first.block, &first.args)
        } else {
            let child = f.create_block();
            emit_tree(f, child, index, start, targets);
            f.create_edge(child, &[])
        }
    };
    let yes = arm(start, &targets[..middle]);
    let no = arm(start + middle, &targets[middle..]);
    let boundary = f.alloc_vreg(f.vreg_data(index).ty);
    let below = f.alloc_vreg(Type::BOOL);
    f.at_end(block).constant(boundary, (start + middle) as i64);
    f.at_end(block).icmp(below, index, boundary, IntCC::LtU);
    f.at_end(block).brcond(below, yes, no);
}
