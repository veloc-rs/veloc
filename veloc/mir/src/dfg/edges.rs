//! Edit installed edges through their metadata and shared operand storage.
use super::DataFlowGraph;
use crate::function::EdgeRef;
use crate::{Block, Inst, Opcode, Value};
use core::ops::Range;

impl DataFlowGraph {
    pub(crate) fn edge_args(&self, edge: EdgeRef) -> (Block, Range<u32>) {
        let inst = &self.instructions[edge.inst];
        let mut found = None;
        let mut index = 0;
        inst.fields.visit_edges(inst.operands.len, |data, args| {
            if index == edge.index {
                found = Some((data.block, args));
            }
            index += 1;
        });
        found.expect("successor position out of bounds")
    }

    /// Return whether the target changed; argument edits do not change the CFG.
    pub(crate) fn redirect_edge(&mut self, edge: EdgeRef, target: Block, args: &[Value]) -> bool {
        let (old_target, group) = self.edge_args(edge);
        let inst = &mut self.instructions[edge.inst];
        let count = args.len().try_into().expect("too many successor arguments");
        inst.operands = self.operands.replace(edge.inst, inst.operands, group, args);
        inst.fields.set_edge_target(edge.index as usize, target);
        inst.fields
            .set_stored_edge_arg_count(edge.index as usize, count);
        old_target != target
    }

    /// Filter every occurrence of the target, including repeated table cases.
    pub(crate) fn retain_edge_args(&mut self, inst: Inst, target: Block, keep: &[bool]) {
        let stored = &mut self.instructions[inst];
        let mut compact = self.operands.compact(stored.operands);
        let mut selected = smallvec::SmallVec::<[usize; 2]>::new();
        let mut index = 0;
        let count = keep.iter().filter(|&&retain| retain).count() as u32;
        stored
            .fields
            .visit_edges(stored.operands.len, |edge, args| {
                if edge.block == target {
                    assert_eq!(
                        edge.len as usize,
                        keep.len(),
                        "successor argument count mismatch"
                    );
                    compact.retain(args, |index| keep[index]);
                    selected.push(index);
                }
                index += 1;
            });
        stored.operands = compact.finish();
        // All old ranges have been consumed; only now change stored lengths.
        for index in selected {
            stored.fields.set_stored_edge_arg_count(index, count);
        }
    }

    /// Keep one branch edge and its existing uses, without an owned snapshot.
    pub(crate) fn fold_to_edge(&mut self, edge: EdgeRef) {
        assert!(matches!(
            self.opcode(edge.inst),
            Opcode::Jump | Opcode::Br | Opcode::BrTable
        ));
        let (block, args) = self.edge_args(edge);
        let old = self.instructions[edge.inst].operands;
        let mut compact = self.operands.compact(old);
        compact.retain(0..args.start, |_| false);
        compact.retain(args.end..old.len, |_| false);
        let operands = compact.finish();

        self.instructions[edge.inst].replace_with_jump(block, operands);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SuccessorData;

    #[test]
    fn repeated_edges_keep_argument_ranges_and_uses_after_editing() {
        let mut dfg = DataFlowGraph::new();
        let edges = [
            SuccessorData::new(Block(1), &[Value(1), Value(2)]),
            SuccessorData::new(Block(2), &[Value(3)]),
            SuccessorData::new(Block(1), &[Value(2), Value(1)]),
        ];
        let inst =
            dfg.create_inst(|w| w.br_table(Value(0), edges.iter().map(SuccessorData::as_view)));
        dfg.retain_edge_args(inst, Block(1), &[false, true]);
        dfg.redirect_edge(EdgeRef { inst, index: 1 }, Block(2), &[Value(3), Value(4)]);
        let view = dfg.inst(inst);
        assert_eq!(view.successor(0).unwrap().args, &[Value(2)]);
        assert_eq!(view.successor(1).unwrap().args, &[Value(3), Value(4)]);
        assert_eq!(view.successor(2).unwrap().args, &[Value(1)]);
        dfg.fold_to_edge(EdgeRef { inst, index: 2 });
        assert_eq!(dfg.opcode(inst), Opcode::Jump);
        assert_eq!(dfg.operands(inst), &[Value(1)]);
        assert_eq!(dfg.uses(Value(1)).count(), 1);
        for value in [Value(0), Value(2), Value(3), Value(4)] {
            assert_eq!(dfg.uses(value).count(), 0);
        }
    }
}
