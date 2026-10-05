//! Edit installed edges through their metadata and shared operand storage.
use super::{DataFlowGraph, OperandRange};
use crate::function::EdgeRef;
use crate::{Block, Inst, Opcode, Successor, Value};
use core::ops::Range;

impl DataFlowGraph {
    pub(crate) fn edge_args(&self, edge: EdgeRef) -> (Block, Range<u32>) {
        let inst = &self.instructions[edge.inst];
        let mut found = None;
        let mut index = 0;
        inst.fields
            .visit_edges(&self.fields, inst.operands.len, |data, args| {
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
        let old_len = inst.operands.len;
        inst.operands = self.operands.replace(edge.inst, inst.operands, group, args);
        let mut index = 0;
        inst.fields
            .edit_edges(&mut self.fields, old_len, |data, _| {
                if index == edge.index {
                    data.block = target;
                    data.len = args.len().try_into().expect("too many successor arguments");
                }
                index += 1;
            });
        old_target != target
    }

    /// Filter every occurrence of the target, including repeated table cases.
    pub(crate) fn retain_edge_args(&mut self, inst: Inst, target: Block, keep: &[bool]) {
        let inst = &mut self.instructions[inst];
        let mut compact = self.operands.compact(inst.operands);
        inst.fields
            .edit_edges(&mut self.fields, inst.operands.len, |edge, args| {
                if edge.block == target {
                    assert_eq!(
                        edge.len as usize,
                        keep.len(),
                        "successor argument count mismatch"
                    );
                    let mut len = 0;
                    compact.retain(args, |index| {
                        len += u32::from(keep[index]);
                        keep[index]
                    });
                    edge.len = len;
                }
            });
        inst.operands = compact.finish();
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

        // Transfer ownership of the compacted range across opcode replacement.
        // The writer releases the old fields, but must not release these uses.
        self.instructions[edge.inst].operands = OperandRange::default();
        self.replace_inst(edge.inst, |writer| {
            writer.jump(Successor { block, args: &[] })
        });
        let inst = &mut self.instructions[edge.inst];
        inst.fields
            .edit_edges(&mut self.fields, 0, |data, _| data.len = operands.len);
        inst.operands = operands;
    }
}
