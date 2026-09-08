//! Structural editing. Type contracts and dominance remain explicit validation.
use super::Function;
use crate::{Block, Inst, InstDraft, Type, Value};
use alloc::vec::Vec;

pub struct FunctionEditor<'a> {
    func: &'a mut Function,
}

impl<'a> FunctionEditor<'a> {
    pub(super) fn new(func: &'a mut Function) -> Self {
        Self { func }
    }

    pub fn function(&self) -> &Function {
        self.func
    }

    pub fn set_value_type(&mut self, value: Value, ty: Type) {
        self.func.dfg.set_value_type(value, ty);
    }

    pub fn set_operand(&mut self, inst: Inst, index: u32, value: Value) {
        self.func.dfg.set_operand(inst, index, value);
    }

    pub fn replace_all_uses(&mut self, old: Value, new: Value) {
        self.func.dfg.replace_all_uses(old, new);
    }

    pub fn append_inst(&mut self, block: Block, data: InstDraft, types: &[Type]) -> Inst {
        let control = data.is_terminator();
        let inst = self.func.dfg.create_inst(data);
        self.func.dfg.append_results(inst, types);
        self.func.layout.append_inst(block, inst);
        if control {
            self.sync_edges(block);
        }
        inst
    }

    pub fn insert_after(&mut self, after: Inst, data: InstDraft, types: &[Type]) -> Inst {
        let block = self
            .func
            .layout
            .inst_block(after)
            .expect("anchor not in layout");
        let control = data.is_terminator();
        let inst = self.func.dfg.create_inst(data);
        self.func.dfg.append_results(inst, types);
        self.func.layout.insert_after(after, inst);
        if control {
            self.sync_edges(block);
        }
        inst
    }

    pub fn replace_inst(&mut self, inst: Inst, data: InstDraft) {
        let block = self
            .func
            .layout
            .inst_block(inst)
            .expect("instruction not in layout");
        let control = data.is_terminator() || self.func.dfg.opcode(inst).spec().is_terminator();
        self.func.dfg.replace_inst(inst, data);
        if control {
            self.sync_edges(block);
        }
    }

    pub fn erase_inst(&mut self, inst: Inst) {
        self.erase_insts(&[inst]);
    }

    pub fn erase_insts(&mut self, insts: &[Inst]) {
        let mut blocks: Vec<_> = insts
            .iter()
            .filter(|&&inst| self.func.dfg.opcode(inst).spec().is_terminator())
            .map(|&inst| {
                self.func
                    .layout
                    .inst_block(inst)
                    .expect("instruction not in layout")
            })
            .collect();
        blocks.sort_unstable();
        blocks.dedup();
        self.func.dfg.remove_insts(insts);
        self.func.layout.remove_insts(insts);
        for block in blocks {
            self.sync_edges(block);
        }
    }

    /// Reattach an existing result, preserving its Value identity and all uses.
    pub fn move_result(&mut self, value: Value, to: Inst) {
        self.func.dfg.move_result(value, to);
    }

    fn sync_edges(&mut self, block: Block) {
        let old = core::mem::take(&mut self.func.layout.blocks[block].succs);
        for succ in old {
            self.func.layout.blocks[succ].preds.retain(|&b| b != block);
        }
        let mut successors = Vec::new();
        for &inst in &self.func.layout.blocks[block].insts {
            self.func
                .dfg
                .inst(inst)
                .visit_successors(|call| successors.push(call.block));
        }
        for succ in successors {
            self.func.layout.add_edge(block, succ);
        }
    }
}
