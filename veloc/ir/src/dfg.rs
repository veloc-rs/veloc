use super::inst::InstructionData;
use crate::types::{
    Block, BlockCall, BlockCallData, Inst, JumpTable, JumpTableData, Type, Value, ValueData,
    ValueList, ValueListData,
};
use alloc::string::String;
use alloc::vec::Vec;
use cranelift_entity::{PrimaryMap, SecondaryMap};

#[derive(Debug, Clone)]
pub struct DataFlowGraph {
    pub instructions: PrimaryMap<Inst, InstructionData>,
    pub values: PrimaryMap<Value, ValueData>,
    pub value_names: SecondaryMap<Value, String>,
    pub value_lists: PrimaryMap<ValueList, ValueListData>,
    pub value_pool: Vec<Value>,
    pub block_calls: PrimaryMap<BlockCall, BlockCallData>,
    pub jump_tables: PrimaryMap<JumpTable, JumpTableData>,
}

impl DataFlowGraph {
    pub fn new() -> Self {
        let mut value_lists = PrimaryMap::new();
        // Index 0 is the empty list
        value_lists.push(ValueListData { offset: 0, len: 0 });

        Self {
            instructions: PrimaryMap::new(),
            values: PrimaryMap::new(),
            value_names: SecondaryMap::new(),
            value_lists,
            value_pool: Vec::new(),
            block_calls: PrimaryMap::new(),
            jump_tables: PrimaryMap::new(),
        }
    }

    pub fn push_value_list(&mut self, values: &[Value]) -> ValueList {
        if values.is_empty() {
            return ValueList::empty();
        }
        let offset = self.value_pool.len() as u32;
        let len = values.len() as u32;
        self.value_pool.extend_from_slice(values);
        self.value_lists.push(ValueListData { offset, len })
    }

    pub fn get_value_list(&self, list: ValueList) -> &[Value] {
        let data = self.value_lists[list];
        &self.value_pool[data.offset as usize..(data.offset + data.len) as usize]
    }

    pub fn make_value(&mut self, ty: Type) -> Value {
        self.values.push(ValueData {
            ty,
            defined_by: None,
        })
    }

    pub fn inst_results(&self, inst: Inst) -> Option<Value> {
        // In our current simplified SSA, an instruction has at most one result.
        // We can find the value that is defined by this instruction.
        self.values
            .iter()
            .find(|(_, data)| data.defined_by == Some(inst))
            .map(|(v, _)| v)
    }

    pub fn analyze_successors(&self, inst: Inst) -> Vec<Block> {
        match &self.instructions[inst] {
            InstructionData::Jump { dest } => vec![self.block_calls[*dest].block],
            InstructionData::Br {
                then_dest,
                else_dest,
                ..
            } => {
                vec![
                    self.block_calls[*then_dest].block,
                    self.block_calls[*else_dest].block,
                ]
            }
            InstructionData::BrTable { table, .. } => {
                let mut succs = Vec::new();
                for target_call in &self.jump_tables[*table].targets {
                    succs.push(self.block_calls[*target_call].block);
                }
                succs
            }
            _ => vec![],
        }
    }
}
