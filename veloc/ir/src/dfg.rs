use super::inst::{Inst, InstructionData};
use crate::constant::Constant;
use crate::types::{
    Block, BlockCall, BlockCallData, JumpTable, JumpTableData, Type, Value, ValueData, ValueDef,
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
    pub inst_results: SecondaryMap<Inst, Option<Value>>,
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
            inst_results: SecondaryMap::new(),
            value_lists,
            value_pool: Vec::new(),
            block_calls: PrimaryMap::new(),
            jump_tables: PrimaryMap::new(),
        }
    }

    pub fn make_value_list(&mut self, values: &[Value]) -> ValueList {
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

    pub fn append_result(&mut self, inst: Inst, ty: Type) -> Value {
        let val = self.values.push(ValueData {
            ty,
            def: ValueDef::Inst(inst),
        });
        self.inst_results[inst] = Some(val);
        val
    }

    pub fn append_block_param(&mut self, block: Block, ty: Type) -> Value {
        self.values.push(ValueData {
            ty,
            def: ValueDef::Param(block),
        })
    }

    pub fn inst_results(&self, inst: Inst) -> Option<Value> {
        self.inst_results[inst]
    }

    pub fn inst(&self, inst: Inst) -> &InstructionData {
        &self.instructions[inst]
    }

    pub fn inst_mut(&mut self, inst: Inst) -> &mut InstructionData {
        &mut self.instructions[inst]
    }

    pub fn value_type(&self, val: Value) -> Type {
        self.values[val].ty
    }

    pub fn value_def(&self, val: Value) -> ValueDef {
        self.values[val].def
    }

    pub fn value_inst(&self, val: Value) -> Option<Inst> {
        match self.value_def(val) {
            ValueDef::Inst(inst) => Some(inst),
            ValueDef::Param(_) => None,
        }
    }

    pub fn as_const(&self, val: Value) -> Option<Constant> {
        if let ValueDef::Inst(inst) = self.value_def(val) {
            match &self.instructions[inst] {
                InstructionData::Iconst { value, ty } => match ty {
                    Type::I8 => Some(Constant::I8(*value as i8)),
                    Type::I16 => Some(Constant::I16(*value as i16)),
                    Type::I32 => Some(Constant::I32(*value as i32)),
                    Type::I64 => Some(Constant::I64(*value)),
                    _ => None,
                },
                InstructionData::Fconst { value, ty } => match ty {
                    Type::F32 => Some(Constant::F32(f32::from_bits(*value as u32))),
                    Type::F64 => Some(Constant::F64(f64::from_bits(*value))),
                    _ => None,
                },
                InstructionData::Bconst { value } => Some(Constant::Bool(*value)),
                _ => None,
            }
        } else {
            None
        }
    }

    pub fn block_call_block(&self, call: BlockCall) -> Block {
        self.block_calls[call].block
    }

    pub fn block_call_args(&self, call: BlockCall) -> &[Value] {
        self.get_value_list(self.block_calls[call].args)
    }

    pub fn jump_table_targets(&self, table: JumpTable) -> &[BlockCall] {
        &self.jump_tables[table].targets
    }

    pub fn analyze_successors(&self, inst: Inst) -> Vec<Block> {
        match &self.instructions[inst] {
            InstructionData::Jump { dest } => vec![self.block_call_block(*dest)],
            InstructionData::Br {
                then_dest,
                else_dest,
                ..
            } => {
                vec![
                    self.block_call_block(*then_dest),
                    self.block_call_block(*else_dest),
                ]
            }
            InstructionData::BrTable { table, .. } => {
                let mut succs = Vec::new();
                for &target_call in self.jump_table_targets(*table) {
                    succs.push(self.block_call_block(target_call));
                }
                succs
            }
            _ => vec![],
        }
    }

    pub fn remove_inst(&mut self, inst: Inst) {
        self.instructions[inst] = InstructionData::Nop;
        self.inst_results[inst] = None;
    }
}
