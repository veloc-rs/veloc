use super::inst::{Inst, InstructionData, PtrIndexImm, PtrIndexImmId};
use crate::constant::Constant;
use crate::types::{
    Block, BlockCall, BlockCallData, JumpTable, JumpTableData, Type, Value, ValueData, ValueDef,
    ValueList, ValueListPool,
};
use alloc::string::String;
use alloc::vec::Vec;
use cranelift_entity::{PrimaryMap, SecondaryMap};
use hashbrown::HashMap;

#[derive(Debug, Clone)]
pub struct DataFlowGraph {
    pub instructions: PrimaryMap<Inst, InstructionData>,
    pub values: PrimaryMap<Value, ValueData>,
    pub value_names: SecondaryMap<Value, String>,
    pub inst_results: SecondaryMap<Inst, ValueList>,
    value_list_pool: ValueListPool,
    pub block_calls: PrimaryMap<BlockCall, BlockCallData>,
    pub jump_tables: PrimaryMap<JumpTable, JumpTableData>,
    pub ptr_imm_pool: PrimaryMap<PtrIndexImmId, PtrIndexImm>,
    ptr_imm_map: HashMap<PtrIndexImm, PtrIndexImmId>,
}

impl DataFlowGraph {
    pub fn new() -> Self {
        Self {
            instructions: PrimaryMap::new(),
            values: PrimaryMap::new(),
            value_names: SecondaryMap::new(),
            inst_results: SecondaryMap::new(),
            value_list_pool: ValueListPool::new(),
            block_calls: PrimaryMap::new(),
            jump_tables: PrimaryMap::new(),
            ptr_imm_pool: PrimaryMap::new(),
            ptr_imm_map: HashMap::new(),
        }
    }

    /// 为指令添加多个结果值（支持多返回值）
    pub fn append_results(&mut self, inst: Inst, types: &[Type]) -> ValueList {
        let values: Vec<Value> = types
            .iter()
            .map(|ty| {
                self.values.push(ValueData {
                    ty: ty.clone(),
                    def: ValueDef::Inst(inst),
                })
            })
            .collect();

        let list = self.make_value_list(&values);
        self.inst_results[inst] = list;
        list
    }

    /// 获取指令的所有结果值
    pub fn inst_results(&self, inst: Inst) -> &[Value] {
        self.inst_results[inst].as_slice(&self.value_list_pool)
    }

    pub fn make_ptr_imm(&mut self, offset: i32, scale: u32) -> PtrIndexImmId {
        let key = PtrIndexImm { offset, scale };

        if let Some(&id) = self.ptr_imm_map.get(&key) {
            return id;
        }

        let id = self.ptr_imm_pool.push(key);
        self.ptr_imm_map.insert(key, id);
        id
    }

    /// 通过 ID 获取 PtrIndex 静态配置
    pub fn get_ptr_imm(&self, id: PtrIndexImmId) -> &PtrIndexImm {
        &self.ptr_imm_pool[id]
    }

    /// 从切片创建 ValueList
    pub fn make_value_list(&mut self, values: &[Value]) -> ValueList {
        ValueList::from_slice(values, &mut self.value_list_pool)
    }

    /// 获取 ValueList 的切片引用
    pub fn get_value_list(&self, list: ValueList) -> &[Value] {
        list.as_slice(&self.value_list_pool)
    }

    pub fn append_block_param(&mut self, block: Block, ty: Type) -> Value {
        self.values.push(ValueData {
            ty,
            def: ValueDef::Param(block),
        })
    }

    pub fn inst(&self, inst: Inst) -> &InstructionData {
        &self.instructions[inst]
    }

    pub fn inst_mut(&mut self, inst: Inst) -> &mut InstructionData {
        &mut self.instructions[inst]
    }

    pub fn value_type(&self, val: Value) -> Type {
        self.values[val].ty.clone()
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
            let ty = self.value_type(val);
            match &self.instructions[inst] {
                InstructionData::Iconst { value } => {
                    if ty == Type::I8 {
                        Some(Constant::I8(*value as i8))
                    } else if ty == Type::I16 {
                        Some(Constant::I16(*value as i16))
                    } else if ty == Type::I32 {
                        Some(Constant::I32(*value as i32))
                    } else if ty == Type::I64 {
                        Some(Constant::I64(*value))
                    } else {
                        None
                    }
                }
                InstructionData::Fconst { value } => {
                    if ty == Type::F32 {
                        Some(Constant::F32(f32::from_bits(*value as u32)))
                    } else if ty == Type::F64 {
                        Some(Constant::F64(f64::from_bits(*value)))
                    } else {
                        None
                    }
                }
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
        self.block_calls[call].args.as_slice(&self.value_list_pool)
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
        self.inst_results[inst] = ValueList::default();
    }
}
