use super::inst::{ConstantPoolId, Inst, InstDraft, InstFields, InstructionView, StoredInst};
use crate::constant::Constant;
use crate::types::{Block, Type, Value, ValueData, ValueDef, ValueList, ValueListPool};
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use cranelift_entity::{PrimaryMap, SecondaryMap};
use hashbrown::HashMap;

mod operands;
mod pool;
pub(crate) use operands::OperandRange;
pub use operands::{Use, Uses};

#[derive(Debug, Clone)]
pub struct DataFlowGraph {
    instructions: PrimaryMap<Inst, StoredInst>,
    pub(crate) values: PrimaryMap<Value, ValueData>,
    pub value_names: SecondaryMap<Value, String>,
    pub(crate) inst_results: SecondaryMap<Inst, ValueList>,
    pub(crate) value_list_pool: ValueListPool,
    operands: operands::Operands,
    /// Constant bytes are immutable and may be interned.
    constant_pool: PrimaryMap<ConstantPoolId, Arc<[u8]>>,
    constant_pool_map: HashMap<Arc<[u8]>, ConstantPoolId>,
}

impl DataFlowGraph {
    /// Finalize parser-local function identities without touching SSA operands.
    pub(crate) fn remap_functions(&mut self, map: &[crate::FuncId]) {
        for (_, inst) in &mut self.instructions {
            inst.fields.map_functions(|id| map[id.0 as usize]);
        }
    }

    pub fn new() -> Self {
        Self {
            instructions: PrimaryMap::new(),
            values: PrimaryMap::new(),
            value_names: SecondaryMap::new(),
            inst_results: SecondaryMap::new(),
            value_list_pool: ValueListPool::new(),
            operands: operands::Operands::default(),
            constant_pool: PrimaryMap::new(),
            constant_pool_map: HashMap::new(),
        }
    }

    /// 为指令添加多个结果值（支持多返回值）
    pub fn append_results(&mut self, inst: Inst, types: &[Type]) -> ValueList {
        let values: Vec<Value> = types
            .iter()
            .map(|ty| {
                self.values.push(ValueData {
                    ty: *ty,
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

    /// 获取指令的第一个结果值（如果存在）
    pub fn first_result(&self, inst: Inst) -> Option<Value> {
        self.inst_results(inst).first().copied()
    }

    pub(crate) fn move_result(&mut self, value: Value, to: Inst) {
        let ValueDef::Inst(from) = self.value_def(value) else {
            panic!("cannot move a block parameter");
        };
        if from == to {
            return;
        }
        let mut old = self.inst_results(from).to_vec();
        let index = old
            .iter()
            .position(|&v| v == value)
            .expect("result missing from definition");
        old.remove(index);
        let mut new = self.inst_results(to).to_vec();
        assert!(!new.contains(&value), "duplicate result");
        new.push(value);
        self.inst_results[from] = self.make_value_list(&old);
        self.inst_results[to] = self.make_value_list(&new);
        self.values[value].def = ValueDef::Inst(to);
    }

    /// 从切片创建 ValueList
    pub(crate) fn make_value_list(&mut self, values: &[Value]) -> ValueList {
        ValueList::from_slice(values, &mut self.value_list_pool)
    }

    pub fn append_block_param(&mut self, block: Block, ty: Type) -> Value {
        self.values.push(ValueData {
            ty,
            def: ValueDef::Param(block),
        })
    }

    pub fn opcode(&self, inst: Inst) -> crate::Opcode {
        self.instructions[inst].fields.opcode()
    }

    pub fn inst(&self, inst: Inst) -> InstructionView<'_> {
        let data = &self.instructions[inst];
        data.fields.view(self.operands.get(data.operands))
    }

    /// Copy an instruction into an independent draft without decoding its fields.
    pub fn draft(&self, inst: Inst) -> InstDraft {
        let data = &self.instructions[inst];
        InstDraft {
            fields: data.fields.clone(),
            operands: crate::inst::Arguments::from_slice(self.operands.get(data.operands)),
        }
    }

    pub fn instructions(&self) -> impl ExactSizeIterator<Item = (Inst, InstructionView<'_>)> {
        self.instructions
            .iter()
            .map(|(inst, _)| (inst, self.inst(inst)))
    }

    pub fn create_inst(&mut self, data: InstDraft) -> Inst {
        let InstDraft {
            fields,
            operands: values,
        } = data;
        let inst = self.instructions.push(StoredInst {
            fields,
            operands: OperandRange::default(),
        });
        self.instructions[inst].operands = self.operands.alloc(inst, &values);
        inst
    }

    pub fn value_type(&self, val: Value) -> Type {
        self.values[val].ty
    }

    pub fn values(&self) -> &PrimaryMap<Value, ValueData> {
        &self.values
    }

    /// Change a declared type without validating the instruction's contract.
    pub fn set_value_type(&mut self, value: Value, ty: Type) {
        self.values[value].ty = ty;
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
            match &self.inst(inst) {
                InstructionView::Iconst { value } => {
                    let val = *value as i64;
                    if ty == Type::I8 {
                        Some(Constant::I8(val as i8))
                    } else if ty == Type::I16 {
                        Some(Constant::I16(val as i16))
                    } else if ty == Type::I32 {
                        Some(Constant::I32(val as i32))
                    } else if ty == Type::I64 {
                        Some(Constant::I64(val))
                    } else {
                        None
                    }
                }
                InstructionView::Fconst { value } => {
                    if ty == Type::F32 {
                        Some(Constant::F32(f32::from_bits(*value as u32)))
                    } else if ty == Type::F64 {
                        Some(Constant::F64(f64::from_bits(*value)))
                    } else {
                        None
                    }
                }
                InstructionView::Bconst { value } => Some(Constant::Bool(*value)),
                _ => None,
            }
        } else {
            None
        }
    }

    pub fn remove_inst(&mut self, inst: Inst) {
        assert!(
            self.inst_results(inst).iter().all(|&v| self.use_empty(v)),
            "cannot erase a used definition"
        );
        self.clear_inst(inst);
    }

    fn clear_inst(&mut self, inst: Inst) {
        self.operands
            .release(core::mem::take(&mut self.instructions[inst].operands));
        self.instructions[inst].fields = InstFields::Nop;
        self.inst_results[inst] = ValueList::default();
    }

    /// Erase a closed set, including mutually dependent dead instructions.
    pub fn remove_insts(&mut self, insts: &[Inst]) {
        let dead: hashbrown::HashSet<_> = insts.iter().copied().collect();
        for &inst in insts {
            for &value in self.inst_results(inst) {
                assert!(
                    self.uses(value).all(|site| dead.contains(&site.inst())),
                    "cannot erase a live definition"
                );
            }
        }
        for &inst in insts {
            self.clear_inst(inst);
        }
    }

    /// Replace operand structure. Positions from the previous instruction expire.
    pub fn replace_inst(&mut self, inst: Inst, data: InstDraft) {
        let InstDraft {
            fields,
            operands: values,
        } = data;
        self.operands
            .release(core::mem::take(&mut self.instructions[inst].operands));
        let operands = self.operands.alloc(inst, &values);
        self.instructions[inst] = StoredInst { fields, operands };
    }
}

impl Default for DataFlowGraph {
    fn default() -> Self {
        Self::new()
    }
}
