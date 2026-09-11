use super::inst::{
    ConstantPoolId, FieldPool, Inst, InstDraft, InstructionView, PackedFields, StoredInst,
};
use crate::constant::Constant;
use crate::types::{Block, Type, Value, ValueData, ValueDef, ValueList, ValueListPool};
use alloc::boxed::Box;
use alloc::sync::Arc;
use cranelift_entity::{PrimaryMap, SecondaryMap};
use hashbrown::HashMap;

mod operands;
mod pool;
pub(crate) use operands::OperandRange;
pub use operands::{Use, Uses};

#[derive(Debug, Clone)]
pub struct DataFlowGraph {
    instructions: PrimaryMap<Inst, StoredInst>,
    fields: FieldPool,
    pub(crate) values: PrimaryMap<Value, ValueData>,
    // Debug names are sparse metadata, not one String header per preceding value.
    value_names: HashMap<Value, Box<str>>,
    inst_results: SecondaryMap<Inst, ValueList>,
    value_list_pool: ValueListPool,
    operands: operands::Operands,
    /// Constant bytes are immutable and may be interned.
    constant_pool: PrimaryMap<ConstantPoolId, Arc<[u8]>>,
    constant_pool_map: HashMap<Arc<[u8]>, ConstantPoolId>,
}

impl DataFlowGraph {
    /// Finalize parser-local function identities without touching SSA operands.
    pub(crate) fn remap_functions(&mut self, map: &[crate::FuncId]) {
        for (_, inst) in &mut self.instructions {
            inst.fields
                .map_functions(&mut self.fields, |id| map[id.0 as usize]);
        }
    }

    pub fn new() -> Self {
        Self {
            instructions: PrimaryMap::new(),
            fields: FieldPool::default(),
            values: PrimaryMap::new(),
            value_names: HashMap::new(),
            inst_results: SecondaryMap::new(),
            value_list_pool: ValueListPool::new(),
            operands: operands::Operands::default(),
            constant_pool: PrimaryMap::new(),
            constant_pool_map: HashMap::new(),
        }
    }

    /// 为指令添加多个结果值（支持多返回值）
    pub fn append_results(&mut self, inst: Inst, types: &[Type]) -> ValueList {
        assert!(self.inst_results(inst).is_empty(), "results already bound");
        let values = types.iter().map(|&ty| {
            self.values.push(ValueData {
                ty,
                def: ValueDef::Inst(inst),
            })
        });
        let list = ValueList::from_iter(values, &mut self.value_list_pool);
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
        let index = self
            .inst_results(from)
            .iter()
            .position(|&v| v == value)
            .expect("result missing from definition");
        assert!(!self.inst_results(to).contains(&value), "duplicate result");
        self.inst_results[from].remove(index, &mut self.value_list_pool);
        self.inst_results[to].push(value, &mut self.value_list_pool);
        self.values[value].def = ValueDef::Inst(to);
    }

    pub fn append_block_param(&mut self, block: Block, ty: Type) -> Value {
        self.values.push(ValueData {
            ty,
            def: ValueDef::Param(block),
        })
    }

    pub fn opcode(&self, inst: Inst) -> crate::Opcode {
        self.instructions[inst].fields.opcode(&self.fields)
    }

    pub fn inst(&self, inst: Inst) -> InstructionView<'_> {
        let data = &self.instructions[inst];
        data.fields
            .view(self.operands.get(data.operands), &self.fields)
    }

    /// Decode persistent storage into an independently editable draft.
    pub fn draft(&self, inst: Inst) -> InstDraft {
        self.inst(inst).to_draft()
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
        let fields = fields.pack(&mut self.fields);
        let inst = self.instructions.push(StoredInst {
            fields,
            operands: OperandRange::default(),
        });
        self.instructions[inst].operands = self.operands.alloc(inst, &values);
        inst
    }

    /// Bind reserved result IDs to an instruction, preserving existing uses.
    /// The caller must supply distinct, not-yet-defined values from this DFG.
    /// Result types and counts are intentionally not validated here.
    pub(crate) fn bind_results(&mut self, inst: Inst, results: &[(Value, Type)]) {
        assert!(self.inst_results(inst).is_empty(), "results already bound");
        let values = results.iter().map(|&(value, ty)| {
            self.values[value] = ValueData {
                ty,
                def: ValueDef::Inst(inst),
            };
            value
        });
        self.inst_results[inst] = ValueList::from_iter(values, &mut self.value_list_pool);
    }

    pub fn value_type(&self, val: Value) -> Type {
        self.values[val].ty
    }

    pub fn value_name(&self, value: Value) -> &str {
        self.value_names.get(&value).map_or("", AsRef::as_ref)
    }

    pub fn set_value_name(&mut self, value: Value, name: &str) {
        if name.is_empty() {
            self.value_names.remove(&value);
        } else {
            self.value_names.insert(value, name.into());
        }
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

    /// Read a scalar literal without traversing expression graphs.
    pub fn as_scalar_const(&self, val: Value) -> Option<crate::ScalarConst> {
        let inst = self.value_inst(val)?;
        let value = match self.inst(inst) {
            InstructionView::Iconst { value } => value.into(),
            InstructionView::Fconst { value } => value.into(),
            InstructionView::Bconst { value } => crate::ScalarConst::from(value),
            _ => return None,
        };
        (self.value_type(val) == value.ty()).then_some(value)
    }

    pub fn as_const(&self, val: Value) -> Option<Constant> {
        if let Some(value) = self.as_scalar_const(val) {
            return Some(value.into());
        }
        let ty = self.value_type(val);
        match self.inst(self.value_inst(val)?) {
            InstructionView::Vconst { value } => (value.ty() == ty).then(|| value.into()),
            InstructionView::Unary {
                opcode: crate::Opcode::Splat,
                arg,
            } => {
                let vector = ty.as_vector()?;
                let scalar = self.as_scalar_const(arg)?;
                if scalar.ty() != vector.element_type().as_type() {
                    return None;
                }
                crate::VectorConst::splat(scalar, vector.lane_count(), vector.is_scalable())
                    .map(Into::into)
            }
            _ => None,
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
        self.instructions[inst].fields.release(&mut self.fields);
        self.instructions[inst].fields = PackedFields::Nop;
        self.inst_results[inst].clear(&mut self.value_list_pool);
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
        self.instructions[inst].fields.release(&mut self.fields);
        let fields = fields.pack(&mut self.fields);
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
