use super::inst::{FieldPool, Inst, InstFields, InstView, InstWriter, StoredInst};
use crate::constant::Constant;
use crate::types::{Block, Type, Value, ValueData, ValueDef, ValueList, ValueListPool};
use alloc::boxed::Box;
use cranelift_entity::{PrimaryMap, SecondaryMap};
use hashbrown::HashMap;

mod operands;
mod pool;
pub(crate) use operands::OperandRange;
pub use operands::{Use, Uses};

#[derive(Debug, Clone, Default)]
pub struct BlockData {
    pub params: alloc::vec::Vec<Value>,
}

#[derive(Debug, Clone)]
pub struct DataFlowGraph {
    pub(crate) blocks: PrimaryMap<crate::Block, BlockData>,
    pub(crate) instructions: PrimaryMap<Inst, StoredInst>,
    pub(crate) fields: FieldPool,
    pub(crate) values: PrimaryMap<Value, ValueData>,
    // Debug names are sparse metadata, not one String header per preceding value.
    value_names: HashMap<Value, Box<str>>,
    inst_results: SecondaryMap<Inst, ValueList>,
    value_list_pool: ValueListPool,
    operands: operands::Operands,
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
            blocks: PrimaryMap::new(),
            instructions: PrimaryMap::new(),
            fields: FieldPool::default(),
            values: PrimaryMap::new(),
            value_names: HashMap::new(),
            inst_results: SecondaryMap::new(),
            value_list_pool: ValueListPool::new(),
            operands: operands::Operands::default(),
        }
    }

    /// 为指令添加多个结果值（支持多返回值）
    pub(crate) fn append_results(&mut self, inst: Inst, types: &[Type]) -> ValueList {
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

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn block_params(&self, block: Block) -> &[Value] {
        &self.blocks[block].params
    }

    pub fn inst_count(&self) -> usize {
        self.instructions.len()
    }

    pub fn create_block(&mut self) -> crate::Block {
        self.blocks.push(BlockData::default())
    }

    pub fn append_block_param(&mut self, block: Block, ty: Type) -> Value {
        let value = self.values.push(ValueData {
            ty,
            def: ValueDef::Param(block),
        });
        self.blocks[block].params.push(value);
        value
    }

    pub fn opcode(&self, inst: Inst) -> crate::Opcode {
        self.instructions[inst].fields.opcode(&self.fields)
    }

    pub fn inst(&self, inst: Inst) -> InstView<'_> {
        let data = &self.instructions[inst];
        data.fields
            .view(self.operands.get(data.operands), &self.fields)
    }

    pub fn instructions(&self) -> impl ExactSizeIterator<Item = (Inst, InstView<'_>)> {
        self.instructions
            .iter()
            .map(|(inst, _)| (inst, self.inst(inst)))
    }

    pub fn writer(&mut self) -> InstWriter<'_> {
        InstWriter {
            dfg: self,
            target: None,
        }
    }

    /// Internal construction primitive. Function-level insertion belongs to
    /// `FuncEditor`, which also updates layout and CFG state.
    pub(crate) fn create_inst(&mut self, build: impl FnOnce(InstWriter<'_>) -> Inst) -> Inst {
        build(self.writer())
    }

    pub(crate) fn write_inst(
        &mut self,
        target: Option<Inst>,
        fields: InstFields,
        values: &[Value],
    ) -> Inst {
        let inst = if let Some(inst) = target {
            self.instructions[inst].fields.release(&mut self.fields);
            self.operands
                .release(core::mem::take(&mut self.instructions[inst].operands));
            self.instructions[inst].fields = fields;
            inst
        } else {
            self.instructions.push(StoredInst {
                fields,
                operands: OperandRange::default(),
            })
        };
        self.instructions[inst].operands = self.operands.alloc(inst, values);
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
    /// Construction/parser escape hatch. Normal transformations must preserve
    /// the instruction contract and use a typed editor operation instead.
    pub(crate) fn set_value_type(&mut self, value: Value, ty: Type) {
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
            InstView::Iconst { value } => value.into(),
            InstView::Fconst { value } => value.into(),
            InstView::Bconst { value } => crate::ScalarConst::from(value),
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
            InstView::Vconst { value } => (value.ty() == ty).then(|| value.into()),
            InstView::Unary {
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

    fn clear_inst(&mut self, inst: Inst) {
        self.operands
            .release(core::mem::take(&mut self.instructions[inst].operands));
        self.instructions[inst].fields.release(&mut self.fields);
        self.instructions[inst].fields = InstFields::Nop;
        self.inst_results[inst].clear(&mut self.value_list_pool);
    }

    /// Erase a closed set, including mutually dependent dead instructions.
    pub(crate) fn remove_insts(&mut self, insts: &[Inst]) {
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
    /// Low-level replacement primitive; `FuncEditor::replace_inst` also keeps
    /// the surrounding CFG synchronized.
    pub(crate) fn replace_inst(&mut self, inst: Inst, build: impl FnOnce(InstWriter<'_>) -> Inst) {
        let result = build(InstWriter {
            dfg: self,
            target: Some(inst),
        });
        assert_eq!(
            result, inst,
            "replacement must write the selected instruction"
        );
    }
}

impl Default for DataFlowGraph {
    fn default() -> Self {
        Self::new()
    }
}
