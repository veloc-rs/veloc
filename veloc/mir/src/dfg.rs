use super::inst::{
    ConstantPoolData, ConstantPoolId, Inst, InstructionData, PtrIndexImm, PtrIndexImmId,
    VectorExtData, VectorExtId, VectorMemExtId, VectorMemOptions,
};
use crate::constant::Constant;
use crate::types::{
    Block, BlockCall, BlockCallData, JumpTable, JumpTableData, Type, Value, ValueData, ValueDef,
    ValueList, ValueListPool,
};
use alloc::string::String;
use alloc::vec::Vec;
use cranelift_entity::{PrimaryMap, SecondaryMap};
use hashbrown::HashMap;

mod pool;
pub use pool::PoolKey;

#[derive(Debug, Clone)]
pub struct DataFlowGraph {
    pub instructions: PrimaryMap<Inst, InstructionData>,
    pub values: PrimaryMap<Value, ValueData>,
    pub value_names: SecondaryMap<Value, String>,
    pub inst_results: SecondaryMap<Inst, ValueList>,
    pub(crate) value_list_pool: ValueListPool,
    pub block_calls: PrimaryMap<BlockCall, BlockCallData>,
    pub jump_tables: PrimaryMap<JumpTable, JumpTableData>,
    ptr_imm_pool: PrimaryMap<PtrIndexImmId, PtrIndexImm>,
    ptr_imm_map: HashMap<PtrIndexImm, PtrIndexImmId>,
    /// 向量操作扩展信息池 (mask, evl)
    vector_ext_pool: PrimaryMap<VectorExtId, VectorExtData>,
    vector_ext_map: HashMap<VectorExtData, VectorExtId>,
    /// 向量内存操作扩展配置池
    vector_mem_ext_pool: PrimaryMap<VectorMemExtId, VectorMemOptions>,
    vector_mem_ext_map: HashMap<VectorMemOptions, VectorMemExtId>,
    /// 常量数据池
    constant_pool: PrimaryMap<ConstantPoolId, ConstantPoolData>,
    constant_pool_map: HashMap<ConstantPoolData, ConstantPoolId>,
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
            vector_ext_pool: PrimaryMap::new(),
            vector_ext_map: HashMap::new(),
            vector_mem_ext_pool: PrimaryMap::new(),
            vector_mem_ext_map: HashMap::new(),
            constant_pool: PrimaryMap::new(),
            constant_pool_map: HashMap::new(),
        }
    }

    // ======================================
    // 向量操作辅助方法
    // ======================================

    /// 创建向量操作扩展信息
    pub fn make_vector_ext(&mut self, mask: Value, evl: Option<Value>) -> VectorExtId {
        self.intern_vector_ext(VectorExtData { mask, evl })
    }

    pub fn intern_vector_ext(&mut self, data: VectorExtData) -> VectorExtId {
        VectorExtId::insert(self, data)
    }

    /// 创建向量内存操作扩展配置
    pub fn make_vector_mem_ext(&mut self, data: VectorMemOptions) -> VectorMemExtId {
        VectorMemExtId::insert(self, data)
    }

    /// 向常量池添加数据
    pub fn make_constant_pool_data(&mut self, data: ConstantPoolData) -> ConstantPoolId {
        let ConstantPoolData::Bytes(bytes) = data;
        ConstantPoolId::insert(self, bytes)
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

    pub fn make_ptr_imm(&mut self, offset: i32, scale: u32) -> PtrIndexImmId {
        self.intern_ptr_imm(PtrIndexImm { offset, scale })
    }

    pub fn intern_ptr_imm(&mut self, key: PtrIndexImm) -> PtrIndexImmId {
        PtrIndexImmId::insert(self, key)
    }

    pub fn ptr_imm(&self, id: PtrIndexImmId) -> Option<&PtrIndexImm> {
        id.get(self)
    }

    pub fn vector_ext(&self, id: VectorExtId) -> Option<&VectorExtData> {
        id.get(self)
    }

    pub fn vector_mem_ext(&self, id: VectorMemExtId) -> Option<&VectorMemOptions> {
        id.get(self)
    }

    pub fn constant_pool_data(&self, id: ConstantPoolId) -> Option<&ConstantPoolData> {
        self.constant_pool.get(id)
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
            let ty = self.value_type(val);
            match &self.instructions[inst] {
                InstructionData::Iconst { value } => {
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

    pub fn make_block_call(&mut self, block: Block, args: &[Value]) -> BlockCall {
        let args = self.make_value_list(args);
        self.block_calls.push(BlockCallData { block, args })
    }

    /// Store case destinations followed by the required default destination.
    pub fn make_jump_table(&mut self, cases: &[BlockCall], default: BlockCall) -> JumpTable {
        let mut targets = Vec::with_capacity(cases.len() + 1);
        targets.extend_from_slice(cases);
        targets.push(default);
        self.jump_tables.push(JumpTableData { targets })
    }

    pub fn block_call_args(&self, call: BlockCall) -> &[Value] {
        self.block_calls[call].args.as_slice(&self.value_list_pool)
    }

    pub fn jump_table_targets(&self, table: JumpTable) -> &[BlockCall] {
        &self.jump_tables[table].targets
    }

    pub fn remove_inst(&mut self, inst: Inst) {
        self.instructions[inst] = InstructionData::Nop;
        self.inst_results[inst] = ValueList::default();
    }

    /// 替换指定指令的数据内容。
    pub fn replace_inst(&mut self, inst: Inst, data: InstructionData) {
        self.instructions[inst] = data;
    }

    /// 替换指令中使用的 Value。
    pub fn replace_value_in_inst(&mut self, inst: Inst, old_val: Value, new_val: Value) {
        let mut data = core::mem::replace(&mut self.instructions[inst], InstructionData::Nop);
        data.replace_value(self, old_val, new_val);
        self.instructions[inst] = data;
    }

    // ======================================
    // 操作数访问与替换 Helper 方法
    // ======================================

    /// 遍历 Value 列表
    pub fn visit_value_list<F>(&self, list: ValueList, mut f: F)
    where
        F: FnMut(Value),
    {
        for &v in self.get_value_list(list) {
            f(v);
        }
    }

    /// 替换 Value 列表中的值
    pub fn replace_value_list(&mut self, list: &mut ValueList, old: Value, new: Value) {
        let mut values = list.as_slice(&self.value_list_pool).to_vec();
        let mut changed = false;
        for v in &mut values {
            if *v == old {
                *v = new;
                changed = true;
            }
        }
        if changed {
            *list = ValueList::from_slice(&values, &mut self.value_list_pool);
        }
    }

    /// 遍历 Block 调用的参数
    pub fn visit_block_call<F>(&self, call: BlockCall, f: F)
    where
        F: FnMut(Value),
    {
        self.visit_value_list(self.block_calls[call].args, f);
    }

    /// 替换 Block 调用中的参数
    pub fn replace_block_call(&mut self, call: BlockCall, old: Value, new_val: Value) {
        let mut args = self.block_calls[call].args;
        self.replace_value_list(&mut args, old, new_val);
        self.block_calls[call].args = args;
    }

    /// 遍历跳转表中的所有参数
    pub fn visit_jump_table<F>(&self, table: JumpTable, mut f: F)
    where
        F: FnMut(Value),
    {
        for &dest in &self.jump_tables[table].targets {
            self.visit_block_call(dest, &mut f);
        }
    }

    /// 替换跳转表中的所有参数
    pub fn replace_jump_table(&mut self, table: JumpTable, old: Value, new_val: Value) {
        let targets = self.jump_tables[table].targets.clone();
        for dest in targets {
            self.replace_block_call(dest, old, new_val);
        }
    }

    /// 遍历向量扩展信息 (mask, evl)
    pub fn visit_vector_ext<F>(&self, ext: VectorExtId, mut f: F)
    where
        F: FnMut(Value),
    {
        let data = self
            .vector_ext(ext)
            .expect("instruction refers to a missing vector extension");
        f(data.mask);
        if let Some(evl) = data.evl {
            f(evl);
        }
    }

    /// 替换向量扩展信息中的值 (并处理 interning)
    pub fn replace_vector_ext(&mut self, ext: &mut VectorExtId, old: Value, new: Value) {
        let mut data = *self
            .vector_ext(*ext)
            .expect("instruction refers to a missing vector extension");
        let mut changed = false;
        if data.mask == old {
            data.mask = new;
            changed = true;
        }
        if let Some(ref mut evl) = data.evl
            && *evl == old
        {
            *evl = new;
            changed = true;
        }
        if changed {
            *ext = self.make_vector_ext(data.mask, data.evl);
        }
    }

    /// 遍历向量内存扩展配置中的值
    pub fn visit_vector_mem_ext<F>(&self, ext: VectorMemExtId, mut f: F)
    where
        F: FnMut(Value),
    {
        let data = self
            .vector_mem_ext(ext)
            .expect("instruction refers to a missing vector memory extension");
        if let Some(mask) = data.mask {
            f(mask);
        }
        if let Some(evl) = data.evl {
            f(evl);
        }
    }

    /// 替换向量内存扩展配置中的值 (并处理 interning)
    pub fn replace_vector_mem_ext(&mut self, ext: &mut VectorMemExtId, old: Value, new: Value) {
        let mut data = *self
            .vector_mem_ext(*ext)
            .expect("instruction refers to a missing vector memory extension");
        let mut changed = false;
        if let Some(ref mut mask) = data.mask
            && *mask == old
        {
            *mask = new;
            changed = true;
        }
        if let Some(ref mut evl) = data.evl
            && *evl == old
        {
            *evl = new;
            changed = true;
        }
        if changed {
            *ext = self.make_vector_mem_ext(data);
        }
    }
}

impl Default for DataFlowGraph {
    fn default() -> Self {
        Self::new()
    }
}
