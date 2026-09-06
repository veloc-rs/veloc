use super::function::{Function, StackSlotData};
use super::inst::{ConstantPoolData, Inst, InstructionData, VectorMemOptions};
use super::opcode::Opcode;
use super::types::{
    Block, BlockCall, FuncId, Signature, StackSlot, Type, Value, ValueList, Variable,
};
use crate::opspec::ResultTypes;
use crate::types::JumpTableData;
use crate::{CallConv, Intrinsic, Linkage, Module, ModuleData, Result, SigId};
use alloc::vec::Vec;
use hashbrown::HashMap;
use smallvec::SmallVec;

pub struct ModuleBuilder {
    data: ModuleData,
}

impl ModuleBuilder {
    pub fn new() -> Self {
        Self {
            data: ModuleData::default(),
        }
    }

    pub fn declare_function(&mut self, name: String, sig_id: SigId, linkage: Linkage) -> FuncId {
        self.data.declare_function(name, sig_id, linkage)
    }

    pub fn make_signature(
        &mut self,
        params: Vec<Type>,
        ret: Vec<Type>,
        call_conv: CallConv,
    ) -> SigId {
        let sig = Signature::new(params, ret, call_conv);
        self.data.intern_signature(sig)
    }

    pub fn get_func_id(&self, name: &str) -> Option<FuncId> {
        self.data.get_func_id(name)
    }

    pub fn builder(&mut self, func_id: FuncId) -> FunctionBuilder<'_> {
        FunctionBuilder::new(&mut self.data, func_id)
    }

    pub fn add_global(&mut self, name: String, ty: Type, linkage: Linkage) {
        self.data.add_global(name, ty, linkage);
    }

    pub fn validate(&self) -> Result<()> {
        self.data.validate()
    }

    pub fn build(self) -> Module {
        Module::new(self.data)
    }

    pub fn build_data(self) -> ModuleData {
        self.data
    }
}

impl Default for ModuleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct FunctionBuilder<'a> {
    module: &'a mut ModuleData,
    func_id: FuncId,
    current_block: Option<Block>,
    // 变量的类型映射
    var_types: HashMap<Variable, Type>,
    // 每个 Block 对变量的最新定义: Block -> Variable -> Value
    def_map: HashMap<Block, HashMap<Variable, Value>>,
    // 未密封 Block 中待处理的 Phi 节点: Block -> Variable -> Phi Value
    incomplete_phis: HashMap<Block, Vec<(Variable, Value)>>,
}

impl<'a> FunctionBuilder<'a> {
    pub(crate) fn new(module: &'a mut ModuleData, func_id: FuncId) -> Self {
        let mut builder = Self {
            module,
            func_id,
            current_block: None,
            var_types: HashMap::new(),
            def_map: HashMap::new(),
            incomplete_phis: HashMap::new(),
        };

        if let Some(entry) = builder.func().entry_block {
            builder.current_block = Some(entry);
        }

        builder
    }

    pub fn init_entry_block(&mut self) -> Block {
        let entry = self.create_block();
        self.switch_to_block(entry);
        self.seal_block(entry);

        let sig_id = self.func().signature;
        let sig = self.module.signatures[sig_id].clone();
        for ty in sig.params.iter().cloned() {
            self.add_block_param(entry, ty);
        }
        entry
    }

    pub fn current_block(&self) -> Option<Block> {
        self.current_block
    }

    pub fn func(&self) -> &Function {
        &self.module.functions[self.func_id]
    }

    pub fn func_mut(&mut self) -> &mut Function {
        &mut self.module.functions[self.func_id]
    }

    pub fn func_signature(&self, func_id: FuncId) -> SigId {
        self.module.functions[func_id].signature
    }

    pub fn signature(&self, sig_id: SigId) -> &Signature {
        &self.module.signatures[sig_id]
    }

    /// Compute the result types of an instruction.
    /// Uses SmallVec to avoid heap allocation for most instructions (0-2 results).
    fn inst_result_types(&self, data: &InstructionData) -> SmallVec<[Type; 2]> {
        let spec = data.opcode().spec();
        let mut operands = SmallVec::<[Type; 4]>::new();
        data.visit_type_operands(&self.func().dfg, |value| {
            operands.push(self.value_type(value));
        });

        match spec
            .type_scheme
            .infer_results(&operands)
            .unwrap_or_else(|error| {
                panic!(
                    "{} constructed with invalid operand types: {error:?}",
                    spec.mnemonic
                )
            }) {
            ResultTypes::Inferred(types) => types,
            ResultTypes::Explicit => {
                panic!("{} requires an explicit result type", spec.mnemonic)
            }
            ResultTypes::Signature => {
                let sig_id = data
                    .call_info()
                    .expect("signature results require call metadata")
                    .signature
                    .resolve(self.module)
                    .expect("call refers to a missing function or signature");
                self.module.signatures[sig_id]
                    .returns
                    .iter()
                    .copied()
                    .collect()
            }
        }
    }

    fn assert_inst_matches_spec(&self, data: &InstructionData, results: &[Type]) {
        let spec = data.opcode().spec();
        assert!(
            data.matches_format(&self.func().dfg, spec.format),
            "{} constructed with the wrong instruction format",
            spec.mnemonic
        );
        let mut operands = SmallVec::<[Type; 4]>::new();
        data.visit_type_operands(&self.func().dfg, |value| {
            operands.push(self.value_type(value));
        });
        assert!(
            spec.type_scheme.validate(&operands, results).is_ok(),
            "{} constructed with types outside its type scheme",
            spec.mnemonic
        );
    }

    fn push_inst(&mut self, block: Block, data: InstructionData) -> Option<Value> {
        let inst = self.push_inst_raw(block, data);
        self.func().dfg.first_result(inst)
    }

    fn push_inst_raw(&mut self, block: Block, data: InstructionData) -> Inst {
        let types = self.inst_result_types(&data);
        self.append_inst(block, data, &types)
    }

    fn append_inst(&mut self, block: Block, data: InstructionData, types: &[Type]) -> Inst {
        self.assert_inst_matches_spec(&data, types);
        let func = self.func_mut();
        let (dfg, layout) = (&func.dfg, &mut func.layout);
        data.visit_successors(dfg, |call| {
            layout.add_edge(block, dfg.block_call_block(call))
        });
        let inst = self.func_mut().dfg.instructions.push(data);
        self.func_mut().layout.append_inst(block, inst);
        if !types.is_empty() {
            self.func_mut().dfg.append_results(inst, types);
        }
        inst
    }

    /// Push an instruction whose result type is selected by the caller.
    fn push_inst_with_type(&mut self, block: Block, data: InstructionData, ty: Type) -> Value {
        let inst = self.append_inst(block, data, &[ty]);
        self.func()
            .dfg
            .first_result(inst)
            .expect("an explicitly typed instruction must produce one result")
    }

    pub fn make_value_list(&mut self, values: &[Value]) -> ValueList {
        self.func_mut().dfg.make_value_list(values)
    }

    pub fn make_block_call(&mut self, block: Block, args: &[Value]) -> BlockCall {
        self.func_mut().dfg.make_block_call(block, args)
    }

    pub fn create_block(&mut self) -> Block {
        self.func_mut().layout.create_block()
    }

    pub fn create_stack_slot(&mut self, size: u32) -> StackSlot {
        self.func_mut().stack_slots.push(StackSlotData { size })
    }

    pub fn switch_to_block(&mut self, block: Block) {
        if !self.func().layout.block_order.contains(&block) {
            self.func_mut().layout.append_block(block);
        }
        self.current_block = Some(block);
        if !self.func().is_defined() {
            self.func_mut().entry_block = Some(block);
        }
    }

    pub fn block_params(&self, block: Block) -> &[Value] {
        &self.func().layout.blocks[block].params
    }

    pub fn value_type(&self, val: Value) -> Type {
        self.func().dfg.value_type(val)
    }

    pub fn set_value_name(&mut self, val: Value, name: &str) {
        self.func_mut().dfg.value_names[val] = name.to_string();
    }

    pub fn add_block_param(&mut self, block: Block, ty: Type) -> Value {
        let val = self.func_mut().dfg.append_block_param(block, ty);
        self.func_mut().layout.blocks[block].params.push(val);
        val
    }

    pub fn func_params(&self) -> &[Value] {
        if let Some(entry) = self.func().entry_block {
            self.block_params(entry)
        } else {
            &[]
        }
    }

    pub fn func_param(&self, index: usize) -> Value {
        self.func_params()[index]
    }

    pub fn ins(&mut self) -> InstBuilder<'_, 'a> {
        InstBuilder { builder: self }
    }

    pub fn in_new_block<F>(&mut self, f: F) -> Block
    where
        F: FnOnce(&mut InstBuilder<'_, 'a>),
    {
        let block = self.create_block();
        self.switch_to_block(block);
        let mut ins = self.ins();
        f(&mut ins);
        block
    }

    pub fn is_current_block_terminated(&self) -> bool {
        let block = self.current_block.expect("No current block");
        if let Some(&last_inst) = self.func().layout.blocks[block].insts.last() {
            self.func().dfg.inst(last_inst).is_terminator()
        } else {
            false
        }
    }

    pub fn if_else<T, E>(&mut self, condition: Value, then_body: T, else_body: E)
    where
        T: FnOnce(&mut FunctionBuilder),
        E: FnOnce(&mut FunctionBuilder),
    {
        let then_block = self.create_block();
        let else_block = self.create_block();
        let merge_block = self.create_block();

        // Entry
        self.ins().br(condition, then_block, &[], else_block, &[]);

        // 密封 then/else block，因为它们的前驱（当前 block）已经确定
        self.seal_block(then_block);
        self.seal_block(else_block);

        // Then 路径
        self.switch_to_block(then_block);
        then_body(self);
        if !self.is_current_block_terminated() {
            self.ins().jump(merge_block, &[]);
        }

        // Else 路径
        self.switch_to_block(else_block);
        else_body(self);
        if !self.is_current_block_terminated() {
            self.ins().jump(merge_block, &[]);
        }

        // 汇合点
        self.switch_to_block(merge_block);
        // 密封 merge block，因为 then 和 else 路径都已经处理完毕
        self.seal_block(merge_block);
    }

    pub fn while_loop<C, B>(&mut self, cond_body: C, loop_body: B)
    where
        C: FnOnce(&mut FunctionBuilder) -> Value,
        B: FnOnce(&mut FunctionBuilder),
    {
        let header_block = self.create_block();
        let body_block = self.create_block();
        let exit_block = self.create_block();

        // 1. 进入循环头
        self.ins().jump(header_block, &[]);

        // 2. 循环头 (Header): 判断条件
        // 注意：Header 不能立即密封，因为它有一个来自循环体底部的回边
        self.switch_to_block(header_block);
        let condition = cond_body(self);
        self.ins().br(condition, body_block, &[], exit_block, &[]);

        // 3. 循环体 (Body)
        // Body 的前驱只有 Header，已知且唯一，可以密封
        self.seal_block(body_block);
        self.switch_to_block(body_block);
        loop_body(self);
        if !self.is_current_block_terminated() {
            self.ins().jump(header_block, &[]);
        }

        // 4. 密封 Header: 此时 Entry -> Header 和 Body -> Header 两个边都已建立
        self.seal_block(header_block);

        // 5. 退出循环
        self.switch_to_block(exit_block);
        self.seal_block(exit_block);
    }

    pub fn declare_var(&mut self, var: Variable, ty: Type) {
        self.var_types.insert(var, ty);
    }

    pub fn def_var(&mut self, var: Variable, val: Value) {
        let block = self.current_block.expect("No current block");
        self.def_map.entry(block).or_default().insert(var, val);
    }

    pub fn use_var(&mut self, var: Variable) -> Value {
        let block = self.current_block.expect("No current block");
        self.use_var_on_block(block, var)
    }

    fn use_var_on_block(&mut self, block: Block, var: Variable) -> Value {
        if let Some(val) = self.def_map.get(&block).and_then(|m| m.get(&var)) {
            *val
        } else {
            self.use_var_recursive(block, var)
        }
    }

    fn use_var_recursive(&mut self, block: Block, var: Variable) -> Value {
        let val;
        if !self.func().layout.blocks[block].is_sealed {
            // Incomplete phi
            let ty = self.var_types[&var];
            val = self.add_block_param(block, ty);
            self.incomplete_phis
                .entry(block)
                .or_default()
                .push((var, val));
        } else {
            let preds = self.func().layout.blocks[block].preds.clone();
            if preds.len() == 1 {
                val = self.use_var_on_block(preds[0], var);
            } else {
                let ty = self.var_types[&var];
                val = self.add_block_param(block, ty);
                // Break recursion
                self.def_map.entry(block).or_default().insert(var, val);
                self.add_phi_operands(block, var, val);
            }
        }
        self.def_map.entry(block).or_default().insert(var, val);
        val
    }

    fn add_phi_operands(&mut self, block: Block, var: Variable, phi: Value) {
        let index = self.func().layout.blocks[block]
            .params
            .iter()
            .position(|&v| v == phi)
            .expect("Phi not found in block params");
        let preds = self.func().layout.blocks[block].preds.clone();
        for p in preds {
            let val = self.use_var_on_block(p, var);
            self.add_block_param_to_jump(p, block, index, val);
        }
    }

    pub fn seal_block(&mut self, block: Block) {
        if self.func().layout.blocks[block].is_sealed {
            return;
        }
        if let Some(phis) = self.incomplete_phis.remove(&block) {
            for (var, phi) in phis {
                self.add_phi_operands(block, var, phi);
            }
        }
        self.func_mut().layout.blocks[block].is_sealed = true;
    }

    pub fn seal_all_blocks(&mut self) {
        let blocks = self.func().layout.block_order.clone();
        for block in blocks {
            self.seal_block(block);
        }
    }

    fn add_block_param_to_jump(&mut self, pred: Block, target: Block, index: usize, val: Value) {
        if let Some(&last_inst) = self.func().layout.blocks[pred].insts.last() {
            let dfg = &mut self.module.functions[self.func_id].dfg;
            let idata = dfg.inst(last_inst).clone();

            match idata {
                InstructionData::Jump { mut dest } => {
                    let mut dest_data = dfg.block_calls[dest];
                    if dest_data.block == target {
                        let mut vec = dfg.block_call_args(dest).to_vec();
                        if index >= vec.len() {
                            vec.resize(index + 1, val);
                        } else {
                            vec[index] = val;
                        }
                        dest_data.args = dfg.make_value_list(&vec);
                        dest = dfg.block_calls.push(dest_data);
                        *dfg.inst_mut(last_inst) = InstructionData::Jump { dest };
                    }
                }
                InstructionData::Br {
                    condition,
                    mut then_dest,
                    mut else_dest,
                } => {
                    let mut changed = false;
                    let mut then_data = dfg.block_calls[then_dest];
                    if then_data.block == target {
                        let mut vec = dfg.block_call_args(then_dest).to_vec();
                        if index >= vec.len() {
                            vec.resize(index + 1, val);
                        } else {
                            vec[index] = val;
                        }
                        then_data.args = dfg.make_value_list(&vec);
                        then_dest = dfg.block_calls.push(then_data);
                        changed = true;
                    }
                    let mut else_data = dfg.block_calls[else_dest];
                    if else_data.block == target {
                        let mut vec = dfg.block_call_args(else_dest).to_vec();
                        if index >= vec.len() {
                            vec.resize(index + 1, val);
                        } else {
                            vec[index] = val;
                        }
                        else_data.args = dfg.make_value_list(&vec);
                        else_dest = dfg.block_calls.push(else_data);
                        changed = true;
                    }

                    if changed {
                        *dfg.inst_mut(last_inst) = InstructionData::Br {
                            condition,
                            then_dest,
                            else_dest,
                        };
                    }
                }
                InstructionData::BrTable {
                    index: idx_val,
                    table,
                } => {
                    let mut targets_data = dfg.jump_table_targets(table).to_vec();
                    let mut changed = false;
                    for target_call in targets_data.iter_mut() {
                        let mut dest_data = dfg.block_calls[*target_call];
                        if dest_data.block == target {
                            let mut vec = dfg.block_call_args(*target_call).to_vec();
                            if index >= vec.len() {
                                vec.resize(index + 1, val);
                            } else {
                                vec[index] = val;
                            }
                            dest_data.args = dfg.make_value_list(&vec);
                            *target_call = dfg.block_calls.push(dest_data);
                            changed = true;
                        }
                    }

                    if changed {
                        let new_table = dfg.jump_tables.push(JumpTableData {
                            targets: targets_data,
                        });
                        *dfg.inst_mut(last_inst) = InstructionData::BrTable {
                            index: idx_val,
                            table: new_table,
                        };
                    }
                }
                _ => {}
            }
        }
    }
}

pub struct InstBuilder<'b, 'a> {
    builder: &'b mut FunctionBuilder<'a>,
}

impl<'b, 'a> InstBuilder<'b, 'a> {
    pub fn block(&self) -> Block {
        self.builder.current_block.expect("No current block")
    }

    pub fn builder(&mut self) -> &mut FunctionBuilder<'a> {
        self.builder
    }

    pub fn param(&self, index: usize) -> Value {
        self.builder.func_param(index)
    }

    pub fn params(&self) -> &[Value] {
        self.builder.func_params()
    }

    pub fn value_type(&self, val: Value) -> Type {
        self.builder.value_type(val)
    }

    pub(crate) fn push(&mut self, data: InstructionData) -> Option<Value> {
        let block = self.block();
        self.builder.push_inst(block, data)
    }

    pub(crate) fn push_with_type(&mut self, data: InstructionData, ty: Type) -> Value {
        let block = self.block();
        self.builder.push_inst_with_type(block, data, ty)
    }

    pub(crate) fn push_raw(&mut self, data: InstructionData) -> Inst {
        let block = self.block();
        self.builder.push_inst_raw(block, data)
    }

    pub(crate) fn result_pair(&self, inst: Inst) -> (Value, Value) {
        let results = self.builder.func().dfg.inst_results(inst);
        (results[0], results[1])
    }

    pub fn i32const(&mut self, val: i32) -> Value {
        self.iconst(val as i64 as u64, Type::I32)
    }

    pub fn i64const(&mut self, val: i64) -> Value {
        self.iconst(val as u64, Type::I64)
    }

    pub fn f32const(&mut self, val: f32) -> Value {
        self.fconst(val.to_bits() as u64, Type::F32)
    }

    pub fn f64const(&mut self, val: f64) -> Value {
        self.fconst(val.to_bits(), Type::F64)
    }

    pub fn vconst(&mut self, ty: Type, data: Vec<u8>) -> Value {
        let pool_id = self
            .builder
            .func_mut()
            .dfg
            .make_constant_pool_data(ConstantPoolData::Bytes(data));
        self.push_with_type(InstructionData::Vconst { pool_id }, ty)
    }

    pub fn i8x16const(&mut self, values: [i8; 16]) -> Value {
        let data = values.iter().map(|&v| v as u8).collect();
        self.vconst(Type::I8X16, data)
    }

    pub fn i16x8const(&mut self, values: [i16; 8]) -> Value {
        let mut data = Vec::with_capacity(16);
        for &v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        self.vconst(Type::I16X8, data)
    }

    pub fn i32x4const(&mut self, values: [i32; 4]) -> Value {
        let mut data = Vec::with_capacity(16);
        for &v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        self.vconst(Type::I32X4, data)
    }

    pub fn i64x2const(&mut self, values: [i64; 2]) -> Value {
        let mut data = Vec::with_capacity(16);
        for &v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        self.vconst(Type::I64X2, data)
    }

    pub fn f32x4const(&mut self, values: [f32; 4]) -> Value {
        let mut data = Vec::with_capacity(16);
        for &v in &values {
            data.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        self.vconst(Type::F32X4, data)
    }

    pub fn f64x2const(&mut self, values: [f64; 2]) -> Value {
        let mut data = Vec::with_capacity(16);
        for &v in &values {
            data.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        self.vconst(Type::F64X2, data)
    }

    pub fn ptr_index(&mut self, ptr: Value, index: Value, scale: u32, offset: i32) -> Value {
        let imm_id = self.builder.func_mut().dfg.make_ptr_imm(offset, scale);
        self.push(InstructionData::PtrIndex { ptr, index, imm_id })
            .unwrap()
    }

    pub fn call(&mut self, func_id: FuncId, args: &[Value]) -> Inst {
        let args = self.builder.make_value_list(args);
        self.push_raw(InstructionData::Call { func_id, args })
    }

    pub fn call_indirect(&mut self, sig_id: SigId, ptr: Value, args: &[Value]) -> Inst {
        let args = self.builder.make_value_list(args);
        self.push_raw(InstructionData::CallIndirect { ptr, args, sig_id })
    }

    pub fn jump(&mut self, destination: Block, args: &[Value]) {
        let dest = self.builder.make_block_call(destination, args);
        self.push(InstructionData::Jump { dest });
    }

    pub fn br(
        &mut self,
        condition: Value,
        then_block: Block,
        then_args: &[Value],
        else_block: Block,
        else_args: &[Value],
    ) {
        debug_assert_eq!(
            self.builder.value_type(condition),
            Type::BOOL,
            "Condition for br must be a bool"
        );
        let then_dest = self.builder.make_block_call(then_block, then_args);
        let else_dest = self.builder.make_block_call(else_block, else_args);
        self.push(InstructionData::Br {
            condition,
            then_dest,
            else_dest,
        });
    }

    pub fn br_table(&mut self, index: Value, default_call: BlockCall, targets: &[BlockCall]) {
        debug_assert_eq!(
            self.builder.value_type(index),
            Type::I32,
            "Index for br_table must be an i32"
        );
        let table = self
            .builder
            .func_mut()
            .dfg
            .make_jump_table(targets, default_call);
        self.push(InstructionData::BrTable { index, table });
    }

    pub fn ret(&mut self, values: &[Value]) {
        let value_list = self.builder.make_value_list(values);
        self.push(InstructionData::Return { values: value_list });
    }

    /// Call an intrinsic function.
    /// Returns the instruction handle, use `dfg.inst_results(inst)` to get return values.
    pub fn call_intrinsic(&mut self, intrinsic: Intrinsic, sig_id: SigId, args: &[Value]) -> Inst {
        let args = self.builder.make_value_list(args);
        self.push_raw(InstructionData::CallIntrinsic {
            intrinsic,
            args,
            sig_id,
        })
    }

    // ======================================
    // 向量操作构建方法
    // ======================================

    /// 向量重排/混洗 (Shuffle)
    /// 根据掩码从两个输入向量中选择元素
    pub fn shuffle(&mut self, v1: Value, v2: Value, mask_id: crate::inst::ConstantPoolId) -> Value {
        self.push(InstructionData::Shuffle {
            args: [v1, v2],
            mask: mask_id,
        })
        .unwrap()
    }

    /// 插入标量到向量的指定通道
    pub fn insert_element(&mut self, vector: Value, scalar: Value, lane_index: u32) -> Value {
        let lane_val = self.i32const(lane_index as i32);
        self.insertelement(vector, scalar, lane_val)
    }

    /// 从向量提取指定通道的标量
    pub fn extract_element(&mut self, vector: Value, lane_index: u32) -> Value {
        let lane_val = self.i32const(lane_index as i32);
        self.extractelement(vector, lane_val)
    }

    /// 带扩展信息的向量操作 (用于 RISC-V V / AVX-512 带 Mask/EVL)
    ///
    /// # Arguments
    /// * `opcode` - 操作码 (如 IAdd, FMul 等)
    /// * `args` - 输入参数
    /// * `mask` - 谓词/掩码 (boolean vector)
    /// * `evl` - 显式向量长度 (i32), None 表示使用默认 VL
    /// * `result_ty` - 结果类型
    pub fn vector_op_ext(
        &mut self,
        opcode: Opcode,
        args: &[Value],
        mask: Value,
        evl: Option<Value>,
        result_ty: Type,
    ) -> Value {
        let args_list = self.builder.make_value_list(args);
        let ext_id = self.builder.func_mut().dfg.make_vector_ext(mask, evl);

        self.push_with_type(
            InstructionData::VectorOpWithExt {
                opcode,
                args: args_list,
                ext: ext_id,
            },
            result_ty,
        )
    }

    /// 固定步长向量加载 (Strided Load)
    pub fn load_stride(
        &mut self,
        ty: Type,
        ptr: Value,
        stride: Value,
        options: VectorMemOptions,
    ) -> Value {
        let ext = self.builder.func_mut().dfg.make_vector_mem_ext(options);

        self.push_with_type(InstructionData::VectorLoadStrided { ptr, stride, ext }, ty)
    }

    /// 固定步长向量存储 (Strided Store)
    pub fn store_stride(
        &mut self,
        value: Value,
        ptr: Value,
        stride: Value,
        options: VectorMemOptions,
    ) {
        let ext = self.builder.func_mut().dfg.make_vector_mem_ext(options);
        let args = self
            .builder
            .func_mut()
            .dfg
            .make_value_list(&[ptr, stride, value]);

        self.push(InstructionData::VectorStoreStrided { args, ext });
    }

    /// 离散向量加载 (Gather)
    /// base_ptr + index[i] * scale
    pub fn gather(
        &mut self,
        ty: Type,
        ptr: Value,
        index: Value,
        options: VectorMemOptions,
    ) -> Value {
        let ext = self.builder.func_mut().dfg.make_vector_mem_ext(options);

        self.push_with_type(InstructionData::VectorGather { ptr, index, ext }, ty)
    }

    /// 离散向量存储 (Scatter)
    pub fn scatter(&mut self, value: Value, ptr: Value, index: Value, options: VectorMemOptions) {
        let ext = self.builder.func_mut().dfg.make_vector_mem_ext(options);
        let args = self
            .builder
            .func_mut()
            .dfg
            .make_value_list(&[ptr, index, value]);

        self.push(InstructionData::VectorScatter { args, ext });
    }
}
