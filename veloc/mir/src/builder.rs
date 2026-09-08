use super::function::{Function, StackSlotData};
use super::inst::{Inst, InstDraft, VectorExtData};
use super::types::{Block, BlockCall, FuncId, Signature, StackSlot, Type, Value, Variable};
use crate::Opcode;
use crate::{CallConv, Intrinsic, Linkage, Module, ModuleData, Result, SigId};
use alloc::vec::Vec;
use hashbrown::HashMap;

include!(concat!(env!("OUT_DIR"), "/builders.rs"));

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

    pub fn make_block_call(&mut self, block: Block, args: &[Value]) -> BlockCall {
        BlockCall::new(block, args)
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
            self.func().dfg.opcode(last_inst).spec().is_terminator()
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
        let Some(&inst) = self.func().layout.blocks[pred].insts.last() else {
            return;
        };
        let mut data = self.func().dfg.draft(inst);
        if data.set_successor_arg(target, index, val) {
            self.func_mut().edit().replace_inst(inst, data);
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

    /// Insert an instruction with caller-supplied result types, without validation.
    /// Referenced storage and the current block must exist. Run the validator
    /// before passing untrusted or potentially invalid IR to later stages.
    pub fn insert(&mut self, data: InstDraft, types: &[Type]) -> Inst {
        let block = self.block();
        self.builder
            .func_mut()
            .edit()
            .append_inst(block, data, types)
    }

    /// Insert a fixed number of results without checking the type contract.
    fn emit<const N: usize>(&mut self, data: InstDraft, types: [Type; N]) -> [Value; N] {
        let inst = self.insert(data, &types);
        self.builder
            .func()
            .dfg
            .inst_results(inst)
            .try_into()
            .expect("insert must create one result per supplied type")
    }

    /// Resolve dynamic result types, such as a call's signature, before insertion.
    fn insert_inferred(&mut self, data: InstDraft) -> Inst {
        let types = data
            .result_types(&self.builder.func().dfg, self.builder.module, &[])
            .unwrap_or_else(|error| panic!("{}: {error}", data.opcode().spec().mnemonic));
        self.insert(data, &types)
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

    pub fn i8x16const(&mut self, values: [i8; 16]) -> Value {
        let data = values.iter().map(|&v| v as u8).collect();
        self.vconst(data, Type::I8X16)
    }

    pub fn i16x8const(&mut self, values: [i16; 8]) -> Value {
        let mut data = Vec::with_capacity(16);
        for &v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        self.vconst(data, Type::I16X8)
    }

    pub fn i32x4const(&mut self, values: [i32; 4]) -> Value {
        let mut data = Vec::with_capacity(16);
        for &v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        self.vconst(data, Type::I32X4)
    }

    pub fn i64x2const(&mut self, values: [i64; 2]) -> Value {
        let mut data = Vec::with_capacity(16);
        for &v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        self.vconst(data, Type::I64X2)
    }

    pub fn f32x4const(&mut self, values: [f32; 4]) -> Value {
        let mut data = Vec::with_capacity(16);
        for &v in &values {
            data.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        self.vconst(data, Type::F32X4)
    }

    pub fn f64x2const(&mut self, values: [f64; 2]) -> Value {
        let mut data = Vec::with_capacity(16);
        for &v in &values {
            data.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        self.vconst(data, Type::F64X2)
    }

    pub fn call(&mut self, func_id: FuncId, args: &[Value]) -> Inst {
        self.insert_inferred(InstDraft::call(func_id, args))
    }

    pub fn call_indirect(&mut self, sig_id: SigId, ptr: Value, args: &[Value]) -> Inst {
        self.insert_inferred(InstDraft::call_indirect(ptr, args, sig_id))
    }

    pub fn jump(&mut self, destination: Block, args: &[Value]) {
        self.insert(
            InstDraft::jump(crate::Successor {
                block: destination,
                args,
            }),
            &[],
        );
    }

    pub fn br(
        &mut self,
        condition: Value,
        then_block: Block,
        then_args: &[Value],
        else_block: Block,
        else_args: &[Value],
    ) {
        self.insert(
            InstDraft::br(
                condition,
                crate::Successor {
                    block: then_block,
                    args: then_args,
                },
                crate::Successor {
                    block: else_block,
                    args: else_args,
                },
            ),
            &[],
        );
    }

    pub fn br_table(&mut self, index: Value, default_call: BlockCall, targets: &[BlockCall]) {
        self.insert(
            InstDraft::br_table(
                index,
                targets
                    .iter()
                    .map(BlockCall::as_view)
                    .chain(core::iter::once(default_call.as_view())),
            ),
            &[],
        );
    }

    pub fn ret(&mut self, values: &[Value]) {
        self.insert(InstDraft::ret(values), &[]);
    }

    /// Call an intrinsic function.
    /// Returns the instruction handle, use `dfg.inst_results(inst)` to get return values.
    pub fn call_intrinsic(&mut self, intrinsic: Intrinsic, sig_id: SigId, args: &[Value]) -> Inst {
        self.insert_inferred(InstDraft::call_intrinsic(intrinsic, args, sig_id))
    }

    // ======================================
    // 向量操作构建方法
    // ======================================

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
        let ext = VectorExtData { mask, evl };

        let [result] = self.emit(
            InstDraft::vector_op_with_ext(opcode, args, ext),
            [result_ty],
        );
        result
    }
}
