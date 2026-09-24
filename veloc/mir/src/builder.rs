use super::function::{FuncBody, FuncEditor, InstCursor};
use super::inst::{InstWriter, VectorExtData};
use super::types::{Block, BlockCall, FuncId, Type, Value, Variable};
use crate::Opcode;
use crate::{CallConv, Linkage, Module, Result, SigId};
use alloc::vec::Vec;
use hashbrown::{HashMap, HashSet};

include!(concat!(env!("OUT_DIR"), "/builders.rs"));

pub struct ModuleBuilder {
    data: Module,
}

impl ModuleBuilder {
    pub fn new() -> Self {
        Self {
            data: Module::default(),
        }
    }

    pub fn with_types(types: alloc::sync::Arc<veloc_types::TypeContext>) -> Self {
        Self {
            data: Module::with_types(types),
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
        self.data
            .types_mut()
            .intern_signature(&params, &ret, call_conv)
    }

    pub fn find_function(&self, name: &str) -> Option<FuncId> {
        self.data.find_function(name)
    }

    /// Start a new definition. An existing body must be edited instead.
    pub fn define(&mut self, func_id: FuncId) -> SsaBuilder<'_> {
        SsaBuilder::new(&mut self.data, func_id)
    }

    pub fn add_global(&mut self, name: String, ty: Type, linkage: Linkage) {
        self.data.add_global(name, ty, linkage);
    }

    pub fn validate(&self) -> Result<()> {
        self.data.validate()
    }

    pub fn build(self) -> Module {
        self.data
    }
}

impl Default for ModuleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Incremental SSA construction state for one function.
///
/// Structural mutations and instruction insertion are performed by
/// `FuncEditor`; this type only tracks source variables and the sealing state
/// needed to materialize their SSA definitions.
pub struct SsaBuilder<'a> {
    function: &'a mut FuncBody,
    decls: &'a cranelift_entity::PrimaryMap<FuncId, crate::FuncDecl>,
    signatures: &'a veloc_types::Signatures,
    current_block: Block,
    // 变量的类型映射
    var_types: HashMap<Variable, Type>,
    // 每个 Block 对变量的最新定义: Block -> Variable -> Value
    def_map: HashMap<Block, HashMap<Variable, Value>>,
    // 未密封 Block 中待处理的 Phi 节点: Block -> Variable -> Phi Value
    incomplete_phis: HashMap<Block, Vec<(Variable, Value)>>,
    // Sealing belongs to this SSA construction session, not the finished IR.
    sealed: HashSet<Block>,
}

impl<'a> SsaBuilder<'a> {
    pub(crate) fn new(module: &'a mut Module, func_id: FuncId) -> Self {
        let (decls, signatures, function) = module.define_body(func_id);
        let entry = function.entry_block();
        Self {
            function,
            decls,
            signatures,
            current_block: entry,
            var_types: HashMap::new(),
            def_map: HashMap::new(),
            incomplete_phis: HashMap::new(),
            sealed: HashSet::from([entry]),
        }
    }

    pub fn current_block(&self) -> Block {
        self.current_block
    }

    pub fn func(&self) -> &FuncBody {
        self.function
    }

    fn edit(&mut self) -> FuncEditor<'_> {
        self.function.edit()
    }

    /// Complete SSA construction before exposing unrestricted body edits.
    pub fn finish(mut self) -> FuncEditor<'a> {
        self.seal_all_blocks();
        self.function.edit()
    }

    pub fn create_block(&mut self) -> Block {
        self.edit().create_block()
    }

    pub fn switch_to_block(&mut self, block: Block) {
        if !self.func().layout().contains_block(block) {
            self.edit().append_block(block);
        }
        self.current_block = block;
    }

    /// Add an explicit parameter before SSA variable resolution starts in this block.
    pub fn add_block_param(&mut self, block: Block, ty: Type) -> Value {
        assert!(!self.sealed.contains(&block), "block already sealed");
        assert!(
            !self.def_map.contains_key(&block),
            "SSA variable resolution already started"
        );
        self.edit().append_block_param(block, ty)
    }

    pub fn ins(&mut self) -> InstCursor<'_, '_> {
        let block = self.current_block;
        self.function
            .edit()
            .at_end(block, self.decls, self.signatures)
    }

    /// Temporarily insert at block start without changing the SSA current block.
    pub fn at_start(&mut self, block: Block) -> InstCursor<'_, '_> {
        self.function
            .edit()
            .at_start(block, self.decls, self.signatures)
    }

    pub fn is_current_block_terminated(&self) -> bool {
        let block = self.current_block;
        if let Some(last_inst) = self.func().layout().last_inst(block) {
            self.func().dfg().opcode(last_inst).spec().is_terminator()
        } else {
            false
        }
    }

    pub fn if_else<T, E>(&mut self, condition: Value, then_body: T, else_body: E)
    where
        T: FnOnce(&mut SsaBuilder),
        E: FnOnce(&mut SsaBuilder),
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
        C: FnOnce(&mut SsaBuilder) -> Value,
        B: FnOnce(&mut SsaBuilder),
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
        let block = self.current_block;
        self.def_map.entry(block).or_default().insert(var, val);
    }

    pub fn use_var(&mut self, var: Variable) -> Value {
        let block = self.current_block;
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
        if !self.sealed.contains(&block) {
            // Incomplete phi
            let ty = self.var_types[&var];
            val = self.edit().append_block_param(block, ty);
            self.incomplete_phis
                .entry(block)
                .or_default()
                .push((var, val));
        } else {
            let preds = &self.func().cfg().blocks[block].preds;
            if let &[pred] = preds.as_slice() {
                val = self.use_var_on_block(pred, var);
            } else {
                let ty = self.var_types[&var];
                val = self.edit().append_block_param(block, ty);
                // Break recursion
                self.def_map.entry(block).or_default().insert(var, val);
                self.add_phi_operands(block, var, val);
            }
        }
        self.def_map.entry(block).or_default().insert(var, val);
        val
    }

    fn add_phi_operands(&mut self, block: Block, var: Variable, phi: Value) {
        let index = self.func().dfg().blocks[block]
            .params
            .iter()
            .position(|&v| v == phi)
            .expect("Phi not found in block params");
        let preds = self.func().cfg().blocks[block].preds.clone();
        for p in preds {
            let val = self.use_var_on_block(p, var);
            self.add_block_param_to_jump(p, block, index, val);
        }
    }

    pub fn seal_block(&mut self, block: Block) {
        if self.sealed.contains(&block) {
            return;
        }
        if let Some(phis) = self.incomplete_phis.remove(&block) {
            for (var, phi) in phis {
                self.add_phi_operands(block, var, phi);
            }
        }
        self.sealed.insert(block);
    }

    pub fn seal_all_blocks(&mut self) {
        let blocks = self.func().layout().block_order().collect::<Vec<_>>();
        for block in blocks {
            self.seal_block(block);
        }
    }

    fn add_block_param_to_jump(&mut self, pred: Block, target: Block, index: usize, val: Value) {
        let Some(inst) = self.func().layout().last_inst(pred) else {
            return;
        };
        self.edit().edit_successors(inst, |edge| {
            if edge.block() == target {
                edge.set_arg(index, val);
            }
        });
    }
}

impl<'ctx, 'body> InstCursor<'ctx, 'body> {
    pub fn i32const(&mut self, val: i32) -> Value {
        self.iconst(val.into())
    }

    pub fn i64const(&mut self, val: i64) -> Value {
        self.iconst(val.into())
    }

    pub fn f32const(&mut self, val: f32) -> Value {
        self.fconst(val.into())
    }

    pub fn f64const(&mut self, val: f64) -> Value {
        self.fconst(val.into())
    }

    /// Materialize a constant in this function. Dense handles must belong to it.
    pub fn constant(&mut self, value: crate::Constant) -> Value {
        let ty = value.ty();
        let data = |writer: InstWriter<'_>| {
            if let Some(value) = value.as_scalar() {
                writer.scalar_const(value)
            } else {
                writer.vconst(value.as_vector().expect("supported constant form"))
            }
        };
        let [result] = self.emit(data, [ty]);
        result
    }

    pub fn i8x16const(&mut self, values: [i8; 16]) -> Value {
        let data = values.iter().map(|&v| v as u8).collect();
        self.dense_const(data, crate::Type::I8X16)
    }

    pub fn i16x8const(&mut self, values: [i16; 8]) -> Value {
        let mut data = Vec::with_capacity(16);
        for &v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        self.dense_const(data, crate::Type::I16X8)
    }

    pub fn i32x4const(&mut self, values: [i32; 4]) -> Value {
        let mut data = Vec::with_capacity(16);
        for &v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        self.dense_const(data, crate::Type::I32X4)
    }

    pub fn i64x2const(&mut self, values: [i64; 2]) -> Value {
        let mut data = Vec::with_capacity(16);
        for &v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        self.dense_const(data, crate::Type::I64X2)
    }

    pub fn f32x4const(&mut self, values: [f32; 4]) -> Value {
        let mut data = Vec::with_capacity(16);
        for &v in &values {
            data.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        self.dense_const(data, crate::Type::F32X4)
    }

    pub fn f64x2const(&mut self, values: [f64; 2]) -> Value {
        let mut data = Vec::with_capacity(16);
        for &v in &values {
            data.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        self.dense_const(data, crate::Type::F64X2)
    }

    pub fn jump(&mut self, destination: Block, args: &[Value]) {
        self.insert(
            |writer: InstWriter<'_>| {
                writer.jump(crate::Successor {
                    block: destination,
                    args,
                })
            },
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
            |writer: InstWriter<'_>| {
                writer.br(
                    condition,
                    crate::Successor {
                        block: then_block,
                        args: then_args,
                    },
                    crate::Successor {
                        block: else_block,
                        args: else_args,
                    },
                )
            },
            &[],
        );
    }

    pub fn br_table(&mut self, index: Value, default_call: BlockCall, targets: &[BlockCall]) {
        self.insert(
            |writer: InstWriter<'_>| {
                writer.br_table(
                    index,
                    targets
                        .iter()
                        .map(BlockCall::as_view)
                        .chain(core::iter::once(default_call.as_view())),
                )
            },
            &[],
        );
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
            |writer: InstWriter<'_>| writer.vector_op_with_ext(opcode, args, ext),
            [result_ty],
        );
        result
    }
}
