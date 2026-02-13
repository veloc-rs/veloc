use super::function::{Function, StackSlotData};
use super::inst::{FloatCC, InstructionData, IntCC, Opcode};
use super::types::{
    Block, BlockCall, FuncId, JumpTable, Signature, StackSlot, Type, Value, ValueList, Variable,
};
use crate::types::{BlockCallData, JumpTableData};
use crate::{Linkage, Module, ModuleData, Result, SigId};
use alloc::sync::Arc;
use alloc::vec::Vec;
use hashbrown::HashMap;

pub struct ModuleBuilder {
    data: ModuleData,
}

impl ModuleBuilder {
    pub fn new() -> Self {
        Self {
            data: ModuleData::default(),
        }
    }

    pub fn declare_function(
        &mut self,
        name: String,
        signature: Signature,
        linkage: Linkage,
    ) -> FuncId {
        self.data.declare_function(name, signature, linkage)
    }

    pub fn get_func_id(&self, name: &str) -> Option<FuncId> {
        self.data.get_func_id(name)
    }

    pub fn intern_signature(&mut self, signature: Signature) -> SigId {
        self.data.intern_signature(signature)
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
        Module {
            inner: Arc::new(self.data),
        }
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
    // 跟踪已密封的 Block。密封意味着该 Block 的所有前驱都已确定。
    sealed_blocks: Vec<Block>,
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
            sealed_blocks: Vec::new(),
            incomplete_phis: HashMap::new(),
        };

        // 自动初始化 entry block 和参数
        if builder.func().layout.block_order.is_empty() {
            let entry = builder.create_block();
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            let sig_id = builder.func().signature;
            let sig = builder.module.signatures[sig_id].clone();
            for ty in sig.params {
                builder.add_block_param(entry, ty);
            }
        } else if let Some(entry) = builder.func().entry_block {
            builder.current_block = Some(entry);
        }

        builder
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

    pub fn push_inst(&mut self, block: Block, mut data: InstructionData) -> Option<Value> {
        match &mut data {
            InstructionData::Call {
                func_id, ret_ty, ..
            } => {
                let sig_id = self.func_signature(*func_id);
                *ret_ty = self.signature(sig_id).ret;
            }
            InstructionData::CallIndirect { sig_id, ret_ty, .. } => {
                *ret_ty = self.signature(*sig_id).ret;
            }
            InstructionData::Select { then_val, ty, .. } => {
                *ty = self.func().dfg.values[*then_val].ty;
            }
            _ => {}
        }

        let ty = data.result_type();
        let inst = self.func_mut().dfg.instructions.push(data);
        let succs = self.func().dfg.analyze_successors(inst);
        self.func_mut().layout.append_inst(block, inst);

        for succ in succs {
            self.func_mut().layout.add_edge(block, succ);
        }

        if ty != Type::Void {
            let val = self.func_mut().dfg.make_value(ty);
            self.func_mut().dfg.values[val].defined_by = Some(inst);
            Some(val)
        } else {
            None
        }
    }

    pub fn push_value_list(&mut self, values: &[Value]) -> ValueList {
        self.func_mut().dfg.push_value_list(values)
    }

    pub fn make_block_call(&mut self, block: Block, args: &[Value]) -> BlockCall {
        let args_list = self.push_value_list(args);
        self.func_mut().dfg.block_calls.push(BlockCallData {
            block,
            args: args_list,
        })
    }

    pub fn make_jump_table(&mut self, targets: Vec<BlockCall>) -> JumpTable {
        self.func_mut().dfg.jump_tables.push(JumpTableData {
            targets: targets.into_boxed_slice(),
        })
    }

    pub fn is_pristine(&self) -> bool {
        self.current_block.is_some()
            && self.func().layout.blocks[self.current_block.unwrap()]
                .insts
                .is_empty()
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
        if self.func().entry_block.is_none() {
            self.func_mut().entry_block = Some(block);
        }
    }

    pub fn block_params(&self, block: Block) -> &[Value] {
        &self.func().layout.blocks[block].params
    }

    pub fn block_params_len(&self, block: Block) -> usize {
        self.func().layout.blocks[block].params.len()
    }

    pub fn entry_block(&self) -> Option<Block> {
        self.func().entry_block
    }

    pub fn value_type(&self, val: Value) -> Type {
        self.func().dfg.values[val].ty
    }

    pub fn set_value_name(&mut self, val: Value, name: &str) {
        self.func_mut().dfg.value_names[val] = name.to_string();
    }

    pub fn add_block_param(&mut self, block: Block, ty: Type) -> Value {
        let val = self.func_mut().dfg.make_value(ty);
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
            let idata = &self.func().dfg.instructions[last_inst];
            matches!(
                idata,
                InstructionData::Jump { .. }
                    | InstructionData::Br { .. }
                    | InstructionData::BrTable { .. }
                    | InstructionData::Return { .. }
                    | InstructionData::Unreachable
            )
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
        self.def_map
            .entry(block)
            .or_insert_with(HashMap::new)
            .insert(var, val);
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
        if !self.sealed_blocks.contains(&block) {
            // Incomplete phi
            let ty = self.var_types[&var];
            val = self.add_block_param(block, ty);
            self.incomplete_phis
                .entry(block)
                .or_insert_with(Vec::new)
                .push((var, val));
        } else {
            let preds = self.func().layout.blocks[block].preds.clone();
            if preds.len() == 1 {
                val = self.use_var_on_block(preds[0], var);
            } else {
                let ty = self.var_types[&var];
                val = self.add_block_param(block, ty);
                // Break recursion
                self.def_map
                    .entry(block)
                    .or_insert_with(HashMap::new)
                    .insert(var, val);
                self.add_phi_operands(block, var, val);
            }
        }
        self.def_map
            .entry(block)
            .or_insert_with(HashMap::new)
            .insert(var, val);
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
        if self.sealed_blocks.contains(&block) {
            return;
        }
        if let Some(phis) = self.incomplete_phis.remove(&block) {
            for (var, phi) in phis {
                self.add_phi_operands(block, var, phi);
            }
        }
        self.sealed_blocks.push(block);
    }

    fn add_block_param_to_jump(&mut self, pred: Block, target: Block, index: usize, val: Value) {
        if let Some(&last_inst) = self.func().layout.blocks[pred].insts.last() {
            let dfg = &mut self.module.functions[self.func_id].dfg;
            let idata = dfg.instructions[last_inst].clone();

            match idata {
                InstructionData::Jump { mut dest } => {
                    let mut dest_data = dfg.block_calls[dest];
                    if dest_data.block == target {
                        let mut vec = dfg.get_value_list(dest_data.args).to_vec();
                        if index >= vec.len() {
                            vec.resize(index + 1, val);
                        } else {
                            vec[index] = val;
                        }
                        dest_data.args = dfg.push_value_list(&vec);
                        dest = dfg.block_calls.push(dest_data);
                        dfg.instructions[last_inst] = InstructionData::Jump { dest };
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
                        let mut vec = dfg.get_value_list(then_data.args).to_vec();
                        if index >= vec.len() {
                            vec.resize(index + 1, val);
                        } else {
                            vec[index] = val;
                        }
                        then_data.args = dfg.push_value_list(&vec);
                        then_dest = dfg.block_calls.push(then_data);
                        changed = true;
                    }
                    let mut else_data = dfg.block_calls[else_dest];
                    if else_data.block == target {
                        let mut vec = dfg.get_value_list(else_data.args).to_vec();
                        if index >= vec.len() {
                            vec.resize(index + 1, val);
                        } else {
                            vec[index] = val;
                        }
                        else_data.args = dfg.push_value_list(&vec);
                        else_dest = dfg.block_calls.push(else_data);
                        changed = true;
                    }

                    if changed {
                        dfg.instructions[last_inst] = InstructionData::Br {
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
                    let mut targets_data = dfg.jump_tables[table].targets.clone();
                    let mut changed = false;
                    for target_call in targets_data.iter_mut() {
                        let mut dest_data = dfg.block_calls[*target_call];
                        if dest_data.block == target {
                            let mut vec = dfg.get_value_list(dest_data.args).to_vec();
                            if index >= vec.len() {
                                vec.resize(index + 1, val);
                            } else {
                                vec[index] = val;
                            }
                            dest_data.args = dfg.push_value_list(&vec);
                            *target_call = dfg.block_calls.push(dest_data);
                            changed = true;
                        }
                    }

                    if changed {
                        let new_table = dfg.jump_tables.push(JumpTableData {
                            targets: targets_data,
                        });
                        dfg.instructions[last_inst] = InstructionData::BrTable {
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

    fn push(&mut self, data: InstructionData) -> Option<Value> {
        let block = self.block();
        self.builder.push_inst(block, data)
    }

    pub fn iconst(&mut self, ty: Type, val: i64) -> Value {
        self.push(InstructionData::Iconst { value: val, ty })
            .unwrap()
    }

    pub fn i32const(&mut self, val: i32) -> Value {
        self.iconst(Type::I32, val as i64)
    }

    pub fn i64const(&mut self, val: i64) -> Value {
        self.iconst(Type::I64, val)
    }

    pub fn fconst(&mut self, ty: Type, val: u64) -> Value {
        self.push(InstructionData::Fconst { value: val, ty })
            .unwrap()
    }

    pub fn f32const(&mut self, val: f32) -> Value {
        self.fconst(Type::F32, val.to_bits() as u64)
    }

    pub fn f64const(&mut self, val: f64) -> Value {
        self.fconst(Type::F64, val.to_bits())
    }

    pub fn bconst(&mut self, val: bool) -> Value {
        self.push(InstructionData::Bconst { value: val }).unwrap()
    }

    pub fn iadd(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = self.builder.value_type(lhs);
        self.push(InstructionData::Binary {
            opcode: Opcode::Iadd,
            args: [lhs, rhs],
            ty,
        })
        .unwrap()
    }

    pub fn isub(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = self.builder.value_type(lhs);
        self.push(InstructionData::Binary {
            opcode: Opcode::Isub,
            args: [lhs, rhs],
            ty,
        })
        .unwrap()
    }

    pub fn imul(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = self.builder.value_type(lhs);
        self.push(InstructionData::Binary {
            opcode: Opcode::Imul,
            args: [lhs, rhs],
            ty,
        })
        .unwrap()
    }

    pub fn fadd(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = self.builder.value_type(lhs);
        self.push(InstructionData::Binary {
            opcode: Opcode::Fadd,
            args: [lhs, rhs],
            ty,
        })
        .unwrap()
    }

    pub fn fsub(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = self.builder.value_type(lhs);
        self.push(InstructionData::Binary {
            opcode: Opcode::Fsub,
            args: [lhs, rhs],
            ty,
        })
        .unwrap()
    }

    pub fn fmul(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = self.builder.value_type(lhs);
        self.push(InstructionData::Binary {
            opcode: Opcode::Fmul,
            args: [lhs, rhs],
            ty,
        })
        .unwrap()
    }

    pub fn idiv(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = self.builder.value_type(lhs);
        self.push(InstructionData::Binary {
            opcode: Opcode::Idiv,
            args: [lhs, rhs],
            ty,
        })
        .unwrap()
    }

    pub fn udiv(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = self.builder.value_type(lhs);
        self.push(InstructionData::Binary {
            opcode: Opcode::Udiv,
            args: [lhs, rhs],
            ty,
        })
        .unwrap()
    }

    pub fn fdiv(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = self.builder.value_type(lhs);
        self.push(InstructionData::Binary {
            opcode: Opcode::Fdiv,
            args: [lhs, rhs],
            ty,
        })
        .unwrap()
    }

    pub fn irem(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = self.builder.value_type(lhs);
        self.push(InstructionData::Binary {
            opcode: Opcode::Irem,
            args: [lhs, rhs],
            ty,
        })
        .unwrap()
    }

    pub fn urem(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = self.builder.value_type(lhs);
        self.push(InstructionData::Binary {
            opcode: Opcode::Urem,
            args: [lhs, rhs],
            ty,
        })
        .unwrap()
    }

    pub fn and(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = self.builder.value_type(lhs);
        self.push(InstructionData::Binary {
            opcode: Opcode::And,
            args: [lhs, rhs],
            ty,
        })
        .unwrap()
    }

    pub fn or(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = self.builder.value_type(lhs);
        self.push(InstructionData::Binary {
            opcode: Opcode::Or,
            args: [lhs, rhs],
            ty,
        })
        .unwrap()
    }

    pub fn xor(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = self.builder.value_type(lhs);
        self.push(InstructionData::Binary {
            opcode: Opcode::Xor,
            args: [lhs, rhs],
            ty,
        })
        .unwrap()
    }

    pub fn shl(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = self.builder.value_type(lhs);
        self.push(InstructionData::Binary {
            opcode: Opcode::Shl,
            args: [lhs, rhs],
            ty,
        })
        .unwrap()
    }

    pub fn shr_s(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = self.builder.value_type(lhs);
        self.push(InstructionData::Binary {
            opcode: Opcode::ShrS,
            args: [lhs, rhs],
            ty,
        })
        .unwrap()
    }

    pub fn shr_u(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = self.builder.value_type(lhs);
        self.push(InstructionData::Binary {
            opcode: Opcode::ShrU,
            args: [lhs, rhs],
            ty,
        })
        .unwrap()
    }

    pub fn rotl(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = self.builder.value_type(lhs);
        self.push(InstructionData::Binary {
            opcode: Opcode::Rotl,
            args: [lhs, rhs],
            ty,
        })
        .unwrap()
    }

    pub fn rotr(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = self.builder.value_type(lhs);
        self.push(InstructionData::Binary {
            opcode: Opcode::Rotr,
            args: [lhs, rhs],
            ty,
        })
        .unwrap()
    }

    pub fn icmp(&mut self, kind: IntCC, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::IntCompare {
            kind,
            args: [lhs, rhs],
            ty: Type::Bool,
        })
        .unwrap()
    }

    pub fn fcmp(&mut self, kind: FloatCC, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::FloatCompare {
            kind,
            args: [lhs, rhs],
            ty: Type::Bool,
        })
        .unwrap()
    }

    pub fn eq(&mut self, lhs: Value, rhs: Value) -> Value {
        self.icmp(IntCC::Eq, lhs, rhs)
    }

    pub fn ne(&mut self, lhs: Value, rhs: Value) -> Value {
        self.icmp(IntCC::Ne, lhs, rhs)
    }

    pub fn lt_s(&mut self, lhs: Value, rhs: Value) -> Value {
        self.icmp(IntCC::LtS, lhs, rhs)
    }

    pub fn lt_u(&mut self, lhs: Value, rhs: Value) -> Value {
        self.icmp(IntCC::LtU, lhs, rhs)
    }

    pub fn gt_s(&mut self, lhs: Value, rhs: Value) -> Value {
        self.icmp(IntCC::GtS, lhs, rhs)
    }

    pub fn gt_u(&mut self, lhs: Value, rhs: Value) -> Value {
        self.icmp(IntCC::GtU, lhs, rhs)
    }

    pub fn le_s(&mut self, lhs: Value, rhs: Value) -> Value {
        self.icmp(IntCC::LeS, lhs, rhs)
    }

    pub fn le_u(&mut self, lhs: Value, rhs: Value) -> Value {
        self.icmp(IntCC::LeU, lhs, rhs)
    }

    pub fn ge_s(&mut self, lhs: Value, rhs: Value) -> Value {
        self.icmp(IntCC::GeS, lhs, rhs)
    }

    pub fn ge_u(&mut self, lhs: Value, rhs: Value) -> Value {
        self.icmp(IntCC::GeU, lhs, rhs)
    }

    pub fn feq(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fcmp(FloatCC::Eq, lhs, rhs)
    }

    pub fn fne(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fcmp(FloatCC::Ne, lhs, rhs)
    }

    pub fn flt(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fcmp(FloatCC::Lt, lhs, rhs)
    }

    pub fn fgt(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fcmp(FloatCC::Gt, lhs, rhs)
    }

    pub fn fle(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fcmp(FloatCC::Le, lhs, rhs)
    }

    pub fn fge(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fcmp(FloatCC::Ge, lhs, rhs)
    }

    pub fn eqz(&mut self, val: Value) -> Value {
        self.push(InstructionData::Unary {
            opcode: Opcode::Eqz,
            arg: val,
            ty: Type::Bool,
        })
        .unwrap()
    }

    pub fn clz(&mut self, val: Value) -> Value {
        let ty = self.builder.value_type(val);
        self.push(InstructionData::Unary {
            opcode: Opcode::Clz,
            arg: val,
            ty,
        })
        .unwrap()
    }

    pub fn ctz(&mut self, val: Value) -> Value {
        let ty = self.builder.value_type(val);
        self.push(InstructionData::Unary {
            opcode: Opcode::Ctz,
            arg: val,
            ty,
        })
        .unwrap()
    }

    pub fn popcnt(&mut self, val: Value) -> Value {
        let ty = self.builder.value_type(val);
        self.push(InstructionData::Unary {
            opcode: Opcode::Popcnt,
            arg: val,
            ty,
        })
        .unwrap()
    }

    pub fn ineg(&mut self, val: Value) -> Value {
        let ty = self.builder.value_type(val);
        self.push(InstructionData::Unary {
            opcode: Opcode::Ineg,
            arg: val,
            ty,
        })
        .unwrap()
    }

    pub fn fneg(&mut self, val: Value) -> Value {
        let ty = self.builder.value_type(val);
        self.push(InstructionData::Unary {
            opcode: Opcode::Fneg,
            arg: val,
            ty,
        })
        .unwrap()
    }

    pub fn abs(&mut self, val: Value) -> Value {
        let ty = self.builder.value_type(val);
        self.push(InstructionData::Unary {
            opcode: Opcode::Abs,
            arg: val,
            ty,
        })
        .unwrap()
    }

    pub fn sqrt(&mut self, val: Value) -> Value {
        let ty = self.builder.value_type(val);
        self.push(InstructionData::Unary {
            opcode: Opcode::Sqrt,
            arg: val,
            ty,
        })
        .unwrap()
    }

    pub fn ceil(&mut self, val: Value) -> Value {
        let ty = self.builder.value_type(val);
        self.push(InstructionData::Unary {
            opcode: Opcode::Ceil,
            arg: val,
            ty,
        })
        .unwrap()
    }

    pub fn floor(&mut self, val: Value) -> Value {
        let ty = self.builder.value_type(val);
        self.push(InstructionData::Unary {
            opcode: Opcode::Floor,
            arg: val,
            ty,
        })
        .unwrap()
    }

    pub fn trunc(&mut self, val: Value) -> Value {
        let ty = self.builder.value_type(val);
        self.push(InstructionData::Unary {
            opcode: Opcode::Trunc,
            arg: val,
            ty,
        })
        .unwrap()
    }

    pub fn nearest(&mut self, val: Value) -> Value {
        let ty = self.builder.value_type(val);
        self.push(InstructionData::Unary {
            opcode: Opcode::Nearest,
            arg: val,
            ty,
        })
        .unwrap()
    }

    pub fn min(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = self.builder.value_type(lhs);
        self.push(InstructionData::Binary {
            opcode: Opcode::Min,
            args: [lhs, rhs],
            ty,
        })
        .unwrap()
    }

    pub fn max(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = self.builder.value_type(lhs);
        self.push(InstructionData::Binary {
            opcode: Opcode::Max,
            args: [lhs, rhs],
            ty,
        })
        .unwrap()
    }

    pub fn copysign(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = self.builder.value_type(lhs);
        self.push(InstructionData::Binary {
            opcode: Opcode::Copysign,
            args: [lhs, rhs],
            ty,
        })
        .unwrap()
    }

    pub fn extend_s(&mut self, val: Value, ty: Type) -> Value {
        self.push(InstructionData::Unary {
            opcode: Opcode::ExtendS,
            arg: val,
            ty,
        })
        .unwrap()
    }

    pub fn extend_u(&mut self, val: Value, ty: Type) -> Value {
        self.push(InstructionData::Unary {
            opcode: Opcode::ExtendU,
            arg: val,
            ty,
        })
        .unwrap()
    }

    pub fn wrap(&mut self, val: Value, ty: Type) -> Value {
        self.push(InstructionData::Unary {
            opcode: Opcode::Wrap,
            arg: val,
            ty,
        })
        .unwrap()
    }

    pub fn trunc_s(&mut self, val: Value, ty: Type) -> Value {
        self.push(InstructionData::Unary {
            opcode: Opcode::TruncS,
            arg: val,
            ty,
        })
        .unwrap()
    }

    pub fn trunc_u(&mut self, val: Value, ty: Type) -> Value {
        self.push(InstructionData::Unary {
            opcode: Opcode::TruncU,
            arg: val,
            ty,
        })
        .unwrap()
    }

    pub fn convert_s(&mut self, val: Value, ty: Type) -> Value {
        self.push(InstructionData::Unary {
            opcode: Opcode::ConvertS,
            arg: val,
            ty,
        })
        .unwrap()
    }

    pub fn convert_u(&mut self, val: Value, ty: Type) -> Value {
        self.push(InstructionData::Unary {
            opcode: Opcode::ConvertU,
            arg: val,
            ty,
        })
        .unwrap()
    }

    pub fn promote(&mut self, val: Value, ty: Type) -> Value {
        self.push(InstructionData::Unary {
            opcode: Opcode::Promote,
            arg: val,
            ty,
        })
        .unwrap()
    }

    pub fn demote(&mut self, val: Value, ty: Type) -> Value {
        self.push(InstructionData::Unary {
            opcode: Opcode::Demote,
            arg: val,
            ty,
        })
        .unwrap()
    }

    pub fn reinterpret(&mut self, val: Value, ty: Type) -> Value {
        self.push(InstructionData::Unary {
            opcode: Opcode::Reinterpret,
            arg: val,
            ty,
        })
        .unwrap()
    }

    pub fn load(&mut self, ty: Type, ptr: Value, offset: u32) -> Value {
        self.push(InstructionData::Load { ptr, offset, ty })
            .unwrap()
    }

    pub fn store(&mut self, value: Value, ptr: Value, offset: u32) {
        self.push(InstructionData::Store { ptr, value, offset });
    }

    pub fn stack_load(&mut self, ty: Type, slot: StackSlot, offset: u32) -> Value {
        self.push(InstructionData::StackLoad { slot, offset, ty })
            .unwrap()
    }

    pub fn stack_store(&mut self, value: Value, slot: StackSlot, offset: u32) {
        self.push(InstructionData::StackStore {
            slot,
            value,
            offset,
        });
    }

    pub fn stack_addr(&mut self, slot: StackSlot, offset: u32) -> Value {
        self.push(InstructionData::StackAddr { slot, offset })
            .unwrap()
    }

    pub fn gep(&mut self, ptr: Value, offset: Value) -> Value {
        self.push(InstructionData::Gep { ptr, offset }).unwrap()
    }

    pub fn int_to_ptr(&mut self, arg: Value) -> Value {
        self.push(InstructionData::IntToPtr { arg }).unwrap()
    }

    pub fn ptr_to_int(&mut self, arg: Value, ty: Type) -> Value {
        self.push(InstructionData::PtrToInt { arg, ty }).unwrap()
    }

    pub fn call(&mut self, func_id: FuncId, args: &[Value]) -> Option<Value> {
        let args = self.builder.push_value_list(args);
        self.push(InstructionData::Call {
            func_id,
            args,
            ret_ty: Type::Void,
        })
    }

    pub fn call_indirect(&mut self, sig_id: SigId, ptr: Value, args: &[Value]) -> Option<Value> {
        let args = self.builder.push_value_list(args);
        self.push(InstructionData::CallIndirect {
            ptr,
            args,
            sig_id,
            ret_ty: Type::Void,
        })
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
        let then_dest = self.builder.make_block_call(then_block, then_args);
        let else_dest = self.builder.make_block_call(else_block, else_args);
        self.push(InstructionData::Br {
            condition,
            then_dest,
            else_dest,
        });
    }

    pub fn br_table(&mut self, index: Value, default_call: BlockCall, targets: &[BlockCall]) {
        let mut target_calls = Vec::with_capacity(targets.len() + 1);
        target_calls.push(default_call);
        target_calls.extend_from_slice(targets);
        let table = self.builder.func_mut().dfg.jump_tables.push(JumpTableData {
            targets: target_calls.into_boxed_slice(),
        });
        self.push(InstructionData::BrTable { index, table });
    }

    pub fn ret(&mut self, value: Option<Value>) {
        self.push(InstructionData::Return { value });
    }

    pub fn unreachable(&mut self) {
        self.push(InstructionData::Unreachable);
    }

    pub fn select(&mut self, condition: Value, then_val: Value, else_val: Value) -> Value {
        self.push(InstructionData::Select {
            condition,
            then_val,
            else_val,
            ty: Type::Void,
        })
        .unwrap()
    }
}
