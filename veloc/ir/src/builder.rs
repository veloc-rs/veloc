use super::function::{Function, StackSlotData};
use super::inst::InstructionData;
use super::opcode::{FloatCC, IntCC, MemFlags, Opcode};
use super::types::{
    Block, BlockCall, FuncId, JumpTable, Signature, StackSlot, Type, Value, ValueList, Variable,
};
use crate::types::{BlockCallData, JumpTableData};
use crate::{CallConv, Intrinsic, Linkage, Module, ModuleData, Result, SigId};
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

    pub fn declare_function(&mut self, name: String, sig_id: SigId, linkage: Linkage) -> FuncId {
        self.data.declare_function(name, sig_id, linkage)
    }

    pub fn make_signature(&mut self, params: Vec<Type>, ret: Type, call_conv: CallConv) -> SigId {
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
        for &ty in sig.params.iter() {
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

    pub fn push_inst(&mut self, block: Block, data: InstructionData) -> Option<Value> {
        let ty = self.inst_result_type(&data);
        let inst = self.func_mut().dfg.instructions.push(data);

        self.func_mut().layout.append_inst(block, inst);

        if ty != Type::Void {
            Some(self.func_mut().dfg.append_result(inst, ty))
        } else {
            self.func_mut().dfg.inst_results[inst] = None;
            None
        }
    }

    /// Compute the result type of an instruction.
    /// This method requires `&self` to access operand types and module signatures.
    fn inst_result_type(&self, data: &InstructionData) -> Type {
        use super::inst::InstructionData;
        match data {
            // Unary/Binary ops: result type same as input operand
            InstructionData::Unary { arg, .. } => self.value_type(*arg),
            InstructionData::Binary { args, .. } => self.value_type(args[0]),

            // Load/StackLoad: type must be provided by caller (via push_inst_with_type)
            // These will be handled specially
            InstructionData::Load { .. } => {
                panic!("Load result type must be provided explicitly")
            }
            InstructionData::StackLoad { .. } => {
                panic!("StackLoad result type must be provided explicitly")
            }

            // Constants: type must be provided by caller
            InstructionData::Iconst { .. } => {
                panic!("Iconst result type must be provided explicitly")
            }
            InstructionData::Fconst { .. } => {
                panic!("Fconst result type must be provided explicitly")
            }
            InstructionData::PtrToInt { .. } => {
                panic!("PtrToInt result type must be provided explicitly")
            }
            InstructionData::ExtractValue { .. } => {
                panic!("ExtractValue result type must be provided explicitly")
            }

            // Instructions with fixed result type
            InstructionData::Bconst { .. } => Type::Bool,
            InstructionData::StackAddr { .. } => Type::Ptr,
            InstructionData::IntToPtr { .. } => Type::Ptr,
            InstructionData::PtrOffset { .. } => Type::Ptr,
            InstructionData::PtrIndex { .. } => Type::Ptr,

            // Comparisons: always return Bool
            InstructionData::IntCompare { .. } | InstructionData::FloatCompare { .. } => Type::Bool,

            // Select: result type same as then_val/else_val
            InstructionData::Select { then_val, .. } => self.value_type(*then_val),

            // Instructions that never produce a value
            InstructionData::Store { .. }
            | InstructionData::StackStore { .. }
            | InstructionData::Jump { .. }
            | InstructionData::Br { .. }
            | InstructionData::BrTable { .. }
            | InstructionData::Return { .. }
            | InstructionData::Unreachable => Type::Void,

            // Call instructions: return type is obtained from the signature
            InstructionData::Call { func_id, .. } => {
                let sig_id = self.module.functions[*func_id].signature;
                self.module.signatures[sig_id].ret
            }
            InstructionData::CallIndirect { sig_id, .. } => self.module.signatures[*sig_id].ret,
            InstructionData::CallIntrinsic { sig_id, .. } => self.module.signatures[*sig_id].ret,

            // Multi-value construction
            InstructionData::ConstructMulti { .. } => Type::MultiValue,

            InstructionData::Nop => Type::Void,
        }
    }

    /// Push an instruction that requires explicit result type (like Load, Iconst, etc.)
    fn push_inst_with_type(
        &mut self,
        block: Block,
        data: InstructionData,
        ty: Type,
    ) -> Option<Value> {
        let inst = self.func_mut().dfg.instructions.push(data);
        self.func_mut().layout.append_inst(block, inst);

        if ty != Type::Void {
            Some(self.func_mut().dfg.append_result(inst, ty))
        } else {
            self.func_mut().dfg.inst_results[inst] = None;
            None
        }
    }

    pub fn make_value_list(&mut self, values: &[Value]) -> ValueList {
        self.func_mut().dfg.make_value_list(values)
    }

    pub fn make_block_call(&mut self, block: Block, args: &[Value]) -> BlockCall {
        let args_list = self.make_value_list(args);
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
        if !self.func().is_defined() {
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
        self.func_mut().layout.blocks[block].is_sealed = true;
    }

    pub fn seal_all_blocks(&mut self) {
        let blocks: Vec<_> = self.func().layout.block_order.iter().copied().collect();
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
                            targets: targets_data.into_boxed_slice(),
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

    fn push(&mut self, data: InstructionData) -> Option<Value> {
        let block = self.block();
        self.builder.push_inst(block, data)
    }

    fn push_with_type(&mut self, data: InstructionData, ty: Type) -> Option<Value> {
        let block = self.block();
        self.builder.push_inst_with_type(block, data, ty)
    }

    pub fn iconst(&mut self, ty: Type, val: i64) -> Value {
        match ty {
            Type::I8 | Type::I16 | Type::I32 | Type::I64 => self
                .push_with_type(InstructionData::Iconst { value: val }, ty)
                .unwrap(),
            _ => panic!("iconst only supports integer types"),
        }
    }

    pub fn i32const(&mut self, val: i32) -> Value {
        self.iconst(Type::I32, val as i64)
    }

    pub fn i64const(&mut self, val: i64) -> Value {
        self.iconst(Type::I64, val)
    }

    pub fn fconst(&mut self, ty: Type, val: u64) -> Value {
        self.push_with_type(InstructionData::Fconst { value: val }, ty)
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
        self.push(InstructionData::Binary {
            opcode: Opcode::Iadd,
            args: [lhs, rhs],
        })
        .unwrap()
    }

    pub fn isub(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::Binary {
            opcode: Opcode::Isub,
            args: [lhs, rhs],
        })
        .unwrap()
    }

    pub fn imul(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::Binary {
            opcode: Opcode::Imul,
            args: [lhs, rhs],
        })
        .unwrap()
    }

    pub fn fadd(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::Binary {
            opcode: Opcode::Fadd,
            args: [lhs, rhs],
        })
        .unwrap()
    }

    pub fn fsub(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::Binary {
            opcode: Opcode::Fsub,
            args: [lhs, rhs],
        })
        .unwrap()
    }

    pub fn fmul(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::Binary {
            opcode: Opcode::Fmul,
            args: [lhs, rhs],
        })
        .unwrap()
    }

    pub fn idiv(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::Binary {
            opcode: Opcode::DivS,
            args: [lhs, rhs],
        })
        .unwrap()
    }

    pub fn udiv(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::Binary {
            opcode: Opcode::DivU,
            args: [lhs, rhs],
        })
        .unwrap()
    }

    pub fn fdiv(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::Binary {
            opcode: Opcode::Fdiv,
            args: [lhs, rhs],
        })
        .unwrap()
    }

    pub fn irem(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::Binary {
            opcode: Opcode::RemS,
            args: [lhs, rhs],
        })
        .unwrap()
    }

    pub fn urem(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::Binary {
            opcode: Opcode::RemU,
            args: [lhs, rhs],
        })
        .unwrap()
    }

    pub fn and(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::Binary {
            opcode: Opcode::And,
            args: [lhs, rhs],
        })
        .unwrap()
    }

    pub fn or(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::Binary {
            opcode: Opcode::Or,
            args: [lhs, rhs],
        })
        .unwrap()
    }

    pub fn xor(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::Binary {
            opcode: Opcode::Xor,
            args: [lhs, rhs],
        })
        .unwrap()
    }

    pub fn shl(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::Binary {
            opcode: Opcode::Shl,
            args: [lhs, rhs],
        })
        .unwrap()
    }

    pub fn shr_s(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::Binary {
            opcode: Opcode::ShrS,
            args: [lhs, rhs],
        })
        .unwrap()
    }

    pub fn shr_u(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::Binary {
            opcode: Opcode::ShrU,
            args: [lhs, rhs],
        })
        .unwrap()
    }

    pub fn rotl(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::Binary {
            opcode: Opcode::Rotl,
            args: [lhs, rhs],
        })
        .unwrap()
    }

    pub fn rotr(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::Binary {
            opcode: Opcode::Rotr,
            args: [lhs, rhs],
        })
        .unwrap()
    }

    pub fn icmp(&mut self, kind: IntCC, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::IntCompare {
            kind,
            args: [lhs, rhs],
        })
        .unwrap()
    }

    pub fn fcmp(&mut self, kind: FloatCC, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::FloatCompare {
            kind,
            args: [lhs, rhs],
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
        self.push_with_type(
            InstructionData::Unary {
                opcode: Opcode::Eqz,
                arg: val,
            },
            Type::Bool,
        )
        .unwrap()
    }

    pub fn clz(&mut self, val: Value) -> Value {
        self.push(InstructionData::Unary {
            opcode: Opcode::Clz,
            arg: val,
        })
        .unwrap()
    }

    pub fn ctz(&mut self, val: Value) -> Value {
        self.push(InstructionData::Unary {
            opcode: Opcode::Ctz,
            arg: val,
        })
        .unwrap()
    }

    pub fn popcnt(&mut self, val: Value) -> Value {
        self.push(InstructionData::Unary {
            opcode: Opcode::Popcnt,
            arg: val,
        })
        .unwrap()
    }

    pub fn ineg(&mut self, val: Value) -> Value {
        self.push(InstructionData::Unary {
            opcode: Opcode::Ineg,
            arg: val,
        })
        .unwrap()
    }

    pub fn fneg(&mut self, val: Value) -> Value {
        self.push(InstructionData::Unary {
            opcode: Opcode::Fneg,
            arg: val,
        })
        .unwrap()
    }

    pub fn abs(&mut self, val: Value) -> Value {
        self.push(InstructionData::Unary {
            opcode: Opcode::Abs,
            arg: val,
        })
        .unwrap()
    }

    pub fn sqrt(&mut self, val: Value) -> Value {
        self.push(InstructionData::Unary {
            opcode: Opcode::Sqrt,
            arg: val,
        })
        .unwrap()
    }

    pub fn ceil(&mut self, val: Value) -> Value {
        self.push(InstructionData::Unary {
            opcode: Opcode::Ceil,
            arg: val,
        })
        .unwrap()
    }

    pub fn floor(&mut self, val: Value) -> Value {
        self.push(InstructionData::Unary {
            opcode: Opcode::Floor,
            arg: val,
        })
        .unwrap()
    }

    pub fn trunc(&mut self, val: Value) -> Value {
        self.push(InstructionData::Unary {
            opcode: Opcode::Trunc,
            arg: val,
        })
        .unwrap()
    }

    pub fn nearest(&mut self, val: Value) -> Value {
        self.push(InstructionData::Unary {
            opcode: Opcode::Nearest,
            arg: val,
        })
        .unwrap()
    }

    pub fn min(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::Binary {
            opcode: Opcode::Min,
            args: [lhs, rhs],
        })
        .unwrap()
    }

    pub fn max(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::Binary {
            opcode: Opcode::Max,
            args: [lhs, rhs],
        })
        .unwrap()
    }

    pub fn copysign(&mut self, lhs: Value, rhs: Value) -> Value {
        self.push(InstructionData::Binary {
            opcode: Opcode::Copysign,
            args: [lhs, rhs],
        })
        .unwrap()
    }

    pub fn extend_s(&mut self, val: Value, ty: Type) -> Value {
        self.push_with_type(
            InstructionData::Unary {
                opcode: Opcode::ExtendS,
                arg: val,
            },
            ty,
        )
        .unwrap()
    }

    pub fn extend_u(&mut self, val: Value, ty: Type) -> Value {
        self.push_with_type(
            InstructionData::Unary {
                opcode: Opcode::ExtendU,
                arg: val,
            },
            ty,
        )
        .unwrap()
    }

    pub fn wrap(&mut self, val: Value, ty: Type) -> Value {
        self.push_with_type(
            InstructionData::Unary {
                opcode: Opcode::Wrap,
                arg: val,
            },
            ty,
        )
        .unwrap()
    }

    pub fn trunc_s(&mut self, val: Value, ty: Type) -> Value {
        self.push_with_type(
            InstructionData::Unary {
                opcode: Opcode::TruncS,
                arg: val,
            },
            ty,
        )
        .unwrap()
    }

    pub fn trunc_u(&mut self, val: Value, ty: Type) -> Value {
        self.push_with_type(
            InstructionData::Unary {
                opcode: Opcode::TruncU,
                arg: val,
            },
            ty,
        )
        .unwrap()
    }

    pub fn convert_s(&mut self, val: Value, ty: Type) -> Value {
        self.push_with_type(
            InstructionData::Unary {
                opcode: Opcode::ConvertS,
                arg: val,
            },
            ty,
        )
        .unwrap()
    }

    pub fn convert_u(&mut self, val: Value, ty: Type) -> Value {
        self.push_with_type(
            InstructionData::Unary {
                opcode: Opcode::ConvertU,
                arg: val,
            },
            ty,
        )
        .unwrap()
    }

    pub fn promote(&mut self, val: Value, ty: Type) -> Value {
        self.push_with_type(
            InstructionData::Unary {
                opcode: Opcode::Promote,
                arg: val,
            },
            ty,
        )
        .unwrap()
    }

    pub fn demote(&mut self, val: Value, ty: Type) -> Value {
        self.push_with_type(
            InstructionData::Unary {
                opcode: Opcode::Demote,
                arg: val,
            },
            ty,
        )
        .unwrap()
    }

    pub fn reinterpret(&mut self, val: Value, ty: Type) -> Value {
        self.push_with_type(
            InstructionData::Unary {
                opcode: Opcode::Reinterpret,
                arg: val,
            },
            ty,
        )
        .unwrap()
    }

    pub fn load(&mut self, ty: Type, ptr: Value, offset: u32, flags: MemFlags) -> Value {
        self.push_with_type(InstructionData::Load { ptr, offset, flags }, ty)
            .unwrap()
    }

    pub fn store(&mut self, value: Value, ptr: Value, offset: u32, flags: MemFlags) {
        self.push(InstructionData::Store {
            ptr,
            value,
            offset,
            flags,
        });
    }

    pub fn stack_load(&mut self, ty: Type, slot: StackSlot, offset: u32) -> Value {
        self.push_with_type(InstructionData::StackLoad { slot, offset }, ty)
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

    pub fn ptr_offset(&mut self, ptr: Value, offset: i32) -> Value {
        self.push(InstructionData::PtrOffset { ptr, offset })
            .unwrap()
    }

    pub fn ptr_index(&mut self, ptr: Value, index: Value, scale: u32, offset: i32) -> Value {
        self.push(InstructionData::PtrIndex {
            ptr,
            index,
            scale,
            offset,
        })
        .unwrap()
    }

    pub fn int_to_ptr(&mut self, arg: Value) -> Value {
        self.push(InstructionData::IntToPtr { arg }).unwrap()
    }

    pub fn ptr_to_int(&mut self, arg: Value, _ty: Type) -> Value {
        self.push(InstructionData::PtrToInt { arg }).unwrap()
    }

    pub fn call(&mut self, func_id: FuncId, args: &[Value]) -> Option<Value> {
        let args = self.builder.make_value_list(args);
        self.push(InstructionData::Call { func_id, args })
    }

    pub fn call_indirect(&mut self, sig_id: SigId, ptr: Value, args: &[Value]) -> Option<Value> {
        let args = self.builder.make_value_list(args);
        self.push(InstructionData::CallIndirect { ptr, args, sig_id })
    }

    pub fn jump(&mut self, destination: Block, args: &[Value]) {
        let dest = self.builder.make_block_call(destination, args);
        let block = self.block();
        self.push(InstructionData::Jump { dest });
        self.builder.func_mut().layout.add_edge(block, destination);
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
            Type::Bool,
            "Condition for br must be a bool"
        );
        let then_dest = self.builder.make_block_call(then_block, then_args);
        let else_dest = self.builder.make_block_call(else_block, else_args);
        let block = self.block();
        self.push(InstructionData::Br {
            condition,
            then_dest,
            else_dest,
        });
        self.builder.func_mut().layout.add_edge(block, then_block);
        self.builder.func_mut().layout.add_edge(block, else_block);
    }

    pub fn br_table(&mut self, index: Value, default_call: BlockCall, targets: &[BlockCall]) {
        debug_assert_eq!(
            self.builder.value_type(index),
            Type::I32,
            "Index for br_table must be an i32"
        );
        let block = self.block();
        let mut target_calls = Vec::with_capacity(targets.len() + 1);
        target_calls.push(default_call);
        target_calls.extend_from_slice(targets);

        for &call in &target_calls {
            let target_block = self.builder.func().dfg.block_call_block(call);
            self.builder.func_mut().layout.add_edge(block, target_block);
        }

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
        debug_assert_eq!(
            self.builder.value_type(condition),
            Type::Bool,
            "Condition for select must be a bool"
        );
        let ty = self.builder.value_type(then_val);
        debug_assert_eq!(
            ty,
            self.builder.value_type(else_val),
            "Select types must match"
        );
        self.push(InstructionData::Select {
            condition,
            then_val,
            else_val,
        })
        .unwrap()
    }

    /// Call an intrinsic function.
    /// Returns `None` if the intrinsic returns void, otherwise returns the result value.
    pub fn call_intrinsic(
        &mut self,
        intrinsic: Intrinsic,
        sig_id: SigId,
        args: &[Value],
    ) -> Option<Value> {
        let args = self.builder.make_value_list(args);
        self.push(InstructionData::CallIntrinsic {
            intrinsic,
            args,
            sig_id,
        })
    }

    /// Extract a single value from a multi-value.
    /// The `index` specifies which element to extract (0-indexed).
    /// The `ty` specifies the expected type of the extracted value.
    pub fn extract_value(&mut self, val: Value, index: u32, _ty: Type) -> Value {
        debug_assert_eq!(
            self.builder.value_type(val),
            Type::MultiValue,
            "Can only extract from MultiValue type"
        );
        self.push(InstructionData::ExtractValue { val, index })
            .unwrap()
    }

    /// Construct a multi-value from multiple values.
    /// This is typically used at the end of a function to bundle multiple return values.
    pub fn construct_multi(&mut self, values: &[Value]) -> Value {
        for &v in values {
            let t = self.builder.value_type(v);
            debug_assert!(
                t != Type::Void && t != Type::MultiValue,
                "Cannot include Void or MultiValue in ConstructMulti"
            );
        }
        let values = self.builder.make_value_list(values);
        self.push(InstructionData::ConstructMulti { values })
            .unwrap()
    }
}
