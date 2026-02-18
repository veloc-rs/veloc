use crate::inst::Inst;
use crate::{Block, Function, InstructionData, ModuleData, Opcode, Result, Type, Value};
use alloc::string::String;
use core::fmt;

#[derive(Debug, Clone)]
pub enum ValidationError {
    EmptyBlock(Block),
    NoTerminator(Block),
    TypeMismatch {
        opcode: Opcode,
        expected: Type,
        got: Type,
    },
    ReturnMismatch {
        expected: Type,
        got: Type,
    },
    OperandTypeMismatch {
        inst: Inst,
        lhs: Type,
        rhs: Type,
    },
    ConditionNotBool(Inst, Type),
    SelectMismatch {
        inst: Inst,
        expected: Type,
        then_val: Type,
        else_val: Type,
    },
    InvalidConversion {
        inst: Inst,
        opcode: Opcode,
        from: Type,
        to: Type,
    },
    PointerArithmetic(Inst, Opcode),
    UnsealedBlock(Block),
    Other(String),
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBlock(b) => write!(f, "Block {:?} is empty", b),
            Self::NoTerminator(b) => write!(f, "Block {:?} does not end with a terminator", b),
            Self::TypeMismatch {
                opcode,
                expected,
                got,
            } => {
                write!(
                    f,
                    "Instruction {:?} type mismatch: expected {:?}, got {:?}",
                    opcode, expected, got
                )
            }
            Self::ReturnMismatch { expected, got } => {
                write!(
                    f,
                    "Return type mismatch: expected {:?}, got {:?}",
                    expected, got
                )
            }
            Self::OperandTypeMismatch { inst, lhs, rhs } => {
                write!(
                    f,
                    "Operand type mismatch for {:?}: lhs {:?}, rhs {:?}",
                    inst, lhs, rhs
                )
            }
            Self::ConditionNotBool(inst, got) => {
                write!(f, "Condition for {:?} must be Bool, got {:?}", inst, got)
            }
            Self::SelectMismatch {
                inst,
                expected,
                then_val,
                else_val,
            } => {
                write!(
                    f,
                    "Select {:?} type mismatch: expected {:?}, got {:?} and {:?}",
                    inst, expected, then_val, else_val
                )
            }
            Self::InvalidConversion {
                inst,
                opcode,
                from,
                to,
            } => {
                write!(
                    f,
                    "Invalid conversion in {:?} for {:?}: from {:?} to {:?}",
                    inst, opcode, from, to
                )
            }
            Self::PointerArithmetic(inst, opcode) => {
                write!(
                    f,
                    "Pointer arithmetic not allowed for instruction {:?} ({:?}). Use PtrOffset or PtrIndex instead.",
                    inst, opcode
                )
            }
            Self::UnsealedBlock(b) => write!(f, "Block {:?} is not sealed", b),
            Self::Other(s) => write!(f, "{}", s),
        }
    }
}

impl ModuleData {
    pub fn validate(&self) -> Result<()> {
        for (_, func) in self.functions.iter() {
            func.validate(self).map_err(|e| {
                crate::Error::Message(alloc::format!("In function {}: {}", func.name, e))
            })?;
        }
        Ok(())
    }
}

impl Function {
    pub fn validate(&self, module: &ModuleData) -> Result<()> {
        let dfg = &self.dfg;
        let layout = &self.layout;

        for block in &layout.block_order {
            let block_data = &layout.blocks[*block];
            if !block_data.is_sealed {
                return Err(ValidationError::UnsealedBlock(*block).into());
            }
            let insts = &block_data.insts;
            if insts.is_empty() {
                return Err(ValidationError::EmptyBlock(*block).into());
            }
            for &inst in insts {
                self.validate_inst(module, inst)?;
            }

            let last_inst = *insts.last().unwrap();
            if !dfg.instructions[last_inst].is_terminator() {
                return Err(ValidationError::NoTerminator(*block).into());
            }
        }
        Ok(())
    }

    fn validate_inst(&self, module: &ModuleData, inst: Inst) -> Result<()> {
        let dfg = &self.dfg;
        let data = &dfg.instructions[inst];
        let val_ty = |v: Value| dfg.values[v].ty;

        match data {
            InstructionData::Unary { opcode, arg } => {
                let arg_ty = val_ty(*arg);
                // Get the result type from the instruction's result
                let result_ty = dfg
                    .inst_results(inst)
                    .map(|v| val_ty(v))
                    .unwrap_or(Type::Void);

                if arg_ty == Type::Ptr || result_ty == Type::Ptr {
                    return Err(ValidationError::PointerArithmetic(inst, *opcode).into());
                }

                match opcode {
                    Opcode::Ineg | Opcode::Clz | Opcode::Ctz | Opcode::Popcnt => {
                        if !result_ty.is_integer() || arg_ty != result_ty {
                            return Err(ValidationError::TypeMismatch {
                                opcode: *opcode,
                                expected: result_ty,
                                got: arg_ty,
                            }
                            .into());
                        }
                    }
                    Opcode::Fneg
                    | Opcode::Abs
                    | Opcode::Sqrt
                    | Opcode::Ceil
                    | Opcode::Floor
                    | Opcode::Trunc
                    | Opcode::Nearest => {
                        if !result_ty.is_float() || arg_ty != result_ty {
                            return Err(ValidationError::TypeMismatch {
                                opcode: *opcode,
                                expected: result_ty,
                                got: arg_ty,
                            }
                            .into());
                        }
                    }
                    Opcode::Eqz => {
                        if !arg_ty.is_integer() || result_ty != Type::Bool {
                            return Err(ValidationError::TypeMismatch {
                                opcode: Opcode::Eqz,
                                expected: Type::Bool,
                                got: result_ty,
                            }
                            .into());
                        }
                    }
                    // Conversion operators
                    Opcode::ExtendS | Opcode::ExtendU => {
                        let is_valid = if arg_ty == Type::Bool {
                            *opcode == Opcode::ExtendU && result_ty.is_integer()
                        } else {
                            arg_ty.is_integer()
                                && result_ty.is_integer()
                                && arg_ty.size_bytes() < result_ty.size_bytes()
                        };

                        if !is_valid {
                            return Err(ValidationError::InvalidConversion {
                                inst,
                                opcode: *opcode,
                                from: arg_ty,
                                to: result_ty,
                            }
                            .into());
                        }
                    }
                    Opcode::Wrap => {
                        if !arg_ty.is_integer()
                            || !result_ty.is_integer()
                            || arg_ty.size_bytes() <= result_ty.size_bytes()
                        {
                            return Err(ValidationError::InvalidConversion {
                                inst,
                                opcode: Opcode::Wrap,
                                from: arg_ty,
                                to: result_ty,
                            }
                            .into());
                        }
                    }
                    Opcode::Demote => {
                        if arg_ty != Type::F64 || result_ty != Type::F32 {
                            return Err(ValidationError::InvalidConversion {
                                inst,
                                opcode: Opcode::Demote,
                                from: arg_ty,
                                to: result_ty,
                            }
                            .into());
                        }
                    }
                    Opcode::TruncS | Opcode::TruncU => {
                        if !arg_ty.is_float() || !result_ty.is_integer() {
                            return Err(ValidationError::InvalidConversion {
                                inst,
                                opcode: *opcode,
                                from: arg_ty,
                                to: result_ty,
                            }
                            .into());
                        }
                    }
                    Opcode::ConvertS | Opcode::ConvertU => {
                        if !arg_ty.is_integer() || !result_ty.is_float() {
                            return Err(ValidationError::InvalidConversion {
                                inst,
                                opcode: *opcode,
                                from: arg_ty,
                                to: result_ty,
                            }
                            .into());
                        }
                    }
                    Opcode::Reinterpret => {
                        if arg_ty.size_bytes() != result_ty.size_bytes() || arg_ty == result_ty {
                            return Err(ValidationError::InvalidConversion {
                                inst,
                                opcode: Opcode::Reinterpret,
                                from: arg_ty,
                                to: result_ty,
                            }
                            .into());
                        }
                    }
                    _ => {}
                }
            }
            InstructionData::Binary { opcode, args } => {
                let lhs_ty = val_ty(args[0]);
                let rhs_ty = val_ty(args[1]);
                let result_ty = dfg
                    .inst_results(inst)
                    .map(|v| val_ty(v))
                    .unwrap_or(Type::Void);

                if lhs_ty == Type::Ptr || rhs_ty == Type::Ptr || result_ty == Type::Ptr {
                    return Err(ValidationError::PointerArithmetic(inst, *opcode).into());
                }

                if lhs_ty != result_ty || rhs_ty != result_ty {
                    return Err(ValidationError::OperandTypeMismatch {
                        inst,
                        lhs: lhs_ty,
                        rhs: rhs_ty,
                    }
                    .into());
                }
            }
            InstructionData::Load { ptr, .. } => {
                if val_ty(*ptr) != Type::Ptr {
                    return Err(ValidationError::TypeMismatch {
                        opcode: Opcode::Load,
                        expected: Type::Ptr,
                        got: val_ty(*ptr),
                    }
                    .into());
                }
                let result_ty = dfg
                    .inst_results(inst)
                    .map(|v| val_ty(v))
                    .unwrap_or(Type::Void);
                if result_ty == Type::Void {
                    return Err(ValidationError::Other(alloc::format!(
                        "Cannot load void type at {:?}",
                        inst
                    ))
                    .into());
                }
            }
            InstructionData::Store { ptr, value, .. } => {
                if val_ty(*ptr) != Type::Ptr {
                    return Err(ValidationError::TypeMismatch {
                        opcode: Opcode::Store,
                        expected: Type::Ptr,
                        got: val_ty(*ptr),
                    }
                    .into());
                }
                if val_ty(*value) == Type::Void {
                    return Err(ValidationError::Other(alloc::format!(
                        "Cannot store void type at {:?}",
                        inst
                    ))
                    .into());
                }
            }
            InstructionData::StackLoad { .. } => {
                let result_ty = dfg
                    .inst_results(inst)
                    .map(|v| val_ty(v))
                    .unwrap_or(Type::Void);
                if result_ty == Type::Void {
                    return Err(ValidationError::Other(alloc::format!(
                        "Cannot stack_load void type at {:?}",
                        inst
                    ))
                    .into());
                }
            }
            InstructionData::StackStore { value, .. } => {
                if val_ty(*value) == Type::Void {
                    return Err(ValidationError::Other(alloc::format!(
                        "Cannot stack_store void type at {:?}",
                        inst
                    ))
                    .into());
                }
            }
            InstructionData::StackAddr { .. } => {}
            InstructionData::PtrOffset { ptr, .. } => {
                if val_ty(*ptr) != Type::Ptr {
                    return Err(ValidationError::TypeMismatch {
                        opcode: Opcode::PtrOffset,
                        expected: Type::Ptr,
                        got: val_ty(*ptr),
                    }
                    .into());
                }
            }
            InstructionData::PtrIndex { ptr, index, .. } => {
                if val_ty(*ptr) != Type::Ptr {
                    return Err(ValidationError::TypeMismatch {
                        opcode: Opcode::PtrIndex,
                        expected: Type::Ptr,
                        got: val_ty(*ptr),
                    }
                    .into());
                }
                if !val_ty(*index).is_integer() {
                    return Err(ValidationError::Other(alloc::format!(
                        "PtrIndex index must be an integer, got {:?}",
                        val_ty(*index)
                    ))
                    .into());
                }
            }
            InstructionData::IntToPtr { arg } => {
                if !val_ty(*arg).is_integer() {
                    return Err(ValidationError::TypeMismatch {
                        opcode: Opcode::IntToPtr,
                        expected: Type::I64,
                        got: val_ty(*arg),
                    }
                    .into());
                }
            }
            InstructionData::PtrToInt { arg } => {
                if val_ty(*arg) != Type::Ptr {
                    return Err(ValidationError::TypeMismatch {
                        opcode: Opcode::PtrToInt,
                        expected: Type::Ptr,
                        got: val_ty(*arg),
                    }
                    .into());
                }
                let result_ty = dfg
                    .inst_results(inst)
                    .map(|v| val_ty(v))
                    .unwrap_or(Type::Void);
                if !result_ty.is_integer() {
                    return Err(ValidationError::TypeMismatch {
                        opcode: Opcode::PtrToInt,
                        expected: Type::I64,
                        got: result_ty,
                    }
                    .into());
                }
            }
            InstructionData::Iconst { .. }
            | InstructionData::Fconst { .. }
            | InstructionData::Bconst { .. } => {}
            InstructionData::Call { func_id, args } => {
                let callee = &module.functions[*func_id];
                let sig = &module.signatures[callee.signature];

                let call_args = dfg.get_value_list(*args);
                if call_args.len() != sig.params.len() {
                    return Err(ValidationError::Other(alloc::format!(
                        "Call to {} argument count mismatch: expected {}, got {}",
                        callee.name,
                        sig.params.len(),
                        call_args.len()
                    ))
                    .into());
                }

                for (i, (&arg, &expected_ty)) in call_args.iter().zip(sig.params.iter()).enumerate()
                {
                    let got_ty = val_ty(arg);
                    if got_ty != expected_ty {
                        return Err(ValidationError::Other(alloc::format!(
                            "Call to {} argument {} type mismatch: expected {:?}, got {:?}",
                            callee.name,
                            i,
                            expected_ty,
                            got_ty
                        ))
                        .into());
                    }
                }
            }
            InstructionData::CallIndirect { ptr, args, sig_id } => {
                if val_ty(*ptr) != Type::Ptr {
                    return Err(ValidationError::TypeMismatch {
                        opcode: Opcode::CallIndirect,
                        expected: Type::Ptr,
                        got: val_ty(*ptr),
                    }
                    .into());
                }

                let sig = &module.signatures[*sig_id];
                let call_args = dfg.get_value_list(*args);
                if call_args.len() != sig.params.len() {
                    return Err(ValidationError::Other(alloc::format!(
                        "CallIndirect argument count mismatch: expected {}, got {}",
                        sig.params.len(),
                        call_args.len()
                    ))
                    .into());
                }

                for (i, (&arg, &expected_ty)) in call_args.iter().zip(sig.params.iter()).enumerate()
                {
                    let got_ty = val_ty(arg);
                    if got_ty != expected_ty {
                        return Err(ValidationError::Other(alloc::format!(
                            "CallIndirect argument {} type mismatch: expected {:?}, got {:?}",
                            i,
                            expected_ty,
                            got_ty
                        ))
                        .into());
                    }
                }
            }
            InstructionData::Jump { dest } => {
                let dest_data = dfg.block_calls[*dest];
                let target_block = dest_data.block;
                let expected_params = &self.layout.blocks[target_block].params;
                let actual_args = dfg.get_value_list(dest_data.args);

                if expected_params.len() != actual_args.len() {
                    return Err(ValidationError::Other(alloc::format!(
                        "Jump to block {:?} has {} arguments, but block expects {}",
                        target_block,
                        actual_args.len(),
                        expected_params.len()
                    ))
                    .into());
                }

                for (i, (&param, &arg)) in
                    expected_params.iter().zip(actual_args.iter()).enumerate()
                {
                    let expected_ty = dfg.values[param].ty;
                    let actual_ty = dfg.values[arg].ty;
                    if expected_ty != actual_ty {
                        return Err(ValidationError::Other(alloc::format!(
                            "Jump to block {:?} argument {} type mismatch: expected {:?}, got {:?}",
                            target_block,
                            i,
                            expected_ty,
                            actual_ty
                        ))
                        .into());
                    }
                }
            }
            InstructionData::Br {
                condition,
                then_dest,
                else_dest,
            } => {
                let cond_ty = val_ty(*condition);
                if cond_ty != Type::Bool {
                    return Err(ValidationError::ConditionNotBool(inst, cond_ty).into());
                }

                for dest in [then_dest, else_dest] {
                    let dest_data = dfg.block_calls[*dest];
                    let target_block = dest_data.block;
                    let expected_params = &self.layout.blocks[target_block].params;
                    let actual_args = dfg.get_value_list(dest_data.args);

                    if expected_params.len() != actual_args.len() {
                        return Err(ValidationError::Other(alloc::format!(
                            "Branch to block {:?} has {} arguments, but block expects {}",
                            target_block,
                            actual_args.len(),
                            expected_params.len()
                        ))
                        .into());
                    }

                    for (i, (&param, &arg)) in
                        expected_params.iter().zip(actual_args.iter()).enumerate()
                    {
                        let expected_ty = dfg.values[param].ty;
                        let actual_ty = dfg.values[arg].ty;
                        if expected_ty != actual_ty {
                            return Err(ValidationError::Other(alloc::format!(
                                "Branch to block {:?} argument {} type mismatch: expected {:?}, got {:?}",
                                target_block, i, expected_ty, actual_ty
                            ))
                            .into());
                        }
                    }
                }
            }
            InstructionData::BrTable { index, table } => {
                let idx_ty = val_ty(*index);
                if idx_ty != Type::I32 {
                    return Err(ValidationError::TypeMismatch {
                        opcode: Opcode::BrTable,
                        expected: Type::I32,
                        got: idx_ty,
                    }
                    .into());
                }

                let table_data = &dfg.jump_tables[*table];
                for target_call in table_data.targets.iter() {
                    let dest_data = dfg.block_calls[*target_call];
                    let target_block = dest_data.block;
                    let expected_params = &self.layout.blocks[target_block].params;
                    let actual_args = dfg.get_value_list(dest_data.args);

                    if expected_params.len() != actual_args.len() {
                        return Err(ValidationError::Other(alloc::format!(
                            "BrTable target block {:?} has {} arguments, but block expects {}",
                            target_block,
                            actual_args.len(),
                            expected_params.len()
                        ))
                        .into());
                    }

                    for (i, (&param, &arg)) in
                        expected_params.iter().zip(actual_args.iter()).enumerate()
                    {
                        let expected_ty = dfg.values[param].ty;
                        let actual_ty = dfg.values[arg].ty;
                        if expected_ty != actual_ty {
                            return Err(ValidationError::Other(alloc::format!(
                                "BrTable target block {:?} argument {} type mismatch: expected {:?}, got {:?}",
                                target_block, i, expected_ty, actual_ty
                            ))
                            .into());
                        }
                    }
                }
            }
            InstructionData::Return { value } => {
                let sig = &module.signatures[self.signature];
                let expected = sig.ret;
                let got = value.map(|v| val_ty(v)).unwrap_or(Type::Void);
                if expected != got {
                    return Err(ValidationError::ReturnMismatch { expected, got }.into());
                }
            }
            InstructionData::Select {
                condition,
                then_val,
                else_val,
            } => {
                let cond_ty = val_ty(*condition);
                if cond_ty != Type::Bool {
                    return Err(ValidationError::ConditionNotBool(inst, cond_ty).into());
                }
                let t_ty = val_ty(*then_val);
                let f_ty = val_ty(*else_val);
                let result_ty = dfg
                    .inst_results(inst)
                    .map(|v| val_ty(v))
                    .unwrap_or(Type::Void);
                if t_ty != f_ty || t_ty != result_ty {
                    return Err(ValidationError::SelectMismatch {
                        inst,
                        expected: result_ty,
                        then_val: t_ty,
                        else_val: f_ty,
                    }
                    .into());
                }
            }
            InstructionData::IntCompare { args, .. }
            | InstructionData::FloatCompare { args, .. } => {
                let lhs_ty = val_ty(args[0]);
                let rhs_ty = val_ty(args[1]);
                if lhs_ty != rhs_ty {
                    return Err(ValidationError::OperandTypeMismatch {
                        inst,
                        lhs: lhs_ty,
                        rhs: rhs_ty,
                    }
                    .into());
                }
                let result_ty = dfg
                    .inst_results(inst)
                    .map(|v| val_ty(v))
                    .unwrap_or(Type::Void);
                if result_ty != Type::Bool {
                    return Err(ValidationError::ConditionNotBool(inst, result_ty).into());
                }
            }
            InstructionData::CallIntrinsic {
                intrinsic: _,
                args,
                sig_id: _,
            } => {
                // Validate that the intrinsic is valid by checking its name
                // Note: validation logic can be extended here
                let _args = dfg.get_value_list(*args);
            }
            InstructionData::ExtractValue { val, index } => {
                let value_type = val_ty(*val);
                // The value being extracted from must be a MultiValue
                if value_type != Type::MultiValue {
                    return Err(ValidationError::TypeMismatch {
                        opcode: Opcode::ExtractValue,
                        expected: Type::MultiValue,
                        got: value_type,
                    }
                    .into());
                }
                let result_ty = dfg
                    .inst_results(inst)
                    .map(|v| val_ty(v))
                    .unwrap_or(Type::Void);
                // Get the component types of the MultiValue by finding its producer
                if let Some(component_types) = self.get_multivalue_components(*val, module) {
                    let idx = *index as usize;
                    if idx >= component_types.len() {
                        return Err(ValidationError::Other(alloc::format!(
                            "ExtractValue index {} out of bounds (MultiValue has {} components)",
                            idx,
                            component_types.len()
                        ))
                        .into());
                    }
                    let expected_ty = component_types[idx];
                    if expected_ty != result_ty {
                        return Err(ValidationError::TypeMismatch {
                            opcode: Opcode::ExtractValue,
                            expected: expected_ty,
                            got: result_ty,
                        }
                        .into());
                    }
                }
                // If we can't determine the component types (e.g., from a parameter),
                // we skip the detailed check
            }
            InstructionData::ConstructMulti { values } => {
                let _vals = dfg.get_value_list(*values);
                // All values being combined must not be Void or MultiValue
                for &v in _vals {
                    let t = val_ty(v);
                    if t == Type::Void || t == Type::MultiValue {
                        return Err(ValidationError::Other(alloc::format!(
                            "ConstructMulti cannot include values of type {:?}",
                            t
                        ))
                        .into());
                    }
                }
            }
            InstructionData::Unreachable => {}
            InstructionData::Nop => {}
        }

        Ok(())
    }

    /// Get the component types of a MultiValue by analyzing its producer.
    /// Returns None if the producer cannot be determined (e.g., block parameter).
    fn get_multivalue_components(&self, val: Value, module: &ModuleData) -> Option<Vec<Type>> {
        use crate::ValueDef;

        match self.dfg.value_def(val) {
            ValueDef::Inst(inst) => {
                let data = &self.dfg.instructions[inst];
                match data {
                    InstructionData::Call { func_id, .. } => {
                        let callee = &module.functions[*func_id];
                        let sig = &module.signatures[callee.signature];
                        if sig.ret == Type::MultiValue {
                            // For now, we don't store multi-value component types in signature
                            // This would need to be extended when signatures support multi-value returns
                            None
                        } else {
                            None
                        }
                    }
                    InstructionData::CallIndirect { sig_id, .. } => {
                        let sig = &module.signatures[*sig_id];
                        if sig.ret == Type::MultiValue {
                            None // Same as above
                        } else {
                            None
                        }
                    }
                    InstructionData::CallIntrinsic { sig_id, .. } => {
                        let sig = &module.signatures[*sig_id];
                        if sig.ret == Type::MultiValue {
                            None // Same as above
                        } else {
                            None
                        }
                    }
                    InstructionData::ConstructMulti { values } => {
                        let vals = self.dfg.get_value_list(*values);
                        Some(vals.iter().map(|&v| self.dfg.value_type(v)).collect())
                    }
                    _ => None,
                }
            }
            ValueDef::Param(_) => None, // Block parameters don't have known component types
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Linkage;
    use crate::builder::ModuleBuilder;

    #[test]
    fn test_unsealed_block_validation() {
        let mut mb = ModuleBuilder::new();
        let sig_id = mb.make_signature(vec![], Type::Void, crate::CallConv::SystemV);
        let func_id = mb.declare_function("test".to_string(), sig_id, Linkage::Export);
        let mut builder = mb.builder(func_id);

        // Explicitly initialize entry block
        builder.init_entry_block();
        // and give it a terminator
        builder.ins().ret(None);

        // Create an additional block and don't seal it
        let block = builder.create_block();
        builder.switch_to_block(block);
        builder.ins().ret(None);

        drop(builder);
        let res = mb.validate();
        match res {
            Err(e) => {
                let err = e.to_string();
                if !err.contains("is not sealed") {
                    panic!("Expected 'is not sealed' error, got: {}", err);
                }
            }
            Ok(_) => panic!("Expected validation error, got Ok"),
        }
    }
}
