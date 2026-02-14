use crate::{Block, Function, Inst, InstructionData, ModuleData, Opcode, Result, Type, Value};
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
            let insts = &layout.blocks[*block].insts;
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
            InstructionData::Unary { opcode, arg, ty } => {
                let arg_ty = val_ty(*arg);
                if arg_ty == Type::Ptr || *ty == Type::Ptr {
                    return Err(ValidationError::PointerArithmetic(inst, *opcode).into());
                }

                match opcode {
                    Opcode::Ineg | Opcode::Clz | Opcode::Ctz | Opcode::Popcnt => {
                        if !ty.is_integer() || arg_ty != *ty {
                            return Err(ValidationError::TypeMismatch {
                                opcode: *opcode,
                                expected: *ty,
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
                        if !ty.is_float() || arg_ty != *ty {
                            return Err(ValidationError::TypeMismatch {
                                opcode: *opcode,
                                expected: *ty,
                                got: arg_ty,
                            }
                            .into());
                        }
                    }
                    Opcode::Eqz => {
                        if !arg_ty.is_integer() || *ty != Type::Bool {
                            return Err(ValidationError::TypeMismatch {
                                opcode: Opcode::Eqz,
                                expected: Type::Bool,
                                got: *ty,
                            }
                            .into());
                        }
                    }
                    // Conversion operators
                    Opcode::ExtendS | Opcode::ExtendU => {
                        let is_valid = if arg_ty == Type::Bool {
                            *opcode == Opcode::ExtendU && ty.is_integer()
                        } else {
                            arg_ty.is_integer()
                                && ty.is_integer()
                                && arg_ty.size_bytes() < ty.size_bytes()
                        };

                        if !is_valid {
                            return Err(ValidationError::InvalidConversion {
                                inst,
                                opcode: *opcode,
                                from: arg_ty,
                                to: *ty,
                            }
                            .into());
                        }
                    }
                    Opcode::Wrap => {
                        if !arg_ty.is_integer()
                            || !ty.is_integer()
                            || arg_ty.size_bytes() <= ty.size_bytes()
                        {
                            return Err(ValidationError::InvalidConversion {
                                inst,
                                opcode: Opcode::Wrap,
                                from: arg_ty,
                                to: *ty,
                            }
                            .into());
                        }
                    }
                    Opcode::Promote => {
                        if arg_ty != Type::F32 || *ty != Type::F64 {
                            return Err(ValidationError::InvalidConversion {
                                inst,
                                opcode: Opcode::Promote,
                                from: arg_ty,
                                to: *ty,
                            }
                            .into());
                        }
                    }
                    Opcode::Demote => {
                        if arg_ty != Type::F64 || *ty != Type::F32 {
                            return Err(ValidationError::InvalidConversion {
                                inst,
                                opcode: Opcode::Demote,
                                from: arg_ty,
                                to: *ty,
                            }
                            .into());
                        }
                    }
                    Opcode::TruncS | Opcode::TruncU => {
                        if !arg_ty.is_float() || !ty.is_integer() {
                            return Err(ValidationError::InvalidConversion {
                                inst,
                                opcode: *opcode,
                                from: arg_ty,
                                to: *ty,
                            }
                            .into());
                        }
                    }
                    Opcode::ConvertS | Opcode::ConvertU => {
                        if !arg_ty.is_integer() || !ty.is_float() {
                            return Err(ValidationError::InvalidConversion {
                                inst,
                                opcode: *opcode,
                                from: arg_ty,
                                to: *ty,
                            }
                            .into());
                        }
                    }
                    Opcode::Reinterpret => {
                        if arg_ty.size_bytes() != ty.size_bytes() || arg_ty == *ty {
                            return Err(ValidationError::InvalidConversion {
                                inst,
                                opcode: Opcode::Reinterpret,
                                from: arg_ty,
                                to: *ty,
                            }
                            .into());
                        }
                    }
                    _ => {}
                }
            }
            InstructionData::Binary { opcode, args, ty } => {
                let lhs_ty = val_ty(args[0]);
                let rhs_ty = val_ty(args[1]);

                if lhs_ty == Type::Ptr || rhs_ty == Type::Ptr || *ty == Type::Ptr {
                    return Err(ValidationError::PointerArithmetic(inst, *opcode).into());
                }

                if lhs_ty != *ty || rhs_ty != *ty {
                    return Err(ValidationError::OperandTypeMismatch {
                        inst,
                        lhs: lhs_ty,
                        rhs: rhs_ty,
                    }
                    .into());
                }
            }
            InstructionData::Load { ptr, ty, .. } => {
                if val_ty(*ptr) != Type::Ptr {
                    return Err(ValidationError::TypeMismatch {
                        opcode: Opcode::Load,
                        expected: Type::Ptr,
                        got: val_ty(*ptr),
                    }
                    .into());
                }
                if *ty == Type::Void {
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
            InstructionData::StackLoad { ty, .. } => {
                if *ty == Type::Void {
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
            InstructionData::PtrToInt { arg, ty } => {
                if val_ty(*arg) != Type::Ptr {
                    return Err(ValidationError::TypeMismatch {
                        opcode: Opcode::PtrToInt,
                        expected: Type::Ptr,
                        got: val_ty(*arg),
                    }
                    .into());
                }
                if !ty.is_integer() {
                    return Err(ValidationError::TypeMismatch {
                        opcode: Opcode::PtrToInt,
                        expected: Type::I64,
                        got: *ty,
                    }
                    .into());
                }
            }
            InstructionData::Iconst { .. }
            | InstructionData::Fconst { .. }
            | InstructionData::Bconst { .. } => {}
            InstructionData::Call {
                func_id,
                args,
                ret_ty,
            } => {
                let callee = &module.functions[*func_id];
                let sig = &module.signatures[callee.signature];

                if sig.ret != *ret_ty {
                    return Err(ValidationError::TypeMismatch {
                        opcode: Opcode::Call,
                        expected: sig.ret,
                        got: *ret_ty,
                    }
                    .into());
                }

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
            InstructionData::CallIndirect {
                ptr,
                args,
                sig_id,
                ret_ty,
            } => {
                if val_ty(*ptr) != Type::Ptr {
                    return Err(ValidationError::TypeMismatch {
                        opcode: Opcode::CallIndirect,
                        expected: Type::Ptr,
                        got: val_ty(*ptr),
                    }
                    .into());
                }

                let sig = &module.signatures[*sig_id];
                if sig.ret != *ret_ty {
                    return Err(ValidationError::TypeMismatch {
                        opcode: Opcode::CallIndirect,
                        expected: sig.ret,
                        got: *ret_ty,
                    }
                    .into());
                }

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
                let _dest_data = dfg.block_calls[*dest];
                // TODO: we need to check block parameter types, but layout doesn't have it easily accessible here
                // without looking at layout.blocks[dest].params and then checking their types in dfg.
            }
            InstructionData::Br { condition, .. } => {
                let cond_ty = val_ty(*condition);
                if cond_ty != Type::Bool {
                    return Err(ValidationError::ConditionNotBool(inst, cond_ty).into());
                }
            }
            InstructionData::BrTable { index, .. } => {
                if !val_ty(*index).is_integer() {
                    return Err(ValidationError::Other(alloc::format!(
                        "br_table index must be integer"
                    ))
                    .into());
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
                ty,
            } => {
                let cond_ty = val_ty(*condition);
                if cond_ty != Type::Bool {
                    return Err(ValidationError::ConditionNotBool(inst, cond_ty).into());
                }
                let t_ty = val_ty(*then_val);
                let f_ty = val_ty(*else_val);
                if t_ty != f_ty || t_ty != *ty {
                    return Err(ValidationError::SelectMismatch {
                        inst,
                        expected: *ty,
                        then_val: t_ty,
                        else_val: f_ty,
                    }
                    .into());
                }
            }
            InstructionData::IntCompare { args, ty, .. }
            | InstructionData::FloatCompare { args, ty, .. } => {
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
                if *ty != Type::Bool {
                    return Err(ValidationError::ConditionNotBool(inst, *ty).into());
                }
            }
            InstructionData::Unreachable => {}
        }

        Ok(())
    }
}
