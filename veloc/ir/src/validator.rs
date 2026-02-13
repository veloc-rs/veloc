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
            Self::PointerArithmetic(inst, opcode) => {
                write!(
                    f,
                    "Pointer arithmetic not allowed for instruction {:?} ({:?}). Use GEP instead.",
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

                // Pointers cannot be operands or results of Unary arithmetic/logic
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
                            return Err(ValidationError::ConditionNotBool(inst, *ty).into());
                        }
                    }
                    _ => {}
                }
            }
            InstructionData::Binary { opcode, args, ty } => {
                let lhs_ty = val_ty(args[0]);
                let rhs_ty = val_ty(args[1]);

                // Pointers cannot participate in Binary arithmetic/logic
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
            InstructionData::Load { ptr, .. } | InstructionData::Store { ptr, .. } => {
                if val_ty(*ptr) != Type::Ptr {
                    return Err(ValidationError::TypeMismatch {
                        opcode: data.opcode(),
                        expected: Type::Ptr,
                        got: val_ty(*ptr),
                    }
                    .into());
                }
            }
            InstructionData::Gep { ptr, offset } => {
                if val_ty(*ptr) != Type::Ptr {
                    return Err(ValidationError::TypeMismatch {
                        opcode: Opcode::Gep,
                        expected: Type::Ptr,
                        got: val_ty(*ptr),
                    }
                    .into());
                }
                if !val_ty(*offset).is_integer() {
                    return Err(ValidationError::Other(alloc::format!(
                        "GEP offset must be an integer, got {:?}",
                        val_ty(*offset)
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
            InstructionData::CallIndirect { ptr, .. } => {
                if val_ty(*ptr) != Type::Ptr {
                    return Err(ValidationError::TypeMismatch {
                        opcode: Opcode::CallIndirect,
                        expected: Type::Ptr,
                        got: val_ty(*ptr),
                    }
                    .into());
                }
                // TODO: check argument types
            }
            InstructionData::IntCompare { kind: _, args, ty } => {
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
            InstructionData::FloatCompare { kind: _, args, ty } => {
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
            InstructionData::Jump { .. } => {
                // TODO: check block arguments type match
            }
            InstructionData::Br { condition, .. } => {
                let cond_ty = val_ty(*condition);
                if cond_ty != Type::Bool {
                    return Err(ValidationError::ConditionNotBool(inst, cond_ty).into());
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
            InstructionData::Call { func_id, .. } => {
                let callee = &module.functions[*func_id];
                let _sig = &module.signatures[callee.signature];
                // TODO: check argument types
            }
            InstructionData::Select {
                condition,
                then_val,
                else_val,
                ..
            } => {
                let cond_ty = val_ty(*condition);
                if cond_ty != Type::Bool {
                    return Err(ValidationError::ConditionNotBool(inst, cond_ty).into());
                }
                let t_ty = val_ty(*then_val);
                let f_ty = val_ty(*else_val);
                if t_ty != f_ty {
                    return Err(ValidationError::OperandTypeMismatch {
                        inst,
                        lhs: t_ty,
                        rhs: f_ty,
                    }
                    .into());
                }
            }
            _ => {}
        }

        Ok(())
    }
}
