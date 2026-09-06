use crate::inst::{ConstantPoolData, Inst, VectorExtId, VectorMemExtId};
use crate::opspec::OpConstraint;
use crate::{
    Block, BlockCall, Function, InstructionData, IntCC, ModuleData, Opcode, Result, SigId, Type,
    Value, ValueList,
};
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
        index: usize,
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
            Self::EmptyBlock(block) => write!(f, "Block {:?} is empty", block),
            Self::NoTerminator(block) => {
                write!(f, "Block {:?} does not end with a terminator", block)
            }
            Self::TypeMismatch {
                opcode,
                expected,
                got,
            } => write!(
                f,
                "Instruction {:?} type mismatch: expected {:?}, got {:?}",
                opcode, expected, got
            ),
            Self::ReturnMismatch {
                index,
                expected,
                got,
            } => write!(
                f,
                "Return type mismatch at index {}: expected {:?}, got {:?}",
                index, expected, got
            ),
            Self::OperandTypeMismatch { inst, lhs, rhs } => write!(
                f,
                "Operand type mismatch for {:?}: lhs {:?}, rhs {:?}",
                inst, lhs, rhs
            ),
            Self::ConditionNotBool(inst, got) => {
                write!(f, "Condition for {:?} must be Bool, got {:?}", inst, got)
            }
            Self::SelectMismatch {
                inst,
                expected,
                then_val,
                else_val,
            } => write!(
                f,
                "Select {:?} type mismatch: expected {:?}, got {:?} and {:?}",
                inst, expected, then_val, else_val
            ),
            Self::InvalidConversion {
                inst,
                opcode,
                from,
                to,
            } => write!(
                f,
                "Invalid conversion in {:?} for {:?}: from {:?} to {:?}",
                inst, opcode, from, to
            ),
            Self::PointerArithmetic(inst, opcode) => write!(
                f,
                "Pointer arithmetic not allowed for instruction {:?} ({:?})",
                inst, opcode
            ),
            Self::UnsealedBlock(block) => write!(f, "Block {:?} is not sealed", block),
            Self::Other(message) => write!(f, "{}", message),
        }
    }
}

impl ModuleData {
    pub fn validate(&self) -> Result<()> {
        for (_, function) in self.functions.iter() {
            function.validate(self).map_err(|error| {
                crate::Error::Message(alloc::format!("In function {}: {}", function.name, error))
            })?;
        }
        Ok(())
    }
}

impl Function {
    pub fn validate(&self, module: &ModuleData) -> Result<()> {
        for &block in &self.layout.block_order {
            let block_data = &self.layout.blocks[block];
            if !block_data.is_sealed {
                return Err(ValidationError::UnsealedBlock(block).into());
            }
            if block_data.insts.is_empty() {
                return Err(ValidationError::EmptyBlock(block).into());
            }
            for &inst in &block_data.insts {
                self.validate_inst(module, inst)?;
            }
            let terminator = *block_data.insts.last().unwrap();
            if !self.dfg.instructions[terminator].is_terminator() {
                return Err(ValidationError::NoTerminator(block).into());
            }
        }
        Ok(())
    }

    fn validate_inst(&self, module: &ModuleData, inst: Inst) -> Result<()> {
        let data = &self.dfg.instructions[inst];
        let opcode = data.opcode();
        let spec = opcode.spec();

        if !data.matches_format(&self.dfg, spec.format) {
            return self.fail(alloc::format!(
                "{} at {:?} is stored in an incompatible instruction format",
                spec.mnemonic,
                inst
            ));
        }

        let mut operands = alloc::vec::Vec::new();
        data.visit_type_operands(&self.dfg, |value| {
            operands.push(self.dfg.value_type(value));
        });
        let results = self
            .dfg
            .inst_results(inst)
            .iter()
            .map(|&value| self.dfg.value_type(value))
            .collect::<alloc::vec::Vec<_>>();
        spec.type_scheme
            .validate(&operands, &results)
            .map_err(|error| {
                crate::Error::from(ValidationError::Other(alloc::format!(
                    "{} type scheme violation at {:?}: {:?}",
                    spec.mnemonic,
                    inst,
                    error
                )))
            })?;

        for &constraint in spec.constraints {
            self.validate_constraint(inst, data, constraint, &results)?;
        }

        if let Some(call) = data.call_info() {
            let Some(signature) = call.signature.resolve(module) else {
                return self.fail(alloc::format!(
                    "{} at {:?} refers to a missing function or signature",
                    spec.mnemonic,
                    inst
                ));
            };
            self.validate_signature(module, inst, signature, call.args, spec.mnemonic)?;
        }
        let mut successors = Ok(());
        data.visit_successors(&self.dfg, |call| {
            if successors.is_ok() {
                successors = self.validate_block_call(call, spec.mnemonic);
            }
        });
        successors?;

        match data {
            InstructionData::BrTable { table, .. }
                if self.dfg.jump_table_targets(*table).is_empty() =>
            {
                return self.fail("branch table must contain a default destination".into());
            }
            InstructionData::Return { values } => {
                self.validate_values(
                    "return",
                    self.dfg.get_value_list(*values),
                    &module.signatures[self.signature].returns,
                )?;
            }
            InstructionData::VectorOpWithExt { ext, .. } => {
                let vector_ty = results.first().copied().ok_or_else(|| {
                    crate::Error::from(ValidationError::Other(alloc::format!(
                        "predicated {} at {:?} has no result",
                        spec.mnemonic,
                        inst
                    )))
                })?;
                if !vector_ty.is_vector() {
                    return self.fail(alloc::format!(
                        "predicated {} at {:?} must produce a vector, got {}",
                        spec.mnemonic,
                        inst,
                        vector_ty
                    ));
                }
                self.validate_vector_ext(inst, opcode, *ext, vector_ty)?;
            }
            InstructionData::VectorLoadStrided { ext, .. }
            | InstructionData::VectorGather { ext, .. } => {
                self.validate_vector_mem_ext(inst, opcode, *ext, results[0])?;
            }
            InstructionData::VectorStoreStrided { args, ext }
            | InstructionData::VectorScatter { args, ext } => {
                let values = self.dfg.get_value_list(*args);
                self.validate_vector_mem_ext(inst, opcode, *ext, self.dfg.value_type(values[2]))?;
            }
            _ => {}
        }

        Ok(())
    }

    fn validate_constraint(
        &self,
        inst: Inst,
        data: &InstructionData,
        constraint: OpConstraint,
        results: &[Type],
    ) -> Result<()> {
        match (constraint, data) {
            (OpConstraint::PointerComparison, InstructionData::IntCompare { kind, args }) => {
                if self.dfg.value_type(args[0]).is_ptr() && !matches!(kind, IntCC::Eq | IntCC::Ne) {
                    return self.fail(alloc::format!(
                        "pointer comparison at {:?} only supports eq and ne",
                        inst
                    ));
                }
            }
            (OpConstraint::NonZeroScale, InstructionData::PtrIndex { imm_id, .. }) => {
                let imm = self.dfg.ptr_imm(*imm_id).ok_or_else(|| {
                    crate::Error::from(ValidationError::Other(alloc::format!(
                        "ptr-index at {:?} refers to missing immediate {:?}",
                        inst,
                        imm_id
                    )))
                })?;
                if imm.scale == 0 {
                    return self.fail(alloc::format!(
                        "ptr-index scale at {:?} must be non-zero",
                        inst
                    ));
                }
            }
            (OpConstraint::VectorConstant, InstructionData::Vconst { pool_id }) => {
                let expected = results
                    .first()
                    .and_then(|ty| ty.min_size_bytes())
                    .ok_or_else(|| {
                        crate::Error::from(ValidationError::Other(alloc::format!(
                            "vconst at {:?} has no statically sized vector result",
                            inst
                        )))
                    })? as usize;
                let data = self.dfg.constant_pool_data(*pool_id).ok_or_else(|| {
                    crate::Error::from(ValidationError::Other(alloc::format!(
                        "vconst at {:?} refers to missing constant {:?}",
                        inst,
                        pool_id
                    )))
                })?;
                let ConstantPoolData::Bytes(bytes) = data;
                if bytes.len() != expected {
                    return self.fail(alloc::format!(
                        "vconst at {:?} requires {} bytes, got {}",
                        inst,
                        expected,
                        bytes.len()
                    ));
                }
            }
            (OpConstraint::ShuffleMask, InstructionData::Shuffle { mask, .. }) => {
                let result_ty = results[0];
                if !result_ty.is_fixed() {
                    return self.fail(alloc::format!(
                        "shuffle at {:?} requires a fixed-width vector",
                        inst
                    ));
                }
                let data = self.dfg.constant_pool_data(*mask).ok_or_else(|| {
                    crate::Error::from(ValidationError::Other(alloc::format!(
                        "shuffle at {:?} refers to missing mask {:?}",
                        inst,
                        mask
                    )))
                })?;
                let ConstantPoolData::Bytes(bytes) = data;
                let lanes = usize::from(result_ty.lane_count());
                if bytes.len() != lanes || bytes.iter().any(|&lane| usize::from(lane) >= 2 * lanes)
                {
                    return self.fail(alloc::format!(
                        "shuffle mask at {:?} must contain {} selectors in 0..{}",
                        inst,
                        lanes,
                        2 * lanes
                    ));
                }
            }
            _ => unreachable!(
                "OpSpec constraint {:?} is incompatible with {:?}",
                constraint, data
            ),
        }
        Ok(())
    }

    fn validate_signature(
        &self,
        module: &ModuleData,
        inst: Inst,
        sig_id: SigId,
        args: ValueList,
        name: &str,
    ) -> Result<()> {
        let signature = &module.signatures[sig_id];
        self.validate_values(name, self.dfg.get_value_list(args), &signature.params)?;

        let results = self.dfg.inst_results(inst);
        if results.len() != signature.returns.len() {
            return self.fail(alloc::format!(
                "{} result count mismatch: expected {}, got {}",
                name,
                signature.returns.len(),
                results.len()
            ));
        }
        for (index, (&result, &expected)) in
            results.iter().zip(signature.returns.iter()).enumerate()
        {
            let got = self.dfg.value_type(result);
            if got != expected {
                return self.fail(alloc::format!(
                    "{} result {} type mismatch: expected {}, got {}",
                    name,
                    index,
                    expected,
                    got
                ));
            }
        }
        Ok(())
    }

    fn validate_values(&self, name: &str, values: &[Value], expected: &[Type]) -> Result<()> {
        if values.len() != expected.len() {
            return self.fail(alloc::format!(
                "{} value count mismatch: expected {}, got {}",
                name,
                expected.len(),
                values.len()
            ));
        }
        for (index, (&value, &expected)) in values.iter().zip(expected).enumerate() {
            let got = self.dfg.value_type(value);
            if got != expected {
                return self.fail(alloc::format!(
                    "{} value {} type mismatch: expected {}, got {}",
                    name,
                    index,
                    expected,
                    got
                ));
            }
        }
        Ok(())
    }

    fn validate_block_call(&self, call: BlockCall, kind: &str) -> Result<()> {
        let call_data = &self.dfg.block_calls[call];
        let params = &self.layout.blocks[call_data.block].params;
        self.validate_values(
            kind,
            self.dfg.get_value_list(call_data.args),
            &self.value_types(params),
        )
    }

    fn value_types(&self, values: &[Value]) -> alloc::vec::Vec<Type> {
        values
            .iter()
            .map(|&value| self.dfg.value_type(value))
            .collect()
    }

    fn validate_vector_ext(
        &self,
        inst: Inst,
        opcode: Opcode,
        ext: VectorExtId,
        vector_ty: Type,
    ) -> Result<()> {
        let ext = self.dfg.vector_ext(ext).ok_or_else(|| {
            crate::Error::from(ValidationError::Other(alloc::format!(
                "vector operation at {:?} refers to missing extension {:?}",
                inst,
                ext
            )))
        })?;
        self.validate_mask(inst, opcode, ext.mask, vector_ty)?;
        if let Some(evl) = ext.evl {
            self.validate_evl(inst, opcode, evl)?;
        }
        Ok(())
    }

    fn validate_vector_mem_ext(
        &self,
        inst: Inst,
        opcode: Opcode,
        ext: VectorMemExtId,
        vector_ty: Type,
    ) -> Result<()> {
        let ext = self.dfg.vector_mem_ext(ext).ok_or_else(|| {
            crate::Error::from(ValidationError::Other(alloc::format!(
                "vector memory operation at {:?} refers to missing extension {:?}",
                inst,
                ext
            )))
        })?;
        if let Some(mask) = ext.mask {
            self.validate_mask(inst, opcode, mask, vector_ty)?;
        }
        if let Some(evl) = ext.evl {
            self.validate_evl(inst, opcode, evl)?;
        }
        Ok(())
    }

    fn validate_mask(
        &self,
        inst: Inst,
        opcode: Opcode,
        mask: Value,
        vector_ty: Type,
    ) -> Result<()> {
        let mask_ty = self.dfg.value_type(mask);
        if !mask_ty.is_predicate() || mask_ty.vector_shape() != vector_ty.vector_shape() {
            return self.fail(alloc::format!(
                "{} mask at {:?} must match vector shape {:?}, got {}",
                opcode.spec().mnemonic,
                inst,
                vector_ty.vector_shape(),
                mask_ty
            ));
        }
        Ok(())
    }

    fn validate_evl(&self, inst: Inst, opcode: Opcode, evl: Value) -> Result<()> {
        let evl_ty = self.dfg.value_type(evl);
        if evl_ty != Type::I32 {
            return self.fail(alloc::format!(
                "{} EVL at {:?} must be i32, got {}",
                opcode.spec().mnemonic,
                inst,
                evl_ty
            ));
        }
        Ok(())
    }

    fn fail<T>(&self, message: String) -> Result<T> {
        Err(ValidationError::Other(message).into())
    }
}

#[cfg(test)]
mod tests {
    use crate::builder::ModuleBuilder;
    use crate::{CallConv, IntCC, Linkage, Type};

    #[test]
    fn branch_tables_require_a_default_destination() {
        let mut module = ModuleBuilder::new();
        let sig = module.make_signature(vec![], vec![], CallConv::SystemV);
        let func = module.declare_function("empty-table".into(), sig, Linkage::Local);
        {
            let mut builder = module.builder(func);
            let entry = builder.init_entry_block();
            let default = builder.make_block_call(entry, &[]);
            let index = builder.ins().i32const(0);
            builder.ins().br_table(index, default, &[]);
        }
        module.validate().unwrap();
        let mut module = module.build_data();
        let func = &mut module.functions[func];
        let inst = *func.layout.blocks[func.entry_block.unwrap()]
            .insts
            .last()
            .unwrap();
        let crate::InstructionData::BrTable { table, .. } = func.dfg.instructions[inst] else {
            unreachable!()
        };
        func.dfg.jump_tables[table].targets.clear();
        assert!(
            module
                .validate()
                .unwrap_err()
                .to_string()
                .contains("default destination")
        );
    }

    #[test]
    fn test_unsealed_block_validation() {
        let mut module = ModuleBuilder::new();
        let signature = module.make_signature(vec![], vec![], CallConv::SystemV);
        let function = module.declare_function("test".to_string(), signature, Linkage::Export);
        let mut builder = module.builder(function);

        builder.init_entry_block();
        builder.ins().ret(&[]);
        let block = builder.create_block();
        builder.switch_to_block(block);
        builder.ins().ret(&[]);

        drop(builder);
        let error = module.validate().unwrap_err().to_string();
        assert!(error.contains("is not sealed"), "unexpected error: {error}");
    }

    #[test]
    fn opspec_rejects_wrong_arithmetic_family() {
        let scheme = crate::Opcode::IAdd.spec().type_scheme;
        assert!(
            scheme
                .validate(&[Type::F32, Type::F32], &[Type::F32])
                .is_err()
        );
        assert!(
            scheme
                .validate(&[Type::I32, Type::I32], &[Type::I32])
                .is_ok()
        );
    }

    #[test]
    fn opspec_validates_conversion_widths() {
        let extend = crate::Opcode::ExtendS.spec().type_scheme;
        assert!(extend.validate(&[Type::I32], &[Type::I64]).is_ok());
        assert!(extend.validate(&[Type::I64], &[Type::I32]).is_err());
    }

    #[test]
    fn opspec_rejects_wrong_vector_constant_size() {
        let mut module = ModuleBuilder::new();
        let signature = module.make_signature(vec![], vec![], CallConv::SystemV);
        let function =
            module.declare_function("bad-vconst".to_string(), signature, Linkage::Export);
        let mut builder = module.builder(function);
        builder.init_entry_block();
        builder.ins().vconst(Type::I32X4, vec![0; 4]);
        builder.ins().ret(&[]);
        drop(builder);

        let error = module.validate().unwrap_err().to_string();
        assert!(
            error.contains("requires 16 bytes"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn opspec_rejects_zero_pointer_scale() {
        let mut module = ModuleBuilder::new();
        let signature =
            module.make_signature(vec![Type::PTR, Type::I32], vec![], CallConv::SystemV);
        let function = module.declare_function("bad-scale".to_string(), signature, Linkage::Export);
        let mut builder = module.builder(function);
        builder.init_entry_block();
        let ptr = builder.func_param(0);
        let index = builder.func_param(1);
        builder.ins().ptr_index(ptr, index, 0, 0);
        builder.ins().ret(&[]);
        drop(builder);

        let error = module.validate().unwrap_err().to_string();
        assert!(error.contains("scale") && error.contains("non-zero"));
    }

    #[test]
    fn opspec_restricts_pointer_ordering() {
        let mut module = ModuleBuilder::new();
        let signature =
            module.make_signature(vec![Type::PTR, Type::PTR], vec![], CallConv::SystemV);
        let function =
            module.declare_function("bad-ptr-compare".to_string(), signature, Linkage::Export);
        let mut builder = module.builder(function);
        builder.init_entry_block();
        let lhs = builder.func_param(0);
        let rhs = builder.func_param(1);
        builder.ins().icmp(IntCC::LtU, lhs, rhs);
        builder.ins().ret(&[]);
        drop(builder);

        let error = module.validate().unwrap_err().to_string();
        assert!(error.contains("only supports eq and ne"));
    }
}
