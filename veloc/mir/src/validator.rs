use crate::dfg::PoolKey;
use crate::inst::{Inst, VectorExtId, VectorMemExtId};
use crate::{
    Block, BlockCall, Function, InstructionData, ModuleData, Opcode, Result, SigId, Type, Value,
    ValueList,
};
use alloc::string::String;
use core::fmt;
use smallvec::SmallVec;

include!(concat!(env!("OUT_DIR"), "/validation.rs"));

#[derive(Debug, Clone)]
pub enum ValidationError {
    EmptyBlock(Block),
    NoTerminator(Block),
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
            Self::UnsealedBlock(block) => write!(f, "Block {:?} is not sealed", block),
            Self::Other(message) => f.write_str(message),
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

        let mut operands = SmallVec::<[Type; 4]>::new();
        data.visit_type_operands(&self.dfg, |value| {
            operands.push(self.dfg.value_type(value));
        });
        let results = self
            .dfg
            .inst_results(inst)
            .iter()
            .map(|&value| self.dfg.value_type(value))
            .collect::<SmallVec<[Type; 2]>>();
        opcode
            .validate_types(&operands, &results)
            .map_err(|error| {
                crate::Error::from(ValidationError::Other(alloc::format!(
                    "{} type scheme violation at {:?}: {:?}",
                    spec.mnemonic,
                    inst,
                    error
                )))
            })?;

        self.validate_constraints(inst, data, &operands, &results)?;

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
                    module.signatures[self.signature].returns.iter().copied(),
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

    #[cold]
    fn constraint_error(&self, inst: Inst, message: &str) -> crate::Error {
        ValidationError::Other(alloc::format!(
            "{} constraint at {:?}: {}",
            self.dfg.instructions[inst].opcode().spec().mnemonic,
            inst,
            message
        ))
        .into()
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
        self.validate_values(
            name,
            self.dfg.get_value_list(args),
            signature.params.iter().copied(),
        )?;

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

    fn validate_values(
        &self,
        name: &str,
        values: &[Value],
        expected: impl ExactSizeIterator<Item = Type>,
    ) -> Result<()> {
        if values.len() != expected.len() {
            return self.fail(alloc::format!(
                "{} value count mismatch: expected {}, got {}",
                name,
                expected.len(),
                values.len()
            ));
        }
        for (index, (&value, expected)) in values.iter().zip(expected).enumerate() {
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
            params.iter().map(|&value| self.dfg.value_type(value)),
        )
    }

    fn validate_vector_ext(
        &self,
        inst: Inst,
        opcode: Opcode,
        ext: VectorExtId,
        vector_ty: Type,
    ) -> Result<()> {
        let ext = ext.get(&self.dfg).ok_or_else(|| {
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
        let ext = ext.get(&self.dfg).ok_or_else(|| {
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
        let Some(vector) = vector_ty.as_vector() else {
            return self.fail(alloc::format!(
                "{} mask at {:?} requires a vector type, got {}",
                opcode.spec().mnemonic,
                inst,
                vector_ty
            ));
        };
        if !mask_ty.is_predicate()
            || mask_ty
                .as_vector()
                .is_none_or(|mask| mask.shape() != vector.shape())
        {
            return self.fail(alloc::format!(
                "{} mask at {:?} must match vector shape {:?}, got {}",
                opcode.spec().mnemonic,
                inst,
                vector.shape(),
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
    fn many_arguments_and_results_preserve_validation_and_diagnostics() {
        let types = [
            Type::I8,
            Type::I16,
            Type::I32,
            Type::I64,
            Type::F32,
            Type::F64,
            Type::BOOL,
        ];
        let mut module = ModuleBuilder::new();
        let sig = module.make_signature(types.to_vec(), types.to_vec(), CallConv::SystemV);
        let callee = module.declare_function("callee".into(), sig, Linkage::Local);
        {
            let mut builder = module.builder(callee);
            builder.init_entry_block();
            let params = builder.func_params().to_vec();
            builder.ins().ret(&params);
        }
        let caller = module.declare_function("caller".into(), sig, Linkage::Local);
        let (target, last_param) = {
            let mut builder = module.builder(caller);
            builder.init_entry_block();
            let args = builder.func_params().to_vec();
            let call = builder.ins().call(callee, &args);
            let results = builder.func().dfg.inst_results(call).to_vec();
            let target = builder.create_block();
            let params = types
                .iter()
                .map(|&ty| builder.add_block_param(target, ty))
                .collect::<Vec<_>>();
            builder.ins().jump(target, &results);
            builder.switch_to_block(target);
            builder.seal_block(target);
            builder.ins().ret(&params);
            (target, *params.last().unwrap())
        };
        module.validate().unwrap();
        let mut module = module.build_data();

        // A mismatch beyond the inline capacity must not be skipped.
        module.functions[caller].dfg.values[last_param].ty = Type::I32;
        let error = module.validate().unwrap_err().to_string();
        assert!(error.contains("value 6 type mismatch"), "{error}");
        module.functions[caller].dfg.values[last_param].ty = Type::BOOL;
        module.validate().unwrap();

        module.functions[caller].layout.blocks[target].params.pop();
        let error = module.validate().unwrap_err().to_string();
        assert!(
            error.contains("value count mismatch: expected 6, got 7"),
            "{error}"
        );
    }

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
        let scheme = crate::Opcode::IAdd;
        assert!(
            scheme
                .validate_types(&[Type::F32, Type::F32], &[Type::F32])
                .is_err()
        );
        assert!(
            scheme
                .validate_types(&[Type::I32, Type::I32], &[Type::I32])
                .is_ok()
        );
    }

    #[test]
    fn opspec_validates_conversion_widths() {
        let extend = crate::Opcode::ExtendS;
        assert!(extend.validate_types(&[Type::I32], &[Type::I64]).is_ok());
        assert!(extend.validate_types(&[Type::I64], &[Type::I32]).is_err());
    }

    #[test]
    fn opspec_rejects_wrong_vector_constant_size() {
        let mut module = ModuleBuilder::new();
        let signature = module.make_signature(vec![], vec![], CallConv::SystemV);
        let function =
            module.declare_function("bad-vconst".to_string(), signature, Linkage::Export);
        let mut builder = module.builder(function);
        builder.init_entry_block();
        builder.ins().vconst(vec![0; 4], Type::I32X4);
        builder.ins().ret(&[]);
        drop(builder);

        let error = module.validate().unwrap_err().to_string();
        assert!(
            error.contains("vector constant byte count must match its type"),
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
        builder.ins().ptr_index(
            ptr,
            index,
            crate::inst::PtrIndexImm {
                scale: 0,
                offset: 0,
            },
        );
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

    #[test]
    fn generated_shuffle_constraints_check_shape_length_and_selectors() {
        let scalable = Type::I32
            .as_scalar()
            .unwrap()
            .vector(4, true)
            .unwrap()
            .as_type();
        for (ty, mask, expected) in [
            (Type::I32X4, vec![0, 7, 1, 4], None),
            (Type::I32X4, vec![0, 1, 2], Some("mask length")),
            (
                Type::I32X4,
                vec![0, 1, 2, 8],
                Some("selector is out of range"),
            ),
            (scalable, vec![0, 1, 2, 3], Some("fixed-width vector")),
        ] {
            let mut module = ModuleBuilder::new();
            let sig = module.make_signature(vec![ty, ty], vec![], CallConv::SystemV);
            let id = module.declare_function("shuffle-check".into(), sig, Linkage::Local);
            {
                let mut builder = module.builder(id);
                builder.init_entry_block();
                let lhs = builder.func_param(0);
                let rhs = builder.func_param(1);
                builder.ins().shuffle(lhs, rhs, mask);
                builder.ins().ret(&[]);
            }
            let result = module.validate();
            if let Some(expected) = expected {
                assert!(result.unwrap_err().to_string().contains(expected));
            } else {
                result.unwrap();
            }
        }
    }

    #[test]
    fn generated_pool_projection_reports_missing_data_without_panicking() {
        let mut module = ModuleBuilder::new();
        let sig = module.make_signature(vec![], vec![], CallConv::SystemV);
        let id = module.declare_function("missing-pool".into(), sig, Linkage::Local);
        {
            let mut builder = module.builder(id);
            builder.init_entry_block();
            let value = builder.ins().vconst(vec![0; 16], Type::I32X4);
            let inst = builder.func().dfg.value_inst(value).unwrap();
            builder.func_mut().dfg.instructions[inst] = crate::InstructionData::Vconst {
                pool_id: crate::inst::ConstantPoolId(u32::MAX),
            };
            builder.ins().ret(&[]);
        }
        assert!(
            module
                .validate()
                .unwrap_err()
                .to_string()
                .contains("vconst constraint")
        );
    }
}
