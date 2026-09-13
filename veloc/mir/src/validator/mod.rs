//! MIR validation: module types, instruction contracts, SSA and ownership.
use crate::host::{ConstContext, VerifyContext};
use crate::inst::Inst;
use crate::{Block, Function, InstView, ModuleData, Opcode, Result, Successor, Type, Value};
use alloc::string::String;
use core::fmt;
use smallvec::SmallVec;

mod control;
mod ownership;
mod types;

include!(concat!(env!("OUT_DIR"), "/validation.rs"));

#[derive(Debug, Clone)]
pub enum ValidationError {
    EmptyBlock(Block),
    NoTerminator(Block),
    Other(String),
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBlock(block) => write!(f, "Block {:?} is empty", block),
            Self::NoTerminator(block) => {
                write!(f, "Block {:?} does not end with a terminator", block)
            }
            Self::Other(message) => f.write_str(message),
        }
    }
}

impl ModuleData {
    pub fn validate(&self) -> Result<()> {
        types::validate(self)?;
        for (_, function) in self.functions.iter() {
            function.validate_body(self).map_err(|error| {
                crate::Error::Message(alloc::format!("In function {}: {}", function.name, error))
            })?;
        }
        Ok(())
    }
}

impl Function {
    pub fn validate(&self, module: &ModuleData) -> Result<()> {
        types::validate(module)?;
        self.validate_body(module)
    }

    fn validate_body(&self, module: &ModuleData) -> Result<()> {
        let structure = control::Structure::check(self, module)?;
        let constants = ConstContext::new(&self.dfg);
        let context = VerifyContext::new(module, self.signature);
        for &block in &self.layout.block_order {
            let block_data = &self.layout.blocks[block];
            for &inst in &block_data.insts {
                self.validate_inst(module, inst, &constants, &context)?;
            }
        }
        structure.check_ssa(self)?;
        ownership::validate(self)
    }

    fn validate_inst(
        &self,
        module: &ModuleData,
        inst: Inst,
        constants: &ConstContext<'_>,
        context: &VerifyContext<'_>,
    ) -> Result<()> {
        let data = &self.dfg.inst(inst);
        let opcode = data.opcode();
        let spec = opcode.spec();

        if !data.matches_format(spec.format) {
            return self.fail(alloc::format!(
                "{} at {:?} is stored in an incompatible instruction format",
                spec.mnemonic,
                inst
            ));
        }

        let mut operands = SmallVec::<[Type; 4]>::new();
        data.visit_type_operands(|value| {
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

        self.validate_constraints(module, inst, data, &operands, &results, constants, context)?;

        data.try_visit_successors(|call| self.validate_block_call(call, spec.mnemonic))
    }

    #[cold]
    fn constraint_error(&self, inst: Inst, message: &str) -> crate::Error {
        ValidationError::Other(alloc::format!(
            "{} constraint at {:?}: {}",
            self.dfg.opcode(inst).spec().mnemonic,
            inst,
            message
        ))
        .into()
    }

    fn validate_values(
        &self,
        name: &str,
        role: &str,
        values: &[Value],
        expected: impl ExactSizeIterator<Item = Type>,
    ) -> Result<()> {
        if values.len() != expected.len() {
            return self.fail(alloc::format!(
                "{} {} count mismatch: expected {}, got {}",
                name,
                role,
                expected.len(),
                values.len()
            ));
        }
        for (index, (&value, expected)) in values.iter().zip(expected).enumerate() {
            let got = self.dfg.value_type(value);
            if got != expected {
                return self.fail(alloc::format!(
                    "{} {} {} type mismatch: expected {}, got {}",
                    name,
                    role,
                    index,
                    expected,
                    got
                ));
            }
        }
        Ok(())
    }

    fn validate_block_call(&self, call: Successor<'_>, kind: &str) -> Result<()> {
        let params = &self.layout.blocks[call.block].params;
        self.validate_values(
            kind,
            "value",
            call.args,
            params.iter().map(|&value| self.dfg.value_type(value)),
        )
    }

    fn fail<T>(&self, message: String) -> Result<T> {
        Err(ValidationError::Other(message).into())
    }
}

#[cfg(test)]
mod tests {
    use crate::builder::ModuleBuilder;
    use crate::{CallConv, Linkage, Type};

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
        // Keep all uses attached so this case isolates the edge arity error.
        let func = &mut module.functions[caller];
        let ret = *func.layout.blocks[target].insts.last().unwrap();
        let params = func.layout.blocks[target].params.clone();
        func.edit()
            .replace_inst(ret, |writer: crate::InstWriter<'_>| writer.ret(&params));
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
        let crate::InstView::BrTable { index, .. } = func.dfg.inst(inst) else {
            unreachable!()
        };
        func.dfg
            .replace_inst(inst, |writer: crate::InstWriter<'_>| {
                writer.br_table(index, [])
            });
        assert!(
            module
                .validate()
                .unwrap_err()
                .to_string()
                .contains("default destination")
        );
    }

    #[test]
    fn explicit_ssa_does_not_require_builder_sealing() {
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
        module.validate().unwrap();
    }

    #[test]
    fn generated_pool_projection_reports_missing_data_without_panicking() {
        let mut module = ModuleBuilder::new();
        let sig = module.make_signature(vec![], vec![], CallConv::SystemV);
        let id = module.declare_function("missing-pool".into(), sig, Linkage::Local);
        {
            let mut builder = module.builder(id);
            builder.init_entry_block();
            let value = builder.ins().i32x4const([0; 4]);
            let inst = builder.func().dfg.value_inst(value).unwrap();
            builder
                .func_mut()
                .dfg
                .replace_inst(inst, |writer: crate::InstWriter<'_>| {
                    writer.vconst(crate::VectorConst::dense(
                        crate::Type::I32X4.as_vector().unwrap(),
                        crate::inst::ConstantPoolId(u32::MAX),
                    ))
                });
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
