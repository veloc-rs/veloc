//! MIR validation: module types, instruction contracts, SSA and ownership.
use crate::host::VerifyContext;
use crate::inst::Inst;
use crate::{Block, FunctionRef, InstView, Module, Opcode, Result, Successor, Type, Value};
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

impl Module {
    pub fn validate(&self) -> Result<()> {
        types::validate(self)?;
        for (_, function) in self.functions() {
            function.validate_body(self).map_err(|error| {
                crate::Error::Message(alloc::format!(
                    "In function {}: {}",
                    function.decl.name,
                    error
                ))
            })?;
        }
        Ok(())
    }
}

impl FunctionRef<'_> {
    pub(crate) fn validate_expressions(&self, module: &Module, candidates: &[Inst]) -> Result<()> {
        self.validate(module)?;
        let body = self.body.expect("defined function");
        let dfg = body.dfg();
        let context = VerifyContext::new(module, self);
        for &inst in candidates {
            if body.layout().inst_block(inst).is_some() || !dfg.inst(inst).can_speculate() {
                return self.fail(format!("invalid floating expression {inst}"));
            }
            for &value in dfg.operands(inst) {
                let Some(data) = dfg.values().get(value) else {
                    return self.fail(format!("unknown operand {value} at {inst}"));
                };
                if let crate::ValueDef::Inst(def) = data.def {
                    if !dfg.inst_results(def).contains(&value)
                        || (body.layout().inst_block(def).is_none()
                            && (!candidates.contains(&def) || def.0 >= inst.0))
                    {
                        return self
                            .fail(format!("invalid candidate dependency {value} at {inst}"));
                    }
                }
            }
            self.validate_inst(module, inst, &context)?;
        }
        Ok(())
    }

    pub fn validate(&self, module: &Module) -> Result<()> {
        types::validate(module)?;
        self.validate_body(module)
    }

    fn validate_body(&self, module: &Module) -> Result<()> {
        if module.signatures().get(self.decl.signature).is_none() {
            return self.fail("unknown function signature".into());
        }
        let Some(body) = self.body else {
            return Ok(());
        };
        let structure = control::Structure::check(self, module)?;
        let context = VerifyContext::new(module, self);
        for block in body.layout().block_order() {
            for inst in body.layout().block_insts(block) {
                self.validate_inst(module, inst, &context)?;
            }
        }
        structure.check_ssa(self)?;
        ownership::validate(self)
    }

    fn validate_inst(
        &self,
        module: &Module,
        inst: Inst,
        context: &VerifyContext<'_>,
    ) -> Result<()> {
        let body = self.body.expect("defined function");
        let data = &body.dfg().inst(inst);
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
            operands.push(body.dfg().value_type(value));
        });
        let results = body
            .dfg()
            .inst_results(inst)
            .iter()
            .map(|&value| body.dfg().value_type(value))
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

        self.validate_constraints(
            &body.dfg(),
            module,
            inst,
            data,
            &operands,
            &results,
            context,
        )?;

        data.try_visit_successors(|call| self.validate_block_call(call, spec.mnemonic))
    }

    #[cold]
    fn constraint_error(&self, inst: Inst, message: &str) -> crate::Error {
        let body = self.body.expect("defined function");
        ValidationError::Other(alloc::format!(
            "{} constraint at {:?}: {}",
            body.dfg().opcode(inst).spec().mnemonic,
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
        let body = self.body.expect("defined function");
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
            let got = body.dfg().value_type(value);
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
        let body = self.body.expect("defined function");
        let params = &body.dfg().blocks[call.block].params;
        self.validate_values(
            kind,
            "value",
            call.args,
            params.iter().map(|&value| body.dfg().value_type(value)),
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
            let mut builder = module.define(callee);
            let params = builder.func().params().to_vec();
            builder.ins().ret(&params);
        }
        let caller = module.declare_function("caller".into(), sig, Linkage::Local);
        let (target, last_param) = {
            let mut builder = module.define(caller);
            let args = builder.func().params().to_vec();
            let call = builder.ins().call(callee, &args);
            let results = builder.func().dfg().inst_results(call).to_vec();
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
        let mut module = module.build();

        // A mismatch beyond the inline capacity must not be skipped.
        module.bodies[caller]
            .as_deref_mut()
            .unwrap()
            .edit()
            .set_value_type(last_param, Type::I32);
        let error = module.validate().unwrap_err().to_string();
        assert!(error.contains("value 6 type mismatch"), "{error}");
        module.bodies[caller]
            .as_deref_mut()
            .unwrap()
            .edit()
            .set_value_type(last_param, Type::BOOL);
        module.validate().unwrap();

        let func = module.bodies[caller].as_deref_mut().unwrap();
        let entry = func.entry_block();
        let jump = func.layout().last_inst(entry).unwrap();
        let args = func.params()[..6].to_vec();
        let edge = crate::BlockCall::new(target, &args);
        func.edit()
            .replace_inst(jump, |writer| writer.jump(edge.as_view()));
        let error = module.validate().unwrap_err().to_string();
        assert!(
            error.contains("value count mismatch: expected 7, got 6"),
            "{error}"
        );
    }

    #[test]
    fn branch_tables_require_a_default_destination() {
        let mut module = ModuleBuilder::new();
        let sig = module.make_signature(vec![], vec![], CallConv::SystemV);
        let func = module.declare_function("empty-table".into(), sig, Linkage::Local);
        {
            let mut builder = module.define(func);
            let entry = builder.func().entry_block();
            let default = crate::BlockCall::new(entry, &[]);
            let index = builder.ins().i32const(0);
            builder.ins().br_table(index, default, &[]);
        }
        module.validate().unwrap();
        let mut module = module.build();
        let func = module.bodies[func].as_deref_mut().unwrap();
        let inst = func.layout().last_inst(func.entry_block()).unwrap();
        let crate::InstView::BrTable { index, .. } = func.dfg().inst(inst) else {
            unreachable!()
        };
        func.edit()
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
        let mut builder = module.define(function);

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
            let mut builder = module.define(id);
            let value = builder.ins().i32x4const([0; 4]);
            let inst = builder.func().dfg().value_inst(value).unwrap();
            builder.ins().ret(&[]);
            builder
                .finish()
                .replace_inst(inst, |writer: crate::InstWriter<'_>| {
                    writer.vconst(crate::VectorConst::dense(
                        crate::Type::I32X4.as_vector().unwrap(),
                        crate::inst::ConstantPoolId(u32::MAX),
                    ))
                });
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
