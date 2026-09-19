mod assembly;
mod contracts;
mod cpu;
mod encoding;
mod generate;
mod select;
mod selection;

use crate::target::ast::Def;
use crate::target::{ExtractorDef, OperandConstraint, parser};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub(crate) struct FinalInstDef {
    operands: Vec<OperandConstraint>,
    reg_classes: Vec<(String, Vec<String>)>,
    ties: Vec<(usize, usize)>,
    implicit_uses: Vec<String>,
    implicit_defs: Vec<String>,
    clobbers: Vec<String>,
    schedule_latency: Option<u32>,
    flow: String,
    memory: Option<(String, u32)>,
    encoding: Option<String>,
    is_pseudo: bool,
    assembly: Option<assembly::Assembly>,
    requires: Vec<String>,
}

fn collect_extractors(
    module: &crate::target::ast::Module,
) -> Result<HashMap<String, ExtractorDef>, String> {
    let mut extractors = HashMap::new();
    for def in &module.defs {
        if let Def::Extractor(extractor) = def {
            if extractors
                .insert(extractor.name.clone(), extractor.clone())
                .is_some()
            {
                return Err(format!("duplicate extractor {}", extractor.name));
            }
        }
    }
    Ok(extractors)
}

pub(crate) struct Plan {
    module: crate::target::ast::Module,
    extractors: HashMap<String, ExtractorDef>,
    instructions: HashMap<String, FinalInstDef>,
    arch: String,
    context: String,
    cpu: cpu::Plan,
}
impl Plan {
    pub(crate) fn prepare(
        input: &crate::Source,
        arch: &str,
        context: &str,
        definitions: &crate::Source,
        input_definitions: Option<(&str, &crate::Source)>,
    ) -> Result<Self, crate::SourceError> {
        let input_error =
            |message: String| input.locate(crate::Error::at(input.text(), 0, message));
        let definition_error =
            |message: String| definitions.locate(crate::Error::at(definitions.text(), 0, message));
        if !crate::rules::identifier(arch) {
            return Err(input_error(
                "architecture must be a Rust module identifier".into(),
            ));
        }
        crate::interfaces::rust_path(input.text(), 0, context).map_err(|e| input.locate(e))?;
        input.check_imports()?;
        let contracts = definitions.contracts()?;
        let mut module = parser::declarations(input.text(), input.declarations())
            .map_err(|e| input.locate(e))?;
        module
            .defs
            .extend(contracts::registers(definitions.declarations()).map_err(&definition_error)?);
        let input_contracts = input_definitions
            .map(|(_, source)| source.contracts())
            .transpose()?;
        if let Some((dialect, source)) = input_definitions {
            let types = crate::types::Types::compile(source.declarations(), source.text())
                .map_err(|e| source.locate(e))?;
            for def in &mut module.defs {
                if let Def::SelectRule(rule) = def {
                    selection::resolve(rule, dialect, input_contracts.as_ref().unwrap(), &types)
                        .map_err(&input_error)?;
                    select::check_temps(rule).map_err(&input_error)?;
                }
            }
        } else if module
            .defs
            .iter()
            .any(|def| matches!(def, Def::SelectRule(_)))
        {
            return Err(input_error(
                "selection requires input operation definitions".into(),
            ));
        }
        let extractors = collect_extractors(&module).map_err(&input_error)?;
        select::check_predicates(&module).map_err(&input_error)?;
        let mut final_inst_defs =
            contracts::compile(contracts, &module).map_err(&definition_error)?;
        for def in &module.defs {
            if let Def::SelectRule(rule) = def {
                select::check_construction(rule, &final_inst_defs).map_err(&input_error)?;
            }
        }
        encoding::compile(definitions, &mut final_inst_defs).map_err(&definition_error)?;
        assembly::compile(definitions.declarations(), &mut final_inst_defs)
            .map_err(&definition_error)?;
        for (name, inst) in &final_inst_defs {
            if inst.schedule_latency.is_some()
                && (inst.memory.is_some()
                    || inst.flow != "Next"
                    || inst.is_pseudo
                    || !inst.implicit_uses.is_empty()
                    || !inst.implicit_defs.is_empty()
                    || inst.clobbers.iter().any(|reg| reg != "EFLAGS")
                    || inst.operands.iter().any(|op| {
                        matches!(
                            op,
                            OperandConstraint::Block(_)
                                | OperandConstraint::Global(_)
                                | OperandConstraint::StackSlot(_)
                        )
                    }))
            {
                return Err(definition_error(format!(
                    "{name}: scheduled instructions must have explicit register dependencies and no control/stack operands; only EFLAGS clobbers are supported"
                )));
            }
        }
        // Validate fallible target metadata even when only one artifact is requested.
        let cpu = cpu::Plan::prepare(&module).map_err(&input_error)?;
        generate::check_abi_descriptors(&module).map_err(&input_error)?;
        Ok(Self {
            module,
            extractors,
            instructions: final_inst_defs,
            arch: arch.into(),
            context: context.into(),
            cpu,
        })
    }
    pub(crate) fn emit(&self, kind: crate::Emit) -> String {
        let module = &self.module;
        let extractors = &self.extractors;
        let final_inst_defs = &self.instructions;
        let arch = self.arch.as_str();
        let mut fragment = String::new();
        match kind {
            crate::Emit::Encoder => encoding::generate(&mut fragment, final_inst_defs),
            crate::Emit::Assembly => assembly::generate(&mut fragment, final_inst_defs),
            crate::Emit::Selector => select::generate_select_instruction(
                &mut fragment,
                module,
                extractors,
                final_inst_defs,
                arch,
                &self.context,
            ),
            crate::Emit::Target => {}
            _ => unreachable!("not a target artifact"),
        }
        if kind != crate::Emit::Target {
            return fragment;
        }

        let mut output = String::new();
        generate::generate_header(&mut output, arch);
        generate::generate_register_descriptors(&mut output, &module);
        self.cpu.generate(&mut output);
        generate::generate_abi_descriptors(&mut output, &module);
        generate::generate_enum(&mut output, &final_inst_defs);
        generate::generate_enum_conversions(&mut output, &final_inst_defs);
        generate::generate_target_inst_metadata(&mut output, &module, &final_inst_defs);
        generate::generate_validation(&mut output, &final_inst_defs);
        select::generate_generic_inst_metadata(&mut output, &module, &final_inst_defs);
        encoding::generate(&mut output, &final_inst_defs);
        assembly::generate(&mut output, &final_inst_defs);
        select::generate_select_instruction(
            &mut output,
            &module,
            &extractors,
            &final_inst_defs,
            arch,
            &self.context,
        );

        output
    }
}
