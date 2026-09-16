mod assembly;
mod contracts;
mod encoding;
mod generate;
mod preprocess;
mod select;

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
    copy_bits: Option<u32>,
}

fn parse_input(input: &str) -> Result<crate::target::ast::Module, String> {
    match parser::parse(input) {
        Ok(m) => Ok(m),
        Err(e) => {
            use miette::Diagnostic;
            let report = if e.source_code().is_some() {
                miette::Report::new(e)
            } else {
                miette::Report::new(e).with_source_code(input.to_string())
            };
            Err(format!("{:?}", report))
        }
    }
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

pub fn compile(
    input: &str,
    arch: &str,
    definitions: &veloc_opgen::Source,
) -> Result<String, String> {
    let contracts = definitions.contracts().map_err(|error| error.to_string())?;
    let input = preprocess::preprocess_isle(input)?;
    let mut module = parse_input(&input)?;
    module
        .defs
        .extend(contracts::registers(definitions.declarations())?);
    for def in &module.defs {
        if let Def::SelectRule(rule) = def {
            select::check_temps(rule)?;
        }
    }
    let extractors = collect_extractors(&module)?;
    let mut final_inst_defs = contracts::compile(contracts, &module)?;
    encoding::compile(definitions, &mut final_inst_defs)?;
    assembly::compile(definitions.declarations(), &mut final_inst_defs)?;
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
            return Err(format!(
                "{name}: scheduled instructions must have explicit register dependencies and no control/stack operands; only EFLAGS clobbers are supported"
            ));
        }
    }
    let needs_positional_helpers = select::module_has_positional_rules(&module);

    let mut output = String::new();
    generate::generate_header(&mut output, arch, needs_positional_helpers);
    generate::generate_register_descriptors(&mut output, &module);
    generate::generate_cpu_info(&mut output, &module);
    generate::generate_abi_descriptors(&mut output, &module)?;
    generate::generate_enum(&mut output, &final_inst_defs);
    generate::generate_enum_conversions(&mut output, &final_inst_defs);
    generate::generate_target_inst_metadata(&mut output, &module, &final_inst_defs);
    generate::generate_validation(&mut output, &final_inst_defs);
    select::generate_generic_inst_metadata(&mut output, &module, &final_inst_defs);
    encoding::generate(&mut output, &final_inst_defs);
    assembly::generate(&mut output, &final_inst_defs);
    select::generate_select_instruction(&mut output, &module, &extractors, &final_inst_defs, arch);

    Ok(output)
}
