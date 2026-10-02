mod assembly;
mod contracts;
mod cpu;
mod encoding;
mod generate;
mod select;
mod selection;

use crate::target::ast::{AttributeKind, Def};
use crate::target::{ExtractorDef, OperandConstraint, parser};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub(crate) struct FinalInstDef {
    operands: Vec<OperandConstraint>,
    reg_classes: Vec<(String, Vec<String>)>,
    state_operands: Vec<String>,
    value_types: Vec<(String, crate::types::TypeSet)>,
    ties: Vec<(usize, usize)>,
    clobbers: Vec<String>,
    schedule_class: Option<String>,
    movable: bool,
    flow: String,
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
    context: Option<String>,
    cpu: cpu::Plan,
    input_layouts: std::collections::BTreeMap<String, crate::storage::operands::Projection>,
    payloads: Option<crate::storage::payload::Payloads>,
}
impl Plan {
    pub(crate) fn prepare(
        source: &crate::Source,
        arch: &str,
        context: Option<&str>,
        input_definitions: Option<(&str, &crate::Source)>,
    ) -> Result<Self, crate::SourceError> {
        let error = |message: String| source.locate(crate::Error::at(source.text(), 0, message));
        if !crate::rules::identifier(arch) {
            return Err(error(
                "architecture must be a Rust module identifier".into(),
            ));
        }
        if let Some(context) = context {
            crate::interfaces::rust_path(source.text(), 0, context)
                .map_err(|e| source.locate(e))?;
        }
        source.check_imports()?;
        let contracts = source.contracts()?;
        let mut module = parser::declarations(source.text(), source.declarations())
            .map_err(|e| source.locate(e))?;
        module
            .defs
            .extend(contracts::registers(source.declarations()).map_err(&error)?);
        let input_contracts = input_definitions
            .map(|(_, source)| source.contracts())
            .transpose()?;
        let mut input_layouts = std::collections::BTreeMap::new();
        let mut payloads = None;
        if let Some((dialect, source)) = input_definitions {
            let definitions = source.parse()?;
            if let crate::storage::Strategy::Operands(storage) = &definitions.storage.strategy {
                payloads = storage.payloads.clone();
            }
            for op in definitions.ops {
                if let crate::model::Projection::Operands(projection) = op.projection {
                    input_layouts.insert(op.name, projection);
                }
            }
            let types = crate::types::Types::compile(source.declarations(), source.text())
                .map_err(|e| source.locate(e))?;
            for def in &mut module.defs {
                if let Def::SelectRule(rule) = def {
                    selection::resolve(rule, dialect, input_contracts.as_ref().unwrap(), &types)
                        .map_err(&error)?;
                    select::check_temps(rule).map_err(&error)?;
                    select::check_storage(rule, &input_layouts).map_err(&error)?;
                }
            }
        } else if module
            .defs
            .iter()
            .any(|def| matches!(def, Def::SelectRule(_)))
        {
            return Err(error(
                "selection requires input operation definitions".into(),
            ));
        }
        if payloads.is_none()
            && module
                .defs
                .iter()
                .any(|def| matches!(def, Def::SelectRule(_)))
        {
            return Err(error(
                "selection requires payload layouts in its input storage definition".into(),
            ));
        }
        let extractors = collect_extractors(&module).map_err(&error)?;
        select::check_predicates(&module).map_err(&error)?;
        if !extractors.is_empty() && context.is_none() {
            return Err(error(
                "custom selector predicates require a context trait".into(),
            ));
        }
        let types = crate::types::Types::compile(source.declarations(), source.text())
            .map_err(|e| source.locate(e))?;
        let mut final_inst_defs = contracts::compile(contracts, &module, &types).map_err(&error)?;
        for def in &mut module.defs {
            if let Def::SelectRule(rule) = def {
                select::complete_state_results(rule, &final_inst_defs).map_err(&error)?;
                select::check_construction(rule, &final_inst_defs, &types).map_err(&error)?;
                select::lower_state_operands(rule, &final_inst_defs).map_err(&error)?;
            }
        }
        encoding::compile(source, arch, &mut final_inst_defs).map_err(&error)?;
        for (name, inst) in &final_inst_defs {
            if let Some(payloads) = &payloads {
                let fields: Vec<_> = inst
                    .operands
                    .iter()
                    .filter_map(|op| match op {
                        OperandConstraint::Attribute(_, kind) => Some((
                            kind.description().field_variant,
                            String::new(),
                            crate::storage::operands::Shape::One,
                        )),
                        _ => None,
                    })
                    .collect();
                payloads
                    .construct(&fields, "writer", "veloc_lir::Fields")
                    .map_err(|e| error(format!("{name}: {e}")))?;
            }
            if inst.movable
                && (inst.flow != "Next"
                    || inst.is_pseudo
                    || inst.operands.iter().any(|op| {
                        matches!(
                            op,
                            OperandConstraint::Attribute(
                                _,
                                AttributeKind::Block
                                    | AttributeKind::Global
                                    | AttributeKind::StackSlot
                                    | AttributeKind::Call
                            )
                        )
                    }))
            {
                return Err(error(format!(
                    "{name}: movable instructions must not have control or unresolved stack effects"
                )));
            }
        }
        // Validate fallible target metadata even when only one artifact is requested.
        // Declarations own class identities, independently of instruction usage.
        // Every CPU models them all, including classes for optional ISA features.
        let classes: std::collections::BTreeSet<_> = module
            .defs
            .iter()
            .filter_map(|def| match def {
                Def::ScheduleClass(class) => Some(class.name.as_str()),
                _ => None,
            })
            .collect();
        for (name, inst) in &final_inst_defs {
            if let Some(class) = &inst.schedule_class
                && !classes.contains(class.as_str())
            {
                return Err(error(format!(
                    "{name} references undeclared scheduling class {class}"
                )));
            }
        }
        for def in &module.defs {
            if let Def::Cpu(cpu) = def {
                let modeled: std::collections::BTreeSet<_> = cpu
                    .schedule
                    .classes
                    .iter()
                    .map(|cost| cost.class.as_str())
                    .collect();
                for class in classes.difference(&modeled) {
                    return Err(error(format!(
                        "CPU {} is missing scheduling class {class}",
                        cpu.name
                    )));
                }
                for class in modeled.difference(&classes) {
                    return Err(error(format!(
                        "CPU {} references unknown scheduling class {class}",
                        cpu.name
                    )));
                }
            }
        }
        let cpu = cpu::Plan::prepare(&module, &classes).map_err(&error)?;
        generate::check_abi_descriptors(&module).map_err(&error)?;
        Ok(Self {
            module,
            extractors,
            instructions: final_inst_defs,
            arch: arch.into(),
            context: context.map(str::to_owned),
            cpu,
            input_layouts,
            payloads,
        })
    }
    pub(crate) fn emit(&self, kind: crate::Emit) -> String {
        let module = &self.module;
        let extractors = &self.extractors;
        let final_inst_defs = &self.instructions;
        let arch = self.arch.as_str();
        let mut fragment = String::new();
        match kind {
            crate::Emit::Encoder => encoding::generate(&mut fragment, arch, final_inst_defs),
            crate::Emit::Assembly => assembly::generate(&mut fragment, final_inst_defs),
            crate::Emit::Selector => select::generate_select_instruction(
                &mut fragment,
                module,
                extractors,
                final_inst_defs,
                self.context.as_deref(),
                &self.input_layouts,
                self.payloads.as_ref(),
            ),
            crate::Emit::Target => {}
            _ => unreachable!("not a target artifact"),
        }
        if kind != crate::Emit::Target {
            return fragment;
        }

        let mut output = String::new();
        generate::generate_header(&mut output, arch);
        output.push_str("\n// Registers, CPU features and ABI descriptors.\n");
        generate::generate_register_descriptors(&mut output, &module);
        self.cpu.generate(&mut output);
        generate::generate_abi_descriptors(&mut output, &module);
        output.push_str("\n// Target opcodes, metadata and validation.\n");
        generate::generate_enum(&mut output, &final_inst_defs);
        generate::generate_enum_conversions(&mut output, &final_inst_defs);
        generate::generate_target_inst_metadata(&mut output, &module, &final_inst_defs, &self.cpu);
        generate::generate_validation(&mut output, &final_inst_defs);
        output.push_str("\n// Machine-code emission.\n");
        encoding::generate(&mut output, arch, &final_inst_defs);
        output.push_str("\n// Assembly rendering.\n");
        assembly::generate(&mut output, &final_inst_defs);
        select::generate_select_instruction(
            &mut output,
            &module,
            &extractors,
            &final_inst_defs,
            self.context.as_deref(),
            &self.input_layouts,
            self.payloads.as_ref(),
        );

        output
    }
}
