//! Bind instruction fields to the encoder's declared types. Expression syntax,
//! constructors and field checking belong to OpSpec, not this target adapter.
use super::{FinalInstDef, generate::find_operand_info};
use crate::target::{AttributeKind, OperandConstraint};
use crate::{
    Source,
    syntax::{DeclKind, Kind, Node},
};
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;

fn field(index: usize, variant: &str) -> String {
    format!(
        "{{ let veloc_lir::FieldValueRef::{variant}(value) = inst.fields().read({index}) else {{ unreachable!(\"validated instruction field\") }}; *value }}"
    )
}

pub(super) fn compile(
    source: &Source,
    arch: &str,
    instructions: &mut HashMap<String, FinalInstDef>,
) -> Result<(), String> {
    let mut expressions = source.expressions().map_err(|e| e.to_string())?;
    for decl in source.declarations() {
        if !matches!(&decl.kind, DeclKind::Op(_)) {
            continue;
        }
        let inst = instructions
            .get_mut(&decl.name)
            .ok_or_else(|| format!("encoding {} has no instruction", decl.name))?;
        let Some(node) = decl.fields.get("encoding") else {
            if inst.is_pseudo {
                continue;
            }
            return Err(format!("{}: missing encoding", decl.name));
        };
        if inst.is_pseudo {
            return Err(format!("{}: pseudo cannot have an encoding", decl.name));
        }
        let mut bindings = BTreeMap::new();
        for operand in &inst.operands {
            let name = operand.name();
            let (index, _) = find_operand_info(name, &inst.operands).unwrap();
            let (ty, rust) = match operand {
                OperandConstraint::Def(_) => ("Reg", format!("register(inst.results()[{index}])?")),
                OperandConstraint::Use(_) => ("Reg", format!("register(inst.inputs()[{index}])?")),
                OperandConstraint::Attribute(_, kind) => {
                    let attribute = kind.description();
                    let value = field(index, attribute.field_variant);
                    match kind {
                        AttributeKind::StackSlot => (
                            "Address",
                            format!("stack_address(&mfunc.stack_frame, {value})?"),
                        ),
                        AttributeKind::Block => ("Block", format!("inst.edge({value}).block")),
                        AttributeKind::Blocks => ("Block", "&inst.fields().successors().iter().map(|&edge| inst.edge(edge).block).collect::<Vec<_>>()".into()),
                        _ => (attribute.spec_type, value),
                    }
                }
            };
            let kind = if matches!(
                operand,
                OperandConstraint::Attribute(_, AttributeKind::Blocks)
            ) {
                Kind::Call(
                    "sequence".into(),
                    vec![Node {
                        offset: node.offset,
                        kind: Kind::Name("Block".into()),
                    }],
                )
            } else {
                Kind::Name(ty.into())
            };
            bindings.insert(
                name.to_owned(),
                (
                    Node {
                        offset: node.offset,
                        kind,
                    },
                    rust,
                ),
            );
        }
        let rust = expressions
            .rust(
                node,
                "Emission",
                &bindings,
                &format!("veloc_encoder::{arch}::"),
            )
            .map_err(|e| format!("{}: {e}", decl.name))?;
        inst.encoding = Some(rust);
    }
    for (name, inst) in instructions {
        if !inst.is_pseudo && inst.encoding.is_none() {
            return Err(format!("{name}: missing encoding"));
        }
    }
    Ok(())
}

pub(super) fn generate(out: &mut String, arch: &str, instructions: &HashMap<String, FinalInstDef>) {
    writeln!(out, "impl TargetInst {{ pub(crate) fn emission(&self, inst: &veloc_lir::InstRef<'_>, mfunc: &veloc_lir::MachineFunction) -> crate::Result<crate::target::{arch}::emitter::Emission> {{").unwrap();
    writeln!(
        out,
        "use crate::target::{arch}::emitter::{{register, stack_address}}; match self {{"
    )
    .unwrap();
    for (name, inst) in instructions.iter().collect::<BTreeMap<_, _>>() {
        writeln!(out, "Self::{name} => {{").unwrap();
        if let Some(encoding) = &inst.encoding {
            writeln!(out, "Ok({encoding})").unwrap();
        } else {
            writeln!(
                out,
                "Err(crate::Error::codegen(\"pseudo must be expanded before encoding\"))"
            )
            .unwrap();
        }
        writeln!(out, "}},").unwrap();
    }
    writeln!(out, "}} }} }}").unwrap();
}
