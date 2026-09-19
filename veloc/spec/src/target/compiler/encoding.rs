//! Bind instruction fields to the encoder's declared types. Expression syntax,
//! constructors and field checking belong to OpSpec, not this target adapter.
use super::{FinalInstDef, generate::find_operand_info};
use crate::target::OperandConstraint;
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
            let name = match operand {
                OperandConstraint::Def(n)
                | OperandConstraint::Use(n)
                | OperandConstraint::FixedUse { src: n, .. }
                | OperandConstraint::Imm(n)
                | OperandConstraint::StackSlot(n)
                | OperandConstraint::Block(n)
                | OperandConstraint::Global(n)
                | OperandConstraint::Call(n) => n,
            };
            let (index, _) = find_operand_info(name, &inst.operands).unwrap();
            let (ty, rust) = match operand {
                OperandConstraint::Def(_) => ("Reg", format!("register(inst.results()[{index}])?")),
                OperandConstraint::Use(_) | OperandConstraint::FixedUse { .. } => {
                    ("Reg", format!("register(inst.inputs()[{index}])?"))
                }
                OperandConstraint::Imm(_) => ("i64", field(index, "Imm")),
                OperandConstraint::StackSlot(_) => (
                    "Address",
                    format!(
                        "stack_address(&mfunc.stack_frame, {})?",
                        field(index, "StackSlot")
                    ),
                ),
                OperandConstraint::Block(_) => (
                    "Block",
                    format!("inst.edge({}).block", field(index, "Edge")),
                ),
                OperandConstraint::Global(_) => ("Global", field(index, "Global")),
                OperandConstraint::Call(_) => ("CallInfo", field(index, "Call")),
            };
            bindings.insert(
                name.clone(),
                (
                    Node {
                        offset: node.offset,
                        kind: Kind::Name(ty.into()),
                    },
                    rust,
                ),
            );
        }
        let rust = expressions
            .rust(node, "Emission", &bindings, "veloc_encoder::x86_64::")
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

pub(super) fn generate(out: &mut String, instructions: &HashMap<String, FinalInstDef>) {
    writeln!(out, "impl TargetInst {{ pub fn emit(&self, emitter: &mut crate::Emitter, inst: &veloc_lir::InstRef<'_>, mfunc: &veloc_lir::MachineFunction) -> crate::Result<()> {{").unwrap();
    writeln!(out, "use crate::target::x86_64::emitter::{{register, stack_address, encode_instruction}}; match self {{").unwrap();
    for (name, inst) in instructions.iter().collect::<BTreeMap<_, _>>() {
        writeln!(out, "Self::{name} => {{").unwrap();
        if let Some(encoding) = &inst.encoding {
            writeln!(out, "encode_instruction(emitter, {encoding})").unwrap();
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
