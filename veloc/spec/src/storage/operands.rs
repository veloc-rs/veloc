//! Operand-array storage strategy. Logical signatures and semantics belong to the shared model.
use crate::model::{Definitions, Op, ParamKind, TypeList};
use crate::syntax::{Kind, Node, Record};
use crate::{Error, model};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

#[derive(Debug)]
pub(crate) struct Operands {
    formats: BTreeMap<String, Format>,
    prefix: String,
}
#[derive(Debug)]
struct Format {
    name: String,
    view: String,
    accessor: String,
    fields: Vec<(String, Role)>,
    lengths: Vec<usize>,
}
pub(crate) struct Projection {
    pub arity: usize,
    pub flow: String,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Def,
    Use,
    TiedDefUse,
    Imm,
    FImm,
    StackSlot,
    Block,
    OptionalUse,
    IntCC,
    FloatCC,
    Index,
    Uses,
    CallShape,
}

impl Role {
    fn parse(source: &str, node: Node) -> Result<Self, Error> {
        let offset = node.offset;
        Ok(match model::name(source, node)?.as_str() {
            "Def" => Self::Def,
            "Use" => Self::Use,
            "TiedDefUse" => Self::TiedDefUse,
            "Imm" => Self::Imm,
            "FImm" => Self::FImm,
            "StackSlot" => Self::StackSlot,
            "Block" => Self::Block,
            "OptionalUse" => Self::OptionalUse,
            "IntCC" => Self::IntCC,
            "FloatCC" => Self::FloatCC,
            "Index" => Self::Index,
            "Uses" => Self::Uses,
            "CallShape" => Self::CallShape,
            _ => return Err(Error::at(source, offset, "unknown machine operand role")),
        })
    }

    fn view_type(self) -> &'static str {
        match self {
            Self::Def | Self::Use | Self::TiedDefUse => "Reg",
            Self::Imm => "i64",
            Self::FImm => "f64",
            Self::StackSlot => "StackSlot",
            Self::Block => "Block",
            Self::OptionalUse => "Option<Reg>",
            Self::IntCC => "IntCC",
            Self::FloatCC => "FloatCC",
            Self::Index => "usize",
            Self::Uses => "SmallVec<[Reg; 2]>",
            Self::CallShape => "CallShape",
        }
    }

    fn decoder(self) -> &'static str {
        match self {
            Self::Def => "expect_def_reg",
            Self::Use => "expect_use_reg",
            Self::TiedDefUse => "expect_tied_def_reg",
            Self::Imm => "expect_imm",
            Self::FImm => "expect_fimm",
            Self::StackSlot => "expect_stackslot",
            Self::Block => "expect_block",
            Self::OptionalUse => "expect_optional_use_reg",
            Self::IntCC => "expect_intcc",
            Self::FloatCC => "expect_floatcc",
            Self::Index => "expect_nonnegative_imm_usize",
            Self::Uses => "collect_use_regs_from",
            Self::CallShape => "decode_call_shape_field",
        }
    }

    fn builder_type(self) -> &'static str {
        match self {
            Self::Def | Self::TiedDefUse => "Writable<Reg>",
            Self::OptionalUse => "Reg",
            Self::Index => "i64",
            _ => self.view_type(),
        }
    }

    fn encode(self, name: &str) -> String {
        let variant = match self {
            Self::Def => "Def",
            Self::Use | Self::OptionalUse => "Use",
            Self::TiedDefUse => "TiedDefUse",
            Self::Imm | Self::Index => "Imm",
            Self::FImm => "FImm",
            Self::StackSlot => "StackSlot",
            Self::Block => "Block",
            Self::IntCC => {
                return format!("MachineOperand::CondCode(CondCode::Int({name}))");
            }
            Self::FloatCC => return format!("MachineOperand::CondCode(CondCode::Float({name}))"),
            Self::Uses | Self::CallShape => unreachable!("variable fields have dedicated builders"),
        };
        format!("MachineOperand::{variant}({name})")
    }

    fn variable(self) -> bool {
        matches!(self, Self::Uses | Self::CallShape)
    }
    fn optional(self) -> bool {
        matches!(self, Self::OptionalUse)
    }
}

fn finish(source: &str, record: &Record) -> Result<(), Error> {
    if let Some((key, value)) = record.fields.first_key_value() {
        return Err(Error::at(
            source,
            value.offset,
            format!("unknown field `{key}`"),
        ));
    }
    Ok(())
}

fn snake(name: &str) -> String {
    let chars: Vec<_> = name.chars().collect();
    let mut out = String::new();
    for (i, &c) in chars.iter().enumerate() {
        if c.is_ascii_uppercase()
            && i > 0
            && (chars[i - 1].is_ascii_lowercase()
                || chars.get(i + 1).is_some_and(char::is_ascii_lowercase))
        {
            out.push('_');
        }
        out.push(c.to_ascii_lowercase());
    }
    out
}

pub(crate) fn compile(records: &[Record], source: &str, prefix: String) -> Result<Operands, Error> {
    let mut formats = BTreeMap::new();
    let mut symbols = BTreeSet::new();
    for mut record in records.iter().filter(|r| r.kind == "format").cloned() {
        model::identifier(source, record.offset, &record.name)?;
        let node = record
            .fields
            .remove("fields")
            .ok_or_else(|| Error::at(source, record.offset, "format requires fields"))?;
        let mut fields = Vec::new();
        let mut names = BTreeSet::new();
        for field in model::list(source, node)? {
            let Kind::Call(name, mut args) = field.kind else {
                return Err(Error::at(source, field.offset, "expected field(Role)"));
            };
            model::identifier(source, field.offset, &name)?;
            if args.len() != 1 || !names.insert(name.clone()) {
                return Err(Error::at(
                    source,
                    field.offset,
                    "duplicate field or invalid role arguments",
                ));
            }
            fields.push((name, Role::parse(source, args.remove(0))?));
        }
        let variable = fields.iter().any(|(_, r)| r.variable());
        if variable && (fields.len() != 1) {
            return Err(Error::at(
                source,
                record.offset,
                "variable codec must describe the entire operand sequence",
            ));
        }
        let lengths = if let Some(node) = record.fields.remove("lengths") {
            if variable {
                return Err(Error::at(
                    source,
                    node.offset,
                    "variable codec cannot have fixed lengths",
                ));
            }
            let lengths = model::list(source, node)?
                .into_iter()
                .map(|node| match node.kind {
                    Kind::Number(n) => Ok(n as usize),
                    _ => Err(Error::at(source, node.offset, "expected operand count")),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let required = fields
                .iter()
                .rposition(|(_, r)| !r.optional())
                .map_or(0, |i| i + 1);
            if lengths.is_empty()
                || lengths.iter().any(|&n| n < required || n > fields.len())
                || lengths.iter().copied().collect::<BTreeSet<_>>().len() != lengths.len()
            {
                return Err(Error::at(
                    source,
                    record.offset,
                    "invalid or duplicate operand counts",
                ));
            }
            lengths
        } else if variable {
            Vec::new()
        } else {
            vec![fields.len()]
        };
        let view = record
            .fields
            .remove("view")
            .map(|n| model::name(source, n))
            .transpose()?
            .unwrap_or_else(|| format!("{}Inst", record.name));
        let accessor = record
            .fields
            .remove("accessor")
            .map(|n| model::name(source, n))
            .transpose()?
            .unwrap_or_else(|| format!("as_{}", snake(&record.name)));
        for symbol in [&view, &accessor] {
            model::identifier(source, record.offset, symbol)?;
            if !symbols.insert(symbol.clone()) {
                return Err(Error::at(
                    source,
                    record.offset,
                    "duplicate generated symbol",
                ));
            }
        }
        finish(source, &record)?;
        let name = record.name.clone();
        if formats
            .insert(
                name,
                Format {
                    name: record.name,
                    view,
                    accessor,
                    fields,
                    lengths,
                },
            )
            .is_some()
        {
            return Err(Error::at(source, record.offset, "duplicate machine format"));
        }
    }

    Ok(Operands { formats, prefix })
}
impl Operands {
    pub(crate) fn format_count(&self) -> usize {
        self.formats.len()
    }

    pub(crate) fn mnemonic(&self, name: &str) -> String {
        name.strip_prefix(&self.prefix)
            .unwrap_or(name)
            .to_ascii_lowercase()
    }

    pub(crate) fn project(
        &self,
        source: &str,
        offset: usize,
        format: &str,
        params: &[model::Param],
        signature: &model::TypeDef,
        flow: Option<Node>,
    ) -> Result<Projection, Error> {
        let fail = |message| Error::at(source, offset, message);
        let shape = self
            .formats
            .get(format)
            .ok_or_else(|| fail("unknown operand format"))?;
        let flow = flow
            .map(|n| model::name(source, n))
            .transpose()?
            .unwrap_or_else(|| "Next".into());
        if !matches!(flow.as_str(), "Next" | "Jump" | "Return" | "Call" | "Trap") {
            return Err(fail("unknown control flow kind"));
        }
        let valid = |role: Role, param: &model::Param| match role {
            Role::Use | Role::OptionalUse | Role::TiedDefUse => param.kind == ParamKind::Value,
            _ => matches!(&param.kind, ParamKind::Property(ty) if ty == match role {
                Role::Imm | Role::Index => "i64", Role::FImm => "f64", Role::StackSlot => "StackSlot",
                Role::Block => "Block", Role::IntCC => "IntCC", Role::FloatCC => "FloatCC",
                _ => "",
            }),
        };
        if let [(_, role)] = shape.fields.as_slice() {
            match role {
                Role::Uses => {
                    if params.len() != 1
                        || params[0].kind != ParamKind::Values
                        || signature.results != TypeList::Fixed(vec![])
                    {
                        return Err(fail(
                            "use-list storage requires variadic inputs and no results",
                        ));
                    }
                    return Ok(Projection { arity: 0, flow });
                }
                Role::CallShape => {
                    if params.len() != 2
                        || params[1].kind != ParamKind::Values
                        || !(params[0].kind == ParamKind::Value
                            || params[0].kind == ParamKind::Property("SymbolId".into()))
                        || signature.results != TypeList::Signature
                        || flow != "Call"
                    {
                        return Err(fail(
                            "call storage requires a callee, variadic arguments, signature results and Call flow",
                        ));
                    }
                    return Ok(Projection { arity: 0, flow });
                }
                _ => {}
            }
        }
        let TypeList::Fixed(results) = &signature.results else {
            return Err(fail("fixed operand storage requires fixed results"));
        };
        let mut choices = Vec::new();
        for &arity in &shape.lengths {
            let fields = &shape.fields[..arity];
            let defs = fields
                .iter()
                .filter(|(_, r)| matches!(r, Role::Def | Role::TiedDefUse))
                .count();
            let inputs = fields
                .iter()
                .filter(|(_, r)| *r != Role::Def)
                .collect::<Vec<_>>();
            if defs == results.len()
                && inputs.len() == params.len()
                && inputs
                    .iter()
                    .zip(params)
                    .all(|((name, role), param)| name == &param.name && valid(*role, param))
            {
                choices.push(arity);
            }
        }
        let [arity] = choices.as_slice() else {
            return Err(fail(
                "operand storage does not uniquely match the logical signature",
            ));
        };
        Ok(Projection {
            arity: *arity,
            flow,
        })
    }

    pub(crate) fn generate(&self, defs: &Definitions) -> String {
        let mut out = String::from("// @generated from LIR definitions by veloc-opgen.\n");
        out.push_str(&crate::generate::opcode_enum(defs, "GenericOpcode"));
        out.push_str(
            "impl GenericOpcode {\npub const fn control(self) -> ControlFlow { match self {\n",
        );
        for inst in &defs.ops {
            writeln!(
                out,
                "Self::{} => ControlFlow::{},",
                inst.name,
                inst.operands().flow
            )
            .unwrap();
        }
        out.push_str("} }\n}\n");
        out.push_str(
            "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GenericInstSchema {\n",
        );
        for f in self.formats.values() {
            writeln!(out, "{},", f.name).unwrap();
        }
        out.push_str("}\n#[derive(Debug, Clone, PartialEq)]\npub enum DecodedGenericInst {\n");
        for f in self.formats.values() {
            writeln!(out, "{}({}),", f.name, f.view).unwrap();
        }
        out.push_str("}\nimpl GenericInstSchema {\npub fn for_opcode(op: GenericOpcode) -> Self { match op {\n");
        for inst in &defs.ops {
            writeln!(
                out,
                "GenericOpcode::{} => Self::{},",
                inst.name, inst.format
            )
            .unwrap();
        }
        out.push_str("} }\n}\nfn decode_simple_generic(inst: &MachineInst, schema: GenericInstSchema) -> crate::error::Result<DecodedGenericInst> { match schema {\n");
        for f in self.formats.values() {
            writeln!(
                out,
                "GenericInstSchema::{} => inst.{}().map(DecodedGenericInst::{}),",
                f.name, f.accessor, f.name
            )
            .unwrap();
        }
        out.push_str("} }\n");
        for f in self.formats.values() {
            self.emit_format(&mut out, defs, f);
        }
        for inst in &defs.ops {
            self.emit_builder(&mut out, inst);
        }
        out
    }

    fn emit_format(&self, out: &mut String, defs: &Definitions, f: &Format) {
        writeln!(
            out,
            "#[derive(Debug, Clone, PartialEq)]\npub struct {} {{",
            f.view
        )
        .unwrap();
        for (name, role) in &f.fields {
            writeln!(out, "pub {name}: {},", role.view_type()).unwrap();
        }
        writeln!(
            out,
            "}}\nimpl MachineInst {{ pub fn {}(&self) -> crate::error::Result<{}> {{",
            f.accessor, f.view
        )
        .unwrap();
        writeln!(out, "self.expect_schema(GenericInstSchema::{})?;", f.name).unwrap();
        // Arity belongs to an opcode, even when several opcodes share a view.
        if !f.lengths.is_empty() {
            out.push_str(
                "let valid_len = match self.generic_opcode().expect(\"schema checked\") {\n",
            );
            for inst in defs.ops.iter().filter(|i| i.format == f.name) {
                let lengths = vec![inst.operands().arity];
                writeln!(
                    out,
                    "GenericOpcode::{} => matches!(self.operands.len(), {}),",
                    inst.name,
                    lengths
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(" | ")
                )
                .unwrap();
            }
            out.push_str("_ => unreachable!(\"schema checked\"),\n};\n");
            writeln!(
                out,
                "if !valid_len {{ return Err(self.decode_error(\"invalid {} operand count\")); }}",
                f.name
            )
            .unwrap();
        }
        writeln!(out, "Ok({} {{", f.view).unwrap();
        for (index, (name, role)) in f.fields.iter().enumerate() {
            writeln!(
                out,
                "{name}: self.{}({index}, \"invalid {}.{name} operand\")?,",
                role.decoder(),
                f.name
            )
            .unwrap();
        }
        out.push_str("}) } }\n");
    }

    fn emit_builder(&self, out: &mut String, inst: &Op) {
        let f = &self.formats[&inst.format];
        if f.fields.iter().any(|(_, r)| r.variable()) {
            return;
        }
        let count = inst.operands().arity;
        let fields = &f.fields[..count];
        let name = inst
            .name
            .strip_prefix(&self.prefix)
            .unwrap_or(&inst.name)
            .to_ascii_lowercase();
        writeln!(
            out,
            "impl MachineInst {{ pub fn build_{name}({}) -> Self {{",
            fields
                .iter()
                .map(|(name, role)| format!("{name}: {}", role.builder_type()))
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap();
        writeln!(out, "Self::build_generic(MachineOpcode::Generic(GenericOpcode::{}), smallvec::smallvec![{}])\n}} }}", inst.name,
            fields.iter().map(|(name, role)| role.encode(name)).collect::<Vec<_>>().join(", ")).unwrap();
    }
}
