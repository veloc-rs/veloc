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
    fields: Vec<(String, Role)>,
}
pub(crate) struct Projection {
    pub arity: usize,
    pub flow: Flow,
    // Builder parameters are logical results followed by logical inputs.
    args: Vec<Argument>,
    // Expressions in physical field order, resolved from explicit bindings.
    fields: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Flow {
    Next,
    Jump,
    Return,
    Call,
    Trap,
}

impl Flow {
    fn parse(source: &str, node: Option<Node>) -> Result<Self, Error> {
        let Some(node) = node else {
            return Ok(Self::Next);
        };
        let offset = node.offset;
        match model::name(source, node)?.as_str() {
            "Next" => Ok(Self::Next),
            "Jump" => Ok(Self::Jump),
            "Return" => Ok(Self::Return),
            "Call" => Ok(Self::Call),
            "Trap" => Ok(Self::Trap),
            _ => Err(Error::at(source, offset, "unknown control flow kind")),
        }
    }

    pub fn traits(self) -> impl Iterator<Item = String> {
        let names: &[&str] = match self {
            Self::Next => &[],
            Self::Jump | Self::Return => &["TERMINATOR"],
            Self::Call => &["MAY_TRAP"],
            Self::Trap => &["TERMINATOR", "ABORT", "MAY_TRAP"],
        };
        names.iter().map(|name| (*name).to_owned())
    }
}

struct Argument {
    name: String,
    role: Role,
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
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
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
            _ => return None,
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
            Self::Uses => "RegList<'a>",
            Self::CallShape => "CallShape<'a>",
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
            Self::Uses => "borrow_use_regs_from",
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
}

pub(crate) fn is_role(name: &str) -> bool {
    Role::from_name(name).is_some()
}

pub(crate) fn compile(
    records: &[Record],
    source: &str,
    prefix: String,
    data: &crate::model::data::Types,
) -> Result<Operands, Error> {
    if let Some(binding) = records.iter().find(|r| r.kind == "layout") {
        return Err(Error::at(
            source,
            binding.offset,
            "operand views are derived from structs; layout overrides are not supported",
        ));
    }
    let mut formats = BTreeMap::new();
    for shape in &data.records {
        let users = records.iter().filter(|r| r.kind == "op").any(|op| {
            matches!(
                op.fields.get("storage"),
                Some(Node { kind: Kind::Object(name, _), .. }) if name == &shape.name
            )
        });
        let roles = shape.fields.iter().any(
            |f| matches!(&f.ty, crate::model::records::PropertyType::Named(ty) if is_role(ty)),
        );
        if !users && !roles {
            continue;
        }
        let offset = records
            .iter()
            .find(|r| r.kind == "struct" && r.name == shape.name)
            .expect("checked struct")
            .offset;
        let fields = shape
            .fields
            .iter()
            .map(|f| {
                let crate::model::records::PropertyType::Named(ty) = &f.ty else {
                    return Err(Error::at(source, offset, "expected machine operand role"));
                };
                Ok((
                    f.name.clone(),
                    Role::from_name(ty)
                        .ok_or_else(|| Error::at(source, offset, "unknown machine operand role"))?,
                ))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let variable = fields.iter().any(|(_, r)| r.variable());
        if variable && (fields.len() != 1) {
            return Err(Error::at(
                source,
                offset,
                "variable codec must describe the entire operand sequence",
            ));
        }
        let name = shape.name.clone();
        if formats
            .insert(
                name,
                Format {
                    name: shape.name.clone(),
                    fields,
                },
            )
            .is_some()
        {
            return Err(Error::at(source, offset, "duplicate machine format"));
        }
    }

    Ok(Operands { formats, prefix })
}
impl Operands {
    pub(crate) fn properties(
        &self,
        source: &str,
        offset: usize,
        format: &str,
        mappings: &BTreeMap<String, Node>,
        params: &[crate::syntax::Parameter],
    ) -> Result<BTreeSet<String>, Error> {
        let shape = self
            .formats
            .get(format)
            .ok_or_else(|| Error::at(source, offset, "unknown operand format"))?;
        let mut properties = BTreeSet::new();
        for (field, role) in &shape.fields {
            let Some(node) = mappings.get(field) else {
                continue;
            };
            if matches!(
                role,
                Role::Imm
                    | Role::Block
                    | Role::FImm
                    | Role::StackSlot
                    | Role::IntCC
                    | Role::FloatCC
                    | Role::Index
            ) {
                if let Kind::Name(name) = &node.kind {
                    properties.insert(name.clone());
                }
            } else if *role == Role::CallShape
                && let Kind::Call(_, args) = &node.kind
                && let Some(Node {
                    kind: Kind::Name(name),
                    ..
                }) = args.first()
            {
                // Named direct callees use a symbol property; indirect callees use PTR.
                if params.iter().any(|p| {
                    p.name == *name && matches!(&p.ty.kind, Kind::Name(ty) if ty == "SymbolId")
                }) {
                    properties.insert(name.clone());
                }
            }
        }
        Ok(properties)
    }

    pub(crate) fn record_names(&self) -> Vec<String> {
        self.formats.keys().cloned().collect()
    }
    pub(crate) fn format_count(&self) -> usize {
        self.formats.len()
    }

    pub(crate) fn mnemonic(&self, name: &str) -> String {
        name.strip_prefix(&self.prefix)
            .unwrap_or(name)
            .to_ascii_lowercase()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn project(
        &self,
        source: &str,
        offset: usize,
        format: &str,
        mappings: &BTreeMap<String, Node>,
        params: &[model::Param],
        signature: &model::TypeDef,
        slots: &BTreeMap<String, model::Slot>,
        flow: Option<Node>,
    ) -> Result<Projection, Error> {
        let fail = |message| Error::at(source, offset, message);
        let shape = self
            .formats
            .get(format)
            .ok_or_else(|| fail("unknown operand format"))?;
        let flow = Flow::parse(source, flow)?;
        for (field, node) in mappings {
            if !shape.fields.iter().any(|(name, _)| name == field) {
                return Err(Error::at(
                    source,
                    node.offset,
                    format!("unknown storage field '{field}'"),
                ));
            }
        }
        for (field, _) in &shape.fields {
            if !mappings.contains_key(field) {
                return Err(fail(&format!("missing storage field '{field}'")));
            }
        }
        let mut binding = Bindings {
            source,
            params,
            slots,
            used_params: BTreeSet::new(),
            used_results: BTreeSet::new(),
            args: BTreeMap::new(),
        };
        let mut fields = Vec::new();
        let mut omitted = false;
        for (field, role) in &shape.fields {
            let node = &mappings[field];
            if *role == Role::OptionalUse
                && matches!(&node.kind, Kind::Name(name) if name == "none")
            {
                omitted = true;
                continue;
            }
            if omitted {
                return Err(Error::at(
                    source,
                    node.offset,
                    "only trailing optional operands may be absent",
                ));
            }
            let value = match role {
                Role::Def => binding.result(node, *role)?,
                Role::TiedDefUse => {
                    let [input, result] = call_args(source, node, "tied")? else {
                        return Err(Error::at(
                            source,
                            node.offset,
                            "tied requires an input and a result",
                        ));
                    };
                    binding.input(input, *role, false)?;
                    binding.result(result, *role)?
                }
                Role::OptionalUse => {
                    let [input] = call_args(source, node, "some")? else {
                        return Err(Error::at(
                            source,
                            node.offset,
                            "optional use requires some(input) or none",
                        ));
                    };
                    binding.input(input, *role, true)?
                }
                Role::Uses => {
                    if signature.results != TypeList::Fixed(vec![]) {
                        return Err(fail(
                            "use-list storage requires variadic inputs and no results",
                        ));
                    }
                    binding.input(node, *role, false)?
                }
                Role::CallShape => {
                    let [callee, args] = call_args(source, node, "call")? else {
                        return Err(Error::at(
                            source,
                            node.offset,
                            "call requires a callee and variadic arguments",
                        ));
                    };
                    if signature.results != TypeList::Signature || flow != Flow::Call {
                        return Err(fail(
                            "call storage requires signature results and Call flow",
                        ));
                    }
                    binding.input(callee, *role, false)?;
                    binding.input(args, Role::Uses, false)?;
                    String::new()
                }
                _ => binding.input(node, *role, true)?,
            };
            fields.push(value);
        }
        for (index, param) in params.iter().enumerate() {
            if !binding.used_params.contains(&index) {
                return Err(fail(&format!(
                    "parameter '{}' has no storage mapping",
                    param.name
                )));
            }
        }
        match &signature.results {
            TypeList::Fixed(results) => {
                for index in 0..results.len() {
                    if !binding.used_results.contains(&index) {
                        return Err(fail(&format!(
                            "result {index} has no storage mapping; name it in the signature"
                        )));
                    }
                }
            }
            TypeList::Signature if shape.fields.iter().any(|(_, r)| *r == Role::CallShape) => {}
            _ => return Err(fail("fixed operand storage requires fixed results")),
        }
        Ok(Projection {
            arity: fields.len(),
            flow,
            args: binding.args.into_values().collect(),
            fields,
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
                "Self::{} => ControlFlow::{:?},",
                inst.name,
                inst.operands().flow
            )
            .unwrap();
        }
        out.push_str("} }\n}\n");
        let lifetime = if self
            .formats
            .values()
            .any(|f| f.fields.iter().any(|(_, r)| r.variable()))
        {
            "<'a>"
        } else {
            ""
        };
        writeln!(
            out,
            "#[derive(Debug, Clone, Copy)] pub enum InstView{lifetime} {{"
        )
        .unwrap();
        for f in self.formats.values() {
            let borrowed = if f.fields.iter().any(|(_, r)| r.variable()) {
                "<'a>"
            } else {
                ""
            };
            writeln!(out, "{}({}Inst{borrowed}),", f.name, f.name).unwrap();
        }
        out.push_str("}\n");
        for f in self.formats.values() {
            let ops: Vec<_> = defs.ops.iter().filter(|op| op.format == f.name).collect();
            if ops.len() > 1 {
                writeln!(out, "#[allow(non_camel_case_types)] #[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum {}Opcode {{", f.name).unwrap();
                for op in &ops {
                    writeln!(
                        out,
                        "{},",
                        op.name.strip_prefix(&self.prefix).unwrap_or(&op.name)
                    )
                    .unwrap();
                }
                out.push_str("}\n");
            }
            let borrowed = if f.fields.iter().any(|(_, r)| r.variable()) {
                "<'a>"
            } else {
                ""
            };
            writeln!(
                out,
                "#[derive(Debug, Clone, Copy)] pub struct {}Inst{borrowed} {{",
                f.name
            )
            .unwrap();
            if ops.len() > 1 {
                writeln!(out, "pub opcode: {}Opcode,", f.name).unwrap();
            }
            for (name, role) in &f.fields {
                writeln!(out, "pub {name}: {},", role.view_type()).unwrap();
            }
            out.push_str("}\n");
        }
        let borrowed = if lifetime.is_empty() { "" } else { "<'_>" };
        writeln!(out, "impl MachineInst {{ pub fn generic_view(&self) -> crate::error::Result<InstView{borrowed}> {{").unwrap();
        out.push_str("Ok(match self.generic_opcode() {\n");
        for op in &defs.ops {
            let f = &self.formats[&op.format];
            let shared = defs
                .ops
                .iter()
                .filter(|other| other.format == f.name)
                .count()
                > 1;
            writeln!(out, "Some(GenericOpcode::{}) => {{", op.name).unwrap();
            if !f.fields.iter().any(|(_, r)| r.variable()) {
                let count = op.operands().arity;
                let invalid = if count == 0 {
                    "!self.operands.is_empty()".to_owned()
                } else {
                    format!("self.operands.len() != {count}")
                };
                writeln!(out, "if {invalid} {{ return Err(self.decode_error(\"invalid {} operand count\")); }}", f.name).unwrap();
            }
            writeln!(out, "InstView::{}({}Inst {{", f.name, f.name).unwrap();
            if shared {
                writeln!(
                    out,
                    "opcode: {}Opcode::{},",
                    f.name,
                    op.name.strip_prefix(&self.prefix).unwrap_or(&op.name)
                )
                .unwrap();
            }
            for (index, (name, role)) in f.fields.iter().enumerate() {
                writeln!(
                    out,
                    "{name}: self.{}({index}, \"invalid {}.{name} operand\")?,",
                    role.decoder(),
                    f.name
                )
                .unwrap();
            }
            out.push_str("})\n},\n");
        }
        out.push_str(
            "_ => return Err(self.decode_error(\"expected a generic opcode\")),\n})\n} }\n",
        );
        for inst in &defs.ops {
            self.emit_builder(&mut out, inst);
        }
        out
    }

    fn emit_builder(&self, out: &mut String, inst: &Op) {
        let f = &self.formats[&inst.format];
        if f.fields.iter().any(|(_, r)| r.variable()) {
            return;
        }
        let projection = inst.operands();
        let name = self.mnemonic(&inst.name);
        writeln!(
            out,
            "impl MachineInst {{ pub fn build_{name}({}) -> Self {{",
            projection
                .args
                .iter()
                .map(|arg| format!("{}: {}", arg.name, arg.role.builder_type()))
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap();
        writeln!(out, "Self::build_generic(MachineOpcode::Generic(GenericOpcode::{}), smallvec::smallvec![{}])\n}} }}", inst.name,
            f.fields.iter().zip(&projection.fields).map(|((_, role), name)| role.encode(name)).collect::<Vec<_>>().join(", ")).unwrap();
    }
}

fn call_args<'a>(source: &str, node: &'a Node, expected: &str) -> Result<&'a [Node], Error> {
    match &node.kind {
        Kind::Call(name, args) if name == expected => Ok(args),
        _ => Err(Error::at(
            source,
            node.offset,
            format!("expected {expected}(...) storage binding"),
        )),
    }
}

/// Resolve logical names once; no opcode/field-name conventions or candidate arities.
struct Bindings<'a> {
    source: &'a str,
    params: &'a [model::Param],
    slots: &'a BTreeMap<String, model::Slot>,
    used_params: BTreeSet<usize>,
    used_results: BTreeSet<usize>,
    args: BTreeMap<(bool, usize), Argument>,
}

impl Bindings<'_> {
    fn input(&mut self, node: &Node, role: Role, argument: bool) -> Result<String, Error> {
        let name = model::name(self.source, node.clone())?;
        let (index, param) = self
            .params
            .iter()
            .enumerate()
            .find(|(_, p)| p.name == name)
            .ok_or_else(|| {
                Error::at(
                    self.source,
                    node.offset,
                    format!("unknown input '{name}' in storage mapping"),
                )
            })?;
        let valid = match role {
            Role::Use | Role::OptionalUse | Role::TiedDefUse => param.kind == ParamKind::Value,
            Role::Uses => param.kind == ParamKind::Values,
            Role::CallShape => {
                param.kind == ParamKind::Value
                    || param.kind == ParamKind::Property("SymbolId".into())
            }
            _ => matches!(&param.kind, ParamKind::Property(ty) if ty == match role {
                Role::Imm | Role::Index => "i64", Role::FImm => "f64",
                Role::StackSlot => "StackSlot", Role::Block => "Block",
                Role::IntCC => "IntCC", Role::FloatCC => "FloatCC", _ => "",
            }),
        };
        if !valid {
            return Err(Error::at(
                self.source,
                node.offset,
                format!("input '{name}' is incompatible with its storage role"),
            ));
        }
        if !self.used_params.insert(index) {
            return Err(Error::at(
                self.source,
                node.offset,
                format!("input '{name}' is stored more than once"),
            ));
        }
        if argument {
            self.args.insert(
                (true, index),
                Argument {
                    name: name.clone(),
                    role,
                },
            );
        }
        Ok(name)
    }

    fn result(&mut self, node: &Node, role: Role) -> Result<String, Error> {
        let name = model::name(self.source, node.clone())?;
        let slot = self
            .slots
            .get(&name)
            .filter(|slot| slot.result)
            .ok_or_else(|| {
                Error::at(
                    self.source,
                    node.offset,
                    format!("unknown result '{name}' in storage mapping"),
                )
            })?;
        let index = usize::from(slot.index);
        if !self.used_results.insert(index) {
            return Err(Error::at(
                self.source,
                node.offset,
                format!("result '{name}' is stored more than once"),
            ));
        }
        self.args.insert(
            (false, index),
            Argument {
                name: name.clone(),
                role,
            },
        );
        Ok(name)
    }
}
