use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::Error;
use crate::model::records::{Policy, RecordDef};
use crate::syntax::{Decl, DeclKind, Kind, Node};

pub(crate) mod compact;
mod generate;
pub(crate) mod operands;

#[derive(Debug)]
pub(crate) enum Strategy {
    Packed,
    Operands(operands::Operands),
}

pub(crate) fn constructor_name(name: &str) -> String {
    let mut method = String::new();
    for (i, ch) in name.chars().enumerate() {
        if i > 0 && ch.is_uppercase() {
            method.push('_');
        }
        method.extend(ch.to_lowercase());
    }
    if method == "return" {
        "ret".into()
    } else {
        method
    }
}

#[derive(Debug)]
pub(crate) struct Format {
    pub name: String,
    pub arity: Option<usize>,
    pub fields: Vec<Field>,
}

#[derive(Debug)]
pub(crate) struct Storage {
    pub strategy: Strategy,
    pub formats: Vec<Format>,
    layouts: Vec<Layout>,
    pub records: Vec<RecordDef>,
    pub alternatives: Vec<Alternative>,
}

#[derive(Debug)]
pub(crate) struct Alternative {
    pub name: String,
    pub fields: Vec<Field>,
    pub formats: Vec<String>,
    pub text: Node,
    pub constraints: Option<Node>,
}

#[derive(Debug)]
struct Layout {
    offset: usize,
    name: String,
    fields: Vec<Field>,
    opcode: OpcodeSource,
    format: FormatSource,
    canonical: bool,
    text: Option<Node>,
    constraints: Option<Node>,
}

#[derive(Clone, Debug)]
pub(crate) struct Field {
    pub(crate) name: String,
    pub(crate) ty: FieldType,
    pub(crate) rust: String,
    pub(crate) policy: Policy,
}

#[derive(Clone, Debug)]
pub(crate) enum FieldType {
    Named(String),
    Values(usize),
}

#[derive(Debug)]
enum OpcodeSource {
    Fixed(String),
    Dynamic(usize),
}

#[derive(Debug)]
enum FormatSource {
    Fixed(String),
    Arity { field: usize, formats: Vec<String> },
}

impl FieldType {
    pub(crate) fn named(&self, expected: &str) -> bool {
        matches!(self, Self::Named(name) if name == expected)
    }

    fn schema_type(&self) -> String {
        match self {
            Self::Named(name) => name.clone(),
            Self::Values(n) => format!("values({n})"),
        }
    }
}

impl Field {
    pub(crate) fn traversal(&self) -> Option<&'static str> {
        if matches!(self.ty, FieldType::Values(_)) {
            return Some("array");
        }
        self.policy.references.traversal()
    }

    fn arity(&self) -> Option<usize> {
        match self.ty {
            FieldType::Values(n) => Some(n),
            _ => self.policy.references.arity(),
        }
    }
}

impl Layout {
    fn arity(&self) -> Option<usize> {
        self.fields
            .iter()
            .try_fold(0usize, |n, field| n.checked_add(field.arity()?))
    }

    fn pattern(&self) -> String {
        if self.fields.is_empty() {
            return format!("Self::{}", self.name);
        }
        let fields = self
            .fields
            .iter()
            .enumerate()
            .map(|(i, f)| format!("{}: _field{i}", f.name))
            .collect::<Vec<_>>()
            .join(", ");
        format!("Self::{} {{ {fields} }}", self.name)
    }
}

/// Compile physical layouts and their logical format/text projections from one
/// field schema. Opcode/type declarations are checked by the enclosing model.
pub(crate) fn compile(
    records: &[Decl],
    source: &str,
    data: &crate::model::data::Types,
) -> Result<Storage, Error> {
    if let Some(record) = records
        .iter()
        .find(|r| matches!(&r.kind, DeclKind::Fields(kind) if kind == "storage"))
    {
        if records
            .iter()
            .filter(|r| matches!(&r.kind, DeclKind::Fields(kind) if kind == "storage"))
            .count()
            != 1
            || record.name != "Operands"
        {
            return Err(Error::at(
                source,
                record.offset,
                "expected one storage Operands declaration",
            ));
        }
        let operands = operands::compile(records, source, data)?;
        return Ok(Storage {
            strategy: Strategy::Operands(operands),
            formats: Vec::new(),
            layouts: Vec::new(),
            records: Vec::new(),
            alternatives: Vec::new(),
        });
    }
    let properties = data.records.clone();
    let mut layouts = Vec::new();
    let mut names = BTreeSet::new();
    let mut methods: BTreeSet<String> = ["copy", "write", "from_values"]
        .into_iter()
        .map(str::to_owned)
        .collect();
    let mut used = BTreeSet::new();
    for op in records
        .iter()
        .filter(|r| matches!(&r.kind, DeclKind::Op(_)))
    {
        if let Some(Node {
            kind: Kind::Object(name, _),
            ..
        }) = op.fields.get("storage")
        {
            used.insert(name.clone());
        }
    }
    for layout in records
        .iter()
        .filter(|r| matches!(&r.kind, DeclKind::Fields(kind) if kind == "layout"))
    {
        used.extend(layout_targets(layout, source)?);
        if !data.records.iter().any(|r| r.name == layout.name) {
            return Err(Error::at(
                source,
                layout.offset,
                format!("unknown layout struct `{}`", layout.name),
            ));
        }
    }
    for record in records
        .iter()
        .filter(|r| matches!(&r.kind, DeclKind::Fields(kind) if kind == "struct"))
    {
        let binding = records.iter().find(|r| {
            matches!(&r.kind, DeclKind::Fields(kind) if kind == "layout") && r.name == record.name
        });
        if !used.contains(&record.name) && binding.is_none() {
            continue;
        }
        if !names.insert(record.name.clone()) {
            return Err(Error::at(source, record.offset, "duplicate storage layout"));
        }
        let method = constructor_name(&record.name);
        crate::model::identifier(source, record.offset, &method)?;
        if !methods.insert(method.clone()) {
            return Err(Error::at(
                source,
                record.offset,
                format!("conflicting instruction constructor `{method}`"),
            ));
        }
        layouts.push(parse_layout(
            record,
            binding,
            records,
            source,
            &properties,
            &data.rust,
        )?);
    }
    let formats = layouts
        .iter()
        .filter(|layout| layout.canonical)
        .map(|layout| Format {
            name: layout.name.clone(),
            arity: layout.arity(),
            fields: layout.fields.clone(),
        })
        .collect::<Vec<_>>();
    validate_links(&layouts, &formats, source)?;
    // Only records actually used by instruction layouts belong in operand storage.
    let properties = properties
        .into_iter()
        .filter(|r| {
            layouts
                .iter()
                .any(|l| l.fields.iter().any(|f| f.ty.named(&r.name)))
        })
        .collect::<Vec<_>>();
    for record in &properties {
        for field in &record.fields {
            use crate::model::records::PropertyType;
            let ty = match &field.ty {
                PropertyType::Named(ty) | PropertyType::Optional(ty) => ty.as_str(),
                PropertyType::Values(_) | PropertyType::Array(_, _) | PropertyType::Sequence(_) => {
                    return Err(Error::at(
                        source,
                        0,
                        "nested fixed SSA arrays are not supported by operand storage",
                    ));
                }
            };
            if !field.policy.references.is_operand()
                && (matches!(field.ty, PropertyType::Optional(_)) || data.contains_value(ty))
            {
                return Err(Error::at(
                    source,
                    records
                        .iter()
                        .find(|r| {
                            matches!(&r.kind, DeclKind::Fields(kind) if kind == "struct")
                                && r.name == record.name
                        })
                        .unwrap()
                        .offset,
                    "operand storage supports direct Value/optional(Value) fields, not nested SSA or optional non-SSA fields",
                ));
            }
        }
    }
    Ok(Storage {
        strategy: Strategy::Packed,

        records: properties,
        alternatives: layouts
            .iter()
            .filter(|layout| !layout.canonical)
            .map(|layout| Alternative {
                constraints: layout.constraints.clone(),
                name: layout.name.clone(),
                fields: layout.fields.clone(),
                formats: match &layout.format {
                    FormatSource::Fixed(name) => vec![name.clone()],
                    FormatSource::Arity { formats, .. } => formats.clone(),
                },
                text: layout
                    .text
                    .clone()
                    .expect("alternate layouts require a text adapter"),
            })
            .collect(),
        formats,
        layouts,
    })
}

impl Storage {
    pub(crate) fn properties(
        &self,
        source: &str,
        offset: usize,
        format: &str,
        mappings: &BTreeMap<String, Node>,
        params: &[crate::syntax::Parameter],
    ) -> Result<BTreeSet<String>, Error> {
        if let Strategy::Operands(operands) = &self.strategy {
            return operands.properties(source, offset, format, mappings, params);
        }
        let layout = self
            .layouts
            .iter()
            .find(|l| l.name == format)
            .or_else(|| {
                self.layouts.iter().find(|l| match &l.format {
                    FormatSource::Fixed(name) => name == format,
                    FormatSource::Arity { formats, .. } => formats.iter().any(|n| n == format),
                })
            })
            .ok_or_else(|| {
                Error::at(source, offset, format!("unknown storage layout `{format}`"))
            })?;
        let mut properties = BTreeSet::new();
        for field in &layout.fields {
            if field.policy.references.is_data()
                && !matches!(field.ty, FieldType::Values(_))
                && let Some(node) = mappings.get(&field.name)
            {
                match &node.kind {
                    Kind::Name(name) => {
                        properties.insert(name.clone());
                    }
                    Kind::Call(name, args) if name == "pool" => {
                        if let [
                            Node {
                                kind: Kind::Name(name),
                                ..
                            },
                        ] = args.as_slice()
                        {
                            properties.insert(name.clone());
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(properties)
    }

    pub(crate) fn instructions(&self) -> String {
        generate::instructions(&self.layouts, &self.records)
    }

    pub(crate) fn format_code(&self) -> String {
        generate_formats(&self.formats)
    }
}

fn layout_targets(layout: &Decl, source: &str) -> Result<Vec<String>, Error> {
    let node = required(layout, "format", source)?;
    let (kind, args) = call(node, source)?;
    match (kind, args) {
        ("fixed", [target]) => Ok(vec![name(target, source)?.to_owned()]),
        ("arity", [_, targets]) => list(targets, source)?
            .iter()
            .map(|n| name(n, source).map(str::to_owned))
            .collect(),
        _ => Err(Error::at(
            source,
            node.offset,
            "expected fixed(Format) or arity(field, [Formats])",
        )),
    }
}

fn parse_layout(
    record: &Decl,
    binding: Option<&Decl>,
    declarations: &[Decl],
    source: &str,
    records: &[RecordDef],
    rust: &crate::model::records::RustTypes,
) -> Result<Layout, Error> {
    let is_format = binding.is_none();
    let targets = match binding {
        Some(layout) => layout_targets(layout, source)?,
        None => vec![record.name.clone()],
    };
    let users = declarations
        .iter()
        .filter(|r| matches!(&r.kind, DeclKind::Op(_)))
        .filter(|op| {
            matches!(
                op.fields.get("storage"),
                Some(Node { kind: Kind::Object(target, _), .. }) if targets.contains(target)
            )
        })
        .collect::<Vec<_>>();
    let shape = records
        .iter()
        .find(|r| r.name == record.name)
        .expect("checked record");
    let mut fields = shape
        .fields
        .iter()
        .map(|f| {
            let ty = match &f.ty {
                crate::model::records::PropertyType::Named(name) => FieldType::Named(name.clone()),
                crate::model::records::PropertyType::Values(n) => FieldType::Values(*n),
                crate::model::records::PropertyType::Optional(_)
                | crate::model::records::PropertyType::Sequence(_)
                | crate::model::records::PropertyType::Array(_, _) => {
                    return Err(Error::at(
                        source,
                        record.offset,
                        "optional or array primary fields require an operand storage adapter",
                    ));
                }
            };
            Ok(Field {
                name: f.name.clone(),
                policy: f.policy.clone(),
                rust: match &ty {
                    FieldType::Named(name) => rust.qualified(name),
                    FieldType::Values(n) => format!("[{}; {n}]", rust.qualified("Value")),
                },
                ty,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let opcode = if let [op] = users.as_slice() {
        OpcodeSource::Fixed(op.name.clone())
    } else {
        if fields.iter().any(|f| f.name == "opcode") {
            return Err(Error::at(
                source,
                record.offset,
                "opcode is reserved for the generated instruction tag",
            ));
        }
        fields.insert(
            0,
            Field {
                name: "opcode".into(),
                policy: Policy::default(),
                ty: FieldType::Named("Opcode".into()),
                rust: "crate::Opcode".into(),
            },
        );
        OpcodeSource::Dynamic(0)
    };
    if let Some(binding) = binding {
        for (key, node) in &binding.fields {
            if !matches!(key.as_str(), "format" | "text" | "verify") {
                return Err(Error::at(
                    source,
                    node.offset,
                    format!("unknown layout field `{key}`"),
                ));
            }
        }
    }
    let record = binding.unwrap_or(record);
    let flags_fields = fields
        .iter()
        .filter(|f| f.ty.named("MemFlags") || f.ty.named("VectorMemOptions"))
        .count();
    if flags_fields > 1 {
        return Err(Error::at(
            source,
            record.offset,
            "a layout has at most one memory flags source",
        ));
    }

    let (format, text) = if is_format {
        (FormatSource::Fixed(record.name.clone()), None)
    } else {
        let node = required(record, "format", source)?;
        let (kind, args) = call(node, source)?;
        let format = match (kind, args) {
            ("fixed", [target]) => FormatSource::Fixed(name(target, source)?.to_owned()),
            ("arity", [values, formats]) => {
                let index = field_index(&fields, name(values, source)?, values, source)?;
                if !fields[index].policy.references.is_operands() {
                    return Err(Error::at(
                        source,
                        values.offset,
                        "arity layout requires a variadic ValueList field",
                    ));
                }
                if fields
                    .iter()
                    .enumerate()
                    .any(|(i, f)| i != index && f.arity() != Some(0))
                {
                    return Err(Error::at(
                        source,
                        node.offset,
                        "arity layout must carry all primary operands in one list",
                    ));
                }
                let mut names = BTreeSet::new();
                let mut targets = Vec::new();
                for target in list(formats, source)? {
                    let name = name(target, source)?;
                    if !names.insert(name) {
                        return Err(Error::at(
                            source,
                            target.offset,
                            "duplicate arity target format",
                        ));
                    }
                    targets.push(name.to_owned());
                }
                if targets.is_empty() {
                    return Err(Error::at(
                        source,
                        formats.offset,
                        "arity layout requires a target format",
                    ));
                }
                FormatSource::Arity {
                    field: index,
                    formats: targets,
                }
            }
            _ => {
                return Err(Error::at(
                    source,
                    node.offset,
                    "expected fixed(Format) or arity(field, [Formats])",
                ));
            }
        };
        let text = record.fields.get("text").cloned().ok_or_else(|| {
            Error::at(
                source,
                record.offset,
                "alternate layout has no text adapter",
            )
        })?;
        (format, Some(text))
    };
    let layout = Layout {
        constraints: record.fields.get("verify").cloned(),
        offset: record.offset,
        name: record.name.clone(),
        fields,
        opcode,
        format,
        canonical: is_format,
        text,
    };

    validate_runtime_contract(&layout, source)?;
    Ok(layout)
}

/// Existing MIR consumers and custom syntax hooks destructure these public
/// layouts. Their field contracts must be checked before generating Rust.
/// New values-only formats have no such ABI and retain arbitrary field names.
fn validate_runtime_contract(layout: &Layout, source: &str) -> Result<(), Error> {
    type Fields = &'static [(&'static str, &'static str)];
    let (fields, fixed): (Fields, Option<&str>) = match layout.name.as_str() {
        "Unary" => (&[("opcode", "Opcode"), ("arg", "Value")], None),
        "Binary" => (&[("opcode", "Opcode"), ("args", "values(2)")], None),
        "Ternary" => (&[("opcode", "Opcode"), ("args", "values(3)")], None),
        "Iconst" => (&[("value", "Int")], Some("Iconst")),
        "Fconst" => (&[("value", "Float")], Some("Fconst")),
        "Bconst" => (&[("value", "bool")], Some("Bconst")),
        "Vconst" => (&[("value", "VectorConst")], Some("Vconst")),
        "Load" => (
            &[("ptr", "Value"), ("offset", "u32"), ("flags", "MemFlags")],
            Some("Load"),
        ),
        "Store" => (
            &[
                ("ptr", "Value"),
                ("value", "Value"),
                ("offset", "u32"),
                ("flags", "MemFlags"),
            ],
            Some("Store"),
        ),
        "Alloca" => (&[("size", "u32"), ("align", "u32")], Some("Alloca")),
        "PtrOffset" => (&[("ptr", "Value"), ("offset", "i32")], Some("PtrOffset")),
        "PtrIndex" => (
            &[
                ("ptr", "Value"),
                ("index", "Value"),
                ("imm_id", "PtrIndexImm"),
            ],
            Some("PtrIndex"),
        ),
        "IntToPtr" | "PtrToInt" => (&[("arg", "Value")], Some(layout.name.as_str())),
        "Call" => (
            &[("func_id", "FuncId"), ("args", "ValueList")],
            Some("Call"),
        ),
        "CallIndirect" => (
            &[("ptr", "Value"), ("args", "ValueList"), ("sig_id", "SigId")],
            Some("CallIndirect"),
        ),
        "CallIntrinsic" => (
            &[
                ("intrinsic", "Intrinsic"),
                ("args", "ValueList"),
                ("sig_id", "SigId"),
            ],
            Some("CallIntrinsic"),
        ),
        "Jump" => (&[("dest", "BlockCall")], Some("Jump")),
        "Br" => (
            &[
                ("condition", "Value"),
                ("then_dest", "BlockCall"),
                ("else_dest", "BlockCall"),
            ],
            Some("Br"),
        ),
        "BrTable" => (
            &[("index", "Value"), ("table", "JumpTable")],
            Some("BrTable"),
        ),
        "Return" => (&[("values", "ValueList")], Some("Return")),
        "IntCompare" => (&[("kind", "IntCC"), ("args", "values(2)")], Some("Icmp")),
        "FloatCompare" => (&[("kind", "FloatCC"), ("args", "values(2)")], Some("Fcmp")),
        "VectorLoadStrided" => (
            &[
                ("ptr", "Value"),
                ("stride", "Value"),
                ("ext", "VectorMemOptions"),
            ],
            Some("LoadStride"),
        ),
        "VectorStoreStrided" => (
            &[("args", "values(3)"), ("ext", "VectorMemOptions")],
            Some("StoreStride"),
        ),
        "VectorGather" => (
            &[
                ("ptr", "Value"),
                ("index", "Value"),
                ("ext", "VectorMemOptions"),
            ],
            Some("Gather"),
        ),
        "VectorScatter" => (
            &[("args", "values(3)"), ("ext", "VectorMemOptions")],
            Some("Scatter"),
        ),
        "Shuffle" => (
            &[("args", "values(2)"), ("mask", "ConstantPoolId")],
            Some("Shuffle"),
        ),
        "Nop" | "Unreachable" => (&[], Some(layout.name.as_str())),
        "VectorOpWithExt" => (
            &[
                ("opcode", "Opcode"),
                ("args", "ValueList"),
                ("ext", "VectorExtData"),
            ],
            None,
        ),
        _ => return Ok(()),
    };
    let actual = layout
        .fields
        .iter()
        .filter(|f| !f.ty.named("Opcode"))
        .collect::<Vec<_>>();
    let fields = fields
        .iter()
        .filter(|(_, ty)| *ty != "Opcode")
        .copied()
        .collect::<Vec<_>>();
    let matching_fields = actual.len() == fields.len()
        && actual
            .into_iter()
            .zip(&fields)
            .all(|(field, &(name, ty))| field.name == name && field.ty.schema_type() == ty);
    if !matching_fields {
        let expected = fields
            .iter()
            .map(|(name, ty)| format!("{name}({ty})"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(Error::at(
            source,
            layout.offset,
            format!(
                "layout `{}` field contract requires [{expected}] in operand order",
                layout.name
            ),
        ));
    }
    if let Some(expected) = fixed
        && !matches!(&layout.opcode, OpcodeSource::Fixed(actual) if actual == expected)
    {
        return Err(Error::at(
            source,
            layout.offset,
            format!(
                "layout `{}` opcode contract requires fixed({expected})",
                layout.name
            ),
        ));
    }
    if layout.name == "VectorOpWithExt" && !matches!(layout.format, FormatSource::Arity { .. }) {
        return Err(Error::at(
            source,
            layout.offset,
            "VectorOpWithExt requires an arity-based format adapter",
        ));
    }
    Ok(())
}

fn validate_links(layouts: &[Layout], formats: &[Format], source: &str) -> Result<(), Error> {
    let formats = formats
        .iter()
        .map(|f| (f.name.as_str(), f))
        .collect::<BTreeMap<_, _>>();
    for layout in layouts {
        match &layout.format {
            FormatSource::Fixed(name) => {
                let Some(format) = formats.get(name.as_str()) else {
                    return Err(Error::at(
                        source,
                        layout.offset,
                        format!("unknown format `{name}`"),
                    ));
                };
                if layout.arity() != format.arity {
                    return Err(Error::at(
                        source,
                        layout.offset,
                        "alternate layout and format arities differ",
                    ));
                }
            }
            FormatSource::Arity {
                formats: targets, ..
            } => {
                for target in targets {
                    let Some(format) = formats.get(target.as_str()) else {
                        return Err(Error::at(
                            source,
                            layout.offset,
                            format!("unknown format `{target}`"),
                        ));
                    };
                    if format.arity.is_none() || !value_only(&format.fields) {
                        return Err(Error::at(
                            source,
                            layout.offset,
                            "arity targets must use fixed-arity value layouts",
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

fn value_only(fields: &[Field]) -> bool {
    fields.iter().all(|f| {
        f.policy.references.is_operand()
            || f.ty.named("Opcode")
            || matches!(f.ty, FieldType::Values(_))
    })
}

fn generate_formats(formats: &[Format]) -> String {
    let mut out = String::from(
        "// @generated from operation storage definitions.\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum OpFormat {\n",
    );
    for format in formats {
        writeln!(out, "    {},", format.name).unwrap();
    }
    out.push_str("}\nimpl OpFormat {\n    pub const fn fixed_value_arity(self) -> Option<usize> {\n        match self {\n");
    for format in formats {
        let arity = format
            .arity
            .map_or("None".to_owned(), |n| format!("Some({n})"));
        writeln!(out, "            Self::{} => {arity},", format.name).unwrap();
    }
    out.push_str("        }\n    }\n}\n");
    out
}

fn required<'a>(record: &'a Decl, field: &str, source: &str) -> Result<&'a Node, Error> {
    record
        .fields
        .get(field)
        .ok_or_else(|| Error::at(source, record.offset, format!("missing `{field}` field")))
}

fn list<'a>(node: &'a Node, source: &str) -> Result<&'a [Node], Error> {
    match &node.kind {
        Kind::List(nodes) => Ok(nodes),
        _ => Err(Error::at(source, node.offset, "expected a list")),
    }
}

fn call<'a>(node: &'a Node, source: &str) -> Result<(&'a str, &'a [Node]), Error> {
    match &node.kind {
        Kind::Call(name, args) => Ok((name, args)),
        _ => Err(Error::at(source, node.offset, "expected a call")),
    }
}

fn name<'a>(node: &'a Node, source: &str) -> Result<&'a str, Error> {
    match &node.kind {
        Kind::Name(name) => Ok(name),
        _ => Err(Error::at(source, node.offset, "expected a name")),
    }
}

fn field_index(fields: &[Field], name: &str, node: &Node, source: &str) -> Result<usize, Error> {
    fields.iter().position(|f| f.name == name).ok_or_else(|| {
        Error::at(
            source,
            node.offset,
            format!("unknown storage field `{name}`"),
        )
    })
}
