use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::Error;
use crate::records::RecordDef;
use crate::syntax::{Kind, Node, Record};

#[derive(Debug)]
pub(crate) struct Format {
    pub name: String,
    pub arity: Option<usize>,
    pub fixed_opcode: Option<String>,
    pub fields: Vec<Field>,
}

#[derive(Debug)]
pub(crate) struct Storage {
    pub formats: Vec<Format>,
    pub instructions: String,
    pub formats_code: String,
    pub records: Vec<RecordDef>,
    pub alternatives: Vec<Alternative>,
}

#[derive(Debug)]
pub(crate) struct Alternative {
    pub name: String,
    pub fields: Vec<Field>,
    pub formats: Vec<String>,
    pub text: Node,
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
}

#[derive(Clone, Debug)]
pub(crate) struct Field {
    pub(crate) name: String,
    pub(crate) ty: FieldType,
}

#[derive(Clone, Debug)]
pub(crate) enum FieldType {
    Named(String),
    Values(usize),
    List(usize),
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
    fn named(&self, expected: &str) -> bool {
        matches!(self, Self::Named(name) if name == expected)
    }

    fn auxiliary(&self) -> bool {
        self.named("VectorExtId") || self.named("VectorMemExtId")
    }

    fn arity(&self) -> Option<usize> {
        match self {
            Self::Values(n) | Self::List(n) => Some(*n),
            Self::Named(name) => match name.as_str() {
                "Value" => Some(1),
                "ValueList" | "BlockCall" | "JumpTable" => None,
                _ => Some(0),
            },
        }
    }

    fn rust_type(&self) -> String {
        match self {
            Self::Named(name) => name.clone(),
            Self::Values(n) => format!("[Value; {n}]"),
            Self::List(_) => "ValueList".to_owned(),
        }
    }

    fn schema_type(&self) -> String {
        match self {
            Self::Named(name) => name.clone(),
            Self::Values(n) => format!("values({n})"),
            Self::List(n) => format!("list({n})"),
        }
    }

    pub(crate) fn qualified_type(&self) -> String {
        match self {
            Self::Values(n) => format!("[crate::Value; {n}]"),
            Self::List(_) => "crate::ValueList".to_owned(),
            Self::Named(name) => match name.as_str() {
                "u32" | "u64" | "i32" | "bool" => name.clone(),
                "PtrIndexImmId" | "ConstantPoolId" | "VectorExtId" | "VectorMemExtId" => {
                    format!("crate::inst::{name}")
                }
                _ => format!("crate::{name}"),
            },
        }
    }

    fn traversal(&self) -> Option<&'static str> {
        match self {
            Self::Values(_) => Some("array"),
            Self::List(_) => Some("value_list"),
            Self::Named(name) => match name.as_str() {
                "Value" => Some("value"),
                "ValueList" => Some("value_list"),
                "BlockCall" => Some("block_call"),
                "JumpTable" => Some("jump_table"),
                "VectorExtId" => Some("vector_ext"),
                "VectorMemExtId" => Some("vector_mem_ext"),
                _ => None,
            },
        }
    }
}

impl Layout {
    fn arity(&self) -> Option<usize> {
        self.fields
            .iter()
            .try_fold(0usize, |n, field| n.checked_add(field.ty.arity()?))
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
pub(crate) fn compile(records: &[Record], source: &str) -> Result<Storage, Error> {
    let properties = crate::records::compile(records, source)?;
    let mut layouts = Vec::new();
    let mut names = BTreeSet::new();
    for record in records {
        if !matches!(record.kind.as_str(), "format" | "layout") {
            continue;
        }
        identifier(&record.name, record.offset, source)?;
        if !names.insert(record.name.clone()) {
            return Err(Error::at(source, record.offset, "duplicate storage layout"));
        }
        layouts.push(parse_layout(record, source)?);
    }
    let formats = layouts
        .iter()
        .filter(|layout| layout.canonical)
        .map(|layout| Format {
            name: layout.name.clone(),
            arity: layout.arity(),
            fixed_opcode: match &layout.opcode {
                OpcodeSource::Fixed(name) => Some(name.clone()),
                OpcodeSource::Dynamic(_) => None,
            },
            fields: layout.fields.clone(),
        })
        .collect::<Vec<_>>();
    validate_links(&layouts, &formats, records, source)?;
    Ok(Storage {
        instructions: crate::records::generate(&properties) + &generate_instructions(&layouts),
        formats_code: generate_formats(&formats),
        records: properties,
        alternatives: layouts
            .iter()
            .filter(|layout| !layout.canonical)
            .map(|layout| Alternative {
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
    })
}

fn parse_layout(record: &Record, source: &str) -> Result<Layout, Error> {
    let is_format = record.kind == "format";
    let allowed: &[&str] = if is_format {
        &["fields", "opcode"]
    } else {
        &["fields", "opcode", "format", "text"]
    };
    for (name, value) in &record.fields {
        if !allowed.contains(&name.as_str()) {
            return Err(Error::at(
                source,
                value.offset,
                format!("unknown layout field `{name}`"),
            ));
        }
    }
    let mut fields = Vec::new();
    let mut names = BTreeSet::new();
    for node in list(required(record, "fields", source)?, source)? {
        let Kind::Call(name, args) = &node.kind else {
            return Err(Error::at(source, node.offset, "expected field(type)"));
        };
        identifier(name, node.offset, source)?;
        if !names.insert(name.clone()) {
            return Err(Error::at(
                source,
                node.offset,
                format!("duplicate storage field `{name}`"),
            ));
        }
        if args.len() != 1 {
            return Err(Error::at(
                source,
                node.offset,
                "a storage field has exactly one type",
            ));
        }
        fields.push(Field {
            name: name.clone(),
            ty: field_type(&args[0], source)?,
        });
    }
    let opcode_node = required(record, "opcode", source)?;
    let (kind, args) = call(opcode_node, source)?;
    if args.len() != 1 {
        return Err(Error::at(
            source,
            opcode_node.offset,
            "expected fixed(Opcode) or dynamic(field)",
        ));
    }
    let opcode_name = name(&args[0], source)?;
    let opcode = match kind {
        "fixed" => {
            identifier(opcode_name, args[0].offset, source)?;
            OpcodeSource::Fixed(opcode_name.to_owned())
        }
        "dynamic" => {
            let index = field_index(&fields, opcode_name, &args[0], source)?;
            if !fields[index].ty.named("Opcode") {
                return Err(Error::at(
                    source,
                    args[0].offset,
                    "dynamic opcode field must have type Opcode",
                ));
            }
            OpcodeSource::Dynamic(index)
        }
        _ => {
            return Err(Error::at(
                source,
                opcode_node.offset,
                "expected fixed(Opcode) or dynamic(field)",
            ));
        }
    };
    let opcode_fields = fields.iter().filter(|f| f.ty.named("Opcode")).count();
    if opcode_fields != usize::from(matches!(opcode, OpcodeSource::Dynamic(_))) {
        return Err(Error::at(
            source,
            record.offset,
            "each Opcode field must be the dynamic opcode source",
        ));
    }
    let flags_fields = fields
        .iter()
        .filter(|f| f.ty.named("MemFlags") || f.ty.named("VectorMemExtId"))
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
                if !fields[index].ty.named("ValueList") {
                    return Err(Error::at(
                        source,
                        values.offset,
                        "arity layout requires a variadic ValueList field",
                    ));
                }
                if fields
                    .iter()
                    .enumerate()
                    .any(|(i, f)| i != index && !f.ty.auxiliary() && f.ty.arity() != Some(0))
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
        "Iconst" | "Fconst" => (&[("value", "u64")], Some(layout.name.as_str())),
        "Bconst" => (&[("value", "bool")], Some("Bconst")),
        "Vconst" => (&[("pool_id", "ConstantPoolId")], Some("Vconst")),
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
        "StackLoad" | "StackAddr" => (
            &[("slot", "StackSlot"), ("offset", "u32")],
            Some(layout.name.as_str()),
        ),
        "StackStore" => (
            &[("slot", "StackSlot"), ("value", "Value"), ("offset", "u32")],
            Some("StackStore"),
        ),
        "PtrOffset" => (&[("ptr", "Value"), ("offset", "i32")], Some("PtrOffset")),
        "PtrIndex" => (
            &[
                ("ptr", "Value"),
                ("index", "Value"),
                ("imm_id", "PtrIndexImmId"),
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
                ("ext", "VectorMemExtId"),
            ],
            Some("LoadStride"),
        ),
        "VectorStoreStrided" => (
            &[("args", "list(3)"), ("ext", "VectorMemExtId")],
            Some("StoreStride"),
        ),
        "VectorGather" => (
            &[
                ("ptr", "Value"),
                ("index", "Value"),
                ("ext", "VectorMemExtId"),
            ],
            Some("Gather"),
        ),
        "VectorScatter" => (
            &[("args", "list(3)"), ("ext", "VectorMemExtId")],
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
                ("ext", "VectorExtId"),
            ],
            None,
        ),
        _ => return Ok(()),
    };
    let matching_fields = layout.fields.len() == fields.len()
        && layout
            .fields
            .iter()
            .zip(fields)
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
    let matching_opcode = match (&layout.opcode, fixed) {
        (OpcodeSource::Fixed(actual), Some(expected)) => actual == expected,
        (OpcodeSource::Dynamic(index), None) => layout.fields[*index].name == "opcode",
        _ => false,
    };
    if !matching_opcode {
        let expected = fixed.map_or("dynamic(opcode)".to_owned(), |name| {
            format!("fixed({name})")
        });
        return Err(Error::at(
            source,
            layout.offset,
            format!(
                "layout `{}` opcode contract requires {expected}",
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

fn field_type(node: &Node, source: &str) -> Result<FieldType, Error> {
    match &node.kind {
        Kind::Name(name)
            if [
                "Opcode",
                "Value",
                "ValueList",
                "BlockCall",
                "JumpTable",
                "VectorExtId",
                "VectorMemExtId",
                "MemFlags",
                "FuncId",
                "SigId",
                "StackSlot",
                "PtrIndexImmId",
                "ConstantPoolId",
                "Intrinsic",
                "IntCC",
                "FloatCC",
                "u32",
                "u64",
                "i32",
                "bool",
            ]
            .contains(&name.as_str()) =>
        {
            Ok(FieldType::Named(name.clone()))
        }
        Kind::Call(kind, args) if matches!(kind.as_str(), "values" | "list") && args.len() == 1 => {
            let n = number(&args[0], source)?;
            if n == 0 || n > u8::MAX as usize {
                return Err(Error::at(
                    source,
                    node.offset,
                    "operand group size must be in 1..=255",
                ));
            }
            Ok(if kind == "values" {
                FieldType::Values(n)
            } else {
                FieldType::List(n)
            })
        }
        _ => Err(Error::at(source, node.offset, "unknown storage field type")),
    }
}

fn validate_links(
    layouts: &[Layout],
    formats: &[Format],
    records: &[Record],
    source: &str,
) -> Result<(), Error> {
    let opcodes = records
        .iter()
        .filter(|r| r.kind == "op")
        .map(|r| (r.name.as_str(), r))
        .collect::<BTreeMap<_, _>>();
    let formats = formats
        .iter()
        .map(|f| (f.name.as_str(), f))
        .collect::<BTreeMap<_, _>>();
    for layout in layouts {
        if let OpcodeSource::Fixed(opcode) = &layout.opcode {
            let Some(record) = opcodes.get(opcode.as_str()) else {
                return Err(Error::at(
                    source,
                    layout.offset,
                    format!("unknown fixed opcode `{opcode}`"),
                ));
            };
            let storage = required(record, "storage", source)?;
            let Kind::Object(target, _) = &storage.kind else {
                return Err(Error::at(
                    source,
                    storage.offset,
                    "expected a storage mapping",
                ));
            };
            let matches = match &layout.format {
                FormatSource::Fixed(format) => format == target,
                FormatSource::Arity { formats, .. } => formats.iter().any(|f| f == target),
            };
            if !matches {
                return Err(Error::at(
                    source,
                    layout.offset,
                    format!("fixed opcode `{opcode}` requires format `{target}`"),
                ));
            }
        }
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
        f.ty.named("Value") || f.ty.named("Opcode") || matches!(f.ty, FieldType::Values(_))
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

fn generate_instructions(layouts: &[Layout]) -> String {
    let mut out = String::from(
        "// @generated from operation storage definitions.\n#[derive(Debug, Clone)]\npub enum InstructionData {\n",
    );
    for layout in layouts {
        if layout.fields.is_empty() {
            writeln!(out, "    {},", layout.name).unwrap();
        } else {
            writeln!(out, "    {} {{", layout.name).unwrap();
            for field in &layout.fields {
                writeln!(out, "        {}: {},", field.name, field.ty.rust_type()).unwrap();
            }
            out.push_str("    },\n");
        }
    }
    out.push_str("}\n#[allow(unused_variables)]\nimpl InstructionData {\n    pub fn opcode(&self) -> Opcode {\n        match self {\n");
    for layout in layouts {
        let value = match &layout.opcode {
            OpcodeSource::Fixed(opcode) => format!("Opcode::{opcode}"),
            OpcodeSource::Dynamic(index) => format!("*_field{index}"),
        };
        writeln!(out, "            {} => {value},", layout.pattern()).unwrap();
    }
    out.push_str("        }\n    }\n    pub fn matches_format(&self, dfg: &DataFlowGraph, format: OpFormat) -> bool {\n        match self {\n");
    for layout in layouts {
        let mut condition = match &layout.format {
            FormatSource::Fixed(name) => format!("format == OpFormat::{name}"),
            FormatSource::Arity { field, formats } => {
                let targets = formats
                    .iter()
                    .map(|f| format!("OpFormat::{f}"))
                    .collect::<Vec<_>>()
                    .join(" | ");
                format!(
                    "matches!(format, {targets}) && format.fixed_value_arity() == Some(dfg.get_value_list(*_field{field}).len())"
                )
            }
        };
        for (i, field) in layout.fields.iter().enumerate() {
            if let FieldType::List(n) = field.ty {
                write!(condition, " && dfg.get_value_list(*_field{i}).len() == {n}").unwrap();
            }
        }
        writeln!(out, "            {} => {condition},", layout.pattern()).unwrap();
    }
    out.push_str("        }\n    }\n");
    for (name, auxiliary) in [("visit_type_operands", false), ("visit_operands", true)] {
        writeln!(out, "    pub fn {name}<F: FnMut(Value)>(&self, dfg: &DataFlowGraph, mut f: F) {{\n        match self {{").unwrap();
        for layout in layouts {
            writeln!(out, "            {} => {{", layout.pattern()).unwrap();
            for (i, field) in layout.fields.iter().enumerate() {
                if field.ty.auxiliary() && !auxiliary {
                    continue;
                }
                if let Some(kind) = field.ty.traversal() {
                    let visit = match kind {
                        "value" => format!("f(*_field{i});"),
                        "array" => format!("for &value in _field{i} {{ f(value); }}"),
                        _ => format!("dfg.visit_{kind}(*_field{i}, &mut f);"),
                    };
                    writeln!(out, "                {visit}").unwrap();
                }
            }
            out.push_str("            },\n");
        }
        out.push_str("        }\n    }\n");
    }
    out.push_str("    pub fn replace_value(&mut self, dfg: &mut DataFlowGraph, old_val: Value, new_val: Value) {\n        match self {\n");
    for layout in layouts {
        writeln!(out, "            {} => {{", layout.pattern()).unwrap();
        for (i, field) in layout.fields.iter().enumerate() {
            if let Some(kind) = field.ty.traversal() {
                let replace = match kind {
                    "value" => format!("if *_field{i} == old_val {{ *_field{i} = new_val; }}"),
                    "array" => format!(
                        "for value in _field{i} {{ if *value == old_val {{ *value = new_val; }} }}"
                    ),
                    "block_call" | "jump_table" => {
                        format!("dfg.replace_{kind}(*_field{i}, old_val, new_val);")
                    }
                    _ => format!("dfg.replace_{kind}(_field{i}, old_val, new_val);"),
                };
                writeln!(out, "                {replace}").unwrap();
            }
        }
        out.push_str("            },\n");
    }
    out.push_str("        }\n    }\n    pub fn memory_flags(&self, dfg: &DataFlowGraph) -> Option<MemFlags> {\n        match self {\n");
    for layout in layouts {
        let value = layout.fields.iter().enumerate().find_map(|(i, f)| {
            if f.ty.named("MemFlags") {
                Some(format!("Some(*_field{i})"))
            } else if f.ty.named("VectorMemExtId") {
                Some(format!("Some(crate::dfg::PoolKey::get(*_field{i}, dfg).expect(\"instruction refers to a missing vector memory extension\").flags)"))
            } else {
                None
            }
        }).unwrap_or_else(|| "None".to_owned());
        writeln!(out, "            {} => {value},", layout.pattern()).unwrap();
    }
    out.push_str("        }\n    }\n    /// Construct a values-only or nullary instruction in its canonical layout.\n    pub fn from_values(opcode: Opcode, values: &[Value]) -> Option<Self> {\n        match opcode.spec().format {\n");
    for layout in layouts {
        if !layout.canonical || !value_only(&layout.fields) {
            continue;
        }
        let arity = layout.arity().expect("inline values have a fixed arity");
        let mut index = 0;
        let mut fields = Vec::new();
        for field in &layout.fields {
            let value = match &field.ty {
                FieldType::Values(n) => {
                    let args = (index..index + n)
                        .map(|i| format!("values[{i}]"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    index += n;
                    format!("[{args}]")
                }
                FieldType::Named(name) if name == "Value" => {
                    let value = format!("values[{index}]");
                    index += 1;
                    value
                }
                FieldType::Named(name) if name == "Opcode" => "opcode".to_owned(),
                _ => unreachable!("validated values-only format"),
            };
            fields.push(if field.name == value {
                value
            } else {
                format!("{}: {value}", field.name)
            });
        }
        let construct = if fields.is_empty() {
            format!("Self::{}", layout.name)
        } else {
            format!("Self::{} {{ {} }}", layout.name, fields.join(", "))
        };
        let arity_check = if arity == 0 {
            "values.is_empty()".into()
        } else {
            format!("values.len() == {arity}")
        };
        writeln!(
            out,
            "            OpFormat::{} if {arity_check} => Some({construct}),",
            layout.name
        )
        .unwrap();
    }
    out.push_str("            _ => None,\n        }\n    }\n}\n");
    out
}

fn required<'a>(record: &'a Record, field: &str, source: &str) -> Result<&'a Node, Error> {
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

fn number(node: &Node, source: &str) -> Result<usize, Error> {
    match node.kind {
        Kind::Number(number) => Ok(number as usize),
        _ => Err(Error::at(source, node.offset, "expected a number")),
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

fn identifier(name: &str, offset: usize, source: &str) -> Result<(), Error> {
    let valid = !name.is_empty()
        && name
            .bytes()
            .next()
            .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
        && name != "_"
        && ![
            "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum",
            "extern", "false", "fn", "for", "gen", "if", "impl", "in", "let", "loop", "match",
            "mod", "move", "mut", "pub", "ref", "return", "self", "Self", "static", "struct",
            "super", "trait", "true", "type", "unsafe", "use", "where", "while", "yield",
            "abstract", "become", "box", "do", "final", "macro", "override", "priv", "try",
            "typeof", "unsized", "virtual",
        ]
        .contains(&name);
    if valid {
        Ok(())
    } else {
        Err(Error::at(
            source,
            offset,
            format!("invalid Rust identifier `{name}`"),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::compile;
    use crate::syntax;

    #[test]
    fn rejects_format_level_text_definitions() {
        let source = "format Binary { fields: [opcode(Opcode), args(values(2))], opcode: dynamic(opcode), text: Text { args: [args] } }";
        let error = compile(&syntax::parse(source).unwrap(), source).unwrap_err();
        assert!(error.message.contains("text"));
    }

    #[test]
    fn rejects_missing_or_mistyped_opcode_field() {
        for fields in ["[arg(Value)]", "[opcode(Value)]"] {
            let source = format!("format Unary {{ fields: {fields}, opcode: dynamic(opcode) }}");
            assert!(compile(&syntax::parse(&source).unwrap(), &source).is_err());
        }
    }

    #[test]
    fn rejects_unknown_layout_target() {
        let source = "layout Extended { fields: [opcode(Opcode), args(ValueList)], opcode: dynamic(opcode), format: arity(args, [Missing]), text: Text { args: [args] } }";
        let error = compile(&syntax::parse(source).unwrap(), source).unwrap_err();
        assert!(error.message.contains("unknown format"));
    }

    #[test]
    fn rejects_unknown_fields_and_storage_types() {
        for source in [
            "format Unary { fields: [arg(Unrecognized)], opcode: dynamic(arg) }",
            "format Unary { fields: [opcode(Opcode), arg(Value)], opcode: dynamic(opcode), typo: true }",
            "format Unary { fields: [opcode(Opcode), arg(Value), arg(Value)], opcode: dynamic(opcode) }",
        ] {
            assert!(compile(&syntax::parse(source).unwrap(), source).is_err());
        }
    }

    #[test]
    fn rejects_an_opcode_that_violates_an_existing_layout_contract() {
        let source = "format Iconst { fields: [value(u64)], opcode: fixed(Fconst) }";
        let error = compile(&syntax::parse(source).unwrap(), source).unwrap_err();
        assert!(
            error
                .message
                .contains("opcode contract requires fixed(Iconst)")
        );
    }

    #[test]
    fn rejects_alternate_layout_with_wrong_fixed_opcode() {
        let source = r#"
            format Unary { fields: [opcode(Opcode), arg(Value)], opcode: dynamic(opcode) }
            format Binary { fields: [opcode(Opcode), args(values(2))], opcode: dynamic(opcode) }
            op Neg(arg: I32) -> (result: I32) { storage: Unary { arg: arg } }
            layout Pair { fields: [args(values(2))], opcode: fixed(Neg), format: fixed(Binary), text: Text { args: [args] } }
        "#;
        let error = compile(&syntax::parse(source).unwrap(), source).unwrap_err();
        assert!(error.message.contains("requires format `Unary`"));
    }
}
