//! Register/attribute storage. A checked layout is shared by all emitters.
use crate::model::records::PropertyType;
use crate::model::{Op, ParamKind, TypeList};
use crate::syntax::{Kind, Node, Record};
use crate::{Error, model};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug)]
pub(crate) struct Operands {
    pub(crate) formats: BTreeMap<String, Format>,
    pub(crate) prefix: String,
    pub(crate) opcode: String,
    pub(crate) view: String,
    pub(crate) reader: String,
    pub(crate) writer: String,
    pub(crate) register: String,
    pub(crate) register_rust: String,
    pub(crate) attributes: String,
    pub(crate) control: Option<(String, String, BTreeSet<String>)>,
}
#[derive(Debug)]
pub(crate) struct Format {
    pub(crate) name: String,
    pub(crate) fields: Vec<Field>,
}
#[derive(Debug, Clone)]
pub(crate) struct Field {
    pub(crate) name: String,
    pub(crate) ty: String,
    pub(crate) rust: String,
    pub(crate) shape: Shape,
    pub(crate) codec: Option<String>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Shape {
    One,
    Optional,
    Sequence,
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Domain {
    Result,
    Input,
    Attribute,
}
impl Domain {
    pub(crate) fn accessor(self) -> &'static str {
        match self {
            Self::Result => "results",
            Self::Input => "inputs",
            Self::Attribute => "fields",
        }
    }
}
// Resolved physical slot. Results follow the logical signature; the other
// domains follow declaration order. An absent optional field consumes no slot.
pub(crate) struct Member {
    pub(crate) field: Field,
    pub(crate) domain: Domain,
    pub(crate) index: usize,
    pub(crate) binding: Option<String>,
}
// One checked plan drives construction, borrowed access and opt-in validation.
pub(crate) struct Projection {
    pub flow: String,
    pub(crate) members: Vec<Member>,
    pub(crate) args: Vec<Argument>,
    pub(crate) counts: [usize; 3],
    pub(crate) tails: [bool; 3],
}
pub(crate) struct Argument {
    pub(crate) name: String,
    pub(crate) rust: String,
}
pub(crate) fn domain_index(domain: Domain) -> usize {
    match domain {
        Domain::Result => 0,
        Domain::Input => 1,
        Domain::Attribute => 2,
    }
}

pub(crate) fn compile(
    records: &[Record],
    source: &str,
    data: &model::data::Types,
) -> Result<Operands, Error> {
    let record = records
        .iter()
        .find(|r| r.kind == "storage")
        .expect("storage declaration");
    let mut config = model::Fields::new(source, record.clone());
    let prefix = match config.optional("prefix") {
        Some(Node {
            kind: Kind::Text(prefix),
            ..
        }) => prefix,
        Some(node) => {
            return Err(Error::at(
                source,
                node.offset,
                "expected opcode prefix string",
            ));
        }
        None => String::new(),
    };
    let opcode = model::name(source, config.take("opcode")?)?;
    let view = model::name(source, config.take("view")?)?;
    let reader = model::name(source, config.take("reader")?)?;
    let writer = model::name(source, config.take("writer")?)?;
    let names = [&opcode, &view, &reader, &writer];
    if names.iter().collect::<BTreeSet<_>>().len() != names.len() {
        return Err(config.error("generated type names must be distinct"));
    }
    for name in names {
        model::identifier(source, record.offset, name)?;
        if data.names.contains(name) || data.rust.contains(name) {
            return Err(config.error(&format!(
                "generated type '{name}' conflicts with a declaration"
            )));
        }
    }
    let register = model::name(source, config.take("register")?)?;
    if !data.rust.contains(&register) {
        return Err(config.error("register requires a declared Rust type"));
    }
    let attributes = model::name(source, config.take("attributes")?)?;
    let attr_enum = data
        .enums
        .iter()
        .find(|e| e.name == attributes)
        .ok_or_else(|| config.error("attributes requires an enum declaration"))?;
    let mut codecs = BTreeMap::new();
    for (variant, payload) in &attr_enum.variants {
        let [PropertyType::Named(ty)] = payload.as_slice() else {
            return Err(config.error("attribute variants require one named payload type"));
        };
        if ty == &register {
            return Err(config.error("registers cannot also be attribute payloads"));
        }
        if codecs
            .insert(ty.clone(), format!("{attributes}::{variant}"))
            .is_some()
        {
            return Err(config.error("attribute payload types must be unique"));
        }
    }
    let control = if let Some(node) = config.optional("control") {
        let path = model::name(source, node)?;
        let (owner, default) = path
            .rsplit_once("::")
            .ok_or_else(|| config.error("control requires Enum::DefaultVariant"))?;
        let en = data
            .enums
            .iter()
            .find(|e| e.name == owner)
            .ok_or_else(|| config.error("unknown control enum"))?;
        if en.variants.iter().any(|(_, args)| !args.is_empty()) {
            return Err(config.error("control variants cannot have payloads"));
        }
        let names: BTreeSet<_> = en.variants.iter().map(|(n, _)| n.clone()).collect();
        if !names.contains(default) {
            return Err(config.error("unknown default control variant"));
        }
        Some((owner.to_owned(), default.to_owned(), names))
    } else {
        None
    };
    config.finish()?;
    if let Some(r) = records.iter().find(|r| r.kind == "layout") {
        return Err(Error::at(
            source,
            r.offset,
            "operand layouts are derived from structs",
        ));
    }
    let mut formats = BTreeMap::new();
    for shape in &data.records {
        let used = records.iter().filter(|r| r.kind == "op").any(|op| matches!(op.fields.get("storage"), Some(Node { kind: Kind::Object(name, _), .. }) if name == &shape.name));
        if !used {
            continue;
        }
        let mut fields = Vec::new();
        for field in &shape.fields {
            let (ty, cardinality) = match &field.ty {
                PropertyType::Named(ty) => (ty, Shape::One),
                PropertyType::Optional(ty) => (ty, Shape::Optional),
                PropertyType::Sequence(ty) => (ty, Shape::Sequence),
                _ => {
                    return Err(Error::at(
                        source,
                        0,
                        "operand fields require a named, optional or sequence type",
                    ));
                }
            };
            let codec = if ty == &register {
                None
            } else {
                Some(
                    codecs
                        .get(ty)
                        .ok_or_else(|| {
                            Error::at(
                                source,
                                0,
                                format!("no attribute variant stores type '{ty}'"),
                            )
                        })?
                        .clone(),
                )
            };
            if cardinality == Shape::Sequence && codec.is_some() {
                return Err(Error::at(
                    source,
                    0,
                    "attribute sequences require a typed attribute pool; this storage supports register sequences",
                ));
            }
            fields.push(Field {
                name: field.name.clone(),
                ty: ty.clone(),
                rust: data.rust.rust(ty),
                shape: cardinality,
                codec,
            });
        }
        formats.insert(
            shape.name.clone(),
            Format {
                name: shape.name.clone(),
                fields,
            },
        );
    }
    Ok(Operands {
        formats,
        prefix,
        opcode,
        view,
        reader,
        writer,
        register_rust: data.rust.rust(&register),
        register,
        attributes,
        control,
    })
}
impl Field {
    pub(crate) fn view_type(&self) -> String {
        match self.shape {
            Shape::One => self.rust.clone(),
            Shape::Optional => format!("Option<{}>", self.rust),
            Shape::Sequence => format!("&'a [{}]", self.rust),
        }
    }
}
impl Member {
    pub(crate) fn read_from(&self, receiver: &str) -> String {
        if self.binding.is_none() {
            return "None".into();
        }
        let access = self.domain.accessor();
        let value = if let Some(codec) = &self.field.codec {
            format!(
                "match {receiver}.fields()[{}] {{ {codec}(value) => value, _ => panic!(\"invalid instruction field\") }}",
                self.index
            )
        } else if self.field.shape == Shape::Sequence {
            format!("&{receiver}.{access}()[{}..]", self.index)
        } else {
            format!("{receiver}.{access}()[{}]", self.index)
        };
        if self.field.shape == Shape::Optional {
            format!("Some({value})")
        } else {
            value
        }
    }
}
impl Projection {
    pub(crate) fn inputs(&self) -> crate::model::access::Inputs {
        use crate::model::access::Access;
        self.members
            .iter()
            .filter(|m| m.domain != Domain::Result)
            .filter_map(|m| {
                let name = m.binding.as_ref()?;
                let access = Access::Field(m.field.name.clone());
                let access = if m.field.shape == Shape::Optional {
                    Access::Required(Box::new(access))
                } else {
                    access
                };
                Some((name.clone(), access))
            })
            .collect()
    }
    pub(crate) fn projections(
        &self,
        op: &Op,
        required: impl Fn(String) -> String,
    ) -> std::collections::BTreeMap<String, String> {
        crate::model::access::projections(
            op,
            "self",
            |name| {
                self.members
                    .iter()
                    .find(|m| m.field.name == name)
                    .expect("checked field")
                    .read_from("self")
            },
            required,
        )
        .into_iter()
        .collect()
    }
}
impl Operands {
    pub(crate) fn construction(
        &self,
        op: &Op,
        receiver: &str,
        result: &str,
        local: impl Fn(&str) -> String,
    ) -> crate::generate::construction::Write {
        use crate::generate::construction::{Write, slice};
        let plan = op.operands();
        let mut args = vec![format!("{}::{}", self.opcode, op.name)];
        for domain in [Domain::Result, Domain::Input, Domain::Attribute] {
            let mut members: Vec<_> = plan
                .members
                .iter()
                .filter(|m| m.domain == domain && m.binding.is_some())
                .collect();
            members.sort_by_key(|m| m.index);
            let tail = if members
                .last()
                .is_some_and(|m| m.field.shape == Shape::Sequence)
            {
                Some(local(members.pop().unwrap().binding.as_ref().unwrap()))
            } else {
                None
            };
            let items: Vec<_> = members
                .iter()
                .map(|m| {
                    let value = local(m.binding.as_ref().unwrap());
                    if let Some(codec) = &m.field.codec {
                        format!("{codec}({value})")
                    } else if domain == Domain::Result {
                        format!("{result}({value})")
                    } else {
                        value
                    }
                })
                .collect();
            args.push(slice(
                if domain == Domain::Attribute {
                    &self.attributes
                } else {
                    &self.register_rust
                },
                &items,
                tail,
            ));
        }
        Write {
            callee: format!("{receiver}.write"),
            args,
        }
    }
    pub(crate) fn properties(
        &self,
        source: &str,
        offset: usize,
        format: &str,
        mappings: &BTreeMap<String, Node>,
        _params: &[crate::syntax::Parameter],
    ) -> Result<BTreeSet<String>, Error> {
        let shape = self
            .formats
            .get(format)
            .ok_or_else(|| Error::at(source, offset, "unknown operand format"))?;
        let mut properties = BTreeSet::new();
        for field in &shape.fields {
            if field.codec.is_none() {
                continue;
            }
            let Some(mut node) = mappings.get(&field.name) else {
                continue;
            };
            if let Kind::Call(name, args) = &node.kind {
                if name == "some" && args.len() == 1 {
                    node = &args[0];
                }
            }
            if let Kind::Name(name) = &node.kind {
                if name != "none" {
                    properties.insert(name.clone());
                }
            }
        }
        Ok(properties)
    }
    pub(crate) fn record_names(&self) -> Vec<String> {
        self.formats
            .keys()
            .cloned()
            .chain(core::iter::once(self.attributes.clone()))
            .collect()
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
        let fail = |message: &str| Error::at(source, offset, message);
        let shape = self
            .formats
            .get(format)
            .ok_or_else(|| fail("unknown operand format"))?;
        if mappings.len() != shape.fields.len()
            || mappings
                .keys()
                .any(|n| !shape.fields.iter().any(|f| f.name == *n))
        {
            return Err(fail("storage mapping must bind every field exactly once"));
        }
        let flow = match (&self.control, flow) {
            (Some((_, default, names)), flow) => {
                let name = flow
                    .map(|n| model::name(source, n))
                    .transpose()?
                    .unwrap_or_else(|| default.clone());
                if !names.contains(&name) {
                    return Err(fail("unknown control flow variant"));
                }
                name
            }
            (None, None) => String::new(),
            (None, Some(_)) => {
                return Err(fail("flow requires a declared control enum in storage"));
            }
        };
        let mut used_params = BTreeSet::new();
        let mut used_results = BTreeSet::new();
        let mut all_results = false;
        let mut args = BTreeMap::new();
        let mut members = Vec::new();
        let mut counts = [0; 3];
        let mut tails = [false; 3];
        let result_count = match &signature.results {
            TypeList::Fixed(results) => Some(results.len()),
            TypeList::Signature => None,
            _ => return Err(fail("unsupported result contract for register storage")),
        };
        for field in &shape.fields {
            let mut node = &mappings[&field.name];
            if field.shape == Shape::Optional {
                match &node.kind {
                    Kind::Name(n) if n == "none" => {
                        members.push(Member {
                            field: field.clone(),
                            domain: if field.codec.is_some() {
                                Domain::Attribute
                            } else {
                                Domain::Input
                            },
                            index: 0,
                            binding: None,
                        });
                        continue;
                    }
                    Kind::Call(n, items) if n == "some" && items.len() == 1 => node = &items[0],
                    _ => return Err(fail("optional fields require some(value) or none")),
                }
            }
            if matches!(&node.kind, Kind::Call(n, items) if n == "results" && items.is_empty()) {
                if field.ty != self.register
                    || field.shape != Shape::Sequence
                    || all_results
                    || !used_results.is_empty()
                {
                    return Err(fail(
                        "results() requires one register sequence and cannot mix with individual results",
                    ));
                }
                if params.iter().any(|p| p.name == field.name) {
                    return Err(fail("result slice name conflicts with an input"));
                }
                all_results = true;
                tails[0] = result_count.is_none();
                counts[0] = result_count.unwrap_or(0);
                args.insert(
                    (false, 0),
                    Argument {
                        name: field.name.clone(),
                        rust: format!("&[{}]", self.register_rust),
                    },
                );
                members.push(Member {
                    field: field.clone(),
                    domain: Domain::Result,
                    index: 0,
                    binding: Some(field.name.clone()),
                });
                continue;
            }
            let name = model::name(source, node.clone())?;
            let result = slots.get(&name).filter(|s| s.result);
            let (domain, order) = if let Some(result) = result {
                if field.ty != self.register
                    || field.shape != Shape::One
                    || all_results
                    || !used_results.insert(result.index)
                {
                    return Err(fail("invalid or repeated result binding"));
                }
                (Domain::Result, usize::from(result.index))
            } else {
                let (index, param) = params
                    .iter()
                    .enumerate()
                    .find(|(_, p)| p.name == name)
                    .ok_or_else(|| fail(&format!("unknown input '{name}'")))?;
                if !used_params.insert(index) {
                    return Err(fail("input is stored more than once"));
                }
                let compatible = if field.ty == self.register {
                    param.kind
                        == if field.shape == Shape::Sequence {
                            ParamKind::Values
                        } else {
                            ParamKind::Value
                        }
                } else {
                    param.kind == ParamKind::Property(field.ty.clone())
                };
                if !compatible {
                    return Err(fail(&format!(
                        "input '{name}' is incompatible with field '{}'",
                        field.name
                    )));
                }
                (
                    if field.codec.is_some() {
                        Domain::Attribute
                    } else {
                        Domain::Input
                    },
                    index,
                )
            };
            let d = domain_index(domain);
            if tails[d] {
                return Err(fail("a storage domain can have only one trailing sequence"));
            }
            let index = if domain == Domain::Result {
                order
            } else {
                counts[d]
            };
            if field.shape == Shape::Sequence {
                tails[d] = true;
            } else {
                counts[d] += 1;
            }
            let rust = if field.shape == Shape::Sequence {
                format!("&[{}]", field.rust)
            } else if domain == Domain::Result {
                "Self::Def".to_owned()
            } else {
                field.rust.clone()
            };
            args.insert(
                (domain != Domain::Result, order),
                Argument {
                    name: name.clone(),
                    rust,
                },
            );
            members.push(Member {
                field: field.clone(),
                domain,
                index,
                binding: Some(name),
            });
        }
        if used_params.len() != params.len() {
            return Err(fail("every input requires a storage mapping"));
        }
        if !all_results && result_count != Some(used_results.len()) {
            return Err(fail("every result requires a storage mapping"));
        }
        Ok(Projection {
            flow,
            members,
            args: args.into_values().collect(),
            counts,
            tails,
        })
    }
}
