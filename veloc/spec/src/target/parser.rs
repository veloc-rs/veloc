//! Checked target projections from the shared Spec declaration AST.
//! Lexing, imports, templates and source diagnostics belong to the common frontend.
use super::ast::*;
mod selection;
use crate::{
    Error,
    syntax::{Decl, DeclKind, Kind, Node},
};
use std::collections::{BTreeMap, BTreeSet};

struct Reader<'a> {
    source: &'a str,
    aliases: BTreeMap<String, Node>,
    types: crate::types::Types,
    bindings: crate::interfaces::Bindings,
}
impl Reader<'_> {
    fn cpu_schedule(&self, node: &Node) -> Result<CpuSchedule, Error> {
        let positive = |n: &Node| {
            u32::try_from(self.number(n)?)
                .ok()
                .filter(|&v| v > 0)
                .ok_or_else(|| self.error(n, "scheduling costs must be positive u32 values"))
        };
        let mut result = CpuSchedule::default();
        for (field, value) in self.record(node)? {
            match field.as_str() {
                "issue_width" => result.issue_width = positive(value)?,
                "resources" | "classes" => {
                    let resource = field == "resources";
                    let allowed: &[&str] = if resource {
                        &["name", "units"]
                    } else {
                        &["class", "resource", "latency", "occupancy"]
                    };
                    let mut names = BTreeSet::new();
                    for entry in self.list(value)? {
                        let fields = self.record(entry)?;
                        for key in fields.keys() {
                            if !allowed.contains(&key.as_str()) {
                                return Err(self.error(
                                    entry,
                                    &format!("unknown scheduling {field} field {key}"),
                                ));
                            }
                        }
                        let required = |key| {
                            fields.get(key).ok_or_else(|| {
                                self.error(
                                    entry,
                                    &format!("missing scheduling {field} field {key}"),
                                )
                            })
                        };
                        let name = if resource {
                            self.name(required("name")?)?
                        } else {
                            let node = required("class")?;
                            let Kind::Name(name) = &node.kind else {
                                return Err(
                                    self.error(node, "expected a scheduling class reference")
                                );
                            };
                            name.clone()
                        };
                        if !names.insert(name.clone()) {
                            return Err(self.error(
                                entry,
                                &format!("duplicate scheduling {field} name {name}"),
                            ));
                        }
                        if resource {
                            result.resources.push(ScheduleResource {
                                name,
                                units: positive(required("units")?)?,
                            });
                        } else {
                            result.classes.push(ScheduleCost {
                                class: name,
                                resource: self.name(required("resource")?)?,
                                latency: positive(required("latency")?)?,
                                occupancy: positive(required("occupancy")?)?,
                            });
                        }
                    }
                }
                _ => return Err(self.error(value, "unknown CPU scheduling field")),
            }
        }
        for class in &result.classes {
            if !result.resources.iter().any(|r| r.name == class.resource) {
                return Err(self.error(
                    node,
                    &format!("unknown scheduling resource {}", class.resource),
                ));
            }
        }
        Ok(result)
    }

    fn error(&self, node: &Node, message: &str) -> Error {
        Error::at(self.source, node.offset, message)
    }
    fn name(&self, node: &Node) -> Result<String, Error> {
        match &node.kind {
            Kind::Name(n) | Kind::Text(n) => Ok(n.clone()),
            _ => Err(self.error(node, "expected a name")),
        }
    }
    fn number(&self, node: &Node) -> Result<i64, Error> {
        match &node.kind {
            Kind::Number(n) => Ok(i64::from(*n)),
            Kind::Integer(n) => {
                i64::try_from(*n).map_err(|_| self.error(node, "integer exceeds i64"))
            }
            _ => Err(self.error(node, "expected an integer")),
        }
    }
    fn list<'a>(&self, node: &'a Node) -> Result<&'a [Node], Error> {
        if let Kind::List(nodes) = &node.kind {
            Ok(nodes)
        } else {
            Err(self.error(node, "expected a list"))
        }
    }
    fn names(&self, node: &Node) -> Result<Vec<String>, Error> {
        self.list(node)?.iter().map(|n| self.name(n)).collect()
    }
    fn feature_names(&self, node: &Node) -> Result<Vec<String>, Error> {
        self.list(node)?
            .iter()
            .map(|node| match &node.kind {
                Kind::Name(name) => Ok(name.clone()),
                _ => Err(self.error(node, "expected a reference to a declared feature")),
            })
            .collect()
    }
    fn record<'a>(&self, node: &'a Node) -> Result<&'a BTreeMap<String, Node>, Error> {
        if let Kind::Record(fields) = &node.kind {
            Ok(fields)
        } else {
            Err(self.error(node, "expected a record"))
        }
    }
    fn required<'a>(&self, d: &'a Decl, key: &str) -> Result<&'a Node, Error> {
        d.fields
            .get(key)
            .ok_or_else(|| Error::at(self.source, d.offset, format!("missing target field {key}")))
    }
    fn fields(&self, d: &Decl, allowed: &[&str]) -> Result<(), Error> {
        for (key, value) in &d.fields {
            if !allowed.contains(&key.as_str()) {
                return Err(self.error(value, &format!("unknown {} field {key}", d.name)));
            }
        }
        Ok(())
    }
    fn pattern(&self, n: &Node) -> Result<Pattern, Error> {
        Ok(match &n.kind {
            Kind::Name(v) if v.starts_with("CC::") => Pattern::CondCode(match &v[4..] {
                "E" => CondCode::E,
                "NE" => CondCode::NE,
                "L" => CondCode::L,
                "LE" => CondCode::LE,
                "G" => CondCode::G,
                "GE" => CondCode::GE,
                "B" => CondCode::B,
                "BE" => CondCode::BE,
                "A" => CondCode::A,
                "AE" => CondCode::AE,
                _ => return Err(self.error(n, "unknown condition code")),
            }),
            Kind::Name(v) => Pattern::Variable(v.clone()),
            Kind::TypedCall(name, types, args) if name == "Value" => {
                let ([ty], [value]) = (types.as_slice(), args.as_slice()) else {
                    return Err(
                        self.error(n, "Value requires one type domain and one value binding")
                    );
                };
                let name = match &value.kind {
                    Kind::Name(name) => name.clone(),
                    _ => return Err(self.error(value, "expected a value binding")),
                };
                let types = self.selection_domain(ty)?;
                Pattern::Typed { name, types }
            }
            Kind::Number(_) | Kind::Integer(_) => Pattern::IntConst(self.number(n)?),
            Kind::Object(path, fields) => {
                let (schema, opcode) = path
                    .split_once("::")
                    .ok_or_else(|| self.error(n, "expected Schema::Opcode pattern"))?;
                Pattern::Schema {
                    schema: schema.into(),
                    opcode: opcode.into(),
                    args: fields
                        .iter()
                        .map(|(name, p)| {
                            Ok(PatternArg::Named {
                                name: name.clone(),
                                pattern: Box::new(self.pattern(p)?),
                            })
                        })
                        .collect::<Result<_, Error>>()?,
                }
            }
            Kind::Call(name, args) if name == "bind" => {
                let [node, inner] = args.as_slice() else {
                    return Err(self.error(n, "bind requires a node name and a pattern"));
                };
                Pattern::NodeBind {
                    node: self.name(node)?,
                    inner: Box::new(self.pattern(inner)?),
                }
            }
            Kind::Call(name, args) if name == "all" => Pattern::And(
                args.iter()
                    .map(|p| self.pattern(p))
                    .collect::<Result<_, _>>()?,
            ),
            Kind::Call(name, args) if name == "stackslot" => {
                let [p] = args.as_slice() else {
                    return Err(self.error(n, "stackslot requires one pattern"));
                };
                Pattern::StackSlot(Box::new(self.pattern(p)?))
            }
            Kind::Call(name, args) => Pattern::Opcode {
                opcode: name.clone(),
                ty: None,
                args: args
                    .iter()
                    .map(|p| self.pattern(p).map(PatternArg::Positional))
                    .collect::<Result<_, _>>()?,
            },
            _ => return Err(self.error(n, "expected an instruction pattern")),
        })
    }
    fn constructor(&self, n: &Node) -> Result<Constructor, Error> {
        Ok(match &n.kind {
            Kind::Name(v) => Constructor::Variable(v.clone()),
            Kind::Number(_) | Kind::Integer(_) => Constructor::Imm(self.number(n)?),
            Kind::Call(name, args) if name == "reg" => {
                let [r] = args.as_slice() else {
                    return Err(self.error(n, "reg requires one register"));
                };
                Constructor::Reg(self.name(r)?)
            }
            Kind::Call(name, args) => Constructor::Inst {
                opcode: name.clone(),
                args: args
                    .iter()
                    .map(|n| self.constructor(n))
                    .collect::<Result<_, _>>()?,
            },
            _ => return Err(self.error(n, "expected an instruction constructor")),
        })
    }
    fn abi_rules(&self, n: &Node) -> Result<Vec<AbiRuleDef>, Error> {
        self.list(n)?.iter().map(|node| {
            let (node, transport) = if let Kind::Call(name, args) = &node.kind
                && name == "bitcast"
            {
                let [ty, action] = args.as_slice() else {
                    return Err(self.error(node, "expected bitcast(type, allocation)"));
                };
                let types = self.selection_domain(ty)?;
                if types.len() != 1 {
                    return Err(self.error(ty, "ABI transport must be one concrete type"));
                }
                (action, types.into_iter().next())
            } else {
                (node, None)
            };
            let Kind::Call(name, args) = &node.kind else {
                return Err(self.error(node, "expected an ABI allocation action"));
            };
            let action = match (name.as_str(), args.as_slice()) {
                ("assign", [_, regs]) => AbiActionDef::Reg {
                    regs: self.names(regs)?, shadows: Vec::new(),
                },
                ("shadow", [_, regs, shadows]) => AbiActionDef::Reg {
                    regs: self.names(regs)?, shadows: self.names(shadows)?,
                },
                ("stack", [_, size, align]) => AbiActionDef::Stack {
                    size: u32::try_from(self.number(size)?)
                        .map_err(|_| self.error(size, "invalid ABI stack size"))?,
                    align: u32::try_from(self.number(align)?)
                        .map_err(|_| self.error(align, "invalid ABI stack alignment"))?,
                },
                _ => return Err(self.error(node, "expected assign(types, regs), shadow(types, regs, shadows), or stack(types, size, align)")),
            };
            Ok(AbiRuleDef { types: self.selection_domain(&args[0])?, transport, action })
        }).collect()
    }
    fn selection_domain(&self, node: &Node) -> Result<Vec<String>, Error> {
        let domain =
            crate::rules::typed::domain(self.source, node, &self.aliases, &mut BTreeSet::new())?;
        domain
            .iter()
            .map(|ty| {
                let Some(("Type", member)) = ty.split_once("::") else {
                    return Err(self.error(node, "expected a logical Type constant"));
                };
                if !self.types.exact.contains_key(member) {
                    return Err(self.error(node, &format!("unknown logical type {ty}")));
                }
                let binding =
                    self.bindings.0.get("Type").ok_or_else(|| {
                        self.error(node, "Type requires an imported Rust binding")
                    })?;
                Ok(format!("{}::{member}", binding.path))
            })
            .collect()
    }

    fn selection_type(&self, node: &Node) -> Result<(String, Vec<Vec<String>>), Error> {
        let (name, args) = match &node.kind {
            Kind::Name(name) => (name, &[][..]),
            Kind::Call(name, args) => (name, args.as_slice()),
            _ => return Err(self.error(node, "expected a qualified operation type")),
        };
        if !name.contains("::") {
            return Err(self.error(node, "expected a qualified operation type"));
        }
        Ok((
            name.clone(),
            args.iter()
                .map(|arg| self.selection_domain(arg))
                .collect::<Result<_, _>>()?,
        ))
    }

    fn selection(&self, d: &Decl, signature: &crate::syntax::Signature) -> Result<Vec<Def>, Error> {
        let [root] = signature.params.as_slice() else {
            return Err(Error::at(
                self.source,
                d.offset,
                "selection requires one root parameter",
            ));
        };
        if root.moves
            || !signature.generics.is_empty()
            || !matches!(&signature.results, crate::syntax::Results::Fixed(values) if values.is_empty())
        {
            return Err(Error::at(
                self.source,
                d.offset,
                "selection requires a plain root parameter and no results",
            ));
        }
        let (opcode, type_args) = self.selection_type(&root.ty)?;
        let cases = self.list(self.required(d, "cases")?)?;
        if cases.is_empty() {
            return Err(Error::at(
                self.source,
                d.offset,
                "selection requires at least one case",
            ));
        }
        cases
            .iter()
            .map(|case| self.selection_case(case, &root.name, &opcode, &type_args))
            .collect()
    }

    fn declaration(&self, d: &Decl) -> Result<Option<Def>, Error> {
        let DeclKind::Fields(kind) = &d.kind else {
            return Ok(None);
        };
        Ok(Some(match kind.as_str() {
            "predicate" => {
                self.fields(d, &["params"])?;
                Def::Decl(DeclDef {
                    name: d.name.clone(),
                    params: self.names(self.required(d, "params")?)?,
                })
            }
            "extractor" => {
                self.fields(d, &["params", "pattern"])?;
                Def::Extractor(ExtractorDef {
                    name: d.name.clone(),
                    args: self.names(self.required(d, "params")?)?,
                    body: self.pattern(self.required(d, "pattern")?)?,
                })
            }
            "feature" => {
                self.fields(d, &["doc", "requires"])?;
                Def::Feature(FeatureDef {
                    name: d.name.clone(),
                    doc: d
                        .fields
                        .get("doc")
                        .map(|n| self.name(n))
                        .transpose()?
                        .unwrap_or_default(),
                    requires: d
                        .fields
                        .get("requires")
                        .map(|n| self.feature_names(n))
                        .transpose()?
                        .unwrap_or_default(),
                })
            }
            "schedule_class" => {
                self.fields(d, &["doc"])?;
                if d.name == "None" {
                    return Err(Error::at(
                        self.source,
                        d.offset,
                        "None is reserved for unscheduled instructions",
                    ));
                }
                Def::ScheduleClass(ScheduleClassDef {
                    name: d.name.clone(),
                    doc: d
                        .fields
                        .get("doc")
                        .map(|n| self.name(n))
                        .transpose()?
                        .unwrap_or_default(),
                })
            }
            "cpu" => {
                self.fields(d, &["name", "features", "schedule"])?;
                Def::Cpu(CpuDef {
                    name: d
                        .fields
                        .get("name")
                        .map(|n| self.name(n))
                        .transpose()?
                        .unwrap_or_else(|| d.name.clone()),
                    features: self.feature_names(self.required(d, "features")?)?,
                    schedule: d
                        .fields
                        .get("schedule")
                        .map(|n| self.cpu_schedule(n))
                        .transpose()?
                        .unwrap_or_default(),
                })
            }
            "data_layout" => {
                self.fields(d, &["endian", "pointer", "types"])?;
                let endian = self.required(d, "endian")?;
                let little_endian = match self.name(endian)?.as_str() {
                    "little" => true,
                    "big" => false,
                    _ => return Err(self.error(endian, "expected little or big endianness")),
                };
                let number = |n: &Node| -> Result<u32, Error> {
                    u32::try_from(self.number(n)?)
                        .map_err(|_| self.error(n, "layout size exceeds u32"))
                };
                let pointer_node = self.required(d, "pointer")?;
                let pointer = self.record(pointer_node)?;
                if pointer.len() != 2
                    || !pointer.contains_key("size")
                    || !pointer.contains_key("align")
                {
                    return Err(self.error(pointer_node, "pointer layout requires size and align"));
                }
                let mut types = Vec::new();
                for node in self.list(self.required(d, "types")?)? {
                    let fields = self.record(node)?;
                    if fields.len() != 3
                        || !fields.contains_key("types")
                        || !fields.contains_key("size")
                        || !fields.contains_key("align")
                    {
                        return Err(self.error(node, "type layout requires types, size and align"));
                    }
                    let (size, align) = (number(&fields["size"])?, number(&fields["align"])?);
                    for ty in self.selection_domain(&fields["types"])? {
                        types.push(TypeLayoutDef { ty, size, align });
                    }
                }
                Def::DataLayout(DataLayoutDef {
                    name: d.name.clone(),
                    little_endian,
                    pointer_size: number(&pointer["size"])?,
                    pointer_align: number(&pointer["align"])?,
                    types,
                })
            }
            "abi" => {
                self.fields(
                    d,
                    &[
                        "arch",
                        "layout",
                        "stack",
                        "args",
                        "variadic",
                        "returns",
                        "preserved",
                    ],
                )?;
                let mut stack = AbiStackDef::default();
                for (field, n) in self.record(self.required(d, "stack")?)? {
                    match field.as_str() {
                        "align" => {
                            stack.align = Some(
                                u32::try_from(self.number(n)?)
                                    .map_err(|_| self.error(n, "alignment exceeds u32"))?,
                            )
                        }
                        "reserved" => {
                            stack.reserved = u32::try_from(self.number(n)?)
                                .map_err(|_| self.error(n, "invalid reserved stack size"))?;
                        }
                        _ => return Err(self.error(n, "unknown ABI stack field")),
                    }
                }
                Def::Abi(AbiDef {
                    name: d.name.clone(),
                    arch: self.name(self.required(d, "arch")?)?,
                    layout: self.name(self.required(d, "layout")?)?,
                    stack,
                    args: self.abi_rules(self.required(d, "args")?)?,
                    variadic: d
                        .fields
                        .get("variadic")
                        .map(|n| self.abi_rules(n))
                        .transpose()?,
                    returns: self.abi_rules(self.required(d, "returns")?)?,
                    preserved: self.names(self.required(d, "preserved")?)?,
                })
            }
            // These declarations are checked by the other Spec consumers.
            "struct" | "enum" | "encoding" => return Ok(None),
            _ => {
                return Err(Error::at(
                    self.source,
                    d.offset,
                    format!("unsupported target declaration {kind}"),
                ));
            }
        }))
    }
}

pub(crate) fn declarations(source: &str, decls: &[Decl]) -> Result<Module, Error> {
    let reader = Reader {
        source,
        aliases: decls
            .iter()
            .filter_map(|d| match &d.kind {
                DeclKind::TypeSet(node) => Some((d.name.clone(), node.clone())),
                _ => None,
            })
            .collect(),
        types: crate::types::Types::compile(decls, source)?,
        bindings: crate::interfaces::Bindings::compile(decls, source)?,
    };
    let mut defs = Vec::new();
    let mut names = BTreeSet::new();
    for d in decls {
        if let DeclKind::Select(signature) = &d.kind {
            defs.extend(reader.selection(d, signature)?);
            continue;
        }
        if let Some(def) = reader.declaration(d)? {
            if !names.insert((d.tag(), d.name.clone())) {
                return Err(Error::at(source, d.offset, "duplicate target declaration"));
            }
            defs.push(def);
        }
    }
    Ok(Module { defs })
}
pub fn parse(source: &str) -> Result<Module, Error> {
    declarations(source, &crate::syntax::parse(source)?)
}
