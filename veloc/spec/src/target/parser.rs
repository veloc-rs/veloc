//! Checked target projections from the shared Spec declaration AST.
//! Lexing, imports, templates and source diagnostics belong to the common frontend.
use super::ast::*;
use crate::{
    Error,
    syntax::{Decl, DeclKind, Kind, Node},
};
use std::collections::{BTreeMap, BTreeSet};

struct Reader<'a> {
    source: &'a str,
}
impl Reader<'_> {
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
    fn abi_regs(&self, n: &Node) -> Result<Vec<AbiClassRegsDef>, Error> {
        self.record(n)?
            .iter()
            .map(|(class, regs)| {
                Ok(AbiClassRegsDef {
                    class: class.clone(),
                    regs: self.names(regs)?,
                })
            })
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
                        .map(|n| self.names(n))
                        .transpose()?
                        .unwrap_or_default(),
                })
            }
            "cpu" => {
                self.fields(d, &["name", "features"])?;
                Def::Cpu(CpuDef {
                    name: d
                        .fields
                        .get("name")
                        .map(|n| self.name(n))
                        .transpose()?
                        .unwrap_or_else(|| d.name.clone()),
                    features: self.names(self.required(d, "features")?)?,
                })
            }
            "select" => {
                self.fields(d, &["match", "emit", "temps", "covers", "cost"])?;
                let patterns = self
                    .list(self.required(d, "match")?)?
                    .iter()
                    .map(|n| self.pattern(n))
                    .collect::<Result<Vec<_>, _>>()?;
                if patterns.len() != 1 {
                    return Err(Error::at(
                        self.source,
                        d.offset,
                        "selection requires exactly one root pattern",
                    ));
                }
                let temps = d
                    .fields
                    .get("temps")
                    .map(|n| {
                        self.record(n)?
                            .iter()
                            .map(|(name, ty)| Ok((name.clone(), self.name(ty)?)))
                            .collect::<Result<Vec<_>, Error>>()
                    })
                    .transpose()?
                    .unwrap_or_default();
                Def::SelectRule(SelectRuleDef {
                    covers: d
                        .fields
                        .get("covers")
                        .map(|n| self.names(n))
                        .transpose()?
                        .unwrap_or_default(),
                    cost: d
                        .fields
                        .get("cost")
                        .map(|n| {
                            u32::try_from(self.number(n)?)
                                .map_err(|_| self.error(n, "cost must be a nonnegative u32"))
                        })
                        .transpose()?
                        .unwrap_or(1),
                    patterns,
                    temps,
                    emit: self.constructor(self.required(d, "emit")?)?,
                })
            }
            "abi" => {
                self.fields(
                    d,
                    &[
                        "arch",
                        "stack",
                        "args",
                        "returns",
                        "preserved",
                        "classifier",
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
                        "incoming" | "outgoing" => {
                            let Kind::Call(name, args) = &n.kind else {
                                return Err(self.error(n, "expected base or slot"));
                            };
                            let [a, b] = args.as_slice() else {
                                return Err(self.error(n, "expected two arguments"));
                            };
                            if field == "incoming" && name == "base" {
                                stack.incoming_base = Some((
                                    self.name(a)?,
                                    i32::try_from(self.number(b)?)
                                        .map_err(|_| self.error(b, "offset exceeds i32"))?,
                                ));
                            } else if field == "outgoing" && name == "slot" {
                                stack.outgoing_slot = Some((
                                    u32::try_from(self.number(a)?)
                                        .map_err(|_| self.error(a, "size exceeds u32"))?,
                                    u32::try_from(self.number(b)?)
                                        .map_err(|_| self.error(b, "alignment exceeds u32"))?,
                                ));
                            } else {
                                return Err(
                                    self.error(n, "expected incoming base or outgoing slot")
                                );
                            }
                        }
                        _ => return Err(self.error(n, "unknown ABI stack field")),
                    }
                }
                Def::Abi(AbiDef {
                    name: d.name.clone(),
                    arch: self.name(self.required(d, "arch")?)?,
                    stack,
                    args: self.abi_regs(self.required(d, "args")?)?,
                    returns: self.abi_regs(self.required(d, "returns")?)?,
                    preserved: self
                        .record(self.required(d, "preserved")?)?
                        .iter()
                        .map(|(bank, n)| {
                            Ok(AbiPreservedSetDef {
                                bank: bank.clone(),
                                regs: self.names(n)?,
                            })
                        })
                        .collect::<Result<_, Error>>()?,
                    classifier: d
                        .fields
                        .get("classifier")
                        .map(|n| self.name(n))
                        .transpose()?,
                })
            }
            // These declarations are checked by the other Spec consumers.
            "struct" | "enum" | "encoding" | "assembly" => return Ok(None),
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
    let reader = Reader { source };
    let mut defs = Vec::new();
    let mut names = BTreeSet::new();
    for d in decls {
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
