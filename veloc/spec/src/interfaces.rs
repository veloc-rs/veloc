//! Rust interface contracts. Paths and signatures are data, never runtime type identities.
use crate::{
    Error, Source,
    syntax::{Decl, DeclKind, FunctionBody, Kind, Node, Results},
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

#[derive(Debug, Clone)]
pub struct Binding {
    pub path: String,
    pub interface: Option<String>,
    mixed: bool,
}

#[derive(Debug, Clone, Default)]
pub struct Bindings(pub BTreeMap<String, Binding>);

pub fn rust_binding(record: &Decl) -> Option<&Node> {
    match &record.kind {
        DeclKind::Type { binding, .. } if matches!(&binding.kind, Kind::Call(name, _) if name == "rust") => {
            Some(binding)
        }
        _ => None,
    }
}

pub fn rust_path(source: &str, offset: usize, path: &str) -> Result<(), Error> {
    let parts = path.split("::").collect::<Vec<_>>();
    if parts.len() < 2
        || parts.iter().any(|part| {
            let mut chars = part.chars();
            !chars
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                || !chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        })
    {
        return Err(Error::at(
            source,
            offset,
            "Rust type path must be nonempty and qualified",
        ));
    }
    Ok(())
}

fn binding_path(source: &str, node: &Node) -> Result<String, Error> {
    if let Kind::Call(name, args) = &node.kind
        && name == "rust"
        && let [
            Node {
                kind: Kind::Text(path),
                ..
            },
        ] = args.as_slice()
    {
        rust_path(source, node.offset, path)?;
        return Ok(path.clone());
    }
    Err(Error::at(
        source,
        node.offset,
        "rust requires one type path string",
    ))
}

impl Bindings {
    pub fn compile(records: &[Decl], source: &str) -> Result<Self, Error> {
        let mut result = Self::default();
        for record in records {
            let Some(node) = rust_binding(record) else {
                continue;
            };
            if primitive(&record.name) {
                return Err(Error::at(
                    source,
                    record.offset,
                    "Rust type binding conflicts with a built-in type",
                ));
            }
            let path = binding_path(source, node)?;
            let interface = record
                .fields
                .get("trait")
                .map(|n| binding_path(source, n))
                .transpose()?;
            let mut has_const = false;
            let mut has_runtime = false;
            for method in record.members() {
                let is_const = match &method.kind {
                    DeclKind::Constant(_) => true,
                    DeclKind::Function {
                        signature,
                        body: FunctionBody::Rust { .. },
                    } => signature.is_const,
                    _ => continue,
                };
                if is_const {
                    has_const = true;
                } else {
                    has_runtime = true;
                }
            }
            if result
                .0
                .insert(
                    record.name.clone(),
                    Binding {
                        path,
                        interface,
                        mixed: has_const && has_runtime,
                    },
                )
                .is_some()
            {
                return Err(Error::at(
                    source,
                    record.offset,
                    "duplicate Rust type binding",
                ));
            }
        }
        Ok(result)
    }
    pub fn method_trait(&self, name: &str, namespace: &str, is_const: bool) -> String {
        let binding = &self.0[name];
        let mut path = binding
            .interface
            .clone()
            .unwrap_or_else(|| format!("{namespace}::{name}"));
        if is_const && binding.mixed {
            path.push_str("Const");
        }
        path
    }
}

pub(crate) fn primitive(name: &str) -> bool {
    matches!(
        name,
        "bool"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "u128"
            | "i8"
            | "i16"
            | "i32"
            | "i64"
            | "i128"
            | "f32"
            | "f64"
    )
}

/// Neutral signature types, usable without loading an implementation.
#[derive(Debug, Clone)]
pub enum Type {
    Named(String),
    Ref(Box<Self>),
    Optional(Box<Self>),
    Sequence(Box<Self>),
    Array(Box<Self>, u32),
}

impl Type {
    fn parse(node: &Node, source: &str, known: &BTreeSet<String>) -> Result<Self, Error> {
        match &node.kind {
            Kind::Ref(inner) => Ok(Self::Ref(Box::new(Self::parse(inner, source, known)?))),
            Kind::Name(name) if name == "Self" || primitive(name) || known.contains(name) => {
                Ok(Self::Named(name.clone()))
            }
            Kind::Call(name, args)
                if matches!(name.as_str(), "optional" | "sequence") && args.len() == 1 =>
            {
                let inner = Box::new(Self::parse(&args[0], source, known)?);
                Ok(if name == "optional" {
                    Self::Optional(inner)
                } else {
                    Self::Sequence(inner)
                })
            }
            Kind::Call(name, args) if name == "array" && args.len() == 2 => {
                let Kind::Number(n) = args[1].kind else {
                    return Err(Error::at(
                        source,
                        args[1].offset,
                        "array length must be a number",
                    ));
                };
                Ok(Self::Array(
                    Box::new(Self::parse(&args[0], source, known)?),
                    n,
                ))
            }
            _ => Err(Error::at(source, node.offset, "unknown interface type")),
        }
    }
    fn rust(&self, bindings: &Bindings) -> String {
        match self {
            Self::Ref(t) => format!("&{}", t.rust(bindings)),
            Self::Named(n) => bindings
                .0
                .get(n)
                .map_or_else(|| n.clone(), |b| b.path.clone()),
            Self::Optional(t) => format!("Option<{}>", t.rust(bindings)),
            Self::Sequence(t) => format!("&[{}]", t.rust(bindings)),
            Self::Array(t, n) => format!("[{}; {n}]", t.rust(bindings)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Method {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub result: Type,
    pub is_const: bool,
    pub constant: bool,
}

/// Build only interface declarations: no foreign functions are executed and no impls are emitted.
pub fn declarations(records: &[Decl], source: &str, namespace: &str) -> Result<String, Error> {
    let bindings = Bindings::compile(records, source)?;
    let known = records
        .iter()
        .filter(|r| (matches!(&r.kind, DeclKind::Type { .. }) || matches!(&r.kind, DeclKind::Fields(kind) if matches!(kind.as_str(), "struct" | "enum"))))
        .map(|r| r.name.clone())
        .collect::<BTreeSet<_>>();
    let mut groups = BTreeMap::<String, Vec<Method>>::new();
    let mut names = BTreeSet::new();
    for (owner, record) in crate::syntax::walk(records) {
        let Some(owner) = owner else {
            continue;
        };
        let method = record.name.as_str();
        if !matches!(
            &record.kind,
            DeclKind::Constant(_)
                | DeclKind::Function {
                    body: FunctionBody::Rust { .. },
                    ..
                }
        ) {
            continue;
        }
        let Some(_) = bindings.0.get(owner) else {
            return Err(Error::at(
                source,
                record.offset,
                "method requires a Rust-bound type",
            ));
        };
        if !names.insert((owner, method)) {
            return Err(Error::at(
                source,
                record.offset,
                "duplicate method declaration",
            ));
        }
        if matches!(
            record.body(),
            Some(FunctionBody::Rust { path: Some(_), .. })
        ) {
            return Err(Error::at(
                source,
                record.offset,
                "Rust methods use the type's trait binding; declare a free function for a direct Rust call",
            ));
        }
        let (params, result, is_const, constant) = if let DeclKind::Constant(ty) = &record.kind {
            (Vec::new(), Type::parse(ty, source, &known)?, true, true)
        } else {
            let signature = record.signature().expect("function signature");
            if !signature.generics.is_empty() {
                return Err(Error::at(
                    source,
                    record.offset,
                    "method requires no generics",
                ));
            }
            let Results::Fixed(results) = &signature.results else {
                return Err(Error::at(
                    source,
                    record.offset,
                    "method requires one result type",
                ));
            };
            let [result] = results.as_slice() else {
                return Err(Error::at(
                    source,
                    record.offset,
                    "method requires one result type",
                ));
            };
            let mut seen = BTreeSet::new();
            let params = signature
                .params
                .iter()
                .map(|p| {
                    if p.moves || !seen.insert(&p.name) {
                        return Err(Error::at(
                            source,
                            p.offset,
                            "method parameters must be distinct immutable values",
                        ));
                    }
                    Ok((p.name.clone(), Type::parse(&p.ty, source, &known)?))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let result = Type::parse(&result.ty, source, &known)?;
            (params, result, signature.is_const, false)
        };
        let path = bindings.method_trait(owner, namespace, is_const);
        if let Some(name) = path.strip_prefix(&format!("{namespace}::")) {
            if name.contains("::") {
                return Err(Error::at(
                    source,
                    record.offset,
                    "trait must be directly inside the output namespace",
                ));
            }
            let group = groups.entry(name.into()).or_default();
            if group
                .iter()
                .any(|m| m.is_const != is_const || m.name == method)
            {
                return Err(Error::at(
                    source,
                    record.offset,
                    "conflicting generated trait methods",
                ));
            }
            group.push(Method {
                name: method.into(),
                params,
                result,
                is_const,
                constant,
            });
        }
    }
    let mut out =
        String::from("// @generated interface contracts; implementations belong to Rust.\n");
    for (name, methods) in groups {
        let qualifier = if methods[0].is_const { "const " } else { "" };
        writeln!(out, "pub {qualifier}trait {name}: Sized {{").unwrap();
        for method in methods {
            let params = method
                .params
                .iter()
                .map(|(name, ty)| {
                    if name == "self" {
                        if matches!(ty, Type::Ref(_)) {
                            "&self".into()
                        } else {
                            "self".into()
                        }
                    } else {
                        format!("{name}: {}", ty.rust(&bindings))
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            if method.constant {
                writeln!(
                    out,
                    "const {}: {};",
                    method.name,
                    method.result.rust(&bindings)
                )
                .unwrap();
            } else {
                writeln!(
                    out,
                    "fn {}({params}) -> {};",
                    method.name,
                    method.result.rust(&bindings)
                )
                .unwrap();
            }
        }
        out.push_str("}\n");
    }
    Ok(out)
}

pub fn generate(source: &Source, namespace: &str) -> Result<String, Error> {
    // Signatures must not acquire type declarations through unrelated sibling imports.
    let owners = source
        .files()
        .iter()
        .enumerate()
        .flat_map(|(i, file)| {
            source.declarations()[file.declarations.clone()]
                .iter()
                .filter(|r| (matches!(&r.kind, DeclKind::Type { .. }) || matches!(&r.kind, DeclKind::Fields(kind) if matches!(kind.as_str(), "struct" | "enum"))))
                .map(move |r| (r.name.as_str(), i))
        })
        .collect::<BTreeMap<_, _>>();
    fn check(
        node: &Node,
        source: &Source,
        visible: &BTreeSet<usize>,
        owners: &BTreeMap<&str, usize>,
    ) -> Result<(), Error> {
        match &node.kind {
            Kind::Ref(inner) => check(inner, source, visible, owners),
            Kind::Name(name)
                if owners
                    .get(name.as_str())
                    .is_some_and(|i| !visible.contains(i)) =>
            {
                Err(Error::at(
                    source.text(),
                    node.offset,
                    format!("`{name}` is not imported in this file"),
                ))
            }
            Kind::Call(_, args) => {
                for arg in args {
                    check(arg, source, visible, owners)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
    for file in source.files() {
        for (_, record) in crate::syntax::walk(&source.declarations()[file.declarations.clone()]) {
            if let DeclKind::Constant(ty) = &record.kind {
                check(ty, source, &file.visible, &owners)?;
            }
            if let Some(sig) = record.signature() {
                for p in &sig.params {
                    check(&p.ty, source, &file.visible, &owners)?;
                }
                if let Results::Fixed(results) = &sig.results {
                    for r in results {
                        check(&r.ty, source, &file.visible, &owners)?;
                    }
                }
            }
        }
    }
    declarations(source.declarations(), source.text(), namespace)
}
