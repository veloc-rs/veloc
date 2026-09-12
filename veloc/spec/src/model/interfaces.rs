//! Typed, pure projections. The language knows values, types and data
//! constructors; domain-specific queries and helper functions live in defs.
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use super::{
    Definitions, Param, ParamKind, Pattern, TypeDef, builtins::Builtins, data,
    records::PropertyType,
};
use crate::{
    Error,
    syntax::{Kind, Node, Record, Results},
};

/// Expression types differ from storage types: Value(PTR) is a checked SSA
/// reference, erased to Value only when emitting a runtime query result.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Ty {
    Named(String),
    Value(Option<String>),
    Optional(Box<Ty>),
    Array(Box<Ty>, usize),
}

impl Ty {
    fn named(name: &str) -> Self {
        Self::Named(name.into())
    }
    fn rust(&self) -> String {
        match self {
            Self::Named(name) if name == "Type" => "crate::Type".into(),
            Self::Named(name) => name.clone(),
            Self::Value(_) => "crate::Value".into(),
            Self::Optional(ty) => format!("Option<{}>", ty.rust()),
            Self::Array(ty, n) => format!("[{}; {n}]", ty.rust()),
        }
    }
    fn accepts(&self, actual: &Self) -> bool {
        self == actual
            || match (self, actual) {
                (Self::Value(None), Self::Value(_)) => true,
                (Self::Optional(a), Self::Optional(b)) => a.accepts(b),
                (Self::Array(a, n), Self::Array(b, m)) => n == m && a.accepts(b),
                _ => false,
            }
    }
    fn property(ty: &PropertyType) -> Self {
        match ty {
            PropertyType::Named(name) if name == "Value" => Self::Value(None),
            PropertyType::Named(name) => Self::Named(name.clone()),
            PropertyType::Optional(name) => {
                Self::Optional(Box::new(Self::property(&PropertyType::Named(name.clone()))))
            }
            PropertyType::Values(n) => Self::Array(Box::new(Self::Value(None)), *n),
        }
    }
}

#[derive(Clone)]
pub(crate) struct Expr {
    ty: Ty,
    kind: ExprKind,
}

#[derive(Clone)]
enum ExprKind {
    Constant(data::Value),
    Parameter(usize),
    Operand(String),
    ResultType(usize),
    TypeOf(Box<Expr>),
    Convert(Box<Expr>),
    Field(Box<Expr>, String),
    Record(BTreeMap<String, Expr>),
    Variant(String, Vec<Expr>),
    Some(Box<Expr>),
    Array(Vec<Expr>),
}

impl Expr {
    fn substitute(&self, args: &[Expr]) -> Self {
        if let ExprKind::Parameter(index) = self.kind {
            return args[index].clone();
        }
        let kind = match &self.kind {
            ExprKind::TypeOf(e) => ExprKind::TypeOf(Box::new(e.substitute(args))),
            ExprKind::Convert(e) => ExprKind::Convert(Box::new(e.substitute(args))),
            ExprKind::Some(e) => ExprKind::Some(Box::new(e.substitute(args))),
            ExprKind::Field(e, name) => {
                return Self::field(e.substitute(args), name, self.ty.clone());
            }
            ExprKind::Record(fields) => ExprKind::Record(
                fields
                    .iter()
                    .map(|(n, e)| (n.clone(), e.substitute(args)))
                    .collect(),
            ),
            ExprKind::Variant(name, fields) => ExprKind::Variant(
                name.clone(),
                fields.iter().map(|e| e.substitute(args)).collect(),
            ),
            ExprKind::Array(fields) => {
                ExprKind::Array(fields.iter().map(|e| e.substitute(args)).collect())
            }
            other => other.clone(),
        };
        Self {
            ty: self.ty.clone(),
            kind,
        }
    }

    fn field(base: Self, name: &str, ty: Ty) -> Self {
        if let ExprKind::Record(fields) = &base.kind {
            return fields[name].clone();
        }
        Self {
            ty,
            kind: ExprKind::Field(Box::new(base), name.into()),
        }
    }

    /// Static metadata can use a projection only when the selected expression
    /// contains no operand/result reads. It never runs a runtime query.
    fn constant_node(&self, offset: usize) -> Option<Node> {
        fn constant(value: &data::Value, offset: usize) -> Node {
            let kind = match value {
                data::Value::Number(n) => Kind::Number(*n),
                data::Value::Bool(b) => Kind::Name(b.to_string()),
                data::Value::Flags(_, members) => Kind::List(
                    members
                        .iter()
                        .map(|m| Node {
                            offset,
                            kind: Kind::Name(m.clone()),
                        })
                        .collect(),
                ),
                data::Value::Record(name, fields) => Kind::Object(
                    name.clone(),
                    fields
                        .iter()
                        .map(|(n, v)| (n.clone(), constant(v, offset)))
                        .collect(),
                ),
                data::Value::Variant(_, name, args) => Kind::Call(
                    name.clone(),
                    args.iter().map(|v| constant(v, offset)).collect(),
                ),
                data::Value::None => Kind::Name("none".into()),
                data::Value::Some(v) => Kind::Call("some".into(), vec![constant(v, offset)]),
                data::Value::Empty(_) => Kind::Name("empty".into()),
            };
            Node { offset, kind }
        }
        let kind = match &self.kind {
            ExprKind::Constant(value) => return Some(constant(value, offset)),
            ExprKind::Convert(value) => return value.constant_node(offset),
            ExprKind::Record(fields) => Kind::Object(
                self.ty.rust(),
                fields
                    .iter()
                    .map(|(n, e)| Some((n.clone(), e.constant_node(offset)?)))
                    .collect::<Option<_>>()?,
            ),
            ExprKind::Variant(name, args) => Kind::Call(
                name.clone(),
                args.iter()
                    .map(|e| e.constant_node(offset))
                    .collect::<Option<_>>()?,
            ),
            ExprKind::Some(e) => Kind::Call("some".into(), vec![e.constant_node(offset)?]),
            ExprKind::Array(args) => Kind::List(
                args.iter()
                    .map(|e| e.constant_node(offset))
                    .collect::<Option<_>>()?,
            ),
            _ => return None,
        };
        Some(Node { offset, kind })
    }

    fn rust(&self, locals: &BTreeMap<String, String>) -> String {
        match &self.kind {
            ExprKind::Constant(data::Value::Number(n)) => format!("{n}{}", self.ty.rust()),
            ExprKind::Constant(v) => v.rust(""),
            ExprKind::Parameter(_) => {
                unreachable!("functions are expanded during definition checking")
            }
            ExprKind::Operand(name) => locals[name].clone(),
            ExprKind::ResultType(0) => "dfg.value_type(*results.first()?)".into(),
            ExprKind::ResultType(index) => format!("dfg.value_type(*results.get({index})?)"),
            ExprKind::TypeOf(e) => format!("dfg.value_type({})", e.rust(locals)),
            ExprKind::Convert(e) => format!("{}::from({})", self.ty.rust(), e.rust(locals)),
            ExprKind::Field(e, name) => format!("({}).{name}", e.rust(locals)),
            ExprKind::Record(fields) => format!(
                "{} {{ {} }}",
                self.ty.rust(),
                fields
                    .iter()
                    .map(|(n, e)| format!("{n}: {}", e.rust(locals)))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            ExprKind::Variant(name, args) => {
                let path = format!("{}::{name}", self.ty.rust());
                if args.is_empty() {
                    path
                } else {
                    format!(
                        "{path}({})",
                        args.iter()
                            .map(|e| e.rust(locals))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
            }
            ExprKind::Some(e) => format!("Some({})", e.rust(locals)),
            ExprKind::Array(args) => format!(
                "[{}]",
                args.iter()
                    .map(|e| e.rust(locals))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

struct Interface {
    fields: Vec<(String, Ty)>,
}
#[derive(Clone)]
struct Function {
    params: Vec<Ty>,
    result: Ty,
    body: Expr,
}

#[derive(Default)]
pub(crate) struct Library {
    interfaces: BTreeMap<String, Interface>,
    functions: BTreeMap<String, Function>,
}

struct Checker<'a> {
    source: &'a str,
    library: &'a mut Library,
    data: &'a data::Types,
    builtins: &'a Builtins,
    types: &'a crate::types::Types,
    declarations: &'a [Record],
    active: BTreeSet<String>,
}

impl Library {
    /// Storage structs used as query data still need their ordinary Rust type.
    pub fn data_types(&self) -> BTreeSet<&str> {
        fn visit<'a>(ty: &'a Ty, names: &mut BTreeSet<&'a str>) {
            match ty {
                Ty::Named(name) => {
                    names.insert(name);
                }
                Ty::Optional(ty) | Ty::Array(ty, _) => visit(ty, names),
                Ty::Value(_) => {}
            }
        }
        let mut names = BTreeSet::new();
        for interface in self.interfaces.values() {
            for (_, ty) in &interface.fields {
                visit(ty, &mut names);
            }
        }
        names
    }

    pub fn compile(
        declarations: &[Record],
        source: &str,
        data: &data::Types,
        builtins: &Builtins,
        types: &crate::types::Types,
    ) -> Result<Self, Error> {
        let mut library = Self::default();
        let mut checker = Checker {
            source,
            library: &mut library,
            data,
            builtins,
            types,
            declarations,
            active: BTreeSet::new(),
        };
        for declaration in declarations.iter().filter(|d| d.kind == "interface") {
            if matches!(
                declaration.name.as_str(),
                "Value" | "InstructionQuery" | "InstructionView"
            ) || data.records.iter().any(|r| r.name == declaration.name)
                || data.enums.iter().any(|e| e.name == declaration.name)
                || builtins.flags.contains_key(&declaration.name)
                || builtins.encodings.contains_key(&declaration.name)
            {
                return Err(Error::at(
                    source,
                    declaration.offset,
                    "interface conflicts with a data type",
                ));
            }
            let mut fields = declaration.fields.iter().collect::<Vec<_>>();
            fields.sort_by_key(|(_, node)| node.offset);
            let fields = fields
                .into_iter()
                .map(|(name, node)| {
                    super::identifier(source, node.offset, name)?;
                    Ok((name.clone(), checker.ty(node)?))
                })
                .collect::<Result<_, Error>>()?;
            checker
                .library
                .interfaces
                .insert(declaration.name.clone(), Interface { fields });
        }
        // Inline query data must be finite, just like ordinary struct data.
        for name in checker.library.interfaces.keys() {
            checker.check_cycle(name, &mut BTreeSet::new())?;
        }
        for declaration in declarations.iter().filter(|d| d.kind == "fn") {
            checker.function(&declaration.name, declaration.offset)?;
        }
        Ok(library)
    }

    pub fn bind(
        &mut self,
        source: &str,
        node: Option<Node>,
        params: &[Param],
        signature: &TypeDef,
        vocabulary: super::Vocabulary<'_>,
    ) -> Result<BTreeMap<String, Expr>, Error> {
        let super::Vocabulary {
            types,
            builtins,
            data,
        } = vocabulary;
        let mut bindings = BTreeMap::new();
        let Some(node) = node else {
            return Ok(bindings);
        };
        let mut operand = 0;
        let env = params
            .iter()
            .filter_map(|param| {
                let ty = match &param.kind {
                    ParamKind::Value => {
                        let pattern = signature.operands.patterns().and_then(|p| p.get(operand));
                        operand += 1;
                        Ty::Value(match pattern {
                            Some(Pattern::Exact(name)) => Some(name.clone()),
                            _ => None,
                        })
                    }
                    ParamKind::Property(name) => Ty::named(name),
                    _ => return None,
                };
                Some((
                    param.name.clone(),
                    Expr {
                        ty,
                        kind: ExprKind::Operand(param.name.clone()),
                    },
                ))
            })
            .collect();
        let mut checker = Checker {
            source,
            library: self,
            data,
            builtins,
            types,
            declarations: &[],
            active: BTreeSet::new(),
        };
        for node in super::list(source, node)? {
            let expression = checker.expr(&node, None, &env, Some(signature))?;
            let Ty::Named(name) = &expression.ty else {
                return Err(Error::at(
                    source,
                    node.offset,
                    "implementation must return an interface",
                ));
            };
            if !checker.library.interfaces.contains_key(name) {
                return Err(Error::at(
                    source,
                    node.offset,
                    "implementation must return an interface",
                ));
            }
            if bindings.insert(name.clone(), expression).is_some() {
                return Err(Error::at(
                    source,
                    node.offset,
                    "duplicate interface implementation",
                ));
            }
        }
        Ok(bindings)
    }

    pub fn metadata(
        &mut self,
        source: &str,
        node: &mut Node,
        bindings: &BTreeMap<String, Expr>,
        data: &data::Types,
        builtins: &Builtins,
        types: &crate::types::Types,
    ) -> Result<(), Error> {
        let mut checker = Checker {
            source,
            library: self,
            data,
            builtins,
            types,
            declarations: &[],
            active: BTreeSet::new(),
        };
        fn visit(
            checker: &mut Checker<'_>,
            node: &mut Node,
            env: &BTreeMap<String, Expr>,
        ) -> Result<(), Error> {
            if matches!(&node.kind, Kind::Call(name,_) if name == "field" || checker.library.functions.contains_key(name))
            {
                let expr = checker.expr(node, None, env, None)?;
                *node = expr.constant_node(node.offset).ok_or_else(|| {
                    Error::at(
                        checker.source,
                        node.offset,
                        "metadata projection must be compile-time constant",
                    )
                })?;
            } else {
                match &mut node.kind {
                    Kind::Object(_, fields) => {
                        for node in fields.values_mut() {
                            visit(checker, node, env)?;
                        }
                    }
                    Kind::Call(_, args) | Kind::List(args) => {
                        for node in args {
                            visit(checker, node, env)?;
                        }
                    }
                    _ => {}
                }
            }
            Ok(())
        }
        visit(&mut checker, node, bindings)
    }
}

impl Checker<'_> {
    fn ty(&self, node: &Node) -> Result<Ty, Error> {
        let fail = || Error::at(self.source, node.offset, "unknown projection type");
        match &node.kind {
            Kind::Name(name) if name == "Value" => Ok(Ty::Value(None)),
            Kind::Name(name)
                if matches!(
                    name.as_str(),
                    "Type" | "i64" | "i32" | "u32" | "u64" | "u8" | "bool"
                ) || self.data.records.iter().any(|r| r.name == *name)
                    || self.data.enums.iter().any(|e| e.name == *name)
                    || self.builtins.flags.contains_key(name)
                    || self.builtins.encodings.contains_key(name)
                    || self
                        .declarations
                        .iter()
                        .any(|d| d.kind == "interface" && d.name == *name)
                    || self.library.interfaces.contains_key(name) =>
            {
                Ok(Ty::named(name))
            }
            Kind::Call(name, args) if name == "Value" && args.len() == 1 => {
                let Kind::Name(ty) = &args[0].kind else {
                    return Err(fail());
                };
                if !self.types.exact.contains_key(ty) {
                    return Err(fail());
                }
                Ok(Ty::Value(Some(ty.clone())))
            }
            Kind::Call(name, args) if name == "optional" && args.len() == 1 => {
                Ok(Ty::Optional(Box::new(self.ty(&args[0])?)))
            }
            Kind::Call(name, args) if name == "array" && args.len() == 2 => {
                let Kind::Number(n) = args[1].kind else {
                    return Err(fail());
                };
                if n > 255 {
                    return Err(Error::at(
                        self.source,
                        node.offset,
                        "projection array length exceeds 255",
                    ));
                }
                Ok(Ty::Array(Box::new(self.ty(&args[0])?), n as usize))
            }
            _ => Err(fail()),
        }
    }

    fn fields(&self, name: &str) -> Option<Vec<(String, Ty)>> {
        self.library
            .interfaces
            .get(name)
            .map(|i| i.fields.clone())
            .or_else(|| {
                self.data.records.iter().find(|r| r.name == name).map(|r| {
                    r.fields
                        .iter()
                        .map(|f| (f.name.clone(), Ty::property(&f.ty)))
                        .collect()
                })
            })
    }

    fn check_cycle(&self, name: &str, active: &mut BTreeSet<String>) -> Result<(), Error> {
        if !active.insert(name.into()) {
            return Err(Error::at(self.source, 0, "recursive inline interface type"));
        }
        fn visit(
            checker: &Checker<'_>,
            ty: &Ty,
            active: &mut BTreeSet<String>,
        ) -> Result<(), Error> {
            match ty {
                Ty::Named(name) if checker.library.interfaces.contains_key(name) => {
                    checker.check_cycle(name, active)
                }
                Ty::Optional(ty) | Ty::Array(ty, _) => visit(checker, ty, active),
                _ => Ok(()),
            }
        }
        for (_, ty) in &self.library.interfaces[name].fields {
            visit(self, ty, active)?;
        }
        active.remove(name);
        Ok(())
    }

    fn function(&mut self, name: &str, offset: usize) -> Result<Function, Error> {
        if let Some(function) = self.library.functions.get(name) {
            return Ok(function.clone());
        }
        let declaration = self
            .declarations
            .iter()
            .find(|d| d.kind == "fn" && d.name == name)
            .cloned()
            .ok_or_else(|| {
                Error::at(
                    self.source,
                    offset,
                    format!("unknown projection function `{name}`"),
                )
            })?;
        if !self.active.insert(name.into()) || self.active.len() > 64 {
            return Err(Error::at(
                self.source,
                offset,
                "recursive projection functions are not supported",
            ));
        }
        if matches!(
            name,
            "field" | "type" | "result_type" | "some" | "i64" | "u64" | "u32" | "i32"
        ) {
            return Err(Error::at(
                self.source,
                offset,
                "function conflicts with a projection primitive",
            ));
        }
        let signature = declaration.signature.clone().unwrap();
        let Results::Fixed(results) = signature.results else {
            return Err(Error::at(
                self.source,
                offset,
                "function requires one result type",
            ));
        };
        if !signature.generics.is_empty() || results.len() != 1 {
            return Err(Error::at(
                self.source,
                offset,
                "function requires one result type and no generics",
            ));
        }
        let result = self.ty(&results[0].ty)?;
        let mut params = Vec::new();
        let mut env = BTreeMap::new();
        for param in signature.params {
            super::identifier(self.source, param.offset, &param.name)?;
            if param.moves || param.property {
                return Err(Error::at(
                    self.source,
                    param.offset,
                    "projection parameters are plain immutable values",
                ));
            }
            let ty = self.ty(&param.ty)?;
            let expr = Expr {
                ty: ty.clone(),
                kind: ExprKind::Parameter(params.len()),
            };
            params.push(ty);
            if env.insert(param.name, expr).is_some() {
                return Err(Error::at(
                    self.source,
                    param.offset,
                    "duplicate function parameter",
                ));
            }
        }
        let mut fields = super::Fields::new(self.source, declaration);
        let value = fields.take("value")?;
        fields.finish()?;
        let body = self.expr(&value, Some(&result), &env, None)?;
        let function = Function {
            params,
            result,
            body,
        };
        self.active.remove(name);
        self.library.functions.insert(name.into(), function.clone());
        Ok(function)
    }

    fn expr(
        &mut self,
        node: &Node,
        expected: Option<&Ty>,
        env: &BTreeMap<String, Expr>,
        signature: Option<&TypeDef>,
    ) -> Result<Expr, Error> {
        let fail = |message: &str| Error::at(self.source, node.offset, message);
        let expr = match &node.kind {
            Kind::Name(name) if env.contains_key(name) => env[name].clone(),
            Kind::Object(name, values) => {
                let fields = self
                    .fields(name)
                    .ok_or_else(|| fail("unknown projection struct or interface"))?;
                if values.len() != fields.len()
                    || values.keys().any(|n| !fields.iter().any(|(f, _)| f == n))
                {
                    return Err(fail("projection fields must exactly match the declaration"));
                }
                let mut checked = BTreeMap::new();
                for (name, ty) in fields {
                    let value = values
                        .get(&name)
                        .ok_or_else(|| fail("missing projection field"))?;
                    checked.insert(name, self.expr(value, Some(&ty), env, signature)?);
                }
                Expr {
                    ty: Ty::named(name),
                    kind: ExprKind::Record(checked),
                }
            }
            Kind::Call(name, args) if name == "field" && args.len() == 2 => {
                let base = self.expr(&args[0], None, env, signature)?;
                let (Ty::Named(ty), Kind::Name(field)) = (&base.ty, &args[1].kind) else {
                    return Err(fail("field requires a struct and field name"));
                };
                let fields = self
                    .fields(ty)
                    .ok_or_else(|| fail("field requires a struct or interface"))?;
                let ty = fields
                    .into_iter()
                    .find(|(n, _)| n == field)
                    .ok_or_else(|| fail("unknown projection field"))?
                    .1;
                Expr::field(base, field, ty)
            }
            Kind::Call(name, args) if name == "type" && args.len() == 1 => {
                let value = self.expr(&args[0], Some(&Ty::Value(None)), env, signature)?;
                Expr {
                    ty: Ty::named("Type"),
                    kind: ExprKind::TypeOf(Box::new(value)),
                }
            }
            Kind::Call(name, args) if name == "result_type" && args.len() == 1 => {
                let Kind::Number(index) = args[0].kind else {
                    return Err(fail("result_type requires a constant index"));
                };
                let results = signature
                    .and_then(|s| s.results.patterns())
                    .ok_or_else(|| {
                        fail("result_type requires a fixed operation result signature")
                    })?;
                if index as usize >= results.len() {
                    return Err(fail("result_type index is out of range"));
                }
                Expr {
                    ty: Ty::named("Type"),
                    kind: ExprKind::ResultType(index as usize),
                }
            }
            Kind::Call(name, args)
                if matches!(name.as_str(), "i64" | "u64" | "u32" | "i32") && args.len() == 1 =>
            {
                let value = self.expr(&args[0], None, env, signature)?;
                let Ty::Named(from) = &value.ty else {
                    return Err(fail("integer conversion requires an integer"));
                };
                if from != name
                    && !matches!(
                        (from.as_str(), name.as_str()),
                        ("u8", "u32" | "i32" | "u64" | "i64")
                            | ("u32", "u64" | "i64")
                            | ("i32", "i64")
                    )
                {
                    return Err(fail("projection conversion must be lossless"));
                }
                Expr {
                    ty: Ty::named(name),
                    kind: ExprKind::Convert(Box::new(value)),
                }
            }
            Kind::Call(name, args) if name == "some" && args.len() == 1 => {
                let inner = match expected {
                    Some(Ty::Optional(inner)) => Some(inner.as_ref()),
                    _ => None,
                };
                let value = self.expr(&args[0], inner, env, signature)?;
                Expr {
                    ty: Ty::Optional(Box::new(value.ty.clone())),
                    kind: ExprKind::Some(Box::new(value)),
                }
            }
            Kind::Call(name, args)
                if self.library.functions.contains_key(name)
                    || self
                        .declarations
                        .iter()
                        .any(|d| d.kind == "fn" && d.name == *name) =>
            {
                let function = self.function(name, node.offset)?;
                if function.params.len() != args.len() {
                    return Err(fail("projection function argument count mismatch"));
                }
                let args = args
                    .iter()
                    .zip(&function.params)
                    .map(|(arg, ty)| self.expr(arg, Some(ty), env, signature))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut expr = function.body.substitute(&args);
                expr.ty = function.result;
                expr
            }
            Kind::List(nodes) if matches!(expected, Some(Ty::Array(_, _))) => {
                let Some(Ty::Array(ty, n)) = expected else {
                    unreachable!()
                };
                if nodes.len() != *n {
                    return Err(fail("projection array length mismatch"));
                }
                Expr {
                    ty: expected.unwrap().clone(),
                    kind: ExprKind::Array(
                        nodes
                            .iter()
                            .map(|n| self.expr(n, Some(ty), env, signature))
                            .collect::<Result<_, _>>()?,
                    ),
                }
            }
            _ => {
                let ty = expected
                    .cloned()
                    .or_else(|| match &node.kind {
                        Kind::Number(_) => Some(Ty::named("u32")),
                        Kind::Name(n) if matches!(n.as_str(), "true" | "false") => {
                            Some(Ty::named("bool"))
                        }
                        _ => None,
                    })
                    .ok_or_else(|| fail("unknown projection name or expression"))?;
                if let Ty::Named(name) = &ty
                    && let Some(en) = self.data.enums.iter().find(|e| e.name == *name)
                {
                    let (name, args) = match &node.kind {
                        Kind::Call(name, args) => (name, args.as_slice()),
                        Kind::Name(name) => (name, &[][..]),
                        _ => return Err(fail("expected enum constructor")),
                    };
                    let (_, params) = en
                        .variants
                        .iter()
                        .find(|(n, _)| n == name)
                        .ok_or_else(|| fail("unknown projection enum variant"))?;
                    let params = params.iter().map(Ty::property).collect::<Vec<_>>();
                    if params.len() != args.len() {
                        return Err(fail("enum argument count mismatch"));
                    }
                    let args = args
                        .iter()
                        .zip(&params)
                        .map(|(arg, ty)| self.expr(arg, Some(ty), env, signature))
                        .collect::<Result<_, _>>()?;
                    Expr {
                        ty,
                        kind: ExprKind::Variant(name.clone(), args),
                    }
                } else if matches!(&node.kind,Kind::Name(n) if n=="none")
                    && matches!(ty, Ty::Optional(_))
                {
                    Expr {
                        ty,
                        kind: ExprKind::Constant(data::Value::None),
                    }
                } else {
                    let Ty::Named(name) = &ty else {
                        return Err(fail("expected typed projection expression"));
                    };
                    let value = self.data.value(
                        self.source,
                        &PropertyType::Named(name.clone()),
                        node.clone(),
                        self.builtins,
                    )?;
                    Expr {
                        ty,
                        kind: ExprKind::Constant(value),
                    }
                }
            }
        };
        if expected.is_some_and(|ty| !ty.accepts(&expr.ty)) {
            return Err(fail(&format!(
                "projection type mismatch: expected {:?}, found {:?}",
                expected.unwrap(),
                expr.ty
            )));
        }
        Ok(expr)
    }
}

pub(crate) fn generate(defs: &Definitions, formats: &[usize]) -> String {
    if defs.interfaces.interfaces.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "/// A statically dispatched projection over an instruction's logical fields.\npub trait InstructionQuery: Sized { fn query(view: &InstructionView<'_>, dfg: &crate::dfg::DataFlowGraph, results: &[crate::Value]) -> Option<Self>; }\nimpl InstructionView<'_> { pub fn query<Q: InstructionQuery>(&self, dfg: &crate::dfg::DataFlowGraph, results: &[crate::Value]) -> Option<Q> { Q::query(self, dfg, results) } }\n",
    );
    for (name, interface) in &defs.interfaces.interfaces {
        writeln!(
            out,
            "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct {name} {{"
        )
        .unwrap();
        for (field, ty) in &interface.fields {
            writeln!(out, "pub {field}: {},", ty.rust()).unwrap();
        }
        out.push_str("}\n");
        writeln!(out,"impl InstructionQuery for {name} {{ fn query(view: &InstructionView<'_>, dfg: &crate::dfg::DataFlowGraph, results: &[crate::Value]) -> Option<Self> {{ let _ = (dfg, results); match view.opcode() {{").unwrap();
        for (op, &format) in defs.ops.iter().zip(formats) {
            let Some(expr) = op.interfaces.get(name) else {
                continue;
            };
            let format = &defs.storage.formats[format];
            let fields = format
                .fields
                .iter()
                .enumerate()
                .map(|(i, f)| format!("{}: _f{i}", f.name))
                .collect::<Vec<_>>()
                .join(", ");
            let locals = crate::generate::packing::projections(
                op,
                format,
                "dfg",
                |name| {
                    format!(
                        "*_f{}",
                        format.fields.iter().position(|f| f.name == name).unwrap()
                    )
                },
                |v| format!("{v}?"),
            )
            .into_iter()
            .collect();
            writeln!(out,"crate::Opcode::{} => {{ let InstructionView::{} {{ {fields} }} = view else {{ return None; }}; Some({}) }},",op.name,format.name,expr.rust(&locals)).unwrap();
        }
        out.push_str("_ => None, } } }\n");
    }
    out
}
