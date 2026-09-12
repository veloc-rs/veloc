//! Shared typed, pure expressions for verification, queries and static metadata. The language knows values, types and data
//! constructors; domain-specific queries and helper functions live in defs.
use std::collections::{BTreeMap, BTreeSet};

use super::{Param, ParamKind, Pattern, TypeDef, data, encoding::Encodings, records::PropertyType};
use crate::types::TypeSet;
mod evaluate;

use crate::{
    Error,
    syntax::{Kind, Node, Record, Results},
};

fn possible(
    types: &crate::types::Types,
    pattern: &Pattern,
    signature: Option<&TypeDef>,
) -> Option<TypeSet> {
    match pattern {
        Pattern::Set(set) | Pattern::Bind(_, set) | Pattern::Property(_, set) => Some(set.clone()),
        Pattern::Exact(name) => types.exact.get(name).cloned(),
        Pattern::Same(slot) => {
            let signature = signature?;
            let patterns = match &signature.operands {
                super::TypeList::Fixed(p) | super::TypeList::Variadic(p) => p,
                _ => return None,
            };
            patterns
                .iter()
                .chain(signature.results.patterns().unwrap_or(&[]))
                .find_map(|p| match p {
                    Pattern::Bind(id, set) if id == slot => Some(set.clone()),
                    _ => None,
                })
        }
        _ => None,
    }
}

/// Expression types differ from storage types: Value(PTR) is a checked SSA
/// reference, erased to Value only when emitting a runtime query result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Ty {
    Named(String),
    Value(Option<String>),
    Optional(Box<Ty>),
    Array(Box<Ty>, usize),
    Sequence(Box<Ty>),
}

impl Ty {
    fn borrows(&self) -> bool {
        match self {
            Self::Sequence(_) => true,
            Self::Optional(t) | Self::Array(t, _) => t.borrows(),
            _ => false,
        }
    }
    pub(crate) fn rust(&self, types: &super::records::RustTypes) -> String {
        match self {
            Self::Named(n) => types.qualified(n),
            Self::Value(_) => types.qualified("Value"),
            Self::Optional(t) => format!("Option<{}>", t.rust(types)),
            Self::Array(t, n) => format!("[{}; {n}]", t.rust(types)),
            Self::Sequence(t) => format!("&[{}]", t.rust(types)),
        }
    }
    fn integer(&self) -> bool {
        matches!(self, Self::Named(n) if matches!(n.as_str(), "i128" | "i64" | "i32" | "u64" | "u32" | "u8"))
    }
    fn fits(&self, n: i128) -> bool {
        matches!(self, Self::Named(name) if data::fits_number(name, n))
    }
    fn named(name: &str) -> Self {
        Self::Named(name.into())
    }
    fn name(&self) -> &str {
        let Self::Named(name) = self else {
            unreachable!("checked expression requires a named type")
        };
        name
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
    fn property(ty: &PropertyType, rust: &super::records::RustTypes) -> Self {
        match ty {
            PropertyType::Named(name) if rust.policy(name).references.is_operand() => {
                Self::Value(None)
            }
            PropertyType::Named(name) => Self::Named(name.clone()),
            PropertyType::Optional(name) => Self::Optional(Box::new(Self::property(
                &PropertyType::Named(name.clone()),
                rust,
            ))),
            PropertyType::Values(n) => Self::Array(Box::new(Self::Value(None)), *n),
        }
    }
}

#[derive(Clone)]
pub(crate) struct Expr {
    pub(crate) ty: Ty,
    pub(crate) kind: ExprKind,
    types: Option<crate::types::TypeSet>,
}

#[derive(Clone)]
pub(crate) enum ExprKind {
    Constant(data::Value),
    Integer(i128),
    Unary(&'static str, Box<Expr>),
    Binary(&'static str, Box<Expr>, Box<Expr>),
    Query(Query, Box<Expr>),
    Results,
    Type(String),
    Slice(Box<Expr>, Box<Expr>, bool),
    Matches(Box<Expr>, Box<Expr>),
    All(Box<Expr>, usize, Box<Expr>),
    Bound(usize),
    Parameter(usize),
    Operand(String),
    ResultType(usize),
    Convert(Box<Expr>),
    Field(Box<Expr>, String),
    Record(BTreeMap<String, Expr>),
    Variant(String, Vec<Expr>),
    Some(Box<Expr>),
    Try(Box<Expr>),
    Host(HostMethod, Vec<Expr>),
    Rust(RustCall, Vec<Expr>),
    Array(Vec<Expr>),
}

impl Expr {
    pub(crate) fn is_const(&self) -> bool {
        self.const_safe(false)
    }

    pub(crate) fn const_type_query(&self, params: &[Param]) -> bool {
        self.type_only(params) && self.const_safe(true)
    }

    fn const_safe(&self, types: bool) -> bool {
        use ExprKind as E;
        let safe = |expr: &Self| expr.const_safe(types);
        match &self.kind {
            E::Constant(_) | E::Integer(_) | E::Type(_) | E::Parameter(_) | E::Bound(_) => true,
            E::Results | E::ResultType(_) | E::Query(Query::TypeOf, _) => types,
            E::Rust(binding, args) => binding.is_const && args.iter().all(safe),
            E::Unary(_, v)
            | E::Convert(v)
            | E::Some(v)
            | E::Try(v)
            | E::Field(v, _)
            | E::Query(Query::Len, v) => safe(v),
            E::Binary(_, a, b) | E::All(a, _, b) => safe(a) && safe(b),
            E::Record(fields) => fields.values().all(safe),
            E::Variant(_, args) | E::Array(args) => args.iter().all(safe),
            _ => false,
        }
    }

    pub(crate) fn literal(
        ty: &PropertyType,
        value: data::Value,
        rust: &super::records::RustTypes,
    ) -> Self {
        Self::new(Ty::property(ty, rust), ExprKind::Constant(value))
    }

    pub(crate) fn const_rust(&self, prefix: &str) -> String {
        let mut emitter = Emitter::query(BTreeMap::new());
        emitter.constant = true;
        emitter.prefix = prefix;
        emitter.term(self)
    }

    /// Helpers have already been expanded, so nested calls need no separate
    /// dependency registry. The emitted trait calls enforce the host contract.
    pub(crate) fn uses_host(&self) -> bool {
        match &self.kind {
            ExprKind::Host(..) => true,
            ExprKind::Unary(_, v)
            | ExprKind::Query(_, v)
            | ExprKind::Convert(v)
            | ExprKind::Some(v)
            | ExprKind::Try(v)
            | ExprKind::Field(v, _) => v.uses_host(),
            ExprKind::Binary(_, a, b)
            | ExprKind::Matches(a, b)
            | ExprKind::Slice(a, b, _)
            | ExprKind::All(a, _, b) => a.uses_host() || b.uses_host(),
            ExprKind::Record(fields) => fields.values().any(Self::uses_host),
            ExprKind::Rust(_, values) | ExprKind::Array(values) | ExprKind::Variant(_, values) => {
                values.iter().any(Self::uses_host)
            }
            ExprKind::Constant(_)
            | ExprKind::Integer(_)
            | ExprKind::Results
            | ExprKind::Type(_)
            | ExprKind::Bound(_)
            | ExprKind::Parameter(_)
            | ExprKind::Operand(_)
            | ExprKind::ResultType(_) => false,
        }
    }
    fn new(ty: Ty, kind: ExprKind) -> Self {
        Self {
            ty,
            kind,
            types: None,
        }
    }
    fn boolean(value: bool) -> Self {
        Self::new(
            Ty::named("bool"),
            ExprKind::Constant(data::Value::Bool(value)),
        )
    }
    pub fn is_bool(&self, value: bool) -> bool {
        matches!(self.kind, ExprKind::Constant(data::Value::Bool(v)) if v == value)
    }
    fn binary(ty: Ty, op: &'static str, lhs: Self, rhs: Self) -> Self {
        // These identities preserve short circuiting and fallible RHS queries.
        if (op == "&&" && lhs.is_bool(false)) || (op == "||" && lhs.is_bool(true)) {
            return lhs;
        }
        if (op == "&&" && lhs.is_bool(true)) || (op == "||" && lhs.is_bool(false)) {
            return rhs;
        }
        if (op == "&&" && rhs.is_bool(true)) || (op == "||" && rhs.is_bool(false)) {
            return lhs;
        }
        Self::new(ty, ExprKind::Binary(op, Box::new(lhs), Box::new(rhs)))
    }

    fn wide(self) -> Self {
        if self.ty.integer() && self.ty != Ty::named("i128") {
            Self::new(Ty::named("i128"), ExprKind::Convert(Box::new(self)))
        } else {
            self
        }
    }
    fn substitute(&self, args: &[Expr], next: &mut usize) -> Self {
        self.expand(args, &mut BTreeMap::new(), next)
    }

    fn expand(&self, args: &[Expr], locals: &mut BTreeMap<usize, usize>, next: &mut usize) -> Self {
        if let ExprKind::Parameter(index) = self.kind {
            return args[index].clone();
        }
        let kind = match &self.kind {
            ExprKind::Unary(op, e) => ExprKind::Unary(op, Box::new(e.expand(args, locals, next))),
            ExprKind::Binary(op, a, b) => {
                return Self::binary(
                    self.ty.clone(),
                    op,
                    a.expand(args, locals, next),
                    b.expand(args, locals, next),
                );
            }
            ExprKind::Query(q, e) => {
                let value = e.expand(args, locals, next);
                return Self {
                    types: if matches!(q, Query::TypeOf) {
                        value.types.clone()
                    } else {
                        None
                    },
                    ty: self.ty.clone(),
                    kind: ExprKind::Query(*q, Box::new(value)),
                };
            }
            ExprKind::Slice(a, b, prefix) => ExprKind::Slice(
                Box::new(a.expand(args, locals, next)),
                Box::new(b.expand(args, locals, next)),
                *prefix,
            ),
            ExprKind::Matches(a, b) => ExprKind::Matches(
                Box::new(a.expand(args, locals, next)),
                Box::new(b.expand(args, locals, next)),
            ),
            ExprKind::All(a, id, body) => {
                let sequence = a.expand(args, locals, next);
                let fresh = *next;
                *next += 1;
                let old = locals.insert(*id, fresh);
                let body = body.expand(args, locals, next);
                if let Some(old) = old {
                    locals.insert(*id, old);
                } else {
                    locals.remove(id);
                }
                ExprKind::All(Box::new(sequence), fresh, Box::new(body))
            }
            ExprKind::Bound(id) => ExprKind::Bound(*locals.get(id).unwrap_or(id)),
            ExprKind::Try(e) => ExprKind::Try(Box::new(e.expand(args, locals, next))),
            ExprKind::Rust(binding, values) => {
                let args = values
                    .iter()
                    .map(|v| v.expand(args, locals, next))
                    .collect::<Vec<_>>();
                ExprKind::Rust(binding.clone(), args)
            }
            ExprKind::Host(method, values) => ExprKind::Host(
                method.clone(),
                values
                    .iter()
                    .map(|e| e.expand(args, locals, next))
                    .collect(),
            ),
            ExprKind::Convert(e) => ExprKind::Convert(Box::new(e.expand(args, locals, next))),
            ExprKind::Some(e) => ExprKind::Some(Box::new(e.expand(args, locals, next))),
            ExprKind::Field(e, name) => {
                return Self::field(e.expand(args, locals, next), name, self.ty.clone());
            }
            ExprKind::Record(fields) => ExprKind::Record(
                fields
                    .iter()
                    .map(|(n, e)| (n.clone(), e.expand(args, locals, next)))
                    .collect(),
            ),
            ExprKind::Variant(name, fields) => ExprKind::Variant(
                name.clone(),
                fields
                    .iter()
                    .map(|e| e.expand(args, locals, next))
                    .collect(),
            ),
            ExprKind::Array(fields) => ExprKind::Array(
                fields
                    .iter()
                    .map(|e| e.expand(args, locals, next))
                    .collect(),
            ),
            other => other.clone(),
        };
        Self {
            types: self.types.clone(),
            ty: self.ty.clone(),
            kind,
        }
    }

    fn field(base: Self, name: &str, ty: Ty) -> Self {
        if let ExprKind::Record(fields) = &base.kind {
            return fields[name].clone();
        }
        Self {
            types: None,
            ty,
            kind: ExprKind::Field(Box::new(base), name.into()),
        }
    }
}

#[derive(Clone)]
pub(crate) struct Interface {
    pub(crate) fields: Vec<(String, Ty)>,
}
#[derive(Clone)]
pub(crate) struct RustCall {
    pub constant: bool,
    is_const: bool,
    path: String,
    method: Option<(String, String)>,
}

#[derive(Clone)]
struct Function {
    constant: bool,
    params: Vec<Ty>,
    result: Ty,
    body: Expr,
}

#[derive(Clone)]
pub(crate) struct HostMethod {
    interface: String,
    name: String,
    params: Vec<(String, Ty)>,
    result: Ty,
}

#[derive(Clone, Default)]
pub(crate) struct Library {
    pub(crate) interfaces: BTreeMap<String, Interface>,
    functions: BTreeMap<String, Function>,
    hosts: BTreeMap<String, HostMethod>,
    methods: String,
    bindings: String,
}

struct Checker<'a> {
    source: &'a str,
    library: &'a mut Library,
    data: &'a data::Types,
    encodings: &'a Encodings,
    types: &'a crate::types::Types,
    comparisons: &'a [super::comparisons::Comparison],
    declarations: &'a [Record],
    active: BTreeSet<String>,
    // Verification arithmetic widens integers; typed helper arguments do not.
    verification: bool,
    next_local: usize,
}

fn operands(
    params: &[Param],
    signature: Option<&TypeDef>,
    types: &crate::types::Types,
) -> BTreeMap<String, Expr> {
    let mut index = 0;
    let mut env = BTreeMap::new();
    for param in params {
        let ty = match &param.kind {
            ParamKind::Value => Ty::Value(None),
            ParamKind::Values => Ty::Sequence(Box::new(Ty::Value(None))),
            ParamKind::Property(name) if name == "Bytes" => {
                Ty::Sequence(Box::new(Ty::named("i128")))
            }
            ParamKind::Property(name) => Ty::named(name),
            _ => continue,
        };
        let mut expr = Expr::new(ty, ExprKind::Operand(param.name.clone()));
        if param.kind == ParamKind::Value {
            if let Some(pattern) = signature
                .and_then(|s| s.operands.patterns())
                .and_then(|p| p.get(index))
            {
                expr.types = possible(types, pattern, signature);
                if let Pattern::Exact(name) = pattern {
                    expr.ty = Ty::Value(Some(name.clone()));
                }
            }
            index += 1;
        }
        env.insert(param.name.clone(), expr);
    }

    env
}

impl Library {
    pub(crate) fn method_code(&self) -> String {
        format!(
            "pub mod type_methods {{\n{}\n}}\n{}",
            self.methods, self.bindings
        )
    }

    pub(crate) fn host_code(&self, types: &super::records::RustTypes) -> String {
        use std::fmt::Write;
        let mut groups = BTreeMap::<&str, Vec<&HostMethod>>::new();
        for method in self.hosts.values() {
            groups.entry(&method.interface).or_default().push(method);
        }
        let mut out = String::new();
        for (name, methods) in groups {
            writeln!(out,"/// Read-only deterministic queries. Implementations must honor this trusted contract.\npub trait {name} {{").unwrap();
            for method in methods {
                let params = method
                    .params
                    .iter()
                    .map(|(name, ty)| format!(", {name}: {}", ty.rust(types)))
                    .collect::<String>();
                writeln!(
                    out,
                    "fn {}(&self{params}) -> {};",
                    method.name,
                    method.result.rust(types)
                )
                .unwrap();
            }
            out.push_str("}\n");
        }
        out
    }
    #[allow(clippy::too_many_arguments)]
    pub fn verify(
        &mut self,
        source: &str,
        nodes: Vec<Node>,
        params: &[Param],
        signature: Option<&TypeDef>,
        results: &BTreeMap<String, super::Slot>,
        vocabulary: super::Vocabulary<'_>,
    ) -> Result<Vec<super::constraints::Constraint>, Error> {
        let super::Vocabulary {
            types,
            encodings,
            data,
            comparisons,
        } = vocabulary;
        let mut env = operands(params, signature, types);
        for (name, slot) in results {
            let mut value = Expr::new(Ty::named("Type"), ExprKind::ResultType(slot.index as usize));
            if !slot.result {
                let operand = params
                    .iter()
                    .filter(|p| p.kind == ParamKind::Value)
                    .nth(slot.index as usize)
                    .expect("generic operand binding");
                let input = env[&operand.name].clone();
                value.kind = ExprKind::Query(Query::TypeOf, Box::new(input));
            }
            value.types = signature
                .and_then(|s| {
                    if slot.result {
                        s.results.patterns()
                    } else {
                        s.operands.patterns()
                    }
                })
                .and_then(|p| p.get(slot.index as usize))
                .and_then(|p| possible(types, p, signature));
            env.insert(name.clone(), value);
        }
        let mut checker = Checker {
            source,
            library: self,
            data,
            encodings,
            types,
            comparisons,
            declarations: &[],
            active: BTreeSet::new(),
            verification: true,
            next_local: 0,
        };
        let mut result = Vec::new();
        for node in nodes {
            let (condition, text) = if let Kind::Call(name, args) = &node.kind
                && name == "require"
            {
                let [
                    condition,
                    Node {
                        kind: Kind::Text(message),
                        ..
                    },
                ] = args.as_slice()
                else {
                    return Err(Error::at(
                        source,
                        node.offset,
                        "require expects a predicate and a diagnostic string",
                    ));
                };
                (condition, message.clone())
            } else {
                (&node, super::constraints::describe(&node))
            };
            let mut condition =
                checker.expr(condition, Some(&Ty::named("bool")), &env, signature)?;
            if let Some(value) = condition.literal_bool(checker.types) {
                condition = Expr::boolean(value);
            }
            if condition.is_bool(false) {
                return Err(Error::at(
                    source,
                    node.offset,
                    format!("constraint is always false: {text}"),
                ));
            }
            result.push(super::constraints::Constraint {
                type_only: condition.type_only(params),
                condition,
                text,
            });
        }
        Ok(result)
    }
    /// Storage structs used as query data still need their ordinary Rust type.
    pub fn data_types(&self) -> BTreeSet<&str> {
        fn visit<'a>(ty: &'a Ty, names: &mut BTreeSet<&'a str>) {
            match ty {
                Ty::Named(name) => {
                    names.insert(name);
                }
                Ty::Optional(ty) | Ty::Array(ty, _) | Ty::Sequence(ty) => visit(ty, names),
                Ty::Value(_) => {}
            }
        }
        let mut names = BTreeSet::new();
        for function in self.functions.values() {
            for ty in function.params.iter().chain([&function.result]) {
                visit(ty, &mut names);
            }
        }
        for method in self.hosts.values() {
            for (_, ty) in &method.params {
                visit(ty, &mut names);
            }
            visit(&method.result, &mut names);
        }
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
        vocabulary: super::Vocabulary<'_>,
    ) -> Result<Self, Error> {
        let super::Vocabulary {
            data,
            encodings,
            types,
            comparisons,
        } = vocabulary;
        let mut library = Self {
            methods: crate::interfaces::declarations(declarations, source, "crate::type_methods")?,
            ..Self::default()
        };
        let mut checker = Checker {
            source,
            library: &mut library,
            data,
            encodings,
            types,
            declarations,
            comparisons,
            active: BTreeSet::new(),
            verification: false,
            next_local: 0,
        };
        for declaration in declarations.iter().filter(|d| d.kind == "interface") {
            if matches!(
                declaration.name.as_str(),
                "Value" | "InstructionQuery" | "InstView"
            ) || data.rust.contains(&declaration.name)
                || data.records.iter().any(|r| r.name == declaration.name)
                || data.enums.iter().any(|e| e.name == declaration.name)
                || encodings.contains_key(&declaration.name)
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
                    let ty = checker.ty(node)?;
                    if ty.borrows() {
                        return Err(Error::at(source,node.offset,"interface fields must be owned; consume borrowed host sequences inside a helper"));
                    }
                    Ok((name.clone(), ty))
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
        for owner in declarations.iter().filter(|d| d.kind == "extern-interface") {
            if !declarations.iter().any(|d| {
                d.kind == "extern-fn"
                    && d.name
                        .split_once("::")
                        .is_some_and(|(name, _)| name == owner.name)
            }) {
                return Err(Error::at(
                    source,
                    owner.offset,
                    "host interface needs at least one method",
                ));
            }
        }
        for declaration in declarations.iter().filter(|d| d.kind == "extern-fn") {
            let (interface, name) = declaration
                .name
                .split_once("::")
                .expect("parsed external method");
            let signature = declaration.signature.as_ref().unwrap();
            let Results::Fixed(results) = &signature.results else {
                return Err(Error::at(
                    source,
                    declaration.offset,
                    "host method requires one result type",
                ));
            };
            if !signature.generics.is_empty() || results.len() != 1 {
                return Err(Error::at(
                    source,
                    declaration.offset,
                    "host method requires one result type and no generics",
                ));
            }
            let mut names = BTreeSet::new();
            let params = signature
                .params
                .iter()
                .map(|p| {
                    super::identifier(source, p.offset, &p.name)?;
                    if p.moves || !names.insert(&p.name) {
                        return Err(Error::at(
                            source,
                            p.offset,
                            "host parameters must be distinct immutable values",
                        ));
                    }
                    Ok((p.name.clone(), checker.ty(&p.ty)?))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let method = HostMethod {
                interface: interface.into(),
                name: name.into(),
                params,
                result: checker.ty(&results[0].ty)?,
            };
            checker
                .library
                .hosts
                .insert(declaration.name.clone(), method);
        }
        for declaration in declarations
            .iter()
            .filter(|d| matches!(d.kind.as_str(), "fn" | "const"))
        {
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
            encodings,
            data,
            comparisons,
        } = vocabulary;
        let mut bindings = BTreeMap::new();
        let Some(node) = node else {
            return Ok(bindings);
        };
        let env = operands(params, Some(signature), types);
        let mut checker = Checker {
            source,
            library: self,
            data,
            encodings,
            types,
            declarations: &[],
            comparisons,
            active: BTreeSet::new(),
            verification: false,
            next_local: 0,
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
        vocabulary: super::Vocabulary<'_>,
    ) -> Result<(), Error> {
        let super::Vocabulary {
            data,
            encodings,
            types,
            comparisons,
        } = vocabulary;
        let mut checker = Checker {
            source,
            library: self,
            data,
            encodings,
            types,
            declarations: &[],
            comparisons,
            active: BTreeSet::new(),
            verification: false,
            next_local: 0,
        };
        fn visit(
            checker: &mut Checker<'_>,
            node: &mut Node,
            env: &BTreeMap<String, Expr>,
        ) -> Result<(), Error> {
            let method = matches!(
                &node.kind,
                Kind::Method(..)
                    | Kind::Member(..)
                    | Kind::Try(_)
                    | Kind::Unary(..)
                    | Kind::Binary(..)
            );
            if method
                || matches!(&node.kind, Kind::Call(name,_) if name == "field" || checker.library.functions.contains_key(name))
            {
                let expr = checker.expr(node, None, env, None)?;
                if let Some(value) = expr.constant_node(checker.types, node.offset) {
                    *node = value;
                } else if !expr.is_const() {
                    return Err(Error::at(
                        checker.source,
                        node.offset,
                        "metadata projection must be compile-time constant",
                    ));
                }
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

    pub fn metadata_field(
        &mut self,
        source: &str,
        node: &Node,
        ty: &PropertyType,
        env: &BTreeMap<String, Expr>,
        vocabulary: super::Vocabulary<'_>,
    ) -> Result<Expr, Error> {
        let super::Vocabulary {
            data,
            encodings,
            types,
            comparisons,
        } = vocabulary;
        let mut checker = Checker {
            source,
            library: self,
            data,
            encodings,
            types,
            declarations: &[],
            comparisons,
            active: BTreeSet::new(),
            verification: false,
            next_local: 0,
        };
        let expr = checker.expr(node, Some(&Ty::property(ty, &data.rust)), env, None)?;
        if !expr.is_const() {
            return Err(Error::at(
                source,
                node.offset,
                "metadata projection must be compile-time constant",
            ));
        }
        if let Some(value) = expr.constant_node(types, node.offset) {
            return Ok(Expr::literal(
                ty,
                data.value(source, ty, value, encodings)?,
                &data.rust,
            ));
        }
        Ok(expr)
    }
}

impl Checker<'_> {
    fn binary(
        &mut self,
        op: &'static str,
        lhs: &Node,
        rhs: &Node,
        expected: Option<&Ty>,
        env: &BTreeMap<String, Expr>,
        signature: Option<&TypeDef>,
    ) -> Result<Expr, Error> {
        let hint = if self.verification {
            None
        } else {
            expected.filter(|ty| ty.integer()).cloned().or_else(|| {
                if matches!(lhs.kind, Kind::Number(_) | Kind::Integer(_)) {
                    self.expr(rhs, None, env, signature)
                        .ok()
                        .map(|e| e.ty)
                        .filter(Ty::integer)
                } else {
                    None
                }
            })
        };
        let mut a = self.expr(lhs, hint.as_ref(), env, signature)?;
        if matches!(a.ty, Ty::Optional(_)) {
            return Err(Error::at(
                self.source,
                lhs.offset,
                "optional operands must be unwrapped with ?",
            ));
        }
        if self.verification {
            a = a.wide();
        }
        let numeric = a.ty.integer();
        let mut b = self.expr(
            rhs,
            if self.verification && numeric {
                None
            } else {
                Some(&a.ty)
            },
            env,
            signature,
        )?;
        if self.verification {
            b = b.wide();
        }
        if a.ty != b.ty {
            return Err(Error::at(
                self.source,
                rhs.offset,
                "binary expression operand types differ",
            ));
        }
        let boolean = a.ty == Ty::named("bool");
        let ty = match op {
            "&&" | "||" if boolean => Ty::named("bool"),
            "+" | "-" | "*" | "&" | "|" if numeric => a.ty.clone(),
            "<" | "<=" | ">" | ">=" if numeric => Ty::named("bool"),
            "==" | "!="
                if numeric
                    || boolean
                    || matches!(&a.ty, Ty::Named(n) if n=="Type" || n=="Shape" || n=="TypeBits" || self.comparisons.iter().any(|c|c.name==*n) || self.data.enums.iter().any(|e|e.name==*n))
                    || matches!(&a.ty, Ty::Sequence(_)) =>
            {
                Ty::named("bool")
            }
            _ => {
                return Err(Error::at(
                    self.source,
                    lhs.offset,
                    "unsupported expression operator or operand type",
                ));
            }
        };
        let numeric_constants = numeric
            && a.constant_node(self.types, lhs.offset).is_some()
            && b.constant_node(self.types, rhs.offset).is_some();
        let expr = Expr::binary(ty, op, a, b);
        if numeric_constants
            && matches!(op, "+" | "-" | "*" | "&" | "|")
            && expr.constant_node(self.types, lhs.offset).is_none()
        {
            return Err(Error::at(
                self.source,
                lhs.offset,
                "expression arithmetic overflow",
            ));
        }
        if let Some(node) = expr.constant_node(self.types, lhs.offset) {
            match node.kind {
                Kind::Name(b) if b == "true" || b == "false" => {
                    return Ok(Expr::boolean(b == "true"));
                }
                Kind::Number(n) => return Ok(Expr::new(expr.ty, ExprKind::Integer(i128::from(n)))),
                Kind::Integer(n) => return Ok(Expr::new(expr.ty, ExprKind::Integer(n))),
                _ => {}
            }
        }
        Ok(expr)
    }

    fn associated(&mut self, offset: usize, owner: &str, field: &str) -> Result<Expr, Error> {
        if let Some(comparison) = self.comparisons.iter().find(|c| c.name == owner) {
            if !comparison.has_variant(field) {
                return Err(Error::at(self.source, offset, "unknown comparison variant"));
            }
            return Ok(Expr::new(
                Ty::named(owner),
                ExprKind::Variant(field.into(), Vec::new()),
            ));
        }
        let name = format!("{owner}::{field}");
        let function = self.function(&name, offset)?;
        if !function.constant {
            return Err(Error::at(
                self.source,
                offset,
                "associated member requires a constant; use () to call a function",
            ));
        }
        Ok(function.body)
    }

    fn member(
        &mut self,
        offset: usize,
        receiver: &Node,
        field: &str,
        env: &BTreeMap<String, Expr>,
        signature: Option<&TypeDef>,
    ) -> Result<Expr, Error> {
        let value = self.expr(receiver, None, env, signature)?;
        let Ty::Named(name) = &value.ty else {
            return Err(Error::at(
                self.source,
                offset,
                "field access requires a struct",
            ));
        };
        let fields = self
            .fields(name)
            .ok_or_else(|| Error::at(self.source, offset, "field access requires a struct"))?;
        let ty = fields
            .into_iter()
            .find(|(n, _)| n == field)
            .ok_or_else(|| Error::at(self.source, offset, format!("unknown field {name}.{field}")))?
            .1;
        Ok(Expr::field(value, field, ty))
    }

    fn query(
        &mut self,
        offset: usize,
        name: &str,
        args: &[Node],
        env: &BTreeMap<String, Expr>,
        signature: Option<&TypeDef>,
    ) -> Result<Expr, Error> {
        let fail = |msg: &str| Error::at(self.source, offset, msg);
        if name == "results" {
            if !args.is_empty() {
                return Err(fail("results takes no arguments"));
            }
            if signature.is_none() {
                return Err(fail("results requires an instruction context"));
            }
            return Ok(Expr::new(
                Ty::Sequence(Box::new(Ty::named("Type"))),
                ExprKind::Results,
            ));
        }
        if name == "all" {
            let [
                sequence,
                Node {
                    kind: Kind::Lambda(name, body),
                    ..
                },
            ] = args
            else {
                return Err(fail("all expects a sequence and |name| predicate"));
            };
            let sequence = self.expr(sequence, None, env, signature)?;
            let (Ty::Sequence(element) | Ty::Optional(element) | Ty::Array(element, _)) =
                &sequence.ty
            else {
                return Err(fail("all expects a finite sequence or optional value"));
            };
            let id = self.next_local;
            self.next_local += 1;
            let mut locals = env.clone();
            locals.insert(
                name.clone(),
                Expr::new(*element.clone(), ExprKind::Bound(id)),
            );
            let body = self.expr(body, Some(&Ty::named("bool")), &locals, signature)?;
            return Ok(Expr::new(
                Ty::named("bool"),
                ExprKind::All(Box::new(sequence), id, Box::new(body)),
            ));
        }
        if matches!(name, "prefix" | "suffix" | "matches") {
            let [lhs, rhs] = args else {
                return Err(fail("sequence operation expects two arguments"));
            };
            let a = self.expr(lhs, None, env, signature)?;
            let b = self.expr(rhs, None, env, signature)?;
            if name == "matches" {
                if a.ty != Ty::Sequence(Box::new(Ty::Value(None)))
                    || b.ty != Ty::Sequence(Box::new(Ty::named("Type")))
                {
                    return Err(fail("matches expects value and type sequences"));
                }
                return Ok(Expr::new(
                    Ty::named("bool"),
                    ExprKind::Matches(Box::new(a), Box::new(b)),
                ));
            }
            if !matches!(a.ty, Ty::Sequence(_)) || !b.ty.integer() {
                return Err(fail("slice expects a sequence and integer"));
            }
            return Ok(Expr::new(
                a.ty.clone(),
                ExprKind::Slice(Box::new(a), Box::new(b.wide()), name == "prefix"),
            ));
        }
        let [arg] = args else {
            return Err(fail("len expects one argument"));
        };
        let value = self.expr(arg, None, env, signature)?;
        if !matches!(value.ty, Ty::Sequence(_) | Ty::Array(_, _)) {
            return Err(fail("len expects a sequence"));
        }
        Ok(Expr::new(
            Ty::named("i128"),
            ExprKind::Query(Query::Len, Box::new(value)),
        ))
    }

    fn ty(&self, node: &Node) -> Result<Ty, Error> {
        let fail = || Error::at(self.source, node.offset, "unknown projection type");
        match &node.kind {
            Kind::Name(name) if name == "Value" && self.data.rust.contains(name) => {
                Ok(Ty::Value(None))
            }
            Kind::Name(name)
                if super::records::primitive(name)
                    || self.data.rust.contains(name)
                    || self.data.records.iter().any(|r| r.name == *name)
                    || self.data.enums.iter().any(|e| e.name == *name)
                    || self.comparisons.iter().any(|c| c.name == *name)
                    || self.encodings.contains_key(name)
                    || self
                        .declarations
                        .iter()
                        .any(|d| d.kind == "interface" && d.name == *name)
                    || self.library.interfaces.contains_key(name) =>
            {
                Ok(Ty::named(name))
            }
            Kind::Call(name, args)
                if name == "Value" && args.len() == 1 && self.data.rust.contains(name) =>
            {
                let ty = self.types.exact_name(&args[0]).ok_or_else(fail)?;
                Ok(Ty::Value(Some(ty)))
            }
            Kind::Call(name, args) if name == "sequence" && args.len() == 1 => {
                Ok(Ty::Sequence(Box::new(self.ty(&args[0])?)))
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
                        .map(|f| (f.name.clone(), Ty::property(&f.ty, &self.data.rust)))
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
            .find(|d| matches!(d.kind.as_str(), "fn" | "const") && d.name == name)
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
            "field"
                | "type"
                | "result_type"
                | "some"
                | "i128"
                | "i64"
                | "u64"
                | "u32"
                | "i32"
                | "all"
                | "prefix"
                | "suffix"
                | "matches"
                | "results"
                | "require"
        ) || name == "len"
        {
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
        let owner = name.split_once("::").map(|(owner, _)| owner);
        fn resolve_self(node: &Node, owner: Option<&str>) -> Node {
            let mut node = node.clone();
            match &mut node.kind {
                Kind::Name(name) if name == "Self" => {
                    if let Some(owner) = owner {
                        *name = owner.into();
                    }
                }
                Kind::Call(_, args) => {
                    for arg in args {
                        *arg = resolve_self(arg, owner);
                    }
                }
                _ => {}
            }
            node
        }
        let result = self.ty(&resolve_self(&results[0].ty, owner))?;
        let mut params = Vec::new();
        let mut env = BTreeMap::new();
        for param in signature.params {
            if param.name != "self" || owner.is_none() {
                super::identifier(self.source, param.offset, &param.name)?;
            }
            if param.moves {
                return Err(Error::at(
                    self.source,
                    param.offset,
                    "projection parameters are plain immutable values",
                ));
            }
            let ty = self.ty(&resolve_self(&param.ty, owner))?;
            let expr = Expr {
                types: None,
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
        let body = match declaration.body.expect("function has a body") {
            crate::syntax::FunctionBody::Rust { offset, path } => {
                let path = if let Some(path) = path {
                    path
                } else {
                    let owner = owner.expect("implicit Rust binding is a method");
                    if !self.data.rust.contains(owner) {
                        return Err(Error::at(
                            self.source,
                            offset,
                            "method requires a Rust-bound type",
                        ));
                    }
                    format!(
                        "{}::{}",
                        self.data.rust.rust(owner),
                        name.rsplit("::").next().unwrap()
                    )
                };
                super::records::rust_path(self.source, offset, &path)?;
                if owner.is_none() {
                    use std::fmt::Write;
                    writeln!(
                        self.library.bindings,
                        "const _: fn({}) -> {} = {path};",
                        params
                            .iter()
                            .map(|t| t.rust(&self.data.rust))
                            .collect::<Vec<_>>()
                            .join(", "),
                        result.rust(&self.data.rust)
                    )
                    .unwrap();
                }
                let args = params
                    .iter()
                    .enumerate()
                    .map(|(i, ty)| Expr::new(ty.clone(), ExprKind::Parameter(i)))
                    .collect();
                Expr::new(
                    result.clone(),
                    ExprKind::Rust(
                        RustCall {
                            constant: declaration.kind == "const",
                            is_const: signature.is_const,
                            path,
                            method: owner.map(|owner| {
                                (
                                    format!(
                                        "<{} as {}>",
                                        self.data.rust.rust(owner),
                                        self.data.rust.method_trait(owner, signature.is_const)
                                    ),
                                    name.rsplit("::").next().unwrap().into(),
                                )
                            }),
                        },
                        args,
                    ),
                )
            }
            crate::syntax::FunctionBody::Value(value) => {
                self.expr(&value, Some(&result), &env, None)?
            }
        };
        if signature.is_const && !body.is_const() {
            return Err(Error::at(
                self.source,
                declaration.offset,
                "const function calls a runtime-only operation",
            ));
        }
        let function = Function {
            constant: declaration.kind == "const",
            params,
            result,
            body,
        };
        self.active.remove(name);
        self.library.functions.insert(name.into(), function.clone());
        Ok(function)
    }

    fn method(
        &mut self,
        offset: usize,
        receiver: &Node,
        method: &str,
        args: &[Node],
        env: &BTreeMap<String, Expr>,
        signature: Option<&TypeDef>,
    ) -> Result<Expr, Error> {
        let value = self.expr(receiver, None, env, signature)?;
        let owner = match &value.ty {
            Ty::Named(name) => name.as_str(),
            Ty::Value(_) => "Value",
            _ => {
                return Err(Error::at(
                    self.source,
                    offset,
                    "method requires a named receiver type",
                ));
            }
        };
        let function = self.function(&format!("{owner}::{method}"), offset)?;
        self.call(function, vec![value], args, offset, env, signature)
    }

    fn call(
        &mut self,
        function: Function,
        mut checked: Vec<Expr>,
        args: &[Node],
        offset: usize,
        env: &BTreeMap<String, Expr>,
        signature: Option<&TypeDef>,
    ) -> Result<Expr, Error> {
        if function.constant {
            return Err(Error::at(
                self.source,
                offset,
                "associated constants are values, not functions",
            ));
        }
        if function.params.len() != checked.len() + args.len() {
            return Err(Error::at(
                self.source,
                offset,
                "projection function argument count mismatch",
            ));
        }
        for (arg, ty) in args.iter().zip(&function.params[checked.len()..]) {
            // Parameter arithmetic retains its declared width inside wide verification expressions.
            let wide = self.verification;
            self.verification = false;
            let value = self.expr(arg, Some(ty), env, signature);
            self.verification = wide;
            checked.push(value?);
        }
        let mut expr = function.body.substitute(&checked, &mut self.next_local);
        expr.ty = function.result;
        Ok(expr)
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
            _ if self.types.exact_name(node).is_some()
                && !matches!(&node.kind, Kind::Name(name) if env.contains_key(name)) =>
            {
                let name = self.types.exact_name(node).unwrap();
                let mut expr = Expr::new(Ty::named("Type"), ExprKind::Type(name.clone()));
                expr.types = self.types.exact.get(&name).cloned();
                expr
            }
            Kind::Name(name) if name.contains("::") => {
                let (owner, field) = name.rsplit_once("::").unwrap();
                self.associated(node.offset, owner, field)?
            }
            Kind::Method(receiver, method, args) => {
                self.method(node.offset, receiver, method, args, env, signature)?
            }
            Kind::Member(receiver, field) => {
                self.member(node.offset, receiver, field, env, signature)?
            }
            Kind::Try(value) => {
                let value = self.expr(value, None, env, signature)?;
                let Ty::Optional(inner) = &value.ty else {
                    return Err(fail("? requires an optional value"));
                };
                Expr::new(*inner.clone(), ExprKind::Try(Box::new(value)))
            }
            Kind::Call(name, args) if self.library.hosts.contains_key(name) => {
                let method = self.library.hosts[name].clone();
                if args.len() != method.params.len() {
                    return Err(fail("host method argument count mismatch"));
                }
                let args = args
                    .iter()
                    .zip(&method.params)
                    .map(|(arg, (_, ty))| {
                        let wide = self.verification;
                        self.verification = false;
                        let result = self.expr(arg, Some(ty), env, signature);
                        self.verification = wide;
                        result
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Expr::new(method.result.clone(), ExprKind::Host(method, args))
            }
            Kind::Unary("-", value) if matches!(value.kind, Kind::Number(_) | Kind::Integer(_)) => {
                let n = match value.kind {
                    Kind::Number(n) => -i128::from(n),
                    Kind::Integer(n) => n
                        .checked_neg()
                        .ok_or_else(|| fail("expression arithmetic overflow"))?,
                    _ => unreachable!(),
                };
                let ty = expected.cloned().unwrap_or_else(|| Ty::named("i128"));
                if !ty.fits(n) {
                    return Err(fail("integer literal is out of range for its type"));
                }
                Expr::new(ty, ExprKind::Integer(n))
            }
            Kind::Unary(op, value) => {
                let value = self.expr(value, expected, env, signature)?;
                let value = if self.verification {
                    value.wide()
                } else {
                    value
                };
                if (*op == "!" && value.ty != Ty::named("bool"))
                    || (*op == "-" && !value.ty.integer())
                {
                    return Err(fail("invalid unary operand type"));
                }
                let constant = value.constant_node(self.types, node.offset).is_some();
                let expr = Expr::new(value.ty.clone(), ExprKind::Unary(op, Box::new(value)));
                if constant && expr.constant_node(self.types, node.offset).is_none() {
                    return Err(fail("expression arithmetic overflow"));
                }
                expr
            }
            Kind::Binary(op, lhs, rhs) => self.binary(op, lhs, rhs, expected, env, signature)?,
            Kind::Number(_) | Kind::Integer(_) => {
                let n = match node.kind {
                    Kind::Number(n) => i128::from(n),
                    Kind::Integer(n) => n,
                    _ => unreachable!(),
                };
                let ty = expected
                    .cloned()
                    .unwrap_or_else(|| Ty::named(if self.verification { "i128" } else { "u32" }));
                if !ty.fits(n) {
                    return Err(fail("integer literal is out of range for its type"));
                }
                Expr::new(ty, ExprKind::Integer(n))
            }
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
                    types: None,
                    ty: Ty::named(name),
                    kind: ExprKind::Record(checked),
                }
            }
            Kind::Call(name, args) if name == "field" && args.len() == 2 => {
                let Kind::Name(field) = &args[1].kind else {
                    return Err(fail("field requires a field name"));
                };
                self.member(node.offset, &args[0], field, env, signature)?
            }
            Kind::Call(name, args) if name == "type" && args.len() == 1 => {
                let value = self.expr(&args[0], None, env, signature)?;
                if !matches!(&value.ty, Ty::Named(n) if matches!(n.as_str(), "Int" | "Float" | "VectorConst"))
                {
                    // SSA references and typed constants have an IR type.
                    if !matches!(value.ty, Ty::Value(_)) {
                        return Err(fail("type expects an SSA value or typed property"));
                    }
                }
                Expr {
                    types: value.types.clone(),
                    ty: Ty::named("Type"),
                    kind: ExprKind::Query(Query::TypeOf, Box::new(value)),
                }
            }
            Kind::Call(name, args) if name == "result_type" && args.len() == 1 => {
                let index = match args[0].kind {
                    Kind::Number(n) => n as usize,
                    Kind::Integer(n) => {
                        usize::try_from(n).map_err(|_| fail("result_type index is out of range"))?
                    }
                    _ => return Err(fail("result_type requires a constant index")),
                };
                let results = signature
                    .and_then(|s| s.results.patterns())
                    .ok_or_else(|| {
                        fail("result_type requires a fixed operation result signature")
                    })?;
                if index >= results.len() {
                    return Err(fail("result_type index is out of range"));
                }
                Expr {
                    types: possible(self.types, &results[index], signature),
                    ty: Ty::named("Type"),
                    kind: ExprKind::ResultType(index),
                }
            }
            Kind::Call(name, args)
                if matches!(name.as_str(), "i128" | "i64" | "u64" | "u32" | "i32")
                    && args.len() == 1 =>
            {
                let value = self.expr(&args[0], None, env, signature)?;
                let Ty::Named(from) = &value.ty else {
                    return Err(fail("integer conversion requires an integer"));
                };
                if from != name
                    && !(name == "i128" && value.ty.integer())
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
                    types: None,
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
                    types: None,
                    ty: Ty::Optional(Box::new(value.ty.clone())),
                    kind: ExprKind::Some(Box::new(value)),
                }
            }
            Kind::Call(name, args)
                if self.library.functions.contains_key(name)
                    || self
                        .declarations
                        .iter()
                        .any(|d| matches!(d.kind.as_str(), "fn" | "const") && d.name == *name) =>
            {
                let function = self.function(name, node.offset)?;
                self.call(function, Vec::new(), args, node.offset, env, signature)?
            }
            Kind::List(nodes) if matches!(expected, Some(Ty::Array(_, _))) => {
                let Some(Ty::Array(ty, n)) = expected else {
                    unreachable!()
                };
                if nodes.len() != *n {
                    return Err(fail("projection array length mismatch"));
                }
                Expr {
                    types: None,
                    ty: expected.unwrap().clone(),
                    kind: ExprKind::Array(
                        nodes
                            .iter()
                            .map(|n| self.expr(n, Some(ty), env, signature))
                            .collect::<Result<_, _>>()?,
                    ),
                }
            }
            Kind::Call(name, args)
                if matches!(
                    name.as_str(),
                    "len" | "all" | "prefix" | "suffix" | "matches" | "results"
                ) =>
            {
                self.query(node.offset, name, args, env, signature)?
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
                    .ok_or_else(|| {
                        fail(&format!(
                            "unknown expression name or operation: {}",
                            super::constraints::describe(node)
                        ))
                    })?;
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
                    let params = params
                        .iter()
                        .map(|ty| Ty::property(ty, &self.data.rust))
                        .collect::<Vec<_>>();
                    if params.len() != args.len() {
                        return Err(fail("enum argument count mismatch"));
                    }
                    let args = args
                        .iter()
                        .zip(&params)
                        .map(|(arg, ty)| self.expr(arg, Some(ty), env, signature))
                        .collect::<Result<_, _>>()?;
                    Expr {
                        types: None,
                        ty,
                        kind: ExprKind::Variant(name.clone(), args),
                    }
                } else if matches!(&node.kind,Kind::Name(n) if n=="none")
                    && matches!(ty, Ty::Optional(_))
                {
                    Expr {
                        types: None,
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
                        self.encodings,
                    )?;
                    Expr {
                        types: None,
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

#[derive(Debug, Clone, Copy)]
pub(crate) enum Query {
    TypeOf,
    Len,
}

pub(crate) struct Emitter<'a> {
    pub projections: BTreeMap<String, String>,
    pub error: Option<String>,
    pub storage_used: std::cell::Cell<bool>,
    pub dfg: &'a str,
    pub results: &'a str,
    pub result_values: bool,
    pub operand_types: BTreeMap<String, String>,
    pub instruction: bool,
    pub constant: bool,
    pub const_failure: &'a str,
    pub prefix: &'a str,
}

impl<'a> Emitter<'a> {
    pub fn query(projections: BTreeMap<String, String>) -> Self {
        Self {
            projections,
            error: None,
            storage_used: std::cell::Cell::new(false),
            dfg: "dfg",
            results: "results",
            result_values: true,
            operand_types: BTreeMap::new(),
            instruction: false,
            constant: false,
            const_failure: "panic!(\"invalid constant expression\")",
            prefix: "",
        }
    }
    fn required(&self, value: String) -> String {
        if self.constant {
            return format!(
                "(match {value} {{ Some(value) => value, None => {} }})",
                self.const_failure
            );
        }
        match &self.error {
            None => format!("{value}?"),
            Some(error) if self.instruction => format!("{value}.ok_or_else(|| {error})?"),
            Some(error) => format!("{value}.ok_or({error})?"),
        }
    }
    fn operand(&self, term: &Expr) -> String {
        let code = self.term(term);
        if matches!(
            term.kind,
            ExprKind::Binary(..) | ExprKind::Query(Query::Len, _)
        ) {
            format!("({code})")
        } else {
            code
        }
    }

    pub fn term(&self, term: &Expr) -> String {
        let receiver = self.dfg.strip_prefix('&').unwrap_or(self.dfg);
        let numeric = |code: String| {
            if term.ty == Ty::named("i128") {
                if self.constant {
                    format!("(({code}) as i128)")
                } else {
                    format!("i128::from({code})")
                }
            } else {
                code
            }
        };
        match &term.kind {
            ExprKind::Constant(data::Value::Bool(value)) => value.to_string(),
            ExprKind::Constant(value) => value.rust(if !self.prefix.is_empty() {
                self.prefix
            } else if self.error.is_some() {
                "crate::inst::"
            } else {
                ""
            }),
            ExprKind::Integer(value) => format!("{value}{}", term.ty.name()),

            ExprKind::Parameter(_) => unreachable!("helper calls are expanded before generation"),
            ExprKind::Operand(name) => {
                self.storage_used.set(true);
                self.projections[name].clone()
            }
            ExprKind::Bound(id) => numeric(format!("_v{id}")),
            ExprKind::Field(value, field) => {
                let value = self.term(value);
                numeric(format!(
                    "({}).{field}",
                    value.strip_prefix('*').unwrap_or(&value)
                ))
            }
            ExprKind::ResultType(index) => {
                if self.result_values {
                    let get = if *index == 0 {
                        format!("{}.first()", self.results)
                    } else {
                        format!("{}.get({index})", self.results)
                    };
                    format!("({}).value_type(*{})", receiver, self.required(get))
                } else {
                    format!("{}[{index}]", self.results)
                }
            }
            ExprKind::Results if self.result_values => format!(
                "&{}.iter().map(|&v| ({}).value_type(v)).collect::<alloc::vec::Vec<_>>()",
                self.results, receiver
            ),
            ExprKind::Results => self.results.into(),
            ExprKind::Type(name) => crate::types::rust_type(name),
            ExprKind::Slice(sequence, index, prefix) => {
                let sequence = self.term(sequence);
                let index = self.required(format!("usize::try_from({}).ok()", self.term(index)));
                let range = if *prefix {
                    format!("..{index}")
                } else {
                    format!("{index}..")
                };
                self.required(format!("({sequence}).get({range})"))
            }
            ExprKind::Matches(values, types) => format!(
                "{{ let values = {}; let types = {}; values.len() == types.len() && values.iter().zip(types.iter()).all(|(&v, &ty)| ({}).value_type(v) == ty) }}",
                self.term(values),
                self.term(types),
                receiver
            ),
            ExprKind::Unary("!", value) => format!("!({})", self.term(value)),
            ExprKind::Unary("-", value) => {
                self.required(format!("({}).checked_neg()", self.term(value)))
            }
            ExprKind::Unary(_, _) => unreachable!("checked unary operator"),
            ExprKind::Binary(op, lhs, rhs) => match *op {
                "+" | "-" | "*" => {
                    let (lhs, rhs) = (self.term(lhs), self.term(rhs));
                    let method = match *op {
                        "+" => "checked_add",
                        "-" => "checked_sub",
                        _ => "checked_mul",
                    };
                    self.required(format!("({lhs}).{method}({rhs})"))
                }
                _ => format!("{} {op} {}", self.operand(lhs), self.operand(rhs)),
            },
            ExprKind::Query(query, value) => {
                if let Query::TypeOf = query
                    && let ExprKind::Operand(name) = &value.kind
                    && let Some(ty) = self.operand_types.get(name)
                {
                    return ty.clone();
                }
                let sort = &value.ty;
                let value = self.term(value);
                match query {
                    Query::TypeOf => {
                        if matches!(sort, Ty::Named(n) if matches!(n.as_str(), "Int" | "Float" | "VectorConst"))
                        {
                            format!("({value}).ty()")
                        } else {
                            format!("({receiver}).value_type({value})")
                        }
                    }
                    Query::Len => format!("({value}).len() as i128"),
                }
            }

            ExprKind::Convert(e) if self.constant => {
                format!("(({}) as {})", self.term(e), term.ty.name())
            }
            ExprKind::Convert(e) => format!("{}::from({})", term.ty.name(), self.term(e)),
            ExprKind::Record(fields) => format!(
                "{}{} {{ {} }}",
                if !self.prefix.is_empty() {
                    self.prefix
                } else if self.error.is_some() {
                    "crate::inst::"
                } else {
                    ""
                },
                term.ty.name(),
                fields
                    .iter()
                    .map(|(n, e)| format!("{n}: {}", self.term(e)))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            ExprKind::Variant(name, args) => {
                let prefix = if self.constant {
                    self.prefix
                } else {
                    "crate::inst::"
                };
                let path = format!("{prefix}{}::{name}", term.ty.name());
                if args.is_empty() {
                    path
                } else {
                    format!(
                        "{path}({})",
                        args.iter()
                            .map(|e| self.term(e))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
            }
            ExprKind::Try(e) => self.required(format!("({})", self.term(e))),
            ExprKind::Rust(binding, args) => {
                let path = binding.method.as_ref().map_or_else(
                    || binding.path.clone(),
                    |(owner, method)| format!("{owner}::{method}"),
                );
                if binding.constant {
                    path
                } else {
                    format!(
                        "{path}({})",
                        args.iter()
                            .map(|arg| self.term(arg))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
            }
            ExprKind::Host(method, args) => format!(
                "crate::host::traits::{}::{}(&_host{})",
                method.interface,
                method.name,
                args.iter()
                    .map(|e| format!(", {}", self.term(e)))
                    .collect::<String>()
            ),
            ExprKind::Some(e) => format!("Some({})", self.term(e)),
            ExprKind::Array(args) => format!(
                "[{}]",
                args.iter()
                    .map(|e| self.term(e))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            ExprKind::All(sequence, id, body)
                if self.constant && matches!(sequence.ty, Ty::Optional(_)) =>
            {
                format!(
                    "(match {} {{ Some(_v{id}) => {}, None => true }})",
                    self.term(sequence),
                    self.term(body)
                )
            }
            ExprKind::All(sequence, id, body) if self.constant => format!(
                "{{ let seq{id} = {}; let mut i{id} = 0; let mut ok{id} = true; while i{id} < seq{id}.len() {{ let _v{id} = seq{id}[i{id}]; if !({}) {{ ok{id} = false; break; }} i{id} += 1; }} ok{id} }}",
                self.term(sequence),
                self.term(body)
            ),
            ExprKind::All(sequence, id, body) => format!(
                "{{ let mut ok{id} = true; for &_v{id} in ({}).iter() {{ if !({}) {{ ok{id} = false; break; }} }} ok{id} }}",
                self.term(sequence),
                self.term(body)
            ),
        }
    }
}
