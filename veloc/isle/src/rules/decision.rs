use super::typed::Signature;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use veloc_opgen::{
    Definitions, Error, interfaces,
    syntax::{Decl, DeclKind, Kind, Node},
};

pub struct DecisionRust<'a> {
    /// Explicit namespace of the supplied OpSpec definitions.
    pub dialect: &'a str,
    pub function: &'a str,
    pub opcode: &'a str,
    pub result: &'a str,
    /// Runtime adapter for a generated value rewrite: (name, body) -> action.
    pub value_rule: &'a str,
}

pub fn decisions(
    source: &str,
    defs: &Definitions,
    config: DecisionRust<'_>,
) -> Result<String, Error> {
    if !super::identifier(config.function) || !super::identifier(config.dialect) {
        return Err(Error::at(source, 0, "invalid Rust function name"));
    }
    interfaces::rust_path(source, 0, config.opcode)?;
    interfaces::rust_path(source, 0, config.value_rule)?;
    let declarations = veloc_opgen::syntax::parse(source)?;
    let bindings = interfaces::Bindings::compile(&declarations, source)?;
    let types: BTreeMap<_, _> = declarations
        .iter()
        .filter_map(|d| matches!(d.kind, DeclKind::Type { .. }).then_some((d.name.as_str(), d)))
        .collect();
    let aliases = declarations
        .iter()
        .filter_map(|d| match &d.kind {
            DeclKind::TypeSet(node) => Some((d.name.clone(), node.clone())),
            _ => None,
        })
        .collect();
    let operations: BTreeMap<_, _> = defs.operations().map(|op| (op.name.clone(), op)).collect();
    let result = bindings
        .0
        .get(config.result)
        .ok_or_else(|| Error::at(source, 0, "unknown decision result type"))?;
    let result_path = result.path.clone();
    let mut out = interfaces::declarations(&declarations, source, "host")?;
    let mut bodies = String::new();
    let mut hosts = BTreeMap::new();
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut names = BTreeSet::new();
    for d in &declarations {
        if matches!(d.kind, DeclKind::Type { .. } | DeclKind::TypeSet(_)) {
            continue;
        }
        if !names.insert(&d.name) {
            return Err(Error::at(source, d.offset, "duplicate decision rule"));
        }
        if d.fields
            .keys()
            .any(|key| !matches!(key.as_str(), "when" | "action" | "replace"))
        {
            return Err(Error::at(source, d.offset, "unknown decision rule field"));
        }
        let (sig, roots) =
            Signature::parse(source, d, &aliases, &operations, defs, config.dialect)?;
        for (name, owner) in &sig.hosts {
            if name == "opcode" {
                return Err(Error::at(
                    source,
                    d.offset,
                    "opcode is reserved for the matching adapter",
                ));
            }
            if !types.contains_key(owner.as_str()) {
                return Err(Error::at(
                    source,
                    d.offset,
                    format!("unknown host type {owner}"),
                ));
            }
            if let Some(previous) = hosts.insert(name.clone(), owner.clone()) {
                if previous != *owner {
                    return Err(Error::at(
                        source,
                        d.offset,
                        "conflicting host parameter types",
                    ));
                }
            }
        }
        let (emit, replacement) = match (d.fields.get("action"), d.fields.get("replace")) {
            (Some(action), None) => (action, false),
            (None, Some(replace)) => (replace, true),
            _ => {
                return Err(Error::at(
                    source,
                    d.offset,
                    "specify exactly one action or replace",
                ));
            }
        };
        let expressions = Expressions {
            source,
            types: &types,
            bindings: &bindings,
            hosts: &sig.hosts,
            signature: &sig,
        };
        let mut guards = Vec::new();
        let sets = |types: Vec<&super::typed::Ty>| -> Result<String, Error> {
            let mut sets = Vec::new();
            for ty in types {
                sets.push(format!(
                    "&[{}]",
                    ty.domain
                        .iter()
                        .map(|n| expressions.constant(n, d.offset))
                        .collect::<Result<Vec<_>, _>>()?
                        .join(", ")
                ));
            }
            Ok(format!("&[{}]", sets.join(", ")))
        };
        // Dynamic nodes (calls/returns) keep their declared host adapter.
        // Fixed-arity nodes derive all type tests from their OpSpec signature.
        let structural = sig.dynamic;
        if !structural {
            guards.push(format!(
                "query.signature({}, {})",
                sets(sig.results.iter().collect())?,
                sets(sig.inputs.iter().map(|(_, t)| t).collect())?
            ));
            for name in sig.generics.keys() {
                let indices = sig
                    .results
                    .iter()
                    .chain(sig.inputs.iter().map(|(_, t)| t))
                    .enumerate()
                    .filter(|(_, ty)| &ty.name == name)
                    .map(|(i, _)| i.to_string())
                    .collect::<Vec<_>>();
                if indices.len() > 1 {
                    guards.push(format!("query.same(&[{}])", indices.join(", ")));
                }
            }
        }
        if let Some(when) = d.fields.get("when") {
            guards.push(expressions.rust(when)?);
        }
        let guard = if guards.is_empty() {
            "true".into()
        } else {
            guards.join(" && ")
        };
        let mut seen = BTreeSet::new();
        for name in roots {
            let op = operations
                .get(&name)
                .ok_or_else(|| Error::at(source, d.offset, format!("unknown operation {name}")))?;
            if !seen.insert(name.clone()) {
                return Err(Error::at(source, d.offset, "repeated operation pattern"));
            }
            // Host-backed memory/control rules intentionally retain their
            // instruction-specific adapter. Pure value rules are checked here.
            if op.signature.is_ok() && !structural {
                sig.check_call(
                    source,
                    d.offset,
                    op,
                    &sig.inputs
                        .iter()
                        .map(|(_, t)| t.clone())
                        .collect::<Vec<_>>(),
                    &sig.results,
                    defs,
                )?;
            }
            let action = if replacement {
                if let Err(message) = &op.signature {
                    return Err(Error::at(
                        source,
                        d.offset,
                        format!(
                            "value rewrite cannot discard instruction effects or attributes: {message}"
                        ),
                    ));
                }
                if structural || sig.results.len() != 1 {
                    return Err(Error::at(
                        source,
                        emit.offset,
                        "value replacement requires one explicit result",
                    ));
                }
                let mut insts = Vec::new();
                let (ty, value) =
                    sig.expression(source, emit, &operations, defs, &mut insts, config.dialect)?;
                if ty != sig.results[0]
                    && !(ty.domain.len() == 1 && ty.domain == sig.results[0].domain)
                {
                    return Err(Error::at(
                        source,
                        emit.offset,
                        "replacement result does not match rule result",
                    ));
                }
                let function = format!("rewrite_{}_case{}", d.name, seen.len());
                writeln!(bodies, "fn {function}<C: ValueRewrite>(ctx: &mut C) {{").unwrap();
                for (i, _) in sig.inputs.iter().enumerate() {
                    writeln!(bodies, "let input{i} = ctx.input({i});").unwrap();
                }
                for (generic, _) in &sig.generics {
                    let (result, index) = sig.anchor(generic).unwrap();
                    writeln!(
                        bodies,
                        "let ty{} = ctx.value_type({result}, {index});",
                        sig.generics.keys().position(|n| n == generic).unwrap()
                    )
                    .unwrap();
                }
                for inst in &insts {
                    let ty = if sig.generics.contains_key(&inst.ty.name) {
                        format!(
                            "ty{}",
                            sig.generics
                                .keys()
                                .position(|n| n == &inst.ty.name)
                                .unwrap()
                        )
                    } else {
                        expressions.constant(&inst.ty.name, emit.offset)?
                    };
                    let destination = if inst.result == value {
                        "Some(0)"
                    } else {
                        "None"
                    };
                    let binding = if inst.result == value {
                        String::new()
                    } else {
                        format!("let {} = ", inst.result)
                    };
                    writeln!(
                        bodies,
                        "{binding}ctx.emit({}::{}, {ty}, &[{}], {destination});",
                        config.opcode,
                        inst.op,
                        inst.inputs.join(", ")
                    )
                    .unwrap();
                }
                if !insts.iter().any(|i| i.result == value) {
                    writeln!(bodies, "ctx.bind(0, {value});").unwrap();
                }
                bodies.push_str("}\n");
                format!("{}({:?}, |ctx| {function}(ctx))", config.value_rule, d.name)
            } else {
                expressions.rust(emit)?
            };
            groups.entry(name).or_default().push(format!(
                "// rule {}\nif {guard} {{ return Some({action}); }}",
                d.name
            ));
        }
    }
    // The query adapter is the matcher input, not a lexical variable available
    // to rules. Referring to it in a rule requires an explicit host parameter.
    if hosts.get("query").is_some_and(|owner| owner != "Query") {
        return Err(Error::at(
            source,
            0,
            "query is reserved for the Query matching adapter",
        ));
    }
    hosts.insert("query".into(), "Query".into());
    let args = hosts
        .iter()
        .map(|(name, ty)| format!("{name}: &impl {ty}"))
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(
        out,
        "pub fn {}(opcode: {}, {args}) -> Option<{result_path}> {{ match opcode {{",
        config.function, config.opcode
    )
    .unwrap();
    for (opcode, rules) in groups {
        writeln!(
            out,
            "{}::{opcode} => {{ {} None }},",
            config.opcode,
            rules.join("\n")
        )
        .unwrap();
    }
    out.push_str("_ => None,\n} }\n");
    let ty = &bindings
        .0
        .get("Type")
        .ok_or_else(|| Error::at(source, 0, "value rules require a Type binding"))?
        .path;
    writeln!(out, "#[allow(dead_code)]\n    pub trait ValueRewrite {{
        type Value: Copy;
        fn input(&self, index: usize) -> Self::Value;
        fn value_type(&self, result: bool, index: usize) -> {ty};
        fn emit(&mut self, opcode: {}, ty: {ty}, inputs: &[Self::Value], result: Option<usize>) -> Self::Value;
        fn bind(&mut self, result: usize, value: Self::Value);
    }}", config.opcode).unwrap();
    out.push_str(&bodies);
    Ok(out)
}

struct Expressions<'a> {
    source: &'a str,
    types: &'a BTreeMap<&'a str, &'a Decl>,
    bindings: &'a interfaces::Bindings,
    hosts: &'a [(String, String)],
    signature: &'a Signature,
}
impl Expressions<'_> {
    fn constant(&self, name: &str, offset: usize) -> Result<String, Error> {
        let fail = || Error::at(self.source, offset, format!("undeclared constant {name}"));
        let (owner, member) = name.split_once("::").ok_or_else(fail)?;
        let declaration = self.types.get(owner).ok_or_else(fail)?;
        if !declaration
            .members()
            .iter()
            .any(|d| d.name == member && matches!(d.kind, DeclKind::Constant { .. }))
        {
            return Err(fail());
        }
        Ok(format!(
            "<{} as {owner}>::{member}",
            self.bindings.0.get(owner).ok_or_else(fail)?.path
        ))
    }
    fn match_expr(
        &self,
        value: &Node,
        arms: &[veloc_opgen::syntax::MatchArm],
    ) -> Result<String, Error> {
        let wildcard = |node: &Node| matches!(&node.kind, Kind::Name(name) if name == "_");
        if arms
            .last()
            .is_none_or(|arm| !wildcard(&arm.pattern) || arm.guard.is_some())
        {
            return Err(Error::at(
                self.source,
                value.offset,
                "match requires a final unguarded _ fallback",
            ));
        }
        let mut binding = "__match_value".to_owned();
        while self.hosts.iter().any(|(name, _)| name == &binding) {
            binding.push('_');
        }
        let mut out = format!("match {} {{", self.rust(value)?);
        let mut covered = BTreeSet::new();
        for (index, arm) in arms.iter().enumerate() {
            let is_wildcard = wildcard(&arm.pattern);
            if is_wildcard && arm.guard.is_none() && index + 1 != arms.len() {
                return Err(Error::at(
                    self.source,
                    arm.pattern.offset,
                    "unreachable arm after _ fallback",
                ));
            }
            let pattern = if is_wildcard {
                None
            } else {
                match &arm.pattern.kind {
                    Kind::Name(name)
                        if name.contains("::") || matches!(name.as_str(), "true" | "false") => {}
                    Kind::Number(_) | Kind::Integer(_) => {}
                    _ => {
                        return Err(Error::at(
                            self.source,
                            arm.pattern.offset,
                            "match patterns must be declared constants, literals, or _",
                        ));
                    }
                }
                Some(self.rust(&arm.pattern)?)
            };
            if let Some(pattern) = &pattern {
                if covered.contains(pattern) {
                    return Err(Error::at(
                        self.source,
                        arm.pattern.offset,
                        "unreachable repeated match pattern",
                    ));
                }
                if arm.guard.is_none() {
                    covered.insert(pattern.clone());
                }
            }
            let mut conditions = Vec::new();
            if let Some(pattern) = pattern {
                conditions.push(format!("{binding} == {pattern}"));
            }
            if let Some(guard) = &arm.guard {
                conditions.push(format!("({})", self.rust(guard)?));
            }
            if conditions.is_empty() {
                write!(out, "_ => {},", self.rust(&arm.value)?).unwrap();
            } else {
                // Equality supports foreign Rust types without requiring structural
                // constant patterns. The scrutinee is still evaluated exactly once.
                let binding = if is_wildcard { "_" } else { &binding };
                write!(
                    out,
                    "{binding} if {} => {},",
                    conditions.join(" && "),
                    self.rust(&arm.value)?
                )
                .unwrap();
            }
        }
        out.push('}');
        Ok(out)
    }

    fn rust(&self, node: &Node) -> Result<String, Error> {
        let fail = || {
            Error::at(
                self.source,
                node.offset,
                "unsupported expression or undeclared host member",
            )
        };
        Ok(match &node.kind {
            Kind::Match(value, arms) => self.match_expr(value, arms)?,
            Kind::Number(n) => n.to_string(),
            Kind::Integer(n) => n.to_string(),
            Kind::Name(n) if matches!(n.as_str(), "true" | "false") => n.clone(),
            Kind::Name(n) if self.signature.generics.contains_key(n) => {
                let (result, index) = self.signature.anchor(n).unwrap();
                format!("query.value_type({result}, {index})")
            }
            Kind::Name(n) => self.constant(n, node.offset)?,
            Kind::List(nodes) => format!(
                "&[{}]",
                nodes
                    .iter()
                    .map(|n| self.rust(n))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(", ")
            ),
            Kind::Unary(op, n) if *op == "!" => format!("!({})", self.rust(n)?),
            Kind::Binary(op, lhs, rhs) => format!("({} {op} {})", self.rust(lhs)?, self.rust(rhs)?),
            Kind::Method(receiver, method, args) => {
                let Kind::Name(name) = &receiver.kind else {
                    return Err(fail());
                };
                let (_, owner) = self
                    .hosts
                    .iter()
                    .find(|(n, _)| n == name)
                    .ok_or_else(fail)?;
                let declaration = self.types[owner.as_str()]
                    .members()
                    .iter()
                    .find(|d| d.name == *method)
                    .ok_or_else(fail)?;
                let signature = declaration.signature().ok_or_else(fail)?;
                if signature.params.first().is_none_or(|p| p.name != "self")
                    || signature.params.len() != args.len() + 1
                {
                    return Err(fail());
                }
                format!(
                    "{name}.{method}({})",
                    args.iter()
                        .map(|n| self.rust(n))
                        .collect::<Result<Vec<_>, _>>()?
                        .join(", ")
                )
            }
            _ => return Err(fail()),
        })
    }
}
