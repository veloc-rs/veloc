use super::typed::Signature;
use crate::{
    Definitions, Error, interfaces,
    syntax::{Decl, DeclKind, Kind, Node},
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

#[derive(Clone, Copy)]
pub struct DecisionRust<'a> {
    /// Explicit namespace of the supplied OpSpec definitions.
    pub dialect: &'a str,
    pub function: &'a str,
    pub opcode: &'a str,
    pub field: &'a str,
    pub result: &'a str,
    /// Explicit rewrite_interface declaration used for value construction.
    pub value_interface: &'a str,
    /// Value-building adapter: (rewrite context, body) -> rewrite result.
    pub value_adapter: &'a str,
    /// Shared action constructor for generated and host rewrites.
    pub rewrite: &'a str,
    pub legal_action: &'a str,
}

pub fn decisions(
    source: &str,
    defs: &Definitions,
    config: DecisionRust<'_>,
) -> Result<String, Error> {
    compile(source, &crate::syntax::parse(source)?, defs, config)
}

impl crate::Source {
    pub fn decisions(
        &self,
        defs: &Definitions,
        config: DecisionRust<'_>,
    ) -> Result<String, crate::SourceError> {
        self.check_imports()?;
        compile(self.text(), self.declarations(), defs, config).map_err(|e| self.locate(e))
    }
}

fn compile(
    source: &str,
    declarations: &[Decl],
    defs: &Definitions,
    config: DecisionRust<'_>,
) -> Result<String, Error> {
    if !super::identifier(config.function) || !super::identifier(config.dialect) {
        return Err(Error::at(source, 0, "invalid Rust function name"));
    }
    interfaces::rust_path(source, 0, config.opcode)?;
    interfaces::rust_path(source, 0, config.field)?;
    interfaces::rust_path(source, 0, config.value_adapter)?;
    interfaces::rust_path(source, 0, config.rewrite)?;
    interfaces::rust_path(source, 0, config.legal_action)?;
    let bindings = interfaces::Bindings::compile(declarations, source)?;
    let interface = ValueInterface::compile(declarations, source, config, &bindings)?;
    let emit_method = &interface.emit;
    let bound = &interface.contract;
    let logical_types = crate::types::Types::compile(declarations, source)?;
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
    let functions = super::functions::Functions::compile(source, declarations, &aliases)?;
    functions.validate(source, &operations, defs, config.dialect)?;
    let result = bindings
        .0
        .get(config.result)
        .ok_or_else(|| Error::at(source, 0, "unknown decision result type"))?;
    let result_path = result.path.clone();
    let mut out = interfaces::declarations(declarations, source, "host")?;
    let mut bodies = String::new();
    let mut hosts = BTreeMap::new();
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut rewrites = BTreeMap::new();
    for d in declarations {
        if !matches!(d.kind, DeclKind::Rewrite(_)) {
            continue;
        }
        if d.fields.len() != 1
            || !(d.fields.contains_key("replace") || d.fields.contains_key("rust"))
        {
            return Err(Error::at(
                source,
                d.offset,
                "rewrite requires a replace body or Rust binding",
            ));
        }
        let (sig, roots) =
            Signature::parse(source, d, &aliases, &operations, defs, config.dialect)?;
        if !sig.hosts.is_empty() || roots.len() != 1 {
            return Err(Error::at(
                source,
                d.offset,
                "rewrite requires one node and no host parameters",
            ));
        }
        let action = if let Some(binding) = d.fields.get("rust") {
            let Kind::Text(path) = &binding.kind else {
                return Err(Error::at(
                    source,
                    binding.offset,
                    "expected Rust function path",
                ));
            };
            interfaces::rust_path(source, binding.offset, path)?;
            // Check every binding, including templates not selected by any rule.
            writeln!(
                bodies,
                "fn rewrite_{}_host() -> {result_path} {{ {}({:?}, {path}) }}",
                d.name, config.rewrite, d.name
            )
            .unwrap();
            format!("rewrite_{}_host()", d.name)
        } else {
            format!("rewrite_{}_case1_action()", d.name)
        };
        if rewrites
            .insert(d.name.clone(), (sig, roots, action))
            .is_some()
        {
            return Err(Error::at(source, d.offset, "duplicate rewrite"));
        }
    }
    let mut names = BTreeSet::new();
    for d in declarations {
        if matches!(d.kind, DeclKind::Type { .. } | DeclKind::TypeSet(_))
            || matches!(&d.kind, DeclKind::Fields(k) if k == "rewrite_interface")
        {
            continue;
        }
        if !names.insert(&d.name) {
            return Err(Error::at(source, d.offset, "duplicate decision rule"));
        }
        if matches!(d.kind, DeclKind::Function { .. }) {
            continue;
        }
        if matches!(d.kind, DeclKind::Rewrite(_)) && d.fields.contains_key("rust") {
            continue;
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
            logical_types: &logical_types,
            bindings: &bindings,
            hosts: &sig.hosts,
            signature: &sig,
            rewrites: &rewrites,
            roots: &roots,
            legal_action: config.legal_action,
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
        for name in roots.iter().cloned() {
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
                let (ty, value) = sig.expression(
                    source,
                    emit,
                    &operations,
                    defs,
                    &mut insts,
                    config.dialect,
                    &mut BTreeMap::new(),
                    &functions,
                    &mut Vec::new(),
                )?;
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
                let value_ty = &interface.value;
                let type_ty = &bindings.0["Type"].path;
                let mut parameters = vec!["ctx: &mut C".to_owned()];
                let mut arguments = vec!["builder".to_owned()];
                for (i, _) in sig.inputs.iter().enumerate() {
                    parameters.push(format!("input{i}: {value_ty}"));
                    arguments.push(format!("inputs[{i}]"));
                }
                for (i, generic) in sig.generics.keys().enumerate() {
                    if !insts.iter().any(|inst| {
                        inst.ty.name == *generic
                            || matches!(&inst.op, super::typed::Call::Host { types, .. }
                            if types.iter().any(|ty| ty.name == *generic))
                    }) {
                        continue;
                    }
                    parameters.push(format!("ty{i}: {type_ty}"));
                    let (result, index) = sig.anchor(generic).unwrap();
                    let index = if result {
                        index
                    } else {
                        sig.results.len() + index
                    };
                    arguments.push(format!("_types[{index}]"));
                }
                parameters.push(format!("destination: Option<{value_ty}>"));
                arguments.push("Some(destination)".into());
                writeln!(
                    bodies,
                    "#[allow(unused_variables)]\nfn {function}<C: {bound}>({}) -> {value_ty} {{",
                    parameters.join(", ")
                )
                .unwrap();
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
                        "destination"
                    } else {
                        "None"
                    };
                    let binding = format!("let {} = ", inst.result);
                    match &inst.op {
                        super::typed::Call::Integer(op, immediate) => {
                            let variant = operations[op]
                                .attributes
                                .first()
                                .ok_or_else(|| {
                                    Error::at(
                                        source,
                                        emit.offset,
                                        "integer attribute requires a storage codec",
                                    )
                                })?
                                .2
                                .as_str();
                            writeln!(bodies, "{binding}ctx.{emit_method}({}::{op}, {ty}, &[], &[{}::{variant}({immediate})], {destination});", config.opcode, config.field).unwrap();
                        }
                        super::typed::Call::Attributed(op, attributes) => {
                            let fields = attributes
                                .iter()
                                .map(|(variant, name)| {
                                    Ok(format!(
                                        "{}::{variant}({})",
                                        config.field,
                                        expressions.constant(name, emit.offset)?
                                    ))
                                })
                                .collect::<Result<Vec<_>, Error>>()?
                                .join(", ");
                            writeln!(bodies, "{binding}ctx.{emit_method}({}::{op}, {ty}, &[{}], &[{fields}], {destination});",
                                config.opcode, inst.inputs.join(", ")).unwrap();
                        }
                        super::typed::Call::Instruction(op) => {
                            writeln!(
                                bodies,
                                "{binding}ctx.{emit_method}({}::{op}, {ty}, &[{}], &[], {destination});",
                                config.opcode,
                                inst.inputs.join(", ")
                            )
                            .unwrap();
                        }
                        super::typed::Call::Host { name, types } => {
                            let mut args = types
                                .iter()
                                .map(|ty| {
                                    if let Some(i) = sig.generics.keys().position(|n| n == &ty.name)
                                    {
                                        Ok(format!("ty{i}"))
                                    } else {
                                        expressions.constant(&ty.name, emit.offset)
                                    }
                                })
                                .collect::<Result<Vec<_>, Error>>()?;
                            args.extend(inst.inputs.iter().cloned());
                            writeln!(
                                bodies,
                                "let {} = build_{name}(ctx, {});",
                                inst.result,
                                args.join(", ")
                            )
                            .unwrap();
                        }
                    }
                }
                writeln!(bodies, "{value}\n}}").unwrap();
                writeln!(bodies,
                    "fn {function}_action() -> {result_path} {{ {}({:?}, |ctx| {}(ctx, |builder, inputs, _types, destination| {function}({}))) }}",
                    config.rewrite, d.name, config.value_adapter, arguments.join(", ")
                ).unwrap();
                format!("{function}_action()")
            } else {
                expressions.rust(emit)?
            };
            if matches!(d.kind, DeclKind::Rewrite(_)) {
                continue;
            }
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
        .map(|(name, ty)| {
            let contract = bindings.method_trait(ty, "host", false);
            let contract = contract.strip_prefix("host::").unwrap_or(&contract);
            format!("{name}: &impl {contract}")
        })
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
    out.push_str(&functions.wrappers(ty, bound, &interface.value));
    out.push_str(&bodies);
    Ok(out)
}

struct Expressions<'a> {
    legal_action: &'a str,
    rewrites: &'a BTreeMap<String, (Signature, Vec<String>, String)>,
    roots: &'a [String],
    source: &'a str,
    logical_types: &'a crate::types::Types,
    types: &'a BTreeMap<&'a str, &'a Decl>,
    bindings: &'a interfaces::Bindings,
    hosts: &'a [(String, String)],
    signature: &'a Signature,
}
impl Expressions<'_> {
    fn expand(&self, offset: usize, args: &[Node]) -> Result<String, Error> {
        let [
            Node {
                kind: Kind::Name(name),
                ..
            },
            Node {
                kind: Kind::Name(node),
                ..
            },
        ] = args
        else {
            return Err(Error::at(
                self.source,
                offset,
                "expected expand(rewrite, node)",
            ));
        };
        if node != &self.signature.node {
            return Err(Error::at(
                self.source,
                offset,
                "rewrite argument must be the matched instruction",
            ));
        }
        let (template, roots, action) = self
            .rewrites
            .get(name)
            .ok_or_else(|| Error::at(self.source, offset, format!("unknown rewrite {name}")))?;
        let actual = self.signature;
        if actual.dynamic
            || actual.inputs.len() != template.inputs.len()
            || actual.results.len() != template.results.len()
            || !self.roots.iter().all(|root| roots.contains(root))
        {
            return Err(Error::at(
                self.source,
                offset,
                "rewrite node signature does not match",
            ));
        }
        let mut generics = BTreeMap::new();
        for (expected, actual) in template
            .inputs
            .iter()
            .map(|(_, ty)| ty)
            .chain(&template.results)
            .zip(
                actual
                    .inputs
                    .iter()
                    .map(|(_, ty)| ty)
                    .chain(&actual.results),
            )
        {
            if !actual.domain.iter().all(|ty| expected.domain.contains(ty)) {
                return Err(Error::at(
                    self.source,
                    offset,
                    "rewrite type domain does not match",
                ));
            }
            if template.generics.contains_key(&expected.name) {
                if let Some(previous) = generics.insert(&expected.name, actual) {
                    if previous != actual
                        && !(actual.domain.len() == 1 && actual.domain == previous.domain)
                    {
                        return Err(Error::at(
                            self.source,
                            offset,
                            "rewrite requires equal argument types",
                        ));
                    }
                }
            }
        }
        Ok(action.clone())
    }

    fn constant(&self, name: &str, offset: usize) -> Result<String, Error> {
        let fail = || Error::at(self.source, offset, format!("undeclared constant {name}"));
        let (owner, member) = name.split_once("::").ok_or_else(fail)?;
        if owner == "Type" && self.logical_types.exact.contains_key(member) {
            return Ok(format!(
                "{}::{member}",
                self.bindings.0.get(owner).ok_or_else(fail)?.path
            ));
        }
        let declaration = self.types.get(owner).ok_or_else(fail)?;
        if !declaration
            .members()
            .iter()
            .any(|d| d.name == member && matches!(d.kind, DeclKind::Constant { .. }))
        {
            return Err(fail());
        }
        let interface = self.bindings.method_trait(owner, "host", true);
        let interface = interface.strip_prefix("host::").unwrap_or(&interface);
        Ok(format!(
            "<{} as {}>::{member}",
            self.bindings.0.get(owner).ok_or_else(fail)?.path,
            interface
        ))
    }
    fn match_expr(&self, value: &Node, arms: &[crate::syntax::MatchArm]) -> Result<String, Error> {
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
            Kind::Call(name, args) if name == "expand" => self.expand(node.offset, args)?,
            Kind::Number(n) => n.to_string(),
            Kind::Integer(n) => n.to_string(),
            Kind::Name(n) if n == "legal" => self.legal_action.into(),
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

struct ValueInterface {
    contract: String,
    value: String,
    emit: String,
}
impl ValueInterface {
    fn compile(
        declarations: &[Decl],
        source: &str,
        config: DecisionRust<'_>,
        bindings: &interfaces::Bindings,
    ) -> Result<Self, Error> {
        use crate::syntax::Results;
        let fail = |message| Error::at(source, 0, message);
        let records: Vec<_> = declarations
            .iter()
            .filter(|d| {
                d.name == config.value_interface
                    && matches!(&d.kind, DeclKind::Fields(k) if k == "rewrite_interface")
            })
            .collect();
        let [record] = records.as_slice() else {
            return Err(fail("expected one rewrite_interface declaration"));
        };
        for key in record.fields.keys() {
            if !matches!(key.as_str(), "contract" | "emit") {
                return Err(Error::at(
                    source,
                    record.offset,
                    "unknown rewrite interface role",
                ));
            }
        }
        let name = |role: &str| -> Result<String, Error> {
            match record.fields.get(role) {
                Some(Node {
                    kind: Kind::Name(name),
                    ..
                }) => Ok(name.clone()),
                _ => Err(Error::at(
                    source,
                    record.offset,
                    format!("missing rewrite interface role {role}"),
                )),
            }
        };
        let owner = name("contract")?;
        let declaration = declarations
            .iter()
            .find(|d| d.name == owner && interfaces::rust_binding(d).is_some())
            .ok_or_else(|| fail("rewrite interface requires a Rust-bound type"))?;
        let emit = name("emit")?;
        let method = declaration
            .members()
            .iter()
            .find(|m| m.name == emit)
            .ok_or_else(|| {
                Error::at(
                    source,
                    record.offset,
                    format!("undeclared rewrite method {emit}"),
                )
            })?;
        let known = bindings.0.keys().cloned().collect();
        let signature = method
            .signature()
            .ok_or_else(|| fail("rewrite role requires a method"))?;
        let Results::Fixed(results) = &signature.results else {
            return Err(fail("invalid emit result"));
        };
        let [result] = results.as_slice() else {
            return Err(fail("emit must return one value"));
        };
        let value = interfaces::Type::parse(&result.ty, source, &known)?.rust(bindings);
        let ty = &bindings
            .0
            .get("Type")
            .ok_or_else(|| fail("missing Type binding"))?
            .path;
        let expected = vec![
            config.opcode.to_owned(),
            ty.clone(),
            format!("&[{value}]"),
            format!("&[{}]", config.field),
            format!("Option<{value}>"),
        ];
        let valid_receiver = signature.params.first().is_some_and(|p| {
            p.name == "self" && matches!(&p.ty.kind, Kind::Call(n, _) if n == "mut_ref")
        });
        let actual = signature
            .params
            .iter()
            .skip(1)
            .map(|p| interfaces::Type::parse(&p.ty, source, &known).map(|t| t.rust(bindings)))
            .collect::<Result<Vec<_>, _>>()?;
        if !valid_receiver || actual != expected {
            return Err(Error::at(
                source,
                method.offset,
                "invalid signature for rewrite role emit",
            ));
        }
        let contract = bindings.method_trait(&owner, "host", false);
        let contract = contract
            .strip_prefix("host::")
            .unwrap_or(&contract)
            .to_owned();
        Ok(Self {
            contract,
            value,
            emit,
        })
    }
}
