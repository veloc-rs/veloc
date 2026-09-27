use super::typed::Signature;
use crate::bytecode::intern;
mod predicate;
mod program;
use crate::{
    Definitions, Error, interfaces,
    syntax::{Decl, DeclKind, Kind, Node},
};
use predicate::Test;
use program::Program;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use veloc_bytecode::rewrite::Instruction as Op;

#[derive(Clone, Copy)]
pub struct DecisionRust<'a> {
    /// Explicit namespace of the supplied OpSpec definitions.
    pub dialect: &'a str,
    pub function: &'a str,
    pub opcode: &'a str,
    pub field: &'a str,
    /// Explicit rewrite_interface declaration used for value construction.
    pub value_interface: &'a str,
    /// Runtime module implementing the legalization bytecode contract.
    pub runtime: &'a str,
}

pub fn decisions(
    source: &str,
    defs: &Definitions,
    config: DecisionRust<'_>,
) -> Result<String, Error> {
    compile(
        source,
        &crate::syntax::parse(source)?,
        defs,
        config,
        &|offset| format!("line {}", Error::at(source, offset, "").line),
    )
}

impl crate::Source {
    pub fn decisions(
        &self,
        defs: &Definitions,
        config: DecisionRust<'_>,
    ) -> Result<String, crate::SourceError> {
        self.check_imports()?;
        compile(self.text(), self.declarations(), defs, config, &|offset| {
            let location = self.locate(Error::at(self.text(), offset, ""));
            format!("{}:{}", location.path.display(), location.diagnostic.line)
        })
        .map_err(|e| self.locate(e))
    }
}

fn compile(
    source: &str,
    declarations: &[Decl],
    defs: &Definitions,
    config: DecisionRust<'_>,
    location: &dyn Fn(usize) -> String,
) -> Result<String, Error> {
    if !super::identifier(config.function) || !super::identifier(config.dialect) {
        return Err(Error::at(source, 0, "invalid Rust function name"));
    }
    interfaces::rust_path(source, 0, config.opcode)?;
    interfaces::rust_path(source, 0, config.field)?;
    interfaces::rust_path(source, 0, config.runtime)?;
    let bindings = interfaces::Bindings::compile(declarations, source)?;
    let interface = ValueInterface::compile(declarations, source, config, &bindings)?;
    predicate::validate(declarations, source, &bindings, &interface.value)?;

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

    let mut out = interfaces::declarations(declarations, source, "host")?;
    let mut bodies = String::new();
    let mut hosts = BTreeMap::new();
    let mut groups: BTreeMap<String, crate::rules::graph::Candidates> = BTreeMap::new();
    let mut program = Program::default();
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
            writeln!(bodies, "#[allow(non_upper_case_globals)]\nconst REWRITE_{}: {}::Action = {}::Action::Host {{ name: {:?}, apply: {path} }};",
                d.name, config.runtime, config.runtime, d.name).unwrap();
            format!("REWRITE_{}", d.name)
        } else {
            // Resolved after all templates are compiled, regardless of import order.
            format!("REWRITE_{}", d.name)
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
        if !d.name.is_empty() && !names.insert(&d.name) {
            return Err(Error::at(source, d.offset, "duplicate decision rule"));
        }
        if matches!(d.kind, DeclKind::Function { .. }) {
            continue;
        }
        if matches!(d.kind, DeclKind::Rewrite(_)) && d.fields.contains_key("rust") {
            continue;
        }
        let cases = cases(source, d)?;
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
        for case in &cases {
            let emit = &case.body;
            let replacement = case.replacement;
            let label = format!("{} [{}]", location(case.offset), roots.join(" | "));
            let expressions = Expressions {
                source,
                types: &types,
                logical_types: &logical_types,
                bindings: &bindings,
                hosts: &sig.hosts,
                signature: &sig,
                rewrites: &rewrites,
                roots: &roots,
                legal_action: &format!("{}::Action::Legal", config.runtime),
                defs,
            };
            let mut guards = Vec::new();
            let structural = sig.dynamic;
            if !structural {
                let mut sets = |types: Vec<&super::typed::Ty>| -> Result<Vec<usize>, Error> {
                    types
                        .into_iter()
                        .map(|ty| {
                            let values = ty
                                .domain
                                .iter()
                                .map(|n| expressions.constant(n, d.offset))
                                .collect::<Result<Vec<_>, _>>()?;
                            Ok(intern(&mut program.sets, values))
                        })
                        .collect()
                };
                guards.push(Test::Signature {
                    results: sets(sig.results.iter().collect())?,
                    inputs: sets(sig.inputs.iter().map(|(_, ty)| ty).collect())?,
                });
                for name in sig.generics.keys() {
                    let values: Vec<_> = sig
                        .results
                        .iter()
                        .enumerate()
                        .map(|(i, t)| (i * 2 + 1, t))
                        .chain(sig.inputs.iter().enumerate().map(|(i, (_, t))| (i * 2, t)))
                        .filter(|(_, ty)| &ty.name == name)
                        .map(|(i, _)| i)
                        .collect();
                    if values.len() > 1 {
                        guards.push(Test::Same(values));
                    }
                }
            }
            for condition in &case.guards {
                expressions.guard(condition, &mut program, &mut guards)?;
            }
            let mut seen = BTreeSet::new();
            for name in roots.iter().cloned() {
                let op = operations.get(&name).ok_or_else(|| {
                    Error::at(source, d.offset, format!("unknown operation {name}"))
                })?;
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
                    program.recipe(
                        &sig,
                        &insts,
                        &value,
                        &expressions,
                        config,
                        &label,
                        emit.offset,
                    )?
                } else {
                    expressions.rust(emit)?
                };
                if matches!(d.kind, DeclKind::Rewrite(_)) {
                    writeln!(bodies, "#[allow(non_upper_case_globals)]\nconst REWRITE_{}: {}::Action = {action};", d.name, config.runtime).unwrap();
                    continue;
                }
                let action = intern(&mut program.actions, action);
                let tests = guards
                    .iter()
                    .map(|g| intern(&mut program.tests, g.clone()))
                    .collect();
                groups.entry(name).or_default().push((action, tests));
            }
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
    let mut graph = crate::rules::graph::Graph::default();
    let entries: Vec<_> = groups
        .into_iter()
        .map(|(opcode, candidates)| (opcode, graph.compile(candidates)))
        .collect();
    // Recipe labels already occupy the assembler's label namespace.
    let base = program.labels;
    for node in &graph.nodes {
        program.asm.label();
        match node {
            crate::rules::graph::Node::Reject => program.asm.emit(Op::Reject {}),
            crate::rules::graph::Node::Accept(action) => {
                program.asm.emit(Op::Accept { action: *action })
            }
            crate::rules::graph::Node::Check { test, yes, no } => {
                program.test(*test, base + no);
                program
                    .asm
                    .branch(Op::Jump { target: 0 }, "target", base + yes);
            }
        }
    }
    let encoded = program.asm.finish();
    writeln!(
        out,
        "pub fn {}(opcode: {}) -> Option<(&'static {}::Program, usize)> {{",
        config.function, config.opcode, config.runtime
    )
    .unwrap();
    writeln!(out, "let entry = match opcode {{").unwrap();
    for (opcode, entry) in entries {
        writeln!(
            out,
            "{}::{opcode} => {},",
            config.opcode,
            encoded.labels[base + entry]
        )
        .unwrap();
    }
    writeln!(out, "_ => return None, }}; Some((&PROGRAM, entry)) }}").unwrap();
    if !program.predicates.is_empty() {
        writeln!(
            out,
            "#[allow(unused_parens)]\npub fn predicate({args}, id: usize) -> bool {{ match id {{"
        )
        .unwrap();
        for (id, predicate) in program.predicates.iter().enumerate() {
            writeln!(out, "{id} => {{ {predicate} }},").unwrap();
        }
        writeln!(out, "_ => unreachable!(\"legalization predicate\"), }} }}").unwrap();
    }
    program.render(&mut out, &encoded, config, &interface);
    let ty = &bindings
        .0
        .get("Type")
        .ok_or_else(|| Error::at(source, 0, "value rules require a Type binding"))?
        .path;
    out.push_str(&functions.wrappers(ty, bound, &interface.value));
    out.push_str(&bodies);
    Ok(out)
}

/// A candidate has a read-only guard followed by one deferred action. Builds
/// are checked as a value expression, never executed while choosing a case.
struct Case {
    offset: usize,
    guards: Vec<Node>,
    body: Node,
    replacement: bool,
}

fn cases(source: &str, d: &Decl) -> Result<Vec<Case>, Error> {
    if matches!(d.kind, DeclKind::Rewrite(_)) {
        return Ok(vec![Case {
            offset: d.offset,
            guards: Vec::new(),
            body: d.fields["replace"].clone(),
            replacement: true,
        }]);
    }
    let DeclKind::Select(signature) = &d.kind else {
        return Err(Error::at(source, d.offset, "expected select or rewrite"));
    };
    let root = signature
        .params
        .iter()
        .find(|p| !matches!(p.ty.kind, Kind::Ref(_)))
        .ok_or_else(|| Error::at(source, d.offset, "missing instruction parameter"))?;
    let Some(Node {
        kind: Kind::List(candidates),
        ..
    }) = d.fields.get("cases")
    else {
        return Err(Error::at(source, d.offset, "expected selection cases"));
    };
    if candidates.is_empty() {
        return Err(Error::at(
            source,
            d.offset,
            "select requires at least one case",
        ));
    }
    let is_root = |node: &Node| matches!(&node.kind, Kind::Name(name) if name == &root.name);
    candidates.iter().map(|candidate| {
        let Kind::List(statements) = &candidate.kind else { unreachable!() };
        let Some((last, preceding)) = statements.split_last() else {
            return Err(Error::at(source, candidate.offset, "empty selection case"));
        };
        let mut guards = Vec::new();
        let mut builds = Vec::new();
        for statement in preceding {
            match &statement.kind {
                Kind::Call(name, args) if name == "require" && args.len() == 1 && builds.is_empty() => {
                    guards.push(args[0].clone());
                }
                Kind::Let(name, value) => {
                    let value = construction(source, value)?;
                    builds.push(Node { offset: statement.offset, kind: Kind::Let(name.clone(), Box::new(value)) });
                }
                _ => return Err(Error::at(source, statement.offset,
                    "expected require before construction, or let binding; actions must be last")),
            }
        }
        let Kind::Call(name, args) = &last.kind else {
            return Err(Error::at(source, last.offset, "expected legal, replace or rewrite call"));
        };
        let (body, replacement) = match (name.as_str(), args.as_slice()) {
            ("replace", [node, value]) if is_root(node) => {
                // A previously built value is shared, not built again.
                let value = if matches!(value.kind, Kind::Name(_)) {
                    value.clone()
                } else { construction(source, value)? };
                builds.push(value);
                (Node { offset: last.offset, kind: Kind::List(builds) }, true)
            }
            ("legal", [node]) if is_root(node) && builds.is_empty() => {
                (Node { offset: last.offset, kind: Kind::Name("legal".into()) }, false)
            }
            (name, [node]) if !matches!(name, "legal" | "replace" | "require" | "build")
                && is_root(node) && builds.is_empty() => (last.clone(), false),
            _ => return Err(Error::at(source, last.offset,
                "case must end with legal(root), replace(root, value), or rewrite(root); builds require replace")),
        };
        Ok(Case { offset: candidate.offset, guards, body, replacement })
    }).collect()
}

fn construction(source: &str, node: &Node) -> Result<Node, Error> {
    let Kind::Call(name, args) = &node.kind else {
        return Err(Error::at(
            source,
            node.offset,
            "construction requires build(expression)",
        ));
    };
    if name != "build" || args.len() != 1 {
        return Err(Error::at(
            source,
            node.offset,
            "construction requires build(expression)",
        ));
    }
    fn unwrap(node: &Node) -> Node {
        let mut node = node.clone();
        match &mut node.kind {
            Kind::Call(name, args) if name == "build" && args.len() == 1 => {
                return unwrap(&args[0]);
            }
            Kind::Call(_, args) | Kind::TypedCall(_, _, args) => {
                for arg in args {
                    *arg = unwrap(arg);
                }
            }
            _ => {}
        }
        node
    }
    Ok(unwrap(&args[0]))
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
    defs: &'a Definitions,
}
impl Expressions<'_> {
    fn field(&self, receiver: &Node, field: &str, offset: usize) -> Result<String, Error> {
        use crate::storage::operands::Domain;
        if !self
            .hosts
            .iter()
            .any(|(n, ty)| n == "query" && ty == "Query")
        {
            return Err(Error::at(
                self.source,
                offset,
                "field queries require an explicit query: &Query parameter",
            ));
        }
        let node = Node {
            offset,
            kind: Kind::Member(Box::new(receiver.clone()), field.into()),
        };
        let (domain, index) = self.access(&node)?;
        Ok(match domain {
            Domain::Input | Domain::Result => {
                format!("query.value({}, {index})", domain == Domain::Result)
            }
            Domain::Attribute => format!("query.immediate({index})"),
        })
    }

    fn rewrite(&self, name: &str, offset: usize, args: &[Node]) -> Result<String, Error> {
        let [
            Node {
                kind: Kind::Name(node),
                ..
            },
        ] = args
        else {
            return Err(Error::at(
                self.source,
                offset,
                "rewrite requires one matched instruction argument",
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
            Kind::Call(name, args) => self.rewrite(name, node.offset, args)?,
            Kind::Member(receiver, field) => self.field(receiver, field, node.offset)?,
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
                if matches!(
                    declaration.body(),
                    Some(crate::syntax::FunctionBody::Vm { .. })
                ) {
                    return Err(Error::at(
                        self.source,
                        node.offset,
                        "VM method requires native predicate lowering",
                    ));
                }
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
