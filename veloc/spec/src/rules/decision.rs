use super::typed::Signature;
use crate::bytecode::intern;
mod predicate;
mod program;
use crate::{
    Definitions, Error, interfaces,
    syntax::{Decl, DeclKind, Kind, Node},
};
use predicate::{Pattern, Test};
use program::{Output, Program};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use veloc_bytecode::OperandRef;
use veloc_bytecode::rewrite::RawInstruction as Op;
use veloc_bytecode::signature::{PatternHeader, TypePatterns};

#[derive(Clone, Copy)]
pub struct DecisionRust<'a> {
    /// Explicit namespace of the supplied OpSpec definitions.
    pub dialect: &'a str,
    pub function: &'a str,
    pub opcode: &'a str,
    pub field: &'a str,
    /// Rust value type used by the runtime builder.
    pub value: &'a str,
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
    interfaces::rust_path(source, 0, config.value)?;
    predicate::validate(declarations, source, &bindings, config.value)?;

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
    let mut groups: BTreeMap<String, crate::rules::graph::Candidates> = BTreeMap::new();
    let mut program = Program::default();
    let mut rewrites = BTreeMap::new();
    for d in declarations {
        if !matches!(d.kind, DeclKind::Rewrite(_)) {
            continue;
        }
        if d.fields.len() != 1 || !d.fields.contains_key("replace") {
            return Err(Error::at(
                source,
                d.offset,
                "rewrite requires a replace body",
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
        // Resolved after all templates are compiled, regardless of import order.
        let action = format!("REWRITE_{}", d.name);
        if rewrites
            .insert(d.name.clone(), (sig, roots, action))
            .is_some()
        {
            return Err(Error::at(source, d.offset, "duplicate rewrite"));
        }
    }
    let mut names = BTreeSet::new();
    for d in declarations {
        if matches!(d.kind, DeclKind::Type { .. } | DeclKind::TypeSet(_)) {
            continue;
        }
        if !d.name.is_empty() && !names.insert(&d.name) {
            return Err(Error::at(source, d.offset, "duplicate decision rule"));
        }
        if matches!(d.kind, DeclKind::Function { .. }) {
            continue;
        }
        let cases = cases(source, d)?;
        let (sig, roots) =
            Signature::parse(source, d, &aliases, &operations, defs, config.dialect)?;
        for (_, owner) in &sig.hosts {
            if !types.contains_key(owner.as_str()) {
                return Err(Error::at(
                    source,
                    d.offset,
                    format!("unknown host type {owner}"),
                ));
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
                value_type: config.value,
                signature: &sig,
                rewrites: &rewrites,
                roots: &roots,
                legal_action: &format!("{}::Action::Legal", config.runtime),
                defs,
            };
            let mut guards = Vec::new();
            let structural = sig.dynamic;
            if !structural {
                guards.push(Program::signature(&sig, &expressions, d.offset)?);
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
                // Structural patterns are checked through their named fields;
                // positional value patterns require a signature check.
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
                let update = match &emit.kind {
                    Kind::List(nodes) => nodes
                        .last()
                        .is_some_and(|n| matches!(n.kind, Kind::Object(..))),
                    Kind::Object(..) => true,
                    _ => false,
                };
                let action = if replacement && update {
                    program.update_recipe(
                        &sig,
                        emit,
                        &expressions,
                        &functions,
                        &operations,
                        config,
                        &label,
                    )?
                } else if replacement {
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
                        Output::Value(value),
                        &expressions,
                        config,
                        &label,
                        emit.offset,
                    )?
                } else if matches!(&emit.kind, Kind::Call(name, _) if name == "libcall") {
                    libcall(source, emit, &sig, op, config, &label)?
                } else {
                    expressions.action(emit)?
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
        "static ENTRIES: &[Option<usize>] = &{{ let mut entries = [None; {}];",
        defs.ops.len()
    )
    .unwrap();
    for (opcode, entry) in entries {
        writeln!(
            out,
            "entries[{}::{opcode} as usize] = Some({});",
            config.opcode,
            encoded.labels[base + entry]
        )
        .unwrap();
    }
    out.push_str("entries };\n");
    program.render(&mut out, &encoded, config);
    out.push_str(&bodies);
    Ok(out)
}

/// Libcalls preserve the complete value signature of a concrete root. Keep
/// symbol selection in the rule and ABI construction in the runtime service.
fn libcall(
    source: &str,
    node: &Node,
    sig: &Signature,
    op: &crate::schema::Operation,
    config: DecisionRust<'_>,
    name: &str,
) -> Result<String, Error> {
    let Kind::Call(_, args) = &node.kind else {
        unreachable!()
    };
    let Kind::Text(symbol) = &args[1].kind else {
        return Err(Error::at(
            source,
            args[1].offset,
            "libcall requires a literal symbol name",
        ));
    };
    if symbol.is_empty() || symbol.contains('\0') {
        return Err(Error::at(
            source,
            args[1].offset,
            "invalid libcall symbol name",
        ));
    }
    if let Err(message) = &op.signature {
        return Err(Error::at(
            source,
            node.offset,
            format!("libcall requires a value operation without effects or attributes: {message}"),
        ));
    }
    if sig.dynamic
        || sig.results.is_empty()
        || sig
            .inputs
            .iter()
            .map(|(_, ty)| ty)
            .chain(&sig.results)
            .any(|ty| ty.domain.len() != 1)
    {
        return Err(Error::at(
            source,
            node.offset,
            "libcall requires a fixed signature with concrete input and result types",
        ));
    }
    Ok(format!(
        "{}::Action::Libcall {{ name: {name:?}, symbol: {symbol:?} }}",
        config.runtime
    ))
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
            ("libcall", [node, _]) if is_root(node) && builds.is_empty() => {
                (last.clone(), false)
            }
            (name, [node]) if !matches!(name, "legal" | "replace" | "require" | "build" | "libcall")
                && is_root(node) && builds.is_empty() => (last.clone(), false),
            _ => return Err(Error::at(source, last.offset,
                "case must end with legal(root), replace(root, value), libcall(root, symbol), or rewrite(root); builds require replace")),
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
    value_type: &'a str,
    signature: &'a Signature,
    defs: &'a Definitions,
}
impl Expressions<'_> {
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
    fn action(&self, node: &Node) -> Result<String, Error> {
        match &node.kind {
            Kind::Name(name) if name == "legal" => Ok(self.legal_action.into()),
            Kind::Call(name, args) => self.rewrite(name, node.offset, args),
            _ => Err(Error::at(
                self.source,
                node.offset,
                "expected legal or a declared rewrite",
            )),
        }
    }
}
