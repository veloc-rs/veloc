//! Directed, single-instruction rewrites. Matching is read-only; a successful
//! replacement preserves the root's position and SSA results. Nested operations
//! are inspected but never erased. Expressions and storage mappings are shared
//! with the other spec consumers.
use crate::model::{Op, ParamKind, expr::Emitter};
use crate::syntax::{Kind, Node};
use crate::{Definitions, Error, Source, SourceError};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write,
};

pub(super) fn generate(
    source: &Source,
    defs: &Definitions,
    dialect: &str,
    rust: &str,
    cases: &[&super::optimizer::Case<'_>],
) -> Result<String, SourceError> {
    let fail = |at, message| source.locate(Error::at(source.text(), at, message));
    let mut expressions = source.expressions()?;
    let eligible: BTreeSet<_> = defs
        .operations()
        .filter(|op| op.expression.is_some())
        .map(|op| op.name)
        .collect();
    let mut groups = BTreeMap::<String, Vec<String>>::new();
    let mut functions = String::new();
    let mut next_rule = 0;
    for case in cases {
        let mut compiler = Compiler {
            source,
            defs,
            dialect,
            eligible: &eligible,
            expressions: &mut expressions,
            generics: &case.generics,
            bound_types: BTreeMap::new(),
            bindings: BTreeMap::new(),
            edges: BTreeMap::new(),
            next: 0,
            code: String::new(),
        };
        let args = case.args;
        let root_op = compiler.operation(case.operation, args.len(), case.offset)?;
        let result_count = root_op.signature.results.patterns().unwrap().len();
        for results in &case.results {
            compiler.result_types(Some(results), result_count, true, case.offset)?;
        }
        compiler.pattern(root_op, args, "inst")?;
        if let Some(guard) = case.guard {
            let guard = compiler.expression(guard, "bool")?;
            writeln!(compiler.code, "if !({guard}) {{ return None; }}").unwrap();
        }
        let (replacement, results) = super::optimizer::annotation(case.replacement);
        let Kind::Call(path, args) = &replacement.kind else {
            return Err(fail(
                replacement.offset,
                "replacement must be one operation",
            ));
        };
        let target = compiler.operation(path, args.len(), replacement.offset)?;
        compiler.result_types(results, result_count, false, replacement.offset)?;
        if target.signature.results.patterns().unwrap().len() != result_count {
            return Err(fail(
                replacement.offset,
                "replacement must preserve result arity",
            ));
        }
        next_rule += 1;
        let function = format!(
            "match_{}_rule_{next_rule}",
            crate::storage::constructor_name(&root_op.name)
        );
        groups
            .entry(root_op.name.clone())
            .or_default()
            .push(function.clone());
        if has_successors(root_op) || has_successors(target) {
            let body = compiler.control_replacement(root_op, target, args, replacement.offset)?;
            emit_rule(&mut functions, &function, &body);
            continue;
        }
        let mut inputs = BTreeMap::new();
        let mut values = Vec::new();
        for (param, arg) in target.params.iter().zip(args) {
            let ty = match &param.kind {
                ParamKind::Value => "Value",
                ParamKind::Property(ty) => ty,
                _ => unreachable!("checked fixed operation"),
            };
            let value = compiler.expression(arg, ty)?;
            let name = format!("new_{}", param.name);
            writeln!(compiler.code, "let {name} = {value};").unwrap();
            if param.kind == ParamKind::Value {
                values.push(format!("dfg.value_type({name})"));
            }
            inputs.insert(param.name.clone(), name);
        }
        if root_op.name == target.name {
            let unchanged = target
                .params
                .iter()
                .map(|p| {
                    let original = root_op.inputs[&p.name]
                        .emit(&|f| format!("root_{f}"), &|v| format!("{v}?"));
                    format!("{} == {original}", inputs[&p.name])
                })
                .collect::<Vec<_>>()
                .join(" && ");
            writeln!(
                compiler.code,
                "if {} {{ return None; }}",
                if unchanged.is_empty() {
                    "true"
                } else {
                    &unchanged
                }
            )
            .unwrap();
        }
        writeln!(
            compiler.code,
            "// Check the replacement against the existing results.\nlet operand_types = [{}];",
            values.join(", ")
        )
        .unwrap();
        let results = (0..result_count)
            .map(|i| format!("dfg.value_type(dfg.inst_results(inst)[{i}])"))
            .collect::<Vec<_>>();
        writeln!(
            compiler.code,
            "let result_types = [{}];\nOpcode::{}.validate_types(&operand_types, &result_types).ok()?;",
            results.join(", "),
            target.name
        )
        .unwrap();
        let mut emitter = Emitter::types(target, inputs.clone(), "operand_types", "result_types");
        emitter.dfg = "dfg";
        for constraint in &target.constraints {
            if !constraint.type_only || constraint.binding.is_some() {
                compiler
                    .code
                    .push_str(&constraint.emit(&emitter, "return None"));
            }
        }
        let fields = crate::generate::fields::constructor(
            defs,
            target,
            &inputs,
            "InstFields",
            &format!("Opcode::{}", target.name),
        );
        let operands = target
            .params
            .iter()
            .filter(|p| p.kind == ParamKind::Value)
            .map(|p| inputs[&p.name].as_str())
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            compiler.code,
            "Some(InstRewrite::Replace {{ fields: {fields}, args: smallvec::smallvec![{operands}] }})"
        )
        .unwrap();
        emit_rule(&mut functions, &function, &compiler.code);
    }
    let mut code = format!(
        "// @generated directed instruction rewrites.\n#[allow(unused_imports)] use {rust}::*;\n#[allow(unused_imports)] use {rust}::inst::*;\n"
    );
    writeln!(code, "pub(crate) fn find_rewrite(func: &FuncBody, inst: Inst) -> Option<InstRewrite> {{ match func.dfg().inst(inst).opcode() {{").unwrap();
    for (op, rules) in groups {
        let calls = rules
            .iter()
            .map(|name| format!("{name}(func, inst)"))
            .reduce(|previous, next| format!("{previous}.or_else(|| {next})"))
            .unwrap();
        writeln!(code, "Opcode::{op} => {calls},").unwrap();
    }
    code.push_str("_ => None,\n} }\n");
    code.push_str(&functions.replace("crate::", &format!("{rust}::")));
    Ok(code)
}

/// Match and validate without borrowing the function into the returned edit.
fn emit_rule(out: &mut String, name: &str, body: &str) {
    writeln!(out, "#[allow(unused_variables, unused_parens)]\nfn {name}(func: &FuncBody, inst: Inst) -> Option<InstRewrite> {{\nlet dfg = func.dfg();\n{body}\n}}\n").unwrap();
}

fn has_successors(op: &Op) -> bool {
    op.params
        .iter()
        .any(|p| matches!(p.kind, ParamKind::Successor | ParamKind::Successors))
}

struct Compiler<'a, 'e> {
    source: &'a Source,
    defs: &'a Definitions,
    dialect: &'a str,
    eligible: &'a BTreeSet<String>,
    expressions: &'e mut crate::schema::Expressions<'a>,
    generics: &'e BTreeMap<String, crate::types::TypeSet>,
    bound_types: BTreeMap<String, String>,
    bindings: BTreeMap<String, (Node, String)>,
    edges: BTreeMap<String, EdgeCapture>,
    next: usize,
    code: String,
}

#[derive(Clone)]
enum EdgeIndex {
    Fixed(usize),
    Dynamic(String),
}

impl EdgeIndex {
    fn expression(&self) -> String {
        match self {
            Self::Fixed(index) => format!("{index}usize"),
            Self::Dynamic(expression) => expression.clone(),
        }
    }
}

struct EdgeCapture {
    index: EdgeIndex,
    value: String,
    multiple: bool,
}

impl<'a> Compiler<'a, '_> {
    fn check_type(&mut self, node: &Node, actual: &str, bind: bool) -> Result<(), SourceError> {
        let expected = match &node.kind {
            Kind::Name(name) if name == "_" => return Ok(()),
            Kind::Name(name) if self.bound_types.contains_key(name) => {
                self.bound_types[name].clone()
            }
            Kind::Name(name) if bind && self.generics.contains_key(name) => {
                let ty = format!("type_{name}");
                let accepts = crate::types::generate::accepts(&self.generics[name], &ty);
                writeln!(
                    self.code,
                    "let {ty} = {actual};\nif !({accepts}) {{ return None; }}"
                )
                .unwrap();
                self.bound_types.insert(name.clone(), ty.clone());
                self.bindings.insert(
                    name.clone(),
                    (
                        Node {
                            offset: node.offset,
                            kind: Kind::Name("Type".into()),
                        },
                        ty,
                    ),
                );
                return Ok(());
            }
            Kind::Name(name)
                if self
                    .defs
                    .type_domain(name)
                    .is_some_and(|d| d.is_singleton()) =>
            {
                name.clone()
            }
            _ => return Err(self.fail(node.offset, "expected an exact or bound result type")),
        };
        writeln!(self.code, "if {actual} != {expected} {{ return None; }}").unwrap();
        Ok(())
    }

    fn result_types(
        &mut self,
        types: Option<&[Node]>,
        count: usize,
        bind: bool,
        at: usize,
    ) -> Result<(), SourceError> {
        if let Some(types) = types {
            if types.len() != count {
                return Err(self.fail(at, "result annotation must match instruction result arity"));
            }
            for (i, ty) in types.iter().enumerate() {
                self.check_type(
                    ty,
                    &format!("dfg.value_type(dfg.inst_results(inst)[{i}])"),
                    bind,
                )?;
            }
        }
        Ok(())
    }

    fn capture_edge(
        &mut self,
        arg: &Node,
        prefix: &str,
        capture: EdgeCapture,
    ) -> Result<(), SourceError> {
        let Kind::Name(name) = &arg.kind else {
            return Err(self.fail(arg.offset, "successors must be captured whole"));
        };
        if prefix != "root"
            || !super::identifier(name)
            || self.bindings.contains_key(name)
            || self.generics.contains_key(name)
        {
            return Err(self.fail(
                arg.offset,
                "successor captures belong to the root and cannot shadow values",
            ));
        }
        if name == "_" {
            return Ok(());
        }
        if let Some(previous) = self.edges.get(name) {
            if previous.multiple != capture.multiple {
                return Err(self.fail(
                    arg.offset,
                    "successor and successor-list captures cannot share a name",
                ));
            }
            // Equality includes both destination and the complete argument slice.
            let value = &capture.value;
            let previous = &previous.value;
            let equal = if capture.multiple {
                format!("{value}.iter().eq({previous}.iter())")
            } else {
                format!("{value} == {previous}")
            };
            writeln!(self.code, "if !({equal}) {{ return None; }}").unwrap();
        } else {
            self.edges.insert(name.clone(), capture);
        }
        Ok(())
    }

    fn edge(&self, node: &Node) -> Result<&EdgeCapture, SourceError> {
        if let Kind::Name(name) = &node.kind
            && let Some(edge) = self.edges.get(name)
        {
            return Ok(edge);
        }
        Err(self.fail(
            node.offset,
            "replacement successors must be captured root edges",
        ))
    }

    /// Carry original edge indices through selection; never rebuild successors.
    fn selected_edge(&mut self, node: &Node) -> Result<EdgeIndex, SourceError> {
        match &node.kind {
            Kind::Method(receiver, method, args) if method == "unwrap_or" && args.len() == 1 => {
                let optional = self.optional_edge(receiver)?;
                let fallback = self.selected_edge(&args[0])?.expression();
                Ok(EdgeIndex::Dynamic(format!(
                    "({optional}).unwrap_or({fallback})"
                )))
            }
            Kind::Try(receiver) => Ok(EdgeIndex::Dynamic(format!(
                "({})?",
                self.optional_edge(receiver)?
            ))),
            _ => {
                let edge = self.edge(node)?;
                if edge.multiple {
                    return Err(self.fail(node.offset, "select one successor with get(index)"));
                }
                Ok(edge.index.clone())
            }
        }
    }

    fn optional_edge(&mut self, node: &Node) -> Result<String, SourceError> {
        if let Kind::Method(receiver, method, args) = &node.kind
            && method == "get"
            && let [index] = args.as_slice()
        {
            let edge = self.edge(receiver)?;
            if !edge.multiple {
                return Err(self.fail(receiver.offset, "get requires a captured successor list"));
            }
            let start = edge.index.clone();
            let list = edge.value.clone();
            let value = self.expression(index, "u64")?;
            let index = format!("edge_index_{}", self.next);
            self.next += 1;
            // A missing constant rejects the rule before an out-of-range lookup
            // can fall back to the default edge. Compare before narrowing to usize.
            writeln!(self.code, "let {index} = {value};").unwrap();
            let selected = match start {
                EdgeIndex::Fixed(0) => format!("{index} as usize"),
                start => format!("{} + {index} as usize", start.expression()),
            };
            return Ok(format!(
                "({index} < {list}.len() as u64).then(|| {selected})"
            ));
        }
        Err(self.fail(
            node.offset,
            "expected a successor-list get(index) selection",
        ))
    }

    fn control_replacement(
        &mut self,
        root: &Op,
        target: &Op,
        args: &[Node],
        at: usize,
    ) -> Result<String, SourceError> {
        if !matches!(root.name.as_str(), "Br" | "BrTable") {
            return Err(self.fail(at, "control rewrites require a conditional root"));
        }
        let edge = |arg: &Node| -> Result<usize, SourceError> {
            let capture = self.edge(arg)?;
            if !capture.multiple
                && let EdgeIndex::Fixed(index) = capture.index
            {
                return Ok(index);
            }
            Err(self.fail(
                arg.offset,
                "replacement successors must be captured root edges",
            ))
        };
        let rewrite = match target.name.as_str() {
            "Jump" => {
                let index = match self.selected_edge(&args[0])? {
                    EdgeIndex::Fixed(index) => index.to_string(),
                    EdgeIndex::Dynamic(index) => {
                        format!("({index}).try_into().expect(\"too many successors\")")
                    }
                };
                format!("Some(InstRewrite::SelectEdge({index}))")
            }
            "Br" if target.name == root.name => {
                let yes = edge(&args[1])?;
                let no = edge(&args[2])?;
                let swap_edges = match (yes, no) {
                    (0, 1) => false,
                    (1, 0) => true,
                    _ => {
                        return Err(self.fail(
                            at,
                            "branch replacements may only preserve or swap the original edges",
                        ));
                    }
                };
                let condition = self.expression(&args[0], "Value")?;
                writeln!(self.code, "let condition = {condition};\nif dfg.value_type(condition) != Type::BOOL {{ return None; }}").unwrap();
                let original = |index: usize| {
                    root.inputs[&root.params[index].name]
                        .emit(&|f| format!("root_{f}"), &|v| format!("{v}?"))
                };
                let unchanged = if !swap_edges {
                    format!("condition == {}", original(0))
                } else {
                    format!(
                        "condition == {} && {} == {}",
                        original(0),
                        original(1),
                        original(2)
                    )
                };
                writeln!(self.code, "if {unchanged} {{ return None; }}").unwrap();
                format!("Some(InstRewrite::Branch {{ condition, swap_edges: {swap_edges} }})")
            }
            _ => {
                return Err(self.fail(
                    at,
                    "control replacements must preserve the branch or select one original edge",
                ));
            }
        };
        Ok(format!("{}\n{rewrite}", self.code))
    }

    fn fail(&self, at: usize, message: &str) -> SourceError {
        self.source
            .locate(Error::at(self.source.text(), at, message))
    }
    fn operation(&self, path: &str, arity: usize, at: usize) -> Result<&'a Op, SourceError> {
        let op = super::optimizer::operation(
            self.source.text(),
            self.defs,
            self.dialect,
            path,
            arity,
            at,
        )
        .map_err(|error| self.source.locate(error))?;
        if has_successors(op) {
            self.check_control(op, at)?;
            return Ok(op);
        }
        if !self.eligible.contains(&op.name)
            || !matches!(op.projection, crate::model::Projection::Packed(_))
            || op.signature.results.patterns().is_none()
            || op
                .params
                .iter()
                .any(|p| !matches!(p.kind, ParamKind::Value | ParamKind::Property(_)))
            || op.traits.contains("TERMINATOR")
            || op
                .constraints
                .iter()
                .any(|c| c.condition.context_type().is_some())
        {
            return Err(self.fail(
                at,
                "instruction rewrite requires fixed operands and context-free constraints",
            ));
        }
        Ok(op)
    }

    /// These rules lower to MIR editor operations. Check their signatures and
    /// storage order here so selected indices still name the original edges.
    fn check_control(&self, op: &Op, at: usize) -> Result<(), SourceError> {
        use crate::model::{Binding, Pattern, Projection, TypeList};
        use ParamKind::{Successor, Successors, Value};
        let (kinds, selector): (&[_], _) = match op.name.as_str() {
            "Jump" => (&[Successor], None),
            "Br" => (&[Value, Successor, Successor], Some("Type::BOOL")),
            "BrTable" => (&[Value, Successors, Successor], Some("Type::I32")),
            _ => return Err(self.fail(at, "control rewrites support Jump, Br and BrTable")),
        };
        let operands_match = match (&op.signature.operands, selector) {
            (TypeList::Variadic(patterns), Some(selector)) => {
                matches!(patterns.as_slice(), [Pattern::Exact(ty)] if ty == selector)
            }
            (TypeList::Variadic(patterns), None) => patterns.is_empty(),
            _ => false,
        };
        if !operands_match
            || op.signature.results.patterns() != Some(&[][..])
            || !op.params.iter().map(|p| &p.kind).eq(kinds.iter())
            || op.params.iter().any(|p| p.moves)
            || !op.constraints.is_empty()
        {
            return Err(self.fail(
                at,
                "control rewrite requires the MIR jump signature without additional constraints",
            ));
        }
        let Projection::Packed(bindings) = &op.projection else {
            return Err(self.fail(at, "control rewrites require packed MIR storage"));
        };
        let layout = self
            .defs
            .storage
            .formats
            .iter()
            .find(|f| f.name == op.format)
            .unwrap();
        let mut order = Vec::new();
        for field in &layout.fields {
            match bindings.get(&field.name) {
                Some(Binding::Name(name)) => order.push(name.as_str()),
                Some(Binding::Table { cases, default }) => {
                    order.push(cases);
                    order.push(default);
                }
                _ => return Err(self.fail(at, "control storage must preserve parameter order")),
            }
        }
        if !order
            .into_iter()
            .eq(op.params.iter().map(|p| p.name.as_str()))
        {
            return Err(self.fail(at, "control storage must preserve parameter order"));
        }
        Ok(())
    }

    fn pattern(&mut self, op: &Op, args: &[Node], inst: &str) -> Result<(), SourceError> {
        let id = self.next;
        self.next += 1;
        let prefix = if id == 0 {
            "root".to_owned()
        } else {
            format!("{}_{id}", crate::storage::constructor_name(&op.name))
        };
        let format = self
            .defs
            .storage
            .formats
            .iter()
            .find(|f| f.name == op.format)
            .unwrap();
        let fields = format
            .fields
            .iter()
            .map(|f| {
                if f.ty.named("Opcode") {
                    format!("{}: Opcode::{}", f.name, op.name)
                } else {
                    format!("{}: {prefix}_{}", f.name, f.name)
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            self.code,
            "let InstView::{} {{ {fields} }} = dfg.inst({inst}) else {{ return None; }};",
            op.format
        )
        .unwrap();
        let mut edge_index = EdgeIndex::Fixed(0);
        for (param, arg) in op.params.iter().zip(args) {
            let mut value =
                op.inputs[&param.name].emit(&|f| format!("{prefix}_{f}"), &|v| format!("{v}?"));
            if matches!(param.kind, ParamKind::Successor | ParamKind::Successors) {
                let multiple = param.kind == ParamKind::Successors;
                if multiple {
                    let list = format!("edges_{id}");
                    writeln!(self.code, "let {list} = {value};").unwrap();
                    value = list;
                }
                self.capture_edge(
                    arg,
                    &prefix,
                    EdgeCapture {
                        index: edge_index.clone(),
                        value: value.clone(),
                        multiple,
                    },
                )?;
                edge_index = match (multiple, edge_index) {
                    (false, EdgeIndex::Fixed(index)) => EdgeIndex::Fixed(index + 1),
                    (true, EdgeIndex::Fixed(0)) => EdgeIndex::Dynamic(format!("{value}.len()")),
                    (multiple, index) => EdgeIndex::Dynamic(format!(
                        "({}) + {}",
                        index.expression(),
                        if multiple {
                            format!("{value}.len()")
                        } else {
                            "1".into()
                        }
                    )),
                };
                continue;
            }
            let ty = match &param.kind {
                ParamKind::Value => "Value",
                ParamKind::Property(ty) => ty,
                _ => unreachable!(),
            };
            let mut arg = arg;
            while let Kind::Typed(inner, types) = &arg.kind {
                arg = inner;
                let types = types.as_slice();
                let [annotation] = types else {
                    return Err(self.fail(arg.offset, "operand pattern requires one type"));
                };
                if param.kind != ParamKind::Value {
                    return Err(self.fail(arg.offset, "type annotations require an SSA operand"));
                }
                self.check_type(annotation, &format!("dfg.value_type({value})"), true)?;
            }
            if let Kind::Name(name) = &arg.kind {
                if param.kind == ParamKind::Value && matches!(name.as_str(), "true" | "false") {
                    let negate = if name == "true" { "!" } else { "" };
                    writeln!(
                        self.code,
                        "if {negate}dfg.as_scalar_const({value})?.as_bool()? {{ return None; }}"
                    )
                    .unwrap();
                    continue;
                }
                if self.generics.contains_key(name) {
                    return Err(self.fail(arg.offset, "value shadows a type parameter"));
                }
                if name == "_" {
                    continue;
                }
                if super::identifier(name)
                    && !matches!(name.as_str(), "true" | "false" | "none")
                    && !self.bindings.contains_key(name)
                    && !self.edges.contains_key(name)
                {
                    let capture = format!("capture_{name}");
                    writeln!(self.code, "let {capture} = {value};").unwrap();
                    self.bindings.insert(
                        name.clone(),
                        (
                            Node {
                                offset: arg.offset,
                                kind: Kind::Name(ty.into()),
                            },
                            capture,
                        ),
                    );
                    continue;
                }
            }
            if let Kind::Call(path, nested) = &arg.kind
                && super::optimizer::lookup(self.defs, self.dialect, path).is_some()
                && param.kind == ParamKind::Value
            {
                let child = self.operation(path, nested.len(), arg.offset)?;
                if child.signature.results.patterns().unwrap().len() != 1 {
                    return Err(self.fail(arg.offset, "nested operation must have one result"));
                }
                let name = format!(
                    "{}_inst_{}",
                    crate::storage::constructor_name(&child.name),
                    self.next
                );
                writeln!(self.code, "let {name} = dfg.value_inst({value})?;").unwrap();
                self.pattern(child, nested, &name)?;
            } else {
                let expected = self.expression(arg, ty)?;
                writeln!(self.code, "if {value} != ({expected}) {{ return None; }}").unwrap();
            }
        }
        Ok(())
    }
    fn expression(&mut self, node: &Node, ty: &str) -> Result<String, SourceError> {
        let (node, annotation) = super::optimizer::annotation(node);
        if let Some(types) = annotation {
            let [annotation] = types else {
                return Err(self.fail(node.offset, "replacement operand requires one type"));
            };
            if ty != "Value" {
                return Err(self.fail(node.offset, "type annotations require an SSA operand"));
            }
            let value = self.expression(node, ty)?;
            self.check_type(annotation, &format!("dfg.value_type({value})"), false)?;
            return Ok(value);
        }
        let mut node = node.clone();
        self.constants(&mut node)?;
        self.expressions.rust(&node, ty, &self.bindings, "")
    }
    // Constant facts are supplied by the host; arithmetic and property access
    // then use the ordinary typed expression checker and emitter.
    fn constants(&mut self, node: &mut Node) -> Result<(), SourceError> {
        if let Kind::Call(name, args) = &node.kind
            && name == "constant_bits"
        {
            let [
                Node {
                    kind: Kind::Name(value),
                    ..
                },
            ] = args.as_slice()
            else {
                return Err(self.fail(node.offset, "constant_bits requires a captured value"));
            };
            let Some((ty, value)) = self.bindings.get(value) else {
                return Err(self.fail(node.offset, "unknown constant value"));
            };
            if !matches!(&ty.kind, Kind::Name(name) if name == "Value") {
                return Err(self.fail(node.offset, "constant_bits requires an SSA value"));
            }
            let name = loop {
                let name = format!("__constant{}", self.next);
                self.next += 1;
                if !self.bindings.contains_key(&name) {
                    break name;
                }
            };
            self.bindings.insert(
                name.clone(),
                (
                    Node {
                        offset: node.offset,
                        kind: Kind::Name("u64".into()),
                    },
                    format!("dfg.as_scalar_const({value})?.to_bits()"),
                ),
            );
            node.kind = Kind::Name(name);
            return Ok(());
        }
        crate::syntax::expand::children(node, &mut |node| self.constants(node))
    }
}
