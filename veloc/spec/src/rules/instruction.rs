//! Directed, single-instruction rewrites. Matching is read-only; a successful
//! replacement preserves the root's position and SSA results. Nested operations
//! are inspected but never erased. Expressions and storage mappings are shared
//! with the other spec consumers.
use crate::model::{Op, ParamKind, expr::Emitter};
use crate::syntax::{DeclKind, Kind, Node, Results};
use crate::{Definitions, Error, Source, SourceError};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write,
};

pub(crate) fn generate(
    source: &Source,
    defs: &Definitions,
    dialect: &str,
    rust: &str,
) -> Result<String, SourceError> {
    let fail = |at, message| source.locate(Error::at(source.text(), at, message));
    if !super::identifier(dialect) || !rust.split("::").all(super::identifier) {
        return Err(fail(0, "invalid instruction rewrite binding"));
    }
    let mut expressions = source.expressions()?;
    let eligible: BTreeSet<_> = defs
        .operations()
        .filter(|op| op.expression.is_some())
        .map(|op| op.name)
        .collect();
    let mut groups = BTreeMap::<String, String>::new();
    for decl in source.declarations() {
        let DeclKind::Rule(sig) = &decl.kind else {
            continue;
        };
        let [root] = sig.params.as_slice() else {
            return Err(fail(decl.offset, "instruction rule requires one root"));
        };
        if root.moves
            || !sig.generics.is_empty()
            || !matches!(&sig.results, Results::Fixed(r) if r.is_empty())
            || decl.fields.keys().any(|k| k != "cases")
        {
            return Err(fail(
                decl.offset,
                "instruction rules preserve the root's result types",
            ));
        }
        let Kind::Name(path) = &root.ty.kind else {
            return Err(fail(
                root.ty.offset,
                "expected an unparameterized root operation",
            ));
        };
        let Some(Node {
            kind: Kind::List(cases),
            ..
        }) = decl.fields.get("cases")
        else {
            return Err(fail(decl.offset, "expected instruction cases"));
        };
        if cases.is_empty() {
            return Err(fail(decl.offset, "expected nonempty cases"));
        }
        for case in cases {
            let mut compiler = Compiler {
                source,
                defs,
                dialect,
                eligible: &eligible,
                expressions: &mut expressions,
                bindings: BTreeMap::new(),
                next: 0,
                code: String::new(),
            };
            let Kind::Record(fields) = &case.kind else {
                unreachable!("parsed case")
            };
            let Kind::List(args) = &fields["match"].kind else {
                unreachable!("parsed pattern")
            };
            let root_op = compiler.operation(path, args.len(), case.offset)?;
            compiler.pattern(root_op, args, "inst")?;
            if compiler.bindings.contains_key(&root.name) {
                return Err(fail(case.offset, "pattern binding shadows the root"));
            }
            if let Some(guard) = fields.get("when") {
                let guard = compiler.expression(guard, "bool")?;
                writeln!(compiler.code, "if !({guard}) {{ return None; }}").unwrap();
            }
            let replacement = &fields["emit"];
            let Kind::Call(path, args) = &replacement.kind else {
                return Err(fail(
                    replacement.offset,
                    "replacement must be one operation",
                ));
            };
            let target = compiler.operation(path, args.len(), replacement.offset)?;
            let result_count = root_op.signature.results.patterns().unwrap().len();
            if target.signature.results.patterns().unwrap().len() != result_count {
                return Err(fail(
                    replacement.offset,
                    "replacement must preserve result arity",
                ));
            }
            // Effectful roots can change operands, but cannot change their opcode.
            // Rule authors still own the equivalence of the old and new address.
            if root_op.name != target.name {
                writeln!(compiler.code, "const {{ assert!(Opcode::{}.spec().is_pure() && Opcode::{}.spec().is_pure() && !Opcode::{}.transfers_ownership() && !Opcode::{}.transfers_ownership(), \"opcode-changing rewrites require pure operations\"); }}", root_op.name, target.name, root_op.name, target.name).unwrap();
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
            writeln!(compiler.code, "// Check the replacement before editing the instruction.\nlet operand_types = [{}];", values.join(", ")).unwrap();
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
            let mut emitter =
                Emitter::types(target, inputs.clone(), "operand_types", "result_types");
            emitter.dfg = "dfg";
            for constraint in &target.constraints {
                if !constraint.type_only || constraint.binding.is_some() {
                    compiler
                        .code
                        .push_str(&constraint.emit(&emitter, "return None"));
                }
            }
            let args = target
                .params
                .iter()
                .map(|p| inputs[&p.name].as_str())
                .collect::<Vec<_>>()
                .join(", ");
            // Keep single-field replacements as tuples; empty replacements use ().
            let tuple = if args.is_empty() {
                "()".to_owned()
            } else {
                format!("({args},)")
            };
            writeln!(compiler.code, "Some({tuple})").unwrap();
            let constructor = crate::model::access::constructor(
                target,
                &defs.storage,
                &inputs,
                &format!("Opcode::{}", target.name),
                "writer",
            );
            writeln!(
                groups.entry(root_op.name.clone()).or_default(),
                "let replacement = (|| {{ let dfg = func.dfg(); {} }})();\nif let Some({tuple}) = replacement {{\nfunc.edit().replace_inst(inst, |writer| {constructor});\nreturn true;\n}}",
                compiler.code
            )
            .unwrap();
        }
    }
    let mut code = format!(
        "// @generated directed instruction rewrites.\n#[allow(unused_imports)] use {rust}::*;\n#[allow(unused_imports)] use {rust}::inst::*;\n#[allow(unused_variables)]\npub(super) fn rewrite(func: &mut FuncBody, inst: Inst) -> bool {{\nmatch func.dfg().inst(inst).opcode() {{\n"
    );
    for (op, body) in groups {
        // Definition-owned Rust bindings are relative to the IR crate. Rebase
        // before inserting the configured namespace itself into generated code.
        let body = body.replace("crate::", &format!("{rust}::"));
        writeln!(code, "Opcode::{op} => {{ {body} false }},").unwrap();
    }
    code.push_str("_ => false,\n}\n}\n");
    Ok(code)
}

struct Compiler<'a, 'e> {
    source: &'a Source,
    defs: &'a Definitions,
    dialect: &'a str,
    eligible: &'a BTreeSet<String>,
    expressions: &'e mut crate::schema::Expressions<'a>,
    bindings: BTreeMap<String, (Node, String)>,
    next: usize,
    code: String,
}
impl<'a> Compiler<'a, '_> {
    fn fail(&self, at: usize, message: &str) -> SourceError {
        self.source
            .locate(Error::at(self.source.text(), at, message))
    }
    fn operation(&self, path: &str, arity: usize, at: usize) -> Result<&'a Op, SourceError> {
        let op = path
            .strip_prefix(&format!("{}::", self.dialect))
            .and_then(|name| self.defs.ops.iter().find(|op| op.name == name))
            .ok_or_else(|| self.fail(at, "unknown instruction operation"))?;
        if !self.eligible.contains(&op.name)
            || !matches!(op.projection, crate::model::Projection::Packed(_))
            || op.params.len() != arity
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
    fn pattern(&mut self, op: &Op, args: &[Node], inst: &str) -> Result<(), SourceError> {
        let id = self.next;
        self.next += 1;
        let prefix = if id == 0 {
            "root".to_owned()
        } else {
            format!("node{id}")
        };
        if id != 0 {
            writeln!(
                self.code,
                "if dfg.inst({inst}).opcode() != Opcode::{} {{ return None; }}",
                op.name
            )
            .unwrap();
        }
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
            .map(|f| format!("{}: {prefix}_{}", f.name, f.name))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            self.code,
            "let InstView::{} {{ {fields} }} = dfg.inst({inst}) else {{ return None; }};",
            op.format
        )
        .unwrap();
        for (param, arg) in op.params.iter().zip(args) {
            let ty = match &param.kind {
                ParamKind::Value => "Value",
                ParamKind::Property(ty) => ty,
                _ => unreachable!(),
            };
            let value =
                op.inputs[&param.name].emit(&|f| format!("{prefix}_{f}"), &|v| format!("{v}?"));
            if let Kind::Name(name) = &arg.kind {
                if name == "_" {
                    continue;
                }
                if super::identifier(name)
                    && !matches!(name.as_str(), "true" | "false" | "none")
                    && !self.bindings.contains_key(name)
                {
                    self.bindings.insert(
                        name.clone(),
                        (
                            Node {
                                offset: arg.offset,
                                kind: Kind::Name(ty.into()),
                            },
                            value,
                        ),
                    );
                    continue;
                }
            }
            if let Kind::Call(path, nested) = &arg.kind
                && path.starts_with(&format!("{}::", self.dialect))
                && param.kind == ParamKind::Value
            {
                let child = self.operation(path, nested.len(), arg.offset)?;
                if child.signature.results.patterns().unwrap().len() != 1 {
                    return Err(self.fail(arg.offset, "nested operation must have one result"));
                }
                writeln!(self.code, "const {{ assert!(Opcode::{}.spec().is_pure() && !Opcode::{}.transfers_ownership(), \"nested patterns require pure operations\"); }}", child.name, child.name).unwrap();
                let name = format!("def{}", self.next);
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
