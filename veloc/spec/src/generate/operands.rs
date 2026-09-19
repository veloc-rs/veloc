//! Array-storage Rust host. Logical accesses, queries and contracts are shared.
use crate::model::{Definitions, Op};
use crate::storage::operands::{Domain, Operands, Shape, domain_index};
use std::fmt::Write;

impl Operands {
    fn emit_signature_host(&self, out: &mut String, defs: &Definitions) {
        let mut methods = std::collections::BTreeMap::new();
        for source in defs
            .ops
            .iter()
            .filter_map(|op| op.signature_source.as_ref())
        {
            let (method, ty) = signature_method(source);
            let ty = ty
                .map(|ty| defs.data.rust.qualified(ty))
                .unwrap_or_else(|| self.register_rust.clone());
            methods.insert(method, ty);
        }
        if methods.is_empty() {
            return;
        }
        for (method, ty) in methods {
            writeln!(
                out,
                "fn {method}(self, value: {ty}) -> Option<(&'a [crate::Type], &'a [crate::Type])>;"
            )
            .unwrap();
        }
        let reg = &self.register_rust;
        out.push_str(&format!(r#"
        fn validate_values(self, name: &str, role: &str, values: &[{reg}], expected: &[crate::Type]) -> core::result::Result<(), Self::Error> {{
            if values.len() != expected.len() {{
                return Err(self.error(&alloc::format!("{{name}} {{role}} count mismatch: expected {{}}, got {{}}", expected.len(), values.len())));
            }}
            for (index, (&value, &expected)) in values.iter().zip(expected).enumerate() {{
                let got = self.value_type(value);
                if got != expected {{
                    return Err(self.error(&alloc::format!("{{name}} {{role}} {{index}} type mismatch: expected {{expected}}, got {{got}}")));
                }}
            }}
            Ok(())
        }}
        "#));
    }
    fn view_plan(&self, defs: &Definitions) -> crate::generate::views::View {
        use crate::generate::views::{Field, Representation, Variant, View};
        View {
            name: self.view.clone(),
            representation: Representation::Records,
            variants: self
                .formats
                .values()
                .map(|format| {
                    let ops: Vec<_> = defs
                        .ops
                        .iter()
                        .filter(|op| op.format == format.name)
                        .collect();
                    let opcodes = if ops.len() > 1 {
                        ops.iter().map(|op| op.name.clone()).collect()
                    } else {
                        Vec::new()
                    };
                    let mut fields = Vec::new();
                    if ops.len() > 1 {
                        fields.push(Field {
                            name: "opcode".into(),
                            ty: format!("{}Opcode", format.name),
                        });
                    }
                    fields.extend(format.fields.iter().map(|field| Field {
                        name: field.name.clone(),
                        ty: field.view_type(),
                    }));
                    Variant {
                        name: format.name.clone(),
                        fields,
                        borrowed: format
                            .fields
                            .iter()
                            .any(|f| f.shape == Shape::Sequence || f.borrowed),
                        opcodes,
                    }
                })
                .collect(),
        }
    }

    pub(crate) fn generate(&self, defs: &Definitions) -> String {
        let opcode = &self.opcode;
        let view = &self.view;
        let reader = &self.reader;
        let writer = &self.writer;
        let reg = &self.register_rust;
        let attrs = &self.attributes;
        let value_types = defs.ops.iter().any(|op| op.signature_source.is_some())
            || defs
                .ops
                .iter()
                .flat_map(|op| {
                    op.constraints
                        .iter()
                        .filter(|c| !c.type_only || c.binding.is_some())
                        .map(|c| &c.condition)
                        .chain(op.queries.values())
                })
                .any(crate::model::expr::Expr::needs_value_types);
        let type_method = if value_types {
            format!("fn value_type(self, value: {reg}) -> crate::Type;")
        } else {
            String::new()
        };
        let mut out = String::from("// @generated from storage definitions.\n");
        out.push_str(&"#[derive(Debug, Clone, Copy)] pub struct AttributeList<'a, T> { fields: FieldView<'a>, start: usize, decode: fn($ATTRSRef<'a>) -> T }\nimpl<'a, T> AttributeList<'a, T> { pub fn iter(&self) -> impl DoubleEndedIterator<Item = T> + ExactSizeIterator + '_ { (self.start..self.fields.len()).map(|i| (self.decode)(self.fields.read(i))) } pub fn len(&self) -> usize { self.fields.len() - self.start } pub fn is_empty(&self) -> bool { self.len() == 0 } }\n".replace("$ATTRS", attrs));
        out.push_str(&crate::generate::opcode_enum(defs, &self.opcode));
        if let Some((control, _, _)) = &self.control {
            writeln!(
                out,
                "impl {opcode} {{ pub const fn control(self) -> {control} {{ match self {{"
            )
            .unwrap();
            for op in &defs.ops {
                writeln!(
                    out,
                    "Self::{} => {control}::{},",
                    op.name,
                    op.operands().flow
                )
                .unwrap();
            }
            out.push_str("} } }\n");
        }
        let view_plan = self.view_plan(defs);
        let lifetime = view_plan.lifetime();
        out.push_str(&view_plan.generate());
        writeln!(
            out,
            "pub trait {reader}<'a>: Copy {{
            type Error;
            {type_method}
            fn opcode(self) -> Option<{opcode}>;
            fn results(self) -> &'a [{reg}];
            fn inputs(self) -> &'a [{reg}];
            fn fields(self) -> FieldView<'a>;
            fn error(self, message: &str) -> Self::Error;
            fn view(self) -> {view}{lifetime} {{ match self.opcode() {{"
        )
        .unwrap();
        for op in &defs.ops {
            let f = &self.formats[&op.format];
            writeln!(
                out,
                "Some({opcode}::{}) => {view}::{}({}Inst {{",
                op.name, f.name, f.name
            )
            .unwrap();
            if defs.ops.iter().filter(|op| op.format == f.name).count() > 1 {
                writeln!(out, "opcode: {}Opcode::{},", f.name, op.name).unwrap();
            }
            for m in &op.operands().members {
                writeln!(out, "{}: {},", m.field.name, m.read_from("self")).unwrap();
            }
            out.push_str("}),\n");
        }
        out.push_str("_ => panic!(\"expected instruction from this opcode set\"),\n} }\n");
        self.emit_signature_host(&mut out, defs);
        self.emit_validator(&mut out, defs);
        out.push_str(&crate::generate::queries::generate(
            defs,
            crate::generate::queries::Host::Operands,
        ));
        out.push_str(&crate::generate::ownership::operand_methods(defs));
        out.push_str("}\n");
        writeln!(out, "pub trait {writer}: Sized {{
            type Inst;
            type Def;
            fn reg(value: Self::Def) -> {reg};
            fn write(self, opcode: {opcode}, results: &[{reg}], inputs: &[{reg}], fields: impl IntoIterator<Item = {attrs}>) -> Self::Inst;").unwrap();
        for op in &defs.ops {
            self.emit_builder(&mut out, op);
        }
        out.push_str("}\n");
        out
    }
    fn emit_validator(&self, out: &mut String, defs: &Definitions) {
        let opcode = &self.opcode;
        let contexts = crate::model::constraints::contexts(defs, &[]);
        let params = contexts
            .iter()
            .map(|(ty, i)| format!(", _ctx{i}: &{ty}"))
            .collect::<String>();
        out.push_str(
            &format!("fn validate(self{params}) -> core::result::Result<(), Self::Error> {{ match self.opcode() {{\n"),
        );
        for op in &defs.ops {
            let plan = op.operands();
            writeln!(out, "Some({opcode}::{}) => {{", op.name).unwrap();
            for domain in [Domain::Result, Domain::Input, Domain::Attribute] {
                let d = domain_index(domain);
                let count = plan.counts[d];
                if plan.tails[d] && count == 0 {
                    continue;
                }
                let cmp = if plan.tails[d] { "<" } else { "!=" };
                writeln!(out, "if self.{}().len() {cmp} {count} {{ return Err(self.error(\"invalid {} count\")); }}", domain.accessor(), domain.accessor()).unwrap();
            }
            for m in &plan.members {
                if m.binding.is_some() {
                    if let Some(codec) = &m.field.codec {
                        let (name, variant) = codec.rsplit_once("::").unwrap();
                        let codec = format!("{name}Ref::{variant}");
                        if m.field.shape == Shape::Sequence {
                            writeln!(out, "if ({}..self.fields().len()).any(|i| !matches!(self.fields().read(i), {codec}(_))) {{ return Err(self.error(\"invalid {} field\")); }}", m.index, m.field.name).unwrap();
                            continue;
                        }
                        writeln!(out, "if !matches!(self.fields().read({}), {codec}(_)) {{ return Err(self.error(\"invalid {} field\")); }}", m.index, m.field.name).unwrap();
                    }
                }
            }
            let error = |text: &str| format!("self.error({text:?})");
            let projections = plan.projections(op, |v| {
                format!(
                    "({v}).ok_or_else(|| {})?",
                    error("missing property storage")
                )
            });
            let mut emitter =
                crate::model::expr::Emitter::values(projections, "self", "self.results()");
            out.push_str(&crate::model::constraints::emit_signature(
                op, &emitter.projections,
                |source, value| {
                    let method = signature_method(source).0;
                    format!("self.{method}({value}).ok_or_else(|| self.error(\"missing function or signature\"))?")
                },
                |role, values, types| format!("self.validate_values({:?}, {role:?}, {values}, {types})?;", op.mnemonic),
                "self.results()",
            ));
            out.push_str(&crate::model::constraints::emit_checks(
                &op.constraints,
                &mut emitter,
                &contexts,
                true,
                error,
            ));
            out.push_str("Ok(())\n},\n");
        }
        out.push_str("_ => Err(self.error(\"expected instruction from this opcode set\")),\n} }\n");
    }
    fn emit_builder(&self, out: &mut String, op: &Op) {
        let plan = op.operands();
        let args = plan
            .args
            .iter()
            .map(|a| format!("{}: {}", a.name, a.rust))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            out,
            "fn {}(self, {args}) -> Self::Inst {{",
            crate::model::mnemonic(&op.name)
        )
        .unwrap();
        writeln!(
            out,
            "{}\n}}",
            self.construction(op, "self", "Self::reg", str::to_owned)
                .emit()
        )
        .unwrap();
    }
}

fn signature_method(
    source: &crate::model::SignatureSource,
) -> (&'static str, Option<&'static str>) {
    match source {
        crate::model::SignatureSource::Function(_) => ("function_signature", Some("FuncId")),
        crate::model::SignatureSource::Signature(_) => ("signature", Some("SigId")),
        crate::model::SignatureSource::Value(_) => ("value_signature", None),
    }
}
