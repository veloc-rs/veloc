//! Lower checked value recipes to bytecode; Rust is emitted only for host
//! adapters and static tables, never one function per construction recipe.
use super::*;
use crate::bytecode::{Assembler, Encoded};
use crate::rules::typed::{Call, Inst};
use veloc_bytecode::Lebs;

#[derive(Default)]
pub(super) struct Program {
    pub asm: Assembler,
    pub labels: usize,
    pub tests: Vec<Test>,
    pub sets: Vec<Vec<String>>,
    pub features: Vec<String>,
    pub feature_source: Option<String>,
    pub actions: Vec<String>,
    types: Vec<String>,
    fields: Vec<String>,
    recipes: BTreeMap<Vec<Vec<u8>>, usize>,
}
pub(super) enum Output {
    Value(String),
    Update {
        inputs: Vec<(usize, String)>,
        fields: Vec<(usize, String)>,
    },
}

impl Program {
    pub fn test(&mut self, id: usize, failure: usize) {
        let op = match &self.tests[id] {
            Test::Signature { results, inputs } => Op::CheckSignature {
                results: Lebs::Values(results),
                inputs: Lebs::Values(inputs),
                failure: 0,
            },
            Test::Same(values) => Op::CheckSameType {
                values: Lebs::Values(values),
                failure: 0,
            },
            Test::Type { value, set } => Op::CheckType {
                value: *value,
                set: *set,
                failure: 0,
            },
            Test::Signed {
                field,
                bits,
                expected,
            } => Op::CheckSignedRange {
                field: *field,
                bits: *bits,
                expected: usize::from(*expected),
                failure: 0,
            },
            Test::Features(set) => Op::CheckFeatures {
                set: *set,
                failure: 0,
            },
        };
        self.asm.branch(op, "failure", failure);
    }

    fn ty(
        &mut self,
        name: &str,
        sig: &Signature,
        expr: &Expressions<'_>,
        config: DecisionRust<'_>,
        offset: usize,
    ) -> Result<usize, Error> {
        let source = if let Some((result, index)) = sig
            .generics
            .contains_key(name)
            .then(|| sig.anchor(name).unwrap())
        {
            format!(
                "{}::TypeSource::Value {{ result: {result}, index: {index} }}",
                config.runtime
            )
        } else {
            format!(
                "{}::TypeSource::Exact({})",
                config.runtime,
                expr.constant(name, offset)?
            )
        };
        Ok(intern(&mut self.types, source))
    }

    /// Rebuild the matched instruction with selected scalar fields changed.
    /// Unmentioned operands, results and metadata retain their original meaning.
    pub fn update_recipe(
        &mut self,
        sig: &Signature,
        node: &Node,
        expr: &Expressions<'_>,
        functions: &crate::rules::functions::Functions,
        operations: &BTreeMap<String, crate::schema::Operation>,
        config: DecisionRust<'_>,
        name: &str,
    ) -> Result<String, Error> {
        use crate::storage::operands::Domain;
        let (last, preceding) = match &node.kind {
            Kind::List(nodes) => {
                let (last, preceding) = nodes.split_last().unwrap();
                (last, preceding)
            }
            _ => (node, &[][..]),
        };
        let Kind::Object(root, changes) = &last.kind else {
            unreachable!()
        };
        if root != &sig.node || sig.dynamic {
            return Err(Error::at(
                expr.source,
                last.offset,
                "update requires the matched fixed-signature instruction",
            ));
        }
        // Replacing control edges or satisfying additional verifier predicates
        // requires a dedicated checked operation, not a scalar field update.
        for root in expr.roots {
            let op = expr.defs.ops.iter().find(|op| &op.name == root).unwrap();
            let crate::model::Projection::Operands(layout) = &op.projection else {
                unreachable!()
            };
            if layout.flow != "Next" || operations[root].constrained {
                return Err(Error::at(
                    expr.source,
                    last.offset,
                    "constrained or control instruction requires a checked adapter",
                ));
            }
        }
        let mut insts = Vec::new();
        let mut locals = BTreeMap::new();
        for binding in preceding {
            let Kind::Let(name, value) = &binding.kind else {
                return Err(Error::at(
                    expr.source,
                    binding.offset,
                    "expected construction binding",
                ));
            };
            let value = sig.expression(
                expr.source,
                value,
                operations,
                expr.defs,
                &mut insts,
                config.dialect,
                &mut locals,
                functions,
                &mut Vec::new(),
            )?;
            if locals.insert(name.clone(), value).is_some() {
                return Err(Error::at(
                    expr.source,
                    binding.offset,
                    "duplicate replacement binding",
                ));
            }
        }
        let mut inputs = Vec::new();
        let mut fields = Vec::new();
        for (field, value) in changes {
            let access = Node {
                offset: value.offset,
                kind: Kind::Member(
                    Box::new(Node {
                        offset: value.offset,
                        kind: Kind::Name(root.clone()),
                    }),
                    field.clone(),
                ),
            };
            let (domain, index) = expr.access(&access)?;
            match domain {
                Domain::Input => {
                    let (ty, slot) = sig.expression(
                        expr.source,
                        value,
                        operations,
                        expr.defs,
                        &mut insts,
                        config.dialect,
                        &mut locals,
                        functions,
                        &mut Vec::new(),
                    )?;
                    let expected = &sig
                        .inputs
                        .iter()
                        .find(|(n, _)| n == &format!("{root}.{field}"))
                        .ok_or_else(|| Error::at(expr.source, value.offset, "unknown input field"))?
                        .1;
                    if &ty != expected && !(ty.domain.len() == 1 && ty.domain == expected.domain) {
                        return Err(Error::at(
                            expr.source,
                            value.offset,
                            "updated input type mismatch",
                        ));
                    }
                    inputs.push((index, slot));
                }
                Domain::Attribute => {
                    let source = match &value.kind {
                        Kind::Number(n) => format!(
                            "{}::FieldSource::Constant({}::Imm({n}))",
                            config.runtime, config.field
                        ),
                        Kind::Integer(n) => {
                            let n = i64::try_from(*n).map_err(|_| {
                                Error::at(expr.source, value.offset, "integer literal exceeds i64")
                            })?;
                            format!(
                                "{}::FieldSource::Constant({}::Imm({n}))",
                                config.runtime, config.field
                            )
                        }
                        Kind::Member(..) => {
                            let (domain, index) = expr.access(value)?;
                            if domain != Domain::Attribute {
                                return Err(Error::at(
                                    expr.source,
                                    value.offset,
                                    "expected an i64 attribute",
                                ));
                            }
                            format!("{}::FieldSource::Root({index})", config.runtime)
                        }
                        _ => {
                            return Err(Error::at(
                                expr.source,
                                value.offset,
                                "expected an i64 literal or field",
                            ));
                        }
                    };
                    fields.push((index, source));
                }
                Domain::Result => {
                    return Err(Error::at(
                        expr.source,
                        value.offset,
                        "updates preserve result identities",
                    ));
                }
            }
        }
        if inputs.is_empty() && fields.is_empty() {
            return Err(Error::at(
                expr.source,
                last.offset,
                "empty instruction update",
            ));
        }
        self.recipe(
            sig,
            &insts,
            Output::Update { inputs, fields },
            expr,
            config,
            name,
            node.offset,
        )
    }

    pub fn recipe(
        &mut self,
        sig: &Signature,
        insts: &[Inst],
        output: Output,
        expr: &Expressions<'_>,
        config: DecisionRust<'_>,
        name: &str,
        offset: usize,
    ) -> Result<String, Error> {
        let mut code = Assembler::default();
        let mut values: BTreeMap<_, _> = sig
            .inputs
            .iter()
            .enumerate()
            .map(|(i, _)| (format!("input{i}"), i))
            .collect();
        for inst in insts {
            let inputs: Vec<_> = inst.inputs.iter().map(|v| values[v]).collect();
            let dst = values.len();
            values.insert(inst.result.clone(), dst);
            let ty = self.ty(&inst.ty.name, sig, expr, config, offset)?;
            let (opcode, fields) = match &inst.op {
                Call::Instruction(op) => (op, Vec::new()),
                Call::Integer(op, n) => {
                    let operation = expr.defs.operations().find(|o| o.name == *op).unwrap();
                    let variant = &operation
                        .attributes
                        .first()
                        .ok_or_else(|| {
                            Error::at(expr.source, offset, "integer requires storage codec")
                        })?
                        .2;
                    (
                        op,
                        vec![format!(
                            "{}::FieldSource::Constant({}::{variant}({n}))",
                            config.runtime, config.field
                        )],
                    )
                }
                Call::FieldInteger(op, field) => {
                    let (domain, index) = expr.access(field)?;
                    if domain != crate::storage::operands::Domain::Attribute {
                        return Err(Error::at(
                            expr.source,
                            field.offset,
                            "expected an i64 attribute",
                        ));
                    }
                    (
                        op,
                        vec![format!("{}::FieldSource::Root({index})", config.runtime)],
                    )
                }
                Call::Attributed(op, fields) => (
                    op,
                    fields
                        .iter()
                        .map(|(variant, name)| {
                            Ok(format!(
                                "{}::FieldSource::Constant({}::{variant}({}))",
                                config.runtime,
                                config.field,
                                expr.constant(name, offset)?
                            ))
                        })
                        .collect::<Result<Vec<_>, Error>>()?,
                ),
            };
            let fields: Vec<_> = fields
                .into_iter()
                .map(|f| intern(&mut self.fields, f))
                .collect();
            // Same definition order as the generated opcode enum and decoder.
            let opcode = expr
                .defs
                .ops
                .iter()
                .position(|op| op.name == *opcode)
                .expect("checked construction opcode");
            code.emit(Op::Emit {
                opcode,
                ty,
                inputs: Lebs::Values(&inputs),
                fields: Lebs::Values(&fields),
                dst,
                reuse: usize::from(
                    matches!(&output, Output::Value(value) if &inst.result == value),
                ),
            });
        }
        match output {
            Output::Value(value) => code.emit(Op::Return {
                value: values[&value],
            }),
            Output::Update { inputs, fields } => {
                let inputs: Vec<_> = inputs
                    .iter()
                    .flat_map(|(i, value)| [*i, values[value]])
                    .collect();
                let fields: Vec<_> = fields
                    .into_iter()
                    .flat_map(|(i, source)| [i, intern(&mut self.fields, source)])
                    .collect();
                code.emit(Op::Update {
                    inputs: Lebs::Values(&inputs),
                    fields: Lebs::Values(&fields),
                });
            }
        }
        let entry = *self
            .recipes
            .entry(code.instructions.clone())
            .or_insert_with(|| {
                let label = self.asm.label();
                self.labels += 1;
                self.asm.instructions.extend(code.instructions);
                label
            });
        Ok(format!(
            "{}::Action::Recipe {{ name: {name:?}, entry: ENTRY_{entry}, slots: {} }}",
            config.runtime,
            values.len()
        ))
    }

    pub fn render(&self, out: &mut String, code: &Encoded, config: DecisionRust<'_>) {
        // Decode for human-readable generated output using the same opcode schema.
        writeln!(out, "#[rustfmt::skip]\nconst CODE: &[u8] = &[").unwrap();
        for (bytes, offset) in code.instructions.iter().zip(&code.offsets) {
            let op = Op::read(&mut veloc_bytecode::Reader { bytes, pc: 0 });
            writeln!(out, "    // @{offset:04x} {op:?}").unwrap();
            for b in bytes {
                write!(out, " {b},").unwrap();
            }
            out.push('\n');
        }
        out.push_str("];\n");
        for id in 0..self.labels {
            writeln!(out, "const ENTRY_{id}: usize = {};", code.labels[id]).unwrap();
        }
        writeln!(
            out,
            "pub static {}: {}::Program = {}::Program {{ entries: ENTRIES, code: CODE,",
            config.function.to_uppercase(),
            config.runtime,
            config.runtime
        )
        .unwrap();
        for (name, table) in [
            ("actions", &self.actions),
            ("types", &self.types),
            ("fields", &self.fields),
        ] {
            writeln!(out, "{name}: &[{}],", table.join(",\n")).unwrap();
        }
        writeln!(
            out,
            "sets: &[{}], features: &[{}],",
            self.sets
                .iter()
                .map(|s| format!("&[{}]", s.join(",")))
                .collect::<Vec<_>>()
                .join(","),
            self.features.join(",")
        )
        .unwrap();
        out.push_str("};\n");
    }
}
