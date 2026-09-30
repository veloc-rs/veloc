//! Lower checked value recipes to bytecode; Rust is emitted only for host
//! adapters and static tables, never one function per construction recipe.
use super::*;
use crate::bytecode::{Assembler, Encoded};
use crate::rules::typed::{Call, Inst};
use veloc_bytecode::{Lebs, Words};

#[derive(Default)]
pub(super) struct Program {
    pub asm: Assembler,
    pub labels: usize,
    pub tests: Vec<Test>,
    pub features: Vec<String>,
    pub feature_source: Option<String>,
    pub actions: Vec<String>,
    types: Vec<String>,
    fields: Vec<String>,
    recipes: BTreeMap<Vec<Vec<u8>>, usize>,
    type_constants: Vec<(usize, TypeConstant)>,
}

/// Rust evaluates the host type codec into fixed-width bytecode operands.
/// Keeping their width fixed lets the assembler resolve branches beforehand.
struct TypeConstant {
    offset: usize,
    ty: String,
    exact_pattern: bool,
}

fn pattern_words(patterns: &[Pattern]) -> (Vec<usize>, Vec<TypeConstant>) {
    let mut words = Vec::new();
    let mut constants = Vec::new();
    for pattern in patterns {
        let (header, types): (PatternHeader, &[String]) = match pattern {
            Pattern::Exact(ty) => (PatternHeader::Exact(0), std::slice::from_ref(ty)),
            Pattern::Set(types) => (PatternHeader::Set(types.len()), types),
            Pattern::Bind(types) => (PatternHeader::Bind(types.len()), types),
            Pattern::Same(slot) => (PatternHeader::Same(*slot), &[]),
        };
        let exact = matches!(header, PatternHeader::Exact(_));
        if !exact {
            words.push(header.encode());
        }
        for ty in types {
            constants.push(TypeConstant {
                // Account for the Words length prefix.
                offset: 4 + words.len() * 4,
                ty: ty.clone(),
                exact_pattern: exact,
            });
            words.push(if exact { header.encode() } else { 0 });
        }
    }
    (words, constants)
}

pub(super) enum Output {
    Value(String),
    Update {
        inputs: Vec<(usize, String)>,
        fields: Vec<(usize, String)>,
    },
}

impl Program {
    pub fn signature(
        sig: &Signature,
        expr: &Expressions<'_>,
        offset: usize,
    ) -> Result<Test, Error> {
        let mut bindings = BTreeMap::new();
        let mut pattern = |ty: &crate::rules::typed::Ty| -> Result<Pattern, Error> {
            if let Some(&slot) = bindings.get(&ty.name) {
                return Ok(Pattern::Same(slot));
            }
            let values = ty
                .domain
                .iter()
                .map(|name| expr.constant(name, offset))
                .collect::<Result<Vec<_>, _>>()?;
            if let [ty] = values.as_slice() {
                Ok(Pattern::Exact(ty.clone()))
            } else if sig.generics.contains_key(&ty.name) {
                let slot = bindings.len();
                bindings.insert(ty.name.clone(), slot);
                Ok(Pattern::Bind(values))
            } else {
                Ok(Pattern::Set(values))
            }
        };
        let results = sig
            .results
            .iter()
            .map(&mut pattern)
            .collect::<Result<_, _>>()?;
        let inputs = sig
            .inputs
            .iter()
            .map(|(_, ty)| pattern(ty))
            .collect::<Result<_, _>>()?;
        Ok(Test::Signature { results, inputs })
    }

    pub fn test(&mut self, id: usize, failure: usize) {
        let op = match &self.tests[id] {
            Test::Signature { results, inputs } => {
                let (results, result_types) = pattern_words(results);
                let (inputs, input_types) = pattern_words(inputs);
                let op = Op::CheckSignature {
                    results: TypePatterns::from_words(Words::Values(&results)),
                    inputs: TypePatterns::from_words(Words::Values(&inputs)),
                    failure: 0,
                };
                for (field, constants) in [("results", result_types), ("inputs", input_types)] {
                    let offset = op.field_offset(field).unwrap();
                    for mut constant in constants {
                        constant.offset += offset;
                        self.type_constants
                            .push((self.asm.instructions.len(), constant));
                    }
                }
                self.asm.branch(op, "failure", failure);
                return;
            }
            Test::Type { value, ty } => {
                let op = Op::CheckType {
                    value: *value,
                    ty: 0,
                    failure: 0,
                };
                self.type_constants.push((
                    self.asm.instructions.len(),
                    TypeConstant {
                        offset: op.field_offset("ty").unwrap(),
                        ty: ty.clone(),
                        exact_pattern: false,
                    },
                ));
                op
            }
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
            let operand = if result {
                OperandRef::Result(index)
            } else {
                OperandRef::Input(index)
            };
            format!(
                "{}::TypeSource::Value({}::OperandRef::{operand:?})",
                config.runtime, config.runtime,
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
        writeln!(
            out,
            "// Type operands are placeholders filled by the const expressions below.\n#[rustfmt::skip]\nconst CODE: &[u8] = &{{ let mut code = ["
        )
        .unwrap();
        for (bytes, offset) in code.instructions.iter().zip(&code.offsets) {
            let op = Op::read(&mut veloc_bytecode::Reader { bytes, pc: 0 });
            writeln!(out, "    // @{offset:04x} {op:?}").unwrap();
            for b in bytes {
                write!(out, " {b},").unwrap();
            }
            out.push('\n');
        }
        out.push_str("];\n");
        for (instruction, constant) in &self.type_constants {
            let offset = code.offsets[*instruction] + constant.offset;
            let encoded = format!("{}::TypeCodec::encode({})", config.runtime, constant.ty);
            let encoded = if constant.exact_pattern {
                format!("veloc_bytecode::signature::PatternHeader::Exact({encoded}).encode()")
            } else {
                encoded
            };
            writeln!(out, "// Inline type at @{offset:04x}: {}", constant.ty).unwrap();
            writeln!(out, "let bytes = ({encoded} as u32).to_le_bytes();").unwrap();
            for i in 0..4 {
                writeln!(out, "code[{}] = bytes[{i}];", offset + i).unwrap();
            }
        }
        out.push_str("code };\n");
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
            "required_features: &[{}],",
            self.features
                .iter()
                .map(|set| { format!("{}::FeatureSetRef::new({set})", config.runtime) })
                .collect::<Vec<_>>()
                .join(",")
        )
        .unwrap();
        out.push_str("};\n");
    }
}
