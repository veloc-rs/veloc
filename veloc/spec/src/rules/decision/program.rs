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
    pub predicates: Vec<String>,
    pub tests: Vec<Test>,
    pub sets: Vec<Vec<String>>,
    pub features: Vec<String>,
    pub feature_source: Option<String>,
    pub actions: Vec<String>,
    types: Vec<String>,
    opcodes: Vec<String>,
    fields: Vec<String>,
    functions: Vec<String>,
    recipes: BTreeMap<Vec<Vec<u8>>, usize>,
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
            Test::Host(predicate) => Op::CallPredicate {
                predicate: *predicate,
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

    pub fn recipe(
        &mut self,
        sig: &Signature,
        insts: &[Inst],
        value: &str,
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
            if let Call::Host { name, types } = &inst.op {
                let types = types
                    .iter()
                    .map(|ty| self.ty(&ty.name, sig, expr, config, offset))
                    .collect::<Result<Vec<_>, _>>()?;
                let args = (0..types.len())
                    .map(|i| format!("types[{i}]"))
                    .chain((0..inputs.len()).map(|i| format!("inputs[{i}]")))
                    .collect::<Vec<_>>();
                let function = intern(
                    &mut self.functions,
                    format!(
                        "|ctx, types, inputs| build_{name}(ctx, {})",
                        args.join(", ")
                    ),
                );
                code.emit(Op::Call {
                    function,
                    types: Lebs::Values(&types),
                    inputs: Lebs::Values(&inputs),
                    dst,
                });
                continue;
            }
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
                    (op, vec![format!("{}::{variant}({n})", config.field)])
                }
                Call::Attributed(op, fields) => (
                    op,
                    fields
                        .iter()
                        .map(|(variant, name)| {
                            Ok(format!(
                                "{}::{variant}({})",
                                config.field,
                                expr.constant(name, offset)?
                            ))
                        })
                        .collect::<Result<Vec<_>, Error>>()?,
                ),
                Call::Host { .. } => unreachable!(),
            };
            let fields: Vec<_> = fields
                .into_iter()
                .map(|f| intern(&mut self.fields, f))
                .collect();
            let opcode = intern(&mut self.opcodes, format!("{}::{opcode}", config.opcode));
            code.emit(Op::Emit {
                opcode,
                ty,
                inputs: Lebs::Values(&inputs),
                fields: Lebs::Values(&fields),
                dst,
                reuse: usize::from(inst.result == value),
            });
        }
        code.emit(Op::Return {
            value: values[value],
        });
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

    pub fn render(
        &self,
        out: &mut String,
        code: &Encoded,
        config: DecisionRust<'_>,
        interface: &ValueInterface,
    ) {
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
            "static PROGRAM: {}::Program = {}::Program {{ code: CODE,",
            config.runtime, config.runtime
        )
        .unwrap();
        for (name, table) in [
            ("actions", &self.actions),
            ("types", &self.types),
            ("opcodes", &self.opcodes),
            ("fields", &self.fields),
            ("functions", &self.functions),
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
        writeln!(out, "emit: |ctx, opcode, ty, inputs, fields, result| {}::{}(ctx, opcode, ty, inputs, fields, result),", interface.contract, interface.emit).unwrap();
        out.push_str("};\n");
    }
}
