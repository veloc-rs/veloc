//! Compile the shared matching graph and construction recipes to bytecode.
//! Matching uses tables; schema-generated constructors install complete instructions.
use super::*;
use crate::bytecode::intern;
use veloc_bytecode::{Lebs, Reader, selection::Instruction as Op};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Access {
    Reg,
    Attribute,
}

pub(in super::super) struct Adapters<'a> {
    layouts: &'a BTreeMap<String, crate::storage::operands::Projection>,
    predicates: Vec<String>,
    builders: BTreeMap<String, String>,
}
impl<'a> Adapters<'a> {
    pub(in super::super) fn new(
        layouts: &'a BTreeMap<String, crate::storage::operands::Projection>,
    ) -> Self {
        Self {
            layouts,
            predicates: Vec::new(),
            builders: BTreeMap::new(),
        }
    }
    fn builder(&mut self, opcode: &str, source: &str, definition: &FinalInstDef) -> String {
        use crate::storage::operands::{Domain, Shape};
        let call = definition
            .operands
            .iter()
            .any(|op| matches!(op, OperandConstraint::Call(_)));
        let build = format!("build_{}", opcode.to_ascii_lowercase());
        let adapter = if call {
            format!(
                "construct_{}_from_{}",
                opcode.to_ascii_lowercase(),
                source.to_ascii_lowercase()
            )
        } else {
            format!("construct_{}", opcode.to_ascii_lowercase())
        };
        if self.builders.contains_key(&adapter) {
            return adapter;
        }
        let mut params = vec!["writer: veloc_lir::InstWriter<'_>".to_owned()];
        let mut args = Vec::new();
        let mut results = Vec::new();
        let mut inputs = Vec::new();
        let mut fields = Vec::new();
        let mut reads = String::new();
        for op in &definition.operands {
            let (name, ty, variant) = match op {
                OperandConstraint::Def(name) | OperandConstraint::Use(name) => (name, "Reg", None),
                OperandConstraint::FixedUse { src, .. } => (src, "Reg", None),
                OperandConstraint::Imm(name) => (name, "i64", Some("Imm")),
                OperandConstraint::Block(name) => (name, "veloc_lir::EdgeId", Some("Edge")),
                OperandConstraint::Global(name) => (name, "veloc_lir::SymbolId", Some("Global")),
                OperandConstraint::StackSlot(name) => {
                    (name, "veloc_lir::StackSlot", Some("StackSlot"))
                }
                OperandConstraint::Call(name) => (name, "veloc_lir::CallInfo", Some("Call")),
            };
            let name = format!("operand_{}", sanitize_ident(name));
            params.push(format!("{name}: {ty}"));
            args.push(name.clone());
            if let Some(variant) = variant {
                writeln!(reads, "let FieldValue::{variant}({name}) = fields.next().expect(\"generated field\") else {{ unreachable!(\"generated field type\") }};").unwrap();
                fields.push(format!("FieldValue::{variant}({name})"));
            } else {
                let (domain, list) = if matches!(op, OperandConstraint::Def(_)) {
                    ("results", &mut results)
                } else {
                    ("inputs", &mut inputs)
                };
                writeln!(reads, "let {name} = _{domain}[{}];", list.len()).unwrap();
                list.push(name);
            }
        }
        if !self.builders.contains_key(&build) {
            let mut body = String::new();
            if call {
                params.extend([
                    "abi_args: &[Reg]".into(),
                    "abi_results: &[Reg]".into(),
                    "effects: veloc_lir::RegEffects<&[Reg]>".into(),
                ]);
            }
            writeln!(
                body,
                "fn {build}({}) -> veloc_lir::InstId {{",
                params.join(", ")
            )
            .unwrap();
            if call {
                writeln!(body, "let mut inputs = smallvec::SmallVec::<[Reg; 8]>::from_slice(&[{}]); inputs.extend_from_slice(abi_args);", inputs.join(", ")).unwrap();
                writeln!(body, "let mut results = smallvec::SmallVec::<[Reg; 4]>::from_slice(&[{}]); results.extend_from_slice(abi_results);", results.join(", ")).unwrap();
                writeln!(
                    body,
                    "let metadata = target_inst_metadata(TargetInst::{opcode});"
                )
                .unwrap();
                body.push_str(
                    "let mut uses = smallvec::SmallVec::<[Reg; 4]>::from_slice(metadata.implicit_uses);
                     let mut defs = smallvec::SmallVec::<[Reg; 4]>::from_slice(metadata.implicit_defs);
                     for &reg in effects.uses { if !uses.contains(&reg) { uses.push(reg); } }
                     for &reg in effects.defs { if !defs.contains(&reg) { defs.push(reg); } }
"
                );
                writeln!(body, "writer.with_effects(&uses, &defs).write(veloc_lir::MachineOpcode::Target(TargetInst::{opcode}.as_u32()), &results, &inputs, [{}])", fields.join(", ")).unwrap();
            } else {
                writeln!(
                    body,
                    "TargetInst::{opcode}.write(writer, &[{}], &[{}], [{}])",
                    results.join(", "),
                    inputs.join(", "),
                    fields.join(", ")
                )
                .unwrap();
            }
            body.push_str("}\n");
            self.builders.insert(build.clone(), body);
        }
        let mut body = format!(
            "fn {adapter}(store: &mut veloc_lir::InstInserter<'_>, _source: veloc_lir::InstId, _results: &[Reg], _inputs: &[Reg], _fields: smallvec::SmallVec<[FieldValue; 4]>) -> veloc_lir::InstId {{\n"
        );
        if !fields.is_empty() {
            body.push_str("let mut fields = _fields.into_iter();\n");
        }
        body.push_str(&reads);
        if call {
            // Resolve the variadic input position from the source definition,
            // never from opcode names or runtime instruction matching.
            let input = self.layouts[source]
                .members
                .iter()
                .find(|m| m.domain == Domain::Input && m.field.shape == Shape::Sequence)
                .expect("call construction requires source ABI arguments");
            writeln!(body, "let source = store.inst(_source);").unwrap();
            writeln!(body, "let abi_args = smallvec::SmallVec::<[Reg; 8]>::from_slice(&source.inputs()[{}..]);", input.index).unwrap();
            body.push_str(
                "let abi_results = smallvec::SmallVec::<[Reg; 4]>::from_slice(source.results());\n",
            );
            body.push_str("let effects = source.effects().unwrap_or_default();\nlet uses = smallvec::SmallVec::<[Reg; 4]>::from_slice(effects.uses);\nlet defs = smallvec::SmallVec::<[Reg; 4]>::from_slice(effects.defs);\n");
            args.extend([
                "&abi_args".into(),
                "&abi_results".into(),
                "veloc_lir::RegEffects { uses: &uses, defs: &defs }".into(),
            ]);
        }
        let arguments = if args.is_empty() {
            String::new()
        } else {
            format!(", {}", args.join(", "))
        };
        writeln!(body, "{build}(store.writer(){arguments})\n}}").unwrap();
        self.builders.insert(adapter.clone(), body);
        adapter
    }
    fn field(&self, opcode: &str, field: &str, access: Access) -> String {
        use crate::storage::operands::{Domain, Shape};
        let member = self.layouts[opcode]
            .members
            .iter()
            .find(|m| m.field.name == field)
            .expect("checked storage field");
        assert!(
            member.field.shape != Shape::Sequence,
            "scalar selector access cannot read sequence {opcode}.{field}"
        );
        assert_eq!(
            member.field.codec.is_none(),
            access == Access::Reg,
            "selector field domain mismatch"
        );
        if member.binding.is_none() {
            return "None".into();
        }
        let domain = match member.domain {
            Domain::Input => "Input",
            Domain::Result => "Result",
            Domain::Attribute => "Attribute",
        };
        format!(
            "Some(crate::isel::matching::Field::{domain}({}))",
            member.index
        )
    }
    pub(in super::super) fn emit(
        &self,
        out: &mut String,
        context: &str,
        extractors: &HashMap<String, ExtractorDef>,
        decls: &HashMap<String, DeclDef>,
    ) {
        for builder in self.builders.values() {
            out.push_str(builder);
        }
        writeln!(out, "fn selection_predicate<C: {context}>(_ctx: &C, id: u32, reg: Reg) -> bool {{ let Some(_v) = reg.as_vreg() else {{ return false }}; match id {{").unwrap();
        for (id, name) in self.predicates.iter().enumerate() {
            let condition = generate_pattern_condition(&extractors[name].body, "_v", decls)
                .replace("ctx.", "_ctx.");
            writeln!(out, "{id} => {condition},").unwrap();
        }
        writeln!(out, "_ => unreachable!(\"selection predicate ID\"), }} }}").unwrap();
    }
}

struct Instruction {
    bytes: Vec<u8>,
    failure: Option<usize>,
}
#[derive(Default)]
struct Code {
    instructions: Vec<Instruction>,
    labels: Vec<usize>,
    types: Vec<Vec<String>>,
    integers: Vec<String>,
    opcodes: Vec<String>,
    targets: Vec<String>,
    registers: Vec<u32>,
    accesses: Vec<(String, String, Access)>,
    features: Vec<Vec<String>>,
    values: usize,
    fields: usize,
}
impl Code {
    fn field(
        &mut self,
        adapters: &Adapters<'_>,
        opcode: &str,
        field: &str,
        access: Access,
    ) -> usize {
        let _ = adapters.field(opcode, field, access);
        intern(&mut self.accesses, (opcode.into(), field.into(), access))
    }
    fn op(&mut self, op: Op<'_>) {
        let mut bytes = Vec::new();
        op.encode(&mut bytes);
        self.instructions.push(Instruction {
            bytes,
            failure: None,
        });
    }
    fn branch(&mut self, op: Op<'_>, target: usize) {
        self.op(op);
        self.instructions.last_mut().unwrap().failure = Some(target);
    }
    fn read_reg(
        &mut self,
        plan: &Plan,
        adapters: &mut Adapters,
        root: &str,
        path: &str,
        dst: usize,
    ) {
        let (node, schema, field) = resolve_field(plan, root, path);
        let field = self.field(adapters, schema, field, Access::Reg);
        self.op(Op::ReadReg { dst, node, field });
        self.values = self.values.max(dst + 1);
    }
    fn test(
        &mut self,
        plan: &Plan,
        adapters: &mut Adapters,
        root: &str,
        test: &Test,
        failure: usize,
    ) {
        match test {
            Test::Definition(slot) => {
                let def = &plan.definitions[*slot];
                self.read_reg(plan, adapters, root, &def.input, 0);
                self.branch(
                    Op::GetDef {
                        dst: slot + 1,
                        value: 0,
                        failure: 0,
                    },
                    failure,
                );
                let opcode = intern(&mut self.opcodes, def.opcode.clone());
                self.branch(
                    Op::CheckOpcode {
                        node: slot + 1,
                        opcode,
                        failure: 0,
                    },
                    failure,
                );
            }
            Test::Field {
                field,
                schema,
                guard,
            } => match guard {
                Guard::Types(types) => {
                    self.read_reg(plan, adapters, root, field, 0);
                    let set = intern(&mut self.types, types.clone());
                    self.branch(
                        Op::CheckType {
                            value: 0,
                            set,
                            failure: 0,
                        },
                        failure,
                    );
                }
                Guard::Extractor(name) => {
                    self.read_reg(plan, adapters, root, field, 0);
                    let predicate = intern(&mut adapters.predicates, name.clone());
                    self.branch(
                        Op::CallPredicate {
                            value: 0,
                            id: predicate,
                            failure: 0,
                        },
                        failure,
                    );
                }
                Guard::Integer(_) | Guard::Condition(_) => {
                    let (access, constant) = match guard {
                        Guard::Integer(value) => (Access::Attribute, value.to_string()),
                        Guard::Condition(cc) => (
                            Access::Attribute,
                            format!(
                                "{} as i64",
                                render_cond_code_match(schema, *cc).expect("checked condition")
                            ),
                        ),
                        _ => unreachable!(),
                    };
                    let (node, schema, field) = resolve_field(plan, root, field);
                    let field = self.field(adapters, schema, field, access);
                    let constant = intern(&mut self.integers, constant);
                    self.branch(
                        Op::CheckInt {
                            node,
                            field,
                            constant,
                            failure: 0,
                        },
                        failure,
                    );
                }
            },
            Test::Features(features) => {
                let set = intern(&mut self.features, features.clone());
                self.branch(Op::CheckFeatures { set, failure: 0 }, failure);
            }
            Test::Foldable(slot) => self.branch(
                Op::CheckFoldable {
                    definition: slot + 1,
                    consumer: 0,
                    failure: 0,
                },
                failure,
            ),
        }
    }
    fn recipe(
        &mut self,
        plan: &Plan,
        rule: &SelectRuleDef,
        adapters: &mut Adapters,
        instructions: &HashMap<String, FinalInstDef>,
        regs: &HashMap<String, u32>,
    ) {
        self.op(Op::Accept {});
        let fields = collect_field_variable_bindings(&rule.fields);
        let mut temps = HashMap::new();
        let mut values = 1; // Slot zero is matching scratch, not a rule binding.
        let mut payloads = 0;
        for (name, ty) in &rule.temps {
            temps.insert(name.as_str(), values);
            let ty = intern(&mut self.types, vec![ty.clone()]);
            self.op(Op::MakeTemp { dst: values, ty });
            values += 1;
        }
        let mut builds = Vec::new();
        for constructor in &rule.builds {
            let Constructor::Inst { opcode, args } = constructor else {
                unreachable!("checked constructor")
            };
            let definition = &instructions[opcode];
            let mut cursor = 0;
            let mut source_result = 0;
            let mut lists: [Vec<usize>; 3] = Default::default();
            for (index, operand) in definition.operands.iter().enumerate() {
                let category = match operand {
                    OperandConstraint::Def(_) => 0,
                    OperandConstraint::Use(_) | OperandConstraint::FixedUse { .. } => 1,
                    _ => 2,
                };
                if rule.builds.len() == 1
                    && category == 0
                    && args.len() - cursor == min_explicit_args_from(&definition.operands, index)
                {
                    self.op(Op::ReadResult {
                        dst: values,
                        index: source_result,
                    });
                    lists[0].push(values);
                    values += 1;
                    source_result += 1;
                    continue;
                }
                let arg = &args[cursor];
                cursor += 1;
                let slot = if category != 2 {
                    match arg {
                        Constructor::Variable(name) if temps.contains_key(name.as_str()) => {
                            temps[name.as_str()]
                        }
                        _ => {
                            let dst = values;
                            values += 1;
                            match arg {
                                Constructor::Variable(name) => {
                                    self.read_reg(plan, adapters, &rule.opcode, &fields[name], dst)
                                }
                                Constructor::Reg(name) => {
                                    let reg = intern(&mut self.registers, regs[name]);
                                    self.op(Op::ConstReg { dst, reg });
                                }
                                _ => panic!("non-register target operand"),
                            }
                            dst
                        }
                    }
                } else {
                    let dst = payloads;
                    payloads += 1;
                    match arg {
                        Constructor::Imm(value) => {
                            let imm = intern(&mut self.integers, value.to_string());
                            self.op(Op::ConstImm { dst, imm });
                        }
                        Constructor::Variable(name) => {
                            let (node, schema, field) =
                                resolve_field(plan, &rule.opcode, &fields[name]);
                            let access = Access::Attribute;
                            let field = self.field(adapters, schema, field, access);
                            self.op(Op::ReadField { dst, node, field });
                        }
                        _ => panic!("invalid target payload"),
                    }
                    dst
                };
                lists[category].push(slot);
            }
            assert_eq!(cursor, args.len(), "target operand count");
            let builder = adapters.builder(opcode, &rule.opcode, definition);
            builds.push((intern(&mut self.targets, builder), lists));
        }
        // Read all source fields before any target instruction is installed.
        for (target, [results, inputs, fields]) in builds {
            self.op(Op::BuildInst {
                target,
                results: Lebs::Values(&results),
                inputs: Lebs::Values(&inputs),
                fields: Lebs::Values(&fields),
            });
        }
        self.op(Op::Finish {});
        self.values = self.values.max(values);
        self.fields = self.fields.max(payloads);
    }
    fn describe(&self, inst: &Instruction, adapters: &Adapters) -> String {
        let op = Op::read(&mut Reader {
            bytes: &inst.bytes,
            pc: 0,
        });
        let field = |id: usize| {
            let (schema, name, _) = &self.accesses[id];
            format!("{schema}.{name}")
        };
        match op {
            Op::ReadReg {
                dst,
                node,
                field: f,
            } => format!("v{dst} <- n{node} {}", field(f)),
            Op::ReadField {
                dst,
                node,
                field: f,
            } => format!("f{dst} <- n{node} {}", field(f)),
            Op::GetDef { dst, value, .. } => format!("n{dst} <- def(v{value})"),
            Op::CheckOpcode { node, opcode, .. } => format!("n{node} == {}", self.opcodes[opcode]),
            Op::CheckType { value, set, .. } => {
                format!("v{value} in [{}]", self.types[set].join(", "))
            }
            Op::CheckInt {
                node,
                field: f,
                constant,
                ..
            } => format!("n{node} {} == {}", field(f), self.integers[constant]),
            Op::CheckFeatures { set, .. } => self.features[set].join(" + "),
            Op::CallPredicate { value, id, .. } => format!("{}(v{value})", adapters.predicates[id]),
            Op::CheckFoldable {
                definition,
                consumer,
                ..
            } => format!("n{definition} into n{consumer}"),
            Op::MakeTemp { dst, ty } => format!("v{dst}: {}", self.types[ty].join(", ")),
            Op::ReadResult { dst, index } => format!("v{dst} <- root.results[{index}]"),
            Op::ConstReg { dst, reg } => format!("v{dst} <- preg{}", self.registers[reg]),
            Op::ConstImm { dst, imm } => format!("f{dst} <- {}", self.integers[imm]),
            Op::BuildInst {
                target,
                results,
                inputs,
                fields,
            } => {
                let mut text = self.targets[target].clone();
                for (name, prefix, list) in [
                    ("results", "v", results),
                    ("inputs", "v", inputs),
                    ("fields", "f", fields),
                ] {
                    let slots = list
                        .iter()
                        .map(|id| format!("{prefix}{id}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    write!(text, " {name}=[{slots}]").unwrap();
                }
                text
            }
            Op::Accept {} => "begin construction (no fallback)".into(),
            Op::Finish {} => "return replacement to driver".into(),
            Op::Reject {} | Op::Jump { .. } => String::new(),
        }
    }

    fn emit(&self, out: &mut String, name: &str, entry: usize, insts: usize, adapters: &Adapters) {
        let mut offsets = Vec::new();
        let mut size = 0usize;
        for inst in &self.instructions {
            offsets.push(size);
            size += inst.bytes.len();
        }
        assert!(size <= u32::MAX as usize, "selection program too large");
        writeln!(
            out,
            "// {name}: entry @{:04x}; n0 = root; v = value slot; f = payload slot.",
            offsets[self.labels[entry]]
        )
        .unwrap();
        writeln!(
            out,
            "    #[rustfmt::skip]\n    const {name}_CODE: &[u8] = &["
        )
        .unwrap();
        for (index, inst) in self.instructions.iter().enumerate() {
            let op = Op::read(&mut Reader {
                bytes: &inst.bytes,
                pc: 0,
            });
            let mut description = self.describe(inst, adapters);
            if let Some(label) = inst.failure {
                write!(
                    description,
                    " {}@{:04x}",
                    if matches!(op, Op::Jump { .. }) {
                        ""
                    } else {
                        "else "
                    },
                    offsets[self.labels[label]]
                )
                .unwrap();
            }
            let decoded = format!(
                "{:04x} {:<14} {}",
                offsets[index],
                format!("{:?}", op.opcode()),
                description.trim()
            );
            writeln!(out, "        // @{}", decoded.trim_end()).unwrap();
            let mut bytes = inst.bytes.clone();
            if let Some(label) = inst.failure {
                let field = if matches!(op, Op::Jump { .. }) {
                    "target"
                } else {
                    "failure"
                };
                let offset = op.field_offset(field).expect("branch target field");
                bytes[offset..offset + 4]
                    .copy_from_slice(&veloc_bytecode::encode_u32(offsets[self.labels[label]]));
            }
            for byte in bytes {
                write!(out, " {byte},").unwrap();
            }
            writeln!(out).unwrap();
        }
        writeln!(
            out,
            "    ];\npub(super) const {name}: &Program = &Program {{ code: {name}_CODE, entry: 0x{:04x}, insts: {insts}, values: {}, fields: {},",
            offsets[self.labels[entry]], self.values, self.fields
        )
        .unwrap();
        writeln!(
            out,
            "types: &[{}],",
            self.types
                .iter()
                .map(|ts| format!("&[{}]", ts.join(",")))
                .collect::<Vec<_>>()
                .join(",")
        )
        .unwrap();
        writeln!(out, "integers: &[{}],", self.integers.join(",")).unwrap();
        writeln!(
            out,
            "opcodes: &[{}],",
            self.opcodes
                .iter()
                .map(|op| format!("veloc_lir::GenericOpcode::{op}"))
                .collect::<Vec<_>>()
                .join(",")
        )
        .unwrap();
        writeln!(
            out,
            "targets: &[{}],",
            self.targets.iter().cloned().collect::<Vec<_>>().join(",")
        )
        .unwrap();
        writeln!(
            out,
            "accesses: &[{}],",
            self.accesses
                .iter()
                .map(|(opcode, field, access)| adapters.field(opcode, field, *access))
                .collect::<Vec<_>>()
                .join(",")
        )
        .unwrap();
        writeln!(
            out,
            "features: &[{}],",
            self.features
                .iter()
                .map(|features| {
                    let set = features
                        .iter()
                        .fold("FeatureSet::empty()".to_owned(), |s, f| {
                            format!("{s}.with(Feature::{f})")
                        });
                    format!("{set}.as_words()")
                })
                .collect::<Vec<_>>()
                .join(",")
        )
        .unwrap();
        writeln!(
            out,
            "registers: &[{}], }};",
            self.registers
                .iter()
                .map(|reg| format!("Reg({reg})"))
                .collect::<Vec<_>>()
                .join(",")
        )
        .unwrap();
    }
}

fn resolve_field<'a>(plan: &'a Plan, root: &'a str, path: &'a str) -> (usize, &'a str, &'a str) {
    match path.split_once('.') {
        Some((owner, field)) => {
            let index = plan
                .definitions
                .iter()
                .position(|def| def.name == owner)
                .expect("checked definition binding");
            (index + 1, &plan.definitions[index].opcode, field)
        }
        None => (0, root, path),
    }
}

pub(in super::super) fn emit(
    out: &mut String,
    rules: &[&SelectRuleDef],
    extractors: &HashMap<String, ExtractorDef>,
    instructions: &HashMap<String, FinalInstDef>,
    regs: &HashMap<String, u32>,
    adapters: &mut Adapters,
) {
    let plan = Plan::prepare(rules, extractors, instructions);
    let mut graph = Graph::default();
    let entry = graph.compile(plan.candidates.clone());
    graph.validate(entry, &plan);
    let mut code = Code::default();
    for node in &graph.nodes {
        code.labels.push(code.instructions.len());
        match node {
            Node::Reject => code.op(Op::Reject {}),
            Node::Accept(rule) => {
                code.recipe(&plan, &plan.rules[*rule], adapters, instructions, regs)
            }
            Node::Check { test, yes, no } => {
                code.test(&plan, adapters, &rules[0].opcode, &plan.tests[*test], *no);
                code.branch(Op::Jump { target: 0 }, *yes);
            }
        }
    }
    let name = sanitize_ident(&rules[0].opcode).to_ascii_uppercase();
    code.emit(out, &name, entry, plan.definitions.len() + 1, adapters);
}
