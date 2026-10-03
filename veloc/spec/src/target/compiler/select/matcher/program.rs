//! Compile the shared matching graph and construction recipes to bytecode.
//! Matching uses tables; schema-generated constructors install complete instructions.
use super::*;
use crate::bytecode::intern;
use crate::target::AttributeKind;
use veloc_bytecode::{Lebs, Reader, selection::Instruction as Op};

pub(in super::super) struct Adapters<'a> {
    layouts: &'a BTreeMap<String, crate::storage::operands::Projection>,
    payloads: Option<&'a crate::storage::payload::Payloads>,
    predicates: Vec<String>,
    builders: BTreeMap<String, String>,
}
impl<'a> Adapters<'a> {
    pub(in super::super) fn new(
        layouts: &'a BTreeMap<String, crate::storage::operands::Projection>,
        payloads: Option<&'a crate::storage::payload::Payloads>,
    ) -> Self {
        Self {
            layouts,
            payloads,
            predicates: Vec::new(),
            builders: BTreeMap::new(),
        }
    }
    fn builder(&mut self, opcode: &str, source: &str, definition: &FinalInstDef) -> String {
        use crate::storage::operands::{Domain, Shape};
        let call = definition
            .operands
            .iter()
            .any(|op| matches!(op, OperandConstraint::Attribute(_, AttributeKind::Call)));
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
        let mut params = vec![format!(
            "{}writer: veloc_lir::InstWriter<'_>",
            if call
                || definition
                    .operands
                    .iter()
                    .any(|op| matches!(op, OperandConstraint::Attribute(_, AttributeKind::Blocks)))
            {
                "mut "
            } else {
                ""
            }
        )];
        let mut args = Vec::new();
        let mut results = Vec::new();
        let mut inputs = Vec::new();
        let mut fields = Vec::new();
        let mut reads = String::new();
        for op in &definition.operands {
            let attribute = match op {
                OperandConstraint::Attribute(_, kind) => Some(kind.description()),
                OperandConstraint::Def(_) | OperandConstraint::Use(_) => None,
            };
            let ty = attribute.as_ref().map_or("Reg", |ty| ty.rust_type);
            let name = format!("operand_{}", sanitize_ident(op.name()));
            params.push(format!("{name}: {ty}"));
            args.push(
                if matches!(op, OperandConstraint::Attribute(_, AttributeKind::Blocks)) {
                    format!("&{name}")
                } else {
                    name.clone()
                },
            );
            if let Some(attribute) = attribute {
                let variant = attribute.field_variant;
                if matches!(op, OperandConstraint::Attribute(_, AttributeKind::Blocks)) {
                    writeln!(reads, "let {name}: Vec<_> = fields.by_ref().map(|v| {{ let FieldValue::Edge(edge) = v else {{ unreachable!(\"successor field\") }}; edge }}).collect();").unwrap();
                    fields.push((variant, name, Shape::Sequence));
                    continue;
                }
                writeln!(reads, "let FieldValue::{variant}({name}) = fields.next().expect(\"generated field\") else {{ unreachable!(\"generated field type\") }};").unwrap();
                fields.push((variant, name, Shape::One));
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
        let returns = definition.flow == "Return";
        if !self.builders.contains_key(&build) {
            let mut body = String::new();
            if call {
                params.extend(["abi_args: &[Reg]".into(), "abi_results: &[Reg]".into()]);
            }
            if returns {
                params.push("abi_uses: &[Reg]".into());
            }
            writeln!(
                body,
                "fn {build}({}) -> veloc_lir::InstId {{",
                params.join(", ")
            )
            .unwrap();
            let payload = self
                .payloads
                .expect("selection requires field layouts")
                .construct(&fields, "writer", "veloc_lir::Fields")
                .expect("checked target field layout");
            writeln!(body, "let fields = {payload};").unwrap();
            if call {
                writeln!(body, "let mut inputs = smallvec::SmallVec::<[Reg; 8]>::from_slice(&[{}]); inputs.extend_from_slice(abi_args);", inputs.join(", ")).unwrap();
                writeln!(body, "let mut results = smallvec::SmallVec::<[Reg; 4]>::from_slice(&[{}]); results.extend_from_slice(abi_results);", results.join(", ")).unwrap();
                writeln!(
                    body,
                    "TargetInst::{opcode}.write(writer, &results, &inputs, fields)"
                )
                .unwrap();
            } else if returns {
                writeln!(
                    body,
                    "TargetInst::{opcode}.write(writer, &[{}], abi_uses, fields)",
                    results.join(", ")
                )
                .unwrap();
            } else {
                writeln!(
                    body,
                    "TargetInst::{opcode}.write(writer, &[{}], &[{}], fields)",
                    results.join(", "),
                    inputs.join(", ")
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
            let result = self.layouts[source]
                .members
                .iter()
                .find(|m| m.domain == Domain::Result && m.field.shape == Shape::Sequence)
                .expect("call construction requires source ABI results");
            writeln!(body, "let source = store.inst(_source);").unwrap();
            writeln!(body, "let abi_args = smallvec::SmallVec::<[Reg; 8]>::from_slice(&source.inputs()[{}..]);", input.index).unwrap();
            writeln!(body, "let abi_results = smallvec::SmallVec::<[Reg; 4]>::from_slice(&source.results()[{}..]);", result.index).unwrap();
            // ABI operands are appended after the target's declared operands.
            // Generate their mapping from both schemas, not from runtime values.
            writeln!(body, "let map_input = |index: usize| index.checked_sub({}).expect(\"constraint outside ABI arguments\") + {};", input.index, inputs.len()).unwrap();
            writeln!(body, "let map_result = |index: usize| index.checked_sub({}).expect(\"constraint outside ABI results\") + {};", result.index, results.len()).unwrap();
            body.push_str(
                r#"let constraints = source.constraints().iter().copied().map(|mut constraint| {
    constraint.operand = match constraint.operand {
        veloc_lir::OperandRef::Input(index) => veloc_lir::OperandRef::Input(map_input(index)),
        veloc_lir::OperandRef::Result(index) => veloc_lir::OperandRef::Result(map_result(index)),
    };
    if let veloc_lir::Placement::Reuse(index) = &mut constraint.placement {
        *index = map_input(*index);
    }
    constraint
}).collect();
"#,
            );
            args.extend(["&abi_args".into(), "&abi_results".into()]);
        }
        if returns {
            body.push_str("let abi_uses = smallvec::SmallVec::<[Reg; 4]>::from_slice(store.inst(_source).inputs());\n");
            body.push_str("let constraints = store.inst(_source).constraints().to_vec();\n");
            args.push("&abi_uses".into());
        }
        let arguments = if args.is_empty() {
            String::new()
        } else {
            format!(", {}", args.join(", "))
        };
        let writer = if call || returns {
            "store.writer().with_constraints(constraints)"
        } else {
            "store.writer()"
        };
        writeln!(body, "{build}({writer}{arguments})\n}}").unwrap();
        self.builders.insert(adapter.clone(), body);
        adapter
    }
    fn member(&self, opcode: &str, field: &str) -> &crate::storage::operands::Member {
        let member = self.layouts[opcode]
            .members
            .iter()
            .find(|member| member.field.name == field)
            .expect("checked storage field");
        assert!(
            member.field.shape != crate::storage::operands::Shape::Sequence,
            "scalar selector access cannot read sequence {opcode}.{field}"
        );
        member
    }
    fn operand(&self, opcode: &str, field: &str) -> veloc_bytecode::OperandRef {
        use crate::storage::operands::Domain;
        use veloc_bytecode::OperandRef;
        let member = self.member(opcode, field);
        assert!(member.field.codec.is_none(), "expected register field");
        match member.domain {
            Domain::Input => OperandRef::Input(member.index),
            Domain::Result => OperandRef::Result(member.index),
            Domain::Attribute => panic!("attribute used as a register"),
        }
    }
    fn attribute(&self, opcode: &str, field: &str) -> usize {
        let member = self.member(opcode, field);
        assert!(
            matches!(member.domain, crate::storage::operands::Domain::Attribute),
            "expected attribute field"
        );
        member.index
    }
    pub(in super::super) fn emit(
        &self,
        out: &mut String,
        context: Option<&str>,
        extractors: &HashMap<String, ExtractorDef>,
        decls: &HashMap<String, DeclDef>,
    ) {
        for builder in self.builders.values() {
            out.push_str(builder);
        }
        if self.predicates.is_empty() {
            return;
        }
        let context = context.expect("custom predicates have a checked context");
        writeln!(out, "pub fn selection_predicate<C: {context}>(_ctx: &C, id: u32, reg: Reg) -> bool {{ let Some(_v) = reg.as_vreg() else {{ return false }}; match id {{").unwrap();
        for (id, name) in self.predicates.iter().enumerate() {
            let condition = generate_pattern_condition(&extractors[name].body, "_v", decls)
                .replace("ctx.", "_ctx.");
            writeln!(out, "{id} => {condition},").unwrap();
        }
        writeln!(out, "_ => unreachable!(\"selection predicate ID\"), }} }}").unwrap();
    }
}

/// One independently compiled matching graph in the shared bytecode program.
struct Entry {
    opcode: String,
    label: usize,
    insts: usize,
    values: usize,
    fields: usize,
}

/// Rust evaluates symbolic constants after the assembler resolves byte offsets.
/// Fixed-width operands keep those offsets stable during initialization.
struct InlineConstant {
    instruction: usize,
    offset: usize,
    expression: String,
    width: usize,
}

#[derive(Default)]
struct Code {
    asm: crate::bytecode::Assembler,
    types: Vec<Vec<String>>,
    constants: Vec<InlineConstant>,
    targets: Vec<String>,
    registers: Vec<u32>,
    features: Vec<Vec<String>>,
    values: usize,
    fields: usize,
}
impl Code {
    fn op(&mut self, op: Op<'_>) {
        self.asm.emit(op);
    }
    fn branch(&mut self, op: Op<'_>, target: usize) {
        let field = if matches!(op, Op::Jump { .. }) {
            "target"
        } else {
            "failure"
        };
        self.asm.branch(op, field, target);
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
        let operand = adapters.operand(schema, field);
        self.op(Op::ReadReg { dst, node, operand });
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
            Test::SameValue(lhs, rhs) => {
                self.read_reg(plan, adapters, root, lhs, 0);
                self.read_reg(plan, adapters, root, rhs, 1);
                self.branch(
                    Op::CheckSameValue {
                        lhs: 0,
                        rhs: 1,
                        failure: 0,
                    },
                    failure,
                );
            }
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
                let op = Op::CheckOpcode {
                    node: slot + 1,
                    opcode: 0,
                    failure: 0,
                };
                self.constants.push(InlineConstant {
                    instruction: self.asm.instructions.len(),
                    offset: op.field_offset("opcode").expect("opcode operand"),
                    expression: format!("veloc_lir::GenericOpcode::{} as u32", def.opcode),
                    width: 4,
                });
                self.branch(op, failure);
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
                Guard::IntRange { bits, signed } => {
                    let (node, schema, field) = resolve_field(plan, root, field);
                    let index = adapters.attribute(schema, field);
                    self.branch(
                        Op::CheckIntRange {
                            node,
                            index,
                            bits: usize::from(*bits),
                            signed: usize::from(*signed),
                            failure: 0,
                        },
                        failure,
                    );
                }
                Guard::Integer(_) | Guard::Condition(_) => {
                    let constant = match guard {
                        Guard::Integer(value) => *value,
                        Guard::Condition(_) => 0, // Patched by Rust during static initialization.
                        _ => unreachable!(),
                    };
                    let (node, opcode, field) = resolve_field(plan, root, field);
                    let index = adapters.attribute(opcode, field);
                    let op = Op::CheckInt {
                        node,
                        index,
                        constant,
                        failure: 0,
                    };
                    if let Guard::Condition(cc) = guard {
                        self.constants.push(InlineConstant {
                            instruction: self.asm.instructions.len(),
                            offset: op.field_offset("constant").expect("integer operand"),
                            width: 8,
                            expression: format!(
                                "{} as i64",
                                render_cond_code_match(schema, *cc).expect("checked condition")
                            ),
                        });
                    }
                    self.branch(op, failure);
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
                    OperandConstraint::Use(_) => 1,
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
                            self.op(Op::ConstImm { dst, imm: *value });
                        }
                        Constructor::Variable(name) => {
                            let (node, schema, field) =
                                resolve_field(plan, &rule.opcode, &fields[name]);
                            if matches!(
                                operand,
                                OperandConstraint::Attribute(_, AttributeKind::Blocks)
                            ) {
                                self.op(Op::ReadSuccessors { dst, node });
                            } else {
                                let index = adapters.attribute(schema, field);
                                self.op(Op::ReadField { dst, node, index });
                            }
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
    fn describe(&self, inst: &[u8], adapters: &Adapters) -> String {
        let op = Op::read(&mut Reader { bytes: inst, pc: 0 });
        match op {
            Op::ReadReg { dst, node, operand } => format!("v{dst} <- n{node} {operand:?}"),
            Op::ReadSuccessors { dst, node } => format!("f{dst} <- n{node} successors"),
            Op::ReadField { dst, node, index } => format!("f{dst} <- n{node} attribute {index:?}"),
            Op::GetDef { dst, value, .. } => format!("n{dst} <- def(v{value})"),
            Op::CheckOpcode { node, opcode, .. } => format!("n{node} opcode == {opcode}"),
            Op::CheckType { value, set, .. } => {
                format!("v{value} in [{}]", self.types[set].join(", "))
            }
            Op::CheckSameValue { lhs, rhs, .. } => format!("v{lhs} == v{rhs}"),
            Op::CheckInt {
                node,
                index,
                constant,
                ..
            } => format!("n{node} attribute {index:?} == {constant}"),
            Op::CheckIntRange {
                node,
                index,
                bits,
                signed,
                ..
            } => {
                format!(
                    "n{node} attribute {index:?} fits {}{bits}",
                    if signed != 0 { "i" } else { "u" }
                )
            }
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
            Op::ConstImm { dst, imm } => format!("f{dst} <- {imm}"),
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

    fn emit(&self, out: &mut String, entries: &[Entry], adapters: &Adapters) {
        let encoded = self.asm.finish();
        writeln!(
            out,
            "// Selection bytecode: n0 = root; v = value slot; f = payload slot."
        )
        .unwrap();
        for entry in entries {
            writeln!(
                out,
                "// {}: entry @{:04x}; insts = {}, values = {}, fields = {}.",
                entry.opcode, encoded.labels[entry.label], entry.insts, entry.values, entry.fields
            )
            .unwrap();
        }
        writeln!(
            out,
            "#[rustfmt::skip]\nconst SELECTION_CODE: &[u8] = &{{ let mut code = ["
        )
        .unwrap();
        for (index, bytes) in encoded.instructions.iter().enumerate() {
            let op = Op::read(&mut Reader { bytes, pc: 0 });
            writeln!(
                out,
                "    // @{:04x} {:?}: {}",
                encoded.offsets[index],
                op,
                self.describe(bytes, adapters)
            )
            .unwrap();
            for byte in bytes {
                write!(out, " {byte},").unwrap();
            }
            writeln!(out).unwrap();
        }
        writeln!(out, "];").unwrap();
        for constant in &self.constants {
            let offset = encoded.offsets[constant.instruction] + constant.offset;
            writeln!(
                out,
                "// Inline constant at @{offset:04x}: {}",
                constant.expression
            )
            .unwrap();
            writeln!(out, "let bytes = ({}).to_le_bytes();", constant.expression).unwrap();
            for byte in 0..constant.width {
                writeln!(out, "code[{}] = bytes[{byte}];", offset + byte).unwrap();
            }
        }
        writeln!(out, "code }};\npub static SELECTION_PROGRAM: crate::passes::isel::matching::Program = crate::passes::isel::matching::Program {{ code: SELECTION_CODE,").unwrap();
        writeln!(
            out,
            "entries: {{ let mut entries = [None; veloc_lir::GenericOpcode::COUNT];"
        )
        .unwrap();
        for entry in entries {
            writeln!(out,
                "entries[veloc_lir::GenericOpcode::{} as usize] = Some(crate::passes::isel::matching::Entry {{ offset: {}, insts: {}, values: {}, fields: {} }});",
                entry.opcode, encoded.labels[entry.label], entry.insts, entry.values, entry.fields).unwrap();
        }
        writeln!(out, "entries }},").unwrap();
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
        writeln!(
            out,
            "targets: &[{}],",
            self.targets.iter().cloned().collect::<Vec<_>>().join(",")
        )
        .unwrap();
        writeln!(
            out,
            "required_features: &[{}],",
            self.features
                .iter()
                .map(|features| {
                    let set = features
                        .iter()
                        .fold("FeatureSet::empty()".to_owned(), |s, f| {
                            format!("{s}.with(Feature::{f})")
                        });
                    format!("crate::target::FeatureSetRef::new({set}.as_words())")
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
    groups: &[&[&SelectRuleDef]],
    extractors: &HashMap<String, ExtractorDef>,
    instructions: &HashMap<String, FinalInstDef>,
    regs: &HashMap<String, u32>,
    adapters: &mut Adapters,
) {
    let mut code = Code::default();
    let mut entries = Vec::new();
    let mut label_base = 0;
    for rules in groups {
        let plan = Plan::prepare(rules, extractors, instructions);
        let mut graph = Graph::default();
        let entry = graph.compile(plan.candidates.clone());
        graph.validate(entry, &plan);
        // Pools and bytecode are shared; scratch slots are local to this entry.
        code.values = 0;
        code.fields = 0;
        for (index, node) in graph.nodes.iter().enumerate() {
            let label = code.asm.label();
            assert_eq!(label, label_base + index);
            match node {
                Node::Reject => code.op(Op::Reject {}),
                Node::Accept(rule) => {
                    code.recipe(&plan, &plan.rules[*rule], adapters, instructions, regs)
                }
                Node::Check { test, yes, no } => {
                    code.test(
                        &plan,
                        adapters,
                        &rules[0].opcode,
                        &plan.tests[*test],
                        label_base + *no,
                    );
                    code.branch(Op::Jump { target: 0 }, label_base + *yes);
                }
            }
        }
        entries.push(Entry {
            opcode: rules[0].opcode.clone(),
            label: label_base + entry,
            insts: plan.definitions.len() + 1,
            values: code.values,
            fields: code.fields,
        });
        label_base += graph.nodes.len();
    }
    code.emit(out, &entries, adapters);
}
