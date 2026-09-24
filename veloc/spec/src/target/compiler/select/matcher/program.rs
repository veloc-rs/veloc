//! Compile the shared matching graph and construction recipes to bytecode.
//! Matching uses tables; schema-generated constructors install complete instructions.
use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Access {
    Reg,
    Attribute,
}

pub(in super::super) struct Adapters<'a> {
    layouts: &'a BTreeMap<String, crate::storage::operands::Projection>,
    predicates: Vec<String>,
    encodings: BTreeMap<&'static str, (usize, bool)>,
    builders: BTreeMap<String, String>,
}
fn intern<T: PartialEq>(items: &mut Vec<T>, item: T) -> usize {
    if let Some(id) = items.iter().position(|old| *old == item) {
        id
    } else {
        let id = items.len();
        items.push(item);
        id
    }
}

impl<'a> Adapters<'a> {
    pub(in super::super) fn new(
        layouts: &'a BTreeMap<String, crate::storage::operands::Projection>,
    ) -> Self {
        Self {
            layouts,
            predicates: Vec::new(),
            encodings: BTreeMap::new(),
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
        writeln!(out, "const _: () = {{").unwrap();
        for (op, (arity, branch)) in &self.encodings {
            writeln!(out, "let (arity, branch) = crate::isel::matching::Op::{op}.format(); assert!(arity == {arity} && branch == {branch});").unwrap();
        }
        writeln!(out, "}};").unwrap();
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
    op: &'static str,
    args: Vec<usize>,
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
fn uleb(value: usize) -> Vec<u8> {
    let mut value = u32::try_from(value).expect("selection index overflow");
    let mut bytes = Vec::new();
    loop {
        let byte = (value & 127) as u8;
        value >>= 7;
        bytes.push(byte | if value != 0 { 128 } else { 0 });
        if value == 0 {
            return bytes;
        }
    }
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
    fn op(&mut self, op: &'static str, args: &[usize]) {
        self.instructions.push(Instruction {
            op,
            args: args.into(),
            failure: None,
        });
    }
    fn branch(&mut self, op: &'static str, args: &[usize], target: usize) {
        self.instructions.push(Instruction {
            op,
            args: args.into(),
            failure: Some(target),
        });
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
        self.op("ReadReg", &[dst, node, field]);
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
                self.branch("GetDef", &[slot + 1, 0], failure);
                let opcode = intern(&mut self.opcodes, def.opcode.clone());
                self.branch("CheckOpcode", &[slot + 1, opcode], failure);
            }
            Test::Field {
                field,
                schema,
                guard,
            } => match guard {
                Guard::Types(types) => {
                    self.read_reg(plan, adapters, root, field, 0);
                    let set = intern(&mut self.types, types.clone());
                    self.branch("CheckType", &[0, set], failure);
                }
                Guard::Extractor(name) => {
                    self.read_reg(plan, adapters, root, field, 0);
                    let predicate = intern(&mut adapters.predicates, name.clone());
                    self.branch("CallPredicate", &[0, predicate], failure);
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
                    self.branch("CheckInt", &[node, field, constant], failure);
                }
            },
            Test::Features(features) => {
                let set = intern(&mut self.features, features.clone());
                self.branch("CheckFeatures", &[set], failure);
            }
            Test::Foldable(slot) => self.branch("CheckFoldable", &[slot + 1, 0], failure),
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
        self.op("Accept", &[]);
        let fields = collect_field_variable_bindings(&rule.fields);
        let mut temps = HashMap::new();
        let mut values = 1; // Slot zero is matching scratch, not a rule binding.
        let mut payloads = 0;
        for (name, ty) in &rule.temps {
            temps.insert(name.as_str(), values);
            let ty = intern(&mut self.types, vec![ty.clone()]);
            self.op("MakeTemp", &[values, ty]);
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
                    self.op("ReadResult", &[values, source_result]);
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
                                    self.op("ConstReg", &[dst, reg]);
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
                            self.op("ConstImm", &[dst, imm]);
                        }
                        Constructor::Variable(name) => {
                            let (node, schema, field) =
                                resolve_field(plan, &rule.opcode, &fields[name]);
                            let access = Access::Attribute;
                            let field = self.field(adapters, schema, field, access);
                            self.op("ReadField", &[dst, node, field]);
                        }
                        _ => panic!("invalid target payload"),
                    }
                    dst
                };
                lists[category].push(slot);
            }
            assert_eq!(cursor, args.len(), "target operand count");
            let builder = adapters.builder(opcode, &rule.opcode, definition);
            let mut encoded = vec![intern(&mut self.targets, builder)];
            for list in lists {
                encoded.push(list.len());
                encoded.extend(list);
            }
            builds.push(encoded);
        }
        // Read all source fields before any target instruction is installed.
        for build in builds {
            self.op("BuildInst", &build);
        }
        self.op("Finish", &[]);
        self.values = self.values.max(values);
        self.fields = self.fields.max(payloads);
    }
    fn describe(&self, inst: &Instruction, adapters: &Adapters) -> String {
        let args = &inst.args;
        let field = |id: usize| {
            let (schema, name, _) = &self.accesses[id];
            format!("{schema}.{name}")
        };
        match inst.op {
            "ReadReg" => format!("v{} <- n{} {}", args[0], args[1], field(args[2])),
            "ReadField" => format!("f{} <- n{} {}", args[0], args[1], field(args[2])),
            "GetDef" => format!("n{} <- def(v{})", args[0], args[1]),
            "CheckOpcode" => format!("n{} == {}", args[0], self.opcodes[args[1]]),
            "CheckType" => format!("v{} in [{}]", args[0], self.types[args[1]].join(", ")),
            "CheckInt" => format!(
                "n{} {} == {}",
                args[0],
                field(args[1]),
                self.integers[args[2]]
            ),
            "CheckFeatures" => self.features[args[0]].join(" + "),
            "CallPredicate" => format!("{}(v{})", adapters.predicates[args[1]], args[0]),
            "CheckFoldable" => format!("n{} into n{}", args[0], args[1]),
            "MakeTemp" => format!("v{}: {}", args[0], self.types[args[1]].join(", ")),
            "ReadResult" => format!("v{} <- root.results[{}]", args[0], args[1]),
            "ConstReg" => format!("v{} <- preg{}", args[0], self.registers[args[1]]),
            "ConstImm" => format!("f{} <- {}", args[0], self.integers[args[1]]),
            "BuildInst" => {
                let mut cursor = 1;
                let mut text = self.targets[args[0]].clone();
                for (name, prefix) in [("results", "v"), ("inputs", "v"), ("fields", "f")] {
                    let len = args[cursor];
                    cursor += 1;
                    let slots = args[cursor..cursor + len]
                        .iter()
                        .map(|id| format!("{prefix}{id}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    write!(text, " {name}=[{slots}]").unwrap();
                    cursor += len;
                }
                text
            }
            "Accept" => "begin construction (no fallback)".into(),
            "Finish" => "return replacement to driver".into(),
            "Reject" | "Jump" => String::new(),
            _ => unreachable!("unknown generated selection opcode"),
        }
    }

    fn emit(&self, out: &mut String, name: &str, entry: usize, insts: usize, adapters: &Adapters) {
        let mut offsets = Vec::new();
        let mut size = 0usize;
        for inst in &self.instructions {
            offsets.push(size);
            size += 1
                + inst.args.iter().map(|&a| uleb(a).len()).sum::<usize>()
                + if inst.failure.is_some() { 4 } else { 0 };
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
            let mut description = self.describe(inst, adapters);
            if let Some(label) = inst.failure {
                write!(
                    description,
                    " {}@{:04x}",
                    if inst.op == "Jump" { "" } else { "else " },
                    offsets[self.labels[label]]
                )
                .unwrap();
            }
            let decoded = format!(
                "{:04x} {:<14} {}",
                offsets[index],
                inst.op,
                description.trim()
            );
            writeln!(out, "        // @{}", decoded.trim_end()).unwrap();
            write!(out, "        Op::{} as u8,", inst.op).unwrap();
            for &arg in &inst.args {
                for byte in uleb(arg) {
                    write!(out, " {byte},").unwrap();
                }
            }
            if let Some(label) = inst.failure {
                let dest = u32::try_from(offsets[self.labels[label]]).unwrap();
                for byte in dest.to_le_bytes() {
                    write!(out, " {byte},").unwrap();
                }
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
            Node::Reject => code.op("Reject", &[]),
            Node::Accept(rule) => {
                code.recipe(&plan, &plan.rules[*rule], adapters, instructions, regs)
            }
            Node::Check { test, yes, no } => {
                code.test(&plan, adapters, &rules[0].opcode, &plan.tests[*test], *no);
                code.branch("Jump", &[], *yes);
            }
        }
    }
    for inst in &code.instructions {
        let format = (
            if inst.op == "BuildInst" {
                1
            } else {
                inst.args.len()
            },
            inst.failure.is_some(),
        );
        if let Some(previous) = adapters.encodings.insert(inst.op, format) {
            assert_eq!(previous, format, "inconsistent opcode encoding");
        }
    }
    let name = sanitize_ident(&rules[0].opcode).to_ascii_uppercase();
    code.emit(out, &name, entry, plan.definitions.len() + 1, adapters);
}
