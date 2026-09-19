//! Compile the shared matching graph and construction recipes to bytecode.
//! Rust is emitted only for schema accessors and declared host predicates.
use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Access {
    Reg,
    Integer,
    Condition,
    Imm,
    Edge,
    Global,
    StackSlot,
}

#[derive(Default)]
pub(in super::super) struct Adapters {
    fields: Vec<(String, String, Access)>,
    features: Vec<Vec<String>>,
    predicates: Vec<String>,
    encodings: BTreeMap<&'static str, (usize, bool)>,
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

impl Adapters {
    fn field(&mut self, schema: &str, field: &str, access: Access) -> usize {
        intern(&mut self.fields, (schema.into(), field.into(), access))
    }
    pub(in super::super) fn emit(
        &self,
        out: &mut String,
        context: &str,
        extractors: &HashMap<String, ExtractorDef>,
        decls: &HashMap<String, DeclDef>,
    ) {
        // Symbolic opcodes avoid a second numeric opcode registry. Check the
        // operand encoding contract as well, when rustc compiles the output.
        writeln!(out, "const _: () = {{").unwrap();
        for (op, (arity, branch)) in &self.encodings {
            writeln!(out, "let (arity, branch) = crate::passes::isel::matching::Op::{op}.format(); assert!(arity == {arity} && branch == {branch});").unwrap();
        }
        writeln!(out, "}};").unwrap();
        writeln!(out, "struct SelectorHost<'a, C>(&'a mut C);").unwrap();
        writeln!(out, "impl<C: LoweringContext + crate::target::arch::TargetFeatures<Features = FeatureSet> + {context}> crate::passes::isel::matching::Host for SelectorHost<'_, C> {{").unwrap();
        for (method, ty, category) in [
            ("read_reg", "Option<Reg>", 0),
            ("read_int", "i64", 1),
            ("read_field", "InstField", 2),
        ] {
            writeln!(out, "fn {method}(&self, inst: veloc_lir::InstRef<'_>, field: u32) -> {ty} {{ use veloc_lir::InstRead; match field {{").unwrap();
            for (id, (schema, field, access)) in self.fields.iter().enumerate() {
                let group = match access {
                    Access::Reg => 0,
                    Access::Integer | Access::Condition => 1,
                    _ => 2,
                };
                if group != category {
                    continue;
                }
                let expr = match access {
                    Access::Reg => format!("reg_value(n.{field})"),
                    Access::Integer => format!("n.{field}.into()"),
                    Access::Condition => format!("n.{field} as i64"),
                    Access::Imm => format!("InstField::Imm(n.{field}.into())"),
                    Access::Edge => format!("InstField::Edge(n.{field})"),
                    Access::Global => format!("InstField::Global(n.{field})"),
                    Access::StackSlot => format!("InstField::StackSlot(n.{field})"),
                };
                writeln!(out, "{id} => {{ let veloc_lir::InstView::{schema}(n) = inst.view() else {{ panic!(\"selection field schema mismatch\") }}; {expr} }},").unwrap();
            }
            writeln!(out, "_ => unreachable!(\"selection field ID\"), }} }}").unwrap();
        }
        writeln!(
            out,
            "fn ty(&self, reg: Reg) -> Option<Type> {{ reg.as_vreg().map(|v| self.0.get_type(v)) }}"
        )
        .unwrap();
        writeln!(
            out,
            "fn features(&self, id: u32) -> bool {{ const SETS: &[FeatureSet] = &["
        )
        .unwrap();
        for features in &self.features {
            let expr = features
                .iter()
                .fold("FeatureSet::empty()".to_owned(), |s, f| {
                    format!("{s}.with(Feature::{f})")
                });
            writeln!(out, "{expr},").unwrap();
        }
        writeln!(out, "]; self.0.supports_features(SETS[id as usize]) }}").unwrap();
        writeln!(out, "fn predicate(&self, id: u32, reg: Reg) -> bool {{ let _ctx = &*self.0; let Some(_v) = reg.as_vreg() else {{ return false }}; match id {{").unwrap();
        for (id, name) in self.predicates.iter().enumerate() {
            let condition = generate_pattern_condition(&extractors[name].body, "_v", decls)
                .replace("ctx.", "_ctx.");
            writeln!(out, "{id} => {condition},").unwrap();
        }
        writeln!(out, "_ => unreachable!(\"selection predicate ID\"), }} }}").unwrap();
        writeln!(
            out,
            "fn temporary(&mut self, ty: Type) -> Reg {{ self.0.alloc_tmp(ty) }}"
        )
        .unwrap();
        writeln!(out, "fn build(&self, target: u32, writer: veloc_lir::InstWriter<'_>, results: &[Reg], inputs: &[Reg], fields: &[InstField]) -> veloc_lir::InstId {{ TargetInst::from_u32(target).write(writer, results, inputs, fields) }}").unwrap();
        writeln!(out, "}}").unwrap();
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
        let field = adapters.field(schema, field, Access::Reg);
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
                        Guard::Integer(value) => (Access::Integer, value.to_string()),
                        Guard::Condition(cc) => (
                            Access::Condition,
                            format!(
                                "{} as i64",
                                render_cond_code_match(schema, *cc).expect("checked condition")
                            ),
                        ),
                        _ => unreachable!(),
                    };
                    let (node, schema, field) = resolve_field(plan, root, field);
                    let field = adapters.field(schema, field, access);
                    let constant = intern(&mut self.integers, constant);
                    self.branch("CheckInt", &[node, field, constant], failure);
                }
            },
            Test::Features(features) => {
                let set = intern(&mut adapters.features, features.clone());
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
                                    self.read_reg(plan, adapters, &rule.schema, &fields[name], dst)
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
                                resolve_field(plan, &rule.schema, &fields[name]);
                            let access = match operand {
                                OperandConstraint::Imm(_) => Access::Imm,
                                OperandConstraint::Block(_) => Access::Edge,
                                OperandConstraint::Global(_) => Access::Global,
                                OperandConstraint::StackSlot(_) => Access::StackSlot,
                                _ => unreachable!(),
                            };
                            let field = adapters.field(schema, field, access);
                            self.op("ReadField", &[dst, node, field]);
                        }
                        _ => panic!("invalid target payload"),
                    }
                    dst
                };
                lists[category].push(slot);
            }
            assert_eq!(cursor, args.len(), "target operand count");
            let mut encoded = vec![intern(&mut self.targets, opcode.clone())];
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
            let (schema, name, _) = &adapters.fields[id];
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
            "CheckFeatures" => adapters.features[args[0]].join(" + "),
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
            self.targets
                .iter()
                .map(|op| format!("TargetInst::{op}.as_u32()"))
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
            (index + 1, &plan.definitions[index].schema, field)
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
                code.test(&plan, adapters, &rules[0].schema, &plan.tests[*test], *no);
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
