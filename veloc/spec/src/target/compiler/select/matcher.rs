//! Ordered, pure matching IR. Rule order is preserved on both success and
//! failure edges; Rust expressions are rendered only after graph construction.
use super::*;
use crate::target::ast::DefMatch;
mod program;
pub(super) use program::{Adapters, emit};

#[derive(Clone, Debug, PartialEq)]
enum Guard {
    Types(Vec<String>),
    Integer(i64),
    Condition(CondCode),
    Extractor(String),
}

#[derive(Clone, Debug, PartialEq)]
enum Test {
    Definition(usize),
    Field {
        field: String,
        schema: String,
        guard: Guard,
    },
    Features(Vec<String>),
    Foldable(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Node {
    Reject,
    Accept(usize),
    Check { test: usize, yes: usize, no: usize },
}

type Candidates = Vec<(usize, Vec<usize>)>;

#[derive(Default)]
struct Graph {
    nodes: Vec<Node>,
    interned: HashMap<Node, usize>,
    states: BTreeMap<(Candidates, usize), usize>,
}

impl Graph {
    /// Check definite initialization over the acyclic decision graph. A cache
    /// slot left by a failed candidate must never satisfy another path's reads.
    fn validate(&self, entry: usize, plan: &Plan) {
        let slot = |path: &str| {
            path.split_once('.').map(|(owner, _)| {
                plan.definitions
                    .iter()
                    .position(|def| def.name == owner)
                    .unwrap()
            })
        };
        let mut incoming = vec![None::<BTreeSet<usize>>; self.nodes.len()];
        incoming[entry] = Some(BTreeSet::new());
        for id in (0..self.nodes.len()).rev() {
            let Some(defined) = incoming[id].take() else {
                continue;
            };
            match self.nodes[id] {
                Node::Reject => {}
                Node::Accept(rule) => {
                    for def in &plan.rules[rule].definitions {
                        let index = plan
                            .definitions
                            .iter()
                            .position(|d| d.name == def.name)
                            .unwrap();
                        assert!(
                            defined.contains(&index),
                            "recipe reads an unmatched definition"
                        );
                    }
                }
                Node::Check { test, yes, no } => {
                    let needed = match &plan.tests[test] {
                        Test::Definition(index) => slot(&plan.definitions[*index].input),
                        Test::Field { field, .. } => slot(field),
                        Test::Foldable(index) => Some(*index),
                        Test::Features(_) => None,
                    };
                    if let Some(needed) = needed {
                        assert!(
                            defined.contains(&needed),
                            "test reads an unmatched definition"
                        );
                    }
                    let mut success = defined.clone();
                    let mut failure = defined;
                    if let Test::Definition(index) = plan.tests[test] {
                        success.insert(index);
                        failure.remove(&index);
                    }
                    for (next, state) in [(yes, success), (no, failure)] {
                        assert!(next < id, "matcher graph must be acyclic and postordered");
                        match &mut incoming[next] {
                            Some(old) => old.retain(|slot| state.contains(slot)),
                            old @ None => *old = Some(state),
                        }
                    }
                }
            }
        }
    }

    fn node(&mut self, node: Node) -> usize {
        if let Some(&id) = self.interned.get(&node) {
            return id;
        }
        let id = self.nodes.len();
        self.nodes.push(node);
        self.interned.insert(node, id);
        id
    }

    fn compile(&mut self, candidates: Candidates) -> usize {
        let reject = self.node(Node::Reject);
        self.sequence(candidates, reject)
    }

    fn sequence(&mut self, candidates: Candidates, fallback: usize) -> usize {
        let key = (candidates.clone(), fallback);
        if let Some(&id) = self.states.get(&key) {
            return id;
        }
        let id = match candidates.first() {
            None => fallback,
            Some((rule, tests)) if tests.is_empty() => self.node(Node::Accept(*rule)),
            Some((_, tests)) => {
                // Share contiguous prefixes without expanding a Boolean
                // decision diagram (which can grow exponentially). The suffix
                // remains the fallback for every failure within this group.
                let test = tests[0];
                let count = candidates
                    .iter()
                    .take_while(|(_, ts)| ts.first() == Some(&test))
                    .count();
                let no = self.sequence(candidates[count..].to_vec(), fallback);
                let yes = candidates[..count]
                    .iter()
                    .map(|(r, ts)| (*r, ts[1..].to_vec()))
                    .collect();
                let yes = self.sequence(yes, no);
                self.node(Node::Check { test, yes, no })
            }
        };
        self.states.insert(key, id);
        id
    }
}

struct Plan {
    definitions: Vec<DefMatch>,
    rules: Vec<SelectRuleDef>,
    tests: Vec<Test>,
    candidates: Candidates,
}

impl Plan {
    fn test(&mut self, test: Test) -> usize {
        if let Some(id) = self.tests.iter().position(|t| *t == test) {
            return id;
        }
        let id = self.tests.len();
        self.tests.push(test);
        id
    }

    fn prepare(
        rules: &[&SelectRuleDef],
        extractors: &HashMap<String, ExtractorDef>,
        instructions: &HashMap<String, FinalInstDef>,
    ) -> Self {
        let mut plan = Self {
            definitions: Vec::new(),
            rules: Vec::new(),
            tests: Vec::new(),
            candidates: Vec::new(),
        };
        for rule in rules {
            let mut rule = (*rule).clone();
            let mut names = BTreeMap::<String, String>::new();
            let rename = |path: &str, names: &BTreeMap<String, String>| {
                if let Some((owner, field)) = path.split_once('.') {
                    format!(
                        "{}.{}",
                        names.get(owner).expect("checked definition order"),
                        field
                    )
                } else {
                    path.to_owned()
                }
            };
            let mut tests = Vec::new();
            for def in &mut rule.definitions {
                def.input = rename(&def.input, &names);
                let slot = plan
                    .definitions
                    .iter()
                    .position(|d| {
                        d.input == def.input && d.opcode == def.opcode && d.schema == def.schema
                    })
                    .unwrap_or(plan.definitions.len());
                let canonical = format!("n{slot}");
                names.insert(def.name.clone(), canonical.clone());
                def.name = canonical;
                if slot == plan.definitions.len() {
                    plan.definitions.push(def.clone());
                }
                tests.push(plan.test(Test::Definition(slot)));
            }
            for field in &mut rule.fields {
                let PatternArg::Named { name, .. } = field else {
                    continue;
                };
                *name = rename(name, &names);
            }
            for (field, pattern) in named_args(&rule.fields) {
                let schema = field
                    .split_once('.')
                    .map(|(name, _)| {
                        plan.definitions
                            .iter()
                            .find(|d| d.name == name)
                            .unwrap()
                            .schema
                            .clone()
                    })
                    .unwrap_or_else(|| rule.schema.clone());
                fn guards(
                    pattern: &Pattern,
                    extractors: &HashMap<String, ExtractorDef>,
                    out: &mut Vec<Guard>,
                ) {
                    match pattern.strip_node_binds() {
                        Pattern::Typed { types, .. } => {
                            let mut types = types.clone();
                            types.sort();
                            types.dedup();
                            out.push(Guard::Types(types));
                        }
                        Pattern::IntConst(value) => out.push(Guard::Integer(*value)),
                        Pattern::CondCode(cc) => out.push(Guard::Condition(*cc)),
                        Pattern::Opcode { opcode, .. } if extractors.contains_key(opcode) => {
                            out.push(Guard::Extractor(opcode.clone()))
                        }
                        Pattern::And(parts) => {
                            for part in parts {
                                guards(part, extractors, out);
                            }
                        }
                        _ => {}
                    }
                }
                let mut constraints = Vec::new();
                guards(pattern, extractors, &mut constraints);
                for guard in constraints {
                    tests.push(plan.test(Test::Field {
                        field: field.into(),
                        schema: schema.clone(),
                        guard,
                    }));
                }
            }
            fn features(
                build: &Constructor,
                insts: &HashMap<String, FinalInstDef>,
                out: &mut BTreeSet<String>,
            ) {
                if let Constructor::Inst { opcode, args } = build {
                    out.extend(insts[opcode].requires.iter().cloned());
                    for arg in args {
                        features(arg, insts, out);
                    }
                }
            }
            let mut required = BTreeSet::new();
            for build in &rule.builds {
                features(build, instructions, &mut required);
            }
            if !required.is_empty() {
                tests.push(plan.test(Test::Features(required.into_iter().collect())));
            }
            for def in &rule.definitions {
                let slot = plan
                    .definitions
                    .iter()
                    .position(|d| d.name == def.name)
                    .unwrap();
                tests.push(plan.test(Test::Foldable(slot)));
            }
            plan.candidates.push((plan.rules.len(), tests));
            plan.rules.push(rule);
        }
        plan
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_graph_preserves_order_for_every_predicate_assignment() {
        let cases = vec![
            (0, vec![0, 1]),
            (1, vec![2, 0]),
            (2, vec![1, 3]),
            (3, vec![2]),
            (4, vec![]),
        ];
        let mut graph = Graph::default();
        let entry = graph.compile(cases.clone());
        for bits in 0..16 {
            let expected = cases
                .iter()
                .find(|(_, ts)| ts.iter().all(|&t| bits & (1 << t) != 0))
                .map(|(r, _)| *r);
            let mut pc = entry;
            let actual = loop {
                match graph.nodes[pc] {
                    Node::Reject => break None,
                    Node::Accept(r) => break Some(r),
                    Node::Check { test, yes, no } => {
                        pc = if bits & (1 << test) != 0 { yes } else { no }
                    }
                }
            };
            assert_eq!(actual, expected);
        }
    }
}
