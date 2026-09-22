//! One pattern vocabulary for SSA rewriting and e-class matching.
use alloc::vec::Vec;
use hashbrown::HashMap;
use smallvec::SmallVec;
use veloc_mir::{Opcode, Type};

pub(crate) enum Pattern {
    Variable(usize),
    Constant(u64),
    Apply(Opcode, &'static [Pattern]),
}
pub(crate) struct Condition {
    pub variable: usize,
    pub constant: u64,
    pub equal: bool,
}
pub(crate) struct Rule {
    pub name: &'static str,
    pub types: &'static [Type],
    pub variables: usize,
    pub pattern: Pattern,
    pub replacement: Pattern,
    pub guard: Option<Condition>,
}
include!(concat!(env!("OUT_DIR"), "/equivalences.rs"));

/// SSA has one definition; an e-class may contain several matching definitions.
/// Unknown analysis facts must return None, never a guessed constant.
pub(crate) trait Context {
    type Value: Copy + Eq;
    fn canonical(&self, value: Self::Value) -> Self::Value {
        value
    }
    fn constant(&self, value: Self::Value) -> Option<u64>;
    /// Scan the indexed `Node(opcode, args...)` relation for one e-class.
    fn scan(&self, value: Self::Value, opcode: Opcode, visit: impl FnMut(&[Self::Value]));
    fn literal(&mut self, value: u64) -> Option<Self::Value>;
    fn build(&mut self, opcode: Opcode, args: &[Self::Value]) -> Option<Self::Value>;
}

#[derive(Clone, Copy)]
enum Term {
    Variable(usize),
    Constant(u64),
}

struct Atom {
    result: Option<usize>,
    opcode: Opcode,
    args: Vec<Term>,
}

/// A rule's relational query plan. The first atom is anchored at the root
/// e-class; every later atom is joined through a variable produced by an
/// earlier atom. This is a nested-loop join over indexed facts, not a second
/// pattern-matching semantics.
struct RulePlan {
    variables: usize,
    root: Option<Term>,
    atoms: Vec<Atom>,
}

impl RulePlan {
    fn compile(rule: &Rule) -> Self {
        let mut plan = Self {
            variables: rule.variables,
            root: None,
            atoms: Vec::new(),
        };
        plan.root = match &rule.pattern {
            Pattern::Apply(opcode, patterns) => {
                plan.root_atom(*opcode, patterns);
                None
            }
            pattern => Some(plan.term(pattern)),
        };
        plan
    }

    fn term(&mut self, pattern: &Pattern) -> Term {
        match pattern {
            Pattern::Variable(index) => Term::Variable(*index),
            Pattern::Constant(value) => Term::Constant(*value),
            Pattern::Apply(opcode, patterns) => self.atom(*opcode, patterns),
        }
    }

    fn root_atom(&mut self, opcode: Opcode, patterns: &[Pattern]) {
        let atom = self.atoms.len();
        self.atoms.push(Atom {
            result: None,
            opcode,
            args: Vec::new(),
        });
        let args = patterns.iter().map(|pattern| self.term(pattern)).collect();
        self.atoms[atom].args = args;
    }

    fn atom(&mut self, opcode: Opcode, patterns: &[Pattern]) -> Term {
        let result = self.variables;
        self.variables += 1;
        let atom = self.atoms.len();
        self.atoms.push(Atom {
            result: Some(result),
            opcode,
            args: Vec::new(),
        });
        let args = patterns.iter().map(|pattern| self.term(pattern)).collect();
        self.atoms[atom].args = args;
        Term::Variable(result)
    }
}

/// Indexed relational matcher. Plans are compiled once per generated rule;
/// completed substitutions are retained contiguously for the mutation phase.
pub(crate) struct RelationalMatcher<V> {
    bindings: Vec<Option<V>>,
    output: Vec<Option<V>>,
    count: usize,
    plans: Vec<RulePlan>,
    plan_ids: HashMap<usize, usize>,
}

impl<V: Copy + Eq> RelationalMatcher<V> {
    pub fn new() -> Self {
        Self {
            bindings: Vec::new(),
            output: Vec::new(),
            count: 0,
            plans: Vec::new(),
            plan_ids: HashMap::new(),
        }
    }

    pub fn search<C: Context<Value = V>>(
        &mut self,
        ctx: &C,
        rule: &Rule,
        value: V,
        mask: u64,
        fuel: &mut usize,
    ) {
        let rule_key = rule as *const Rule as usize;
        let plan_id = if let Some(&id) = self.plan_ids.get(&rule_key) {
            id
        } else {
            let id = self.plans.len();
            self.plans.push(RulePlan::compile(rule));
            self.plan_ids.insert(rule_key, id);
            id
        };
        let plan = &self.plans[plan_id];
        self.bindings.resize(plan.variables, None);
        self.bindings.fill(None);
        self.output.clear();
        self.count = 0;
        match_plan(
            ctx,
            plan,
            value,
            &mut self.bindings,
            mask,
            fuel,
            &mut |env, _| {
                if rule.guard.as_ref().is_none_or(|guard| {
                    env[guard.variable]
                        .and_then(|v| ctx.constant(v))
                        .is_some_and(|v| (v == (guard.constant & mask)) == guard.equal)
                }) {
                    self.output.extend_from_slice(env);
                    self.count += 1;
                }
            },
        );
    }

    pub fn matches(&self) -> impl Iterator<Item = &[Option<V>]> {
        let width = self.bindings.len();
        (0..self.count).map(move |i| &self.output[i * width..(i + 1) * width])
    }
}

fn match_plan<C: Context>(
    ctx: &C,
    plan: &RulePlan,
    value: C::Value,
    env: &mut [Option<C::Value>],
    mask: u64,
    fuel: &mut usize,
    visit: &mut dyn FnMut(&mut [Option<C::Value>], &mut usize),
) {
    if let Some(root) = plan.root {
        if !bind_term(ctx, root, value, env, mask) {
            return;
        }
    }
    join_atoms(ctx, plan, 0, value, env, mask, fuel, visit);
}

fn join_atoms<C: Context>(
    ctx: &C,
    plan: &RulePlan,
    index: usize,
    root: C::Value,
    env: &mut [Option<C::Value>],
    mask: u64,
    fuel: &mut usize,
    visit: &mut dyn FnMut(&mut [Option<C::Value>], &mut usize),
) {
    let Some(atom) = plan.atoms.get(index) else {
        visit(env, fuel);
        return;
    };
    let result = atom
        .result
        .and_then(|variable| env[variable])
        .unwrap_or(root);
    ctx.scan(ctx.canonical(result), atom.opcode, |args| {
        if args.len() != atom.args.len() || *fuel == 0 {
            return;
        }
        *fuel -= 1;
        let mut changed: SmallVec<[(usize, Option<C::Value>); 4]> = SmallVec::new();
        for (&term, &arg) in atom.args.iter().zip(args) {
            let arg = ctx.canonical(arg);
            match term {
                Term::Variable(variable) => {
                    if env[variable].is_some_and(|old| ctx.canonical(old) != arg) {
                        for (variable, old) in changed.drain(..) {
                            env[variable] = old;
                        }
                        return;
                    }
                    changed.push((variable, env[variable]));
                    env[variable] = Some(arg);
                }
                Term::Constant(expected) => {
                    if ctx.constant(arg) != Some(expected & mask) {
                        for (variable, old) in changed.drain(..) {
                            env[variable] = old;
                        }
                        return;
                    }
                }
            }
        }
        join_atoms(ctx, plan, index + 1, root, env, mask, fuel, visit);
        for (variable, old) in changed {
            env[variable] = old;
        }
    });
}

fn bind_term<C: Context>(
    ctx: &C,
    term: Term,
    value: C::Value,
    env: &mut [Option<C::Value>],
    mask: u64,
) -> bool {
    let value = ctx.canonical(value);
    match term {
        Term::Variable(variable) if variable != usize::MAX => {
            if env[variable].is_some_and(|old| ctx.canonical(old) != value) {
                false
            } else {
                env[variable] = Some(value);
                true
            }
        }
        Term::Constant(expected) => ctx.constant(value) == Some(expected & mask),
        Term::Variable(_) => true,
    }
}

pub(crate) fn emit<C: Context>(
    ctx: &mut C,
    pattern: &Pattern,
    env: &[Option<C::Value>],
) -> Option<C::Value> {
    match pattern {
        Pattern::Variable(i) => env[*i].map(|v| ctx.canonical(v)),
        Pattern::Constant(value) => ctx.literal(*value),
        Pattern::Apply(opcode, patterns) => {
            let args: SmallVec<[C::Value; 3]> = patterns
                .iter()
                .map(|p| emit(ctx, p, env))
                .collect::<Option<SmallVec<[C::Value; 3]>>>()?;
            ctx.build(*opcode, &args)
        }
    }
}
