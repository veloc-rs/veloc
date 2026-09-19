//! One pattern vocabulary for SSA rewriting and e-class matching.
use alloc::{vec, vec::Vec};
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
    fn alternatives(&self, value: Self::Value, opcode: Opcode, visit: impl FnMut(&[Self::Value]));
    fn literal(&mut self, value: u64) -> Option<Self::Value>;
    fn build(&mut self, opcode: Opcode, args: &[Self::Value]) -> Option<Self::Value>;
}

pub(crate) fn matches<C: Context>(
    ctx: &C,
    rule: &Rule,
    value: C::Value,
    mask: u64,
    fuel: &mut usize,
) -> Vec<Vec<Option<C::Value>>> {
    let bindings = vec![None; rule.variables];
    match_pattern(ctx, &rule.pattern, value, bindings, mask, fuel)
        .into_iter()
        .filter(|env| {
            rule.guard.as_ref().is_none_or(|guard| {
                env[guard.variable]
                    .and_then(|v| ctx.constant(v))
                    .is_some_and(|v| (v == (guard.constant & mask)) == guard.equal)
            })
        })
        .collect()
}

fn match_pattern<C: Context>(
    ctx: &C,
    pattern: &Pattern,
    value: C::Value,
    mut env: Vec<Option<C::Value>>,
    mask: u64,
    fuel: &mut usize,
) -> Vec<Vec<Option<C::Value>>> {
    if *fuel == 0 {
        return Vec::new();
    }
    *fuel -= 1;
    let value = ctx.canonical(value);
    match pattern {
        Pattern::Variable(i) => {
            if env[*i].is_some_and(|old| ctx.canonical(old) != value) {
                return Vec::new();
            }
            env[*i] = Some(value);
            vec![env]
        }
        Pattern::Constant(n) => {
            if ctx.constant(value) == Some(n & mask) {
                vec![env]
            } else {
                Vec::new()
            }
        }
        Pattern::Apply(opcode, patterns) => {
            let mut output = Vec::new();
            ctx.alternatives(value, *opcode, |args| {
                if args.len() != patterns.len() || *fuel == 0 || output.len() >= 16 {
                    return;
                }
                let mut states = vec![env.clone()];
                for (pattern, &arg) in patterns.iter().zip(args) {
                    states = states
                        .into_iter()
                        .flat_map(|state| match_pattern(ctx, pattern, arg, state, mask, fuel))
                        .take(16)
                        .collect();
                }
                output.extend(states.into_iter().take(16 - output.len()));
            });
            output
        }
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
            let args = patterns
                .iter()
                .map(|p| emit(ctx, p, env))
                .collect::<Option<Vec<_>>>()?;
            ctx.build(*opcode, &args)
        }
    }
}
