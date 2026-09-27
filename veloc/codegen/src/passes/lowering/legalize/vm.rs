//! A read-only decision VM followed by a deferred value-rewrite recipe.
//! The driver owns edit tracking, convergence and worklist updates.
use super::contracts::Query as _;
use super::info::{LegalizePolicy, Query, RewriteContext};
use crate::error::Result;
use smallvec::SmallVec;
use veloc_bytecode::{Reader, rewrite::Instruction as Op};
use veloc_lir::{FieldValue, GenericOpcode, Reg};
use veloc_mir::Type;

pub struct Program {
    pub code: &'static [u8],
    pub sets: &'static [&'static [Type]],
    pub features: &'static [&'static [u64]],
    pub actions: &'static [Action],
    pub types: &'static [TypeSource],
    pub opcodes: &'static [GenericOpcode],
    pub fields: &'static [FieldValue],
    pub emit:
        fn(&mut RewriteContext<'_>, GenericOpcode, Type, &[Reg], &[FieldValue], Option<Reg>) -> Reg,
    pub functions: &'static [fn(&mut RewriteContext<'_>, &[Type], &[Reg]) -> Reg],
}

pub enum TypeSource {
    Exact(Type),
    Value { result: bool, index: usize },
}

#[derive(Clone, Copy)]
pub enum Action {
    Legal,
    Host {
        name: &'static str,
        apply: fn(&mut RewriteContext<'_>) -> Result<()>,
    },
    Recipe {
        name: &'static str,
        entry: usize,
        slots: usize,
    },
}

/// Read-only matching shared by verification and the execution engine.
/// Return the table entry directly, without allocating or rebuilding a plan.
pub(super) fn select(
    policy: LegalizePolicy<'_>,
    query: &Query<'_>,
) -> Option<(&'static Program, &'static Action)> {
    let (program, entry) = (policy.program)(query.opcode())?;
    let mut reader = Reader {
        bytes: program.code,
        pc: entry,
    };
    // Low bit selects results; remaining bits are the operand index.
    let ty = |value: usize| query.value_type(value & 1 != 0, (value >> 1) as u32);
    let action = loop {
        match Op::read(&mut reader) {
            Op::Reject {} => return None,
            Op::Jump { target } => reader.pc = target,
            Op::CheckSignature {
                results,
                inputs,
                failure,
            } => {
                let matches = |result, sets: veloc_bytecode::Lebs<'_>| {
                    query.arity(result) == sets.len()
                        && sets.iter().enumerate().all(|(i, set)| {
                            program.sets[set].contains(&query.value_type(result, i as u32))
                        })
                };
                if !matches(true, results) || !matches(false, inputs) {
                    reader.pc = failure;
                }
            }
            Op::CheckSameType { values, failure } => {
                let mut values = values.iter().map(ty);
                if let Some(first) = values.next() {
                    if values.any(|v| v != first) {
                        reader.pc = failure;
                    }
                }
            }
            Op::CheckType {
                value,
                set,
                failure,
            } => {
                if !program.sets[set].contains(&ty(value)) {
                    reader.pc = failure;
                }
            }
            Op::CheckSignedRange {
                field,
                bits,
                expected,
                failure,
            } => {
                let n = query.immediate(field as u32);
                let fits = bits == 64 || (n >= -(1i64 << (bits - 1)) && n < (1i64 << (bits - 1)));
                if fits != (expected != 0) {
                    reader.pc = failure;
                }
            }
            Op::CheckFeatures { set, failure } => {
                if !program.features[set]
                    .iter()
                    .enumerate()
                    .all(|(i, mask)| policy.features.get(i).copied().unwrap_or(0) & mask == *mask)
                {
                    reader.pc = failure;
                }
            }
            Op::CallPredicate { predicate, failure } => {
                if !(policy.predicate.expect("declared Rust predicate"))(predicate, query) {
                    reader.pc = failure;
                }
            }
            Op::Accept { action } => break action,
            _ => panic!("construction instruction in read-only query"),
        }
    };
    Some((program, &program.actions[action]))
}

pub(super) fn apply(
    program: &'static Program,
    entry: usize,
    slots: usize,
    ctx: &mut RewriteContext<'_>,
) -> Result<()> {
    ctx.replace_values(|ctx, inputs, types, destination| {
        // Slots are definitely assigned by the checked, straight-line recipe.
        // Input registers are snapshotted before any construction changes uses.
        let mut values = SmallVec::<[Reg; 16]>::from_elem(destination, slots);
        values[..inputs.len()].copy_from_slice(inputs);
        let ty = |id: usize| match program.types[id] {
            TypeSource::Exact(ty) => ty,
            TypeSource::Value { result, index } => types[if result { index } else { 1 + index }],
        };
        let mut reader = Reader {
            bytes: program.code,
            pc: entry,
        };
        let mut args = SmallVec::<[Reg; 4]>::new();
        loop {
            match Op::read(&mut reader) {
                Op::Emit {
                    opcode,
                    ty: t,
                    inputs,
                    fields,
                    dst,
                    reuse,
                } => {
                    args.clear();
                    args.extend(inputs.iter().map(|i| values[i]));
                    let fields: SmallVec<[FieldValue; 2]> =
                        fields.iter().map(|i| program.fields[i].clone()).collect();
                    values[dst] = (program.emit)(
                        ctx,
                        program.opcodes[opcode],
                        ty(t),
                        &args,
                        &fields,
                        (reuse != 0).then_some(destination),
                    );
                }
                Op::Call {
                    function,
                    types,
                    inputs,
                    dst,
                } => {
                    args.clear();
                    args.extend(inputs.iter().map(|i| values[i]));
                    let types: SmallVec<[Type; 2]> = types.iter().map(ty).collect();
                    values[dst] = program.functions[function](ctx, &types, &args);
                }
                Op::Return { value } => return values[value],
                _ => panic!("query instruction in committed rewrite"),
            }
        }
    })
}
