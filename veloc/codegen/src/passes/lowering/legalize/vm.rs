//! A read-only decision VM followed by a deferred value-rewrite recipe.
//! The driver owns edit tracking, convergence and worklist updates.
use super::info::{LegalizePolicy, RewriteContext};
use crate::error::{Error, Result};
use smallvec::SmallVec;
pub use veloc_bytecode::rewrite::OperandRef;
use veloc_bytecode::{
    Lebs, Reader,
    rewrite::{Instruction as Op, TypePattern},
};
use veloc_lir::{FieldValue, GenericOpcode, Reg};
use veloc_lir::{InstId, MachineFunction};
use veloc_mir::Type;

pub struct Program {
    pub entries: &'static [Option<usize>],
    pub code: &'static [u8],
    pub sets: &'static [&'static [Type]],
    pub features: &'static [&'static [u64]],
    pub actions: &'static [Action],
    pub types: &'static [TypeSource],
    pub fields: &'static [FieldSource],
}

pub enum FieldSource {
    Constant(FieldValue),
    Root(usize),
}
impl FieldSource {
    fn read(&self, ctx: &RewriteContext<'_>) -> FieldValue {
        match self {
            Self::Constant(value) => value.clone(),
            Self::Root(index) => ctx.inst(ctx.root()).fields().at(*index),
        }
    }
}

pub enum TypeSource {
    Exact(Type),
    Value(OperandRef),
}

#[derive(Clone, Copy)]
pub enum Action {
    Legal,
    Recipe {
        name: &'static str,
        entry: usize,
        slots: usize,
    },
    /// Replace the matched value operation with a runtime call. Its fixed
    /// signature is the root's input/result types, checked by the spec compiler.
    Libcall {
        name: &'static str,
        symbol: &'static str,
    },
}

/// Read-only matching shared by verification and the execution engine.
/// Return the table entry directly, without allocating or rebuilding a plan.
pub(super) fn select(
    policy: LegalizePolicy<'_>,
    function: &MachineFunction,
    id: InstId,
) -> Result<Option<(&'static Program, &'static Action)>> {
    let types = InstTypes::new(function, id)?;
    let inst = types.inst;
    let program = policy.program;
    let Some(entry) = program
        .entries
        .get(inst.generic_opcode().unwrap() as usize)
        .copied()
        .flatten()
    else {
        return Ok(None);
    };
    let mut reader = Reader {
        bytes: program.code,
        pc: entry,
    };
    let action = loop {
        match Op::read(&mut reader) {
            Op::Reject {} => return Ok(None),
            Op::Jump { target } => reader.pc = target,
            Op::CheckSignature {
                results,
                inputs,
                failure,
            } => {
                if !types.matches_signature(results, inputs, program.sets) {
                    reader.pc = failure;
                }
            }
            Op::CheckType {
                value,
                set,
                failure,
            } => {
                if !types
                    .get(OperandRef::decode(value))
                    .is_some_and(|ty| program.sets[set].contains(&ty))
                {
                    reader.pc = failure;
                }
            }
            Op::CheckSignedRange {
                field,
                bits,
                expected,
                failure,
            } => {
                let n = immediate(inst, field);
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
            Op::Accept { action } => break action,
            _ => panic!("construction instruction in read-only query"),
        }
    };
    Ok(Some((program, &program.actions[action])))
}

pub(super) fn apply(
    program: &'static Program,
    entry: usize,
    slots: usize,
    ctx: &mut RewriteContext<'_>,
) -> Result<()> {
    let root = ctx.inst(ctx.root());
    let destination = root.results().first().copied();
    let results = root.results().len();
    // Snapshot through the same type view before edits can invalidate borrows.
    let types = InstTypes::new(ctx, ctx.root())?.snapshot()?;
    // Checked recipes assign fresh slots in order; no dummy register values.
    let mut values = SmallVec::<[Reg; 16]>::with_capacity(slots);
    values.extend_from_slice(root.inputs());
    let ty = |id: usize| match program.types[id] {
        TypeSource::Exact(ty) => ty,
        TypeSource::Value(operand) => *operand
            .get(&types[results..], &types[..results])
            .expect("checked recipe type source"),
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
                    fields.iter().map(|i| program.fields[i].read(ctx)).collect();
                assert_eq!(dst, values.len(), "fresh recipe result");
                values.push(ctx.emit(
                    GenericOpcode::from_code(opcode).expect("invalid opcode in generated recipe"),
                    ty(t),
                    &args,
                    &fields,
                    if reuse != 0 {
                        Some(destination.expect("value replacement result"))
                    } else {
                        None
                    },
                ));
            }
            Op::Return { value } => {
                ctx.finish_value(values[value]);
                return Ok(());
            }
            Op::Update { inputs, fields } => {
                let mut changes = SmallVec::<[(usize, Reg); 4]>::new();
                let mut inputs = inputs.iter();
                while let Some(index) = inputs.next() {
                    changes.push((index, values[inputs.next().expect("update input pair")]));
                }
                let mut attributes = SmallVec::<[(usize, FieldValue); 2]>::new();
                let mut fields = fields.iter();
                while let Some(index) = fields.next() {
                    attributes.push((
                        index,
                        program.fields[fields.next().expect("update field pair")].read(ctx),
                    ));
                }
                ctx.update(&changes, &attributes);
                return Ok(());
            }
            _ => panic!("query instruction in committed rewrite"),
        }
    }
}

/// A checked view of semantic SSA types. ABI placement is independent of types.
struct InstTypes<'a> {
    function: &'a MachineFunction,
    inst: veloc_lir::InstRef<'a>,
}

impl<'a> InstTypes<'a> {
    fn matches_signature(&self, results: Lebs<'_>, inputs: Lebs<'_>, sets: &[&[Type]]) -> bool {
        if results.len() != self.inst.results().len() || inputs.len() != self.inst.inputs().len() {
            return false;
        }
        // Bindings belong to this signature match; failed candidates cannot
        // leak type variables into later branches of the decision bytecode.
        let mut bindings = SmallVec::<[Type; 4]>::new();
        results
            .iter()
            .enumerate()
            .map(|(i, pattern)| (OperandRef::Result(i), pattern))
            .chain(
                inputs
                    .iter()
                    .enumerate()
                    .map(|(i, pattern)| (OperandRef::Input(i), pattern)),
            )
            .all(|(operand, pattern)| {
                let Some(ty) = self.get(operand) else {
                    return false;
                };
                match TypePattern::decode(pattern) {
                    TypePattern::Set(set) => sets[set].contains(&ty),
                    TypePattern::Bind(set) => {
                        if !sets[set].contains(&ty) {
                            return false;
                        }
                        bindings.push(ty);
                        true
                    }
                    TypePattern::Same(slot) => ty == bindings[slot],
                }
            })
    }

    fn new(function: &'a MachineFunction, id: InstId) -> Result<Self> {
        let inst = function.inst(id);
        if inst.generic_opcode().is_none() {
            return Err(Error::codegen("expected generic instruction"));
        }
        let regs = || inst.results().iter().chain(inst.inputs());
        if regs().any(|reg| {
            reg.as_vreg()
                .is_some_and(|reg| function.vregs().get(reg).is_none())
        }) {
            return Err(Error::codegen("unknown virtual operand in legalization"));
        }
        if regs().any(|reg| reg.is_preg()) {
            return Err(Error::codegen(
                "generic value operands must be virtual before allocation",
            ));
        }
        Ok(Self { function, inst })
    }

    fn get(&self, operand: OperandRef) -> Option<Type> {
        let reg = operand.get(self.inst.inputs(), self.inst.results())?;
        match reg.as_vreg() {
            Some(reg) => Some(self.function.vregs()[reg].ty),
            None => None,
        }
    }

    /// Recipes need an owned snapshot while they mutate the function. Preserve
    /// the results/inputs split so all type sources still use OperandRef.
    fn snapshot(&self) -> Result<SmallVec<[Type; 4]>> {
        (0..self.inst.results().len())
            .map(OperandRef::Result)
            .chain((0..self.inst.inputs().len()).map(OperandRef::Input))
            .map(|operand| {
                self.get(operand)
                    .ok_or_else(|| Error::codegen("value recipe requires semantic operand types"))
            })
            .collect()
    }
}

fn immediate(inst: veloc_lir::InstRef<'_>, index: usize) -> i64 {
    let veloc_lir::FieldValueRef::Imm(&value) = inst.fields().read(index) else {
        panic!("checked immediate field");
    };
    value
}
