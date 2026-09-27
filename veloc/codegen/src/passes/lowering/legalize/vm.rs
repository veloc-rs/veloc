//! A read-only decision VM followed by a deferred value-rewrite recipe.
//! The driver owns edit tracking, convergence and worklist updates.
use super::info::{LegalizePolicy, RewriteContext};
use crate::error::{Error, Result};
use smallvec::SmallVec;
use veloc_bytecode::{Reader, rewrite::Instruction as Op};
use veloc_lir::{FieldValue, GenericOpcode, Reg};
use veloc_lir::{InstId, InstRead, MachineFunction};
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
    Value { result: bool, index: usize },
}

#[derive(Clone, Copy)]
pub enum Action {
    Legal,
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
    function: &MachineFunction,
    id: InstId,
) -> Result<Option<(&'static Program, &'static Action)>> {
    check_input(function, id)?;
    let inst = function.inst(id);
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
    // Low bit selects results; remaining bits are the operand index.
    let ty = |value: usize| value_type(function, id, value & 1 != 0, value >> 1);
    let action = loop {
        match Op::read(&mut reader) {
            Op::Reject {} => return Ok(None),
            Op::Jump { target } => reader.pc = target,
            Op::CheckSignature {
                results,
                inputs,
                failure,
            } => {
                let matches = |result, sets: veloc_bytecode::Lebs<'_>| {
                    (if result {
                        inst.results().len()
                    } else {
                        inst.inputs().len()
                    }) == sets.len()
                        && sets.iter().enumerate().all(|(i, set)| {
                            program.sets[set].contains(&value_type(function, id, result, i))
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
    let types: SmallVec<[Type; 4]> = root
        .results()
        .iter()
        .chain(root.inputs())
        .map(|reg| ctx.vreg_data(*reg).ty)
        .collect();
    // Checked recipes assign fresh slots in order; no dummy register values.
    let mut values = SmallVec::<[Reg; 16]>::with_capacity(slots);
    values.extend_from_slice(root.inputs());
    let ty = |id: usize| match program.types[id] {
        TypeSource::Exact(ty) => ty,
        TypeSource::Value { result, index } => types[if result { index } else { results + index }],
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

/// Validate the instruction-local ABI boundary before reading operand types.
fn check_input(function: &MachineFunction, id: InstId) -> Result<()> {
    let inst = function.inst(id);
    let vregs = function.vregs();
    if inst.generic_opcode().is_none() {
        return Err(Error::codegen("expected generic instruction"));
    }
    let regs = || inst.results().iter().chain(inst.inputs());
    if regs().any(|reg| reg.as_vreg().is_some_and(|reg| vregs.get(reg).is_none())) {
        return Err(Error::codegen("unknown virtual operand in legalization"));
    }
    if regs().any(|reg| reg.is_preg()) {
        // Physical locations are permitted only at explicit ABI boundaries.
        // In particular, a register name never supplies a semantic type.
        let valid = match inst.view() {
            veloc_lir::InstView::UnaryReg(copy)
                if copy.opcode == veloc_lir::UnaryRegOpcode::Copy =>
            {
                copy.dst.is_vreg() != copy.src.is_vreg()
            }
            veloc_lir::InstView::Call(call) => call
                .args
                .iter()
                .chain(call.results)
                .all(|reg| reg.is_preg()),
            veloc_lir::InstView::CallIndirect(call) => {
                call.callee.is_vreg()
                    && call
                        .args
                        .iter()
                        .chain(call.results)
                        .all(|reg| reg.is_preg())
            }
            veloc_lir::InstView::Return(ret) => ret.values.iter().all(|reg| reg.is_preg()),
            _ => false,
        };
        if !valid {
            return Err(Error::codegen(
                "physical operands require a typed copy or ABI call/return boundary",
            ));
        }
    }

    Ok(())
}

/// Physical Copy endpoints inherit the transfer type from the SSA endpoint.
fn value_type(function: &MachineFunction, id: InstId, result: bool, index: usize) -> Type {
    let inst = function.inst(id);
    let reg = if result {
        inst.results()[index]
    } else {
        inst.inputs()[index]
    };
    if let Some(reg) = reg.as_vreg() {
        return function.vregs()[reg].ty;
    }
    assert_eq!(
        inst.generic_opcode(),
        Some(GenericOpcode::Copy),
        "ABI locations have no standalone value type"
    );
    let value = inst
        .results()
        .iter()
        .chain(inst.inputs())
        .find_map(|reg| reg.as_vreg())
        .expect("typed boundary copy");
    function.vregs()[value].ty
}

fn immediate(inst: veloc_lir::InstRef<'_>, index: usize) -> i64 {
    let veloc_lir::FieldValueRef::Imm(&value) = inst.fields().read(index) else {
        panic!("checked immediate field");
    };
    value
}
