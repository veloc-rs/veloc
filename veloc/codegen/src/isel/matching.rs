//! Selection bytecode. Matching is read-only; Accept enters construction.
//! Accepted recipes insert before the source; the driver finishes replacement and edge transfers.
use super::select::SelectResult;
use smallvec::SmallVec;
use std::vec::Vec;
use veloc_lir::{FieldValue, GenericOpcode, InstId, InstInserter, InstRef, Reg};
use veloc_mir::Type;

use veloc_bytecode::{Reader, selection::Instruction as Op};

pub(crate) struct Program {
    pub code: &'static [u8],
    pub entry: u32,
    pub insts: usize,
    pub values: usize,
    pub fields: usize,
    pub types: &'static [&'static [Type]],
    pub integers: &'static [i64],
    pub opcodes: &'static [GenericOpcode],
    pub targets: &'static [Target],
    pub accesses: &'static [Option<Field>],
    pub features: &'static [&'static [u64]],
    pub registers: &'static [Reg],
}

// Source instructions stay alive until selection commits. Cache locations,
// not owned payloads, so repeated candidates do not clone call signatures.
#[derive(Clone, Copy)]
enum FieldSource {
    Attribute(InstId, usize),
    Imm(i64),
}
/// Physical field positions come from the same checked storage projections as
/// InstView. Optional fields bound to none occupy no slot; sequences are not
/// scalar accesses and are rejected by the selection compiler.
pub(crate) enum Field {
    Input(usize),
    Result(usize),
    Attribute(usize),
}
impl Field {
    fn reg(&self, inst: InstRef<'_>) -> Reg {
        match *self {
            Self::Input(index) => inst.inputs()[index],
            Self::Result(index) => inst.results()[index],
            Self::Attribute(_) => panic!("attribute used as a register"),
        }
    }
    fn integer(&self, inst: InstRef<'_>) -> i64 {
        let Self::Attribute(index) = *self else {
            panic!("register used as an attribute")
        };
        match inst.fields().read(index) {
            veloc_lir::FieldValueRef::Imm(value) => *value,
            veloc_lir::FieldValueRef::IntCC(value) => *value as i64,
            veloc_lir::FieldValueRef::FloatCC(value) => *value as i64,
            _ => panic!("non-integer selection field"),
        }
    }
}

/// Generated construction entry points commit complete, positioned instructions.
pub(crate) type Target =
    fn(&mut InstInserter<'_>, InstId, &[Reg], &[Reg], SmallVec<[FieldValue; 4]>) -> InstId;

/// Debug output describes the actual bytecode, including byte offsets.
#[allow(dead_code)]
pub(crate) fn disassemble(program: &Program, out: &mut dyn core::fmt::Write) -> core::fmt::Result {
    let mut reader = Reader {
        bytes: program.code,
        pc: 0,
    };
    while reader.pc < reader.bytes.len() {
        let pc = reader.pc;
        writeln!(out, "{pc:04x}: {:?}", Op::read(&mut reader))?;
    }
    Ok(())
}

/// One non-monomorphized executor for ordinary tests and construction recipes.
/// Programs are trusted build output, not user-provided bytecode.
#[inline(never)]
pub(crate) fn execute(
    program: &Program,
    features: &[u64],
    predicate: &dyn Fn(u32, Reg) -> bool,
    store: &mut InstInserter<'_>,
    source: InstId,
    out: &mut Vec<InstId>,
    edge_transfers: &mut Vec<(veloc_lir::EdgeId, veloc_lir::EdgeId)>,
) -> Option<SelectResult> {
    let mut reader = Reader {
        bytes: program.code,
        pc: program.entry as usize,
    };
    let mut insts = SmallVec::<[Option<InstId>; 4]>::from_elem(None, program.insts);
    let mut values = SmallVec::<[Option<Reg>; 16]>::from_elem(None, program.values);
    let mut fields = SmallVec::<[Option<FieldSource>; 8]>::from_elem(None, program.fields);
    insts[0] = Some(source);
    let start = out.len();
    let mut accepted = false;
    loop {
        let op = Op::read(&mut reader);
        match op {
            Op::Reject {} => {
                assert!(!accepted);
                return None;
            }
            Op::Jump { target } => {
                reader.pc = target;
            }
            Op::ReadReg { dst, node, field } => {
                values[dst] = program.accesses[field].as_ref().map(|field| {
                    field.reg(store.inst(insts[node].expect("dominating definition")))
                });
            }
            Op::GetDef {
                dst,
                value,
                failure,
            } => {
                assert!(!accepted);
                insts[dst] = values[value]
                    .filter(|reg| reg.is_vreg())
                    .and_then(|reg| store.defs(reg).single().map(|site| site.inst()));
                if !(insts[dst].is_some()) {
                    reader.pc = failure;
                }
            }
            Op::CheckOpcode {
                node,
                opcode,
                failure,
            } => {
                assert!(!accepted);
                if !(store.inst(insts[node].unwrap()).generic_opcode()
                    == Some(program.opcodes[opcode]))
                {
                    reader.pc = failure;
                }
            }
            Op::CheckType {
                value,
                set,
                failure,
            } => {
                assert!(!accepted);
                if !(values[value]
                    .and_then(|reg| reg.is_vreg().then(|| store.vreg_data(reg).ty))
                    .is_some_and(|ty| program.types[set].contains(&ty)))
                {
                    reader.pc = failure;
                }
            }
            Op::CheckInt {
                node,
                field,
                constant,
                failure,
            } => {
                assert!(!accepted);
                if !(program.accesses[field]
                    .as_ref()
                    .map(|field| field.integer(store.inst(insts[node].unwrap())))
                    == Some(program.integers[constant]))
                {
                    reader.pc = failure;
                }
            }
            Op::CheckFeatures { set, failure } => {
                assert!(!accepted);
                if !(program.features[set]
                    .iter()
                    .enumerate()
                    .all(|(i, required)| {
                        features.get(i).copied().unwrap_or(0) & required == *required
                    }))
                {
                    reader.pc = failure;
                }
            }
            Op::CallPredicate { value, id, failure } => {
                assert!(!accepted);
                if !(values[value].is_some_and(|reg| predicate(id as u32, reg))) {
                    reader.pc = failure;
                }
            }
            Op::CheckFoldable {
                definition,
                consumer,
                failure,
            } => {
                assert!(!accepted);
                let definition = insts[definition].unwrap();
                let consumer = insts[consumer].unwrap();
                // Only duplicate pure computation. Other users keep the old
                // definition; DCE may erase it once it becomes unused.
                let inst = store.inst(definition);
                if !(definition != consumer && inst.is_pure_value() && inst.results().len() == 1) {
                    reader.pc = failure;
                }
            }
            Op::Accept {} => {
                assert!(!accepted);
                accepted = true;
            }
            Op::MakeTemp { dst, ty } => {
                assert!(accepted);
                let [ty] = program.types[ty] else {
                    panic!("temporary requires one type")
                };
                values[dst] = Some(store.alloc_vreg(*ty));
            }
            Op::ReadResult { dst, index } => {
                assert!(accepted);
                values[dst] = Some(store.inst(source).results()[index]);
            }
            Op::ConstReg { dst, reg } => {
                assert!(accepted);
                values[dst] = Some(program.registers[reg]);
            }
            Op::ReadField { dst, node, field } => {
                assert!(accepted);
                fields[dst] = program.accesses[field].as_ref().map(|field| {
                    let Field::Attribute(index) = *field else {
                        panic!("register used as an attribute")
                    };
                    FieldSource::Attribute(insts[node].unwrap(), index)
                });
            }
            Op::ConstImm { dst, imm } => {
                assert!(accepted);
                fields[dst] = Some(FieldSource::Imm(program.integers[imm]));
            }
            Op::BuildInst {
                target,
                results: result_slots,
                inputs: input_slots,
                fields: field_slots,
            } => {
                assert!(accepted);
                let mut results = SmallVec::<[Reg; 2]>::new();
                let mut inputs = SmallVec::<[Reg; 4]>::new();
                let mut operands = SmallVec::<[FieldValue; 4]>::new();
                for slot in result_slots.iter() {
                    results.push(values[slot].expect("initialized result"));
                }
                for slot in input_slots.iter() {
                    inputs.push(values[slot].expect("initialized input"));
                }
                for slot in field_slots.iter() {
                    let mut field = match fields[slot].expect("initialized field") {
                        FieldSource::Attribute(inst, index) => store.inst(inst).fields().at(index),
                        FieldSource::Imm(value) => FieldValue::Imm(value),
                    };
                    if let FieldValue::Edge(edge) = &mut field {
                        assert!(
                            !edge_transfers.iter().any(|&(old, _)| old == *edge),
                            "edge transferred twice"
                        );
                        assert!(
                            store.inst(source).edge_ids().any(|id| id == *edge),
                            "edge must belong to selection root"
                        );
                        let copy = store.clone_edge(*edge);
                        edge_transfers.push((*edge, copy));
                        *edge = copy;
                    }
                    operands.push(field);
                }
                out.push(program.targets[target](
                    store, source, &results, &inputs, operands,
                ));
            }
            Op::Finish {} => {
                assert!(accepted);
                return Some(if out.len() - start == 1 {
                    SelectResult::InPlace
                } else {
                    SelectResult::Replace
                });
            }
        }
    }
}
