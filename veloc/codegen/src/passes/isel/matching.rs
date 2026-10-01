//! Selection bytecode. Matching is read-only; Accept enters construction.
//! Accepted recipes insert before the source; the driver finishes replacement and edge transfers.
use crate::target::FeatureSetRef;
use smallvec::SmallVec;
use std::vec::Vec;
use veloc_lir::{FieldValue, GenericOpcode, InstId, InstInserter, InstRef, Reg};
use veloc_mir::Type;

use veloc_bytecode::{Reader, selection::Instruction as Op};

pub struct Program {
    pub entries: [Option<Entry>; GenericOpcode::COUNT],
    pub code: &'static [u8],
    pub types: &'static [&'static [Type]],
    pub targets: &'static [Target],
    pub required_features: &'static [FeatureSetRef<'static>],
    pub registers: &'static [Reg],
}

/// An opcode's entry point and its local scratch requirements.
#[derive(Clone, Copy)]
pub struct Entry {
    pub offset: u32,
    pub insts: usize,
    pub values: usize,
    pub fields: usize,
}

impl Program {
    pub fn entry(&self, opcode: GenericOpcode) -> Option<Entry> {
        self.entries[opcode as usize]
    }
}

// Source instructions stay alive until selection commits. Cache locations,
// not owned payloads, so repeated candidates do not clone call signatures.
#[derive(Clone, Copy)]
enum FieldSource {
    Attribute(InstId, usize),
    Imm(i64),
}
fn integer(inst: InstRef<'_>, index: usize) -> i64 {
    match inst.fields().read(index) {
        veloc_lir::FieldValueRef::Imm(value) => *value,
        veloc_lir::FieldValueRef::IntCC(value) => *value as i64,
        veloc_lir::FieldValueRef::FloatCC(value) => *value as i64,
        _ => panic!("non-integer selection field"),
    }
}

/// Generated construction entry points commit complete, positioned instructions.
pub type Target =
    fn(&mut InstInserter<'_>, InstId, &[Reg], &[Reg], SmallVec<[FieldValue; 4]>) -> InstId;

/// Debug output describes the actual bytecode, including byte offsets.
#[allow(dead_code)]
pub fn disassemble(program: &Program, out: &mut dyn core::fmt::Write) -> core::fmt::Result {
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
pub(super) fn execute(
    program: &Program,
    entry: Entry,
    features: FeatureSetRef<'_>,
    predicate: Option<&(dyn Fn(u32, Reg) -> bool + Send + Sync)>,
    store: &mut InstInserter<'_>,
    source: InstId,
    out: &mut Vec<InstId>,
    edge_transfers: &mut Vec<(veloc_lir::EdgeId, veloc_lir::EdgeId)>,
) -> Option<()> {
    let mut reader = Reader {
        bytes: program.code,
        pc: entry.offset as usize,
    };
    let mut insts = SmallVec::<[Option<InstId>; 4]>::from_elem(None, entry.insts);
    let mut values = SmallVec::<[Option<Reg>; 16]>::from_elem(None, entry.values);
    let mut fields = SmallVec::<[Option<FieldSource>; 8]>::from_elem(None, entry.fields);
    insts[0] = Some(source);
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
            Op::ReadReg { dst, node, operand } => {
                let inst = store.inst(insts[node].expect("dominating definition"));
                values[dst] = Some(
                    *operand
                        .get(inst.inputs(), inst.results())
                        .expect("checked operand position"),
                );
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
                if !(store
                    .inst(insts[node].unwrap())
                    .generic_opcode()
                    .map(|opcode| opcode as usize)
                    == Some(opcode))
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
                    .and_then(|reg| reg.as_vreg().map(|v| store.vregs()[v].ty))
                    .is_some_and(|ty| program.types[set].contains(&ty)))
                {
                    reader.pc = failure;
                }
            }
            Op::CheckInt {
                node,
                index,
                constant,
                failure,
            } => {
                assert!(!accepted);
                if integer(store.inst(insts[node].unwrap()), index) != constant {
                    reader.pc = failure;
                }
            }
            Op::CheckIntRange {
                node,
                index,
                bits,
                signed,
                failure,
            } => {
                assert!(!accepted);
                let value = integer(store.inst(insts[node].unwrap()), index);
                let fits = if signed != 0 {
                    bits == 64 || value == (value << (64 - bits)) >> (64 - bits)
                } else {
                    value >= 0 && (bits == 64 || (value as u64) >> bits == 0)
                };
                if !fits {
                    reader.pc = failure;
                }
            }
            Op::CheckFeatures { set, failure } => {
                assert!(!accepted);
                let required = program.required_features[set];
                if !features.contains_all(required) {
                    reader.pc = failure;
                }
            }
            Op::CallPredicate { value, id, failure } => {
                assert!(!accepted);
                let predicate = predicate.expect("selection program requires host predicates");
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
            Op::ReadField { dst, node, index } => {
                assert!(accepted);
                fields[dst] = Some(FieldSource::Attribute(insts[node].unwrap(), index));
            }
            Op::ConstImm { dst, imm } => {
                assert!(accepted);
                fields[dst] = Some(FieldSource::Imm(imm));
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
                return Some(());
            }
        }
    }
}
