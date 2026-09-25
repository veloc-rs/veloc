//! Selection bytecode. Matching is read-only; Accept enters construction.
//! Accepted recipes insert before the source; the driver finishes replacement and edge transfers.
use super::select::SelectResult;
use smallvec::SmallVec;
use std::vec::Vec;
use veloc_lir::{FieldValue, GenericOpcode, InstId, InstInserter, InstRef, Reg};
use veloc_mir::Type;

use veloc_bytecode::{Reader, opcodes};

// Arity counts ULEB128 operands. Branch adds a little-endian u32 byte offset.
// BuildInst additionally carries three length-prefixed slot lists.
opcodes! {
    pub(crate) enum Op {
        Reject(0, false),
        Jump(0, true),
        ReadReg(3, false),       // value slot, instruction slot, field accessor
        GetDef(2, true),         // instruction slot, value slot, failure
        CheckOpcode(2, true),    // instruction slot, opcode constant, failure
        CheckType(2, true),      // value slot, type-set constant, failure
        CheckInt(3, true),       // instruction slot, field accessor, constant, failure
        CheckFeatures(1, true),  // feature-set constant, failure
        CallPredicate(2, true),  // value slot, host predicate, failure
        CheckFoldable(2, true),  // definition slot, consumer slot, failure
        Accept(0, false),
        MakeTemp(2, false),      // value slot, singleton type-set constant
        ReadResult(2, false),    // value slot, source result index
        ConstReg(2, false),      // value slot, physical register constant
        ReadField(3, false),     // field slot, instruction slot, field accessor
        ConstImm(2, false),      // field slot, integer constant
        BuildInst(1, false),     // target constant, result slots, input slots, field slots
        Finish(0, false),
    }
}

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
        write!(out, "{:04x}: ", reader.pc)?;
        let op = Op::decode(reader.byte());
        write!(out, "{op:?}")?;
        let (arity, branch) = op.format();
        for _ in 0..arity {
            write!(out, " {}", reader.uleb())?;
        }
        if matches!(op, Op::BuildInst) {
            for _ in 0..3 {
                let len = reader.uleb();
                write!(out, " [")?;
                for index in 0..len {
                    if index != 0 {
                        write!(out, ", ")?;
                    }
                    write!(out, "{}", reader.uleb())?;
                }
                write!(out, "]")?;
            }
        }
        if branch {
            write!(out, " -> {:04x}", reader.u32())?;
        }
        writeln!(out)?;
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
        let op = Op::decode(reader.byte());
        match op {
            Op::Reject => {
                assert!(!accepted);
                return None;
            }
            Op::Jump => {
                reader.pc = reader.u32();
            }
            Op::ReadReg => {
                let dst = reader.uleb();
                let node = reader.uleb();
                let field = reader.uleb();
                values[dst] = program.accesses[field].as_ref().map(|field| {
                    field.reg(store.inst(insts[node].expect("dominating definition")))
                });
            }
            Op::GetDef => {
                assert!(!accepted);
                let dst = reader.uleb();
                let value = reader.uleb();
                insts[dst] = values[value]
                    .filter(|reg| reg.is_vreg())
                    .and_then(|reg| store.defs(reg).single().map(|site| site.inst()));
                reader.branch(insts[dst].is_some());
            }
            Op::CheckOpcode => {
                assert!(!accepted);
                let node = reader.uleb();
                let opcode = reader.uleb();
                reader.branch(
                    store.inst(insts[node].unwrap()).generic_opcode()
                        == Some(program.opcodes[opcode]),
                );
            }
            Op::CheckType => {
                assert!(!accepted);
                let value = reader.uleb();
                let set = reader.uleb();
                reader.branch(
                    values[value]
                        .and_then(|reg| reg.is_vreg().then(|| store.vreg_data(reg).ty))
                        .is_some_and(|ty| program.types[set].contains(&ty)),
                );
            }
            Op::CheckInt => {
                assert!(!accepted);
                let node = reader.uleb();
                let field = reader.uleb();
                let constant = reader.uleb();
                reader.branch(
                    program.accesses[field]
                        .as_ref()
                        .map(|field| field.integer(store.inst(insts[node].unwrap())))
                        == Some(program.integers[constant]),
                );
            }
            Op::CheckFeatures => {
                assert!(!accepted);
                let set = reader.uleb();
                reader.branch(
                    program.features[set]
                        .iter()
                        .enumerate()
                        .all(|(i, required)| {
                            features.get(i).copied().unwrap_or(0) & required == *required
                        }),
                );
            }
            Op::CallPredicate => {
                assert!(!accepted);
                let value = reader.uleb();
                let id = reader.uleb();
                reader.branch(values[value].is_some_and(|reg| predicate(id as u32, reg)));
            }
            Op::CheckFoldable => {
                assert!(!accepted);
                let definition = insts[reader.uleb()].unwrap();
                let consumer = insts[reader.uleb()].unwrap();
                // Only duplicate pure computation. Other users keep the old
                // definition; DCE may erase it once it becomes unused.
                let inst = store.inst(definition);
                reader.branch(
                    definition != consumer && inst.is_pure_value() && inst.results().len() == 1,
                );
            }
            Op::Accept => {
                assert!(!accepted);
                accepted = true;
            }
            Op::MakeTemp => {
                assert!(accepted);
                let dst = reader.uleb();
                let ty = reader.uleb();
                let [ty] = program.types[ty] else {
                    panic!("temporary requires one type")
                };
                values[dst] = Some(store.alloc_vreg(*ty));
            }
            Op::ReadResult => {
                assert!(accepted);
                let dst = reader.uleb();
                let index = reader.uleb();
                values[dst] = Some(store.inst(source).results()[index]);
            }
            Op::ConstReg => {
                assert!(accepted);
                let dst = reader.uleb();
                let reg = reader.uleb();
                values[dst] = Some(program.registers[reg]);
            }
            Op::ReadField => {
                assert!(accepted);
                let dst = reader.uleb();
                let node = reader.uleb();
                let field = reader.uleb();
                fields[dst] = program.accesses[field].as_ref().map(|field| {
                    let Field::Attribute(index) = *field else {
                        panic!("register used as an attribute")
                    };
                    FieldSource::Attribute(insts[node].unwrap(), index)
                });
            }
            Op::ConstImm => {
                assert!(accepted);
                let dst = reader.uleb();
                let imm = reader.uleb();
                fields[dst] = Some(FieldSource::Imm(program.integers[imm]));
            }
            Op::BuildInst => {
                assert!(accepted);
                let target = reader.uleb();
                let mut results = SmallVec::<[Reg; 2]>::new();
                let mut inputs = SmallVec::<[Reg; 4]>::new();
                let mut operands = SmallVec::<[FieldValue; 4]>::new();
                for _ in 0..reader.uleb() {
                    results.push(values[reader.uleb()].expect("initialized result"));
                }
                for _ in 0..reader.uleb() {
                    inputs.push(values[reader.uleb()].expect("initialized input"));
                }
                for _ in 0..reader.uleb() {
                    let mut field = match fields[reader.uleb()].expect("initialized field") {
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
            Op::Finish => {
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
