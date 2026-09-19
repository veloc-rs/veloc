//! Selection bytecode. Matching is read-only; Accept enters construction.
//! The selection driver commits detached instructions and edge transfers.
use super::select::SelectResult;
use alloc::vec::Vec;
use smallvec::SmallVec;
use veloc_lir::{
    FieldValue, GenericOpcode, InstBuilder, InstId, InstRef, MachineOpcode, Reg, VRegBuilder,
    VRegData,
};
use veloc_mir::Type;

// Like interpreter::define_opcodes, keep decoding and diagnostics beside the
// instruction declaration. Unlike that VM, indexes here use compact ULEB128.
// The generator references symbolic Op names: numeric codes have one owner.
macro_rules! opcodes {
    ($($name:ident($arity:literal, $branch:literal)),* $(,)?) => {
        #[derive(Clone, Copy, Debug)]
        #[repr(u8)]
        pub(crate) enum Op { $($name),* }
        impl Op {
            fn decode(byte: u8) -> Self {
                match byte {
                    $(x if x == Self::$name as u8 => Self::$name,)*
                    _ => panic!("invalid selection opcode {byte}"),
                }
            }
            pub(crate) const fn format(self) -> (usize, bool) {
                match self { $(Self::$name => ($arity, $branch),)* }
            }
        }
    };
}
// Arity counts ULEB128 operands. Branch adds a little-endian u32 byte offset.
// BuildInst additionally carries three length-prefixed slot lists.
opcodes! {
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

/// Data only: construction uses the common writer, not a target callback.
pub(crate) struct Target {
    pub opcode: u32,
    pub metadata: &'static crate::target::TargetInstMetadata,
}

struct Reader<'a> {
    bytes: &'a [u8],
    pc: usize,
}
impl Reader<'_> {
    fn byte(&mut self) -> u8 {
        let b = self.bytes[self.pc];
        self.pc += 1;
        b
    }
    fn index(&mut self) -> usize {
        let mut value = 0u32;
        for shift in (0..35).step_by(7) {
            let byte = self.byte();
            assert!(shift != 28 || byte & 0xf0 == 0, "selection index overflow");
            value |= u32::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return value as usize;
            }
        }
        unreachable!()
    }
    fn offset(&mut self) -> usize {
        let bytes = self.bytes[self.pc..self.pc + 4].try_into().unwrap();
        self.pc += 4;
        u32::from_le_bytes(bytes) as usize
    }
    fn branch(&mut self, success: bool) {
        let failure = self.offset();
        if !success {
            self.pc = failure;
        }
    }
}

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
            write!(out, " {}", reader.index())?;
        }
        if matches!(op, Op::BuildInst) {
            for _ in 0..3 {
                let len = reader.index();
                write!(out, " [")?;
                for index in 0..len {
                    if index != 0 {
                        write!(out, ", ")?;
                    }
                    write!(out, "{}", reader.index())?;
                }
                write!(out, "]")?;
            }
        }
        if branch {
            write!(out, " -> {:04x}", reader.offset())?;
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
    vregs: &mut VRegBuilder<'_>,
    features: &[u64],
    predicate: &dyn Fn(u32, Reg) -> bool,
    store: &mut InstBuilder<'_>,
    source: InstId,
    out: &mut Vec<InstId>,
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
                reader.pc = reader.offset();
            }
            Op::ReadReg => {
                let dst = reader.index();
                let node = reader.index();
                let field = reader.index();
                values[dst] = program.accesses[field]
                    .as_ref()
                    .map(|field| field.reg(store.get(insts[node].expect("dominating definition"))));
            }
            Op::GetDef => {
                assert!(!accepted);
                let dst = reader.index();
                let value = reader.index();
                insts[dst] = values[value].and_then(|reg| store.def(reg));
                reader.branch(insts[dst].is_some());
            }
            Op::CheckOpcode => {
                assert!(!accepted);
                let node = reader.index();
                let opcode = reader.index();
                reader.branch(
                    store.get(insts[node].unwrap()).generic_opcode()
                        == Some(program.opcodes[opcode]),
                );
            }
            Op::CheckType => {
                assert!(!accepted);
                let value = reader.index();
                let set = reader.index();
                reader.branch(
                    values[value]
                        .and_then(|reg| reg.as_vreg().map(|reg| vregs.get(reg).ty))
                        .is_some_and(|ty| program.types[set].contains(&ty)),
                );
            }
            Op::CheckInt => {
                assert!(!accepted);
                let node = reader.index();
                let field = reader.index();
                let constant = reader.index();
                reader.branch(
                    program.accesses[field]
                        .as_ref()
                        .map(|field| field.integer(store.get(insts[node].unwrap())))
                        == Some(program.integers[constant]),
                );
            }
            Op::CheckFeatures => {
                assert!(!accepted);
                let set = reader.index();
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
                let value = reader.index();
                let id = reader.index();
                reader.branch(values[value].is_some_and(|reg| predicate(id as u32, reg)));
            }
            Op::CheckFoldable => {
                assert!(!accepted);
                let definition = insts[reader.index()].unwrap();
                let consumer = insts[reader.index()].unwrap();
                // Only duplicate pure computation. Other users keep the old
                // definition; DCE may erase it once it becomes unused.
                let inst = store.get(definition);
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
                let dst = reader.index();
                let ty = reader.index();
                let [ty] = program.types[ty] else {
                    panic!("temporary requires one type")
                };
                values[dst] = Some(vregs.alloc(VRegData {
                    ty: *ty,
                    bank: None,
                }));
            }
            Op::ReadResult => {
                assert!(accepted);
                let dst = reader.index();
                let index = reader.index();
                values[dst] = Some(store.get(source).results()[index]);
            }
            Op::ConstReg => {
                assert!(accepted);
                let dst = reader.index();
                let reg = reader.index();
                values[dst] = Some(program.registers[reg]);
            }
            Op::ReadField => {
                assert!(accepted);
                let dst = reader.index();
                let node = reader.index();
                let field = reader.index();
                fields[dst] = program.accesses[field].as_ref().map(|field| {
                    let Field::Attribute(index) = *field else {
                        panic!("register used as an attribute")
                    };
                    FieldSource::Attribute(insts[node].unwrap(), index)
                });
            }
            Op::ConstImm => {
                assert!(accepted);
                let dst = reader.index();
                let imm = reader.index();
                fields[dst] = Some(FieldSource::Imm(program.integers[imm]));
            }
            Op::BuildInst => {
                assert!(accepted);
                let target = reader.index();
                let mut results = SmallVec::<[Reg; 2]>::new();
                let mut inputs = SmallVec::<[Reg; 4]>::new();
                let mut operands = SmallVec::<[FieldValue; 4]>::new();
                for _ in 0..reader.index() {
                    results.push(values[reader.index()].expect("initialized result"));
                }
                for _ in 0..reader.index() {
                    inputs.push(values[reader.index()].expect("initialized input"));
                }
                for _ in 0..reader.index() {
                    let mut field = match fields[reader.index()].expect("initialized field") {
                        FieldSource::Attribute(inst, index) => store.get(inst).fields().at(index),
                        FieldSource::Imm(value) => FieldValue::Imm(value),
                    };
                    if let FieldValue::Edge(edge) = &mut field {
                        *edge = store.replacement_edge(source, *edge);
                    }
                    operands.push(field);
                }
                let target = &program.targets[target];
                out.push(
                    store
                        .writer()
                        .with_effects(target.metadata.implicit_uses, target.metadata.implicit_defs)
                        .write(
                            MachineOpcode::Target(target.opcode),
                            &results,
                            &inputs,
                            operands,
                        ),
                );
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
