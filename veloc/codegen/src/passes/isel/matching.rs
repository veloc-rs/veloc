//! Selection bytecode. Matching is read-only; Accept enters construction.
//! The selection driver commits detached instructions and edge transfers.
use super::select::SelectResult;
use alloc::vec::Vec;
use smallvec::SmallVec;
use veloc_lir::{GenericOpcode, InstBuilder, InstField, InstId, InstRef, InstWriter, Reg};
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
    pub targets: &'static [u32],
    pub registers: &'static [Reg],
}

/// Generated per target/schema, not per rule. Queries and predicates are pure.
/// Field IDs are resolved and checked by Spec; no runtime string lookup.
pub(crate) trait Host {
    fn read_reg(&self, inst: InstRef<'_>, field: u32) -> Option<Reg>;
    fn read_int(&self, inst: InstRef<'_>, field: u32) -> i64;
    fn read_field(&self, inst: InstRef<'_>, field: u32) -> InstField;
    fn ty(&self, reg: Reg) -> Option<Type>;
    fn features(&self, set: u32) -> bool;
    fn predicate(&self, id: u32, reg: Reg) -> bool;
    fn temporary(&mut self, ty: Type) -> Reg;
    fn build(
        &self,
        target: u32,
        writer: InstWriter<'_>,
        results: &[Reg],
        inputs: &[Reg],
        fields: &[InstField],
    ) -> InstId;
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
    host: &mut dyn Host,
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
    let mut fields = SmallVec::<[Option<InstField>; 8]>::from_elem(None, program.fields);
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
                values[dst] = host.read_reg(
                    store.get(insts[node].expect("dominating definition")),
                    field as u32,
                );
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
                        .and_then(|reg| host.ty(reg))
                        .is_some_and(|ty| program.types[set].contains(&ty)),
                );
            }
            Op::CheckInt => {
                assert!(!accepted);
                let node = reader.index();
                let field = reader.index();
                let constant = reader.index();
                reader.branch(
                    host.read_int(store.get(insts[node].unwrap()), field as u32)
                        == program.integers[constant],
                );
            }
            Op::CheckFeatures => {
                assert!(!accepted);
                let set = reader.index();
                reader.branch(host.features(set as u32));
            }
            Op::CallPredicate => {
                assert!(!accepted);
                let value = reader.index();
                let predicate = reader.index();
                reader
                    .branch(values[value].is_some_and(|reg| host.predicate(predicate as u32, reg)));
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
                values[dst] = Some(host.temporary(*ty));
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
                fields[dst] = Some(host.read_field(store.get(insts[node].unwrap()), field as u32));
            }
            Op::ConstImm => {
                assert!(accepted);
                let dst = reader.index();
                let imm = reader.index();
                fields[dst] = Some(InstField::Imm(program.integers[imm]));
            }
            Op::BuildInst => {
                assert!(accepted);
                let target = reader.index();
                let mut results = SmallVec::<[Reg; 2]>::new();
                let mut inputs = SmallVec::<[Reg; 4]>::new();
                let mut operands = SmallVec::<[InstField; 4]>::new();
                for _ in 0..reader.index() {
                    results.push(values[reader.index()].expect("initialized result"));
                }
                for _ in 0..reader.index() {
                    inputs.push(values[reader.index()].expect("initialized input"));
                }
                for _ in 0..reader.index() {
                    let mut field = fields[reader.index()].expect("initialized field");
                    if let InstField::Edge(edge) = &mut field {
                        *edge = store.replacement_edge(source, *edge);
                    }
                    operands.push(field);
                }
                out.push(host.build(
                    program.targets[target],
                    store.writer(),
                    &results,
                    &inputs,
                    &operands,
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
