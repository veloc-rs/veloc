use super::*;
use crate::passes::lowering::legalize::Query;
use veloc_lir::{InstBuild, InstRead};

impl host::ValueBuild for crate::passes::lowering::legalize::ValueRewriter<'_> {
    type Value = Reg;
    fn emit(&mut self, opcode: GenericOpcode, ty: Type, inputs: &[Reg]) -> Reg {
        self.emit(opcode, ty, inputs, None)
    }
    fn emit_integer(&mut self, opcode: GenericOpcode, ty: Type, value: i64) -> Reg {
        self.emit_integer(opcode, ty, value, None)
    }
}
impl host::ValueRewrite for crate::passes::lowering::legalize::ValueRewriter<'_> {
    fn input(&self, index: usize) -> Reg {
        self.input(index)
    }
    fn value_type(&self, result: bool, index: usize) -> Type {
        self.value_type(result, index)
    }
    fn emit_at(
        &mut self,
        opcode: GenericOpcode,
        ty: Type,
        inputs: &[Reg],
        result: Option<usize>,
    ) -> Reg {
        self.emit(opcode, ty, inputs, result)
    }
    fn emit_integer_at(
        &mut self,
        opcode: GenericOpcode,
        ty: Type,
        value: i64,
        result: Option<usize>,
    ) -> Reg {
        self.emit_integer(opcode, ty, value, result)
    }
    fn bind(&mut self, result: usize, value: Reg) {
        self.bind(result, value)
    }
}
// Declared contracts are checked even if a particular target rule does not use
// every method yet.
#[allow(dead_code)]
mod host {
    include!(concat!(env!("OUT_DIR"), "/legalize_x86_64.rs"));
}

#[derive(Debug, Clone, Copy)]
pub struct X86_64Legalizer {
    pub features: generated::FeatureSet,
}
impl host::Instruction for crate::target::x86_64::inst::TargetInst {
    const POPCNT32: Self = Self::X86Popcnt32;
    const POPCNT64: Self = Self::X86Popcnt64;
}
impl host::Target for generated::FeatureSet {
    fn supports(&self, instruction: crate::target::x86_64::inst::TargetInst) -> bool {
        self.contains_all(instruction.required_features())
    }
}

impl TargetLegalizer for X86_64Legalizer {
    fn legalize_target(
        &self,
        inst: &veloc_lir::InstRef<'_>,
    ) -> Result<Option<LegalizeAction>, crate::error::Error> {
        let MachineOpcode::Target(code) = inst.opcode() else {
            unreachable!()
        };
        let opcode = TargetInst::from_u32(code);
        Ok((!opcode.is_pseudo()
            && opcode.has_encoding()
            && self.features.contains_all(opcode.required_features()))
        .then_some(LegalizeAction::Legal))
    }

    fn legalize_action(
        &self,
        query: &Query,
    ) -> Result<Option<LegalizeAction>, crate::error::Error> {
        Ok(host::decide(query.opcode, query, &self.features))
    }
}
impl host::Query for Query {
    fn signature(&self, results: &[&[Type]], inputs: &[&[Type]]) -> bool {
        self.signature(results, inputs)
    }
    fn same(&self, indices: &[u32]) -> bool {
        self.same(indices)
    }
    fn value_type(&self, result: bool, index: u32) -> Type {
        if result {
            self.results[index as usize]
        } else {
            self.inputs[index as usize]
        }
    }
    fn input_is(&self, index: u32, ty: Type) -> bool {
        self.input_is(index, ty)
    }
    fn signed_offset(&self, bits: u32) -> bool {
        self.signed_offset(bits)
    }
}
impl host::Type for Type {
    const BOOL: Self = Self::BOOL;
    const I8: Self = Self::I8;
    const I16: Self = Self::I16;
    const I32: Self = Self::I32;
    const I64: Self = Self::I64;
    const F32: Self = Self::F32;
    const F64: Self = Self::F64;
    const PTR: Self = Self::PTR;
}
fn displacement(
    inst_id: InstId,
    mfunc: &mut MachineFunction,
) -> Result<LegalizeResult, crate::error::Error> {
    let opcode = mfunc.inst(inst_id).generic_opcode().unwrap();

    let inst = mfunc.inst(inst_id);
    let (base, offset, value) = match inst.view() {
        veloc_lir::InstView::LoadOffset(load) => (load.base, load.offset, load.dst),
        veloc_lir::InstView::StoreOffset(store) => (store.base, store.offset, store.src),
        _ => unreachable!("offset memory opcode"),
    };
    let memory = inst.memory();
    // x86 disp32 sign-extends. Materialize the full displacement
    // before the access rather than silently truncating it.
    let displacement = mfunc.editor().alloc_vreg(Type::I64);
    let address = mfunc.editor().alloc_vreg(Type::PTR);
    let constant = mfunc
        .editor()
        .writer()
        .constant(Writable(displacement), offset);
    let add = mfunc
        .editor()
        .writer()
        .ptr_add(Writable(address), base, displacement);
    let access = if opcode == GenericOpcode::OffsetLoad {
        mfunc
            .editor()
            .writer()
            .offset_load(Writable(value), address, 0)
    } else {
        mfunc.editor().writer().offset_store(value, address, 0)
    };
    mfunc.editor().set_inst_memory(access, memory);
    mfunc.editor().replace_inst(inst_id, access);
    return Ok(LegalizeResult::Replace(alloc::vec![constant, add, inst_id]));
}
