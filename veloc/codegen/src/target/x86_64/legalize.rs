use super::inst as generated;
use crate::passes::lowering::legalize::Query;
use crate::passes::lowering::{LegalizeAction, RewriteContext};
use crate::target::TargetLegalizer;
use veloc_lir::{GenericOpcode, Writable};
use veloc_lir::{InstBuild, InstRead};
use veloc_mir::Type;

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
    fn legalize_action(
        &self,
        query: &Query<'_>,
    ) -> Result<Option<LegalizeAction>, crate::error::Error> {
        Ok(host::decide(query.opcode(), query, &self.features))
    }
}
fn displacement(mfunc: &mut RewriteContext<'_>) -> Result<(), crate::error::Error> {
    let inst_id = mfunc.root();
    let opcode = mfunc.inst(inst_id).generic_opcode().unwrap();

    let inst = mfunc.inst(inst_id);
    let (base, offset, value) = match inst.view() {
        veloc_lir::InstView::Load(load) => (load.base, load.offset, load.dst),
        veloc_lir::InstView::Store(store) => (store.base, store.offset, store.src),
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
    let access = if opcode == GenericOpcode::Load {
        mfunc.editor().writer().load(Writable(value), address, 0)
    } else {
        mfunc.editor().writer().store(value, address, 0)
    };
    mfunc.editor().set_inst_memory(access, memory);
    mfunc.editor().replace_inst(inst_id, access);
    mfunc.replace(&[constant, add, inst_id]);
    Ok(())
}
