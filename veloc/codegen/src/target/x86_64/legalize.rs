use super::inst as generated;
use crate::passes::lowering::RewriteContext;
use crate::passes::lowering::legalize::LegalizePolicy;
use veloc_lir::GenericOpcode;
use veloc_lir::{InstBuild, InstRead};
use veloc_mir::Type;

// Declared contracts are checked even if a particular target rule does not use
// every method yet.
#[allow(dead_code)]
mod host {
    include!(concat!(env!("OUT_DIR"), "/legalize_x86_64.rs"));
}

impl host::Instruction for crate::target::x86_64::inst::TargetInst {
    const POPCNT32: &'static [u64] = Self::X86Popcnt32.required_features().as_words();
    const POPCNT64: &'static [u64] = Self::X86Popcnt64.required_features().as_words();
}
impl host::Target for generated::FeatureSet {
    fn words(&self) -> &[u64] {
        self.as_words()
    }
}

pub(super) fn policy(features: &generated::FeatureSet) -> LegalizePolicy<'_> {
    LegalizePolicy {
        program: host::program,
        features: host::Target::words(features),
        predicate: None,
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
    let mut edit = mfunc.editor();
    {
        let mut insert = edit.before(inst_id);
        insert.constant(displacement, offset);
        insert.ptr_add(address, base, displacement);
    }
    let mut writer = edit.replace(inst_id);
    if let Some(memory) = memory {
        writer = writer.with_memory(memory);
    }
    if opcode == GenericOpcode::Load {
        writer.load(value, address, 0);
    } else {
        writer.store(value, address, 0);
    }
    Ok(())
}
