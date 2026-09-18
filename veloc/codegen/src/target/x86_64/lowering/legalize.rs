use super::*;
use crate::passes::lowering::legalize::{Query, Rewrite};
use veloc_lir::{InstBuild, InstRead};

impl host::ValueRewrite for crate::passes::lowering::legalize::ValueRewriter<'_> {
    type Value = Reg;
    fn input(&self, index: usize) -> Reg {
        self.input(index)
    }
    fn value_type(&self, result: bool, index: usize) -> Type {
        self.value_type(result, index)
    }
    fn emit(
        &mut self,
        opcode: GenericOpcode,
        ty: Type,
        inputs: &[Reg],
        result: Option<usize>,
    ) -> Reg {
        self.emit(opcode, ty, inputs, result)
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
    pub cpu: CpuDescription,
}
impl host::Instruction for crate::target::x86_64::isle::TargetInst {
    const POPCNT32: Self = Self::X86Popcnt32;
    const POPCNT64: Self = Self::X86Popcnt64;
}
impl host::Target for CpuDescription {
    fn supports(&self, instruction: crate::target::x86_64::isle::TargetInst) -> bool {
        instruction
            .required_features()
            .iter()
            .all(|feature| self.has_feature(feature))
    }
}

impl TargetLegalizer for X86_64Legalizer {
    fn legalize_action(
        &self,
        query: &Query,
    ) -> Result<Option<LegalizeAction>, crate::error::Error> {
        Ok(host::decide(query.opcode, query, self, &self.cpu))
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
impl host::Recipes for X86_64Legalizer {
    fn legal(&self) -> LegalizeAction {
        LegalizeAction::Legal
    }
    fn unsigned(&self) -> LegalizeAction {
        LegalizeAction::Rewrite(Rewrite {
            name: "unsigned",
            apply: unsigned,
        })
    }
    fn float_sign(&self) -> LegalizeAction {
        LegalizeAction::Rewrite(Rewrite {
            name: "float_sign",
            apply: float_sign,
        })
    }
    fn displacement(&self) -> LegalizeAction {
        LegalizeAction::Rewrite(Rewrite {
            name: "displacement",
            apply: displacement,
        })
    }
    fn bit_count(&self) -> LegalizeAction {
        LegalizeAction::Rewrite(Rewrite {
            name: "bit_count",
            apply: bit_count,
        })
    }
    fn branch_table(&self) -> LegalizeAction {
        LegalizeAction::Rewrite(Rewrite {
            name: "branch_table",
            apply: branch_table,
        })
    }
}

fn unsigned(
    inst_id: InstId,
    mfunc: &mut MachineFunction,
) -> Result<LegalizeResult, crate::error::Error> {
    let mut output = Vec::new();
    let opcode = mfunc.inst(inst_id).generic_opcode().unwrap();

    let inst = mfunc.inst(inst_id);
    let veloc_lir::InstView::UnaryReg(unary) = inst.view() else {
        unreachable!()
    };
    X86_64Lowering::unsigned_conversion(mfunc, &mut output, opcode, unary.dst, unary.src);
    return Ok(LegalizeResult::Replace(output));
}

fn float_sign(
    inst_id: InstId,
    mfunc: &mut MachineFunction,
) -> Result<LegalizeResult, crate::error::Error> {
    let mut output = Vec::new();
    let opcode = mfunc.inst(inst_id).generic_opcode().unwrap();

    let inst = mfunc.inst(inst_id);
    let veloc_lir::InstView::UnaryReg(unary) = inst.view() else {
        unreachable!()
    };
    let float = mfunc.vreg_data(unary.dst).ty;
    let (integer, sign) = if float == Type::F32 {
        (Type::I32, 1i64 << 31)
    } else {
        (Type::I64, i64::MIN)
    };
    let bits = mfunc.editor().alloc_vreg(integer);
    output.push(mfunc.editor().writer().unary(
        MachineOpcode::Generic(GenericOpcode::Bitcast),
        Writable(bits),
        unary.src,
    ));
    let mask = X86_64Lowering::emit_legalize_constant_reg(
        mfunc,
        &mut output,
        integer,
        if opcode == GenericOpcode::Fneg {
            sign
        } else {
            sign.wrapping_sub(1)
        },
    );
    let changed = X86_64Lowering::emit_legalize_binary_reg(
        mfunc,
        &mut output,
        if opcode == GenericOpcode::Fneg {
            GenericOpcode::Xor
        } else {
            GenericOpcode::And
        },
        integer,
        bits,
        mask,
    );
    output.push(mfunc.editor().writer().unary(
        MachineOpcode::Generic(GenericOpcode::Bitcast),
        Writable(unary.dst),
        changed,
    ));
    return Ok(LegalizeResult::Replace(output));
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

fn bit_count(
    inst_id: InstId,
    mfunc: &mut MachineFunction,
) -> Result<LegalizeResult, crate::error::Error> {
    let mut output = Vec::new();
    let opcode = mfunc.inst(inst_id).generic_opcode().unwrap();

    let inst = mfunc.inst(inst_id);
    let veloc_lir::InstView::UnaryReg(unary) = inst.view() else {
        unreachable!("unary legalization opcode");
    };
    let ty = if unary.dst.is_vreg() {
        mfunc.vreg_data(unary.dst).ty
    } else {
        panic!(
            "x86_64 legalization expected virtual register destination for {:?}",
            inst.opcode()
        );
    };
    match opcode {
        GenericOpcode::Ctpop => {
            X86_64Lowering::legalize_ctpop_into(mfunc, &mut output, unary.src, unary.dst, ty)?;
        }
        GenericOpcode::Ctlz => {
            X86_64Lowering::legalize_ctlz_into(mfunc, &mut output, unary.src, unary.dst, ty)?;
        }
        GenericOpcode::Cttz => {
            X86_64Lowering::legalize_cttz_into(mfunc, &mut output, unary.src, unary.dst, ty)?;
        }
        _ => unreachable!(),
    };
    return Ok(LegalizeResult::Replace(output));
}

fn branch_table(
    inst_id: InstId,
    mfunc: &mut MachineFunction,
) -> Result<LegalizeResult, crate::error::Error> {
    let mut output = Vec::new();
    let Some(InstExtra::BrTable(info)) = mfunc.inst_extra(inst_id).map(|e| e.to_owned()) else {
        panic!("missing br_table extra during x86_64 br_table legalization");
    };
    let veloc_lir::InstView::BranchTable(brjt) = mfunc.inst(inst_id).view() else {
        panic!("invalid br_table instruction during x86_64 legalization");
    };

    if info.targets.is_empty() {
        return Ok(LegalizeResult::Replace(output));
    }

    let index = brjt.index;
    let default_target = info.targets.last().unwrap();

    for (case_idx, target) in info.targets[..info.targets.len() - 1].iter().enumerate() {
        let cmp_inst = TargetInst::X86Cmp32ri.write(
            mfunc.editor().writer(),
            &[],
            &[index],
            &[InstField::Imm(case_idx as i64)],
        );
        output.push(cmp_inst);

        let je_inst = TargetInst::X86Je.write(
            mfunc.editor().writer(),
            &[],
            &[],
            &[InstField::Block(target.block)],
        );
        mfunc.editor().set_inst_extra(
            je_inst,
            InstExtra::Branch(veloc_lir::BranchInfo {
                args: target.args.clone(),
            }),
        );
        output.push(je_inst);
    }

    let jmp_inst = TargetInst::X86Jmp.write(
        mfunc.editor().writer(),
        &[],
        &[],
        &[InstField::Block(default_target.block)],
    );
    mfunc.editor().set_inst_extra(
        jmp_inst,
        InstExtra::Branch(veloc_lir::BranchInfo {
            args: default_target.args.clone(),
        }),
    );
    output.push(jmp_inst);
    return Ok(LegalizeResult::Replace(output));
}
