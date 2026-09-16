use super::*;
use veloc_lir::{InstBuild, InstRead};

#[derive(Debug, Clone, Copy)]
pub struct X86_64Legalizer {
    pub(super) lowering: X86_64Lowering,
}

impl X86_64Legalizer {
    pub fn new(cpu: CpuDescription) -> Self {
        Self {
            lowering: X86_64Lowering::new(cpu),
        }
    }
}

impl TargetLegalizer for X86_64Legalizer {
    fn legalize_action(
        &self,
        inst: &veloc_lir::InstRef<'_>,
        mfunc: &MachineFunction,
    ) -> Result<Option<LegalizeAction>, crate::error::Error> {
        let action = crate::legalize_matcher!(inst, mfunc, {
            Arg => {
                [def(any), imm] => legal,
            };
            Ret => {
                seq[..use(any)] => legal,
            };
            Unreachable => {
                [] => legal,
            };
            Br => {
                [block] => legal,
            };
            Brcond => {
                [use(BOOL), block, block] => legal,
            };
            Call => {
                seq[..def(any), global, ..use(any)] => legal,
            };
            Callind => {
                seq[..def(any), use(PTR), ..use(any)] => legal,
            };
            Add | Sub | Mul => {
                [def(scalar_int(32, 64)), use(scalar_int(32, 64)), use(scalar_int(32, 64))]
                    if same_types(0, 1, 2) => legal,
                [def(scalar_int(8, 16)), use(scalar_int(8, 16)), use(scalar_int(8, 16))]
                    if same_types(0, 1, 2) => widen_scalar(I32),
            };
            And | Or | Xor => {
                [def(BOOL), use(BOOL), use(BOOL)] => legal,
                [def(scalar_int(32, 64)), use(scalar_int(32, 64)), use(scalar_int(32, 64))]
                    if same_types(0, 1, 2) => legal,
                [def(scalar_int(8, 16)), use(scalar_int(8, 16)), use(scalar_int(8, 16))]
                    if same_types(0, 1, 2) => widen_scalar(I32),
            };
            Shl | Lshr | Ashr | Rotl | Rotr | Sdiv | Udiv | Srem | Urem => {
                [def(scalar_int(32, 64)), use(scalar_int(32, 64)), use(scalar_int(32, 64))]
                    if same_types(0, 1, 2) => legal,
            };
            Icmp => {
                [def(BOOL), use(int_or_ptr_scalar(32, 64)), use(int_or_ptr_scalar(32, 64)), condcode]
                    if same_types(1, 2) => legal,
            };
            Fcmp => {
                [def(BOOL), use(scalar_float(32, 64)), use(scalar_float(32, 64)), condcode]
                    if same_types(1, 2) => legal,
            };
            Select => {
                [def(BOOL), use(BOOL), use(BOOL), use(BOOL)]
                    if same_types(0, 2, 3) => legal,
                [def(scalar_value(32, 64)), use(BOOL), use(scalar_value(32, 64)), use(scalar_value(32, 64))]
                    if same_types(0, 2, 3) => legal,
            };
            Load => {
                [def(scalar_value(8, 16, 32, 64)), use(PTR)] => legal,
            };
            StackLoad => {
                [def(scalar_value(8, 16, 32, 64)), stackslot] => legal,
            };
            StackAddr => {
                [def(PTR), stackslot] => legal,
            };
            StackStore => {
                [use(scalar_value(8, 16, 32, 64)), stackslot] => legal,
            };
            Store => {
                [use(scalar_value(8, 16, 32, 64)), use(PTR)] => legal,
            };
            OffsetLoad => {
                [def(scalar_value(8, 16, 32, 64)), use(PTR), imm] => legal,
            };
            OffsetStore => {
                [use(scalar_value(8, 16, 32, 64)), use(PTR), imm] => legal,
            };
            IndexedLoad => {
                [def(scalar_numeric(32, 64)), def(PTR), use(PTR), imm] => legal,
            };
            IndexedStore => {
                [def(PTR), use(scalar_numeric(32, 64)), use(PTR), imm] => legal,
            };
            Constant => {
                [def(BOOL), imm] => legal,
                [def(int_or_ptr_scalar(8, 16, 32, 64)), imm] => legal,
            };
            Ieqz => {
                [def(BOOL), use(scalar_int(32, 64))] => legal,
            };
            Fneg | Fabs => {
                [def(scalar_float(32, 64)), use(scalar_float(32, 64))]
                    if same_types(0, 1) => lower,
            };
            Sitofp => {
                [def(scalar_float(32, 64)), use(scalar_int(32, 64))] => legal,
            };
            Fptosi => {
                [def(scalar_int(32, 64)), use(scalar_float(32, 64))] => legal,
            };
            Uitofp => {
                [def(scalar_float(32, 64)), use(scalar_int(32, 64))] => lower,
            };
            Fptoui => {
                [def(scalar_int(32, 64)), use(scalar_float(32, 64))] => lower,
            };
            Fsqrt => {
                [def(scalar_float(32, 64)), use(scalar_float(32, 64))]
                    if same_types(0, 1) => legal,
            };
            Fpext => { [def(F64), use(F32)] => legal, };
            Fptrunc => { [def(F32), use(F64)] => legal, };
            Zext => {
                [def(scalar_int(32, 64)), use(BOOL)] => legal,
                [def(scalar_int(32, 64)), use(scalar_int(8, 16))] => legal,
                [def(I64), use(I32)] => legal,
            };
            Sext => {
                [def(scalar_int(32, 64)), use(scalar_int(8, 16))] => legal,
                [def(I64), use(I32)] => legal,
            };
            Trunc => {
                [def(scalar_int(8, 16, 32)), use(scalar_int(32, 64))] => legal,
            };
            Inttoptr => { [def(PTR), use(I64)] => legal, };
            Ptrtoint => { [def(I64), use(PTR)] => legal, };
            PtrAdd => {
                [def(PTR), use(PTR), use(I64)] => legal,
            };
            Copy => {
                [def(scalar_value(8, 16, 32, 64)), use(scalar_value(8, 16, 32, 64))]
                    if same_types(0, 1) => legal,
            };
            Bitcast => {
                [def(F32), use(I32)] => legal,
                [def(I32), use(F32)] => legal,
                [def(F64), use(I64)] => legal,
                [def(I64), use(F64)] => legal,
            };
            Fconstant => {
                [def(scalar_float(32, 64)), fimm] => legal,
            };
            Fadd | Fsub | Fmul | Fdiv => {
                [def(scalar_float(32, 64)), use(scalar_float(32, 64)), use(scalar_float(32, 64))]
                    if same_types(0, 1, 2) => legal,
            };
            Brjt => {
                [use(I32)] => lower,
            };
            Ctpop | Ctlz | Cttz => {
                [def(scalar_int(32, 64)), use(scalar_int(32, 64))]
                    if same_types(0, 1) => lower,
            };
        })?;
        if action == Some(LegalizeAction::Legal) {
            let offset = match inst.view() {
                veloc_lir::InstView::LoadOffset(load) => Some(load.offset),
                veloc_lir::InstView::StoreOffset(store) => Some(store.offset),
                _ => None,
            };
            if offset.is_some_and(|offset| i32::try_from(offset).is_err()) {
                return Ok(Some(LegalizeAction::Lower));
            }
        }
        Ok(action)
    }

    fn legalize_instruction(
        &self,
        inst_id: veloc_lir::InstId,
        mfunc: &mut veloc_lir::MachineFunction,
    ) -> Result<LegalizeResult, crate::error::Error> {
        let mut output = Vec::new();
        let opcode = mfunc.inst(inst_id).generic_opcode();
        if let Some(opcode) = opcode {
            match opcode {
                GenericOpcode::Uitofp | GenericOpcode::Fptoui => {
                    let inst = mfunc.inst(inst_id);
                    let veloc_lir::InstView::UnaryReg(unary) = inst.view() else {
                        unreachable!()
                    };
                    self.lowering.unsigned_conversion(
                        mfunc,
                        &mut output,
                        opcode,
                        unary.dst,
                        unary.src,
                    );
                    return Ok(LegalizeResult::Replace(output));
                }
                GenericOpcode::Fneg | GenericOpcode::Fabs => {
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
                    let bits = mfunc.alloc_vreg(integer);
                    output.push(mfunc.writer().unary(
                        MachineOpcode::Generic(GenericOpcode::Bitcast),
                        Writable(bits),
                        unary.src,
                    ));
                    let mask = self.lowering.emit_legalize_constant_reg(
                        mfunc,
                        &mut output,
                        integer,
                        if opcode == GenericOpcode::Fneg {
                            sign
                        } else {
                            sign.wrapping_sub(1)
                        },
                    );
                    let changed = self.lowering.emit_legalize_binary_reg(
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
                    output.push(mfunc.writer().unary(
                        MachineOpcode::Generic(GenericOpcode::Bitcast),
                        Writable(unary.dst),
                        changed,
                    ));
                    return Ok(LegalizeResult::Replace(output));
                }
                GenericOpcode::OffsetLoad | GenericOpcode::OffsetStore => {
                    let inst = mfunc.inst(inst_id);
                    let (base, offset, value) = match inst.view() {
                        veloc_lir::InstView::LoadOffset(load) => (load.base, load.offset, load.dst),
                        veloc_lir::InstView::StoreOffset(store) => {
                            (store.base, store.offset, store.src)
                        }
                        _ => unreachable!("offset memory opcode"),
                    };
                    let memory = inst.memory();
                    // x86 disp32 sign-extends. Materialize the full displacement
                    // before the access rather than silently truncating it.
                    let displacement = mfunc.alloc_vreg(Type::I64);
                    let address = mfunc.alloc_vreg(Type::PTR);
                    let constant = mfunc.writer().constant(Writable(displacement), offset);
                    let add = mfunc
                        .writer()
                        .ptr_add(Writable(address), base, displacement);
                    let access = if opcode == GenericOpcode::OffsetLoad {
                        mfunc.writer().offset_load(Writable(value), address, 0)
                    } else {
                        mfunc.writer().offset_store(value, address, 0)
                    };
                    mfunc.set_inst_memory(access, memory);
                    mfunc.replace_inst(inst_id, access);
                    return Ok(LegalizeResult::Replace(alloc::vec![constant, add, inst_id]));
                }
                GenericOpcode::Ctpop | GenericOpcode::Ctlz | GenericOpcode::Cttz => {
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
                            let _ = self.lowering.legalize_ctpop_into(
                                mfunc,
                                &mut output,
                                unary.src,
                                unary.dst,
                                ty,
                            );
                        }
                        GenericOpcode::Ctlz => {
                            let _ = self.lowering.legalize_ctlz_into(
                                mfunc,
                                &mut output,
                                unary.src,
                                unary.dst,
                                ty,
                            );
                        }
                        GenericOpcode::Cttz => {
                            let _ = self.lowering.legalize_cttz_into(
                                mfunc,
                                &mut output,
                                unary.src,
                                unary.dst,
                                ty,
                            );
                        }
                        _ => unreachable!(),
                    };
                    return Ok(LegalizeResult::Replace(output));
                }
                _ => {}
            }
        }

        if matches!(
            mfunc.inst(inst_id).opcode(),
            MachineOpcode::Generic(veloc_lir::GenericOpcode::Brjt)
        ) {
            let Some(InstExtra::BrTable(info)) = mfunc.inst_extra(inst_id).cloned() else {
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
                let cmp_inst = mfunc.writer().write(
                    MachineOpcode::Target(TargetInst::X86Cmp32ri.as_u32()),
                    &[],
                    &[index],
                    &[InstField::Imm(case_idx as i64)],
                );
                output.push(cmp_inst);

                let je_inst = mfunc.writer().write(
                    MachineOpcode::Target(TargetInst::X86Je.as_u32()),
                    &[],
                    &[],
                    &[InstField::Block(target.block)],
                );
                mfunc.set_inst_extra(
                    je_inst,
                    InstExtra::Branch(veloc_lir::BranchInfo {
                        args: target.args.clone(),
                    }),
                );
                output.push(je_inst);
            }

            let jmp_inst = mfunc.writer().write(
                MachineOpcode::Target(TargetInst::X86Jmp.as_u32()),
                &[],
                &[],
                &[InstField::Block(default_target.block)],
            );
            mfunc.set_inst_extra(
                jmp_inst,
                InstExtra::Branch(veloc_lir::BranchInfo {
                    args: default_target.args.clone(),
                }),
            );
            output.push(jmp_inst);
            return Ok(LegalizeResult::Replace(output));
        }

        Err(crate::error::Error::codegen(alloc::format!(
            "x86_64 missing custom legalizer for opcode {:?}",
            mfunc.inst(inst_id).opcode()
        )))
    }
}
