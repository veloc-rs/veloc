use super::*;

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
        inst: &MachineInst,
        mfunc: &MachineFunction<LegalizedLir>,
    ) -> Result<Option<LegalizeAction>, crate::error::Error> {
        let action = crate::legalize_matcher!(inst, mfunc, {
            G_ARG => {
                [def(any), imm] => legal,
            };
            G_RET => {
                seq[..use(any)] => legal,
            };
            G_UNREACHABLE => {
                [] => legal,
            };
            G_BR => {
                [block] => legal,
            };
            G_BRCOND => {
                [use(BOOL), block, block] => legal,
            };
            G_CALL => {
                seq[..def(any), global, ..use(any)] => legal,
            };
            G_CALLIND => {
                seq[..def(any), use(PTR), ..use(any)] => legal,
            };
            G_ADD | G_SUB | G_MUL | G_AND | G_OR | G_XOR => {
                [def(scalar_int(32, 64)), use(scalar_int(32, 64)), use(scalar_int(32, 64))]
                    if same_types(0, 1, 2) => legal,
                [def(scalar_int(8, 16)), use(scalar_int(8, 16)), use(scalar_int(8, 16))]
                    if same_types(0, 1, 2) => widen_scalar(I32),
            };
            G_AND | G_OR | G_XOR => {
                [def(BOOL), use(BOOL), use(BOOL)] => legal,
            };
            G_SHL | G_LSHR | G_ASHR | G_ROTL | G_ROTR | G_SDIV | G_UDIV | G_SREM | G_UREM => {
                [def(scalar_int(32, 64)), use(scalar_int(32, 64)), use(scalar_int(32, 64))]
                    if same_types(0, 1, 2) => legal,
            };
            G_ICMP => {
                [def(BOOL), use(int_or_ptr_scalar(32, 64)), use(int_or_ptr_scalar(32, 64)), condcode]
                    if same_types(1, 2) => legal,
            };
            G_FCMP => {
                [def(BOOL), use(scalar_float(32, 64)), use(scalar_float(32, 64)), condcode]
                    if same_types(1, 2) => legal,
            };
            G_SELECT => {
                [def(BOOL), use(BOOL), use(BOOL), use(BOOL)]
                    if same_types(0, 2, 3) => legal,
                [def(scalar_value(32, 64)), use(BOOL), use(scalar_value(32, 64)), use(scalar_value(32, 64))]
                    if same_types(0, 2, 3) => legal,
            };
            G_LOAD => {
                [def(scalar_value(8, 16, 32, 64)), use(PTR)] => legal,
            };
            G_STACK_LOAD => {
                [def(scalar_value(8, 16, 32, 64)), stackslot] => legal,
            };
            G_STACK_ADDR => {
                [def(PTR), stackslot] => legal,
            };
            G_STACK_STORE => {
                [use(scalar_value(8, 16, 32, 64)), stackslot] => legal,
            };
            G_STORE => {
                [use(scalar_value(8, 16, 32, 64)), use(PTR)] => legal,
            };
            G_OFFSET_LOAD => {
                [def(scalar_value(8, 16, 32, 64)), use(PTR), imm] => legal,
            };
            G_OFFSET_STORE => {
                [use(scalar_value(8, 16, 32, 64)), use(PTR), imm] => legal,
            };
            G_INDEXED_LOAD => {
                [def(scalar_numeric(32, 64)), tied(PTR), use(PTR), imm] => legal,
            };
            G_INDEXED_STORE => {
                [tied(PTR), use(scalar_numeric(32, 64)), use(PTR), imm] => legal,
            };
            G_CONSTANT => {
                [def(BOOL), imm] => legal,
                [def(int_or_ptr_scalar(32, 64)), imm] => legal,
            };
            G_IEQZ => {
                [def(BOOL), use(scalar_int(32, 64))] => legal,
            };
            G_FNEG | G_FABS => {
                [def(scalar_float(32, 64)), use(scalar_float(32, 64))]
                    if same_types(0, 1) => lower,
            };
            G_SITOFP => {
                [def(scalar_float(32, 64)), use(scalar_int(32, 64))] => legal,
            };
            G_FPTOSI => {
                [def(scalar_int(32, 64)), use(scalar_float(32, 64))] => legal,
            };
            G_UITOFP => {
                [def(scalar_float(32, 64)), use(scalar_int(32, 64))] => lower,
            };
            G_FPTOUI => {
                [def(scalar_int(32, 64)), use(scalar_float(32, 64))] => lower,
            };
            G_FSQRT => {
                [def(scalar_float(32, 64)), use(scalar_float(32, 64))]
                    if same_types(0, 1) => legal,
            };
            G_FPEXT => { [def(F64), use(F32)] => legal, };
            G_FPTRUNC => { [def(F32), use(F64)] => legal, };
            G_ZEXT => {
                [def(scalar_int(32, 64)), use(BOOL)] => legal,
                [def(scalar_int(32, 64)), use(scalar_int(8, 16))] => legal,
                [def(I64), use(I32)] => legal,
            };
            G_SEXT => {
                [def(scalar_int(32, 64)), use(scalar_int(8, 16))] => legal,
                [def(I64), use(I32)] => legal,
            };
            G_TRUNC => {
                [def(scalar_int(8, 16, 32)), use(scalar_int(32, 64))] => legal,
            };
            G_INTTOPTR => { [def(PTR), use(I64)] => legal, };
            G_PTRTOINT => { [def(I64), use(PTR)] => legal, };
            G_PTR_ADD => {
                [def(PTR), use(PTR), use(I64)] => legal,
            };
            G_COPY => {
                [def(scalar_value(8, 16, 32, 64)), use(scalar_value(8, 16, 32, 64))]
                    if same_types(0, 1) => legal,
            };
            G_BITCAST => {
                [def(F32), use(I32)] => legal,
                [def(I32), use(F32)] => legal,
                [def(F64), use(I64)] => legal,
                [def(I64), use(F64)] => legal,
            };
            G_FCONSTANT => {
                [def(scalar_float(32, 64)), fimm] => legal,
            };
            G_FADD | G_FSUB | G_FMUL | G_FDIV => {
                [def(scalar_float(32, 64)), use(scalar_float(32, 64)), use(scalar_float(32, 64))]
                    if same_types(0, 1, 2) => legal,
            };
            G_BRJT => {
                [use(I32)] => lower,
            };
            G_CTPOP | G_CTLZ | G_CTTZ => {
                [def(scalar_int(32, 64)), use(scalar_int(32, 64))]
                    if same_types(0, 1) => lower,
            };
        })?;
        if action == Some(LegalizeAction::Legal) {
            let offset = match inst.generic_view()? {
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
        mfunc: &mut veloc_lir::MachineFunction<LegalizedLir>,
    ) -> Result<LegalizeResult, crate::error::Error> {
        let mut output = Vec::new();
        let opcode = mfunc.dfg[inst_id].generic_opcode();
        if let Some(opcode) = opcode {
            match opcode {
                GenericOpcode::G_UITOFP | GenericOpcode::G_FPTOUI => {
                    let inst = mfunc.dfg[inst_id].clone();
                    let veloc_lir::InstView::UnaryReg(unary) = inst.generic_view()? else {
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
                GenericOpcode::G_FNEG | GenericOpcode::G_FABS => {
                    let inst = mfunc.dfg[inst_id].clone();
                    let veloc_lir::InstView::UnaryReg(unary) = inst.generic_view()? else {
                        unreachable!()
                    };
                    let float = mfunc.vreg_data(unary.dst).ty;
                    let (integer, sign) = if float == Type::F32 {
                        (Type::I32, 1i64 << 31)
                    } else {
                        (Type::I64, i64::MIN)
                    };
                    let bits = mfunc.alloc_vreg(integer);
                    output.push(mfunc.alloc_inst(MachineInst::build_unary(
                        MachineOpcode::Generic(GenericOpcode::G_BITCAST),
                        Writable(bits),
                        unary.src,
                    )));
                    let mask = self.lowering.emit_legalize_constant_reg(
                        mfunc,
                        &mut output,
                        integer,
                        if opcode == GenericOpcode::G_FNEG {
                            sign
                        } else {
                            !sign
                        },
                    );
                    let changed = self.lowering.emit_legalize_binary_reg(
                        mfunc,
                        &mut output,
                        if opcode == GenericOpcode::G_FNEG {
                            GenericOpcode::G_XOR
                        } else {
                            GenericOpcode::G_AND
                        },
                        integer,
                        bits,
                        mask,
                    );
                    output.push(mfunc.alloc_inst(MachineInst::build_unary(
                        MachineOpcode::Generic(GenericOpcode::G_BITCAST),
                        Writable(unary.dst),
                        changed,
                    )));
                    return Ok(LegalizeResult::Replace(output));
                }
                GenericOpcode::G_OFFSET_LOAD | GenericOpcode::G_OFFSET_STORE => {
                    let inst = mfunc.dfg[inst_id].clone();
                    let (base, offset, value) = match inst.generic_view()? {
                        veloc_lir::InstView::LoadOffset(load) => (load.base, load.offset, load.dst),
                        veloc_lir::InstView::StoreOffset(store) => {
                            (store.base, store.offset, store.src)
                        }
                        _ => unreachable!("offset memory opcode"),
                    };
                    // x86 disp32 sign-extends. Materialize the full displacement
                    // before the access rather than silently truncating it.
                    let displacement = mfunc.alloc_vreg(Type::I64);
                    let address = mfunc.alloc_vreg(Type::PTR);
                    let constant = mfunc
                        .alloc_inst(MachineInst::build_constant(Writable(displacement), offset));
                    let add = mfunc.alloc_inst(MachineInst::build_ptr_add(
                        Writable(address),
                        base,
                        displacement,
                    ));
                    let mut access = if opcode == GenericOpcode::G_OFFSET_LOAD {
                        MachineInst::build_offset_load(Writable(value), address, 0)
                    } else {
                        MachineInst::build_offset_store(value, address, 0)
                    };
                    access.memory = inst.memory;
                    mfunc.replace_inst(inst_id, access);
                    return Ok(LegalizeResult::Replace(alloc::vec![constant, add, inst_id]));
                }
                GenericOpcode::G_CTPOP | GenericOpcode::G_CTLZ | GenericOpcode::G_CTTZ => {
                    let inst = mfunc.dfg[inst_id].clone();
                    let veloc_lir::InstView::UnaryReg(unary) = inst.generic_view()? else {
                        unreachable!("unary legalization opcode");
                    };
                    let ty = if unary.dst.is_vreg() {
                        mfunc.vreg_data(unary.dst).ty
                    } else {
                        panic!(
                            "x86_64 legalization expected virtual register destination for {:?}",
                            inst.opcode
                        );
                    };
                    match opcode {
                        GenericOpcode::G_CTPOP => {
                            let _ = self.lowering.legalize_ctpop_into(
                                mfunc,
                                &mut output,
                                unary.src,
                                unary.dst,
                                ty,
                            );
                        }
                        GenericOpcode::G_CTLZ => {
                            let _ = self.lowering.legalize_ctlz_into(
                                mfunc,
                                &mut output,
                                unary.src,
                                unary.dst,
                                ty,
                            );
                        }
                        GenericOpcode::G_CTTZ => {
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
            mfunc.dfg[inst_id].opcode,
            MachineOpcode::Generic(veloc_lir::GenericOpcode::G_BRJT)
        ) {
            let Some(InstExtra::BrTable(info)) = mfunc.inst_extra(inst_id).cloned() else {
                panic!("missing br_table extra during x86_64 br_table legalization");
            };
            let veloc_lir::InstView::BranchTable(brjt) = mfunc.dfg[inst_id].generic_view()? else {
                panic!("invalid br_table instruction during x86_64 legalization");
            };

            if info.targets.is_empty() {
                return Ok(LegalizeResult::Replace(output));
            }

            let index = brjt.index;
            let default_target = info.targets.last().unwrap();
            debug_assert!(
                info.targets.iter().all(|target| target.args.is_empty()),
                "edge arguments should be lowered before x86_64 br_table legalization"
            );

            for (case_idx, target) in info.targets[..info.targets.len() - 1].iter().enumerate() {
                let cmp_inst = MachineInst::build_generic(
                    MachineOpcode::Target(TargetInst::X86Cmp32ri.as_u32()),
                    smallvec::smallvec![
                        MachineOperand::Use(index),
                        MachineOperand::Imm(case_idx as i64),
                    ],
                );
                output.push(mfunc.alloc_inst(cmp_inst));

                let je_inst = MachineInst::build_generic(
                    MachineOpcode::Target(TargetInst::X86Je.as_u32()),
                    smallvec::smallvec![MachineOperand::Block(target.block)],
                );
                output.push(mfunc.alloc_inst(je_inst));
            }

            let jmp_inst = MachineInst::build_generic(
                MachineOpcode::Target(TargetInst::X86Jmp.as_u32()),
                smallvec::smallvec![MachineOperand::Block(default_target.block)],
            );
            output.push(mfunc.alloc_inst(jmp_inst));
            return Ok(LegalizeResult::Replace(output));
        }

        Err(crate::error::Error::codegen(alloc::format!(
            "x86_64 missing custom legalizer for opcode {:?}",
            mfunc.dfg[inst_id].opcode
        )))
    }
}
