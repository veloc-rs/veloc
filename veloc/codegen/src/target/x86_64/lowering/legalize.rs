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
        mfunc: &MachineFunction<LegalizedMir>,
    ) -> Result<Option<LegalizeAction>, crate::error::Error> {
        crate::legalize_matcher!(inst, mfunc, {
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
            G_SHL | G_LSHR | G_ASHR => {
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
                [def(scalar_numeric(32, 64)), use(PTR)] => legal,
            };
            G_STORE => {
                [use(scalar_numeric(32, 64)), use(PTR)] => legal,
            };
            G_OFFSET_LOAD => {
                [def(scalar_numeric(32, 64)), use(PTR), imm] => legal,
            };
            G_OFFSET_STORE => {
                [use(scalar_numeric(32, 64)), use(PTR), imm] => legal,
            };
            G_INDEXED_LOAD => {
                [def(scalar_numeric(32, 64)), tied(PTR), use(PTR), imm] => legal,
            };
            G_INDEXED_STORE => {
                [tied(PTR), use(scalar_numeric(32, 64)), use(PTR), imm] => legal,
            };
            G_CONSTANT => {
                [def(int_or_ptr_scalar(32, 64)), imm] => legal,
            };
            G_COPY => {
                [def(scalar_value(32, 64)), use(scalar_value(32, 64))]
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
        })
    }

    fn legalize_instruction(
        &self,
        inst_id: crate::mir::InstId,
        mfunc: &mut crate::mir::MachineFunction<LegalizedMir>,
    ) -> Result<LegalizeResult, crate::error::Error> {
        let mut output = Vec::new();
        let opcode = mfunc.dfg[inst_id].generic_opcode();
        if let Some(opcode) = opcode {
            match opcode {
                GenericOpcode::G_CTPOP | GenericOpcode::G_CTLZ | GenericOpcode::G_CTTZ => {
                    let inst = mfunc.dfg[inst_id].clone();
                    let unary = inst.as_unary_reg().unwrap_or_else(|err| {
                        panic!(
                            "invalid unary opcode {:?} during x86_64 legalization: {}",
                            inst.opcode, err
                        );
                    });
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
            MachineOpcode::Generic(crate::mir::GenericOpcode::G_BRJT)
        ) {
            let Some(InstExtra::BrTable(info)) = mfunc.inst_extra(inst_id).cloned() else {
                panic!("missing br_table extra during x86_64 br_table legalization");
            };
            let Ok(brjt) = mfunc.dfg[inst_id].as_branch_table() else {
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
