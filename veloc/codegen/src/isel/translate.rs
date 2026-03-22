//! IR to Machine IR Translator
//!
//! 将 SSA IR 转换为机器无关的中间表示 (MIR)
//! 这是 GlobalISel 流程的第一步

use crate::error::{Error, Result};
use crate::mir::{
    BrTableInfo, BrTableTarget, BranchCondInfo, BranchInfo, CallInfo, GenericOpcode, InstExtra,
    MachineBlock, MachineFunction, MachineInst, MachineModule, MachineOpcode, MachineOperand, Reg,
};
use alloc::format;
use hashbrown::HashMap;
use veloc_ir::{Function, InstructionData, Module, Opcode, Value};

/// IR 到 MIR 的翻译器
pub struct IRTranslator<'a> {
    module: &'a Module,
}

/// 翻译上下文，用于在翻译过程中共享状态
struct TranslationContext<'a> {
    func: &'a Function,
    mmodule: &'a mut MachineModule,
    mfunc: MachineFunction,
    value_map: HashMap<Value, Reg>,
}

struct TranslatedInst {
    inst: MachineInst,
    extra: Option<InstExtra>,
}

impl TranslatedInst {
    fn with_extra(inst: MachineInst, extra: InstExtra) -> Self {
        Self {
            inst,
            extra: Some(extra),
        }
    }
}

impl From<MachineInst> for TranslatedInst {
    fn from(inst: MachineInst) -> Self {
        Self { inst, extra: None }
    }
}

impl<'a> IRTranslator<'a> {
    pub fn new(module: &'a Module) -> Self {
        Self { module }
    }

    /// 为函数参数生成 G_ARG 指令
    fn lower_arguments(&self, ctx: &mut TranslationContext, mblock: &mut MachineBlock) {
        use crate::mir::Writable;
        for (idx, &param_val) in ctx.func.params().iter().enumerate() {
            let vreg = ctx.value_map[&param_val];
            let arg_inst = MachineInst::build_arg(Writable(vreg), idx as i64);
            Self::emit_inst(ctx, mblock, arg_inst);
        }
    }

    fn emit_inst(
        ctx: &mut TranslationContext,
        mblock: &mut MachineBlock,
        inst: MachineInst,
    ) -> crate::mir::InstId {
        let inst_id = ctx.mfunc.alloc_inst(inst);
        mblock.append_inst_id(inst_id);
        inst_id
    }

    /// 将 IR 模块翻译为 MachineModule
    pub fn translate_module(&self) -> Result<MachineModule> {
        // IR Module 目前没有直接的 name 字段，可以根据需要从其他地方获取或使用默认值
        let mut mmodule = MachineModule::new(alloc::string::String::from("default"));

        for (_, func) in self.module.functions.iter() {
            let mfunc = self.translate_function(func, &mut mmodule)?;
            mmodule.add_function(mfunc);
        }

        Ok(mmodule)
    }

    /// 将 IR 函数翻译为 MachineFunction
    fn translate_function(
        &self,
        func: &Function,
        mmodule: &mut MachineModule,
    ) -> Result<MachineFunction> {
        let mut ctx = TranslationContext {
            func,
            mmodule,
            mfunc: MachineFunction::new(func.name.clone()),
            value_map: HashMap::new(),
        };

        // 1. 预分配所有 Value 对应的 VReg
        for (val, data) in &func.dfg.values {
            let vreg = ctx.mfunc.alloc_vreg(data.ty.clone());
            ctx.value_map.insert(val, vreg);
        }

        // 2. 翻译基本块和指令
        for (idx, &block_id) in func.layout.block_order.iter().enumerate() {
            let mut mblock = MachineBlock::new(block_id);
            mblock.params = func.layout.blocks[block_id]
                .params
                .iter()
                .map(|value| ctx.value_map[value])
                .collect();

            // 如果是入口块，先处理函数参数
            if idx == 0 {
                self.lower_arguments(&mut ctx, &mut mblock);
            }

            for &inst_id in &func.layout.blocks[block_id].insts {
                let translated = self.translate_instruction(inst_id, &mut ctx, &mut mblock)?;
                let m_inst_id = ctx.mfunc.alloc_inst(translated.inst);
                if let Some(extra) = translated.extra {
                    ctx.mfunc.set_inst_extra(m_inst_id, extra);
                }
                mblock.append_inst_id(m_inst_id);
            }

            ctx.mfunc.blocks.push(mblock);
        }

        for &param in func.params() {
            if let Some(&reg) = ctx.value_map.get(&param) {
                ctx.mfunc.params.push(reg);
            }
        }

        Ok(ctx.mfunc)
    }

    /// 翻译单条指令
    fn translate_instruction(
        &self,
        inst_id: veloc_ir::Inst,
        ctx: &mut TranslationContext,
        mblock: &mut MachineBlock,
    ) -> Result<TranslatedInst> {
        use crate::mir::Writable;
        use smallvec::SmallVec;

        let inst_data = &ctx.func.dfg.instructions[inst_id];

        // 获取结果寄存器 (Defs)
        let results = ctx.func.dfg.inst_results(inst_id);
        let mut defs = SmallVec::<[MachineOperand; 1]>::new();
        for &res in results {
            let vreg = ctx.value_map[&res];
            defs.push(MachineOperand::Def(Writable(vreg)));
        }

        match inst_data {
            InstructionData::Binary { opcode, args } => {
                let src0 = ctx.value_map[&args[0]];
                let src1 = ctx.value_map[&args[1]];

                let m_opcode = match opcode {
                    Opcode::IAdd => MachineOpcode::Generic(GenericOpcode::G_ADD),
                    Opcode::ISub => MachineOpcode::Generic(GenericOpcode::G_SUB),
                    Opcode::IMul => MachineOpcode::Generic(GenericOpcode::G_MUL),
                    Opcode::IDivS => MachineOpcode::Generic(GenericOpcode::G_SDIV),
                    Opcode::IDivU => MachineOpcode::Generic(GenericOpcode::G_UDIV),
                    Opcode::IAnd => MachineOpcode::Generic(GenericOpcode::G_AND),
                    Opcode::IOr => MachineOpcode::Generic(GenericOpcode::G_OR),
                    Opcode::IXor => MachineOpcode::Generic(GenericOpcode::G_XOR),
                    Opcode::IShl => MachineOpcode::Generic(GenericOpcode::G_SHL),
                    Opcode::IShrS => MachineOpcode::Generic(GenericOpcode::G_ASHR),
                    Opcode::IShrU => MachineOpcode::Generic(GenericOpcode::G_LSHR),
                    Opcode::FAdd => MachineOpcode::Generic(GenericOpcode::G_FADD),
                    Opcode::FSub => MachineOpcode::Generic(GenericOpcode::G_FSUB),
                    Opcode::FMul => MachineOpcode::Generic(GenericOpcode::G_FMUL),
                    Opcode::FDiv => MachineOpcode::Generic(GenericOpcode::G_FDIV),
                    _ => {
                        return Err(Error::unsupported_binary_opcode(*opcode));
                    }
                };

                Ok(
                    MachineInst::build_binary(m_opcode, defs[0].as_writable().unwrap(), src0, src1)
                        .into(),
                )
            }

            InstructionData::Unary { opcode, arg } => {
                let src = ctx.value_map[arg];

                let m_opcode = match opcode {
                    Opcode::INeg => MachineOpcode::Generic(GenericOpcode::G_NEG),
                    Opcode::IClz => MachineOpcode::Generic(GenericOpcode::G_CTLZ),
                    Opcode::ICtz => MachineOpcode::Generic(GenericOpcode::G_CTTZ),
                    Opcode::IPopcnt => MachineOpcode::Generic(GenericOpcode::G_CTPOP),
                    Opcode::FNeg => MachineOpcode::Generic(GenericOpcode::G_FNEG),
                    Opcode::IEqz => MachineOpcode::Generic(GenericOpcode::G_IEQZ),
                    Opcode::Wrap => MachineOpcode::Generic(GenericOpcode::G_TRUNC),
                    Opcode::ExtendU => MachineOpcode::Generic(GenericOpcode::G_ZEXT),
                    Opcode::ExtendS => MachineOpcode::Generic(GenericOpcode::G_SEXT),
                    Opcode::FloatDemote => MachineOpcode::Generic(GenericOpcode::G_FPTRUNC),
                    Opcode::FloatPromote => MachineOpcode::Generic(GenericOpcode::G_FPEXT),
                    Opcode::FloatToIntU => MachineOpcode::Generic(GenericOpcode::G_FPTOUI),
                    Opcode::FloatToIntS => MachineOpcode::Generic(GenericOpcode::G_FPTOSI),
                    Opcode::IntToFloatU => MachineOpcode::Generic(GenericOpcode::G_UITOFP),
                    Opcode::IntToFloatS => MachineOpcode::Generic(GenericOpcode::G_SITOFP),
                    Opcode::Reinterpret => MachineOpcode::Generic(GenericOpcode::G_BITCAST),
                    _ => {
                        return Err(Error::unsupported_unary_opcode(*opcode));
                    }
                };

                Ok(MachineInst::build_unary(m_opcode, defs[0].as_writable().unwrap(), src).into())
            }

            InstructionData::IntCompare { kind, args } => {
                let src0 = ctx.value_map[&args[0]];
                let src1 = ctx.value_map[&args[1]];

                Ok(
                    MachineInst::build_icmp(defs[0].as_writable().unwrap(), src0, src1, *kind)
                        .into(),
                )
            }

            InstructionData::FloatCompare { kind, args } => {
                let src0 = ctx.value_map[&args[0]];
                let src1 = ctx.value_map[&args[1]];

                Ok(
                    MachineInst::build_fcmp(defs[0].as_writable().unwrap(), src0, src1, *kind)
                        .into(),
                )
            }

            InstructionData::Load { ptr, offset, .. } => {
                let base = ctx.value_map[ptr];
                Ok(MachineInst::build_load_offset(
                    defs[0].as_writable().unwrap(),
                    base,
                    *offset as i64,
                )
                .into())
            }

            InstructionData::Store {
                ptr, value, offset, ..
            } => {
                let val = ctx.value_map[value];
                let base = ctx.value_map[ptr];
                Ok(MachineInst::build_store_offset(val, base, *offset as i64).into())
            }

            InstructionData::Iconst { value: imm } => {
                Ok(MachineInst::build_constant(defs[0].as_writable().unwrap(), *imm as i64).into())
            }

            InstructionData::Fconst { value } => {
                let dst = defs[0].as_writable().unwrap();
                let dst_ty = ctx.mfunc.vreg_data(dst.to_reg()).ty;

                let (bits_ty, bits_imm) = if dst_ty == veloc_ir::Type::F32 {
                    (veloc_ir::Type::I32, (*value as u32) as i64)
                } else if dst_ty == veloc_ir::Type::F64 {
                    (veloc_ir::Type::I64, *value as i64)
                } else {
                    return Err(Error::translate(format!(
                        "Unsupported float constant type: {:?}",
                        dst_ty
                    )));
                };

                let bits_reg = ctx.mfunc.alloc_vreg(bits_ty);
                let bits_inst = MachineInst::build_constant(Writable(bits_reg), bits_imm);
                Self::emit_inst(ctx, mblock, bits_inst);

                Ok(MachineInst::build_unary(
                    MachineOpcode::Generic(GenericOpcode::G_BITCAST),
                    dst,
                    bits_reg,
                )
                .into())
            }

            InstructionData::Jump { dest } => {
                let target = ctx.func.dfg.block_call_block(*dest);
                let args = ctx
                    .func
                    .dfg
                    .block_call_args(*dest)
                    .iter()
                    .map(|value| ctx.value_map[value])
                    .collect::<SmallVec<[Reg; 2]>>();
                let inst = MachineInst::build_br(target);
                if args.is_empty() {
                    Ok(inst.into())
                } else {
                    Ok(TranslatedInst::with_extra(
                        inst,
                        InstExtra::Branch(BranchInfo { args }),
                    ))
                }
            }

            InstructionData::Br {
                condition,
                then_dest,
                else_dest,
            } => {
                let cond_vreg = ctx.value_map[condition];
                let then_args = ctx
                    .func
                    .dfg
                    .block_call_args(*then_dest)
                    .iter()
                    .map(|value| ctx.value_map[value])
                    .collect::<SmallVec<[Reg; 2]>>();
                let else_args = ctx
                    .func
                    .dfg
                    .block_call_args(*else_dest)
                    .iter()
                    .map(|value| ctx.value_map[value])
                    .collect::<SmallVec<[Reg; 2]>>();

                let inst = MachineInst::build_br_cond(
                    cond_vreg,
                    ctx.func.dfg.block_call_block(*then_dest),
                    ctx.func.dfg.block_call_block(*else_dest),
                );
                if then_args.is_empty() && else_args.is_empty() {
                    Ok(inst.into())
                } else {
                    Ok(TranslatedInst::with_extra(
                        inst,
                        InstExtra::BranchCond(BranchCondInfo {
                            then_args,
                            else_args,
                        }),
                    ))
                }
            }

            InstructionData::BrTable { index, .. } => {
                let idx_vreg = ctx.value_map[index];
                let InstructionData::BrTable { table, .. } = inst_data else {
                    unreachable!();
                };
                let targets = ctx
                    .func
                    .dfg
                    .jump_table_targets(*table)
                    .iter()
                    .map(|&call| BrTableTarget {
                        block: ctx.func.dfg.block_call_block(call),
                        args: ctx
                            .func
                            .dfg
                            .block_call_args(call)
                            .iter()
                            .map(|value| ctx.value_map[value])
                            .collect(),
                    })
                    .collect();

                Ok(TranslatedInst::with_extra(
                    MachineInst::build_br_jt(idx_vreg),
                    InstExtra::BrTable(BrTableInfo { targets }),
                ))
            }

            InstructionData::Return { values } => {
                let ret_values = ctx.func.dfg.get_value_list(*values);
                let mut rets = SmallVec::new();
                for &v in ret_values {
                    let vreg = ctx.value_map[&v];
                    rets.push(vreg);
                }
                Ok(MachineInst::build_ret(rets).into())
            }

            InstructionData::Call { func_id, args } => {
                let call_args = ctx.func.dfg.get_value_list(*args);
                let sym_id = ctx
                    .mmodule
                    .symbols_mut()
                    .get_or_create_func(*func_id, self.module);
                let call_inst = MachineInst::build_call(
                    defs.iter().map(|operand| operand.as_writable().unwrap()),
                    sym_id,
                    call_args.iter().map(|value| ctx.value_map[value]),
                );
                let sig_id = self.module.get_function(*func_id).signature;
                let call_info = CallInfo {
                    sig: self.module.get_signature(sig_id).clone(),
                };

                Ok(TranslatedInst::with_extra(
                    call_inst,
                    InstExtra::Call(call_info),
                ))
            }

            InstructionData::CallIndirect { ptr, args, sig_id } => {
                let call_args = ctx.func.dfg.get_value_list(*args);
                let call_inst = MachineInst::build_call_indirect(
                    defs.iter().map(|operand| operand.as_writable().unwrap()),
                    ctx.value_map[ptr],
                    call_args.iter().map(|value| ctx.value_map[value]),
                );
                let call_info = CallInfo {
                    sig: self.module.get_signature(*sig_id).clone(),
                };

                Ok(TranslatedInst::with_extra(
                    call_inst,
                    InstExtra::Call(call_info),
                ))
            }

            InstructionData::Ternary { opcode, args } => {
                let v0 = ctx.value_map[&args[0]];
                let v1 = ctx.value_map[&args[1]];
                let v2 = ctx.value_map[&args[2]];

                match opcode {
                    Opcode::Select => {
                        Ok(MachineInst::build_select(defs[0].as_writable().unwrap(), v0, v1, v2).into())
                    }
                    _ => Err(Error::translate(format!(
                        "Unsupported ternary opcode: {:?}",
                        opcode
                    ))),
                }
            }

            InstructionData::IntToPtr { arg } => {
                let src = ctx.value_map[arg];
                Ok(MachineInst::build_unary(
                    MachineOpcode::Generic(GenericOpcode::G_INTTOPTR),
                    defs[0].as_writable().unwrap(),
                    src,
                )
                .into())
            }

            InstructionData::PtrToInt { arg } => {
                let src = ctx.value_map[arg];
                Ok(MachineInst::build_unary(
                    MachineOpcode::Generic(GenericOpcode::G_PTRTOINT),
                    defs[0].as_writable().unwrap(),
                    src,
                )
                .into())
            }

            InstructionData::PtrOffset { ptr, offset } => {
                use crate::mir::Writable;

                let addr = ctx.value_map[ptr];
                if *offset == 0 {
                    Ok(MachineInst::build_copy(defs[0].as_writable().unwrap(), addr).into())
                } else {
                    let off_reg = ctx.mfunc.alloc_vreg(veloc_ir::Type::I64);
                    Self::emit_inst(
                        ctx,
                        mblock,
                        MachineInst::build_constant(Writable(off_reg), *offset as i64),
                    );

                    let ptr_reg = ctx.mfunc.alloc_vreg(veloc_ir::Type::I64);
                    Self::emit_inst(
                        ctx,
                        mblock,
                        MachineInst::build_binary(
                            MachineOpcode::Generic(GenericOpcode::G_PTR_ADD),
                            Writable(ptr_reg),
                            addr,
                            off_reg,
                        ),
                    );

                    Ok(MachineInst::build_copy(defs[0].as_writable().unwrap(), ptr_reg).into())
                }
            }

            InstructionData::PtrIndex { ptr, index, imm_id } => {
                let base_ptr = ctx.value_map[ptr];
                let idx = ctx.value_map[index];
                let imm = ctx.func.dfg.ptr_imm_pool[*imm_id];

                // 1. scale index: idx * scale
                let scaled_idx = if imm.scale != 1 {
                    let scale_reg = ctx.mfunc.alloc_vreg(veloc_ir::Type::I64);
                    let scale_inst =
                        MachineInst::build_constant(Writable(scale_reg), imm.scale as i64);
                    Self::emit_inst(ctx, mblock, scale_inst);

                    let res_reg = ctx.mfunc.alloc_vreg(veloc_ir::Type::I64);
                    let mul_inst = MachineInst::build_binary(
                        MachineOpcode::Generic(GenericOpcode::G_MUL),
                        Writable(res_reg),
                        idx,
                        scale_reg,
                    );
                    Self::emit_inst(ctx, mblock, mul_inst);
                    res_reg
                } else {
                    idx
                };

                // 2. add offset if any: base_idx = (idx * scale) + offset
                let base_idx = if imm.offset != 0 {
                    let off_reg = ctx.mfunc.alloc_vreg(veloc_ir::Type::I64);
                    let off_inst =
                        MachineInst::build_constant(Writable(off_reg), imm.offset as i64);
                    Self::emit_inst(ctx, mblock, off_inst);

                    let res_reg = ctx.mfunc.alloc_vreg(veloc_ir::Type::I64);
                    let add_inst = MachineInst::build_binary(
                        MachineOpcode::Generic(GenericOpcode::G_ADD),
                        Writable(res_reg),
                        scaled_idx,
                        off_reg,
                    );
                    Self::emit_inst(ctx, mblock, add_inst);
                    res_reg
                } else {
                    scaled_idx
                };

                // 3. ptr_add: ptr + base_idx
                Ok(MachineInst::build_binary(
                    MachineOpcode::Generic(GenericOpcode::G_PTR_ADD),
                    defs[0].as_writable().unwrap(),
                    base_ptr,
                    base_idx,
                )
                .into())
            }
            InstructionData::Unreachable => Ok(MachineInst::build_unreachable().into()),

            _ => Err(Error::translate(format!(
                "InstructionData variant not implemented for translation: {:?}",
                inst_data
            ))),
        }
    }
}
