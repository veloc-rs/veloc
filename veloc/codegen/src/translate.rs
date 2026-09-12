//! Middle-level IR (MIR) to low-level IR (LIR) translator.
//!
//! 将 SSA MIR 转换为面向机器的 LIR。
//! 这是 GlobalISel 流程的第一步

use crate::error::{Error, Result};
use alloc::format;
use cranelift_entity::PrimaryMap;
use veloc_lir::stages::RawLir;
use veloc_lir::{
    BrTableInfo, BrTableTarget, BranchCondInfo, BranchInfo, CallInfo, GenericOpcode, InstExtra,
    MachineBlock, MachineFunction, MachineInst, MachineModule, MachineOpcode, MachineOperand, Reg,
};
use veloc_mir::{Function, InstView, Module, Opcode, Value};

include!(concat!(env!("OUT_DIR"), "/mir_lowering.rs"));

#[cfg(test)]
mod tests;

/// IR 到 LIR 的翻译器
pub struct IRTranslator<'a> {
    module: &'a Module,
    layout: crate::target::arch::DataLayout,
}

/// 翻译上下文，用于在翻译过程中共享状态
struct TranslationContext<'a> {
    func: &'a Function,
    mmodule: &'a mut MachineModule,
    mfunc: MachineFunction<RawLir>,
    value_map: PrimaryMap<Value, Reg>,
    slots: hashbrown::HashMap<veloc_mir::Inst, veloc_lir::StackSlot>,
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
    pub fn new(module: &'a Module, layout: crate::target::arch::DataLayout) -> Self {
        Self { module, layout }
    }

    fn memory_access(
        &self,
        func: &Function,
        inst: veloc_mir::Inst,
    ) -> Result<veloc_lir::MemoryAccess> {
        let source = func
            .memory_access(inst)
            .expect("memory lowering requires an access contract");
        let bytes = source
            .bytes(Some(u32::from(self.layout.pointer_size)))
            .ok_or_else(|| {
                Error::translate(format!(
                    "memory access requires a fixed machine representation: {:?}",
                    source.ty
                ))
            })?;
        let kind = if source.stored.is_some() {
            veloc_lir::MemoryKind::Write
        } else {
            veloc_lir::MemoryKind::Read
        };
        let mut access = veloc_lir::MemoryAccess::new(kind, bytes);
        access.alignment = source.flags.alignment();
        access.volatile = source.flags.is_volatile();
        access.may_trap = func
            .stack_access(source, Some(u32::from(self.layout.pointer_size)))
            .is_none();
        Ok(access)
    }

    /// 为函数参数生成 G_ARG 指令
    fn lower_arguments(&self, ctx: &mut TranslationContext, mblock: &mut MachineBlock) {
        use veloc_lir::Writable;
        for (idx, &param_val) in ctx.func.params().iter().enumerate() {
            let vreg = ctx.value_map[param_val];
            let arg_inst = MachineInst::build_arg(Writable(vreg), idx as i64);
            Self::emit_inst(ctx, mblock, arg_inst);
        }
    }

    fn emit_inst(
        ctx: &mut TranslationContext,
        mblock: &mut MachineBlock,
        inst: MachineInst,
    ) -> veloc_lir::InstId {
        let inst_id = ctx.mfunc.alloc_inst(inst);
        mblock.append_inst_id(inst_id);
        inst_id
    }

    /// 将 IR 模块翻译为 MachineModule
    pub fn translate_module(&self) -> Result<MachineModule> {
        for (_, func) in &self.module.functions {
            if func
                .dfg()
                .values()
                .iter()
                .any(|(_, value)| value.ty.is_callable())
                || self
                    .module
                    .get_signature(func.signature)
                    .params()
                    .iter()
                    .chain(self.module.get_signature(func.signature).returns())
                    .any(|ty| ty.is_callable())
            {
                return Err(Error::message(
                    "typed callables and tail calls require callable/environment and tail-call lowering before native code generation",
                ));
            }
        }
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
    ) -> Result<MachineFunction<RawLir>> {
        let mut ctx = TranslationContext {
            func,
            mmodule,
            mfunc: MachineFunction::<RawLir>::new(func.name.clone()),
            value_map: PrimaryMap::with_capacity(func.dfg().values().len()),
            slots: hashbrown::HashMap::new(),
        };

        for &block in func.layout().block_order() {
            for &inst in &func.layout().blocks()[block].insts {
                if let InstView::Alloca { size, align } = func.dfg().inst(inst) {
                    if Some(block) != func.entry_block {
                        return Err(Error::translate(
                            "non-entry alloca requires dynamic stack lowering",
                        ));
                    }
                    if size == 0 || !align.is_power_of_two() || align > 16 {
                        return Err(Error::translate(
                            "native alloca requires positive size and power-of-two alignment at most 16",
                        ));
                    }
                    if ctx
                        .mfunc
                        .stack_frame
                        .local_size
                        .checked_add(size)
                        .and_then(|n| n.checked_add(align - 1))
                        .is_none_or(|n| n > i32::MAX as u32)
                    {
                        return Err(Error::translate(
                            "native alloca frame exceeds target displacement range",
                        ));
                    }
                    let slot = ctx.mfunc.alloc_stack_slot(size, align);
                    ctx.slots.insert(inst, slot);
                }
            }
        }

        // 1. 预分配所有 Value 对应的 VReg
        for (val, data) in func.dfg().values() {
            let vreg = ctx.mfunc.alloc_vreg(data.ty);
            let mapped = ctx.value_map.push(vreg);
            debug_assert_eq!(mapped, val);
        }

        // 2. 翻译基本块和指令
        for (idx, &block_id) in func.layout().block_order().iter().enumerate() {
            let mut mblock = MachineBlock::new(block_id);
            mblock.params = func.layout().blocks()[block_id]
                .params
                .iter()
                .map(|value| ctx.value_map[*value])
                .collect();

            // 如果是入口块，先处理函数参数
            if idx == 0 {
                self.lower_arguments(&mut ctx, &mut mblock);
            }

            for &inst_id in &func.layout().blocks()[block_id].insts {
                let translated = self.translate_instruction(inst_id, &mut ctx, &mut mblock)?;
                let m_inst_id = ctx.mfunc.alloc_inst(translated.inst);
                if let Some(extra) = translated.extra {
                    ctx.mfunc.set_inst_extra(m_inst_id, extra);
                }
                mblock.append_inst_id(m_inst_id);
            }

            ctx.mfunc.blocks.push(mblock);
        }

        ctx.mfunc
            .params
            .extend(func.params().iter().map(|&param| ctx.value_map[param]));

        Ok(ctx.mfunc)
    }

    /// 翻译单条指令
    fn translate_instruction(
        &self,
        inst_id: veloc_mir::Inst,
        ctx: &mut TranslationContext,
        mblock: &mut MachineBlock,
    ) -> Result<TranslatedInst> {
        use smallvec::SmallVec;
        use veloc_lir::Writable;

        let inst_data = &ctx.func.dfg().inst(inst_id);

        // 获取结果寄存器 (Defs)
        let results = ctx.func.dfg().inst_results(inst_id);
        let mut defs = SmallVec::<[MachineOperand; 4]>::new();
        for &res in results {
            let vreg = ctx.value_map[res];
            defs.push(MachineOperand::Def(Writable(vreg)));
        }

        let spec = inst_data.opcode().spec();
        if matches!(inst_data, InstView::Unary { .. } | InstView::Binary { .. }) {
            let args = ctx.func.dfg().operands(inst_id);
            let operand_types: SmallVec<[_; 2]> = args
                .iter()
                .map(|&value| ctx.func.dfg().value_type(value))
                .collect();
            let result_types: SmallVec<[_; 1]> = results
                .iter()
                .map(|&value| ctx.func.dfg().value_type(value))
                .collect();
            inst_data
                .opcode()
                .validate_types(&operand_types, &result_types)
                .map_err(|error| {
                    Error::translate(format!(
                        "invalid types for {} semantic lowering: {error:?}",
                        spec.mnemonic
                    ))
                })?;
            if let Some(opcode) = direct_lowering(inst_data.opcode()) {
                // The shared binding describes the scalar/per-lane operation.
                // Preserve source types here; target legalization still decides
                // which widths and vector shapes the backend can implement.
                defs.extend(
                    args.iter()
                        .map(|arg| MachineOperand::Use(ctx.value_map[*arg])),
                );
                return Ok(MachineInst::build_generic(MachineOpcode::Generic(opcode), defs).into());
            }
        }

        match inst_data {
            InstView::TailCall { .. } => Err(Error::translate(
                "tail calls require tail-call lowering before native code generation",
            )),
            InstView::Alloca { .. } => {
                let slot = ctx.slots[&inst_id];
                Ok(MachineInst::build_stack_addr(defs[0].as_writable().unwrap(), slot).into())
            }
            InstView::Binary { opcode, args } => {
                let src0 = ctx.value_map[args[0]];
                let src1 = ctx.value_map[args[1]];

                let m_opcode = match opcode {
                    Opcode::IDivS => MachineOpcode::Generic(GenericOpcode::G_SDIV),
                    Opcode::IRemS => MachineOpcode::Generic(GenericOpcode::G_SREM),
                    Opcode::IRemU => MachineOpcode::Generic(GenericOpcode::G_UREM),
                    Opcode::IRotl => MachineOpcode::Generic(GenericOpcode::G_ROTL),
                    Opcode::IRotr => MachineOpcode::Generic(GenericOpcode::G_ROTR),
                    Opcode::IDivU => MachineOpcode::Generic(GenericOpcode::G_UDIV),
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

            InstView::Unary { opcode, arg } => {
                let src = ctx.value_map[*arg];

                let m_opcode = match opcode {
                    // The MIR contract spells negation as `0 - arg`. This
                    // explicit target rule retains G_NEG until compositional
                    // selection can match complete semantic programs.
                    Opcode::INeg => MachineOpcode::Generic(GenericOpcode::G_NEG),
                    Opcode::IClz => MachineOpcode::Generic(GenericOpcode::G_CTLZ),
                    Opcode::ICtz => MachineOpcode::Generic(GenericOpcode::G_CTTZ),
                    Opcode::IPopcnt => MachineOpcode::Generic(GenericOpcode::G_CTPOP),
                    Opcode::FAbs => MachineOpcode::Generic(GenericOpcode::G_FABS),
                    Opcode::FSqrt => MachineOpcode::Generic(GenericOpcode::G_FSQRT),
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

            InstView::IntCompare { kind, args } => {
                let src0 = ctx.value_map[args[0]];
                let src1 = ctx.value_map[args[1]];

                Ok(
                    MachineInst::build_icmp(defs[0].as_writable().unwrap(), src0, src1, *kind)
                        .into(),
                )
            }

            InstView::FloatCompare { kind, args } => {
                let src0 = ctx.value_map[args[0]];
                let src1 = ctx.value_map[args[1]];

                Ok(
                    MachineInst::build_fcmp(defs[0].as_writable().unwrap(), src0, src1, *kind)
                        .into(),
                )
            }

            InstView::Load { ptr, offset, .. } => {
                let base = ctx.value_map[*ptr];
                let access = self.memory_access(ctx.func, inst_id)?;
                Ok(MachineInst::build_offset_load(
                    defs[0].as_writable().unwrap(),
                    base,
                    *offset as i64,
                )
                .with_memory(access)
                .into())
            }

            InstView::Store {
                ptr, value, offset, ..
            } => {
                let val = ctx.value_map[*value];
                let base = ctx.value_map[*ptr];
                let access = self.memory_access(ctx.func, inst_id)?;
                Ok(MachineInst::build_offset_store(val, base, *offset as i64)
                    .with_memory(access)
                    .into())
            }

            InstView::Iconst { value: imm } => Ok(MachineInst::build_constant(
                defs[0].as_writable().unwrap(),
                imm.signed(),
            )
            .into()),

            InstView::Bconst { value } => Ok(MachineInst::build_constant(
                defs[0].as_writable().unwrap(),
                i64::from(*value),
            )
            .into()),

            InstView::Fconst { value } => {
                let dst = defs[0].as_writable().unwrap();
                let dst_ty = ctx.mfunc.vreg_data(dst.to_reg()).ty;

                let (bits_ty, bits_imm) = if dst_ty == veloc_mir::Type::F32 {
                    (veloc_mir::Type::I32, value.to_bits() as u32 as i64)
                } else if dst_ty == veloc_mir::Type::F64 {
                    (veloc_mir::Type::I64, value.to_bits() as i64)
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

            InstView::Jump { dest } => {
                let target = dest.block;
                let args = dest
                    .args
                    .iter()
                    .map(|value| ctx.value_map[*value])
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

            InstView::Br {
                condition,
                then_dest,
                else_dest,
            } => {
                let cond_vreg = ctx.value_map[*condition];
                let then_args = then_dest
                    .args
                    .iter()
                    .map(|value| ctx.value_map[*value])
                    .collect::<SmallVec<[Reg; 2]>>();
                let else_args = else_dest
                    .args
                    .iter()
                    .map(|value| ctx.value_map[*value])
                    .collect::<SmallVec<[Reg; 2]>>();

                let inst = MachineInst::build_brcond(cond_vreg, then_dest.block, else_dest.block);
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

            InstView::BrTable { index, .. } => {
                let idx_vreg = ctx.value_map[*index];
                let InstView::BrTable { table, .. } = inst_data else {
                    unreachable!();
                };
                let targets = table
                    .iter()
                    .map(|call| BrTableTarget {
                        block: call.block,
                        args: call
                            .args
                            .iter()
                            .map(|value| ctx.value_map[*value])
                            .collect(),
                    })
                    .collect();

                Ok(TranslatedInst::with_extra(
                    MachineInst::build_brjt(idx_vreg),
                    InstExtra::BrTable(BrTableInfo { targets }),
                ))
            }

            InstView::Return { values } => {
                let ret_values = *values;
                let mut rets = SmallVec::new();
                for &v in ret_values {
                    let vreg = ctx.value_map[v];
                    rets.push(vreg);
                }
                Ok(MachineInst::build_ret(rets).into())
            }

            InstView::Call { func_id, args } => {
                let call_args = *args;
                let callee = self.module.get_function(*func_id);
                let sym_id = ctx.mmodule.symbols_mut().get_or_create_function(
                    self.module.get_function_name(*func_id),
                    callee.linkage,
                );
                let call_inst = MachineInst::build_call(
                    defs.iter().map(|operand| operand.as_writable().unwrap()),
                    sym_id,
                    call_args.iter().map(|value| ctx.value_map[*value]),
                );
                let sig_id = callee.signature;
                let call_info = CallInfo {
                    sig: self.module.get_signature(sig_id).clone(),
                };

                Ok(TranslatedInst::with_extra(
                    call_inst,
                    InstExtra::Call(call_info),
                ))
            }

            InstView::CallIndirect { ptr, args, sig_id } => {
                let call_args = *args;
                let call_inst = MachineInst::build_call_indirect(
                    defs.iter().map(|operand| operand.as_writable().unwrap()),
                    ctx.value_map[*ptr],
                    call_args.iter().map(|value| ctx.value_map[*value]),
                );
                let call_info = CallInfo {
                    sig: self.module.get_signature(*sig_id).clone(),
                };

                Ok(TranslatedInst::with_extra(
                    call_inst,
                    InstExtra::Call(call_info),
                ))
            }

            InstView::Ternary { opcode, args } => {
                let v0 = ctx.value_map[args[0]];
                let v1 = ctx.value_map[args[1]];
                let v2 = ctx.value_map[args[2]];

                match opcode {
                    Opcode::Select => {
                        Ok(
                            MachineInst::build_select(defs[0].as_writable().unwrap(), v0, v1, v2)
                                .into(),
                        )
                    }
                    _ => Err(Error::translate(format!(
                        "Unsupported ternary opcode: {:?}",
                        opcode
                    ))),
                }
            }

            InstView::IntToPtr { arg } => {
                let src = ctx.value_map[*arg];
                Ok(MachineInst::build_unary(
                    MachineOpcode::Generic(GenericOpcode::G_INTTOPTR),
                    defs[0].as_writable().unwrap(),
                    src,
                )
                .into())
            }

            InstView::PtrToInt { arg } => {
                let src = ctx.value_map[*arg];
                Ok(MachineInst::build_unary(
                    MachineOpcode::Generic(GenericOpcode::G_PTRTOINT),
                    defs[0].as_writable().unwrap(),
                    src,
                )
                .into())
            }

            InstView::PtrOffset { ptr, offset } => {
                use veloc_lir::Writable;

                let addr = ctx.value_map[*ptr];
                if *offset == 0 {
                    Ok(MachineInst::build_copy(defs[0].as_writable().unwrap(), addr).into())
                } else {
                    let off_reg = ctx.mfunc.alloc_vreg(veloc_mir::Type::I64);
                    Self::emit_inst(
                        ctx,
                        mblock,
                        MachineInst::build_constant(Writable(off_reg), *offset as i64),
                    );

                    Ok(
                        MachineInst::build_ptr_add(defs[0].as_writable().unwrap(), addr, off_reg)
                            .into(),
                    )
                }
            }

            InstView::PtrIndex { ptr, index, imm_id } => {
                let base_ptr = ctx.value_map[*ptr];
                let idx = ctx.value_map[*index];
                let imm = *imm_id;

                // 1. scale index: idx * scale
                let scaled_idx = if imm.scale != 1 {
                    let scale_reg = ctx.mfunc.alloc_vreg(veloc_mir::Type::I64);
                    let scale_inst =
                        MachineInst::build_constant(Writable(scale_reg), imm.scale as i64);
                    Self::emit_inst(ctx, mblock, scale_inst);

                    let res_reg = ctx.mfunc.alloc_vreg(veloc_mir::Type::I64);
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
                    let off_reg = ctx.mfunc.alloc_vreg(veloc_mir::Type::I64);
                    let off_inst =
                        MachineInst::build_constant(Writable(off_reg), imm.offset as i64);
                    Self::emit_inst(ctx, mblock, off_inst);

                    let res_reg = ctx.mfunc.alloc_vreg(veloc_mir::Type::I64);
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
            InstView::Unreachable => Ok(MachineInst::build_unreachable().into()),

            _ => Err(Error::translate(format!(
                "InstView variant not implemented for translation: {:?}",
                inst_data
            ))),
        }
    }
}
