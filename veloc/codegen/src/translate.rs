//! Middle-level IR (MIR) to low-level IR (LIR) translator.
//!
//! 将 SSA MIR 转换为面向机器的 LIR。
//! 这是 GlobalISel 流程的第一步

use crate::error::{Error, Result};
use alloc::{format, vec::Vec};
use cranelift_entity::PrimaryMap;
use veloc_lir::InstBuild;
use veloc_lir::{
    BrTableInfo, BrTableTarget, BranchCondInfo, BranchInfo, CallInfo, InstExtra, MachineBlock,
    MachineFunction, MachineModule, Reg,
};
use veloc_mir::{Function, InstView, Module, Opcode, TypeInfo, Value};

/// IR 到 LIR 的翻译器
pub struct IRTranslator<'a> {
    module: &'a Module,
    layout: veloc_types::DataLayout,
}

/// 翻译上下文，用于在翻译过程中共享状态
struct TranslationContext<'a> {
    func: &'a Function,
    mmodule: &'a mut MachineModule,
    mfunc: MachineFunction,
    value_map: PrimaryMap<Value, Reg>,
}

impl<'a> IRTranslator<'a> {
    pub fn new(module: &'a Module, layout: veloc_types::DataLayout) -> Self {
        Self { module, layout }
    }

    fn memory_access(
        &self,
        func: &Function,
        inst: veloc_mir::Inst,
    ) -> Result<veloc_lir::MemoryAccess> {
        let source = inst
            .memory_access(func.dfg())
            .expect("memory lowering requires an access contract");
        let bytes = source.bytes(&self.layout).ok_or_else(|| {
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
        access.may_trap = func.stack_access(source, &self.layout).is_none();
        Ok(access)
    }

    /// 为函数参数生成 Arg 指令
    fn lower_arguments(
        &self,
        ctx: &mut TranslationContext,
        mblock: &mut MachineBlock,
        fresh: bool,
    ) -> Vec<Reg> {
        use veloc_lir::Writable;
        let mut args = Vec::new();
        for (idx, &param_val) in ctx.func.params().iter().enumerate() {
            let original = ctx.value_map[param_val];
            let vreg = if fresh {
                ctx.mfunc.alloc_vreg(ctx.mfunc.vreg_data(original).ty)
            } else {
                original
            };
            args.push(vreg);
            ctx.mfunc.params.push(vreg);
            let id = ctx.mfunc.writer().arg(Writable(vreg), idx as i64);
            mblock.append_inst_id(id);
        }
        args
    }

    /// Translate valid MIR. Callers may run the MIR validator before this stage.
    /// Unsupported backend features are still diagnosed during lowering.
    /// 将 IR 模块翻译为 MachineModule
    pub fn translate_module(&self) -> Result<MachineModule> {
        for (_, func) in &self.module.functions {
            if func.body().is_some_and(|body| {
                body.dfg()
                    .values()
                    .iter()
                    .any(|(_, value)| value.ty.is_callable())
            }) || self
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
        match self.layout.pointer_size {
            4 | 8 => {}
            size => {
                return Err(Error::translate(format!(
                    "unsupported pointer size: {size}"
                )));
            }
        }
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
        if func.body().is_none() {
            return Ok(MachineFunction::new(func.name.clone()));
        }
        let mut ctx = TranslationContext {
            func,
            mmodule,
            mfunc: MachineFunction::new(func.name.clone()),
            value_map: PrimaryMap::with_capacity(func.dfg().values().len()),
        };

        // 1. 预分配所有 Value 对应的 VReg
        for (val, data) in func.dfg().values() {
            let vreg = ctx.mfunc.alloc_vreg(data.ty);
            let mapped = ctx.value_map.push(vreg);
            debug_assert_eq!(mapped, val);
        }

        // 2. 翻译基本块和指令
        let entry = func.entry_block();
        let order = entry.into_iter().chain(
            func.layout()
                .block_order()
                .filter(|&block| Some(block) != entry),
        );
        for block_id in order {
            let mut mblock = MachineBlock::new(block_id);
            mblock.params = func.dfg().blocks()[block_id]
                .params
                .iter()
                .map(|value| ctx.value_map[*value])
                .collect();

            // 如果是入口块，先处理函数参数
            if Some(block_id) == entry {
                if func.cfg().blocks()[block_id].preds.is_empty() {
                    // No backedge: Arg/ABI copies are the sole definitions.
                    mblock.params.clear();
                    self.lower_arguments(&mut ctx, &mut mblock, false);
                } else {
                    // An entry loop needs genuine phi parameters. Give incoming
                    // ABI values a separate entry predecessor and fresh identities.
                    let entry = veloc_mir::Block::from_u32(func.dfg().blocks().len() as u32);
                    let mut incoming = MachineBlock::new(entry);
                    let args = self.lower_arguments(&mut ctx, &mut incoming, true);
                    let jump = ctx.mfunc.writer().br(block_id);
                    ctx.mfunc.set_inst_extra(
                        jump,
                        InstExtra::Branch(BranchInfo {
                            args: args.into_iter().collect(),
                        }),
                    );
                    incoming.append_inst_id(jump);
                    ctx.mfunc.blocks.push(incoming);
                }
            }

            for inst_id in func.layout().block_insts(block_id) {
                self.translate_instruction(inst_id, &mut ctx, &mut mblock)?;
            }

            ctx.mfunc.blocks.push(mblock);
        }

        Ok(ctx.mfunc)
    }

    /// 翻译单条指令
    fn translate_instruction(
        &self,
        inst_id: veloc_mir::Inst,
        ctx: &mut TranslationContext,
        mblock: &mut MachineBlock,
    ) -> Result<()> {
        use smallvec::SmallVec;
        use veloc_lir::Writable;

        let inst_data = &ctx.func.dfg().inst(inst_id);

        let results = ctx.func.dfg().inst_results(inst_id);
        let result = || Writable(ctx.value_map[results[0]]);

        let lowered = match inst_data {
            InstView::Unary { .. }
            | InstView::Binary { .. }
            | InstView::Ternary { .. }
            | InstView::IntToPtr { .. }
            | InstView::PtrToInt { .. } => {
                let args = ctx.func.dfg().operands(inst_id);
                let input = |i: usize| ctx.value_map[args[i]];
                let dst = result();
                let writer = ctx.mfunc.writer();
                match inst_data.opcode() {
                    Opcode::INeg => Ok(writer.neg(dst, input(0))),
                    Opcode::IClz => Ok(writer.ctlz(dst, input(0))),
                    Opcode::ICtz => Ok(writer.cttz(dst, input(0))),
                    Opcode::IPopcnt => Ok(writer.ctpop(dst, input(0))),
                    Opcode::FAbs => Ok(writer.fabs(dst, input(0))),
                    Opcode::FSqrt => Ok(writer.fsqrt(dst, input(0))),
                    Opcode::FNeg => Ok(writer.fneg(dst, input(0))),
                    Opcode::IEqz => Ok(writer.ieqz(dst, input(0))),
                    Opcode::Wrap => Ok(writer.trunc(dst, input(0))),
                    Opcode::ExtendU => Ok(writer.zext(dst, input(0))),
                    Opcode::ExtendS => Ok(writer.sext(dst, input(0))),
                    Opcode::FloatDemote => Ok(writer.fptrunc(dst, input(0))),
                    Opcode::FloatPromote => Ok(writer.fpext(dst, input(0))),
                    Opcode::FloatToIntU => Ok(writer.fptoui(dst, input(0))),
                    Opcode::FloatToIntS => Ok(writer.fptosi(dst, input(0))),
                    Opcode::IntToFloatU => Ok(writer.uitofp(dst, input(0))),
                    Opcode::IntToFloatS => Ok(writer.sitofp(dst, input(0))),
                    Opcode::Reinterpret => Ok(writer.bitcast(dst, input(0))),
                    Opcode::IntToPtr => Ok(writer.inttoptr(dst, input(0))),
                    Opcode::PtrToInt => Ok(writer.ptrtoint(dst, input(0))),
                    Opcode::IAdd => Ok(writer.add(dst, input(0), input(1))),
                    Opcode::ISub => Ok(writer.sub(dst, input(0), input(1))),
                    Opcode::IMul => Ok(writer.mul(dst, input(0), input(1))),
                    Opcode::IAnd => Ok(writer.and(dst, input(0), input(1))),
                    Opcode::IOr => Ok(writer.or(dst, input(0), input(1))),
                    Opcode::IXor => Ok(writer.xor(dst, input(0), input(1))),
                    Opcode::IDivS => Ok(writer.sdiv(dst, input(0), input(1))),
                    Opcode::IDivU => Ok(writer.udiv(dst, input(0), input(1))),
                    Opcode::IRemS => Ok(writer.srem(dst, input(0), input(1))),
                    Opcode::IRemU => Ok(writer.urem(dst, input(0), input(1))),
                    Opcode::IRotl => Ok(writer.rotl(dst, input(0), input(1))),
                    Opcode::IRotr => Ok(writer.rotr(dst, input(0), input(1))),
                    Opcode::IShl => Ok(writer.shl(dst, input(0), input(1))),
                    Opcode::IShrS => Ok(writer.ashr(dst, input(0), input(1))),
                    Opcode::IShrU => Ok(writer.lshr(dst, input(0), input(1))),
                    Opcode::FAdd => Ok(writer.fadd(dst, input(0), input(1))),
                    Opcode::FSub => Ok(writer.fsub(dst, input(0), input(1))),
                    Opcode::FMul => Ok(writer.fmul(dst, input(0), input(1))),
                    Opcode::FDiv => Ok(writer.fdiv(dst, input(0), input(1))),
                    Opcode::Select => Ok(writer.select(dst, input(0), input(1), input(2))),
                    _ => Err(Error::translate(format!(
                        "unsupported arithmetic opcode: {:?}",
                        inst_data.opcode()
                    ))),
                }
            }
            InstView::TailCall { .. } => Err(Error::translate(
                "tail calls require tail-call lowering before native code generation",
            )),
            InstView::Alloca { size, align } => {
                if Some(mblock.id) != ctx.func.entry_block() {
                    return Err(Error::translate(
                        "non-entry alloca requires dynamic stack lowering",
                    ));
                }
                if *size == 0 || !align.is_power_of_two() || *align > 16 {
                    return Err(Error::translate(
                        "native alloca requires positive size and power-of-two alignment at most 16",
                    ));
                }
                if ctx
                    .mfunc
                    .stack_frame
                    .local_size
                    .checked_add(*size)
                    .and_then(|n| n.checked_add(*align - 1))
                    .is_none_or(|n| n > i32::MAX as u32)
                {
                    return Err(Error::translate(
                        "native alloca frame exceeds target displacement range",
                    ));
                }
                let slot = ctx.mfunc.alloc_stack_slot(*size, *align);
                Ok(ctx.mfunc.writer().stack_addr(result(), slot))
            }
            InstView::IntCompare { kind, args } => {
                let src0 = ctx.value_map[args[0]];
                let src1 = ctx.value_map[args[1]];

                Ok(ctx.mfunc.writer().icmp(result(), src0, src1, *kind))
            }

            InstView::FloatCompare { kind, args } => {
                let src0 = ctx.value_map[args[0]];
                let src1 = ctx.value_map[args[1]];

                Ok(ctx.mfunc.writer().fcmp(result(), src0, src1, *kind))
            }

            InstView::Load { ptr, offset, .. } => {
                let base = ctx.value_map[*ptr];
                let access = self.memory_access(ctx.func, inst_id)?;
                Ok(ctx.mfunc.writer().with_memory(access).offset_load(
                    result(),
                    base,
                    *offset as i64,
                ))
            }

            InstView::Store {
                ptr, value, offset, ..
            } => {
                let val = ctx.value_map[*value];
                let base = ctx.value_map[*ptr];
                let access = self.memory_access(ctx.func, inst_id)?;
                Ok(ctx
                    .mfunc
                    .writer()
                    .with_memory(access)
                    .offset_store(val, base, *offset as i64))
            }

            InstView::Iconst { value: imm } => {
                Ok(ctx.mfunc.writer().constant(result(), imm.signed()))
            }

            InstView::Bconst { value } => {
                Ok(ctx.mfunc.writer().constant(result(), i64::from(*value)))
            }

            InstView::Fconst { value } => {
                let dst = result();
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
                let bits_inst = ctx.mfunc.writer().constant(Writable(bits_reg), bits_imm);
                mblock.append_inst_id(bits_inst);

                Ok(ctx.mfunc.writer().bitcast(dst, bits_reg))
            }

            InstView::Jump { dest } => {
                let target = dest.block;
                let args = dest
                    .args
                    .iter()
                    .map(|value| ctx.value_map[*value])
                    .collect::<SmallVec<[Reg; 2]>>();
                let inst = ctx.mfunc.writer().br(target);
                if args.is_empty() {
                    Ok(inst)
                } else {
                    Ok({
                        let id = inst;
                        ctx.mfunc
                            .set_inst_extra(id, InstExtra::Branch(BranchInfo { args }));
                        id
                    })
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

                let inst = ctx
                    .mfunc
                    .writer()
                    .brcond(cond_vreg, then_dest.block, else_dest.block);
                if then_args.is_empty() && else_args.is_empty() {
                    Ok(inst)
                } else {
                    Ok({
                        let id = inst;
                        ctx.mfunc.set_inst_extra(
                            id,
                            InstExtra::BranchCond(BranchCondInfo {
                                then_args,
                                else_args,
                            }),
                        );
                        id
                    })
                }
            }

            InstView::BrTable { index, table } => {
                let idx_vreg = ctx.value_map[*index];
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

                Ok({
                    let id = ctx.mfunc.writer().brjt(idx_vreg);
                    ctx.mfunc
                        .set_inst_extra(id, InstExtra::BrTable(BrTableInfo { targets }));
                    id
                })
            }

            InstView::Return { values } => {
                let ret_values = *values;
                let mut rets = SmallVec::<[Reg; 2]>::new();
                for &v in ret_values {
                    let vreg = ctx.value_map[v];
                    rets.push(vreg);
                }
                Ok(ctx.mfunc.writer().ret(&rets))
            }

            InstView::Call { func_id, args } => {
                let call_args = *args;
                let callee = self.module.get_function(*func_id);
                let sym_id = ctx.mmodule.symbols_mut().get_or_create_function(
                    self.module.get_function_name(*func_id),
                    callee.linkage,
                );
                let call_inst = ctx.mfunc.writer().call(
                    &results
                        .iter()
                        .map(|value| ctx.value_map[*value])
                        .collect::<SmallVec<[Reg; 2]>>(),
                    sym_id,
                    &call_args
                        .iter()
                        .map(|value| ctx.value_map[*value])
                        .collect::<SmallVec<[Reg; 4]>>(),
                );
                let sig_id = callee.signature;
                let call_info = CallInfo {
                    sig: self.module.get_signature(sig_id).clone(),
                };

                Ok({
                    let id = call_inst;
                    ctx.mfunc.set_inst_extra(id, InstExtra::Call(call_info));
                    id
                })
            }

            InstView::CallIndirect { ptr, args, sig_id } => {
                let call_args = *args;
                let call_inst = ctx.mfunc.writer().callind(
                    &results
                        .iter()
                        .map(|value| ctx.value_map[*value])
                        .collect::<SmallVec<[Reg; 2]>>(),
                    ctx.value_map[*ptr],
                    &call_args
                        .iter()
                        .map(|value| ctx.value_map[*value])
                        .collect::<SmallVec<[Reg; 4]>>(),
                );
                let call_info = CallInfo {
                    sig: self.module.get_signature(*sig_id).clone(),
                };

                Ok({
                    let id = call_inst;
                    ctx.mfunc.set_inst_extra(id, InstExtra::Call(call_info));
                    id
                })
            }

            InstView::PtrOffset { ptr, offset } => {
                use veloc_lir::Writable;

                let addr = ctx.value_map[*ptr];
                let addr_ty = if self.layout.pointer_size == 4 {
                    veloc_mir::Type::I32
                } else {
                    veloc_mir::Type::I64
                };
                if *offset == 0 {
                    Ok(ctx.mfunc.writer().copy(result(), addr))
                } else {
                    let off_reg = ctx.mfunc.alloc_vreg(addr_ty);
                    let id = ctx
                        .mfunc
                        .writer()
                        .constant(Writable(off_reg), *offset as i64);
                    mblock.append_inst_id(id);

                    Ok(ctx.mfunc.writer().ptr_add(result(), addr, off_reg))
                }
            }

            InstView::PtrIndex { ptr, index, imm_id } => {
                let base_ptr = ctx.value_map[*ptr];
                let idx = ctx.value_map[*index];
                let addr_ty = if self.layout.pointer_size == 4 {
                    veloc_mir::Type::I32
                } else {
                    veloc_mir::Type::I64
                };
                let index_ty = ctx.func.dfg().value_type(*index);
                // Indices are unsigned bit patterns; signed displacements use
                // the offset field (or an explicit MIR sign extension).
                let idx = if index_ty == addr_ty {
                    idx
                } else {
                    let normalized = ctx.mfunc.alloc_vreg(addr_ty);
                    let from = index_ty.element_bits().expect("integer index width");
                    let to = u32::from(self.layout.pointer_size) * 8;
                    let id = if from < to {
                        ctx.mfunc.writer().zext(Writable(normalized), idx)
                    } else {
                        ctx.mfunc.writer().trunc(Writable(normalized), idx)
                    };
                    mblock.append_inst_id(id);
                    normalized
                };
                let imm = *imm_id;

                // 1. scale index: idx * scale
                let scaled_idx = if imm.scale != 1 {
                    let scale_reg = ctx.mfunc.alloc_vreg(addr_ty);
                    let scale_inst = ctx
                        .mfunc
                        .writer()
                        .constant(Writable(scale_reg), imm.scale as i64);
                    mblock.append_inst_id(scale_inst);

                    let res_reg = ctx.mfunc.alloc_vreg(addr_ty);
                    let mul_inst = ctx.mfunc.writer().mul(Writable(res_reg), idx, scale_reg);
                    mblock.append_inst_id(mul_inst);
                    res_reg
                } else {
                    idx
                };

                // 2. add offset if any: base_idx = (idx * scale) + offset
                let base_idx = if imm.offset != 0 {
                    let off_reg = ctx.mfunc.alloc_vreg(addr_ty);
                    let off_inst = ctx
                        .mfunc
                        .writer()
                        .constant(Writable(off_reg), imm.offset as i64);
                    mblock.append_inst_id(off_inst);

                    let res_reg = ctx.mfunc.alloc_vreg(addr_ty);
                    let add_inst = ctx
                        .mfunc
                        .writer()
                        .add(Writable(res_reg), scaled_idx, off_reg);
                    mblock.append_inst_id(add_inst);
                    res_reg
                } else {
                    scaled_idx
                };

                // 3. ptr_add: ptr + base_idx
                Ok(ctx.mfunc.writer().ptr_add(result(), base_ptr, base_idx))
            }
            InstView::Unreachable => Ok(ctx.mfunc.writer().unreachable()),

            _ => Err(Error::translate(format!(
                "InstView variant not implemented for translation: {:?}",
                inst_data
            ))),
        }?;
        mblock.append_inst_id(lowered);
        Ok(())
    }
}
