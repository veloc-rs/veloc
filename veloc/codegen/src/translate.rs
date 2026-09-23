//! Middle-level IR (MIR) to low-level IR (LIR) translator.
//!
//! Preserves semantic operations and SSA; ABI and target legalization run later.

use crate::error::{Error, Result};
use alloc::{format, vec::Vec};
use cranelift_entity::PrimaryMap;
use smallvec::SmallVec;
use veloc_lir::InstBuild;
use veloc_lir::{BlockId, CallInfo, MachineFunction, MachineModule, Reg};
use veloc_mir::{InstView, Module, Opcode, TypeInfo, Value};

/// IR 到 LIR 的翻译器
pub struct IRTranslator<'a> {
    module: &'a Module,
    layout: veloc_types::DataLayout,
}

/// Function-local mappings and output. Target ABI decisions stay outside translation.
struct FuncTranslator<'a> {
    module: &'a Module,
    layout: veloc_types::DataLayout,
    func: &'a veloc_mir::FuncBody,
    mmodule: &'a mut MachineModule,
    mfunc: MachineFunction,
    value_map: PrimaryMap<Value, Reg>,
    block_map: cranelift_entity::SecondaryMap<veloc_mir::Block, Option<BlockId>>,
}

impl<'a> IRTranslator<'a> {
    pub fn new(module: &'a Module, layout: veloc_types::DataLayout) -> Self {
        Self { module, layout }
    }

    /// Translate valid MIR. Callers may run the MIR validator before this stage.
    /// Unsupported backend features are still diagnosed during lowering.
    /// 将 IR 模块翻译为 MachineModule
    pub fn translate_module(&self) -> Result<MachineModule> {
        for (_, func) in self.module.functions() {
            if func.body.is_some_and(|body| {
                body.dfg()
                    .values()
                    .iter()
                    .any(|(_, value)| value.ty.is_callable())
            }) || self.module.signatures()[func.decl.signature]
                .params()
                .iter()
                .chain(self.module.signatures()[func.decl.signature].returns())
                .any(|ty| ty.is_callable())
            {
                return Err(Error::message(
                    "typed callables and tail calls require callable/environment and tail-call lowering before native code generation",
                ));
            }
        }
        match self.layout.pointer_size {
            4 | 8 => {}
            size => {
                return Err(Error::translate(format!(
                    "unsupported pointer size: {size}"
                )));
            }
        }
        let mut mmodule = MachineModule::new(alloc::string::String::from("default"));

        for (_, func) in self.module.functions().filter(|(_, f)| f.body.is_some()) {
            let mfunc = FuncTranslator::new(
                self.module,
                self.layout,
                func.decl,
                func.body.unwrap(),
                &mut mmodule,
            )
            .translate()?;
            mmodule.add_function(mfunc);
        }

        Ok(mmodule)
    }
}

impl<'a> FuncTranslator<'a> {
    fn new(
        module: &'a Module,
        layout: veloc_types::DataLayout,
        decl: &veloc_mir::FuncDecl,
        func: &'a veloc_mir::FuncBody,
        mmodule: &'a mut MachineModule,
    ) -> Self {
        let block_count = func.layout().block_order().count();
        let inst_count = func.dfg().instructions().len();
        // Selection appends stable target instruction IDs before invalidating
        // their generic roots, so the final ID space is normally near 2x MIR.
        let lir_inst_capacity = inst_count.saturating_mul(2);
        let value_count = func.dfg().values().len();
        Self {
            module,
            layout,
            func,
            mmodule,
            mfunc: MachineFunction::with_capacity(
                decl.name.clone(),
                block_count + 1,
                lir_inst_capacity,
                value_count + func.params().len(),
            ),
            value_map: PrimaryMap::with_capacity(value_count),
            block_map: cranelift_entity::SecondaryMap::with_capacity(block_count),
        }
    }

    fn translate(mut self) -> Result<MachineFunction> {
        let func = self.func;
        // Allocate value identities before translating forward references.
        for (val, data) in func.dfg().values() {
            let vreg = self.mfunc.editor().alloc_vreg(data.ty);
            let mapped = self.value_map.push(vreg);
            debug_assert_eq!(mapped, val);
        }

        // Allocate LIR identities independently; all edges use this explicit map.
        let entry = func.entry_block();
        let order = || {
            core::iter::once(entry)
                .chain(func.layout().block_order().filter(|&block| block != entry))
        };
        let incoming =
            (!func.cfg().blocks()[entry].preds.is_empty()).then_some(self.mfunc.entry_block());
        for block in order() {
            self.block_map[block] = Some(if block == entry && incoming.is_none() {
                self.mfunc.entry_block()
            } else {
                self.mfunc.editor().create_block()
            });
        }
        // The caller enters once; a backedge must enter the MIR block instead.
        if let Some(incoming) = incoming {
            let mut args = Vec::with_capacity(func.params().len());
            for &value in func.params() {
                let reg = self.mfunc.editor().alloc_vreg(func.dfg().value_type(value));
                self.mfunc.editor().append_param(reg);
                args.push(reg);
            }
            let edge = self
                .mfunc
                .editor()
                .create_edge(self.block_map[entry].unwrap(), &args);
            self.mfunc.editor().at_end(incoming).br(edge);
        } else {
            for &value in func.params() {
                self.mfunc.editor().append_param(self.value_map[value]);
            }
        }
        for block_id in order() {
            let mblock = self.block_map[block_id].unwrap();
            if block_id != entry || incoming.is_some() {
                for &value in func.dfg().block_params(block_id) {
                    self.mfunc
                        .editor()
                        .append_block_param(mblock, self.value_map[value]);
                }
            }
            for inst in func.layout().block_insts(block_id) {
                self.translate_instruction(inst, mblock)?;
            }
        }

        Ok(self.mfunc)
    }

    fn memory_access(&self, inst: veloc_mir::Inst) -> Result<veloc_lir::MemoryAccess> {
        let source = inst
            .memory_access(self.func.dfg())
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
        access.may_trap = self.func.stack_access(source, &self.layout).is_none();
        Ok(access)
    }

    fn lower_edge(&mut self, edge: veloc_mir::Successor<'_>) -> veloc_lir::EdgeId {
        let args = self.values(edge.args);
        self.mfunc
            .editor()
            .create_edge(self.block_map[edge.block].unwrap(), &args)
    }

    fn values(&self, values: &[Value]) -> SmallVec<[Reg; 4]> {
        values.iter().map(|&value| self.value_map[value]).collect()
    }

    fn call_info(&self, sig: veloc_mir::SigId) -> CallInfo {
        CallInfo {
            clobbers: Default::default(),
            frame: None,
            stack_args: Default::default(),
            sig: self.module.signatures()[sig].clone(),
        }
    }

    /// Translate one operation, emitting complete LIR instructions in order.
    fn translate_instruction(&mut self, inst_id: veloc_mir::Inst, mblock: BlockId) -> Result<()> {
        let inst_data = &self.func.dfg().inst(inst_id);

        let results = self.func.dfg().inst_results(inst_id);
        let result = || self.value_map[results[0]];

        match inst_data {
            InstView::Unary { .. }
            | InstView::Binary { .. }
            | InstView::Ternary { .. }
            | InstView::IntToPtr { .. }
            | InstView::PtrToInt { .. } => {
                let args = self.func.dfg().operands(inst_id);
                let input = |i: usize| self.value_map[args[i]];
                let dst = result();
                let mut edit = self.mfunc.editor();
                let mut insert = edit.at_end(mblock);
                match inst_data.opcode() {
                    Opcode::INeg => Ok(insert.neg(dst, input(0))),
                    Opcode::IClz => Ok(insert.ctlz(dst, input(0))),
                    Opcode::ICtz => Ok(insert.cttz(dst, input(0))),
                    Opcode::IPopcnt => Ok(insert.ctpop(dst, input(0))),
                    Opcode::FAbs => Ok(insert.fabs(dst, input(0))),
                    Opcode::FSqrt => Ok(insert.fsqrt(dst, input(0))),
                    Opcode::FNeg => Ok(insert.fneg(dst, input(0))),
                    Opcode::IEqz => Ok(insert.ieqz(dst, input(0))),
                    Opcode::Wrap => Ok(insert.trunc(dst, input(0))),
                    Opcode::ExtendU => Ok(insert.zext(dst, input(0))),
                    Opcode::ExtendS => Ok(insert.sext(dst, input(0))),
                    Opcode::FloatDemote => Ok(insert.fptrunc(dst, input(0))),
                    Opcode::FloatPromote => Ok(insert.fpext(dst, input(0))),
                    Opcode::FloatToIntU => Ok(insert.fptoui(dst, input(0))),
                    Opcode::FloatToIntS => Ok(insert.fptosi(dst, input(0))),
                    Opcode::IntToFloatU => Ok(insert.uitofp(dst, input(0))),
                    Opcode::IntToFloatS => Ok(insert.sitofp(dst, input(0))),
                    Opcode::Reinterpret => Ok(insert.bitcast(dst, input(0))),
                    Opcode::IntToPtr => Ok(insert.inttoptr(dst, input(0))),
                    Opcode::PtrToInt => Ok(insert.ptrtoint(dst, input(0))),
                    Opcode::IAdd => Ok(insert.add(dst, input(0), input(1))),
                    Opcode::ISub => Ok(insert.sub(dst, input(0), input(1))),
                    Opcode::IMul => Ok(insert.mul(dst, input(0), input(1))),
                    Opcode::IAnd => Ok(insert.and(dst, input(0), input(1))),
                    Opcode::IOr => Ok(insert.or(dst, input(0), input(1))),
                    Opcode::IXor => Ok(insert.xor(dst, input(0), input(1))),
                    Opcode::IDivS => Ok(insert.sdiv(dst, input(0), input(1))),
                    Opcode::IDivU => Ok(insert.udiv(dst, input(0), input(1))),
                    Opcode::IRemS => Ok(insert.srem(dst, input(0), input(1))),
                    Opcode::IRemU => Ok(insert.urem(dst, input(0), input(1))),
                    Opcode::IRotl => Ok(insert.rotl(dst, input(0), input(1))),
                    Opcode::IRotr => Ok(insert.rotr(dst, input(0), input(1))),
                    Opcode::IShl => Ok(insert.shl(dst, input(0), input(1))),
                    Opcode::IShrS => Ok(insert.ashr(dst, input(0), input(1))),
                    Opcode::IShrU => Ok(insert.lshr(dst, input(0), input(1))),
                    Opcode::FAdd => Ok(insert.fadd(dst, input(0), input(1))),
                    Opcode::FSub => Ok(insert.fsub(dst, input(0), input(1))),
                    Opcode::FMul => Ok(insert.fmul(dst, input(0), input(1))),
                    Opcode::FDiv => Ok(insert.fdiv(dst, input(0), input(1))),
                    Opcode::Select => Ok(insert.select(dst, input(0), input(1), input(2))),
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
                if mblock != self.block_map[self.func.entry_block()].unwrap() {
                    return Err(Error::translate(
                        "non-entry alloca requires dynamic stack lowering",
                    ));
                }
                let slot = self.mfunc.editor().alloc_stack_object(
                    veloc_lir::StackObject::Local,
                    *size,
                    *align,
                );
                Ok(self
                    .mfunc
                    .editor()
                    .at_end(mblock)
                    .stack_addr(result(), slot))
            }
            InstView::IntCompare { kind, args } => {
                let src0 = self.value_map[args[0]];
                let src1 = self.value_map[args[1]];

                Ok(self
                    .mfunc
                    .editor()
                    .at_end(mblock)
                    .icmp(result(), src0, src1, *kind))
            }

            InstView::FloatCompare { kind, args } => {
                let src0 = self.value_map[args[0]];
                let src1 = self.value_map[args[1]];

                Ok(self
                    .mfunc
                    .editor()
                    .at_end(mblock)
                    .fcmp(result(), src0, src1, *kind))
            }

            InstView::Load { ptr, offset, .. } => {
                let base = self.value_map[*ptr];
                let access = self.memory_access(inst_id)?;
                Ok(self.mfunc.editor().at_end(mblock).with_memory(access).load(
                    result(),
                    base,
                    *offset as i64,
                ))
            }

            InstView::Store {
                ptr, value, offset, ..
            } => {
                let val = self.value_map[*value];
                let base = self.value_map[*ptr];
                let access = self.memory_access(inst_id)?;
                Ok(self
                    .mfunc
                    .editor()
                    .at_end(mblock)
                    .with_memory(access)
                    .store(val, base, *offset as i64))
            }

            InstView::Iconst { value: imm } => Ok(self
                .mfunc
                .editor()
                .at_end(mblock)
                .constant(result(), imm.signed())),

            InstView::Bconst { value } => Ok(self
                .mfunc
                .editor()
                .at_end(mblock)
                .constant(result(), i64::from(*value))),

            InstView::Fconst { value } => {
                let dst = result();
                let dst_ty = self.mfunc.vreg_data(dst).ty;

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

                let bits_reg = self.mfunc.editor().alloc_vreg(bits_ty);
                self.mfunc
                    .editor()
                    .at_end(mblock)
                    .constant(bits_reg, bits_imm);

                Ok(self.mfunc.editor().at_end(mblock).bitcast(dst, bits_reg))
            }

            InstView::Jump { dest } => {
                let edge = self.lower_edge(*dest);
                Ok(self.mfunc.editor().at_end(mblock).br(edge))
            }

            InstView::Br {
                condition,
                then_dest,
                else_dest,
            } => {
                let cond_vreg = self.value_map[*condition];
                let yes = self.lower_edge(*then_dest);
                let no = self.lower_edge(*else_dest);
                Ok(self
                    .mfunc
                    .editor()
                    .at_end(mblock)
                    .brcond(cond_vreg, yes, no))
            }

            InstView::BrTable { index, table } => {
                let idx_vreg = self.value_map[*index];
                let edges: SmallVec<[_; 4]> =
                    table.iter().map(|edge| self.lower_edge(edge)).collect();
                Ok(self.mfunc.editor().at_end(mblock).brjt(idx_vreg, &edges))
            }

            InstView::Return { values } => {
                let rets = self.values(values);
                Ok(self.mfunc.editor().at_end(mblock).ret(&rets))
            }

            InstView::Call { func_id, args } => {
                let callee = &self.module.decls[*func_id];
                let symbol = self
                    .mmodule
                    .symbols_mut()
                    .get_or_create_function(&callee.name, callee.linkage);
                let info = self.call_info(callee.signature);
                let results = self.values(results);
                let args = self.values(args);
                Ok(self
                    .mfunc
                    .editor()
                    .at_end(mblock)
                    .call(&results, symbol, &args, info))
            }

            InstView::CallIndirect { ptr, args, sig_id } => {
                let info = self.call_info(*sig_id);
                let results = self.values(results);
                let args = self.values(args);
                let ptr = self.value_map[*ptr];
                Ok(self
                    .mfunc
                    .editor()
                    .at_end(mblock)
                    .callind(&results, ptr, &args, info))
            }

            InstView::PtrOffset { ptr, offset } => {
                let addr = self.value_map[*ptr];
                let addr_ty = if self.layout.pointer_size == 4 {
                    veloc_mir::Type::I32
                } else {
                    veloc_mir::Type::I64
                };
                if *offset == 0 {
                    Ok(self.mfunc.editor().at_end(mblock).copy(result(), addr))
                } else {
                    let off_reg = self.mfunc.editor().alloc_vreg(addr_ty);
                    self.mfunc
                        .editor()
                        .at_end(mblock)
                        .constant(off_reg, *offset as i64);

                    Ok(self
                        .mfunc
                        .editor()
                        .at_end(mblock)
                        .ptr_add(result(), addr, off_reg))
                }
            }

            InstView::PtrIndex { ptr, index, imm_id } => {
                let base_ptr = self.value_map[*ptr];
                let idx = self.value_map[*index];
                let addr_ty = if self.layout.pointer_size == 4 {
                    veloc_mir::Type::I32
                } else {
                    veloc_mir::Type::I64
                };
                let index_ty = self.func.dfg().value_type(*index);
                // Indices are unsigned bit patterns; signed displacements use
                // the offset field (or an explicit MIR sign extension).
                let idx = if index_ty == addr_ty {
                    idx
                } else {
                    let normalized = self.mfunc.editor().alloc_vreg(addr_ty);
                    let from = index_ty.element_bits().expect("integer index width");
                    let to = u32::from(self.layout.pointer_size) * 8;
                    if from < to {
                        self.mfunc.editor().at_end(mblock).zext(normalized, idx)
                    } else {
                        self.mfunc.editor().at_end(mblock).trunc(normalized, idx)
                    };

                    normalized
                };
                let imm = *imm_id;

                // 1. scale index: idx * scale
                let scaled_idx = if imm.scale != 1 {
                    let scale_reg = self.mfunc.editor().alloc_vreg(addr_ty);
                    self.mfunc
                        .editor()
                        .at_end(mblock)
                        .constant(scale_reg, imm.scale as i64);

                    let res_reg = self.mfunc.editor().alloc_vreg(addr_ty);
                    self.mfunc
                        .editor()
                        .at_end(mblock)
                        .mul(res_reg, idx, scale_reg);

                    res_reg
                } else {
                    idx
                };

                // 2. add offset if any: base_idx = (idx * scale) + offset
                let base_idx = if imm.offset != 0 {
                    let off_reg = self.mfunc.editor().alloc_vreg(addr_ty);
                    self.mfunc
                        .editor()
                        .at_end(mblock)
                        .constant(off_reg, imm.offset as i64);

                    let res_reg = self.mfunc.editor().alloc_vreg(addr_ty);
                    self.mfunc
                        .editor()
                        .at_end(mblock)
                        .add(res_reg, scaled_idx, off_reg);

                    res_reg
                } else {
                    scaled_idx
                };

                // 3. ptr_add: ptr + base_idx
                Ok(self
                    .mfunc
                    .editor()
                    .at_end(mblock)
                    .ptr_add(result(), base_ptr, base_idx))
            }
            InstView::Unreachable => Ok(self.mfunc.editor().at_end(mblock).trap()),

            _ => Err(Error::translate(format!(
                "InstView variant not implemented for translation: {:?}",
                inst_data
            ))),
        }?;
        Ok(())
    }
}
