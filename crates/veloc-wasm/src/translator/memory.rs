use super::WasmTranslator;
use crate::vm::{TrapCode, VMMemory};
use veloc::mir::{IntCC, MemFlags, Type as VelocType, Value};
use wasmparser::{BinaryReaderError, MemArg, Operator};

impl<'a> WasmTranslator<'a> {
    pub(super) fn translate_memory(&mut self, op: Operator) -> Result<(), BinaryReaderError> {
        match op {
            Operator::I32Load { memarg } => self.translate_load(VelocType::I32, memarg),
            Operator::I64Load { memarg } => self.translate_load(VelocType::I64, memarg),
            Operator::F32Load { memarg } => self.translate_load(VelocType::F32, memarg),
            Operator::F64Load { memarg } => self.translate_load(VelocType::F64, memarg),
            Operator::I32Load8S { memarg } => {
                self.translate_load(VelocType::I8, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extends(v, VelocType::I32);
                self.stack.push(extended);
            }
            Operator::I32Load8U { memarg } => {
                self.translate_load(VelocType::I8, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extendu(v, VelocType::I32);
                self.stack.push(extended);
            }
            Operator::I32Load16S { memarg } => {
                self.translate_load(VelocType::I16, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extends(v, VelocType::I32);
                self.stack.push(extended);
            }
            Operator::I32Load16U { memarg } => {
                self.translate_load(VelocType::I16, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extendu(v, VelocType::I32);
                self.stack.push(extended);
            }
            Operator::I64Load8S { memarg } => {
                self.translate_load(VelocType::I8, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extends(v, VelocType::I64);
                self.stack.push(extended);
            }
            Operator::I64Load8U { memarg } => {
                self.translate_load(VelocType::I8, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extendu(v, VelocType::I64);
                self.stack.push(extended);
            }
            Operator::I64Load16S { memarg } => {
                self.translate_load(VelocType::I16, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extends(v, VelocType::I64);
                self.stack.push(extended);
            }
            Operator::I64Load16U { memarg } => {
                self.translate_load(VelocType::I16, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extendu(v, VelocType::I64);
                self.stack.push(extended);
            }
            Operator::I64Load32S { memarg } => {
                self.translate_load(VelocType::I32, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extends(v, VelocType::I64);
                self.stack.push(extended);
            }
            Operator::I64Load32U { memarg } => {
                self.translate_load(VelocType::I32, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extendu(v, VelocType::I64);
                self.stack.push(extended);
            }
            Operator::I32Store { memarg } => self.translate_store(VelocType::I32, memarg),
            Operator::I64Store { memarg } => self.translate_store(VelocType::I64, memarg),
            Operator::F32Store { memarg } => self.translate_store(VelocType::F32, memarg),
            Operator::F64Store { memarg } => self.translate_store(VelocType::F64, memarg),
            Operator::I32Store8 { memarg } => {
                let val = self.pop_i32();
                let truncated = self.builder.ins().wrap(val, VelocType::I8);
                self.stack.push(truncated);
                self.translate_store(VelocType::I8, memarg);
            }
            Operator::I32Store16 { memarg } => {
                let val = self.pop_i32();
                let truncated = self.builder.ins().wrap(val, VelocType::I16);
                self.stack.push(truncated);
                self.translate_store(VelocType::I16, memarg);
            }
            Operator::I64Store8 { memarg } => {
                let val = self.pop_i64();
                let truncated = self.builder.ins().wrap(val, VelocType::I8);
                self.stack.push(truncated);
                self.translate_store(VelocType::I8, memarg);
            }
            Operator::I64Store16 { memarg } => {
                let val = self.pop_i64();
                let truncated = self.builder.ins().wrap(val, VelocType::I16);
                self.stack.push(truncated);
                self.translate_store(VelocType::I16, memarg);
            }
            Operator::I64Store32 { memarg } => {
                let val = self.pop_i64();
                let truncated = self.builder.ins().wrap(val, VelocType::I32);
                self.stack.push(truncated);
                self.translate_store(VelocType::I32, memarg);
            }
            Operator::MemorySize { mem, .. } => {
                let (_, len_var) = self.memory_vars[mem as usize];
                let size_bytes = self.builder.use_var(len_var);
                let page_size = self.builder.ins().iconst(65536, VelocType::I64);
                let size_pages = self.builder.ins().idiv_u(size_bytes, page_size);
                let size_i32 = self.builder.ins().wrap(size_pages, VelocType::I32);
                self.stack.push(size_i32);
            }
            Operator::MemoryGrow { mem, .. } => {
                let delta = self.pop();
                let vmctx = self.vmctx.expect("vmctx not set");
                let mem_idx = self.builder.ins().iconst(mem as u64, VelocType::I32);
                let call_inst = self
                    .builder
                    .ins()
                    .call(self.runtime.memory_grow, &[vmctx, mem_idx, delta]);
                let res_val = self.builder.func().dfg.inst_results(call_inst)[0];
                self.stack.push(res_val);
                self.reload_memory(mem);
            }
            Operator::MemoryInit { data_index, mem } => {
                let len = self.pop_i32();
                let src = self.pop_i32();
                let dst = self.pop_i32();
                let vmctx = self.vmctx.expect("vmctx not set");
                let mem_idx = self.builder.ins().iconst(mem as u64, VelocType::I32);
                let data_idx = self.builder.ins().iconst(data_index as u64, VelocType::I32);
                self.builder.ins().call(
                    self.runtime.memory_init,
                    &[vmctx, mem_idx, data_idx, dst, src, len],
                );
            }
            Operator::DataDrop { data_index } => {
                let vmctx = self.vmctx.expect("vmctx not set");
                let data_idx = self.builder.ins().iconst(data_index as u64, VelocType::I32);
                self.builder
                    .ins()
                    .call(self.runtime.data_drop, &[vmctx, data_idx]);
            }
            Operator::MemoryCopy { dst_mem, src_mem } => {
                let len = self.pop_i32();
                let src = self.pop_i32();
                let dst = self.pop_i32();
                let vmctx = self.vmctx.expect("vmctx not set");
                let dst_mem_val = self.builder.ins().iconst(dst_mem as u64, VelocType::I32);
                let src_mem_val = self.builder.ins().iconst(src_mem as u64, VelocType::I32);
                self.builder.ins().call(
                    self.runtime.memory_copy,
                    &[vmctx, dst_mem_val, src_mem_val, dst, src, len],
                );
            }
            Operator::MemoryFill { mem } => {
                let len = self.pop_i32();
                let val = self.pop_i32();
                let dst = self.pop_i32();
                let vmctx = self.vmctx.expect("vmctx not set");
                let mem_idx = self.builder.ins().iconst(mem as u64, VelocType::I32);
                self.builder
                    .ins()
                    .call(self.runtime.memory_fill, &[vmctx, mem_idx, dst, val, len]);
            }
            _ => unreachable!("Non-memory operator in translate_memory"),
        }
        Ok(())
    }

    pub(super) fn reload_memory(&mut self, index: u32) {
        let vmctx = self.vmctx.expect("vmctx not set");
        let (is_imported, offset) = self.offsets.memory_access_info(index);

        let (base, length) = if is_imported {
            // 导入的 memory：先加载指针，再通过指针访问
            let alignment = if offset % 16 == 0 { 16 } else { 8 };
            let def_ptr = self.builder.ins().load(
                vmctx,
                offset,
                MemFlags::new().with_alignment(alignment),
                VelocType::PTR,
            );
            let base = self.builder.ins().load(
                def_ptr,
                VMMemory::base_offset(),
                MemFlags::new().with_alignment(8),
                VelocType::PTR,
            );
            let length = self.builder.ins().load(
                def_ptr,
                VMMemory::current_length_offset(),
                MemFlags::new().with_alignment(8),
                VelocType::I64,
            );
            (base, length)
        } else {
            // 本地 memory：直接通过 VMContext 偏移访问
            let base = self.builder.ins().load(
                vmctx,
                offset + VMMemory::base_offset(),
                MemFlags::new().with_alignment(8),
                VelocType::PTR,
            );
            let length = self.builder.ins().load(
                vmctx,
                offset + VMMemory::current_length_offset(),
                MemFlags::new().with_alignment(8),
                VelocType::I64,
            );
            (base, length)
        };

        if self.use_names {
            self.builder
                .set_value_name(base, &format!("mem{}_base", index));
            self.builder
                .set_value_name(length, &format!("mem{}_len", index));
        }

        let (base_var, len_var) = self.memory_vars[index as usize];
        self.builder.def_var(base_var, base);
        self.builder.def_var(len_var, length);
    }

    fn translate_load(&mut self, ty: VelocType, memarg: MemArg) {
        let addr = self.pop_i32();
        let mem_idx = memarg.memory;
        self.memory_bounds_check(
            mem_idx,
            addr,
            memarg.offset,
            ty.fixed_size_bytes()
                .expect("Wasm loads have fixed-size types"),
        );
        let mem_base = self.get_memory_base(mem_idx);
        let addr_i64 = self.addr_to_i64(addr);
        let actual_ptr = self.builder.ins().ptr_index(
            mem_base,
            addr_i64,
            veloc::mir::inst::PtrIndexImm {
                scale: 1,
                offset: 0,
            },
        );
        let flags = MemFlags::new().with_alignment(1 << memarg.align);
        let res = self
            .builder
            .ins()
            .load(actual_ptr, memarg.offset as u32, flags, ty);
        self.stack.push(res);
    }

    fn translate_store(&mut self, ty: VelocType, memarg: MemArg) {
        let val = self.pop_typed(ty);
        let addr = self.pop_i32();
        let mem_idx = memarg.memory;
        self.memory_bounds_check(
            mem_idx,
            addr,
            memarg.offset,
            ty.fixed_size_bytes()
                .expect("Wasm stores have fixed-size types"),
        );
        let mem_base = self.get_memory_base(mem_idx);
        let addr_i64 = self.addr_to_i64(addr);
        let actual_ptr = self.builder.ins().ptr_index(
            mem_base,
            addr_i64,
            veloc::mir::inst::PtrIndexImm {
                scale: 1,
                offset: 0,
            },
        );
        let flags = MemFlags::new().with_alignment(1 << memarg.align);
        self.builder
            .ins()
            .store(actual_ptr, val, memarg.offset as u32, flags);
    }

    pub(super) fn get_memory_base(&mut self, index: u32) -> Value {
        let (base_var, _) = self.memory_vars[index as usize];
        self.builder.use_var(base_var)
    }

    pub(super) fn memory_bounds_check(
        &mut self,
        index: u32,
        addr: Value,
        offset: u64,
        access_size: u32,
    ) {
        let (_, len_var) = self.memory_vars[index as usize];
        let length = self.builder.use_var(len_var);
        let addr_i64 = self.addr_to_i64(addr);
        let total_offset_imm = offset.wrapping_add(access_size as u64);
        let total_offset = self
            .builder
            .ins()
            .iconst(total_offset_imm as u64, VelocType::I64);
        let effective_end = self.builder.ins().iadd(addr_i64, total_offset);
        let is_oob = self.builder.ins().icmp(IntCC::GtU, effective_end, length);
        self.trap_if(is_oob, TrapCode::MemoryOutOfBounds);
    }

    pub(super) fn addr_to_i64(&mut self, addr: Value) -> Value {
        let addr_ty = self.builder.value_type(addr);
        if addr_ty == VelocType::I64 {
            addr
        } else {
            self.builder.ins().extendu(addr, VelocType::I64)
        }
    }
}
