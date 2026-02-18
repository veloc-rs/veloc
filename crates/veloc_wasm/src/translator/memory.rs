use super::WasmTranslator;
use crate::vm::{TrapCode, VMMemory};
use veloc::ir::{IntCC, MemFlags, Type as VelocType, Value};
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
                let extended = self.builder.ins().extend_s(v, VelocType::I32);
                self.stack.push(extended);
            }
            Operator::I32Load8U { memarg } => {
                self.translate_load(VelocType::I8, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extend_u(v, VelocType::I32);
                self.stack.push(extended);
            }
            Operator::I32Load16S { memarg } => {
                self.translate_load(VelocType::I16, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extend_s(v, VelocType::I32);
                self.stack.push(extended);
            }
            Operator::I32Load16U { memarg } => {
                self.translate_load(VelocType::I16, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extend_u(v, VelocType::I32);
                self.stack.push(extended);
            }
            Operator::I64Load8S { memarg } => {
                self.translate_load(VelocType::I8, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extend_s(v, VelocType::I64);
                self.stack.push(extended);
            }
            Operator::I64Load8U { memarg } => {
                self.translate_load(VelocType::I8, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extend_u(v, VelocType::I64);
                self.stack.push(extended);
            }
            Operator::I64Load16S { memarg } => {
                self.translate_load(VelocType::I16, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extend_s(v, VelocType::I64);
                self.stack.push(extended);
            }
            Operator::I64Load16U { memarg } => {
                self.translate_load(VelocType::I16, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extend_u(v, VelocType::I64);
                self.stack.push(extended);
            }
            Operator::I64Load32S { memarg } => {
                self.translate_load(VelocType::I32, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extend_s(v, VelocType::I64);
                self.stack.push(extended);
            }
            Operator::I64Load32U { memarg } => {
                self.translate_load(VelocType::I32, memarg);
                let v = self.pop();
                let extended = self.builder.ins().extend_u(v, VelocType::I64);
                self.stack.push(extended);
            }
            Operator::I32Store { memarg } => self.translate_store(VelocType::I32, memarg),
            Operator::I64Store { memarg } => self.translate_store(VelocType::I64, memarg),
            Operator::F32Store { memarg } => self.translate_store(VelocType::F32, memarg),
            Operator::F64Store { memarg } => self.translate_store(VelocType::F64, memarg),
            Operator::I32Store8 { memarg } => {
                let val = self.pop();
                let truncated = self.builder.ins().wrap(val, VelocType::I8);
                self.stack.push(truncated);
                self.translate_store(VelocType::I8, memarg);
            }
            Operator::I32Store16 { memarg } => {
                let val = self.pop();
                let truncated = self.builder.ins().wrap(val, VelocType::I16);
                self.stack.push(truncated);
                self.translate_store(VelocType::I16, memarg);
            }
            Operator::I64Store8 { memarg } => {
                let val = self.pop();
                let truncated = self.builder.ins().wrap(val, VelocType::I8);
                self.stack.push(truncated);
                self.translate_store(VelocType::I8, memarg);
            }
            Operator::I64Store16 { memarg } => {
                let val = self.pop();
                let truncated = self.builder.ins().wrap(val, VelocType::I16);
                self.stack.push(truncated);
                self.translate_store(VelocType::I16, memarg);
            }
            Operator::I64Store32 { memarg } => {
                let val = self.pop();
                let truncated = self.builder.ins().wrap(val, VelocType::I32);
                self.stack.push(truncated);
                self.translate_store(VelocType::I32, memarg);
            }
            Operator::MemorySize { mem, .. } => {
                let (_, len_var) = self.memory_vars[mem as usize];
                let size_bytes = self.builder.use_var(len_var);
                let page_size = self.builder.ins().iconst(VelocType::I64, 65536);
                let size_pages = self.builder.ins().udiv(size_bytes, page_size);
                let size_i32 = self.builder.ins().wrap(size_pages, VelocType::I32);
                self.stack.push(size_i32);
            }
            Operator::MemoryGrow { mem, .. } => {
                let delta = self.pop();
                let vmctx = self.vmctx.expect("vmctx not set");
                let mem_idx = self.builder.ins().iconst(VelocType::I32, mem as i64);
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
                let mem_idx = self.builder.ins().iconst(VelocType::I32, mem as i64);
                let data_idx = self.builder.ins().iconst(VelocType::I32, data_index as i64);
                self.builder.ins().call(
                    self.runtime.memory_init,
                    &[vmctx, mem_idx, data_idx, dst, src, len],
                );
            }
            Operator::DataDrop { data_index } => {
                let vmctx = self.vmctx.expect("vmctx not set");
                let data_idx = self.builder.ins().iconst(VelocType::I32, data_index as i64);
                self.builder
                    .ins()
                    .call(self.runtime.data_drop, &[vmctx, data_idx]);
            }
            Operator::MemoryCopy { dst_mem, src_mem } => {
                let len = self.pop_i32();
                let src = self.pop_i32();
                let dst = self.pop_i32();
                let vmctx = self.vmctx.expect("vmctx not set");
                let dst_mem_val = self.builder.ins().iconst(VelocType::I32, dst_mem as i64);
                let src_mem_val = self.builder.ins().iconst(VelocType::I32, src_mem as i64);
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
                let mem_idx = self.builder.ins().iconst(VelocType::I32, mem as i64);
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
        let offset = self.offsets.memory_offset(index);
        let alignment = if offset % 16 == 0 { 16 } else { 8 };
        let def_ptr = self.builder.ins().load(
            VelocType::Ptr,
            vmctx,
            offset,
            MemFlags::new().with_alignment(alignment),
        );
        let base = self.builder.ins().load(
            VelocType::Ptr,
            def_ptr,
            VMMemory::base_offset(),
            MemFlags::new().with_alignment(8),
        );
        let length = self.builder.ins().load(
            VelocType::I64,
            def_ptr,
            VMMemory::current_length_offset(),
            MemFlags::new().with_alignment(8),
        );

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
        let addr = self.pop();
        let mem_idx = memarg.memory;
        self.memory_bounds_check(mem_idx, addr, memarg.offset, ty.size_bytes());
        let mem_base = self.get_memory_base(mem_idx);
        let addr_i64 = self.addr_to_i64(addr);
        let actual_ptr = self.builder.ins().ptr_index(mem_base, addr_i64, 1, 0);
        let flags = MemFlags::new().with_alignment(1 << memarg.align);
        let res = self
            .builder
            .ins()
            .load(ty, actual_ptr, memarg.offset as u32, flags);
        self.stack.push(res);
    }

    fn translate_store(&mut self, ty: VelocType, memarg: MemArg) {
        let val = self.pop();
        let addr = self.pop();
        let mem_idx = memarg.memory;
        self.memory_bounds_check(mem_idx, addr, memarg.offset, ty.size_bytes());
        let mem_base = self.get_memory_base(mem_idx);
        let addr_i64 = self.addr_to_i64(addr);
        let actual_ptr = self.builder.ins().ptr_index(mem_base, addr_i64, 1, 0);
        let flags = MemFlags::new().with_alignment(1 << memarg.align);
        self.builder
            .ins()
            .store(val, actual_ptr, memarg.offset as u32, flags);
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
            .iconst(VelocType::I64, total_offset_imm as i64);
        let effective_end = self.builder.ins().iadd(addr_i64, total_offset);
        let is_oob = self.builder.ins().icmp(IntCC::GtU, effective_end, length);
        self.trap_if(is_oob, TrapCode::MemoryOutOfBounds);
    }

    pub(super) fn addr_to_i64(&mut self, addr: Value) -> Value {
        let addr_ty = self.builder.value_type(addr);
        if addr_ty == VelocType::I64 {
            addr
        } else {
            self.builder.ins().extend_u(addr, VelocType::I64)
        }
    }
}
