use super::WasmTranslator;
use crate::vm::VMTable;
use veloc::ir::{IntCC, MemFlags, Type as VelocType, Value};
use wasmparser::{BinaryReaderError, Operator};

impl<'a> WasmTranslator<'a> {
    pub(super) fn translate_table(&mut self, op: Operator) -> Result<(), BinaryReaderError> {
        match op {
            Operator::TableGet { table } => {
                let index = self.pop();
                self.table_bounds_check(table, index);
                let table_base = self.get_table_base(table);
                let index_i64 = self.addr_to_i64(index);
                let addr = self.builder.ins().ptr_index(table_base, index_i64, 8, 0);
                let res = self.builder.ins().load(
                    VelocType::Ptr,
                    addr,
                    0,
                    MemFlags::new().with_alignment(8),
                );
                self.stack.push(res);
            }
            Operator::TableSet { table } => {
                let func_ref = self.pop();
                let index = self.pop();
                self.table_bounds_check(table, index);
                let table_base = self.get_table_base(table);
                let index_i64 = self.addr_to_i64(index);
                let entry_addr = self.builder.ins().ptr_index(table_base, index_i64, 8, 0);
                self.builder.ins().store(
                    func_ref,
                    entry_addr,
                    0,
                    MemFlags::new().with_alignment(8),
                );
            }
            Operator::TableInit { table, elem_index } => {
                let len = self.pop_i32();
                let src = self.pop_i32();
                let dst = self.pop_i32();
                let vmctx = self.vmctx.expect("vmctx not set");
                let table_idx = self.builder.ins().iconst(VelocType::I32, table as i64);
                let elem_idx = self.builder.ins().iconst(VelocType::I32, elem_index as i64);
                self.builder.ins().call(
                    self.runtime.table_init,
                    &[vmctx, table_idx, elem_idx, dst, src, len],
                );
            }
            Operator::TableCopy {
                dst_table,
                src_table,
            } => {
                let len = self.pop_i32();
                let src = self.pop_i32();
                let dst = self.pop_i32();
                let vmctx = self.vmctx.expect("vmctx not set");
                let dst_table_val = self.builder.ins().iconst(VelocType::I32, dst_table as i64);
                let src_table_val = self.builder.ins().iconst(VelocType::I32, src_table as i64);
                self.builder.ins().call(
                    self.runtime.table_copy,
                    &[vmctx, dst_table_val, src_table_val, dst, src, len],
                );
            }
            Operator::TableGrow { table } => {
                let delta = self.pop_i32();
                let init_val = self.pop();
                let vmctx = self.vmctx.expect("vmctx not set");
                let table_idx = self.builder.ins().iconst(VelocType::I32, table as i64);
                let res = self.builder.ins().call(
                    self.runtime.table_grow,
                    &[vmctx, table_idx, init_val, delta],
                );
                self.stack.push(res.unwrap());
                self.reload_table(table);
            }
            Operator::TableSize { table } => {
                let (_, len_var) = self.table_vars[table as usize];
                let table_len = self.builder.use_var(len_var);
                let table_len_i32 = self.builder.ins().wrap(table_len, VelocType::I32);
                self.stack.push(table_len_i32);
            }
            Operator::TableFill { table } => {
                let len = self.pop_i32();
                let val = self.pop();
                let dst = self.pop_i32();
                let vmctx = self.vmctx.expect("vmctx not set");
                let table_idx = self.builder.ins().iconst(VelocType::I32, table as i64);
                self.builder
                    .ins()
                    .call(self.runtime.table_fill, &[vmctx, table_idx, dst, val, len]);
            }
            Operator::ElemDrop { elem_index } => {
                let vmctx = self.vmctx.expect("vmctx not set");
                let elem_idx = self.builder.ins().iconst(VelocType::I32, elem_index as i64);
                self.builder
                    .ins()
                    .call(self.runtime.elem_drop, &[vmctx, elem_idx]);
            }
            _ => unreachable!("Non-table operator in translate_table"),
        }
        Ok(())
    }

    pub(super) fn reload_table(&mut self, index: u32) {
        let vmctx = self.vmctx.expect("vmctx not set");
        let offset = self.offsets.table_offset(index);
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
            VMTable::base_offset(),
            MemFlags::new().with_alignment(8),
        );
        let length = self.builder.ins().load(
            VelocType::I64,
            def_ptr,
            VMTable::current_elements_offset(),
            MemFlags::new().with_alignment(8),
        );

        if self.use_names {
            self.builder
                .set_value_name(base, &format!("tab{}_base", index));
            self.builder
                .set_value_name(length, &format!("tab{}_len", index));
        }

        let (base_var, len_var) = self.table_vars[index as usize];
        self.builder.def_var(base_var, base);
        self.builder.def_var(len_var, length);
    }

    pub(super) fn get_table_base(&mut self, index: u32) -> Value {
        let (base_var, _) = self.table_vars[index as usize];
        self.builder.use_var(base_var)
    }

    pub(super) fn table_bounds_check(&mut self, table_index: u32, index: Value) {
        let (_, len_var) = self.table_vars[table_index as usize];
        let table_len = self.builder.use_var(len_var);
        let index_i64 = self.addr_to_i64(index);
        let is_oob = self.builder.ins().icmp(IntCC::GeU, index_i64, table_len);
        self.trap_if(is_oob, crate::vm::TrapCode::TableOutOfBounds);
    }
}
