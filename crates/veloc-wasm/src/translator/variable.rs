use super::WasmTranslator;
use veloc::ir::{MemFlags, Value};
use wasmparser::{BinaryReaderError, Operator};

impl<'a> WasmTranslator<'a> {
    pub(super) fn translate_variable(&mut self, op: Operator) -> Result<(), BinaryReaderError> {
        match op {
            Operator::LocalGet { local_index } => {
                let (var, _) = self.locals[local_index as usize];
                let val = self.builder.use_var(var);
                self.stack.push(val);
            }
            Operator::LocalSet { local_index } => {
                let val = self.pop();
                let (var, _) = self.locals[local_index as usize];
                self.builder.def_var(var, val);
            }
            Operator::LocalTee { local_index } => {
                let val = *self.stack.last().expect("stack empty");
                let (var, _) = self.locals[local_index as usize];
                self.builder.def_var(var, val);
            }
            Operator::GlobalGet { global_index } => {
                let ty = self.metadata.globals[global_index as usize].ty;
                let veloc_ty = self.val_type_to_veloc(ty);
                let global_val_ptr = self.get_global_ptr(global_index);
                let val = self
                    .builder
                    .ins()
                    .load(veloc_ty, global_val_ptr, 0, MemFlags::default());
                self.stack.push(val);
            }
            Operator::GlobalSet { global_index } => {
                let val = self.pop();
                let global_val_ptr = self.get_global_ptr(global_index);
                self.builder
                    .ins()
                    .store(val, global_val_ptr, 0, MemFlags::default());
            }
            _ => unreachable!("Non-variable operator in translate_variable"),
        }
        Ok(())
    }

    /// 获取 global 的指针值
    /// 对于导入的 global：返回之前加载的指针变量值
    /// 对于本地的 global：返回 vmctx + 偏移量计算出的指针
    pub(super) fn get_global_ptr(&mut self, index: u32) -> Value {
        match self.global_ptr_vars[index as usize] {
            Some(var) => self.builder.use_var(var),
            None => {
                // 本地 global：计算 vmctx + offset
                let vmctx = self.vmctx.expect("vmctx not set");
                let (_, offset) = self.offsets.global_access_info(index);
                self.builder.ins().ptr_offset(vmctx, offset as i32)
            }
        }
    }
}
