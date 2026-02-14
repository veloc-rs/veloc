use super::WasmTranslator;
use veloc::ir::Value;
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
                let val = self.builder.ins().load(veloc_ty, global_val_ptr, 0);
                self.stack.push(val);
            }
            Operator::GlobalSet { global_index } => {
                let val = self.pop();
                let global_val_ptr = self.get_global_ptr(global_index);
                self.builder.ins().store(val, global_val_ptr, 0);
            }
            _ => unreachable!("Non-variable operator in translate_variable"),
        }
        Ok(())
    }

    pub(super) fn get_global_ptr(&mut self, index: u32) -> Value {
        let var = self.global_ptr_vars[index as usize];
        self.builder.use_var(var)
    }
}
