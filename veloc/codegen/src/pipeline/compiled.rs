use crate::mir::{MachineFunction, SymbolTable};
use crate::pipeline::stages::PrologueEpilogueInserted;
use alloc::string::String;
use alloc::vec::Vec;
use veloc_ir::FuncId;

#[derive(Debug, Clone)]
pub struct CompiledFunction {
    pub func_id: FuncId,
    pub name: String,
    pub machine_function: MachineFunction<PrologueEpilogueInserted>,
    pub emitted: Option<crate::EmittedCode>,
}

#[derive(Debug, Clone)]
pub struct CompiledModule {
    pub name: String,
    pub symbols: SymbolTable,
    pub functions: Vec<CompiledFunction>,
}

impl CompiledModule {
    pub fn new(name: String, symbols: SymbolTable, functions: Vec<CompiledFunction>) -> Self {
        Self {
            name,
            symbols,
            functions,
        }
    }
}
