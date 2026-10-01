use std::string::String;
use std::vec::Vec;
use veloc_lir::{MachineFunction, SymbolTable};
use veloc_mir::FuncId;

#[derive(Debug, Clone)]
pub struct CompiledFunction {
    pub func_id: FuncId,
    /// Definition identity used by section layout and symbolic calls.
    pub symbol: veloc_lir::SymbolId,
    pub name: String,
    pub machine_function: MachineFunction,
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

/// Symbolic code is the sole mutable representation after emission.
#[derive(Debug, Clone)]
pub struct EmittedFunction {
    pub func_id: FuncId,
    pub symbol: veloc_lir::SymbolId,
    pub name: String,
    pub emission: crate::Emitter,
}
#[derive(Debug, Clone)]
pub struct EmissionModule {
    pub name: String,
    pub symbols: SymbolTable,
    pub functions: Vec<EmittedFunction>,
}
