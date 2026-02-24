//! Program runtime management
//!
//! This module provides the main Program structure that manages all loaded
//! modules, host functions, and their interactions.

use crate::bytecode::{CompiledFunction, compile_function};
use crate::host::{HostFunc, HostFuncId, HostFunction};
use crate::runtime::RuntimeModule;
use crate::runtime::ptr::ImportTarget;
use crate::value::{HostFuncArgs, HostFuncRets};
use ::alloc::boxed::Box;
use ::alloc::string::String;
use ::alloc::sync::Arc;
use ::alloc::vec::Vec;
use cranelift_entity::{EntityRef, PrimaryMap, SecondaryMap};
use hashbrown::HashMap;
use veloc_ir::{FuncId, Module, ModuleId};

/// Main program structure managing all modules and host functions
pub struct Program {
    /// Map from host function name to ID
    pub(crate) host_functions: HashMap<String, HostFuncId>,
    /// Storage for host function implementations
    pub(crate) host_functions_list: PrimaryMap<HostFuncId, HostFunc>,
    /// Loaded modules
    pub(crate) modules: PrimaryMap<ModuleId, RuntimeModule>,
}

// ============== Accessors ==============

impl Program {
    /// Get a host function ID by name
    pub fn get_host_function(&self, name: &str) -> Option<HostFuncId> {
        self.host_functions.get(name).copied()
    }

    /// Get the number of loaded modules
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    /// Collect all compiled functions into a Vec
    /// Returns Vec of (module_id, function_id, compiled_function)
    pub fn all_compiled_functions(&self) -> Vec<(ModuleId, FuncId, Arc<CompiledFunction>)> {
        let mut result = Vec::new();
        for (mid, module) in self.modules.iter() {
            for (fid, func) in module.compiled.iter() {
                if let Some(compiled) = func {
                    result.push((mid, fid, compiled.clone()));
                }
            }
        }
        result
    }
}

impl Program {
    /// Create a new empty program
    pub fn new() -> Self {
        Self {
            host_functions: HashMap::new(),
            host_functions_list: PrimaryMap::new(),
            modules: PrimaryMap::new(),
        }
    }

    /// Get a compiled function from a module
    pub(crate) fn get_compiled_func(&self, mid: ModuleId, fid: FuncId) -> Arc<CompiledFunction> {
        self.modules[mid].compiled[fid]
            .clone()
            .expect("Function not compiled or not defined")
    }

    /// Register a module and compile its defined functions
    pub fn register_module(&mut self, module: Module) -> ModuleId {
        // Predict the ModuleId that will be assigned
        let mid = ModuleId::new(self.modules.len());

        let mut compiled: SecondaryMap<FuncId, Option<Arc<CompiledFunction>>> = SecondaryMap::new();
        let mut links: SecondaryMap<FuncId, ImportTarget> = SecondaryMap::new();

        for (fid, func) in module.functions.iter() {
            if func.is_defined() {
                compiled[fid] = Some(Arc::new(compile_function(mid, fid, func)));
            } else {
                compiled[fid] = None;
            }
            links[fid] = ImportTarget::None;
        }

        let actual_mid = self
            .modules
            .push(RuntimeModule::new(module, compiled, links));
        debug_assert_eq!(mid, actual_mid);
        actual_mid
    }

    /// Get a pointer to a host function (for indirect calls)
    pub fn get_host_func_ptr(&self, id: HostFuncId) -> *const u8 {
        crate::runtime::VMFuncPointer::from_host(id).as_ptr()
    }

    /// Get a pointer to an interpreter function (for indirect calls)
    pub fn get_interpreter_func_ptr(&self, module_id: ModuleId, func_id: FuncId) -> *const u8 {
        crate::runtime::VMFuncPointer::from_interpreter(module_id, func_id).as_ptr()
    }

    /// Decode a function pointer to its target
    pub fn decode_ptr(&self, ptr_val: usize) -> Option<ImportTarget> {
        crate::runtime::VMFuncPointer(ptr_val).decode()
    }

    /// Register a raw host function
    pub fn register_raw(&mut self, name: String, f: HostFunction) -> HostFuncId {
        unsafe extern "C" fn trampoline(
            env: *mut u8,
            args_results: *mut crate::value::InterpreterValue,
            arity: usize,
        ) {
            unsafe {
                let func = &*(env as *const HostFunction);
                let args_slice = core::slice::from_raw_parts(args_results, arity);
                let res = func(args_slice);
                *args_results = res;
            }
        }

        self.register_handler(name, f, trampoline)
    }

    /// Register a typed host function
    pub fn register_func<F, Args, Rets>(&mut self, name: String, func: F) -> HostFuncId
    where
        F: Fn(Args) -> Rets + Send + Sync + 'static,
        Args: HostFuncArgs,
        Rets: HostFuncRets,
    {
        unsafe extern "C" fn trampoline<F, Args, Rets>(
            env: *mut u8,
            args_results: *mut crate::value::InterpreterValue,
            arity: usize,
        ) where
            F: Fn(Args) -> Rets + Send + Sync + 'static,
            Args: HostFuncArgs,
            Rets: HostFuncRets,
        {
            unsafe {
                let func = &*(env as *const F);
                let args_slice = core::slice::from_raw_parts(args_results, arity);
                let args = Args::decode(args_slice);
                let rets = func(args);
                let results_slice = core::slice::from_raw_parts_mut(args_results, 8.max(arity));
                rets.encode(results_slice);
            }
        }

        self.register_handler(name, func, trampoline::<F, Args, Rets>)
    }

    fn register_handler<F>(
        &mut self,
        name: String,
        handler: F,
        trampoline: unsafe extern "C" fn(*mut u8, *mut crate::value::InterpreterValue, usize),
    ) -> HostFuncId
    where
        F: Send + Sync + 'static,
    {
        let env = Box::into_raw(Box::new(handler)) as *mut u8;
        let drop_fn = |ptr: *mut u8| unsafe {
            let _ = Box::from_raw(ptr as *mut F);
        };

        let host_func = HostFunc(Arc::new(crate::host::HostFunctionInner {
            handler: trampoline,
            env,
            drop_fn,
        }));

        let id = self.host_functions_list.push(host_func);
        self.host_functions.insert(name, id);
        id
    }

    /// Link an import to another module's function
    pub fn link_import(
        &mut self,
        mid: ModuleId,
        fid: FuncId,
        target_mid: ModuleId,
        target_fid: FuncId,
    ) {
        self.modules[mid].links[fid] = ImportTarget::Module(target_mid, target_fid);
    }

    /// Link an import to a host function
    pub fn link_host(&mut self, mid: ModuleId, fid: FuncId, host_fid: HostFuncId) {
        self.modules[mid].links[fid] = ImportTarget::Host(host_fid);
    }

    /// Automatically link imports based on symbol names
    pub fn auto_link(&mut self) {
        let mut links = Vec::new();

        for (mid, runtime_module) in self.modules.iter() {
            for (fid, func) in runtime_module.ir().functions.iter() {
                if func.linkage == veloc_ir::Linkage::Import {
                    let name = &func.name;

                    // 1. Try to link to host functions
                    if let Some(&host_id) = self.host_functions.get(name) {
                        links.push((mid, fid, ImportTarget::Host(host_id)));
                        continue;
                    }

                    // 2. Try to link to other modules' exports
                    for (target_mid, target_module) in self.modules.iter() {
                        if target_mid == mid {
                            continue;
                        }
                        if let Some(target_fid) = target_module.ir().find_function_by_name(name) {
                            if target_module.ir().get_function(target_fid).linkage
                                == veloc_ir::Linkage::Export
                            {
                                links.push((
                                    mid,
                                    fid,
                                    ImportTarget::Module(target_mid, target_fid),
                                ));
                                break;
                            }
                        }
                    }
                }
            }
        }

        for (mid, fid, target) in links {
            match target {
                ImportTarget::Module(tm, tf) => self.link_import(mid, fid, tm, tf),
                ImportTarget::Host(h) => self.link_host(mid, fid, h),
                ImportTarget::None => {}
            }
        }
    }
}
