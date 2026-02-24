use crate::bytecode::{CompiledFunction, compile_function};
use crate::value::{HostFuncArgs, HostFuncRets, InterpreterValue};
use ::alloc::boxed::Box;
use ::alloc::string::String;
use ::alloc::sync::Arc;
use ::alloc::vec::Vec;
use cranelift_entity::{EntityRef, PrimaryMap, entity_impl};
use hashbrown::HashMap;
use veloc_ir::{FuncId, Module, ModuleId};

pub type HostFunction = Arc<dyn Fn(&[InterpreterValue]) -> InterpreterValue + Send + Sync>;

pub type TrampolineFn =
    unsafe extern "C" fn(env: *mut u8, args_results: *mut InterpreterValue, arity: usize);

pub struct HostFunctionInner {
    pub(crate) handler: TrampolineFn,
    pub(crate) env: *mut u8,
    pub(crate) drop_fn: fn(*mut u8),
}

unsafe impl Send for HostFunctionInner {}
unsafe impl Sync for HostFunctionInner {}

impl Drop for HostFunctionInner {
    fn drop(&mut self) {
        (self.drop_fn)(self.env);
    }
}

#[derive(Clone)]
pub struct HostFunc(pub Arc<HostFunctionInner>);

impl core::fmt::Debug for HostFunc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HostFunc").finish()
    }
}

impl HostFunc {
    pub fn call(&self, args_results: &mut [InterpreterValue], param_count: usize) {
        unsafe {
            (self.0.handler)(self.0.env, args_results.as_mut_ptr(), param_count);
        }
    }
}

/// A reference to a host function identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HostFuncId(u32);
entity_impl!(HostFuncId);

impl core::fmt::Debug for HostFuncId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "host{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImportTarget {
    Module(ModuleId, FuncId),
    Host(HostFuncId),
    /// Internal function or unresolved import
    None,
}

/// A tagged function pointer that can represent either a host function or an interpreter function.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct VMFuncPointer(pub usize);

impl VMFuncPointer {
    pub fn from_host(id: HostFuncId) -> Self {
        Self(((id.index() as usize) << 2) | 2)
    }

    pub fn from_interpreter(mid: ModuleId, fid: FuncId) -> Self {
        let val = ((mid.index() as u64) << 33) | ((fid.index() as u64) << 1) | 1;
        Self(val as usize)
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.0 as *const u8
    }

    pub fn decode(&self) -> Option<ImportTarget> {
        if self.0 & 1 == 1 {
            let mid = (self.0 >> 33) as u32;
            let fid = ((self.0 >> 1) & 0xFFFFFFFF) as u32;
            Some(ImportTarget::Module(
                ModuleId::new(mid as usize),
                FuncId::new(fid as usize),
            ))
        } else if self.0 & 3 == 2 {
            Some(ImportTarget::Host(HostFuncId::new((self.0 >> 2) as usize)))
        } else {
            None
        }
    }
}

pub struct RuntimeModule {
    pub ir: Module,
    pub compiled: Vec<Option<Arc<CompiledFunction>>>,
    pub links: Vec<ImportTarget>,
}

pub struct Program {
    pub host_functions: HashMap<String, HostFuncId>,
    pub host_functions_list: PrimaryMap<HostFuncId, HostFunc>,
    pub modules: PrimaryMap<ModuleId, RuntimeModule>,
}

impl Program {
    pub fn new() -> Self {
        Self {
            host_functions: HashMap::new(),
            host_functions_list: PrimaryMap::new(),
            modules: PrimaryMap::new(),
        }
    }

    pub fn link_import(
        &mut self,
        mid: ModuleId,
        fid: FuncId,
        target_mid: ModuleId,
        target_fid: FuncId,
    ) {
        self.modules[mid].links[fid.index()] = ImportTarget::Module(target_mid, target_fid);
    }

    pub fn link_host(&mut self, mid: ModuleId, fid: FuncId, host_fid: HostFuncId) {
        self.modules[mid].links[fid.index()] = ImportTarget::Host(host_fid);
    }

    pub(crate) fn get_compiled_func(&self, mid: ModuleId, fid: FuncId) -> Arc<CompiledFunction> {
        self.modules[mid].compiled[fid.index()]
            .as_ref()
            .expect("Function not compiled or not defined")
            .clone()
    }

    pub fn register_module(&mut self, module: Module) -> ModuleId {
        // Predict the ModuleId that will be assigned
        let mid = ModuleId::new(self.modules.len());

        let mut compiled = Vec::new();
        let mut links = Vec::new();

        for (fid, func) in module.functions.iter() {
            if func.is_defined() {
                compiled.push(Some(Arc::new(compile_function(mid, fid, func))));
            } else {
                compiled.push(None);
            }
            links.push(ImportTarget::None);
        }

        let actual_mid = self.modules.push(RuntimeModule {
            ir: module,
            compiled,
            links,
        });
        debug_assert_eq!(mid, actual_mid);
        actual_mid
    }

    fn register_handler<F>(
        &mut self,
        name: String,
        handler: F,
        trampoline: TrampolineFn,
    ) -> HostFuncId
    where
        F: Send + Sync + 'static,
    {
        let env = Box::into_raw(Box::new(handler)) as *mut u8;
        let drop_fn = |ptr: *mut u8| unsafe {
            let _ = Box::from_raw(ptr as *mut F);
        };

        let host_func = HostFunc(Arc::new(HostFunctionInner {
            handler: trampoline,
            env,
            drop_fn,
        }));

        let id = self.host_functions_list.push(host_func);
        self.host_functions.insert(name, id);
        id
    }

    pub fn register_raw(&mut self, name: String, f: HostFunction) -> HostFuncId {
        unsafe extern "C" fn trampoline(
            env: *mut u8,
            args_results: *mut InterpreterValue,
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

    pub fn register_func<F, Args, Rets>(&mut self, name: String, func: F) -> HostFuncId
    where
        F: Fn(Args) -> Rets + Send + Sync + 'static,
        Args: HostFuncArgs,
        Rets: HostFuncRets,
    {
        unsafe extern "C" fn trampoline<F, Args, Rets>(
            env: *mut u8,
            args_results: *mut InterpreterValue,
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

    pub fn get_host_func_ptr(&self, id: HostFuncId) -> *const u8 {
        VMFuncPointer::from_host(id).as_ptr()
    }

    pub fn get_interpreter_func_ptr(&self, module_id: ModuleId, func_id: FuncId) -> *const u8 {
        VMFuncPointer::from_interpreter(module_id, func_id).as_ptr()
    }

    pub fn decode_ptr(&self, ptr_val: usize) -> Option<ImportTarget> {
        VMFuncPointer(ptr_val).decode()
    }

    /// Automatically link imports based on symbol names
    pub fn auto_link(&mut self) {
        let mut links = Vec::new();

        for (mid, runtime_module) in self.modules.iter() {
            for (fid, func) in runtime_module.ir.functions.iter() {
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
                        if let Some(target_fid) = target_module.ir.find_function_by_name(name) {
                            if target_module.ir.get_function(target_fid).linkage
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
