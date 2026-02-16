use crate::bytecode::{CompiledFunction, compile_function};
use crate::value::{HostFuncArgs, HostFuncRets, InterpreterValue};
use ::alloc::boxed::Box;
use ::alloc::string::String;
use ::alloc::sync::Arc;
use ::alloc::vec::Vec;
use hashbrown::HashMap;
use veloc_ir::{FuncId, Module};

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
    pub fn call(&self, args: &mut [InterpreterValue]) -> InterpreterValue {
        let arity = args.len();
        unsafe {
            (self.0.handler)(self.0.env, args.as_mut_ptr(), arity);
        }
        if arity > 0 {
            args[0]
        } else {
            InterpreterValue::none()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImportTarget {
    Module(ModuleId, FuncId),
    Host(usize),
}

pub struct Program {
    pub host_functions: HashMap<String, HostFunc>,
    pub host_functions_list: Vec<HostFunc>,
    pub modules: Vec<Module>,
    pub compiled_modules: Vec<Vec<Option<Arc<CompiledFunction>>>>,
    pub import_links: HashMap<(ModuleId, FuncId), ImportTarget>,
}

impl Program {
    pub fn new() -> Self {
        Self {
            host_functions: HashMap::new(),
            host_functions_list: Vec::new(),
            modules: Vec::new(),
            compiled_modules: Vec::new(),
            import_links: HashMap::new(),
        }
    }

    pub fn link_import(
        &mut self,
        mid: ModuleId,
        fid: FuncId,
        target_mid: ModuleId,
        target_fid: FuncId,
    ) {
        self.import_links
            .insert((mid, fid), ImportTarget::Module(target_mid, target_fid));
    }

    pub fn link_host(&mut self, mid: ModuleId, fid: FuncId, host_fid: usize) {
        self.import_links
            .insert((mid, fid), ImportTarget::Host(host_fid));
    }

    pub fn get_compiled_func(&self, mid: ModuleId, fid: FuncId) -> Arc<CompiledFunction> {
        self.compiled_modules[mid.0][fid.0 as usize]
            .as_ref()
            .expect("Function not compiled or not defined")
            .clone()
    }

    pub fn register_module(&mut self, module: Module) -> ModuleId {
        let mid = ModuleId(self.modules.len());

        let mut compiled = Vec::new();
        // Module functions are stored in a PrimaryMap usually, so .values() or similar
        // Let's check how reloc_ir::Module::functions is structured.
        // Usually it's a PrimaryMap<FuncId, Function>
        for (fid, func) in module.functions.iter() {
            if func.is_defined() {
                compiled.push(Some(Arc::new(compile_function(mid, fid, func))));
            } else {
                compiled.push(None);
            }
        }

        self.modules.push(module);
        self.compiled_modules.push(compiled);
        mid
    }

    fn register_handler<F>(&mut self, name: String, handler: F, trampoline: TrampolineFn) -> usize
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

        self.host_functions.insert(name, host_func.clone());
        let id = self.host_functions_list.len();
        self.host_functions_list.push(host_func);
        id
    }

    pub fn register_raw(&mut self, name: String, f: HostFunction) -> usize {
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

    pub fn register_func<F, Args, Rets>(&mut self, name: String, func: F) -> usize
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

    pub fn get_host_func_ptr(&self, id: usize) -> *const u8 {
        let val = ((id as u64) << 2) | 2;
        val as *const u8
    }

    pub fn get_interpreter_func_ptr(&self, module_id: ModuleId, func_id: FuncId) -> *const u8 {
        let mid = module_id.0 as u64;
        let fid = func_id.0 as u64;
        let val = (mid << 33) | (fid << 1) | 1;
        val as *const u8
    }

    pub fn decode_interpreter_ptr(&self, ptr_val: usize) -> Option<(ModuleId, FuncId)> {
        if ptr_val & 1 == 1 {
            let mid = (ptr_val >> 33) as usize;
            let fid = ((ptr_val >> 1) & 0xFFFFFFFF) as u32;
            Some((ModuleId(mid), FuncId(fid)))
        } else {
            None
        }
    }

    pub fn decode_host_ptr(&self, ptr_val: usize) -> Option<usize> {
        if ptr_val & 3 == 2 {
            Some(ptr_val >> 2)
        } else {
            None
        }
    }
}
