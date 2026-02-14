use crate::Extern;
use crate::Result;
use crate::func::IntoFunc;
use crate::instance::VMInstance;
use crate::module::Module;
use crate::module::types::WasmSignature;
use crate::store::{Instance, Store};
use crate::vm::VMFuncRef;
use crate::wasi;
use alloc::format;
use alloc::string::String;
use hashbrown::HashMap;

pub struct Linker {
    definitions: HashMap<(String, String), Extern>,
    definitions_name: HashMap<String, Extern>,
}

impl Linker {
    pub fn new() -> Self {
        Self {
            definitions: HashMap::new(),
            definitions_name: HashMap::new(),
        }
    }

    pub fn define(
        &mut self,
        _store: &Store,
        module: &str,
        name: &str,
        item: impl Into<Extern>,
    ) -> crate::error::Result<&mut Self> {
        self.definitions
            .insert((module.to_string(), name.to_string()), item.into());
        Ok(self)
    }

    pub fn define_name(
        &mut self,
        _store: &Store,
        name: &str,
        item: impl Into<Extern>,
    ) -> crate::error::Result<&mut Self> {
        self.definitions_name.insert(name.to_string(), item.into());
        Ok(self)
    }

    pub fn definitions(&self) -> &HashMap<(String, String), Extern> {
        &self.definitions
    }

    pub fn add_wasi(&mut self, store: &mut Store) -> Result<&mut Self> {
        wasi::add_to_linker(self, store)?;
        Ok(self)
    }

    pub fn func_wrap<Params, Results, F>(
        &mut self,
        store: &mut Store,
        module: &str,
        name: &str,
        f: F,
    ) -> &mut Self
    where
        F: IntoFunc<Params, Results>,
    {
        let (params, results, host_fn) = f.into_func();
        let full_name = format!("{}.{}", module, name);
        let id = store.program.register_host_function(full_name, host_fn);
        let native_call = store.program.get_host_func_ptr(id);

        let sig = WasmSignature::new(params, results);

        let func_ref = VMFuncRef {
            native_call: native_call as *const core::ffi::c_void,
            vmctx: core::ptr::null_mut(),
            type_index: sig.hash,
            offset: 0,
            caller: core::ptr::null_mut(),
        };

        self.definitions.insert(
            (module.to_string(), name.to_string()),
            Extern::Function(func_ref),
        );
        self
    }

    pub fn instantiate(
        &mut self,
        store: &mut Store,
        module: Module,
    ) -> crate::error::Result<Instance> {
        let meta = module.metadata();
        let mut resolved_imports = Vec::with_capacity(meta.imports.len());

        for import in &meta.imports {
            let key = (import.module.clone(), import.field.clone());
            let resolved = self
                .definitions
                .get(&key)
                .cloned()
                .or_else(|| self.definitions_name.get(&import.field).cloned());

            if let Some(ext) = resolved {
                resolved_imports.push(ext);
            } else {
                return Err(crate::error::Error::Message(format!(
                    "Import not found: {}.{}",
                    import.module, import.field
                )));
            }
        }

        VMInstance::new(store, module, &resolved_imports)
    }
}
