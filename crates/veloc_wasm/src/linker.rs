use crate::Extern;
use crate::instance::Instance;
use crate::module::Module;
use crate::store::{InstanceId, Store};
use crate::vm::VMFuncRef;
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use hashbrown::HashMap;
use veloc::interpreter::{HostFunction, InterpreterValue};

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

    pub fn func_wrap(
        &mut self,
        store: &mut Store,
        module: &str,
        name: &str,
        f: impl Fn(&[InterpreterValue]) -> InterpreterValue + Send + Sync + 'static,
    ) -> &mut Self {
        let host_fn = Arc::new(f) as HostFunction;
        let full_name = format!("{}.{}", module, name);
        let id = store.program.register_host_function(full_name, host_fn);
        let native_call = store.program.get_host_func_ptr(id);

        let func_ref = VMFuncRef {
            native_call: native_call as *const core::ffi::c_void,
            vmctx: core::ptr::null_mut(),
            type_index: 0,
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
    ) -> crate::error::Result<InstanceId> {
        let mut extern_imports: HashMap<String, HashMap<String, Extern>> = HashMap::new();

        let meta = module.metadata();
        for import in &meta.imports {
            if let Some(m) = extern_imports.get(&import.module) {
                if m.contains_key(&import.field) {
                    continue;
                }
            }

            let key = (import.module.clone(), import.field.clone());
            let resolved = self
                .definitions
                .get(&key)
                .cloned()
                .or_else(|| self.definitions_name.get(&import.field).cloned());

            if let Some(ext) = resolved {
                let export_map = extern_imports
                    .entry(import.module.clone())
                    .or_insert_with(HashMap::new);
                export_map.insert(import.field.clone(), ext);
            } else {
                return Err(crate::error::Error::Message(format!(
                    "Import not found: {}.{}",
                    import.module, import.field
                )));
            }
        }

        Instance::new(store, module, &extern_imports)
    }
}
