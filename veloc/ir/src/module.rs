use crate::function::Function;
use crate::types::{FuncId, SigId, Signature, Type};
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Deref;
use cranelift_entity::PrimaryMap;
use hashbrown::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    Import,
    Export,
    Local,
}

impl core::fmt::Display for Linkage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Linkage::Import => write!(f, "import"),
            Linkage::Export => write!(f, "export"),
            Linkage::Local => write!(f, "local"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Global {
    pub name: String,
    pub ty: Type,
    pub linkage: Linkage,
}

#[derive(Debug, Default, Clone)]
pub struct ModuleData {
    pub functions: PrimaryMap<FuncId, Function>,
    pub signatures: PrimaryMap<SigId, Signature>,
    pub globals: Vec<Global>,
    sig_map: HashMap<Signature, SigId>,
    /// 当前模块的修订版本
    revision: u64,
}

impl ModuleData {
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn bump_revision(&mut self) {
        self.revision += 1;
    }

    pub fn get_func_id(&self, name: &str) -> Option<FuncId> {
        self.functions
            .iter()
            .find(|(_, f)| f.name == name)
            .map(|(id, _)| id)
    }

    pub fn intern_signature(&mut self, signature: Signature) -> SigId {
        if let Some(&id) = self.sig_map.get(&signature) {
            id
        } else {
            let id = self.signatures.push(signature.clone());
            self.sig_map.insert(signature, id);
            id
        }
    }

    pub fn declare_function(&mut self, name: String, sig_id: SigId, linkage: Linkage) -> FuncId {
        self.functions.push(Function::new(name, sig_id, linkage))
    }

    pub fn add_global(&mut self, name: String, ty: Type, linkage: Linkage) {
        self.globals.push(Global { name, ty, linkage });
    }
}

#[derive(Debug, Default, Clone)]
pub struct Module {
    pub(crate) inner: Arc<ModuleData>,
}

impl Module {
    pub fn find_function_by_name(&self, name: &str) -> Option<FuncId> {
        self.inner.get_func_id(name)
    }

    pub fn get_function(&self, func_id: FuncId) -> &Function {
        &self.inner.functions[func_id]
    }

    pub fn get_function_name(&self, func_id: FuncId) -> &str {
        &self.inner.functions[func_id].name
    }

    pub fn get_signature(&self, sig_id: SigId) -> &Signature {
        &self.inner.signatures[sig_id]
    }
}

impl Deref for Module {
    type Target = ModuleData;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
