use crate::function::{FuncBody, FuncDecl, FunctionRef};
use crate::types::{FuncId, SigId, Signature, Type};
use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Deref;
use cranelift_entity::{PrimaryMap, SecondaryMap};

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

impl Linkage {
    pub fn from_mnemonic(mnemonic: &str) -> Option<Self> {
        match mnemonic {
            "local" => Some(Self::Local),
            "export" => Some(Self::Export),
            "import" => Some(Self::Import),
            _ => None,
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
    pub decls: PrimaryMap<FuncId, FuncDecl>,
    pub bodies: SecondaryMap<FuncId, Option<Box<FuncBody>>>,
    types: Arc<veloc_types::TypeContext>,
    pub globals: Vec<Global>,
}

impl ModuleData {
    /// Bind a module to an existing immutable type universe.
    pub fn with_types(types: Arc<veloc_types::TypeContext>) -> Self {
        Self {
            types,
            decls: PrimaryMap::new(),
            bodies: SecondaryMap::new(),
            globals: Vec::new(),
        }
    }

    pub fn types(&self) -> &veloc_types::TypeContext {
        &self.types
    }

    /// Share type identity without sharing mutable function bodies.
    pub fn shared_types(&self) -> Arc<veloc_types::TypeContext> {
        Arc::clone(&self.types)
    }

    /// Detach a shared context before extending it. Existing IDs remain valid
    /// in this module; newly assigned IDs belong to the detached context only.
    pub fn types_mut(&mut self) -> &mut veloc_types::TypeContext {
        Arc::make_mut(&mut self.types)
    }

    pub fn signatures(&self) -> &veloc_types::Signatures {
        self.types.signatures()
    }

    pub fn get_func_id(&self, name: &str) -> Option<FuncId> {
        self.decls
            .iter()
            .find(|(_, f)| f.name == name)
            .map(|(id, _)| id)
    }

    pub fn intern_signature(&mut self, signature: Signature) -> SigId {
        self.types_mut().insert_signature(signature)
    }

    pub fn declare_function(&mut self, name: String, signature: SigId, linkage: Linkage) -> FuncId {
        self.decls.push(FuncDecl {
            name,
            signature,
            linkage,
        })
    }

    pub fn function(&self, id: FuncId) -> FunctionRef<'_> {
        FunctionRef {
            decl: &self.decls[id],
            body: self.bodies[id].as_deref(),
        }
    }

    pub fn functions(&self) -> impl ExactSizeIterator<Item = (FuncId, FunctionRef<'_>)> {
        self.decls.iter().map(|(id, decl)| {
            (
                id,
                FunctionRef {
                    decl,
                    body: self.bodies[id].as_deref(),
                },
            )
        })
    }

    /// Create a definition and borrow its metadata independently of its body.
    pub(crate) fn define_body(
        &mut self,
        id: FuncId,
    ) -> (
        &PrimaryMap<FuncId, FuncDecl>,
        &veloc_types::Signatures,
        &mut FuncBody,
    ) {
        assert!(self.decls.get(id).is_some(), "unknown function");
        assert!(self.bodies[id].is_none(), "function already defined");
        let params = self.types.signatures()[self.decls[id].signature].params();
        let body = self.bodies[id].get_or_insert_with(|| Box::new(FuncBody::new(params)));
        (&self.decls, self.types.signatures(), body)
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
    pub fn new(data: ModuleData) -> Self {
        Self {
            inner: Arc::new(data),
        }
    }

    pub fn find_function_by_name(&self, name: &str) -> Option<FuncId> {
        self.inner.get_func_id(name)
    }

    pub fn get_function(&self, func_id: FuncId) -> FunctionRef<'_> {
        self.inner.function(func_id)
    }

    pub fn get_function_name(&self, func_id: FuncId) -> &str {
        &self.inner.decls[func_id].name
    }

    pub fn get_signature(&self, sig_id: SigId) -> &Signature {
        &self.inner.signatures()[sig_id]
    }
}

impl Deref for Module {
    type Target = ModuleData;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
