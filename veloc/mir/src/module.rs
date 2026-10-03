use crate::function::{FuncBody, FuncDecl, FunctionRef};
use crate::types::{FuncId, SigId, Signature, Type};
use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
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
    pub data: Option<GlobalData>,
}

/// A data definition and relocations resolved by the native linker.
#[derive(Debug, Clone)]
pub struct GlobalData {
    pub bytes: Vec<u8>,
    pub align: u64,
    pub writable: bool,
    pub relocations: Vec<DataRelocation>,
}

#[derive(Debug, Clone)]
pub struct DataRelocation {
    pub offset: u64,
    pub symbol: String,
    pub addend: i64,
}

/// Owned MIR data. Cloning copies declarations and function bodies; callers
/// that need shared ownership should hold an `Arc<Module>` instead.
/// Construction and sharing do not imply validation.
#[derive(Debug, Default, Clone)]
pub struct Module {
    pub(crate) decls: PrimaryMap<FuncId, FuncDecl>,
    pub(crate) bodies: SecondaryMap<FuncId, Option<Box<FuncBody>>>,
    types: Arc<veloc_types::TypeContext>,
    pub(crate) globals: Vec<Global>,
}

impl Module {
    pub fn decls(&self) -> &PrimaryMap<FuncId, FuncDecl> {
        &self.decls
    }

    pub fn globals(&self) -> &[Global] {
        &self.globals
    }

    pub fn body_mut(&mut self, id: FuncId) -> Option<&mut FuncBody> {
        assert!(self.decls.get(id).is_some(), "unknown function");
        self.bodies[id].as_deref_mut()
    }

    /// Install a transformed definition for a newly declared function. Module
    /// transformations can construct an owned body without an SSA builder.
    pub fn define_function(&mut self, id: FuncId, body: FuncBody) {
        let declaration = &self.decls[id];
        assert!(self.bodies[id].is_none(), "function already defined");
        assert!(
            body.dfg()
                .block_params(body.entry_block())
                .iter()
                .map(|&v| body.dfg().value_type(v))
                .eq(self.signatures()[declaration.signature]
                    .params()
                    .iter()
                    .copied()),
            "entry parameters must match the declaration"
        );
        self.bodies[id] = Some(Box::new(body));
    }

    /// Iterate existing definitions without exposing body insertion or removal.
    pub fn bodies_mut(&mut self) -> impl Iterator<Item = (FuncId, &mut FuncBody)> {
        self.bodies
            .iter_mut()
            .filter_map(|(id, body)| body.as_deref_mut().map(|body| (id, body)))
    }

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

    pub fn find_function(&self, name: &str) -> Option<FuncId> {
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
        let body = self.bodies[id].insert(Box::new(FuncBody::new(params)));
        (&self.decls, self.types.signatures(), body)
    }

    pub fn add_global(&mut self, name: String, ty: Type, linkage: Linkage) -> crate::GlobalId {
        let id = crate::GlobalId(self.globals.len().try_into().expect("too many globals"));
        self.globals.push(Global {
            name,
            ty,
            linkage,
            data: None,
        });
        id
    }

    pub fn define_global(&mut self, id: crate::GlobalId, data: GlobalData) {
        assert!(data.align.is_power_of_two(), "invalid data alignment");
        let global = &mut self.globals[id.0 as usize];
        if global.linkage == Linkage::Import {
            global.linkage = Linkage::Export;
        }
        global.data = Some(data);
    }
}
