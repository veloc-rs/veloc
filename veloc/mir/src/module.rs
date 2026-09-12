use crate::function::Function;
use crate::types::{FuncId, SigId, Signature, Type};
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Deref;
use cranelift_entity::PrimaryMap;
use hashbrown::HashSet;

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
    pub functions: PrimaryMap<FuncId, Function>,
    pub signatures: veloc_types::Signatures,
    pub globals: Vec<Global>,
}

impl ModuleData {
    /// Compare structural types from different module contexts. Compact data
    /// types keep the constant-time path; interned IDs are never compared across
    /// contexts. The worklist also terminates on malformed cyclic signatures.
    pub fn type_eq(&self, lhs: Type, other: &Self, rhs: Type) -> bool {
        if lhs.is_compact() || rhs.is_compact() {
            return lhs == rhs;
        }
        let (Some((lhs, a)), Some((rhs, b))) = (lhs.as_callable(), rhs.as_callable()) else {
            return false;
        };
        a == b && self.signature_eq(lhs, other, rhs)
    }

    pub fn signature_eq(&self, lhs: SigId, other: &Self, rhs: SigId) -> bool {
        if core::ptr::eq(self, other) {
            return lhs == rhs && self.signatures.get(lhs).is_some();
        }
        let (Some(a), Some(b)) = (self.signatures.get(lhs), other.signatures.get(rhs)) else {
            return false;
        };
        // Ordinary function signatures do not need structural traversal or any
        // allocation, including cross-module indirect calls.
        if a.types().iter().all(|ty| ty.is_compact()) {
            return a == b;
        }
        let mut pending = alloc::vec![(lhs, rhs)];
        let mut seen = HashSet::new();
        while let Some((lhs, rhs)) = pending.pop() {
            if !seen.insert((lhs, rhs)) {
                continue;
            }
            let (Some(a), Some(b)) = (self.signatures.get(lhs), other.signatures.get(rhs)) else {
                return false;
            };
            if a.call_conv != b.call_conv
                || a.params().len() != b.params().len()
                || a.returns().len() != b.returns().len()
            {
                return false;
            }
            for (&lhs, &rhs) in a.types().iter().zip(b.types()) {
                if lhs.is_compact() || rhs.is_compact() {
                    if lhs != rhs {
                        return false;
                    }
                } else {
                    let (Some((lhs, a)), Some((rhs, b))) = (lhs.as_callable(), rhs.as_callable())
                    else {
                        return false;
                    };
                    if a != b {
                        return false;
                    }
                    pending.push((lhs, rhs));
                }
            }
        }
        true
    }

    pub fn get_func_id(&self, name: &str) -> Option<FuncId> {
        self.functions
            .iter()
            .find(|(_, f)| f.name == name)
            .map(|(id, _)| id)
    }

    pub fn intern_signature(&mut self, signature: Signature) -> SigId {
        self.signatures.insert(signature)
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
    pub fn new(data: ModuleData) -> Self {
        Self {
            inner: Arc::new(data),
        }
    }

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
