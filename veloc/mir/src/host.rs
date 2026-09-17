//! Rust implementations of the read-only type contracts declared in defs.
//!
//! Verification reads constants and signatures through the current function
//! and its containing module.
use crate::type_methods::{SignatureInfo, VerifyContextInfo};
use crate::{Function, ModuleData, SigId, Type, VectorConst};

impl SignatureInfo for veloc_types::Signature {
    fn params(&self) -> &[Type] {
        self.params()
    }
    fn returns(&self) -> &[Type] {
        self.returns()
    }
    fn types(&self) -> &[Type] {
        self.types()
    }
}

pub struct VerifyContext<'a> {
    module: &'a ModuleData,
    function: &'a Function,
}
impl<'a> VerifyContext<'a> {
    pub fn new(module: &'a ModuleData, function: &'a Function) -> Self {
        Self { module, function }
    }
}
impl VerifyContextInfo for VerifyContext<'_> {
    fn bytes(&self, value: VectorConst) -> Option<&[u8]> {
        value.bytes(self.function.dfg())
    }
    fn function_signature(&self, func: crate::FuncId) -> Option<&veloc_types::Signature> {
        self.signature(self.module.functions.get(func)?.signature)
    }
    fn current_signature(&self) -> Option<&veloc_types::Signature> {
        self.signature(self.function.signature)
    }
    fn signature(&self, sig: SigId) -> Option<&veloc_types::Signature> {
        self.module.signatures().get(sig)
    }
}
