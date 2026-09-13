//! Rust implementations of the read-only type contracts declared in defs.
//!
//! Callers construct and pass the concrete context directly. Constants need only
//! a DFG; instruction verification can also inspect module signatures.
use crate::type_methods::{ConstContextInfo, SignatureInfo, VerifyContextInfo};
use crate::{ModuleData, SigId, Type, VectorConst, dfg::DataFlowGraph};

pub struct ConstContext<'a> {
    dfg: &'a DataFlowGraph,
}
impl<'a> ConstContext<'a> {
    pub fn new(dfg: &'a DataFlowGraph) -> Self {
        Self { dfg }
    }
}
impl ConstContextInfo for ConstContext<'_> {
    fn bytes(&self, value: VectorConst) -> Option<&[u8]> {
        value.bytes(self.dfg)
    }
}

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
    signature: SigId,
}
impl<'a> VerifyContext<'a> {
    pub fn new(module: &'a ModuleData, signature: SigId) -> Self {
        Self { module, signature }
    }
}
impl VerifyContextInfo for VerifyContext<'_> {
    fn function_signature(&self, func: crate::FuncId) -> Option<&veloc_types::Signature> {
        self.signature(self.module.functions.get(func)?.signature)
    }
    fn current_signature(&self) -> Option<&veloc_types::Signature> {
        self.signature(self.signature)
    }
    fn signature(&self, sig: SigId) -> Option<&veloc_types::Signature> {
        self.module.signatures.get(sig)
    }
}
