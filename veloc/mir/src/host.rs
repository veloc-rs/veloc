//! Rust implementations of the read-only capabilities declared in defs.
//!
//! The generated traits own signatures; this module owns access to MIR storage.
//! Queries are deterministic for an unchanged context and do not mutate the IR.
use crate::{ModuleData, SigId, Type, VectorConst, dfg::DataFlowGraph};

pub mod traits {
    include!(concat!(env!("OUT_DIR"), "/host_traits.rs"));
}
use traits::{Constants, Module, Types};

pub(crate) struct Context<'a, M = ()> {
    dfg: &'a DataFlowGraph,
    module: M,
}

pub(crate) struct ModuleState<'a> {
    data: &'a ModuleData,
    signature: SigId,
}

impl<'a> Context<'a> {
    pub(crate) fn new(dfg: &'a DataFlowGraph) -> Self {
        Self { dfg, module: () }
    }

    pub(crate) fn with_module(
        self,
        data: &'a ModuleData,
        signature: SigId,
    ) -> Context<'a, ModuleState<'a>> {
        Context {
            dfg: self.dfg,
            module: ModuleState { data, signature },
        }
    }
}

impl<M> Constants for Context<'_, M> {
    fn is_dense(&self, value: VectorConst) -> bool {
        value.is_dense()
    }
    fn bytes(&self, value: VectorConst) -> Option<&[u8]> {
        value.bytes(self.dfg)
    }
}

impl<M> Types for Context<'_, M> {
    fn signature(&self, ty: Type) -> Option<SigId> {
        ty.as_callable().map(|(sig, _)| sig)
    }
}

impl Module for Context<'_, ModuleState<'_>> {
    fn signature(&self, func: crate::FuncId) -> Option<SigId> {
        self.module.data.functions.get(func).map(|f| f.signature)
    }
    fn current_signature(&self) -> SigId {
        self.module.signature
    }
    fn params(&self, sig: SigId) -> Option<&[Type]> {
        self.module.data.signatures.get(sig).map(|s| s.params())
    }
    fn returns(&self, sig: SigId) -> Option<&[Type]> {
        self.module.data.signatures.get(sig).map(|s| s.returns())
    }
}
