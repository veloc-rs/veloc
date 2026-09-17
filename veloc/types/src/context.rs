//! Type identity and cross-context migration, independent of any IR module.
//! Compact types carry their own identity; callable IDs belong to this context.
//! Entries are immutable and insertion requires exclusive access. An Arc can
//! share a read-only context across modules or compilation tasks without locks.
//! Cloning creates an independent universe preserving the existing ID prefix;
//! additions to clones do not create interchangeable IDs. Import at boundaries.
//! Forward references are permitted during construction; validation/import checks
//! the completed graph. Recursive type definitions are not yet supported.
use crate::{CallConv, SigId, Signature, SignatureError, Signatures, Type};
use alloc::vec::Vec;

#[derive(Debug, Default, Clone)]
pub struct TypeContext {
    signatures: Signatures,
}

impl TypeContext {
    pub fn signatures(&self) -> &Signatures {
        &self.signatures
    }

    pub fn insert_signature(&mut self, signature: Signature) -> SigId {
        self.signatures.insert(signature)
    }

    pub fn intern_signature(
        &mut self,
        params: &[Type],
        returns: &[Type],
        call_conv: CallConv,
    ) -> SigId {
        self.signatures.intern(params, returns, call_conv)
    }

    /// Import the source context and return its signature-ID mapping.
    /// Nested references are remapped before their containing signatures.
    /// Validation happens before any destination entries are inserted.
    pub fn import(&mut self, source: &Self) -> Result<Vec<SigId>, SignatureError> {
        self.signatures.import(&source.signatures)
    }
}
