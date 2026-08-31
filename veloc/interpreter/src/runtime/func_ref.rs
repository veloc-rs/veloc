//! Opaque function-reference handles used by the VM.
//!
//! The WebAssembly runtime currently stores native entry points and interpreter
//! references in the same pointer-sized field. Interpreter references are
//! therefore represented by tagged, non-dereferenceable handles and resolved
//! through `Program`; module/function IDs are never encoded directly.

use crate::host::HostFuncId;
use core::num::NonZeroUsize;
use veloc_ir::{FuncId, ModuleId};

/// Fully classified target of a guest call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallTarget {
    /// A compiled bytecode function. Local calls use the current module ID.
    Bytecode(ModuleId, FuncId),
    /// Reference to a host function.
    Host(HostFuncId),
}

/// An opaque index into `Program`'s function-reference table.
///
/// The low bits distinguish these handles from aligned native entry points.
/// The value is never a dereferenceable address.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FunctionRef(NonZeroUsize);

impl FunctionRef {
    const TAG: usize = 0b10;
    const TAG_MASK: usize = 0b11;

    pub(crate) fn from_index(index: usize) -> Option<Self> {
        let one_based = index.checked_add(1)?;
        let address = one_based.checked_shl(2)? | Self::TAG;
        NonZeroUsize::new(address).map(Self)
    }

    pub(crate) fn index_from_address(address: usize) -> Option<usize> {
        if address & Self::TAG_MASK != Self::TAG {
            return None;
        }
        (address >> 2).checked_sub(1)
    }

    /// Integer representation used in VM ABI storage.
    pub fn address(self) -> usize {
        self.0.get()
    }

    /// Produce the opaque pointer representation required by the current VM ABI.
    /// This pointer must never be dereferenced.
    pub fn as_opaque_ptr(self) -> *const u8 {
        core::ptr::without_provenance(self.address())
    }
}

impl core::fmt::Debug for FunctionRef {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("FunctionRef")
            .field(&Self::index_from_address(self.address()).unwrap())
            .finish()
    }
}
