//! Function pointer encoding/decoding for VM
//!
//! This module provides functionality to encode function references (both host
//! and interpreter functions) into tagged pointers that can be passed around
//! and later decoded.

use crate::host::HostFuncId;
use cranelift_entity::EntityRef;
use veloc_ir::{FuncId, ModuleId};

/// Target of an import resolution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ImportTarget {
    /// Reference to a function in another module
    Module(ModuleId, FuncId),
    /// Reference to a host function
    Host(HostFuncId),
    /// Unresolved import or internal function
    #[default]
    None,
}

/// A tagged function pointer that can represent either a host function
/// or an interpreter function.
///
/// Encoding scheme:
/// - Bit 0 = 1: Interpreter function, bits 1-32 = func_id, bits 33+ = module_id
/// - Bit 0 = 0, Bit 1 = 1: Host function, bits 2+ = host_func_id
/// - Bit 0-1 = 0: Null/invalid
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct VMFuncPointer(pub usize);

impl VMFuncPointer {
    /// Create a pointer from a host function ID
    pub fn from_host(id: HostFuncId) -> Self {
        Self(((id.index() as usize) << 2) | 2)
    }

    /// Create a pointer from an interpreter function reference
    pub fn from_interpreter(mid: ModuleId, fid: FuncId) -> Self {
        let val = ((mid.index() as u64) << 33) | ((fid.index() as u64) << 1) | 1;
        Self(val as usize)
    }

    /// Get the raw pointer value
    pub fn as_ptr(&self) -> *const u8 {
        self.0 as *const u8
    }

    /// Decode the pointer to an import target
    pub fn decode(&self) -> Option<ImportTarget> {
        if self.0 & 1 == 1 {
            let mid = (self.0 >> 33) as u32;
            let fid = ((self.0 >> 1) & 0xFFFFFFFF) as u32;
            Some(ImportTarget::Module(
                ModuleId::new(mid as usize),
                FuncId::new(fid as usize),
            ))
        } else if self.0 & 3 == 2 {
            Some(ImportTarget::Host(HostFuncId::new((self.0 >> 2) as usize)))
        } else {
            None
        }
    }
}

impl core::fmt::Debug for VMFuncPointer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.decode() {
            Some(ImportTarget::Module(mid, fid)) => {
                write!(f, "VMFuncPointer(Module({}, {}))", mid.index(), fid.index())
            }
            Some(ImportTarget::Host(hid)) => {
                write!(f, "VMFuncPointer(Host({}))", hid.index())
            }
            Some(ImportTarget::None) => write!(f, "VMFuncPointer(None)"),
            None => write!(f, "VMFuncPointer(Invalid)",),
        }
    }
}
