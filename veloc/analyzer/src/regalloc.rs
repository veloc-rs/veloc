use crate::liveness::Liveness;
use hashbrown::HashMap;
use veloc_ir::{Function, Value};

/// Register assignment result
pub struct RegAlloc {
    pub assignments: HashMap<Value, u16>,
    pub num_registers: usize,
}

/// Generic Linear Scan Register Allocator
pub struct LinearScan {}

impl LinearScan {
    pub fn new() -> Self {
        Self {}
    }

    pub fn allocate(&self, _func: &Function, _liveness: &Liveness) -> RegAlloc {
        // Placeholder for future robust Linear Scan implementation
        RegAlloc {
            assignments: HashMap::new(),
            num_registers: 0,
        }
    }
}
