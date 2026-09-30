mod allocation;
pub(crate) mod constraints;
mod edges;
mod linear_scan;
mod moves;
mod operands;

pub use allocation::{Allocation, InstAllocation, Transfer};
pub use edges::EdgeAllocation;

pub use linear_scan::RegisterAllocator;
