mod allocation;
mod edges;
mod linear_scan;

pub use allocation::{Allocation, InstAllocation, Transfer};
pub use edges::EdgeAllocation;

pub use linear_scan::RegisterAllocator;
