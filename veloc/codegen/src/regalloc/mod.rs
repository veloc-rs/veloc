mod allocation;
mod edges;
mod linear_scan;

pub use allocation::{Allocation, InstAllocation};
pub use edges::EdgeAllocation;
pub mod regbank_select;

pub use linear_scan::RegisterAllocator;
pub use regbank_select::RegisterBankSelector;
