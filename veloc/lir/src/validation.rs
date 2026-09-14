//! Type contracts are generated separately from construction and instruction access.
pub use veloc_mir::inst::TypeError;

include!(concat!(env!("OUT_DIR"), "/type_rules.rs"));
