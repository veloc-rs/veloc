#![no_std]
extern crate alloc;

pub mod dataflow;
pub mod graph;
pub mod liveness;
pub mod manager;

pub use liveness::*;
pub use manager::*;

/// MIR dominance analysis; the graph algorithm is shared with IR validation.
pub type Dominators = graph::DominatorTree<veloc_mir::Block>;
