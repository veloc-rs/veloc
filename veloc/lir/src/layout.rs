//! Stable machine block and instruction order, independent of storage and CFG.
pub type Layout = veloc_collections::EntityLayout<crate::BlockId, crate::InstId>;
