//! Stable block and instruction order, independent of storage and CFG.
pub type Layout = veloc_collections::EntityLayout<crate::Block, crate::Inst>;
pub type InstOrder = veloc_collections::InstOrder<crate::Inst>;
