//! Typed access to interned instruction properties.
use super::DataFlowGraph;
use crate::inst::ConstantPoolId;
use alloc::sync::Arc;
use alloc::vec::Vec;

impl ConstantPoolId {
    /// Intern immutable bytes. The pool and its lookup index share the payload.
    pub fn insert(dfg: &mut DataFlowGraph, value: Vec<u8>) -> Self {
        if let Some(&id) = dfg.constant_pool_map.get(value.as_slice()) {
            return id;
        }
        let bytes: Arc<[u8]> = value.into();
        let id = dfg.constant_pool.push(Arc::clone(&bytes));
        dfg.constant_pool_map.insert(bytes, id);
        id
    }

    pub fn get(self, dfg: &DataFlowGraph) -> Option<&[u8]> {
        dfg.constant_pool.get(self).map(AsRef::as_ref)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_pool_shares_backing_storage_with_its_index() {
        let mut dfg = DataFlowGraph::new();
        let id = ConstantPoolId::insert(&mut dfg, vec![1, 2, 3]);
        assert_eq!(id, ConstantPoolId::insert(&mut dfg, vec![1, 2, 3]));
        let (key, &indexed_id) = dfg.constant_pool_map.get_key_value(&[1, 2, 3][..]).unwrap();
        assert_eq!(indexed_id, id);
        assert!(Arc::ptr_eq(key, &dfg.constant_pool[id]));
        let cloned = dfg.clone();
        assert!(Arc::ptr_eq(
            &dfg.constant_pool[id],
            &cloned.constant_pool[id]
        ));
        assert_eq!(id.get(&cloned), Some(&[1, 2, 3][..]));
    }
}
