//! Typed access to interned instruction properties.
use super::DataFlowGraph;
use crate::inst::{
    ConstantPoolId, PtrIndexImm, PtrIndexImmId, VectorExtData, VectorExtId, VectorMemExtId,
    VectorMemOptions,
};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::hash::Hash;
use cranelift_entity::{EntityRef, PrimaryMap};
use hashbrown::HashMap;

/// A handle's storage contract, independent of the DFG's concrete pool names.
///
/// Insertion consumes owned data; reads borrow a view without cloning. For byte
/// constants the input is `Vec<u8>` and the view is `[u8]`.
pub trait PoolKey: Copy {
    type Input;
    type View: ?Sized;

    fn insert(dfg: &mut DataFlowGraph, value: Self::Input) -> Self;
    fn get(self, dfg: &DataFlowGraph) -> Option<&Self::View>;
}

fn intern<K: EntityRef, T: Clone + Eq + Hash>(
    pool: &mut PrimaryMap<K, T>,
    index: &mut HashMap<T, K>,
    value: T,
) -> K {
    if let Some(&id) = index.get(&value) {
        return id;
    }
    let id = pool.push(value.clone());
    index.insert(value, id);
    id
}

impl PoolKey for PtrIndexImmId {
    type Input = PtrIndexImm;
    type View = PtrIndexImm;

    fn insert(dfg: &mut DataFlowGraph, value: Self::Input) -> Self {
        intern(&mut dfg.ptr_imm_pool, &mut dfg.ptr_imm_map, value)
    }

    fn get(self, dfg: &DataFlowGraph) -> Option<&Self::View> {
        dfg.ptr_imm_pool.get(self)
    }
}

impl PoolKey for VectorExtId {
    type Input = VectorExtData;
    type View = VectorExtData;

    fn insert(dfg: &mut DataFlowGraph, value: Self::Input) -> Self {
        intern(&mut dfg.vector_ext_pool, &mut dfg.vector_ext_map, value)
    }

    fn get(self, dfg: &DataFlowGraph) -> Option<&Self::View> {
        dfg.vector_ext_pool.get(self)
    }
}

impl PoolKey for VectorMemExtId {
    type Input = VectorMemOptions;
    type View = VectorMemOptions;

    fn insert(dfg: &mut DataFlowGraph, value: Self::Input) -> Self {
        intern(
            &mut dfg.vector_mem_ext_pool,
            &mut dfg.vector_mem_ext_map,
            value,
        )
    }

    fn get(self, dfg: &DataFlowGraph) -> Option<&Self::View> {
        dfg.vector_mem_ext_pool.get(self)
    }
}

impl PoolKey for ConstantPoolId {
    type Input = Vec<u8>;
    type View = [u8];

    fn insert(dfg: &mut DataFlowGraph, value: Self::Input) -> Self {
        if let Some(&id) = dfg.constant_pool_map.get(value.as_slice()) {
            return id;
        }
        let bytes: Arc<[u8]> = value.into();
        let id = dfg.constant_pool.push(Arc::clone(&bytes));
        dfg.constant_pool_map.insert(bytes, id);
        id
    }

    fn get(self, dfg: &DataFlowGraph) -> Option<&Self::View> {
        dfg.constant_pool.get(self).map(AsRef::as_ref)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Value;
    use core::fmt::Debug;

    fn check<K>(dfg: &mut DataFlowGraph, input: K::Input, expected: &K::View)
    where
        K: PoolKey + Debug + Eq,
        K::Input: Clone,
        K::View: Debug + Eq,
    {
        let id = K::insert(dfg, input.clone());
        assert_eq!(id, K::insert(dfg, input));
        assert_eq!(id.get(dfg), Some(expected));
        assert!(core::ptr::eq(id.get(dfg).unwrap(), id.get(dfg).unwrap()));
    }

    #[test]
    fn property_handles_share_the_intern_and_borrow_contract() {
        let mut dfg = DataFlowGraph::new();
        let imm = PtrIndexImm {
            offset: -7,
            scale: 4,
        };
        check::<PtrIndexImmId>(&mut dfg, imm, &imm);
        let ext = VectorExtData {
            mask: Value(3),
            evl: Some(Value(4)),
        };
        check::<VectorExtId>(&mut dfg, ext, &ext);
        let mem = VectorMemOptions {
            mask: Some(Value(3)),
            ..Default::default()
        };
        check::<VectorMemExtId>(&mut dfg, mem, &mem);
        check::<ConstantPoolId>(&mut dfg, vec![0, 0xff, 1], &[0, 0xff, 1]);
        check::<ConstantPoolId>(&mut dfg, vec![], &[]);
        assert!(PtrIndexImmId(999).get(&dfg).is_none());
        assert!(VectorExtId(999).get(&dfg).is_none());
        assert!(VectorMemExtId(999).get(&dfg).is_none());
        assert!(ConstantPoolId(999).get(&dfg).is_none());
    }

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
