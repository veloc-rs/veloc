//! Typed access to interned instruction properties.
use super::DataFlowGraph;
use crate::inst::{
    ConstantPoolData, ConstantPoolId, PtrIndexImm, PtrIndexImmId, VectorExtData, VectorExtId,
    VectorMemExtId, VectorMemOptions,
};
use alloc::vec::Vec;
use core::hash::Hash;
use cranelift_entity::{EntityRef, PrimaryMap};
use hashbrown::HashMap;

/// A handle's storage contract, independent of the DFG's concrete pool names.
///
/// Insertion consumes owned data; reads borrow a view without cloning. For byte
/// constants the input is `Vec<u8>` and the view is `[u8]`, hiding the pool's enum.
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
        intern(
            &mut dfg.constant_pool,
            &mut dfg.constant_pool_map,
            ConstantPoolData::Bytes(value),
        )
    }

    fn get(self, dfg: &DataFlowGraph) -> Option<&Self::View> {
        let ConstantPoolData::Bytes(bytes) = dfg.constant_pool.get(self)?;
        Some(bytes)
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
    fn existing_helpers_and_typed_handles_use_the_same_pools() {
        let mut dfg = DataFlowGraph::new();
        let imm = PtrIndexImm {
            offset: 3,
            scale: 2,
        };
        let id = PtrIndexImmId::insert(&mut dfg, imm);
        assert_eq!(id, dfg.make_ptr_imm(3, 2));
        assert_eq!(id.get(&dfg), dfg.ptr_imm(id));
        let bytes = vec![1, 2, 3];
        let id = ConstantPoolId::insert(&mut dfg, bytes.clone());
        assert_eq!(
            id,
            dfg.make_constant_pool_data(ConstantPoolData::Bytes(bytes))
        );
        let ConstantPoolData::Bytes(stored) = dfg.constant_pool_data(id).unwrap();
        assert!(core::ptr::eq(id.get(&dfg).unwrap(), stored.as_slice()));
    }
}
