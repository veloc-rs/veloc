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
    use crate::{InstDraft, Type, Value, ValueDef};

    #[test]
    fn result_moves_preserve_identity_and_recycle_lists() {
        let mut dfg = DataFlowGraph::new();
        let a = dfg.create_inst(InstDraft::nop());
        let b = dfg.create_inst(InstDraft::nop());
        let list = dfg.append_results(a, &[Type::I32, Type::I64]);
        let values = dfg.inst_results(a).to_vec();
        for _ in 0..100 {
            dfg.move_result(values[1], b);
            assert_eq!(dfg.inst_results(a), &values[..1]);
            assert_eq!(dfg.inst_results(b), &values[1..]);
            assert_eq!(dfg.value_def(values[1]), ValueDef::Inst(b));
            dfg.move_result(values[1], a);
            assert_eq!(dfg.inst_results(a), values);
            assert!(dfg.inst_results(b).is_empty());
        }
        dfg.remove_inst(a);
        let reused = dfg.append_results(b, &[Type::I32, Type::I64]);
        assert_eq!(
            list, reused,
            "erasure must return result storage to the pool"
        );
    }

    #[test]
    fn names_only_allocate_for_named_values() {
        let mut dfg = DataFlowGraph::new();
        let value = Value(1_000_000);
        dfg.set_value_name(value, "late");
        assert_eq!(dfg.value_names.len(), 1);
        assert_eq!(dfg.value_name(value), "late");
        assert_eq!(dfg.value_name(Value(0)), "");
        dfg.set_value_name(value, "");
        assert!(dfg.value_names.is_empty());
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
