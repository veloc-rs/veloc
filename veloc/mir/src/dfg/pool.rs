//! Typed access to interned instruction properties.
use super::DataFlowGraph;
use crate::inst::ConstantPoolId;
use alloc::vec::Vec;

impl ConstantPoolId {
    /// Intern immutable bytes. The pool and its lookup index share the payload.
    pub fn insert(dfg: &mut DataFlowGraph, value: Vec<u8>) -> Self {
        dfg.fields.intern(&value)
    }

    pub fn get(self, dfg: &DataFlowGraph) -> Option<&[u8]> {
        dfg.fields.constant(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Type, Value};

    #[test]
    fn erasure_recycles_result_lists() {
        let mut dfg = DataFlowGraph::new();
        let a = dfg.create_inst(|writer: crate::InstWriter<'_>| writer.nop());
        let b = dfg.create_inst(|writer: crate::InstWriter<'_>| writer.nop());
        let list = dfg.append_results(a, &[Type::I32, Type::I64]);
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
    fn shared_constants_survive_instruction_erasure_and_dfg_clone() {
        let mut dfg = DataFlowGraph::new();
        let id = ConstantPoolId::insert(&mut dfg, vec![7; 16]);
        assert_eq!(id, ConstantPoolId::insert(&mut dfg, vec![7; 16]));
        let ty = crate::Type::I8X16.as_vector().unwrap();
        let a = dfg.writer().vconst(crate::VectorConst::dense(ty, id));
        let b = dfg.writer().copy(a);
        let cloned = dfg.clone();
        dfg.remove_inst(a);
        assert_eq!(id.get(&dfg), Some(&[7; 16][..]));
        dfg.remove_inst(b);
        assert_eq!(id.get(&dfg), id.get(&cloned));
        assert_eq!(
            id.get(&dfg).unwrap().as_ptr(),
            id.get(&cloned).unwrap().as_ptr()
        );
    }
}
