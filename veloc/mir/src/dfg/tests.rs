use super::*;
use crate::type_methods::VectorConstInfo;
use crate::{Type, Value};

#[test]
fn erasure_recycles_result_lists() {
    let mut dfg = DataFlowGraph::new();
    let a = dfg.create_inst(|writer: crate::InstWriter<'_>| writer.nop());
    let b = dfg.create_inst(|writer: crate::InstWriter<'_>| writer.nop());
    let list = dfg.append_results(a, &[Type::I32, Type::I64]);
    dfg.remove_insts(&[a]);
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
fn constants_own_and_share_their_bytes() {
    let mut source = DataFlowGraph::new();
    let ty = crate::Type::I8X16.as_vector().unwrap();
    let constant: crate::Constant = crate::VectorConst::dense(ty, alloc::vec![7u8; 16]).into();
    let literal = source.constant(constant.clone());
    assert_eq!(
        literal,
        source.constant(crate::VectorConst::dense(ty, alloc::vec![7u8; 16]).into())
    );

    let inst = source.writer().nop();
    source.remove_insts(&[inst]);
    let cloned = source.clone();
    assert_eq!(source.as_const(literal), cloned.as_const(literal));

    // Import content, not a source-function ID. The shared allocation survives
    // either function and needs no byte-pool remapping.
    let mut target = DataFlowGraph::new();
    let imported = target.constant(source.as_const(literal).unwrap().clone());
    drop(source);
    let bytes = target
        .as_const(imported)
        .unwrap()
        .as_vector()
        .unwrap()
        .bytes()
        .unwrap();
    let original = constant.as_vector().unwrap().bytes().unwrap();
    let copied = cloned
        .as_const(literal)
        .unwrap()
        .as_vector()
        .unwrap()
        .bytes()
        .unwrap();
    assert_eq!(bytes, &[7; 16]);
    assert_eq!(bytes.as_ptr(), original.as_ptr());
    assert_eq!(bytes.as_ptr(), copied.as_ptr());
}
