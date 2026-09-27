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
    let constant: crate::Constant =
        crate::VectorConst::dense(ty, (0u8..16).collect::<alloc::vec::Vec<_>>()).into();
    let literal = source.constant(constant.clone());
    assert_eq!(
        literal,
        source.constant(
            crate::VectorConst::dense(ty, (0u8..16).collect::<alloc::vec::Vec<_>>()).into()
        )
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
    assert_eq!(bytes, &(0u8..16).collect::<alloc::vec::Vec<_>>());
    assert_eq!(bytes.as_ptr(), original.as_ptr());
    assert_eq!(bytes.as_ptr(), copied.as_ptr());
}

#[test]
fn constant_identity_preserves_type_and_exact_bits() {
    let mut dfg = DataFlowGraph::new();
    let scalar = crate::ScalarConst::from(7i32);
    let a = dfg.constant(scalar.into());
    assert_eq!(a, dfg.constant(scalar.into()));
    assert_ne!(a, dfg.constant(crate::ScalarConst::from(7i64).into()));

    let ty = Type::I32X4.as_vector().unwrap();
    let dense = crate::VectorConst::dense(ty, [7i32.to_le_bytes(); 4].concat());
    let splat = crate::VectorConst::splat(scalar, 4, false).unwrap();
    assert_eq!(dfg.constant(dense.into()), dfg.constant(splat.into()));

    let positive = crate::ScalarConst::from_bits(Type::F64, 0).unwrap();
    let negative = crate::ScalarConst::from_bits(Type::F64, 1u64 << 63).unwrap();
    assert_ne!(dfg.constant(positive.into()), dfg.constant(negative.into()));
    for bits in [0x7ff8000000000001, 0x7ff8000000000002] {
        let nan = crate::ScalarConst::from_bits(Type::F64, bits).unwrap();
        let value = dfg.constant(nan.into());
        assert_eq!(value, dfg.constant(nan.into()));
        assert_eq!(dfg.as_scalar_const(value).unwrap().to_bits(), bits);
    }
}
