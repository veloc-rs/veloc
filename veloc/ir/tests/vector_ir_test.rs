//! Vector IR Integration Tests

use veloc_ir::{
    CallConv, Linkage, MemFlags, Opcode,
    builder::ModuleBuilder,
    types::{ScalarType, Type},
};

#[test]
fn test_simple_vector_add_fixed() {
    let mut mb = ModuleBuilder::new();
    let sig_id = mb.make_signature(vec![], vec![], CallConv::SystemV);
    let func_id = mb.declare_function("test_vadd".to_string(), sig_id, Linkage::Export);
    let mut builder = mb.builder(func_id);
    builder.init_entry_block();

    let v4i32 = Type::new_vector(ScalarType::I32, 4, false);

    let scalar_a = builder.ins().i32const(1);
    let scalar_b = builder.ins().i32const(2);
    let vec_a = builder.ins().splat(scalar_a, v4i32);
    let vec_b = builder.ins().splat(scalar_b, v4i32);
    let vec_c = builder.ins().iadd(vec_a, vec_b);

    assert_eq!(builder.value_type(vec_c), v4i32);

    builder.ins().ret(&[vec_c]);
    builder.seal_all_blocks();
}

#[test]
fn test_vector_splat() {
    let mut mb = ModuleBuilder::new();
    let sig_id = mb.make_signature(vec![], vec![], CallConv::SystemV);
    let func_id = mb.declare_function("test_splat".to_string(), sig_id, Linkage::Export);
    let mut builder = mb.builder(func_id);
    builder.init_entry_block();

    let v8i32 = Type::new_vector(ScalarType::I32, 8, false);
    let v4f64 = Type::new_vector(ScalarType::F64, 4, false);

    let scalar_i = builder.ins().i32const(42);
    let scalar_f = builder.ins().f64const(3.14);

    let vec_i = builder.ins().splat(scalar_i, v8i32);
    let vec_f = builder.ins().splat(scalar_f, v4f64);

    assert_eq!(builder.value_type(vec_i), v8i32);
    assert_eq!(builder.value_type(vec_f), v4f64);

    builder.ins().ret(&[]);
    builder.seal_all_blocks();
}

#[test]
fn test_vector_reduction_ops() {
    let mut mb = ModuleBuilder::new();
    let sig_id = mb.make_signature(vec![], vec![], CallConv::SystemV);
    let func_id = mb.declare_function("test_reduction".to_string(), sig_id, Linkage::Export);
    let mut builder = mb.builder(func_id);
    builder.init_entry_block();

    let v4f32 = Type::new_vector(ScalarType::F32, 4, false);
    let scalar = builder.ins().f32const(1.0);
    let vec = builder.ins().splat(scalar, v4f32);

    let sum = builder.ins().reduce_sum(vec, Type::F32);
    let add = builder.ins().reduce_add(vec, Type::F32);
    let min = builder.ins().reduce_min(vec, Type::F32);
    let max = builder.ins().reduce_max(vec, Type::F32);

    assert_eq!(builder.value_type(sum), Type::F32);
    assert_eq!(builder.value_type(add), Type::F32);
    assert_eq!(builder.value_type(min), Type::F32);
    assert_eq!(builder.value_type(max), Type::F32);

    builder.ins().ret(&[]);
    builder.seal_all_blocks();
}

#[test]
fn test_vector_extract_insert() {
    let mut mb = ModuleBuilder::new();
    let sig_id = mb.make_signature(vec![], vec![], CallConv::SystemV);
    let func_id = mb.declare_function("test_extract_insert".to_string(), sig_id, Linkage::Export);
    let mut builder = mb.builder(func_id);
    builder.init_entry_block();

    let v4i32 = Type::new_vector(ScalarType::I32, 4, false);
    let scalar = builder.ins().i32const(10);
    let vec = builder.ins().splat(scalar, v4i32);

    let extracted = builder.ins().extract_element(vec, 0, Type::I32);
    assert_eq!(builder.value_type(extracted), Type::I32);

    let new_val = builder.ins().i32const(20);
    let inserted = builder.ins().insert_element(vec, new_val, 1, v4i32);
    assert_eq!(builder.value_type(inserted), v4i32);

    builder.ins().ret(&[]);
    builder.seal_all_blocks();
}

#[test]
fn test_vector_with_mask_evl() {
    let mut mb = ModuleBuilder::new();
    let sig_id = mb.make_signature(vec![], vec![], CallConv::SystemV);
    let func_id = mb.declare_function("test_masked".to_string(), sig_id, Linkage::Export);
    let mut builder = mb.builder(func_id);
    builder.init_entry_block();

    let scalable_v4i32 = Type::new_vector(ScalarType::I32, 4, true);

    let scalar_a = builder.ins().i32const(1);
    let scalar_b = builder.ins().i32const(2);
    let vec_a = builder.ins().splat(scalar_a, scalable_v4i32);
    let vec_b = builder.ins().splat(scalar_b, scalable_v4i32);

    let mask = builder.ins().bconst(true);
    let avl = builder.ins().i64const(16);
    let vl = builder.ins().setvl(avl);

    let result = builder.ins().vector_op_ext(
        Opcode::IAdd,
        &[vec_a, vec_b],
        mask,
        Some(vl),
        scalable_v4i32,
    );

    assert_eq!(builder.value_type(result), scalable_v4i32);

    builder.ins().ret(&[]);
    builder.seal_all_blocks();
}

#[test]
fn test_gather_load() {
    let mut mb = ModuleBuilder::new();
    let sig_id = mb.make_signature(vec![], vec![], CallConv::SystemV);
    let func_id = mb.declare_function("test_gather".to_string(), sig_id, Linkage::Export);
    let mut builder = mb.builder(func_id);
    builder.init_entry_block();

    let v4i32 = Type::new_vector(ScalarType::I32, 4, false);
    let v4i64 = Type::new_vector(ScalarType::I64, 4, false);

    let ptr_val = builder.ins().i64const(0x1000);
    let base_ptr = builder.ins().int_to_ptr(ptr_val);
    let idx_val = builder.ins().i64const(0);
    let indices = builder.ins().splat(idx_val, v4i64);

    let loaded = builder
        .ins()
        .gather(base_ptr, indices, 0, None, None, MemFlags::new(), v4i32);

    assert_eq!(builder.value_type(loaded), v4i32);

    builder.ins().ret(&[]);
    builder.seal_all_blocks();
}

#[test]
fn test_strided_load_store() {
    let mut mb = ModuleBuilder::new();
    let sig_id = mb.make_signature(vec![], vec![], CallConv::SystemV);
    let func_id = mb.declare_function("test_strided".to_string(), sig_id, Linkage::Export);
    let mut builder = mb.builder(func_id);
    builder.init_entry_block();

    let v8f32 = Type::new_vector(ScalarType::F32, 8, false);

    let ptr_val = builder.ins().i64const(0x1000);
    let base_ptr = builder.ins().int_to_ptr(ptr_val);
    let stride = builder.ins().i64const(2);

    let loaded = builder
        .ins()
        .load_stride(base_ptr, stride, 0, None, None, MemFlags::new(), v8f32);

    assert_eq!(builder.value_type(loaded), v8f32);

    builder
        .ins()
        .store_stride(base_ptr, stride, loaded, 0, None, None, MemFlags::new());

    builder.ins().ret(&[]);
    builder.seal_all_blocks();
}

#[test]
fn test_setvl() {
    let mut mb = ModuleBuilder::new();
    let sig_id = mb.make_signature(vec![], vec![], CallConv::SystemV);
    let func_id = mb.declare_function("test_setvl".to_string(), sig_id, Linkage::Export);
    let mut builder = mb.builder(func_id);
    builder.init_entry_block();

    let avl = builder.ins().i64const(100);
    let vl = builder.ins().setvl(avl);

    assert_eq!(builder.value_type(vl), Type::EVL);

    builder.ins().ret(&[]);
    builder.seal_all_blocks();
}

#[test]
fn test_vector_types_properties() {
    let v4i32 = Type::new_vector(ScalarType::I32, 4, false);
    assert!(v4i32.is_vector());
    assert!(!v4i32.is_scalable());
    assert!(!v4i32.is_predicate());
    assert_eq!(v4i32.lane_count(), 4);
    assert_eq!(v4i32.element_type(), Type::I32);
    assert_eq!(v4i32.size_bytes(), 16);

    let scalable_v4f32 = Type::new_vector(ScalarType::F32, 4, true);
    assert!(scalable_v4f32.is_vector());
    assert!(scalable_v4f32.is_scalable());

    let mask_fixed = Type::new_predicate(8, false);
    assert!(!mask_fixed.is_vector());
    assert!(mask_fixed.is_predicate());

    let mask_scalable = Type::new_predicate(4, true);
    assert!(mask_scalable.is_scalable());
    assert!(mask_scalable.is_predicate());
}
