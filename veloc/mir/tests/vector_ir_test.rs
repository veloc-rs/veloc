//! Vector IR Integration Tests

use veloc_mir::{CallConv, Linkage, Opcode, VectorMemOptions, builder::ModuleBuilder, types::Type};

#[test]
fn generated_pool_builders_use_logical_parameters() {
    let mut module = ModuleBuilder::new();
    let sig = module.make_signature(
        vec![Type::PTR, Type::I32X4],
        vec![Type::I32X4],
        CallConv::SystemV,
    );
    let func = module.declare_function("pooled".into(), sig, Linkage::Local);
    {
        let mut builder = module.builder(func);
        builder.init_entry_block();
        let ptr = builder.func_param(0);
        let indices = builder.func_param(1);
        let constant = builder.ins().vconst(vec![0; 16], Type::I32X4);
        let shuffled = builder.ins().shuffle(constant, indices, vec![0, 2, 4, 6]);
        builder.ins().scatter(
            ptr,
            indices,
            shuffled,
            VectorMemOptions {
                scale: 4,
                ..Default::default()
            },
        );
        builder.ins().ret(&[shuffled]);
    }
    module.validate().unwrap();
}

#[test]
fn test_simple_vector_add_fixed() {
    let mut mb = ModuleBuilder::new();
    let sig_id = mb.make_signature(vec![], vec![], CallConv::SystemV);
    let func_id = mb.declare_function("test_vadd".to_string(), sig_id, Linkage::Export);
    let mut builder = mb.builder(func_id);
    builder.init_entry_block();

    let v4i32 = Type::I32
        .as_scalar()
        .unwrap()
        .vector(4, false)
        .unwrap()
        .as_type();

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

    let v8i32 = Type::I32
        .as_scalar()
        .unwrap()
        .vector(8, false)
        .unwrap()
        .as_type();
    let v4f64 = Type::F64
        .as_scalar()
        .unwrap()
        .vector(4, false)
        .unwrap()
        .as_type();

    let scalar_i = builder.ins().i32const(42);
    let scalar_f = builder.ins().f64const(core::f64::consts::PI);

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

    let v4f32 = Type::F32
        .as_scalar()
        .unwrap()
        .vector(4, false)
        .unwrap()
        .as_type();
    let scalar = builder.ins().f32const(1.0);
    let vec = builder.ins().splat(scalar, v4f32);

    let sum = builder.ins().reduce_sum(vec);
    let add = builder.ins().reduce_add(vec);
    let min = builder.ins().reduce_min(vec);
    let max = builder.ins().reduce_max(vec);

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

    let v4i32 = Type::I32
        .as_scalar()
        .unwrap()
        .vector(4, false)
        .unwrap()
        .as_type();
    let scalar = builder.ins().i32const(10);
    let vec = builder.ins().splat(scalar, v4i32);

    let extracted = builder.ins().extract_element(vec, 0);
    assert_eq!(builder.value_type(extracted), Type::I32);

    let new_val = builder.ins().i32const(20);
    let inserted = builder.ins().insert_element(vec, new_val, 1);
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

    let scalable_v4i32 = Type::I32
        .as_scalar()
        .unwrap()
        .vector(4, true)
        .unwrap()
        .as_type();

    let scalar_a = builder.ins().i32const(1);
    let scalar_b = builder.ins().i32const(2);
    let vec_a = builder.ins().splat(scalar_a, scalable_v4i32);
    let vec_b = builder.ins().splat(scalar_b, scalable_v4i32);

    let mask_ty = Type::new_mask(4, true).unwrap();
    let mask = builder.ins().vconst(vec![u8::MAX; 4], mask_ty);
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
    drop(builder);
    mb.validate().unwrap();
}

#[test]
fn test_gather_load() {
    let mut mb = ModuleBuilder::new();
    let sig_id = mb.make_signature(vec![], vec![], CallConv::SystemV);
    let func_id = mb.declare_function("test_gather".to_string(), sig_id, Linkage::Export);
    let mut builder = mb.builder(func_id);
    builder.init_entry_block();

    let v4i32 = Type::I32
        .as_scalar()
        .unwrap()
        .vector(4, false)
        .unwrap()
        .as_type();
    let v4i64 = Type::I64
        .as_scalar()
        .unwrap()
        .vector(4, false)
        .unwrap()
        .as_type();

    let ptr_val = builder.ins().i64const(0x1000);
    let base_ptr = builder.ins().inttoptr(ptr_val);
    let idx_val = builder.ins().i64const(0);
    let indices = builder.ins().splat(idx_val, v4i64);

    let loaded = builder
        .ins()
        .gather(base_ptr, indices, VectorMemOptions::default(), v4i32);

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

    let v8f32 = Type::F32
        .as_scalar()
        .unwrap()
        .vector(8, false)
        .unwrap()
        .as_type();

    let ptr_val = builder.ins().i64const(0x1000);
    let base_ptr = builder.ins().inttoptr(ptr_val);
    let stride = builder.ins().i64const(2);

    let loaded = builder
        .ins()
        .load_stride(base_ptr, stride, VectorMemOptions::default(), v8f32);

    assert_eq!(builder.value_type(loaded), v8f32);

    builder
        .ins()
        .store_stride(base_ptr, stride, loaded, VectorMemOptions::default());

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

    assert_eq!(builder.value_type(vl), Type::I32);

    builder.ins().ret(&[]);
    builder.seal_all_blocks();
}

#[test]
fn test_vector_types_properties() {
    let v4i32 = Type::I32
        .as_scalar()
        .unwrap()
        .vector(4, false)
        .unwrap()
        .as_type();
    assert!(v4i32.is_vector());
    assert!(!v4i32.is_scalable());
    assert!(!v4i32.is_predicate());
    assert_eq!(v4i32.lane_count(), 4);
    assert_eq!(v4i32.element_type(), Type::I32);
    assert_eq!(v4i32.fixed_size_bytes(), Some(16));

    let scalable_v4f32 = Type::F32
        .as_scalar()
        .unwrap()
        .vector(4, true)
        .unwrap()
        .as_type();
    assert!(scalable_v4f32.is_vector());
    assert!(scalable_v4f32.is_scalable());

    let mask_fixed = Type::new_mask(8, false).unwrap();
    assert!(mask_fixed.is_vector());
    assert!(mask_fixed.is_predicate());

    let mask_scalable = Type::new_mask(4, true).unwrap();
    assert!(mask_scalable.is_scalable());
    assert!(mask_scalable.is_predicate());
}
