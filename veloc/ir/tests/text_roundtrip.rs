use veloc_ir::{InstructionData, ModuleParser};

const MODULE: &str = r#"
global counter: i32 (local)

export function main(ptr, i32, f32, i32<4>, mask<4>) -> i32
  ss0: size 16
block0(v0: ptr, v1: i32, v2: f32, v3: i32<4>, v4: mask<4>):
  v5 = iconst.i32 -7
  v6 = fconst.f32 0x3f800000
  v7 = bconst.bool true
  v8 = vconst.i32<4> 0x00000000010000000200000003000000
  v9 = iadd.i32 v1, v5
  (sum.v29, overflow.v30) = iadd-with-overflow.i32 v1, v5
  v10 = ineg.i32 v9
  v11 = select.i32 v7, v9, v10
  v12 = icmp.bool lts v9, v10
  v13 = fcmp.bool eq v2, v6
  v14 = load.i32.trusted.align4 v0, offset=4
  store.volatile v14, v0, offset=8
  v15 = stack-load.i32 ss0, offset=4
  stack-store v15, ss0, offset=8
  v16 = stack-addr.ptr ss0
  v17 = ptr-offset.ptr v0, -4
  v18 = ptr-index.ptr v0, v1, scale=4, offset=-8
  v19 = inttoptr.ptr v1
  v20 = ptrtoint.i64 v0
  v21 = call.i32 callee(v11)
  v22 = call-indirect.i32 v0(v11) : (i32) -> i32
  v23 = call-intrinsic.f32 veloc.sin.f32(v2) : (f32) -> f32
  v24 = shuffle.i32<4> v3, v8, mask=0x00010203
  v25 = iadd.i32<4> v3, v8, mask=v4
  v26 = load-stride.i32<4>.align4 v0, stride=v1, offset=-16, mask=v4
  store-stride v26, v0, stride=v1, offset=16, mask=v4
  v27 = gather.i32<4> v0, index=v8, scale=4, mask=v4
  scatter v27, v0, index=v8, scale=4, mask=v4
  nop
  br v12, block1(), block2()
block1():
  jump block3(v21)
block2():
  br-table v1, [block3(v22)], block3(v11)
block3(v28: i32):
  return v28

import function callee(i32) -> i32
"#;

#[test]
fn canonical_module_text_round_trips_every_codec_family() {
    let module = ModuleParser::new().parse(MODULE).unwrap();
    module.validate().unwrap();

    let text = module.to_string();
    let reparsed = ModuleParser::new().parse(&text).unwrap();
    reparsed.validate().unwrap();
    assert_eq!(reparsed.to_string(), text);

    assert!(text.contains("fconst.f32 0x3f800000"));
    assert!(text.contains("vconst.i32<4> 0x00000000010000000200000003000000"));
    assert!(text.contains("mask=0x00010203"));
    assert!(text.contains("(sum.v29, overflow.v30) = iadd-with-overflow.i32 v1, v5"));
    assert!(text.contains("br-table v1, [block3(v22)], block3(v11)"));

    let main = &reparsed.functions[veloc_ir::FuncId(0)];
    let branch_table = main.layout.blocks[veloc_ir::Block(2)]
        .insts
        .last()
        .copied()
        .unwrap();
    let InstructionData::BrTable { table, .. } = main.dfg.inst(branch_table) else {
        panic!("expected branch table");
    };
    let targets = main.dfg.jump_table_targets(*table);
    assert_eq!(main.dfg.block_call_block(targets[0]), veloc_ir::Block(3));
    assert_eq!(main.dfg.block_call_block(targets[1]), veloc_ir::Block(3));
    assert_eq!(main.dfg.block_call_args(targets[0]), &[veloc_ir::Value(22)]);
    assert_eq!(main.dfg.block_call_args(targets[1]), &[veloc_ir::Value(11)]);
    assert_eq!(
        main.layout.blocks[veloc_ir::Block(0)].succs,
        [veloc_ir::Block(1), veloc_ir::Block(2)]
    );
    assert!(!text.contains("preds:"));
}

#[test]
fn malformed_named_fields_are_rejected() {
    let source = r#"
local function bad(ptr) -> i32
block0(v0: ptr):
  v1 = load.i32 v0, mystery=4
  return v1
"#;
    let error = ModuleParser::new().parse(source).unwrap_err().to_string();
    assert!(error.contains("unknown named field `mystery`"), "{error}");
}

#[test]
fn duplicate_ssa_definitions_are_rejected() {
    let source = r#"
local function bad() -> i32
block0():
  value = iconst.i32 1
  value = iconst.i32 2
  return value
"#;
    let error = ModuleParser::new().parse(source).unwrap_err().to_string();
    assert!(error.contains("already-defined"), "{error}");
}

#[test]
fn global_only_modules_round_trip() {
    let module = ModuleParser::new()
        .parse("global lanes: i32<scalable 4> (export)")
        .unwrap();
    let text = module.to_string();
    assert_eq!(ModuleParser::new().parse(&text).unwrap().to_string(), text);
}
