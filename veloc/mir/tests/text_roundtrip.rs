use veloc_mir::{CallConv, InstructionData, Linkage, ModuleBuilder, ModuleParser};

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
  v14 = load.i32.align4 v0, offset=4
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

    let main = &reparsed.functions[veloc_mir::FuncId(0)];
    let branch_table = main.layout.blocks[veloc_mir::Block(2)]
        .insts
        .last()
        .copied()
        .unwrap();
    let InstructionData::BrTable { table, .. } = main.dfg.inst(branch_table) else {
        panic!("expected branch table");
    };
    let targets = main.dfg.jump_table_targets(*table);
    assert_eq!(main.dfg.block_call_block(targets[0]), veloc_mir::Block(3));
    assert_eq!(main.dfg.block_call_block(targets[1]), veloc_mir::Block(3));
    assert_eq!(main.dfg.block_call_args(targets[0]), &[veloc_mir::Value(22)]);
    assert_eq!(main.dfg.block_call_args(targets[1]), &[veloc_mir::Value(11)]);
    assert_eq!(
        main.layout.blocks[veloc_mir::Block(0)].succs,
        [veloc_mir::Block(1), veloc_mir::Block(2)]
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

#[test]
fn fixed_prefixes_allow_forward_instruction_values_in_valid_cfg_order() {
    let mut module = ModuleBuilder::new();
    let signature = module.make_signature(vec![], vec![], CallConv::SystemV);
    let function = module.declare_function("forward_values".into(), signature, Linkage::Export);
    let mut builder = module.builder(function);
    let entry = builder.init_entry_block();
    let call_block = builder.create_block();
    let table_block = builder.create_block();
    let definitions = builder.create_block();
    let exit = builder.create_block();
    builder.ins().jump(definitions, &[]);

    // Definitions dominate their uses in the CFG, but follow them in layout order.
    builder.switch_to_block(definitions);
    let condition = builder.ins().bconst(true);
    let index = builder.ins().i32const(0);
    let address = builder.ins().i64const(0);
    let callee = builder.ins().inttoptr(address);
    builder.ins().jump(call_block, &[]);

    builder.switch_to_block(call_block);
    builder.ins().call_indirect(signature, callee, &[]);
    builder.ins().br(condition, table_block, &[], exit, &[]);
    builder.switch_to_block(table_block);
    let default = builder.make_block_call(exit, &[]);
    builder.ins().br_table(index, default, &[]);
    builder.switch_to_block(exit);
    builder.ins().ret(&[]);
    builder.seal_all_blocks();
    builder.func_mut().layout.block_order = vec![entry, call_block, table_block, definitions, exit];
    module.validate().unwrap();

    let text = module.build().to_string();
    let parsed = ModuleParser::new().parse(&text).unwrap();
    parsed.validate().unwrap();
    assert_eq!(parsed.to_string(), text);
}

#[test]
fn deferred_prefix_checks_reject_wrong_types_and_undefined_values() {
    for instruction in [
        "br v0, block3(), block3()",
        "br-table v0, [], block3()",
        "call-indirect v0() : () -> void\n  jump block3()",
    ] {
        let definitions = "block2():\n  v0 = fconst.f32 0x00000000\n  jump block1()\n";
        let uses = format!("block1():\n  {instruction}\n");
        for forward in [false, true] {
            let blocks = if forward {
                format!("{uses}{definitions}")
            } else {
                format!("{definitions}{uses}")
            };
            let source = format!(
                "local function bad() -> void\nblock0():\n  jump block2()\n{blocks}block3():\n  return\n"
            );
            let error = ModuleParser::new().parse(&source).unwrap_err().to_string();
            assert!(error.contains("invalid operand types"), "{error}");
            assert!(error.contains("got: F32"), "{error}");

            let undefined = source.replace("  v0 = fconst.f32 0x00000000\n", "");
            let error = ModuleParser::new()
                .parse(&undefined)
                .unwrap_err()
                .to_string();
            assert!(error.contains("undefined SSA value `v0`"), "{error}");
        }
    }
}

fn changed_instruction(before: &str, after: &str) -> String {
    assert!(
        MODULE.contains(before),
        "fixture does not contain `{before}`"
    );
    MODULE.replacen(before, after, 1)
}

fn rejected_text(source: &str, expected: &str) {
    let error = match ModuleParser::new().parse(source) {
        Ok(_) => panic!("parser accepted invalid text:\n{source}"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains(expected), "expected `{expected}` in {error}");
}

#[test]
fn vector_memory_rejects_fields_belonging_to_other_addressing_modes() {
    for (before, after, field) in [
        (
            "stride=v1, offset=-16",
            "stride=v1, index=v8, offset=-16",
            "index",
        ),
        (
            "stride=v1, offset=16",
            "stride=v1, index=v8, offset=16",
            "index",
        ),
        (
            "stride=v1, offset=-16",
            "stride=v1, scale=2, offset=-16",
            "scale",
        ),
        (
            "stride=v1, offset=16",
            "stride=v1, scale=2, offset=16",
            "scale",
        ),
        (
            "v27 = gather.i32<4> v0, index=v8",
            "v27 = gather.i32<4> v0, stride=v1, index=v8",
            "stride",
        ),
        (
            "scatter v27, v0, index=v8",
            "scatter v27, v0, stride=v1, index=v8",
            "stride",
        ),
    ] {
        // The old shared whitelist accepted these fields without consuming them.
        rejected_text(&changed_instruction(before, after), field);
    }
}

#[test]
fn duplicate_named_fields_are_rejected_in_each_named_codec_family() {
    for (before, after) in [
        (
            "load.i32.align4 v0, offset=4",
            "load.i32.align4 v0, offset=4, offset=8",
        ),
        (
            "store.volatile v14, v0, offset=8",
            "store.volatile v14, v0, offset=8, offset=4",
        ),
        (
            "stack-load.i32 ss0, offset=4",
            "stack-load.i32 ss0, offset=4, offset=8",
        ),
        (
            "stack-store v15, ss0, offset=8",
            "stack-store v15, ss0, offset=8, offset=4",
        ),
        (
            "stack-addr.ptr ss0",
            "stack-addr.ptr ss0, offset=0, offset=4",
        ),
        ("scale=4, offset=-8", "scale=4, scale=2, offset=-8"),
        (
            "shuffle.i32<4> v3, v8, mask=0x00010203",
            "shuffle.i32<4> v3, v8, mask=0x00010203, mask=0x00010203",
        ),
        (
            "iadd.i32<4> v3, v8, mask=v4",
            "iadd.i32<4> v3, v8, mask=v4, mask=v4",
        ),
        (
            "iadd.i32<4> v3, v8, mask=v4",
            "iadd.i32<4> v3, v8, mask=v4, evl=v1, evl=v1",
        ),
        ("stride=v1, offset=-16", "stride=v1, stride=v1, offset=-16"),
        ("stride=v1, offset=16", "stride=v1, stride=v1, offset=16"),
        (
            "gather.i32<4> v0, index=v8",
            "gather.i32<4> v0, index=v8, index=v8",
        ),
        (
            "scatter v27, v0, index=v8",
            "scatter v27, v0, index=v8, index=v8",
        ),
    ] {
        rejected_text(&changed_instruction(before, after), "duplicate");
    }
}

#[test]
fn default_memory_properties_are_omitted_from_canonical_text() {
    let source = r#"
export function defaults(ptr, i32, i32<4>, mask<4>) -> void
  ss0: size 32
block0(v0: ptr, v1: i32, v2: i32<4>, v3: mask<4>):
  v4 = load.i32.align1 v0, offset=0
  store v4, v0, offset=0
  v5 = stack-load.i32 ss0, offset=0
  stack-store v5, ss0, offset=0
  v6 = stack-addr.ptr ss0, offset=0
  v7 = ptr-index.ptr v0, v1, scale=1, offset=0
  v8 = load-stride.i32<4> v0, stride=v1, offset=0, mask=v3, evl=v1
  store-stride v8, v0, stride=v1, offset=0, mask=v3, evl=v1
  v9 = gather.i32<4> v0, index=v2, scale=1, offset=0, mask=v3, evl=v1
  scatter v9, v0, index=v2, scale=1, offset=0, mask=v3, evl=v1
  v10 = iadd.i32<4> v2, v2, mask=v3, evl=v1
  return
"#;
    let module = ModuleParser::new().parse(source).unwrap();
    module.validate().unwrap();
    let text = module.to_string();
    assert!(!text.contains("offset=0"), "{text}");
    assert!(!text.contains("scale=1"), "{text}");
    assert!(!text.contains(".align1"), "{text}");
    assert_eq!(text.matches("mask=v3, evl=v1").count(), 5);
    assert!(text.contains("ptr-index.ptr v0, v1\n"), "{text}");
    let reparsed = ModuleParser::new().parse(&text).unwrap();
    reparsed.validate().unwrap();
    assert_eq!(reparsed.to_string(), text);
}

#[test]
fn floating_nan_payloads_round_trip_without_host_float_conversion() {
    for (ty, bits, encoded) in [
        ("f32", 0x7fa1_2345, "0x7fa12345"),
        ("f32", 0xffc0_1234, "0xffc01234"),
        ("f64", 0x7ff0_0000_0000_0001, "0x7ff0000000000001"),
    ] {
        let source = format!(
            "local function bits() -> {ty}\nblock0():\n  v0 = fconst.{ty} {encoded}\n  return v0\n"
        );
        let module = ModuleParser::new().parse(&source).unwrap();
        module.validate().unwrap();
        let text = module.to_string();
        assert!(text.contains(encoded), "{text}");
        let reparsed = ModuleParser::new().parse(&text).unwrap();
        reparsed.validate().unwrap();
        let function = &reparsed.functions[veloc_mir::FuncId(0)];
        assert!(matches!(
            function.dfg.instructions.values().next().unwrap(),
            InstructionData::Fconst { value } if *value == bits
        ));
        assert_eq!(reparsed.to_string(), text);
    }
}

#[test]
fn empty_calls_and_branch_tables_preserve_the_default_target() {
    let source = r#"
export function empty(ptr, i32) -> void
block0(v0: ptr, v1: i32):
  call sink()
  call-indirect v0() : () -> void
  br-table v1, [], block1()
block1():
  br-table v1, [block2(), block3()], block4()
block2():
  return
block3():
  return
block4():
  return
import function sink() -> void
"#;
    let module = ModuleParser::new().parse(source).unwrap();
    module.validate().unwrap();
    let text = module.to_string();
    assert!(text.contains("call sink()"), "{text}");
    assert!(text.contains("call-indirect v0() : () -> void"), "{text}");
    assert!(text.contains("br-table v1, [], block1()"), "{text}");
    assert!(
        text.contains("br-table v1, [block2(), block3()], block4()"),
        "{text}"
    );
    let reparsed = ModuleParser::new().parse(&text).unwrap();
    reparsed.validate().unwrap();
    assert_eq!(reparsed.to_string(), text);
    let dfg = &reparsed.functions[veloc_mir::FuncId(0)].dfg;
    let tables = dfg
        .instructions
        .values()
        .filter_map(|data| {
            if let InstructionData::BrTable { table, .. } = data {
                Some(
                    dfg.jump_table_targets(*table)
                        .iter()
                        .map(|&target| dfg.block_call_block(target).0)
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(tables, [vec![1], vec![2, 3, 4]]);
    for data in dfg.instructions.values() {
        if let InstructionData::Call { args, .. } | InstructionData::CallIndirect { args, .. } =
            data
        {
            assert!(dfg.get_value_list(*args).is_empty());
        }
    }
}

#[test]
fn malformed_payloads_are_rejected_across_all_text_codec_families() {
    for (before, after, expected) in [
        ("iadd.i32 v1, v5", "iadd.i32 v1", "operand"),
        ("  nop", "  nop v1", "operand"),
        ("iconst.i32 -7", "iconst.i32 nonsense", "integer"),
        ("fconst.f32 0x3f800000", "fconst.f32 0x100000000", "32 bits"),
        ("bconst.bool true", "bconst.bool maybe", "true"),
        (
            "vconst.i32<4> 0x00000000010000000200000003000000",
            "vconst.i32<4> 0x123",
            "even",
        ),
        (
            "load.i32.align4 v0, offset=4",
            "load.i32.align4 v0, offset=-1",
            "offset",
        ),
        (
            "store.volatile v14, v0, offset=8",
            "store.volatile v14",
            "operand",
        ),
        (
            "stack-load.i32 ss0, offset=4",
            "stack-load.i32 slot0, offset=4",
            "stack slot",
        ),
        (
            "stack-store v15, ss0, offset=8",
            "stack-store v15, v0, offset=8",
            "stack slot",
        ),
        (
            "stack-addr.ptr ss0",
            "stack-addr.ptr ss0, v1",
            "named field",
        ),
        (
            "ptr-offset.ptr v0, -4",
            "ptr-offset.ptr v0, invalid",
            "offset",
        ),
        ("scale=4, offset=-8", "scale=invalid, offset=-8", "scale"),
        ("callee(v11)", "missing(v11)", "function"),
        (
            "call-indirect.i32 v0(v11) : (i32) -> i32",
            "call-indirect.i32 v0(v11)",
            "signature",
        ),
        ("veloc.sin.f32(v2)", "veloc.missing(v2)", "intrinsic"),
        ("jump block3(v21)", "jump block99(v21)", "block"),
        ("br v12, block1(), block2()", "br v12, block1()", "operand"),
        (
            "br-table v1, [block3(v22)], block3(v11)",
            "br-table v1, block3(v22), block3(v11)",
            "[]",
        ),
        ("return v28", "return v28,", "SSA value"),
        (
            "icmp.bool lts v9, v10",
            "icmp.bool invalid v9, v10",
            "condition",
        ),
        (
            "fcmp.bool eq v2, v6",
            "fcmp.bool invalid v2, v6",
            "condition",
        ),
        ("stride=v1, offset=-16", "offset=-16", "stride"),
        ("stride=v1, offset=16", "offset=16", "stride"),
        (
            "gather.i32<4> v0, index=v8, scale=4",
            "gather.i32<4> v0, scale=4",
            "index",
        ),
        (
            "scatter v27, v0, index=v8, scale=4",
            "scatter v27, v0, scale=4",
            "index",
        ),
        (
            "shuffle.i32<4> v3, v8, mask=0x00010203",
            "shuffle.i32<4> v3, v8",
            "mask",
        ),
        (
            "iadd.i32 v1, v5",
            "iadd.i32.volatile v1, v5",
            "memory flags",
        ),
    ] {
        rejected_text(&changed_instruction(before, after), expected);
    }
}
