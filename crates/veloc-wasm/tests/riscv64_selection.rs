#![cfg(all(target_arch = "riscv64", target_os = "linux"))]

use veloc_wasm::engine::{Config, Strategy};
use veloc_wasm::{Engine, Linker, Module, Store, Val};

/// Long regions, shared producers and live values exceeding the GPR capacity.
/// Compare both pipeline levels and CPU models against the interpreter.
#[test]
fn scheduled_integer_dags_match_interpreter() {
    use veloc_wasm::veloc::codegen::{CodegenOptions, OptLevel};
    let mut wat = String::from("(module");
    for bits in [32, 64] {
        let ty = format!("i{bits}");
        wat.push_str(&format!(
            "(func (export \"dag{bits}\") (param {ty} {ty}) (result {ty}) (local {} )",
            vec![ty.clone(); 48].join(" ")
        ));
        for i in 0..24 {
            wat.push_str(&format!(
                "local.get 0 {ty}.const {} {ty}.add local.set {} ",
                i * 13 + 3,
                i + 2
            ));
        }
        for i in 0..24 {
            wat.push_str(&format!(
                "local.get {} local.get 1 {ty}.mul local.get {} {ty}.xor local.set {} ",
                i + 2,
                (i + 7) % 24 + 2,
                i + 26
            ));
        }
        wat.push_str(&format!("{ty}.const 0 "));
        for i in 26..50 {
            wat.push_str(&format!("local.get {i} {ty}.add "));
        }
        wat.push(')');
    }
    wat.push(')');
    let wasm = wat::parse_str(&wat).unwrap();
    let run = |config| {
        let engine = Engine::with_config(config);
        let module = Module::new(&engine, &wasm).unwrap();
        let mut store = Store::new();
        let instance = Linker::new().instantiate(&mut store, module).unwrap();
        let mut results = Vec::new();
        for bits in [32, 64] {
            let func = instance.get_func(&store, &format!("dag{bits}")).unwrap();
            for a in [i64::MIN, i32::MIN as i64, -1, 0, 1, 17, i64::MAX] {
                for b in [-37, 0, 1, 13, i64::MAX] {
                    let val = |v| {
                        if bits == 32 {
                            Val::I32(v as i32)
                        } else {
                            Val::I64(v)
                        }
                    };
                    results.push(func.call(&mut store, &[val(a), val(b)]).unwrap());
                }
            }
        }
        results
    };
    let expected = run(Config {
        strategy: Strategy::Interpreter,
        ..Default::default()
    });
    for cpu in ["generic", "c908"] {
        for level in [OptLevel::None, OptLevel::Default] {
            for mir_level in [0, 1] {
                let actual = run(Config {
                    strategy: Strategy::Jit,
                    cpu: cpu.into(),
                    opt_level: mir_level,
                    codegen: CodegenOptions {
                        opt_level: level,
                        verify: true,
                        ..Default::default()
                    },
                    ..Default::default()
                });
                assert_eq!(actual, expected, "{cpu}, {level:?}, MIR O{mir_level}");
            }
        }
    }
}

/// Exercise immediate boundaries, RV64's sign-extended i32 representation,
/// comparison branches, shared producers and optional extension fallbacks.
#[test]
fn scalar_selection_matches_interpreter() {
    let mut wat = String::from("(module\n");
    let mut cases = Vec::new();
    for bits in [32, 64] {
        let ty = format!("i{bits}");
        let values = [
            i64::MIN,
            i32::MIN as i64,
            -4097,
            -2048,
            -1,
            0,
            1,
            2047,
            2048,
            i32::MAX as i64,
            u32::MAX as i64,
            i64::MAX,
        ];
        for op in [
            "add", "and", "or", "xor", "shl", "shr_s", "shr_u", "rotl", "rotr",
        ] {
            for imm in [-2049, -2048, -1, 0, 1, 3, 31, 32, 63, 64, 2047, 2048] {
                let name = format!("f{}", cases.len());
                wat.push_str(&format!("(func (export \"{name}\") (param {ty}) (result {ty}) local.get 0 {ty}.const {imm} {ty}.{op})\n"));
                let args = values
                    .iter()
                    .map(|&v| {
                        vec![if bits == 32 {
                            Val::I32(v as i32)
                        } else {
                            Val::I64(v)
                        }]
                    })
                    .collect::<Vec<_>>();
                cases.push((name, args));
            }
        }
        for op in [
            "eq", "ne", "lt_s", "le_s", "gt_s", "ge_s", "lt_u", "le_u", "gt_u", "ge_u",
        ] {
            let name = format!("f{}", cases.len());
            wat.push_str(&format!("(func (export \"{name}\") (param {ty} {ty}) (result i32) local.get 0 local.get 1 {ty}.{op} if (result i32) i32.const 17 else i32.const 29 end)\n"));
            let val = |v| {
                if bits == 32 {
                    Val::I32(v as i32)
                } else {
                    Val::I64(v)
                }
            };
            cases.push((
                name,
                values
                    .iter()
                    .flat_map(|&a| values.iter().map(move |&b| vec![val(a), val(b)]))
                    .collect(),
            ));
        }
        for op in ["extend8_s", "extend16_s"] {
            let name = format!("f{}", cases.len());
            wat.push_str(&format!(
                "(func (export \"{name}\") (param {ty}) (result {ty}) local.get 0 {ty}.{op})\n"
            ));
            cases.push((
                name,
                values
                    .iter()
                    .map(|&v| {
                        vec![if bits == 32 {
                            Val::I32(v as i32)
                        } else {
                            Val::I64(v)
                        }]
                    })
                    .collect(),
            ));
        }
    }
    wat.push_str(
        r#"
        (func (export "zext") (param i32) (result i64) local.get 0 i64.extend_i32_u)
        (func (export "shiftadd") (param i64 i64) (result i64)
            local.get 0 local.get 1 i64.const 3 i64.shl i64.add)
        (func (export "shared") (param i32 i32) (result i32) (local i32)
            local.get 0 local.get 1 i32.lt_u local.set 2
            local.get 2 if (result i32) i32.const 5 else i32.const 9 end
            local.get 2 i32.add)
    )"#,
    );
    cases.push((
        "zext".into(),
        [-1, i32::MIN, 0, i32::MAX]
            .into_iter()
            .map(|v| vec![Val::I32(v)])
            .collect(),
    ));
    cases.push((
        "shiftadd".into(),
        vec![
            vec![Val::I64(7), Val::I64(-1)],
            vec![Val::I64(i64::MAX), Val::I64(i64::MAX)],
        ],
    ));
    cases.push((
        "shared".into(),
        vec![
            vec![Val::I32(-1), Val::I32(0)],
            vec![Val::I32(0), Val::I32(-1)],
        ],
    ));
    let wasm = wat::parse_str(&wat).unwrap();
    let run = |config| {
        let engine = Engine::with_config(config);
        let module = Module::new(&engine, &wasm).unwrap();
        let mut store = Store::new();
        let instance = Linker::new().instantiate(&mut store, module).unwrap();
        let mut output = Vec::new();
        for (name, args) in &cases {
            let func = instance.get_func(&store, name).unwrap();
            for args in args {
                output.push(func.call(&mut store, args).unwrap());
            }
        }
        output
    };
    let expected = run(Config {
        strategy: Strategy::Interpreter,
        ..Default::default()
    });
    for opt_level in [0, 1] {
        for (cpu, features) in [
            ("generic", vec![]),
            ("c908", vec![]),
            ("c908", vec!["-Zba"]),
            ("c908", vec!["-Zbb"]),
            ("c908", vec!["-Zba", "-Zbb"]),
        ] {
            let actual = run(Config {
                strategy: Strategy::Jit,
                cpu: cpu.into(),
                cpu_features: features.iter().map(|s| s.to_string()).collect(),
                opt_level,
                codegen: veloc_wasm::veloc::codegen::CodegenOptions {
                    verify: true,
                    ..Default::default()
                },
                ..Default::default()
            });
            assert_eq!(
                actual, expected,
                "cpu={cpu}, features={features:?}, O{opt_level}"
            );
        }
    }
}
