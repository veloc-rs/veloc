#![cfg(all(target_arch = "x86_64", target_os = "linux"))]

use veloc_wasm::engine::{Config, Strategy};
use veloc_wasm::{Engine, Linker, Module, Store, Val};

#[test]
fn multi_results_and_unused_arguments_cross_the_entry_abi() {
    let wasm = wat::parse_str(
        r#"
      (module
        (func (export "constant") (param i64) (result i32)
          i32.const 195940365 return)
        (func $multi (result i32 i64 f32 f64)
          i32.const -7 i64.const 9000000000 f32.const -1.25 f64.const 3.5)
        (func (export "multi") (result i32 i64 f32 f64) call $multi))
    "#,
    )
    .unwrap();
    for strategy in [Strategy::Interpreter, Strategy::Jit] {
        let engine = Engine::with_config(Config {
            strategy,
            ..Config::default()
        });
        let module = Module::new(&engine, &wasm).unwrap();
        let mut store = Store::new();
        let instance = Linker::new().instantiate(&mut store, module).unwrap();
        let constant = instance.get_func(&store, "constant").unwrap();
        assert_eq!(
            constant.call(&mut store, &[Val::I64(42)]).unwrap(),
            vec![Val::I32(195940365)]
        );
        let multi = instance.get_func(&store, "multi").unwrap();
        assert_eq!(
            multi.call(&mut store, &[]).unwrap(),
            vec![
                Val::I32(-7),
                Val::I64(9000000000),
                Val::F32(-1.25),
                Val::F64(3.5)
            ]
        );
    }
}

#[test]
#[ignore = "full CoreMark benchmark; run explicitly with --release --ignored"]
fn coremark_validates_under_jit() {
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_veloc-wasm"))
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/wasm/coremark.wasm"
        ))
        .args(["--strategy", "jit"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    while child.try_wait().unwrap().is_none() {
        if std::time::Instant::now() > deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("CoreMark JIT did not finish within 120 seconds");
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stdout}\n{stderr}");
    assert!(
        stdout.contains("Correct operation validated."),
        "{stdout}\n{stderr}"
    );
    for crc in [
        "seedcrc          : 0xe9f5",
        "[0]crclist       : 0xe714",
        "[0]crcmatrix     : 0x1fd7",
        "[0]crcstate      : 0x8e3a",
    ] {
        assert!(stdout.contains(crc), "missing {crc}:\n{stdout}");
    }
    println!("{stdout}");
}

#[test]
fn host_imports_use_native_bridges_for_calls_and_tables() {
    let wasm = wat::parse_str(
        r#"
        (module
          (type $host (func (param i32 i64 f32 f64 i32 i32 i32 i64) (result f64)))
          (import "env" "host" (func $host (type $host)))
          (table 1 funcref)
          (elem (i32.const 0) $host)
          (func (export "run") (result f64)
            i32.const -7 i64.const -9000000000 f32.const 1.25 f64.const -3.5
            i32.const 17 i32.const 23 i32.const 31 i64.const 10000000000
            call $host
            i32.const -7 i64.const -9000000000 f32.const 1.25 f64.const -3.5
            i32.const 17 i32.const 23 i32.const 31 i64.const 10000000000
            i32.const 0 call_indirect (type $host)
            f64.add))
    "#,
    )
    .unwrap();
    for strategy in [Strategy::Interpreter, Strategy::Jit] {
        let engine = Engine::with_config(Config {
            strategy,
            ..Config::default()
        });
        let module = Module::new(&engine, &wasm).unwrap();
        let mut store = Store::new();
        let mut linker = Linker::new();
        let received = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured = received.clone();
        linker.func_wrap(
            &mut store,
            "env",
            "host",
            move |a: i32, b: i64, c: f32, d: f64, e: i32, f: i32, g: i32, h: i64| {
                captured.lock().unwrap().push((a, b, c, d, e, f, g, h));
                42.25f64
            },
        );
        let instance = linker.instantiate(&mut store, module).unwrap();
        let run = instance.get_func(&store, "run").unwrap();
        assert_eq!(run.call(&mut store, &[]).unwrap(), vec![Val::F64(84.5)]);
        assert_eq!(
            *received.lock().unwrap(),
            vec![(-7, -9000000000, 1.25, -3.5, 17, 23, 31, 10000000000); 2]
        );
    }
}
