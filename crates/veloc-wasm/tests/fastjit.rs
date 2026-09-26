#![cfg(all(target_arch = "x86_64", target_os = "linux"))]

use veloc_wasm::engine::{Config, Strategy};
use veloc_wasm::{Engine, Linker, Module, Store, Val};

#[test]
fn fast_jit_executes_integer_control_flow() {
    let wasm = wat::parse_str(
        r#"(module
            (func (export "sum") (param i32) (result i32)
                (local i32 i32)
                i32.const 0
                local.set 1
                i32.const 0
                local.set 2
                block
                  loop
                    local.get 1
                    local.get 0
                    i32.ge_u
                    br_if 1
                    local.get 2
                    local.get 1
                    i32.add
                    local.set 2
                    local.get 1
                    i32.const 1
                    i32.add
                    local.set 1
                    br 0
                  end
                end
                local.get 2))"#,
    )
    .unwrap();
    let engine = Engine::with_config(Config {
        strategy: Strategy::FastJit,
        ..Config::default()
    });
    let module = Module::new(&engine, &wasm).unwrap();
    let mut store = Store::new();
    let instance = Linker::new().instantiate(&mut store, module).unwrap();
    let sum = instance.get_func(&store, "sum").unwrap();
    assert_eq!(
        sum.call(&mut store, &[Val::I32(10)]).unwrap(),
        vec![Val::I32(45)]
    );
}

#[test]
fn fast_jit_handles_mixed_register_and_stack_arguments() {
    let wasm = wat::parse_str(
        r#"(module
            (func $inner (param i32 f64 i32 f64 f64 f64 f64 f64 f64 f64 f64) (result f64)
                local.get 1
                local.get 10
                f64.add)
            (func (export "mixed") (param i32 f64 i32 f64 f64 f64 f64 f64 f64 f64 f64) (result f64)
                local.get 0
                local.get 1
                local.get 2
                local.get 3
                local.get 4
                local.get 5
                local.get 6
                local.get 7
                local.get 8
                local.get 9
                local.get 10
                call $inner))"#,
    )
    .unwrap();
    let engine = Engine::with_config(Config {
        strategy: Strategy::FastJit,
        ..Config::default()
    });
    let module = Module::new(&engine, &wasm).unwrap();
    let mut store = Store::new();
    let instance = Linker::new().instantiate(&mut store, module).unwrap();
    let mixed = instance.get_func(&store, "mixed").unwrap();
    let args = [
        Val::I32(7),
        Val::F64(1.25),
        Val::I32(8),
        Val::F64(2.0),
        Val::F64(3.0),
        Val::F64(4.0),
        Val::F64(5.0),
        Val::F64(6.0),
        Val::F64(7.0),
        Val::F64(8.0),
        Val::F64(9.5),
    ];
    assert_eq!(
        mixed.call(&mut store, &args).unwrap(),
        vec![Val::F64(10.75)]
    );
}
