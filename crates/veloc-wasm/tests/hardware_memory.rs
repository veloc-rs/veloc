#![cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]

use veloc_wasm::engine::{Config, Strategy};
use veloc_wasm::error::Error;
use veloc_wasm::vm::TrapCode;
use veloc_wasm::{Engine, Linker, Module, Store, Val};

fn engine(hardware_memory_checks: bool, opt_level: u8) -> Engine {
    Engine::with_config(Config {
        strategy: Strategy::Interpreter,
        hardware_memory_checks,
        opt_level,
        ..Config::default()
    })
}

fn out_of_bounds(result: Result<Vec<Val>, Error>) {
    assert!(
        matches!(&result, Err(Error::Trap(TrapCode::MemoryOutOfBounds))),
        "expected memory out of bounds trap, got {result:?}"
    );
}

#[test]
fn protected_memory_traps_and_grows() {
    let wasm = wat::parse_str(
        r#"
        (module
          (memory 1 2)
          (func (export "load8") (param i32) (result i32)
            local.get 0 i32.load8_u)
          (func (export "load32") (param i32) (result i32)
            local.get 0 i32.load)
          (func (export "store32") (param i32)
            local.get 0 i32.const 0x12345678 i32.store)
          (func (export "discard") (param i32)
            local.get 0 i32.load drop)
          (func (export "large_offset") (result i32)
            i32.const -1 i32.load8_u offset=4294967295)
          (func (export "grow") (result i32)
            i32.const 1 memory.grow))
        "#,
    )
    .unwrap();

    for opt_level in [0, 1] {
        let module = Module::new(&engine(true, opt_level), &wasm).unwrap();
        let mut store = Store::new();
        let instance = Linker::new().instantiate(&mut store, module).unwrap();
        let load8 = instance.get_func(&store, "load8").unwrap();
        let load32 = instance.get_func(&store, "load32").unwrap();
        let store32 = instance.get_func(&store, "store32").unwrap();
        let discard = instance.get_func(&store, "discard").unwrap();
        let large_offset = instance.get_func(&store, "large_offset").unwrap();
        let grow = instance.get_func(&store, "grow").unwrap();

        assert_eq!(
            load8.call(&mut store, &[Val::I32(65535)]).unwrap(),
            vec![Val::I32(0)]
        );
        out_of_bounds(load8.call(&mut store, &[Val::I32(65536)]));
        out_of_bounds(load32.call(&mut store, &[Val::I32(65533)]));
        out_of_bounds(store32.call(&mut store, &[Val::I32(65534)]));
        out_of_bounds(discard.call(&mut store, &[Val::I32(65536)]));
        out_of_bounds(large_offset.call(&mut store, &[]));
        assert_eq!(
            load8.call(&mut store, &[Val::I32(65535)]).unwrap(),
            vec![Val::I32(0)]
        );

        assert_eq!(grow.call(&mut store, &[]).unwrap(), vec![Val::I32(1)]);
        assert_eq!(
            load8.call(&mut store, &[Val::I32(65536)]).unwrap(),
            vec![Val::I32(0)]
        );
        out_of_bounds(load8.call(&mut store, &[Val::I32(131072)]));
    }
}

#[test]
fn imported_memory_and_start_function_are_protected() {
    let wasm = wat::parse_str(
        r#"
        (module
          (import "env" "memory" (memory 1 2))
          (func (export "load") (param i32) (result i32)
            local.get 0 i32.load8_u))
        "#,
    )
    .unwrap();
    let engine = engine(true, 1);
    let module = Module::new(&engine, &wasm).unwrap();
    let mut store = Store::new();
    let memory = store.alloc_memory(1, Some(2)).unwrap();
    let mut linker = Linker::new();
    linker
        .define(&store, "env", "memory", store.get_memory(memory))
        .unwrap();
    let instance = linker.instantiate(&mut store, module).unwrap();
    let load = instance.get_func(&store, "load").unwrap();
    out_of_bounds(load.call(&mut store, &[Val::I32(65536)]));

    let start = wat::parse_str(
        r#"
        (module
          (memory 1)
          (func $start i32.const 65536 i32.load8_u drop)
          (start $start))
        "#,
    )
    .unwrap();
    let module = Module::new(&engine, &start).unwrap();
    assert!(matches!(
        Linker::new().instantiate(&mut store, module),
        Err(Error::Trap(TrapCode::MemoryOutOfBounds))
    ));
}

#[test]
fn software_mode_keeps_wasm_bounds_checks() {
    let wasm = wat::parse_str(
        "(module (memory 1) (func (export \"load\") (result i32) i32.const 65536 i32.load8_u))",
    )
    .unwrap();
    let module = Module::new(&engine(false, 1), &wasm).unwrap();
    let mut store = Store::new();
    let instance = Linker::new().instantiate(&mut store, module).unwrap();
    let load = instance.get_func(&store, "load").unwrap();
    assert!(load.call(&mut store, &[]).is_err());
}

#[test]
fn unsupported_strategy_reports_an_error() {
    let wasm = wat::parse_str("(module (memory 1))").unwrap();
    let engine = Engine::with_config(Config {
        strategy: Strategy::Jit,
        hardware_memory_checks: true,
        ..Config::default()
    });
    assert!(matches!(
        Module::new(&engine, &wasm),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn signal_handler_forwards_unrelated_signals() {
    const CHILD: &str = "VELOC_WASM_SIGNAL_CHAIN_CHILD";
    const EXIT_CODE: i32 = 47;

    if std::env::var_os(CHILD).is_some() {
        extern "C" fn previous_handler(_: i32) {
            unsafe { libc::_exit(EXIT_CODE) }
        }
        unsafe {
            libc::signal(libc::SIGSEGV, previous_handler as *const () as usize);
        }
        let wasm = wat::parse_str("(module (memory 1))").unwrap();
        Module::new(&engine(true, 0), &wasm).unwrap();
        unsafe { libc::raise(libc::SIGSEGV) };
        panic!("SIGSEGV did not reach the previous handler");
    }

    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "signal_handler_forwards_unrelated_signals"])
        .env(CHILD, "1")
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(EXIT_CODE));
}
