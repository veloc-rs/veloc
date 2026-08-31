use veloc_wasm::engine::{Config, Strategy};
use veloc_wasm::{Engine, Linker, Module, Store, Val};

#[test]
fn tail_threaded_dispatch_handles_control_flow_and_calls() {
    let wasm = wat::parse_str(
        r#"
        (module
          (type $binary (func (param i32 i32) (result i32)))
          (table 1 funcref)
          (elem (i32.const 0) $add)

          (func $add (type $binary)
            local.get 0
            local.get 1
            i32.add)

          (func $sum_to (param $n i32) (result i32)
            (local $sum i32)
            block $done
              loop $next
                local.get $n
                i32.eqz
                br_if $done
                local.get $sum
                local.get $n
                i32.add
                local.set $sum
                local.get $n
                i32.const 1
                i32.sub
                local.set $n
                br $next
              end
            end
            local.get $sum)

          (func (export "run") (result i32)
            i32.const 1000
            call $sum_to
            i32.const 7
            i32.const 5
            i32.const 0
            call_indirect (type $binary)
            i32.add))
        "#,
    )
    .unwrap();

    let engine = Engine::with_config(Config {
        strategy: Strategy::Interpreter,
        ..Config::default()
    });
    let module = Module::new(&engine, &wasm).unwrap();
    let mut store = Store::new();
    let instance = Linker::new().instantiate(&mut store, module).unwrap();
    let run = instance.get_func(&store, "run").unwrap();

    assert_eq!(run.call(&mut store, &[]).unwrap(), vec![Val::I32(500_512)]);
}
