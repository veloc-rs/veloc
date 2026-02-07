use veloc_wasm::{Engine, Module, Store};

fn main() {
    println!("=== Wasmtime-style JIT Runtime ===");

    // 1. 初始化引擎
    let mut config = veloc_wasm::engine::Config::default();
    config.strategy = veloc_wasm::engine::Strategy::Interpreter;
    let engine = std::sync::Arc::new(Engine::with_config(config));

    // 2. 准备 Wasm 字节码 (包含本地变量读写与复用)
    // 计算: (x = 10; (x + x) * x) = 200
    let wasm_text = r#"
        (module
            (func (export "main") (result i32)
                (local i32)
                i32.const 10
                local.set 0
                (i32.mul
                    (i32.add (local.get 0) (local.get 0))
                    (local.get 0)
                )
            )
        )
    "#;
    let wasm_bin = wat::parse_str(wasm_text).expect("Failed to parse WAT");

    // 3. 编译模块
    println!("[Module] Compiling Wasm with locals and value reuse...");
    let module = Module::new(engine.clone(), &wasm_bin).expect("Failed to create Module");

    // 4. 初始化状态存储
    let mut store = Store::new();

    // 5. 实例化 (使用 Linker)
    println!("[Instance] Instantiating via Linker...");
    let mut linker = veloc_wasm::linker::Linker::new();
    let instance_id = linker
        .instantiate(&mut store, module)
        .expect("Failed to instantiate via Linker");

    // 6. 调用到处函数
    println!("[Runtime] Calling 'main'...");
    let main_func = instance_id
        .get_func(&store, "main")
        .expect("Export 'main' not found");
    let result = main_func.call(&mut store, &[]).expect("wasm trap");

    println!(
        "\n[Result] Execution success! Result: {:?} (Expect: 200)",
        result
    );
}
