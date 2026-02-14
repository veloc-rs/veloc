use anyhow::{Context, Result, anyhow};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use veloc_wasm::{
    Engine, Module, Store,
    engine::{Config, Strategy},
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the Wasm or WAT file
    #[arg(value_name = "FILE")]
    file: PathBuf,

    /// Function to invoke
    #[arg(short, long, default_value = "_start")]
    invoke: String,

    /// Execution strategy (auto, jit, interpreter)
    #[arg(short, long, default_value = "interpreter")]
    strategy: String,

    /// Dump generated IR to stdout
    #[arg(long)]
    dump_ir: bool,

    /// Enabled variable names in IR output (improves readability)
    #[arg(long)]
    ir_names: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 1. 初始化引擎
    let mut config = Config::default();
    config.strategy = match args.strategy.to_lowercase().as_str() {
        "jit" => Strategy::Jit,
        "interpreter" => Strategy::Interpreter,
        _ => Strategy::Auto,
    };
    config.dump_ir = args.dump_ir;
    config.ir_names = args.ir_names;
    let engine = Arc::new(Engine::with_config(config));

    // 2. 读取并解析 Wasm 字节码
    let wasm_bin = if args.file.extension().and_then(|s| s.to_str()) == Some("wat") {
        wat::parse_file(&args.file)
            .with_context(|| format!("Failed to parse WAT file: {:?}", args.file))?
    } else {
        std::fs::read(&args.file)
            .with_context(|| format!("Failed to read Wasm file: {:?}", args.file))?
    };

    // 3. 编译模块
    let module =
        Module::new(&engine, &wasm_bin).map_err(|e| anyhow!("Failed to create Module: {:?}", e))?;

    // 4. 初始化状态存储
    let mut store = Store::new();

    // 5. 实例化 (使用 Linker)
    let mut linker = veloc_wasm::linker::Linker::new();

    // 6. 如果启用了 WASI

    let wasi_ctx = veloc_wasm::wasi::default_wasi_ctx();
    store.set_wasi(wasi_ctx);
    linker
        .add_wasi(&mut store)
        .map_err(|e| anyhow!("Failed to add WASI: {:?}", e))?;

    let instance_id = linker
        .instantiate(&mut store, module)
        .map_err(|e| anyhow!("Failed to instantiate module: {:?}", e))?;

    // 7. 调用导出函数
    let func = instance_id
        .get_func(&store, &args.invoke)
        .ok_or_else(|| anyhow!("Export '{}' not found", args.invoke))?;

    let result = func
        .call(&mut store, &[])
        .map_err(|e| anyhow!("Wasm trap during execution: {:?}", e))?;

    if !result.is_empty() {
        println!("Result: {:?}", result);
    }

    Ok(())
}
