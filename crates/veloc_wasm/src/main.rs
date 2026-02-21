use anyhow::{Context, Result, anyhow};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use veloc_wasm::{
    Engine, Module, Store,
    engine::{Config, Strategy},
};

#[derive(Parser, Debug)]
#[command(author, version, about = "Veloc WebAssembly runtime", long_about = None)]
struct Args {
    /// Path to the Wasm or WAT file
    #[arg(value_name = "FILE")]
    file: PathBuf,

    /// Function to invoke
    #[arg(short, long, default_value = "_start")]
    invoke: String,

    /// Execution strategy
    #[arg(short, long, value_enum, default_value = "interpreter")]
    strategy: Strategy,

    /// Dump generated IR to stdout
    #[arg(long, group = "ir-output")]
    dump_ir: bool,

    /// Output generated IR to a file (does not run the module)
    #[arg(short = 'o', long, value_name = "FILE", group = "ir-output")]
    output_ir: Option<PathBuf>,

    /// Optimization level (0, 1)
    #[arg(short, long, default_value = "1")]
    opt_level: u8,

    /// Output chrome trace JSON to file
    #[arg(long)]
    trace_file: Option<PathBuf>,

    /// Print optimization pass statistics
    #[arg(long)]
    print_stats: bool,

    /// Enable debug tags for optimization passes (e.g., --opt-debug=dce)
    #[arg(long, value_delimiter = ',')]
    opt_debug: Vec<String>,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    // Check if we only need to output IR
    let output_only = args.output_ir.is_some() || args.dump_ir;

    // 1. 初始化引擎
    let config = Config {
        strategy: args.strategy,
        dump_ir: args.dump_ir,
        ir_names: output_only,
        opt_level: args.opt_level,
        output_ir: args.output_ir,
        trace_file: args.trace_file,
        print_stats: args.print_stats,
        opt_debug: args.opt_debug,
    };
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

    // If only outputting IR, we're done
    if output_only {
        return Ok(());
    }

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
