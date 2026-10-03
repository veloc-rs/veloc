use anyhow::{Result, anyhow, bail};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use veloc_wasm::{
    Engine, Module, Store, Val,
    cli_support::{parse_opt_level, parse_val, read_wasm},
    engine::{Config, OptLevel, Strategy},
    linker::Linker,
};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Compare interpreter and JIT results for one Wasm export"
)]
struct Args {
    /// Path to a Wasm or WAT file.
    #[arg(value_name = "FILE")]
    file: PathBuf,

    /// Exported function to invoke.
    #[arg(short, long)]
    invoke: String,

    /// Function arguments, formatted as `i32:1`, `i64:2`, `f32:1.5`, `f64:2.5`.
    #[arg(long = "arg")]
    args: Vec<String>,

    /// Optimization level used by both engines.
    #[arg(short = 'O', long, default_value = "1", value_parser = parse_opt_level)]
    opt_level: OptLevel,

    /// Dump generated IR for both engines.
    #[arg(long)]
    dump_ir: bool,
}

struct RunOutcome {
    results: Result<Vec<Val>>,
}

fn main() -> Result<()> {
    #[cfg(feature = "logging")]
    env_logger::init();
    let args = Args::parse();
    let wasm = read_wasm(&args.file)?;
    let call_args = args
        .args
        .iter()
        .map(|arg| parse_val(arg))
        .collect::<Result<Vec<_>>>()?;

    let interp = run_once(
        Strategy::Interpreter,
        &wasm,
        &args.invoke,
        &call_args,
        args.opt_level,
        args.dump_ir,
    );
    let jit = run_once(
        Strategy::Jit,
        &wasm,
        &args.invoke,
        &call_args,
        args.opt_level,
        args.dump_ir,
    );

    match (&interp.results, &jit.results) {
        (Ok(lhs), Ok(rhs)) if lhs == rhs => {
            println!("match");
            println!("interp: {:?}", lhs);
            println!("jit:    {:?}", rhs);
            Ok(())
        }
        (Ok(lhs), Ok(rhs)) => {
            println!("mismatch");
            println!("interp: {:?}", lhs);
            println!("jit:    {:?}", rhs);
            bail!("interpreter and JIT returned different results");
        }
        (lhs, rhs) => {
            println!("execution diverged");
            print_outcome("interp", lhs);
            print_outcome("jit", rhs);
            bail!("interpreter and JIT did not agree");
        }
    }
}

fn run_once(
    strategy: Strategy,
    wasm: &[u8],
    export: &str,
    args: &[Val],
    opt_level: OptLevel,
    dump_ir: bool,
) -> RunOutcome {
    let outcome = (|| -> Result<Vec<Val>> {
        let config = Config {
            codegen: veloc_wasm::veloc::codegen::CodegenOptions {
                opt_level,
                ..Default::default()
            },
            cpu: "generic".into(),
            cpu_features: Vec::new(),
            strategy,
            memory_checks: veloc_wasm::engine::MemoryChecks::Software,
            dump_ir,
            ir_names: false,
            verify_ir: true,
            fast_egraph: false,
            trace_file: None,
            trace_details: false,
            print_stats: false,
            opt_debug: Vec::new(),
            ..Config::default()
        };
        let engine = Arc::new(Engine::with_config(config)?);
        let module = Module::new(&engine, wasm)
            .map_err(|err| anyhow!("failed to compile module in {:?} mode: {}", strategy, err))?;
        let mut store = Store::new();
        let mut linker = Linker::new();
        let wasi_ctx = veloc_wasm::wasi::default_wasi_ctx();
        store.set_wasi(wasi_ctx);
        linker
            .add_wasi(&mut store)
            .map_err(|err| anyhow!("failed to add WASI in {:?} mode: {}", strategy, err))?;
        let instance = linker.instantiate(&mut store, module).map_err(|err| {
            anyhow!(
                "failed to instantiate module in {:?} mode: {}",
                strategy,
                err
            )
        })?;
        let func = instance
            .get_func(&store, export)
            .ok_or_else(|| anyhow!("export `{}` not found in {:?} mode", export, strategy))?;
        func.call(&mut store, args)
            .map_err(|err| anyhow!("execution failed in {:?} mode: {}", strategy, err))
    })();

    RunOutcome { results: outcome }
}

fn print_outcome(label: &str, outcome: &Result<Vec<Val>>) {
    match outcome {
        Ok(values) => println!("{label}: ok {:?}", values),
        Err(err) => println!("{label}: err {err}"),
    }
}
