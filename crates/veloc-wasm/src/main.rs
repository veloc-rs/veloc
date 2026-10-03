use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::{
    io::{self, Write},
    path::PathBuf,
};
use veloc_wasm::{
    Engine, Linker, Module, Store, Val,
    cli_support::{parse_opt_level, parse_val, read_wasm},
    engine::{Config, OptLevel, Strategy},
    module::Emit,
    veloc::codegen::{CodegenOptions, TargetArch},
    wasi::WasiCtx,
};

#[derive(Parser)]
#[command(version, about = "Veloc WebAssembly runtime")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile, instantiate and execute a WebAssembly module.
    Run(Run),
    /// Emit a compilation product without linking or executing the module.
    Emit(EmitArgs),
    /// List capabilities declared by a native backend.
    Inspect(Inspect),
}

#[derive(Args)]
struct CompileArgs {
    /// Input Wasm file (WAT requires the `wat` build feature).
    #[arg(value_name = "FILE")]
    file: PathBuf,
    /// Backend. Defaults to interpreter for run, jit for object emission.
    #[arg(short, long, value_enum, help_heading = "Compilation")]
    strategy: Option<Strategy>,
    /// Native backend for emission. Running native code requires the host target.
    #[arg(long, value_enum, help_heading = "Compilation")]
    target: Option<Architecture>,
    /// CPU model for native code generation; see `inspect cpus`.
    #[arg(long, help_heading = "Compilation")]
    cpu: Option<String>,
    /// ISA overrides, for example --cpu-features=-Zbb,-Zba.
    #[arg(
        long,
        value_delimiter = ',',
        require_equals = true,
        help_heading = "Compilation"
    )]
    cpu_features: Vec<String>,
    /// Optimization level for both MIR and native code: 0 or 1.
    #[arg(short = 'O', long, default_value = "1", value_parser = parse_opt_level, help_heading = "Compilation")]
    opt_level: OptLevel,
    /// Use a smaller MIR equality-search budget at O1.
    #[arg(long, help_heading = "Compilation")]
    fast_egraph: bool,
    /// Bounds checks: auto selects guard pages for supported native runtimes.
    #[arg(long, value_enum, default_value = "auto", help_heading = "Compilation")]
    memory_checks: veloc_wasm::engine::MemoryChecks,
    /// Print optimized MIR to stderr and continue.
    #[arg(long, help_heading = "Diagnostics")]
    dump_ir: bool,
    /// Print LIR after named passes to stderr; use '*' for every pass.
    #[arg(long, value_delimiter = ',', help_heading = "Diagnostics")]
    dump_after: Vec<String>,
    /// Limit LIR dumps to a function.
    #[arg(long, requires = "dump_after", help_heading = "Diagnostics")]
    dump_function: Option<String>,
    /// Write a Chrome/Perfetto compilation trace.
    #[arg(long, help_heading = "Diagnostics")]
    trace_file: Option<PathBuf>,
    /// Include optimization remarks and LIR snapshots in the trace.
    #[arg(long, requires = "trace_file", help_heading = "Diagnostics")]
    trace_details: bool,
    /// Print compilation timings and metrics to stderr.
    #[arg(long, help_heading = "Diagnostics")]
    print_stats: bool,
    /// Enable optimizer debug tags, for example --opt-debug=dce.
    #[arg(long, value_delimiter = ',', help_heading = "Diagnostics")]
    opt_debug: Vec<String>,
}

#[derive(Args)]
struct Run {
    #[command(flatten)]
    compile: CompileArgs,
    /// Export to call after instantiation.
    #[arg(short, long, default_value = "_start")]
    invoke: String,
    /// Export argument: i32:1, i64:2, f32:1.5 or f64:2.5. Repeat as needed.
    #[arg(long = "arg")]
    args: Vec<String>,
    /// WASI environment entry, KEY=VALUE. Repeat as needed.
    #[arg(long = "env", value_parser = parse_env)]
    env: Vec<(String, String)>,
    /// Print interpreter bytecode to stderr and continue.
    #[arg(long)]
    dump_bytecode: bool,
    /// WASI arguments, following '--'. argv[0] is the input filename.
    #[arg(last = true, value_name = "WASI_ARGS")]
    wasi_args: Vec<String>,
}

#[derive(Clone, Copy, ValueEnum)]
enum Product {
    Mir,
    Bytecode,
    Object,
}

impl From<Product> for Emit {
    fn from(value: Product) -> Self {
        match value {
            Product::Mir => Emit::Mir,
            Product::Bytecode => Emit::Bytecode,
            Product::Object => Emit::Object,
        }
    }
}

#[derive(Args)]
struct EmitArgs {
    #[command(flatten)]
    compile: CompileArgs,
    /// Product to generate. Object files use the selected backend's ELF format.
    #[arg(long, value_enum)]
    emit: Product,
    /// Destination, or '-' for stdout. Object emission requires an explicit destination.
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[derive(Clone, Copy, ValueEnum)]
enum Architecture {
    #[value(name = "x86_64")]
    X86_64,
    Riscv64,
}
impl Architecture {
    fn target(self) -> TargetArch {
        match self {
            Self::X86_64 => TargetArch::X86_64,
            Self::Riscv64 => TargetArch::Riscv64,
        }
    }
    fn host() -> Option<Self> {
        if cfg!(target_arch = "x86_64") {
            Some(Self::X86_64)
        } else if cfg!(target_arch = "riscv64") {
            Some(Self::Riscv64)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum Capability {
    Cpus,
    Features,
}

#[derive(Args)]
struct Inspect {
    #[arg(value_enum)]
    capability: Capability,
    /// Backend to inspect. Required on hosts without a native backend.
    #[arg(long, value_enum)]
    target: Option<Architecture>,
}

impl CompileArgs {
    fn engine(&self, product: Option<Emit>) -> Result<Engine> {
        let strategy = self.strategy.unwrap_or(if product == Some(Emit::Object) {
            Strategy::Jit
        } else {
            Strategy::Interpreter
        });
        let native = strategy != Strategy::Interpreter;
        if product == Some(Emit::Bytecode) && native {
            bail!("--emit bytecode requires --strategy interpreter");
        }
        if product == Some(Emit::Object) && !native {
            bail!("--emit object requires --strategy jit or fast-jit");
        }
        if !native && (self.target.is_some() || self.cpu.is_some() || !self.cpu_features.is_empty())
        {
            bail!("--target, --cpu and --cpu-features require a native strategy");
        }
        if strategy == Strategy::FastJit && (self.cpu.is_some() || !self.cpu_features.is_empty()) {
            bail!("fast-jit does not support CPU model or feature overrides");
        }
        if !self.dump_after.is_empty()
            && (!matches!(strategy, Strategy::Jit | Strategy::Auto) || product == Some(Emit::Mir))
        {
            bail!(
                "--dump-after requires the JIT codegen pipeline; MIR emission stops before codegen"
            );
        }
        if self.opt_level == OptLevel::None && !self.opt_debug.is_empty() {
            bail!("--opt-debug requires -O1");
        }
        if strategy == Strategy::FastJit && !cfg!(all(target_arch = "x86_64", target_os = "linux"))
        {
            bail!("fast-jit requires Linux x86-64");
        }
        let target = if native {
            let target = self.target.or_else(Architecture::host)
                .ok_or_else(|| anyhow!("no native backend for this host; use interpreter, or specify --target for emission"))?.target();
            if product.is_none() && Architecture::host().map(Architecture::target) != Some(target) {
                bail!("run requires the host target; use emit for cross compilation");
            }
            if strategy == Strategy::FastJit && target != TargetArch::X86_64 {
                bail!("fast-jit requires the x86_64 target");
            }
            target
        } else {
            Config::default().target
        };
        Engine::with_config(Config {
            strategy,
            target,
            cpu: self.cpu.clone().unwrap_or_else(|| "generic".into()),
            cpu_features: self.cpu_features.clone(),
            codegen: CodegenOptions {
                opt_level: self.opt_level,
                dump_after: self.dump_after.clone(),
                dump_function: self.dump_function.clone(),
                ..Default::default()
            },
            fast_egraph: self.fast_egraph,
            memory_checks: self.memory_checks,
            dump_ir: self.dump_ir,
            ir_names: self.dump_ir || product.is_some(),
            trace_file: self.trace_file.clone(),
            trace_details: self.trace_details,
            print_stats: self.print_stats,
            opt_debug: self.opt_debug.clone(),
            ..Default::default()
        })
        .map_err(Into::into)
    }
}

fn parse_env(value: &str) -> Result<(String, String), String> {
    let (key, value) = value.split_once('=').ok_or("expected KEY=VALUE")?;
    if key.is_empty() || key.contains('\0') || value.contains('\0') {
        return Err("invalid WASI environment entry".into());
    }
    Ok((key.into(), value.into()))
}

fn run(args: Run) -> Result<()> {
    let call_args = args
        .args
        .iter()
        .map(|arg| parse_val(arg))
        .collect::<Result<Vec<Val>>>()?;
    if args.dump_bytecode
        && args
            .compile
            .strategy
            .is_some_and(|s| s != Strategy::Interpreter)
    {
        bail!("--dump-bytecode requires --strategy interpreter");
    }
    let engine = args.compile.engine(None)?;
    let wasm = read_wasm(&args.compile.file)?;
    let module = Module::new(&engine, &wasm)?;
    if args.dump_bytecode {
        eprint!(
            "{}",
            veloc_wasm::veloc::interpreter::bytecode::format_module(module.ir())?
        );
    }
    let mut argv = vec![
        args.compile
            .file
            .to_string_lossy()
            .into_owned()
            .into_bytes(),
    ];
    argv.extend(args.wasi_args.into_iter().map(String::into_bytes));
    let mut store = Store::new();
    store.set_wasi(WasiCtx::new().with_args(argv).with_env(args.env));
    let mut linker = Linker::new();
    linker.add_wasi(&mut store)?;
    let instance = linker.instantiate(&mut store, module)?;
    let func = instance
        .get_func(&store, &args.invoke)
        .ok_or_else(|| anyhow!("export '{}' not found", args.invoke))?;
    let result = func.call(&mut store, &call_args)?;
    if !result.is_empty() {
        println!("{result:?}");
    }
    Ok(())
}

fn emit(args: EmitArgs) -> Result<()> {
    if matches!(args.emit, Product::Object) && args.output.is_none() {
        bail!("object emission requires -o FILE (or -o - for stdout)");
    }
    let kind = args.emit.into();
    let engine = args.compile.engine(Some(kind))?;
    let output = args.output.unwrap_or_else(|| PathBuf::from("-"));
    if args.compile.trace_file.as_ref() == Some(&output) {
        bail!("trace and compilation product must use different output paths");
    }
    let wasm = read_wasm(&args.compile.file)?;
    let bytes = Module::emit(&engine, &wasm, kind)?;
    if output.as_os_str() == "-" {
        io::stdout().lock().write_all(&bytes)?;
    } else {
        std::fs::write(&output, bytes)
            .with_context(|| format!("failed to write {}", output.display()))?;
    }
    Ok(())
}

fn inspect(args: Inspect) -> Result<()> {
    let arch = args
        .target
        .or_else(Architecture::host)
        .ok_or_else(|| anyhow!("specify --target x86_64 or --target riscv64"))?;
    let capabilities = veloc_wasm::veloc::codegen::target::capabilities(arch.target())?;
    let names = match args.capability {
        Capability::Cpus => capabilities.cpus,
        Capability::Features => capabilities.features,
    };
    let mut out = io::stdout().lock();
    for name in names {
        writeln!(out, "{name}")?;
    }
    Ok(())
}

fn main() -> Result<()> {
    #[cfg(feature = "logging")]
    env_logger::init();
    match Cli::parse().command {
        Command::Run(args) => run(args),
        Command::Emit(args) => emit(args),
        Command::Inspect(args) => inspect(args),
    }
}
