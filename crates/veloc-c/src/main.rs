use clap::{Parser, ValueEnum};
use std::{fs, path::PathBuf, process::Command};
use veloc_codegen::target::TargetArch;
use veloc_codegen::{
    CodegenOptions, CodegenPipeline, OptLevel, TargetConfig, create_target_machine,
};

#[derive(Clone, Copy, ValueEnum)]
enum Emit {
    Ast,
    Mir,
    Obj,
}

/// Compile C with the Veloc frontend, optimizer and native code generator.
#[derive(Parser)]
struct Args {
    input: PathBuf,
    #[arg(short)]
    output: PathBuf,
    #[arg(long, default_value = "riscv64")]
    target: String,
    #[arg(long, default_value = "generic")]
    cpu: String,
    #[arg(long, value_enum, default_value = "obj")]
    emit: Emit,
    #[arg(short = 'O', default_value_t = 1, value_parser = clap::value_parser!(u8).range(0..=1))]
    optimize: u8,
    /// External preprocessor only; optimization and code generation use Veloc.
    #[arg(long, default_value = "clang")]
    cpp: PathBuf,
    #[arg(long)]
    sysroot: Option<PathBuf>,
    #[arg(short = 'I')]
    include: Vec<PathBuf>,
    #[arg(short = 'D')]
    define: Vec<String>,
    /// Input has already been preprocessed (.i files imply this).
    #[arg(long)]
    preprocessed: bool,
    /// Validate MIR and every machine pass (enabled by default in debug builds).
    #[arg(long, default_value_t = cfg!(debug_assertions))]
    verify_ir: bool,
    /// Report compilation stages. Disable when benchmarking compilation latency.
    #[arg(long)]
    print_stats: bool,
    /// Write a Chrome/Perfetto compilation trace.
    #[arg(long)]
    trace_file: Option<PathBuf>,
    /// Use a trained policy with target features from the selected CPU model.
    #[arg(long)]
    policy: Option<PathBuf>,
    /// Record optimizer decisions and features for offline training.
    #[arg(long)]
    policy_trace: Option<PathBuf>,
}

fn main() {
    if let Err(error) = compile(Args::parse()) {
        eprintln!("veloc-c: {error}");
        std::process::exit(1);
    }
}

fn compile(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let profile = veloc_profile::Profile::new(veloc_profile::Config {
        mode: if args.trace_file.is_some() {
            veloc_profile::Mode::Trace
        } else if args.print_stats {
            veloc_profile::Mode::Summary
        } else {
            veloc_profile::Mode::Off
        },
        ..Default::default()
    });
    let compilation = profile.scope("compile", 0);
    emit(&args, &profile)?;
    compilation.success();
    let report = profile.report();
    if args.print_stats {
        eprintln!("{report}");
    }
    if let Some(path) = args.trace_file {
        fs::write(path, report.chrome_trace())?;
    }
    Ok(())
}

fn emit(args: &Args, profile: &veloc_profile::Profile) -> Result<(), Box<dyn std::error::Error>> {
    let arch = match args.target.as_str() {
        "riscv64" => TargetArch::Riscv64,
        _ => return Err("the C frontend currently supports the riscv64 Linux LP64D target".into()),
    };
    let source = if args.preprocessed || args.input.extension().is_some_and(|s| s == "i") {
        fs::read_to_string(&args.input)?
    } else {
        let mut cpp = Command::new(&args.cpp);
        cpp.args(["-E", "-P", "-std=c11", "-x", "c"]);
        cpp.arg(format!("--target={}-linux-gnu", args.target));
        if let Some(root) = &args.sysroot {
            cpp.arg(format!("--sysroot={}", root.display()));
        }
        for path in &args.include {
            cpp.arg("-I").arg(path);
        }
        for define in &args.define {
            cpp.arg("-D").arg(define);
        }
        let result = cpp.arg(&args.input).output()?;
        if !result.status.success() {
            return Err(String::from_utf8_lossy(&result.stderr).into_owned().into());
        }
        String::from_utf8(result.stdout)?
    };
    let ast = profile.measure("parse", 0, || veloc_c::parse(&source))?;
    if matches!(args.emit, Emit::Ast) {
        fs::write(&args.output, format!("{ast:#?}"))?;
        return Ok(());
    }
    let target = create_target_machine(TargetConfig {
        arch,
        cpu: args.cpu.clone(),
        external_calls: veloc_codegen::target::ExternalCalls::Linker,
        ..Default::default()
    })?;
    let c_target = veloc_c::CTargetModel::riscv64_linux(target.desc().data_layout)?;
    let policy = if args.policy.is_some() || args.policy_trace.is_some() {
        let mut policy = veloc_policy::Policy::new(
            veloc_codegen::target::policy_context(&*target),
            [
                veloc_optimizer::passes::inline::InlinePass::POLICY_SCHEMA,
                veloc_codegen::passes::schedule::SchedulePass::POLICY_SCHEMA,
            ],
        )?;
        if let Some(path) = &args.policy {
            policy = policy.with_model(path)?;
        }
        if args.policy_trace.is_some() {
            policy = policy.with_observations();
        }
        Some(std::sync::Arc::new(policy))
    } else {
        None
    };
    let mut module = profile.measure("frontend", 0, || {
        veloc_c::CodeGenContext::new(c_target).generate(&ast)
    })?;
    if args.verify_ir {
        module
            .validate()
            .map_err(|e| format!("frontend produced invalid MIR: {e:?}"))?;
    }
    if args.optimize != 0 {
        veloc_optimizer::PassManager::native_o1(target.desc().data_layout)
            .with_profile(profile.clone())
            .with_policy(policy.clone())
            .run_on_module(&mut module);
        if args.verify_ir {
            module
                .validate()
                .map_err(|e| format!("optimizer produced invalid MIR: {e:?}"))?;
        }
    }
    if matches!(args.emit, Emit::Mir) {
        fs::write(&args.output, module.to_string())?;
    } else {
        let pipeline = CodegenPipeline::new(
            &*target,
            CodegenOptions {
                verify: args.verify_ir,
                policy: policy.clone(),
                opt_level: if args.optimize == 0 {
                    OptLevel::None
                } else {
                    OptLevel::Default
                },
                ..Default::default()
            },
        )
        .with_profile(profile.clone());
        fs::write(&args.output, pipeline.compile_object(&module)?)?;
    }
    if let Some(path) = &args.policy_trace {
        policy.as_ref().unwrap().write_observations(path)?;
    }
    Ok(())
}
