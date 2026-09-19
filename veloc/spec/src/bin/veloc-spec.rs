use clap::{ArgGroup, Parser, builder::TypedValueParser};
use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
};
use veloc_spec::{Decisions, Emit, Options, Source, Target, ValueRules};

/// Compile instruction definitions into selected Rust artifacts.
#[derive(Parser)]
#[command(version, about, group(ArgGroup::new("destination").args(["output", "out_dir"]).required(true)))]
struct Args {
    /// Root definition file; imports are resolved relative to it.
    input: PathBuf,
    /// Artifacts to generate (comma-separated or repeated).
    #[arg(long, required = true, value_delimiter = ',', value_parser = clap::builder::PossibleValuesParser::new(Emit::ALL.iter().map(|e| e.name())).map(|s| s.parse::<Emit>().expect("listed artifact")))]
    emit: Vec<Emit>,
    /// Write one artifact to a file, or '-' for standard output.
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Write selected artifacts under their standard filenames.
    #[arg(long)]
    out_dir: Option<PathBuf>,
    /// Target architecture for machine-specific artifacts.
    #[arg(long, requires = "definitions")]
    arch: Option<String>,
    /// Instruction contracts for machine-specific artifacts.
    #[arg(long)]
    definitions: Option<PathBuf>,
    /// Rust namespace for generated interfaces.
    #[arg(long)]
    namespace: Option<String>,
    /// Source dialect definition module for cross-IR rules.
    #[arg(long)]
    source_definitions: Option<PathBuf>,
    /// Namespace of the source dialect (or decision rule dialect).
    #[arg(long)]
    source_dialect: Option<String>,
    /// Rust source opcode path.
    #[arg(long)]
    source_opcode: Option<String>,
    /// Namespace of the destination dialect.
    #[arg(long)]
    target_dialect: Option<String>,
    /// Rust destination opcode path.
    #[arg(long)]
    target_opcode: Option<String>,
    /// Generated Rust function name.
    #[arg(long)]
    function: Option<String>,
    /// Selector host trait path or generated value-adapter host context name.
    #[arg(long)]
    context: Option<String>,
    /// Infer identity rules for compatible semantic primitives.
    #[arg(long)]
    infer_primitives: bool,
    /// Decision result type declared in the module.
    #[arg(long)]
    result: Option<String>,
    /// Explicit rewrite_interface declaration for value construction.
    #[arg(long)]
    value_interface: Option<String>,
    /// Rust attribute enum used by value construction.
    #[arg(long)]
    field: Option<String>,
    /// Rust adapter for value rewrites.
    #[arg(long)]
    value_adapter: Option<String>,
    /// Shared Rust action constructor for generated and host rewrites.
    #[arg(long)]
    rewrite: Option<String>,
    /// Rust constructor for a legal decision.
    #[arg(long)]
    legal_action: Option<String>,
}

fn required<'a>(
    value: Option<&'a str>,
    option: &str,
) -> Result<&'a str, Box<dyn std::error::Error>> {
    value.ok_or_else(|| format!("selected artifact requires --{option}").into())
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.output.is_some() && args.emit.iter().any(|kind| *kind != args.emit[0]) {
        return Err(
            "--output requires one distinct artifact; use --out-dir for multiple artifacts".into(),
        );
    }
    let source = Source::load(&args.input)?;
    let definitions = args.definitions.as_ref().map(Source::load).transpose()?;
    let source_definitions = args
        .source_definitions
        .as_ref()
        .map(Source::load)
        .transpose()?;
    let rules = if args.emit.contains(&Emit::Rules) {
        Some(ValueRules {
            source: source_definitions
                .as_ref()
                .ok_or("rules require --source-definitions")?,
            target: definitions.as_ref().ok_or("rules require --definitions")?,
            rust: veloc_spec::rules::Rust {
                function: required(args.function.as_deref(), "function")?,
                context: required(args.context.as_deref(), "context")?,
                source: (
                    required(args.source_dialect.as_deref(), "source-dialect")?,
                    required(args.source_opcode.as_deref(), "source-opcode")?,
                ),
                target: (
                    required(args.target_dialect.as_deref(), "target-dialect")?,
                    required(args.target_opcode.as_deref(), "target-opcode")?,
                ),
            },
            infer_primitives: args.infer_primitives,
        })
    } else {
        None
    };
    let decisions = if args.emit.contains(&Emit::Decisions) {
        Some(Decisions {
            definitions: definitions
                .as_ref()
                .ok_or("decisions require --definitions")?,
            rust: veloc_spec::rules::DecisionRust {
                dialect: required(args.source_dialect.as_deref(), "source-dialect")?,
                function: required(args.function.as_deref(), "function")?,
                opcode: required(args.source_opcode.as_deref(), "source-opcode")?,
                result: required(args.result.as_deref(), "result")?,
                value_interface: required(args.value_interface.as_deref(), "value-interface")?,
                field: required(args.field.as_deref(), "field")?,
                value_adapter: required(args.value_adapter.as_deref(), "value-adapter")?,
                rewrite: required(args.rewrite.as_deref(), "rewrite")?,
                legal_action: required(args.legal_action.as_deref(), "legal-action")?,
            },
        })
    } else {
        None
    };
    let target = if let Some(arch) = args.arch.as_deref() {
        Some(Target {
            input: source_definitions
                .as_ref()
                .map(|source| -> Result<_, Box<dyn std::error::Error>> {
                    Ok((
                        required(args.source_dialect.as_deref(), "source-dialect")?,
                        source,
                    ))
                })
                .transpose()?,
            arch,
            context: required(args.context.as_deref(), "context")?,
            definitions: definitions
                .as_ref()
                .ok_or("target requires --definitions")?,
        })
    } else {
        None
    };
    let artifacts = source.generate(
        &args.emit,
        Options {
            target,
            interfaces: args.namespace.as_deref(),
            rules,
            decisions,
        },
    )?;
    if let Some(dir) = args.out_dir {
        artifacts.write(&dir)?;
    } else if let Some(path) = args.output {
        let text = artifacts.get(args.emit[0]).expect("requested artifact");
        if path.as_os_str() == "-" {
            io::stdout().lock().write_all(text.as_bytes())?;
        } else {
            fs::write(path, text)?;
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run(Args::parse()) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
