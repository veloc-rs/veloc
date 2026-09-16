use std::{env, fs};
use veloc_isle::rules::{Dialects, Program, Rust};
use veloc_opgen::Source;

fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let (output, code) = match args {
        [mode, input, definitions, output, arch] if mode == "target" => {
            let contracts = Source::load(definitions)?;
            (output, veloc_isle::target::compile(&fs::read_to_string(input)?, arch, &contracts)?)
        }
        [mode, input, output, source_name, source_path, target_name, target_path] if mode == "rules" => {
            let source = Source::load(source_path)?.parse()?;
            let target = Source::load(target_path)?.parse()?;
            let mut dialects = Dialects::default();
            dialects.insert(source_name, &source)?;
            dialects.insert(target_name, &target)?;
            let mut program = Program::compile(&fs::read_to_string(input)?, &dialects)?;
            program.infer_primitives(&dialects, source_name, target_name)?;
            (output, program.rust(Rust {
                function: "lower",
                context: "Context",
                source: (source_name, "SourceOpcode"),
                target: (target_name, "TargetOpcode"),
            })?)
        }
        _ => return Err("usage:\n  veloc-isle target INPUT DEFINITIONS OUTPUT ARCH\n  veloc-isle rules INPUT OUTPUT SOURCE_NAME SOURCE_OPS TARGET_NAME TARGET_OPS".into()),
    };
    fs::write(output, code)?;
    Ok(())
}

fn main() {
    if let Err(error) = run(&env::args().skip(1).collect::<Vec<_>>()) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
