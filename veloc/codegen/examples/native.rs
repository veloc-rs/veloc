//! Small MIR-to-object driver; linking is intentionally left to the system linker.
use std::{env, error::Error, fs};
use veloc_codegen::{CodegenOptions, CodegenPipeline, TargetConfig, create_target_machine};
use veloc_mir::ModuleParser;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if !(args.len() == 2 || (args.len() == 3 && args[2] == "--no-opt")) {
        return Err("usage: native INPUT.mir OUTPUT.o [--no-opt]".into());
    }
    let source = fs::read_to_string(&args[0])?;
    let module = ModuleParser::new()
        .parse(&source)
        .map_err(|e| e.to_string())?;
    module.validate().map_err(|e| format!("{e:?}"))?;
    let target =
        create_target_machine(TargetConfig::default()).ok_or("x86-64 target unavailable")?;
    let pipeline = CodegenPipeline::with_options(
        &*target,
        CodegenOptions {
            optimize: args.len() == 2,
            ..Default::default()
        },
    );
    fs::write(&args[1], pipeline.compile_object(&module)?)?;
    Ok(())
}
