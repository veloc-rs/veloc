//! File-driven tests of public compiler stages. No shell commands in fixtures.
use std::{fs, path::Path};

use filecheck::{CheckerBuilder, NO_VARIABLES};
use libtest_mimic::{Arguments, Trial};
use veloc_analyzer::{AnalysisManager, UseDefAnalysis};
use veloc_mir::{Module, ModuleParser};
use veloc_optimizer::{Metrics, PassManager, passes::function::simplify::run_simplify};

type Result<T> = std::result::Result<T, String>;

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("cases");
    let mut tests = Vec::new();
    discover(&root, &root, &mut tests).expect("discover file tests");
    assert!(!tests.is_empty(), "no file tests discovered");
    tests.push(Trial::test("runner/rejects-invalid-expectations", || {
        let input = "local function main() -> void\nblock0():\n  return\n";
        for directives in [
            "// check: return",
            "// run: roundtrip",
            "// run: roundtrip\n// run: roundtrip\n// check: return",
            "// run: unknown\n// check: return",
            "// run: roundtrip\n// check: missing",
            "// run: roundtrip\n// check: main\n// not: return",
            "// run: parse-error\n// check: error",
            "// run: validate-error\n// check: error",
        ] {
            if run(&format!("{directives}\n{input}")).is_ok() {
                return Err(format!("accepted invalid expectations: {directives}").into());
            }
        }
        Ok(())
    }));
    libtest_mimic::run(&Arguments::from_args(), tests).exit();
}

fn discover(root: &Path, dir: &Path, tests: &mut Vec<Trial>) -> std::io::Result<()> {
    let mut paths = fs::read_dir(dir)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.sort();
    for path in paths {
        if path.is_dir() {
            discover(root, &path, tests)?;
        } else if matches!(
            path.extension().and_then(|ext| ext.to_str()),
            Some("mir" | "ops")
        ) {
            let text = fs::read_to_string(&path)?;
            let name = path.strip_prefix(root).unwrap().display().to_string();
            // A separator starts an independent input, not a transformation of
            // a shared fixture. Every failure can be reproduced from its section.
            for (index, section) in text.split("// -----").enumerate() {
                if section.trim().is_empty() {
                    continue;
                }
                let (label, source) = if index == 0 {
                    (name.clone(), section.to_string())
                } else {
                    let (label, source) = section.split_once('\n').unwrap_or((section, ""));
                    (format!("{name}/{}", label.trim()), source.to_string())
                };
                tests.push(Trial::test(label, move || run(&source).map_err(Into::into)));
            }
        }
    }
    Ok(())
}

fn run(source: &str) -> Result<()> {
    let mut mode = None;
    let mut checks = CheckerBuilder::new();
    let mut positive = false;
    for line in source
        .lines()
        .filter_map(|line| line.trim().strip_prefix("//"))
    {
        let line = line.trim();
        if let Some(command) = line.strip_prefix("run:") {
            if mode.replace(command.trim()).is_some() {
                return Err("each case must have exactly one run: directive".into());
            }
        } else {
            positive |= ["check:", "unordered:", "nextln:", "sameln:"]
                .iter()
                .any(|prefix| line.starts_with(prefix));
            checks.directive(line).map_err(|error| error.to_string())?;
        }
    }
    let mode = mode.ok_or("missing run: directive")?;
    if !positive {
        return Err("each case must check an observable output, not only absence".into());
    }
    let output = execute(mode, source)?;
    let checker = checks.finish();
    if !checker
        .check(&output, NO_VARIABLES)
        .map_err(|error| error.to_string())?
    {
        let (_, explanation) = checker
            .explain(&output, NO_VARIABLES)
            .map_err(|error| error.to_string())?;
        return Err(format!("{explanation}\nactual output:\n{output}"));
    }
    Ok(())
}

fn rejected<T, E: std::fmt::Display>(result: std::result::Result<T, E>) -> Result<String> {
    match result {
        Ok(_) => Err("expected rejection, but the input was accepted".into()),
        Err(error) => Ok(error.to_string()),
    }
}

fn execute(mode: &str, source: &str) -> Result<String> {
    match mode {
        "opgen-error" => {
            let builtins = concat!(
                include_str!("../../veloc/mir/defs/types.ops"),
                "\n",
                include_str!("../../veloc/mir/defs/builtins.ops"),
                "\n",
                include_str!("../../veloc/mir/defs/comparisons.ops"),
                "\n",
            );
            rejected(veloc_opgen::compile_mir(&format!("{builtins}{source}")))
        }
        "fixture" | "fixture-error" | "fixture-validate-error" => {
            let parsed = veloc_test_mir::ModuleParser::new().parse(source);
            if mode == "fixture-error" {
                // Like normal MIR tests, distinguish parser rejection from a
                // validator failure; a panic never counts as a diagnostic.
                return rejected(parsed);
            }
            let module = parsed.map_err(|error| error.to_string())?;
            if mode == "fixture-validate-error" {
                return rejected(module.validate());
            }
            module.validate().map_err(|error| error.to_string())?;
            let text = module.to_string();
            let reparsed = veloc_test_mir::ModuleParser::new()
                .parse(&text)
                .map_err(|error| error.to_string())?;
            reparsed.validate().map_err(|error| error.to_string())?;
            if reparsed.to_string() != text {
                return Err("fixture text is not canonical after round-trip".into());
            }
            Ok(text)
        }
        "parse-error" => rejected(ModuleParser::new().parse(source)),
        "validate-error" => {
            let module = ModuleParser::new()
                .parse(source)
                .map_err(|error| error.to_string())?;
            rejected(module.validate())
        }
        "roundtrip" | "simplify" | "o1" | "lower" | "execute" => {
            let module = ModuleParser::new()
                .parse(source)
                .map_err(|error| error.to_string())?;
            module.validate().map_err(|error| error.to_string())?;
            let module = match mode {
                "simplify" => simplify(module)?,
                "o1" => optimize(&module),
                "execute" => {
                    let optimized = optimize(&module);
                    optimized.validate().map_err(|error| error.to_string())?;
                    let before = interpret(module)?;
                    let after = interpret(optimized)?;
                    if before != after {
                        return Err(format!(
                            "execution changed after optimization:\n{before}\n{after}"
                        ));
                    }
                    return Ok(before);
                }
                "lower" => {
                    let lir = veloc_codegen::translate::IRTranslator::new(&module)
                        .translate_module()
                        .map_err(|error| error.to_string())?;
                    return Ok(lir
                        .func_order
                        .iter()
                        .map(|&id| {
                            let function = &lir.functions[id];
                            let mut text = function.format_for_dump();
                            // Expose result types as well as the operand identities
                            // already present in the LIR dump.
                            for (reg, data) in &function.vregs {
                                text.push_str(&format!("type v{}: {}\n", reg.as_u32(), data.ty));
                            }
                            text
                        })
                        .collect());
                }
                _ => module,
            };
            roundtrip(&module)
        }
        _ => Err(format!("unknown run mode `{mode}`")),
    }
}

fn optimize(module: &Module) -> Module {
    let mut data = (**module).clone();
    PassManager::new_o1().run_on_module(&mut data);
    Module::new(data)
}

fn interpret(module: Module) -> Result<String> {
    use veloc_interpreter::{Interpreter, Program, VirtualMemory};
    // These portable execution fixtures use stack memory only. Accesses to
    // external memory fail explicitly rather than dereferencing host addresses.
    struct NoMemory;
    impl VirtualMemory for NoMemory {
        fn translate_addr(&self, _: usize, _: usize) -> Option<*mut u8> {
            None
        }
    }
    let main = module
        .find_function_by_name("main")
        .ok_or("execute needs a main function")?;
    let signature = &module.signatures[module.functions[main].signature];
    if !signature.params.is_empty() {
        return Err("execute requires main() with no parameters".into());
    }
    let widths = signature
        .returns
        .iter()
        .map(|ty| {
            ty.as_scalar()
                .and_then(|_| ty.element_bits())
                .filter(|&width| width <= 64)
                .ok_or_else(|| format!("execute does not support return type {ty}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut program = Program::new();
    let id = program
        .builder(module)
        .finish()
        .map_err(|error| error.to_string())?;
    match Interpreter::new().run_function(&program, &NoMemory, id, main, &[]) {
        Ok(values) => {
            if values.len() != widths.len() {
                return Err("interpreter returned the wrong number of values".into());
            }
            // The high bits of an i8/i16/i32 register are not part of the
            // returned value. Compare the bits specified by its MIR type.
            Ok(format!(
                "returned: [{}]",
                values
                    .iter()
                    .zip(widths)
                    .map(|(value, width)| {
                        let bits = value.0 & (u64::MAX >> (64 - width));
                        format!("0x{bits:x}")
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        }
        Err(error) => Ok(format!("trapped: {error}")),
    }
}

fn roundtrip(module: &Module) -> Result<String> {
    module.validate().map_err(|error| error.to_string())?;
    let text = module.to_string();
    let reparsed = ModuleParser::new()
        .parse(&text)
        .map_err(|error| error.to_string())?;
    reparsed.validate().map_err(|error| error.to_string())?;
    if reparsed.to_string() != text {
        return Err(format!("text is not canonical after round-trip:\n{text}"));
    }
    Ok(text)
}

fn simplify(module: Module) -> Result<Module> {
    let mut data = (*module).clone();
    for (_, function) in data.functions.iter_mut() {
        if function.entry_block.is_none() {
            continue;
        }
        let mut analyses = AnalysisManager::new();
        analyses.use_def(function);
        let mut metrics = Metrics::default();
        run_simplify(function, &mut analyses, false, &mut metrics);
        let rebuilt = UseDefAnalysis::new(function);
        for value in function.dfg.values.keys() {
            let mut actual = analyses.use_def(function).users_of(value).to_vec();
            let mut expected = rebuilt.users_of(value).to_vec();
            actual.sort_unstable();
            expected.sort_unstable();
            if actual != expected {
                return Err(format!(
                    "{}: stale use-def cache for {value:?}",
                    function.name
                ));
            }
        }
        let revision = function.revision();
        if run_simplify(function, &mut analyses, false, &mut metrics)
            || function.revision() != revision
        {
            return Err(format!(
                "{}: simplify did not reach a fixed point",
                function.name
            ));
        }
    }
    Ok(Module::new(data))
}
