use anyhow::Result;
use clap::Parser;
use hashbrown::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use wasmparser::RefType;
use wast::parser::{self, ParseBuffer};
use wast::{QuoteWat, Wast, WastArg, WastDirective, WastExecute, WastRet, Wat};

use veloc::interpreter::InterpreterValue;
use veloc_wasm::engine::{Config, Strategy};
use veloc_wasm::instance::ExternMap;
use veloc_wasm::linker::Linker;
use veloc_wasm::{Engine, Extern, InstanceId, Module, Store, Val};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to a .wast file or a directory containing .wast files
    #[arg(value_name = "PATH")]
    path: Option<PathBuf>,

    /// Execution strategy: interp or jit
    #[arg(short, long, default_value = "interp")]
    strategy: String,

    /// Dump IR to stdout
    #[arg(long)]
    dump_ir: bool,

    /// Show detailed test output
    #[arg(short, long)]
    verbose: bool,
}

pub struct SpecRunner {
    engine: Arc<Engine>,
    store: Store,
    instances: Vec<InstanceId>,
    named_instances: HashMap<String, InstanceId>,
    modules: Vec<Module>,
    named_modules: HashMap<String, Module>,
    registered: HashMap<String, ExternMap>,
    mode_name: String,
    dump_ir: bool,
    verbose: bool,
}

impl SpecRunner {
    pub fn new(strategy: Strategy, name: &str, dump_ir: bool, verbose: bool) -> Self {
        let mut config = Config::default();
        config.strategy = strategy;
        let engine = Arc::new(Engine::with_config(config));
        let mut store = Store::new();
        let mut registered = HashMap::new();

        let mut spectest = HashMap::new();
        let mem_id = store.alloc_memory(1, Some(2)).unwrap();
        spectest.insert(
            "memory".to_string(),
            Extern::Memory(store.get_memory(mem_id)),
        );
        let table_id = store.alloc_table(10, RefType::FUNCREF).unwrap();
        spectest.insert(
            "table".to_string(),
            Extern::Table(store.get_table(table_id)),
        );

        // globals
        let g_i32 = store.alloc_global(Val::I32(666), false);
        spectest.insert(
            "global_i32".to_string(),
            Extern::Global(store.get_global(g_i32)),
        );
        let g_i64 = store.alloc_global(Val::I64(666), false);
        spectest.insert(
            "global_i64".to_string(),
            Extern::Global(store.get_global(g_i64)),
        );
        let g_f32 = store.alloc_global(Val::F32(666.6), false);
        spectest.insert(
            "global_f32".to_string(),
            Extern::Global(store.get_global(g_f32)),
        );
        let g_f64 = store.alloc_global(Val::F64(666.6), false);
        spectest.insert(
            "global_f64".to_string(),
            Extern::Global(store.get_global(g_f64)),
        );

        let mut linker = Linker::new();
        linker.func_wrap(&mut store, "spectest", "print_i32", |args| {
            println!("{}: i32", args[0].unwarp_i32());
            InterpreterValue::None
        });
        linker.func_wrap(&mut store, "spectest", "print_i64", |args| {
            println!("{}: i64", args[0].unwarp_i64());
            InterpreterValue::None
        });
        linker.func_wrap(&mut store, "spectest", "print_f32", |args| {
            println!("{}: f32", args[0].unwarp_f32());
            InterpreterValue::None
        });
        linker.func_wrap(&mut store, "spectest", "print_f64", |args| {
            println!("{}: f64", args[0].unwarp_f64());
            InterpreterValue::None
        });

        for ((m_name, f_name), ext) in linker.definitions() {
            if m_name == "spectest" {
                spectest.insert(f_name.clone(), ext.clone());
            }
        }
        registered.insert("spectest".to_string(), spectest);

        Self {
            engine,
            store,
            instances: Vec::new(),
            named_instances: HashMap::new(),
            modules: Vec::new(),
            named_modules: HashMap::new(),
            registered,
            mode_name: name.to_string(),
            dump_ir,
            verbose,
        }
    }

    pub fn instantiate(&mut self, wasm_bin: &[u8], id: Option<&str>) -> Result<InstanceId> {
        let m = Module::new(self.engine.clone(), wasm_bin)?;
        let mut linker = Linker::new();
        for (name, exports) in &self.registered {
            for (field, ext) in exports {
                linker.define(&self.store, name, field, ext.clone())?;
            }
        }
        let inst = linker.instantiate(&mut self.store, m.clone())?;
        self.instances.push(inst);
        self.modules.push(m.clone());
        if let Some(id) = id {
            self.named_instances.insert(id.to_string(), inst);
            self.named_modules.insert(id.to_string(), m);
        }
        Ok(inst)
    }

    pub fn dump_all_ir(&self) {
        for (i, m) in self.modules.iter().enumerate() {
            println!(">>>> DUMP IR for indexed module {} <<<<", i);
            println!("{}", m.ir());
        }
        for (name, m) in self.named_modules.iter() {
            println!(">>>> DUMP IR for named module {:?} <<<<", name);
            println!("{}", m.ir());
        }
        println!(">>>> END DUMP IR <<<<");
    }

    pub fn dump_ir_for(&self, module_name: Option<&str>) {
        if let Some(name) = module_name {
            if let Some(m) = self.named_modules.get(name) {
                println!(">>>> DUMP IR for module {:?} <<<<", name);
                println!("{}", m.ir());
            }
        } else {
            if let Some(m) = self.modules.last() {
                println!(">>>> DUMP IR for last module <<<<");
                println!("{}", m.ir());
            }
        }
        println!(">>>> END DUMP IR <<<<");
    }

    pub fn register(&mut self, name: &str, module: Option<&str>) {
        let inst = module
            .and_then(|m| self.named_instances.get(m).copied())
            .or_else(|| self.instances.last().copied())
            .expect("instance not found");
        self.registered
            .insert(name.to_string(), inst.exports(&self.store));
    }

    pub fn call(&mut self, module: Option<&str>, name: &str, args: &[WastArg]) -> Result<Vec<i64>> {
        let inst = module
            .and_then(|m| self.named_instances.get(m).copied())
            .or_else(|| self.instances.last().copied())
            .expect("instance not found");
        let func = inst
            .get_func(&self.store, name)
            .ok_or_else(|| anyhow::anyhow!("function {} not found", name))?;
        let val_args: Vec<_> = args.iter().map(wast_arg_to_val).collect();
        match func.call(&mut self.store, &val_args) {
            Ok(res) => Ok(res.iter().map(|v| v.as_i64()).collect()),
            Err(e) => Err(anyhow::anyhow!(
                "{} call failed for {}: {}",
                self.mode_name,
                name,
                e
            )),
        }
    }

    pub fn assert_trap(&mut self, module: Option<&str>, name: &str, args: &[WastArg]) {
        let val_args: Vec<_> = args.iter().map(wast_arg_to_val).collect();
        let inst = module
            .and_then(|m| self.named_instances.get(m).copied())
            .or_else(|| self.instances.last().copied())
            .expect("instance not found");
        let func = inst.get_func(&self.store, name).expect("func not found");
        let result = func.call(&mut self.store, &val_args);
        if result.is_ok() {
            if self.dump_ir {
                self.dump_ir_for(module);
            }
            panic!("expected trap but function returned {:?}", result);
        }
    }
}

fn wast_arg_to_val(arg: &WastArg) -> Val {
    match arg {
        WastArg::Core(wast::core::WastArgCore::I32(val)) => Val::I32(*val),
        WastArg::Core(wast::core::WastArgCore::I64(val)) => Val::I64(*val),
        WastArg::Core(wast::core::WastArgCore::F32(val)) => Val::F32(f32::from_bits(val.bits)),
        WastArg::Core(wast::core::WastArgCore::F64(val)) => Val::F64(f64::from_bits(val.bits)),
        WastArg::Core(wast::core::WastArgCore::RefNull(_)) => Val::I64(0),
        WastArg::Core(wast::core::WastArgCore::RefExtern(val)) => Val::I64(*val as i64),
        _ => panic!("Unsupported argument type {:?}", arg),
    }
}

pub fn run_wast_file(path: &Path, strategy: Strategy, dump_ir: bool, verbose: bool) -> Result<()> {
    let mode_name = match strategy {
        Strategy::Interpreter => "interp",
        Strategy::Jit => "jit",
        Strategy::Auto => "auto",
    };
    println!(
        ">>>> TESTING SPEC TEST: {:?} ({}) <<<<",
        path.file_name().unwrap(),
        mode_name
    );
    let contents = fs::read_to_string(path)?;
    let buf = ParseBuffer::new(&contents)?;
    let wast = parser::parse::<Wast>(&buf)
        .map_err(|e| anyhow::anyhow!("failed to parse {:?}: {}", path, e))?;
    let mut runner = SpecRunner::new(strategy, mode_name, dump_ir, verbose);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        for directive in wast.directives {
            match directive {
                WastDirective::Module(mut module) => {
                    let id = match &module {
                        QuoteWat::Wat(Wat::Module(m)) => m.id.map(|i| i.name()),
                        _ => None,
                    };
                    runner.instantiate(&module.encode().unwrap(), id).unwrap();
                }
                WastDirective::Register { name, module, .. } => {
                    runner.register(name, module.map(|m| m.name()));
                }
                WastDirective::Invoke(invoke) => {
                    runner
                        .call(invoke.module.map(|m| m.name()), &invoke.name, &invoke.args)
                        .unwrap();
                }
                WastDirective::AssertReturn {
                    exec,
                    results,
                    span,
                    ..
                } => {
                    if let WastExecute::Invoke(invoke) = exec {
                        let actuals = runner
                            .call(invoke.module.map(|m| m.name()), &invoke.name, &invoke.args)
                            .unwrap();
                        for (j, expected) in results.iter().enumerate() {
                            let actual = actuals[j];
                            match expected {
                                WastRet::Core(wast::core::WastRetCore::I32(val)) => {
                                    if actual as i32 != *val {
                                        if verbose {
                                            let (line, col) = span.linecol_in(&contents);
                                            println!("Assertion failed at {}:{}:{}: expected i32 {}, got {}", path.display(), line + 1, col + 1, val, actual as i32);
                                        }
                                        panic!("i32 mismatch");
                                    }
                                }
                                WastRet::Core(wast::core::WastRetCore::I64(val)) => {
                                    if actual != *val {
                                        if verbose {
                                            let (line, col) = span.linecol_in(&contents);
                                            println!("Assertion failed at {}:{}:{}: expected i64 {}, got {}", path.display(), line + 1, col + 1, val, actual);
                                        }
                                        panic!("i64 mismatch");
                                    }
                                }
                                WastRet::Core(wast::core::WastRetCore::F32(val)) => {
                                    if let wast::core::NanPattern::Value(v) = val {
                                        if actual as u32 != v.bits {
                                            if verbose {
                                                let (line, col) = span.linecol_in(&contents);
                                                println!("Assertion failed at {}:{}:{}: expected f32 bits {:x}, got {:x}", path.display(), line + 1, col + 1, v.bits, actual as u32);
                                            }
                                            panic!("f32 mismatch");
                                        }
                                    }
                                }
                                WastRet::Core(wast::core::WastRetCore::F64(val)) => {
                                    if let wast::core::NanPattern::Value(v) = val {
                                        if actual as u64 != v.bits {
                                            if verbose {
                                                let (line, col) = span.linecol_in(&contents);
                                                println!("Assertion failed at {}:{}:{}: expected f64 bits {:x}, got {:x}", path.display(), line + 1, col + 1, v.bits, actual as u64);
                                            }
                                            panic!("f64 mismatch");
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                WastDirective::AssertTrap { exec, .. } => {
                    if let WastExecute::Invoke(invoke) = exec {
                        runner.assert_trap(
                            invoke.module.map(|m| m.name()),
                            &invoke.name,
                            &invoke.args,
                        );
                    }
                }
                _ => {}
            }
        }
    }));

    match result {
        Ok(_) => Ok(()),
        Err(e) => {
            if dump_ir {
                runner.dump_all_ir();
            }
            std::panic::resume_unwind(e);
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let strategy = match args.strategy.as_str() {
        "interp" => Strategy::Interpreter,
        "jit" => Strategy::Jit,
        _ => anyhow::bail!("Invalid strategy. Use 'interp' or 'jit'."),
    };

    // Determine the root directory of the workspace to find testsuite
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    // If we're in crates/veloc_wasm/tests/spec_test, we need to go up 3 levels to reach workspace root,
    // or just assume we're running from workspace root.
    // Let's use relative path from the crate root.
    let default_tests_dir = manifest_dir.join("../testsuite");

    let path = args.path.unwrap_or(default_tests_dir);

    if path.is_file() {
        run_wast_file(&path, strategy, args.dump_ir, args.verbose)?;
    } else if path.is_dir() {
        let mut paths: Vec<_> = fs::read_dir(&path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                p.extension().map_or(false, |ext| ext == "wast") 
                    && !name.starts_with("simd_")
                    && !name.contains("relaxed")
                    && !name.contains("i16x8")
                    && !name.contains("i32x4")
                    && !name.contains("i64x2")
                    && !name.contains("i8x16")
                    && !name.contains("f32x4")
                    && !name.contains("f64x2")
                    && !name.contains("v128")
            })
            .collect();
        paths.sort();
        let mut passed = 0;
        let mut failed = 0;
        for p in &paths {
            if let Err(e) = run_wast_file(p, strategy, args.dump_ir, args.verbose) {
                eprintln!("FAIL: {:?}\nError: {}", p.file_name().unwrap(), e);
                failed += 1;
            } else {
                passed += 1;
            }
        }
        println!("\nTest Summary: {} passed, {} failed", passed, failed);
        if failed > 0 {
            anyhow::bail!("{} tests failed", failed);
        }
    } else {
        anyhow::bail!("Path does not exist: {:?}", path);
    }
    Ok(())
}
