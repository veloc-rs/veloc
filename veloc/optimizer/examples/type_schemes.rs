//! Synthetic type-scheme and compiler-phase benchmark. Run in release mode.
//! Optional arguments: phase (all/types/build/validate/fold), iteration multiplier.
use std::hint::black_box;
use std::time::Instant;
use veloc_analyzer::AnalysisManager;
use veloc_mir::{CallConv, Linkage, ModuleBuilder, ModuleData, Opcode, Type};
use veloc_optimizer::Metrics;
use veloc_optimizer::passes::function::simplify::run_simplify;

fn module() -> ModuleData {
    let mut module = ModuleBuilder::new();
    let sig = module.make_signature(vec![], vec![Type::I32], CallConv::SystemV);
    let id = module.declare_function("arithmetic".into(), sig, Linkage::Local);
    {
        let mut builder = module.builder(id);
        builder.init_entry_block();
        let mut ins = builder.ins();
        let one = ins.i32const(1);
        let three = ins.i32const(3);
        let mut value = ins.i32const(17);
        for _ in 0..200 {
            value = ins.iadd(value, one);
            value = ins.isub(value, one);
            value = ins.imul(value, three);
            let wide = ins.extends(value, Type::I64);
            value = ins.wrap(wide, Type::I32);
        }
        ins.ret(&[value]);
    }
    module.build_data()
}

fn measure(name: &str, iterations: usize, mut run: impl FnMut()) {
    for _ in 0..3 {
        run();
    }
    let mut samples = Vec::new();
    for _ in 0..9 {
        let start = Instant::now();
        for _ in 0..iterations {
            run();
        }
        samples.push(start.elapsed().as_secs_f64() * 1e9 / iterations as f64);
    }
    samples.sort_by(f64::total_cmp);
    println!(
        "{name}: median {:.1} ns/iteration (min {:.1}, max {:.1})",
        samples[4], samples[0], samples[8]
    );
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let phase = args.first().map_or("all", String::as_str);
    assert!(matches!(
        phase,
        "all" | "types" | "build" | "validate" | "fold"
    ));
    let scale: usize = args.get(1).map_or(1, |s| s.parse().unwrap());
    assert!(scale > 0);
    let selected = |name| phase == "all" || phase == name;
    if selected("types") {
        measure("validate_types (5 calls)", 100_000 * scale, || {
            for op in [Opcode::IAdd, Opcode::ISub, Opcode::IMul] {
                black_box(op)
                    .validate_types(black_box(&[Type::I32, Type::I32]), black_box(&[Type::I32]))
                    .unwrap();
            }
            black_box(Opcode::ExtendS)
                .validate_types(black_box(&[Type::I32]), black_box(&[Type::I64]))
                .unwrap();
            black_box(Opcode::Wrap)
                .validate_types(black_box(&[Type::I64]), black_box(&[Type::I32]))
                .unwrap();
        });
        measure("infer_results (3 calls)", 100_000 * scale, || {
            for op in [Opcode::IAdd, Opcode::ISub, Opcode::IMul] {
                black_box(
                    black_box(op)
                        .infer_result_types(black_box(&[Type::I32, Type::I32]))
                        .unwrap(),
                );
            }
        });
    }
    if selected("build") {
        measure("build + drop (1004 instructions)", 100 * scale, || {
            black_box(module());
        });
    }
    let source = module();
    source.validate().unwrap();
    if selected("validate") {
        measure("validate (1004 instructions)", 300 * scale, || {
            black_box(&source).validate().unwrap();
        });
    }
    if selected("fold") {
        // Clone outside the timed region. Folding changes the IR in place.
        let mut samples = Vec::new();
        for sample in 0..12 {
            let mut elapsed = 0.0;
            for _ in 0..10 * scale {
                let mut data = source.clone();
                let mut analyses = AnalysisManager::new();
                let mut metrics = Metrics::default();
                let start = Instant::now();
                for (_, function) in data.functions.iter_mut() {
                    assert!(run_simplify(function, &mut analyses, false, &mut metrics));
                }
                elapsed += start.elapsed().as_secs_f64();
                black_box(&data);
                data.validate().unwrap();
            }
            if sample >= 3 {
                samples.push(elapsed * 1e9 / (10 * scale) as f64);
            }
        }
        samples.sort_by(f64::total_cmp);
        println!(
            "fold (1000 instructions): median {:.1} ns/iteration (min {:.1}, max {:.1})",
            samples[4], samples[0], samples[8]
        );
    }
}
