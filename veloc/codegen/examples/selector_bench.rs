//! Reproducible table-driven selector and whole-pipeline benchmark.
//! Cloning and hashing are outside the selector timer. Output fingerprints
//! support comparisons with earlier implementations. This synthetic workload
//! does not replace application benchmarks.
use std::{
    hint::black_box,
    time::{Duration, Instant},
};
use veloc_codegen::isel::InstructionSelector;
use veloc_codegen::{CodegenOptions, CodegenPipeline, TargetConfig, create_target_machine};
use veloc_lir::{InstBuild, MachineFunction, MachineOpcode, Type};

fn workload(ty: Type, comparisons: bool) -> MachineFunction {
    let mut f = MachineFunction::new("matcher".into());
    let mut e = f.editor();
    let block = e.create_block();
    let a = e.alloc_vreg(ty);
    let b = e.alloc_vreg(ty);
    e.append_block_param(block, a);
    e.append_block_param(block, b);
    let mut value = a;
    for i in 0..96 {
        let next = e.alloc_vreg(ty);
        if comparisons {
            let flag = e.alloc_vreg(Type::BOOL);
            let cc = [
                veloc_mir::FloatCC::Eq,
                veloc_mir::FloatCC::Ne,
                veloc_mir::FloatCC::Lt,
                veloc_mir::FloatCC::Le,
                veloc_mir::FloatCC::Gt,
                veloc_mir::FloatCC::Ge,
            ][i % 6];
            let cmp = e.writer().fcmp(flag, value, b, cc);
            e.append_inst(block, cmp);
            let select = e.writer().select(next, flag, value, b);
            e.append_inst(block, select);
        } else {
            let constant = e.alloc_vreg(ty);
            let inst = e.writer().constant(constant, (i % 13 + 1) as i64);
            e.append_inst(block, inst);
            let inst = match i % 4 {
                0 => e.writer().add(next, value, constant),
                1 => e.writer().mul(next, value, b),
                2 => e.writer().xor(next, value, constant),
                _ => e.writer().sub(next, value, b),
            };
            e.append_inst(block, inst);
        }
        value = next;
    }
    // An already selected sink keeps the result alive without ABI lowering.
    // This selector-only fixture is never encoded as an executable function.
    let sink = e
        .writer()
        .write(MachineOpcode::Target(0), &[], &[value], []);
    e.append_inst(block, sink);
    f
}

fn hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    })
}

fn memory_workload(stack: bool) -> MachineFunction {
    let mut f = MachineFunction::new("addresses".into());
    let mut e = f.editor();
    let block = e.create_block();
    let base = e.alloc_vreg(Type::PTR);
    let index = e.alloc_vreg(Type::I64);
    e.append_block_param(block, base);
    e.append_block_param(block, index);
    let slot = e.alloc_stack_object(veloc_lir::StackObject::Local, 16, 8);
    let mut live = Vec::new();
    for _ in 0..96 {
        let address = e.alloc_vreg(Type::PTR);
        let addr = if stack {
            e.writer().stack_addr(address, slot)
        } else {
            e.writer().ptr_add(address, base, index)
        };
        e.append_inst(block, addr);
        let value = e.alloc_vreg(Type::I64);
        let load = e
            .writer()
            .with_memory(veloc_lir::MemoryAccess::new(veloc_lir::MemoryKind::Read, 8))
            .load(value, address, 0);
        e.append_inst(block, load);
        live.push(value);
    }
    let sink = e.writer().write(MachineOpcode::Target(0), &[], &live, []);
    e.append_inst(block, sink);
    f
}
fn median(mut values: Vec<Duration>) -> f64 {
    values.sort();
    values[values.len() / 2].as_secs_f64() * 1e6
}
fn main() {
    let target = create_target_machine(TargetConfig::default()).unwrap();
    let selector = InstructionSelector::new(target.selector());
    let mode = "table";
    for (name, source) in [
        ("integer32", workload(Type::I32, false)),
        ("integer64", workload(Type::I64, false)),
        ("float32", workload(Type::F32, true)),
        ("float64", workload(Type::F64, true)),
        ("stack_loads", memory_workload(true)),
        ("indexed_loads", memory_workload(false)),
    ] {
        let mut check = source.clone();
        selector.select(&mut check).unwrap();
        check.check_refs().unwrap();
        let fingerprint = hash(check.format_for_dump().as_bytes());
        let mut times = Vec::new();
        for round in 0..10 {
            let mut batch: Vec<_> = (0..64).map(|_| source.clone()).collect();
            let start = Instant::now();
            for f in &mut batch {
                selector.select(black_box(f)).unwrap();
            }
            let elapsed = start.elapsed();
            black_box(&batch);
            if round != 0 {
                times.push(elapsed / 64);
            }
        }
        println!(
            "{mode} selector {name} median_us={:.3} hash={fingerprint:016x}",
            median(times)
        );
    }
    let module = veloc_mir::ModuleParser::new()
        .parse(include_str!("sum.mir"))
        .unwrap();
    module.validate().unwrap();
    for optimize in [false, true] {
        let pipeline = CodegenPipeline::with_options(
            &*target,
            CodegenOptions {
                optimize,
                ..Default::default()
            },
        );
        let reference = pipeline.compile_functions(&module).unwrap();
        let fingerprint = reference.values().fold(0, |h, code| h ^ hash(code));
        let mut times = Vec::new();
        for round in 0..10 {
            let start = Instant::now();
            for _ in 0..128 {
                black_box(pipeline.compile_functions(black_box(&module)).unwrap());
            }
            if round != 0 {
                times.push(start.elapsed() / 128);
            }
        }
        println!(
            "{mode} pipeline sum optimize={optimize} median_us={:.3} hash={fingerprint:016x}",
            median(times)
        );
    }
}
