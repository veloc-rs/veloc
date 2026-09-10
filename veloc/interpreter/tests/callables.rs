//! File-driven execution plus host ownership boundaries that a single main()
//! file test cannot express (escape, reuse, explicit drop and collection).
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use veloc_interpreter::{HostFunction, Interpreter, InterpreterValue as V, Program, VirtualMemory};
use veloc_mir::{CallConv, FuncId, ModuleId, ModuleParser, Signature, Type};

struct Memory;
impl VirtualMemory for Memory {
    fn translate_addr(&self, address: usize, _: usize) -> Option<*mut u8> {
        Some(address as *mut u8)
    }
}

struct Fixture {
    program: Program,
    module: ModuleId,
    functions: Vec<(String, FuncId)>,
    cleaned: Arc<AtomicUsize>,
}
impl Fixture {
    fn new() -> Self {
        let module = ModuleParser::new()
            .parse(include_str!("callables.mir"))
            .unwrap();
        module.validate().unwrap();
        let functions = module
            .functions
            .iter()
            .map(|(id, f)| (f.name.clone(), id))
            .collect();
        let cleanup = module.find_function_by_name("cleanup").unwrap();
        let cleanup_empty = module.find_function_by_name("cleanup_empty").unwrap();
        let pair = module.find_function_by_name("host_pair").unwrap();
        let cleaned = Arc::new(AtomicUsize::new(0));
        let count = cleaned.clone();
        let mut program = Program::new();
        let host = program.register_host(
            "cleanup".into(),
            HostFunction::new(
                Signature::new(vec![Type::I32], vec![], CallConv::SystemV),
                move |values| {
                    assert_eq!(values[0].unwrap_i32(), 40);
                    count.fetch_add(1, Ordering::Relaxed);
                },
            ),
        );
        let count = cleaned.clone();
        let host_empty = program.register_host(
            "cleanup_empty".into(),
            HostFunction::new(
                Signature::new(vec![], vec![], CallConv::SystemV),
                move |_| {
                    count.fetch_add(1, Ordering::Relaxed);
                },
            ),
        );
        let host_pair = program.register_host(
            "host_pair".into(),
            HostFunction::new(
                Signature::new(
                    vec![Type::I32, Type::I32],
                    vec![Type::I32, Type::I64],
                    CallConv::SystemV,
                ),
                |values| {
                    let captured = values[0].unwrap_i32();
                    let argument = values[1].unwrap_i32();
                    values[0] = V::i32(captured + argument);
                    values[1] = V::i64(i64::from(captured));
                },
            ),
        );
        let mut builder = program.builder(module);
        builder.link_host(cleanup, host).unwrap();
        builder.link_host(cleanup_empty, host_empty).unwrap();
        builder.link_host(pair, host_pair).unwrap();
        let module = builder.finish().unwrap();
        Self {
            program,
            module,
            functions,
            cleaned,
        }
    }

    fn run(
        &self,
        vm: &mut Interpreter,
        name: &str,
        args: &[V],
    ) -> veloc_interpreter::Result<Vec<V>> {
        let id = self.functions.iter().find(|(n, _)| n == name).unwrap().1;
        vm.run_function(&self.program, &Memory, self.module, id, args)
            .map(|values| values.to_vec())
    }
}

#[test]
fn escaped_owned_callable_runs_once_and_drop_is_explicit() {
    let f = Fixture::new();
    let mut vm = Interpreter::new();
    let k = f.run(&mut vm, "make", &[]).unwrap()[0];
    assert_eq!(
        f.run(&mut vm, "tail_call_owned", &[k, V::i32(2)]).unwrap(),
        vec![V::i32(42)]
    );
    assert!(f.run(&mut vm, "tail_call_owned", &[k, V::i32(2)]).is_err());
    assert_eq!(f.cleaned.load(Ordering::Relaxed), 0);
    let k = f.run(&mut vm, "make", &[]).unwrap()[0];
    assert_eq!(
        f.run(&mut vm, "call_owned", &[k, V::i32(2)]).unwrap(),
        vec![V::i32(43)]
    );
    assert!(f.run(&mut vm, "call_owned", &[k, V::i32(2)]).is_err());
    assert_eq!(f.cleaned.load(Ordering::Relaxed), 0);
    let k = f.run(&mut vm, "make", &[]).unwrap()[0];
    let outer = f.run(&mut vm, "wrap", &[k]).unwrap()[0];
    assert!(f.run(&mut vm, "drop_owned", &[k]).is_err());
    f.run(&mut vm, "drop_owned", &[outer]).unwrap();
    assert_eq!(f.cleaned.load(Ordering::Relaxed), 1);
    assert!(f.run(&mut vm, "drop_owned", &[outer]).is_err());
    assert_eq!(vm.live_callables(), 0);
}

#[test]
fn host_cannot_duplicate_or_forge_owned_arguments() {
    let f = Fixture::new();
    let mut vm = Interpreter::new();
    let k = f.run(&mut vm, "make", &[]).unwrap()[0];
    assert!(f.run(&mut vm, "drop_two", &[k, k]).is_err());
    assert!(f.run(&mut Interpreter::new(), "drop_owned", &[k]).is_err());
    assert!(Fixture::new().run(&mut vm, "drop_owned", &[k]).is_err());
    let target = f
        .functions
        .iter()
        .find(|(name, _)| name == "tail_call_owned")
        .unwrap()
        .1;
    let ptr = f.program.func_ref(f.module, target).unwrap().address();
    let error = f
        .run(&mut vm, "raw_call", &[V(ptr as u64), k, V::i32(2)])
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("indirect call signature mismatch")
    );
    f.run(&mut vm, "drop_owned", &[k]).unwrap();
    assert_eq!(f.cleaned.load(Ordering::Relaxed), 1);
}

#[test]
fn trap_aborts_moved_ownership_without_implicit_guest_cleanup() {
    let f = Fixture::new();
    let mut vm = Interpreter::new();
    let k = f.run(&mut vm, "make", &[]).unwrap()[0];
    assert!(f.run(&mut vm, "abort", &[k]).is_err());
    assert!(f.run(&mut vm, "drop_owned", &[k]).is_err());
    assert_eq!(f.cleaned.load(Ordering::Relaxed), 0);
    assert_eq!(vm.live_callables(), 0);
    assert_eq!(
        f.run(&mut vm, "keep_across_gc", &[]).unwrap(),
        vec![V::i32(42)]
    );
}

#[test]
fn tail_host_answers_and_zero_capture_cleanup_use_the_foreign_abi() {
    let f = Fixture::new();
    let mut vm = Interpreter::new();
    let k = f.run(&mut vm, "make_empty", &[]).unwrap()[0];
    f.run(&mut vm, "drop_empty", &[k]).unwrap();
    f.run(&mut vm, "tail_host", &[V::i32(40)]).unwrap();
    assert_eq!(f.cleaned.load(Ordering::Relaxed), 2);
    assert_eq!(vm.live_callables(), 0);
}

#[test]
fn guest_links_compare_structural_signatures_and_preserve_environment_origin() {
    let mut f = Fixture::new();
    let source = ModuleParser::new()
        .parse(include_str!("callables_remote.mir"))
        .unwrap();
    let import = source.find_function_by_name("make").unwrap();
    let run = source.find_function_by_name("run").unwrap();
    let run_tail = source.find_function_by_name("run_tail").unwrap();
    let drop_remote = source.find_function_by_name("drop_remote").unwrap();
    let target = f
        .functions
        .iter()
        .find(|(name, _)| name == "make")
        .unwrap()
        .1;
    let mut builder = f.program.builder(source);
    builder.link_import(import, f.module, target).unwrap();
    let module = builder.finish().unwrap();
    let mut vm = Interpreter::new();
    assert_eq!(
        vm.run_function(&f.program, &Memory, module, run, &[])
            .unwrap(),
        [V::i32(44)]
    );
    assert_eq!(
        vm.run_function(&f.program, &Memory, module, run_tail, &[])
            .unwrap(),
        [V::i32(42)]
    );
    vm.run_function(&f.program, &Memory, module, drop_remote, &[])
        .unwrap();
    assert_eq!(f.cleaned.load(Ordering::Relaxed), 1);
    assert_eq!(vm.live_callables(), 0);
}

#[test]
fn precise_roots_keep_waiting_callers_and_external_shared_values_alive() {
    let f = Fixture::new();
    let mut vm = Interpreter::new();
    let k = f.run(&mut vm, "make_shared", &[]).unwrap()[0];
    assert_eq!(
        f.run(&mut vm, "keep_across_gc", &[]).unwrap(),
        vec![V::i32(42)]
    );
    for _ in 0..2 {
        assert_eq!(
            f.run(&mut vm, "tail_call_shared", &[k, V::i32(2)]).unwrap(),
            vec![V::i32(42)]
        );
        assert_eq!(
            f.run(&mut vm, "call_shared", &[k, V::i32(2)]).unwrap(),
            vec![V::i32(84)]
        );
    }
    assert_eq!(vm.live_callables(), 1);
    vm.release_shared(k).unwrap();
    assert_eq!(vm.live_callables(), 0);
}

#[test]
fn tail_transfers_preserve_borrowed_stack_storage_and_do_not_grow_call_stack() {
    let f = Fixture::new();
    let mut vm = Interpreter::with_stack_limit(64);
    assert_eq!(
        f.run(&mut vm, "local_borrow", &[]).unwrap(),
        vec![V::i32(42)]
    );
    assert_eq!(
        f.run(&mut vm, "tail_loop", &[V::i32(100_000)]).unwrap(),
        vec![V::i32(0)]
    );
    assert_eq!(vm.live_callables(), 0);
}

#[test]
fn ordinary_local_calls_observe_mutated_borrowed_storage() {
    let f = Fixture::new();
    let mut vm = Interpreter::with_stack_limit(64);
    assert_eq!(
        f.run(&mut vm, "local_borrow_across_calls", &[]).unwrap(),
        vec![V::i32(20), V::i32(22)]
    );
    assert_eq!(vm.live_callables(), 0);
}

#[test]
fn host_backed_callables_return_multiple_or_no_values_to_the_caller() {
    let f = Fixture::new();
    let mut vm = Interpreter::new();
    assert_eq!(
        f.run(&mut vm, "call_host_values", &[]).unwrap(),
        vec![V::i32(44), V::i64(40)]
    );
    assert_eq!(
        f.run(&mut vm, "call_host_ignored_result", &[]).unwrap(),
        vec![V::i64(40)]
    );
    assert_eq!(f.cleaned.load(Ordering::Relaxed), 0);
    assert_eq!(
        f.run(&mut vm, "call_host_empty", &[]).unwrap(),
        vec![V::i32(7)]
    );
    assert_eq!(f.cleaned.load(Ordering::Relaxed), 2);
    assert_eq!(vm.live_callables(), 0);
}

#[test]
fn raw_host_imports_cannot_accept_or_produce_callable_handles() {
    for source in [
        "import function host(owned<() -> void>) -> void",
        "import function host() -> shared<() -> void>",
    ] {
        let module = ModuleParser::new().parse(source).unwrap();
        let import = module.find_function_by_name("host").unwrap();
        let signature = module
            .get_signature(module.functions[import].signature)
            .clone();
        let mut program = Program::new();
        let host = program.register_host(
            "host".into(),
            HostFunction::new(signature, |_| panic!("raw callable callback must not run")),
        );
        let mut builder = program.builder(module);
        builder.link_host(import, host).unwrap();
        let error = builder.finish().unwrap_err();
        assert!(error.to_string().contains("host callable imports"));
    }
}
