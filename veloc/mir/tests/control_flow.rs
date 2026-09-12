//! Public editing and validation of the control-flow substrate. Text-only
//! structural and dominance diagnostics live in the file-test corpus.
use veloc_mir::function::Dominators;
use veloc_mir::{Block, EdgeRef, ModuleParser, Value};

#[test]
fn generated_callable_builders_share_ssa_storage_and_explicit_validation() {
    use veloc_mir::{CallConv, CallableKind, Linkage, ModuleBuilder, Type};
    let mut module = ModuleBuilder::new();
    let sig = module.make_signature(vec![Type::I32], vec![Type::I32], CallConv::SystemV);
    let ty = Type::callable(sig, CallableKind::Owned);
    assert!(ty.as_scalar().is_none());
    assert!(ty.as_vector().is_none());
    assert!(!ty.is_integer());
    assert_eq!(ty.bit_size(), None);
    let body = module.declare_function("body".into(), sig, Linkage::Local);
    let empty = module.make_signature(vec![], vec![], CallConv::SystemV);
    let cleanup = module.declare_function("cleanup".into(), empty, Linkage::Local);
    let entry = module.declare_function("entry".into(), sig, Linkage::Local);
    {
        let mut f = module.builder(body);
        f.init_entry_block();
        let arg = f.ins().param(0);
        f.ins().ret(&[arg]);
    }
    {
        let mut f = module.builder(cleanup);
        f.init_entry_block();
        f.ins().ret(&[]);
    }
    {
        let mut f = module.builder(entry);
        f.init_entry_block();
        let arg = f.ins().param(0);
        let k = f.ins().closure_new(body, &[], cleanup, ty);
        f.ins().tail_call_value(k, &[arg]);
    }
    let mut module = module.build_data();
    module.validate().unwrap();
    let function = &mut module.functions[entry];
    function.dfg().check_uses().unwrap();
    let create = function.layout().blocks()[Block(0)].insts[0];
    function
        .edit()
        .replace_inst(create, |writer: veloc_mir::InstWriter<'_>| {
            writer.closure(veloc_mir::Opcode::ClosureShared, body, &[])
        });
    assert!(
        module
            .validate()
            .unwrap_err()
            .to_string()
            .contains("ownership kind mismatch")
    );
}

#[test]
fn deeply_nested_callable_signatures_validate_without_recursive_stack_growth() {
    use veloc_mir::{CallConv, CallableKind, ModuleData, SigId, Signature, Type};
    let mut module = ModuleData::default();
    // Forward references force the validator to visit the entire chain before
    // any suffix has been marked done. Text cannot directly encode these IDs.
    for index in 0..20_000 {
        let params = vec![Type::callable(SigId(index + 1), CallableKind::Shared)];
        module
            .signatures
            .insert(Signature::new(params, vec![], CallConv::SystemV));
    }
    module
        .signatures
        .insert(Signature::new(vec![], vec![], CallConv::SystemV));
    module.validate().unwrap();
}

#[test]
fn callable_signature_diagnostics_identify_cycles_and_unknown_references() {
    use veloc_mir::{CallConv, CallableKind, ModuleData, SigId, Signature, Type};
    let mut module = ModuleData::default();
    let sig = module.signatures.insert(Signature::new(
        vec![Type::callable(SigId(1), CallableKind::Shared)],
        vec![],
        CallConv::SystemV,
    ));
    let error = module.validate().unwrap_err().to_string();
    assert!(error.contains("signature sig0, parameter 0"), "{error}");
    assert!(error.contains("unknown callable signature sig1"), "{error}");

    module.signatures.insert(Signature::new(
        vec![],
        vec![Type::callable(sig, CallableKind::Owned)],
        CallConv::SystemV,
    ));
    let error = module.validate().unwrap_err().to_string();
    assert!(error.contains("recursive callable signature"), "{error}");
    assert!(error.contains("signature sig1"), "{error}");
    assert!(error.contains("active signature sig0"), "{error}");
}

fn example() -> veloc_mir::ModuleData {
    let module = ModuleParser::new()
        .parse(
            "local function choose(bool, i32, i32) -> i32\n\
         block0(v0: bool, v1: i32, v2: i32):\n\
           br v0, block1(v1), block1(v2)\n\
         block1(v3: i32):\n\
           return v3\n\
         block2(v4: i32):\n\
           return v4\n",
        )
        .unwrap();
    (*module).clone()
}

#[test]
fn editing_one_edge_preserves_other_occurrences_and_use_chains() {
    let mut module = example();
    module.validate().unwrap();
    let func = &mut module.functions[veloc_mir::FuncId(0)];
    let inst = func.layout().blocks()[Block(0)].insts[0];
    func.edit().edit_edge(EdgeRef { inst, index: 1 }, |edge| {
        assert_eq!(edge.args(), [Value(2)]);
        edge.set_block(Block(2));
        edge.set_args(&[Value(1)]);
    });
    let mut edges = Vec::new();
    func.dfg()
        .inst(inst)
        .visit_successors(|edge| edges.push((edge.block, edge.args.to_vec())));
    assert_eq!(
        edges,
        [(Block(1), vec![Value(1)]), (Block(2), vec![Value(1)])]
    );
    assert_eq!(func.layout().blocks()[Block(0)].succs, [Block(1), Block(2)]);
    assert_eq!(func.layout().blocks()[Block(1)].preds, [Block(0)]);
    assert_eq!(func.layout().blocks()[Block(2)].preds, [Block(0)]);
    func.dfg().check_uses().unwrap();
    module.validate().unwrap();

    let func = &mut module.functions[veloc_mir::FuncId(0)];
    func.edit()
        .edit_edge(EdgeRef { inst, index: 0 }, |edge| edge.set_block(Block(2)));
    assert!(func.layout().blocks()[Block(1)].preds.is_empty());
    assert_eq!(func.layout().blocks()[Block(2)].preds, [Block(0)]);
    let dom = Dominators::compute(func.layout(), Block(0));
    assert!(dom.dominates(Block(0), Block(2)));
    assert_eq!(dom.immediate_dominator(Block(2)), Some(Block(0)));
    assert!(!dom.is_reachable(Block(1)));
    module.validate().unwrap();
}

#[test]
fn validator_rejects_detached_targets_and_unknown_values_without_panicking() {
    let mut module = example();
    let func = &mut module.functions[veloc_mir::FuncId(0)];
    let inst = func.layout().blocks()[Block(1)].insts[0];
    func.edit().set_operand(inst, 0, Value(1000));
    assert!(
        module
            .validate()
            .unwrap_err()
            .to_string()
            .contains("no attached definition")
    );

    // A detached but allocated block is not a valid destination. Construction
    // accepts handles and leaves this structural contract to validation.
    let mut builder = veloc_mir::ModuleBuilder::new();
    let sig = builder.make_signature(vec![], vec![], veloc_mir::CallConv::SystemV);
    let id = builder.declare_function("bad".into(), sig, veloc_mir::Linkage::Local);
    let mut f = builder.builder(id);
    f.init_entry_block();
    let detached = f.create_block();
    f.ins().jump(detached, &[]);
    drop(f);
    assert!(
        builder
            .validate()
            .unwrap_err()
            .to_string()
            .contains("not in function layout")
    );
}

#[test]
fn edge_argument_replacement_preserves_other_edges() {
    let mut dfg = veloc_mir::dfg::DataFlowGraph::new();
    let inst = dfg.create_inst(|writer| {
        writer.br(
            Value(0),
            veloc_mir::Successor {
                block: Block(1),
                args: &[Value(1), Value(2)],
            },
            veloc_mir::Successor {
                block: Block(1),
                args: &[Value(3)],
            },
        )
    });
    let mut index = 0;
    dfg.edit_successors(inst, |edge| {
        if index == 0 {
            edge.set_args(&[]);
        } else {
            edge.set_args(&[Value(4), Value(5), Value(6)]);
        }
        index += 1;
    });
    let mut operands = Vec::new();
    dfg.inst(inst).visit_operands(|v| operands.push(v));
    assert_eq!(operands, [Value(0), Value(4), Value(5), Value(6)]);
    let mut lengths = Vec::new();
    dfg.inst(inst)
        .visit_successors(|edge| lengths.push(edge.args.len()));
    assert_eq!(lengths, [0, 3]);
}

#[test]
fn fallible_visitors_preserve_order_and_stop_at_the_first_error() {
    use veloc_mir::Successor;
    // Parallel edges must remain separate, in both fixed and variadic storage.
    let edges = [
        Successor {
            block: Block(1),
            args: &[Value(1)],
        },
        Successor {
            block: Block(1),
            args: &[Value(2)],
        },
    ];
    let mut dfg = veloc_mir::dfg::DataFlowGraph::new();
    for inst in [
        dfg.writer().br(Value(0), edges[0], edges[1]),
        dfg.writer().br_table(Value(0), edges),
    ] {
        let view = dfg.inst(inst);
        let mut visited = Vec::new();
        let result = view.try_visit_successors(|edge| {
            visited.push(edge.args[0]);
            Err("stop")
        });
        assert_eq!(result, Err("stop"));
        assert_eq!(visited, [Value(1)]);

        visited.clear();
        view.try_visit_successors::<()>(|edge| {
            visited.push(edge.args[0]);
            Ok(())
        })
        .unwrap();
        assert_eq!(visited, [Value(1), Value(2)]);

        visited.clear();
        let result = view.try_visit_operands(|value| {
            visited.push(value);
            if value == Value(1) {
                Err("stop")
            } else {
                Ok(())
            }
        });
        assert_eq!(result, Err("stop"));
        assert_eq!(visited, [Value(0), Value(1)]);
    }
}
