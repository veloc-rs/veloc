//! Make inter-block data dependencies explicit before duplicating blocks.
//! Function inputs and literals remain globally available. Other values travel
//! through block parameters, so each cloned block can be remapped independently.
use hashbrown::HashMap;
use veloc_mir::{Block, FuncBody, Value, ValueDef, function::EdgeRef};

pub(super) fn localize(f: &mut FuncBody) -> Vec<(Value, Value)> {
    let definitions: Vec<_> = f
        .layout()
        .block_order()
        .flat_map(|block| {
            f.dfg()
                .block_params(block)
                .iter()
                .copied()
                .chain(
                    f.layout()
                        .block_insts(block)
                        .flat_map(|i| f.dfg().inst_results(i).iter().copied()),
                )
                .map(move |value| (value, block))
        })
        .collect();
    let mut transported = Vec::new();
    for (value, definition) in definitions {
        let uses: Vec<_> = f
            .dfg()
            .uses(value)
            .filter_map(|site| {
                let block = f.layout().inst_block(site.inst())?;
                (block != definition).then_some((site.inst(), site.index(), block))
            })
            .collect();
        if uses.is_empty() {
            continue;
        }
        let mut available = HashMap::from([(definition, value)]);
        let mut pending = Vec::new();
        // Rewrite existing operand positions before extending edge arguments:
        // adding an argument to one successor shifts later operand positions.
        for (inst, index, block) in uses {
            let replacement = read(
                f,
                value,
                block,
                &mut available,
                &mut pending,
                &mut transported,
            );
            f.edit().set_operand(inst, index, replacement);
        }
        while let Some(block) = pending.pop() {
            let predecessors = f.cfg().preds(block).to_vec();
            for pred in predecessors {
                let argument = read(
                    f,
                    value,
                    pred,
                    &mut available,
                    &mut pending,
                    &mut transported,
                );
                let inst = f.layout().last_inst(pred).expect("terminated predecessor");
                let mut edges = Vec::new();
                let mut index = 0;
                f.dfg().inst(inst).visit_successors(|edge| {
                    if edge.block == block {
                        let mut args = edge.args.to_vec();
                        args.push(argument);
                        edges.push((EdgeRef { inst, index }, args));
                    }
                    index += 1;
                });
                for (edge, args) in edges {
                    f.edit().redirect_edge(edge, block, &args);
                }
            }
        }
    }
    transported
}

fn read(
    f: &mut FuncBody,
    original: Value,
    block: Block,
    available: &mut HashMap<Block, Value>,
    pending: &mut Vec<Block>,
    transported: &mut Vec<(Value, Value)>,
) -> Value {
    if let Some(&value) = available.get(&block) {
        return value;
    }
    // A valid SSA definition dominates its uses. Backward propagation must
    // reach that definition before reaching function entry or a dead root.
    assert!(block != f.entry_block() && !f.cfg().preds(block).is_empty());
    debug_assert!(!matches!(
        f.dfg().value_def(original),
        ValueDef::Const(_) | ValueDef::FunctionParam(_)
    ));
    let ty = f.dfg().value_type(original);
    let param = f.edit().append_block_param(block, ty);
    // Publish before following predecessors, breaking cycles without recursion.
    available.insert(block, param);
    pending.push(block);
    transported.push((original, param));
    param
}
