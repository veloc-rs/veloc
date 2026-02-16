use hashbrown::HashMap;
use veloc_ir::{Function, InstructionData, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveInterval {
    pub start: u32,
    pub end: u32,
}

/// Result of liveness analysis
pub struct Liveness {
    pub intervals: HashMap<Value, LiveInterval>,
}

pub fn analyze_liveness(func: &Function) -> Liveness {
    let mut intervals: HashMap<Value, LiveInterval> = HashMap::new();
    let mut block_starts = HashMap::new();
    let mut block_ends = HashMap::new();
    let mut inst_pc = 0u32;

    // 1. Initial pass: linearly assign PC and find def/use
    for &block in &func.layout.block_order {
        block_starts.insert(block, inst_pc);

        for &param in &func.layout.blocks[block].params {
            intervals.insert(
                param,
                LiveInterval {
                    start: inst_pc,
                    end: inst_pc,
                },
            );
        }

        for &inst in &func.layout.blocks[block].insts {
            if let Some(v) = func.dfg.inst_results(inst) {
                intervals.insert(
                    v,
                    LiveInterval {
                        start: inst_pc,
                        end: inst_pc,
                    },
                );
            }

            visit_operands(&func.dfg.instructions[inst], &func.dfg, |v| {
                if let Some(int) = intervals.get_mut(&v) {
                    int.end = inst_pc;
                }
            });
            inst_pc += 1;
        }
        block_ends.insert(block, inst_pc);
    }

    // 2. Multi-pass refinement for loops
    let mut changed = true;
    while changed {
        changed = false;
        for &block in &func.layout.block_order {
            let start_pc = block_starts[&block];
            let end_pc = block_ends[&block];

            let mut successors = Vec::new();
            if let Some(&last_inst) = func.layout.blocks[block].insts.last() {
                match &func.dfg.instructions[last_inst] {
                    InstructionData::Jump { dest } => {
                        successors.push(func.dfg.block_calls[*dest].block);
                    }
                    InstructionData::Br {
                        then_dest,
                        else_dest,
                        ..
                    } => {
                        successors.push(func.dfg.block_calls[*then_dest].block);
                        successors.push(func.dfg.block_calls[*else_dest].block);
                    }
                    InstructionData::BrTable { table, .. } => {
                        for &dest in &func.dfg.jump_tables[*table].targets {
                            successors.push(func.dfg.block_calls[dest].block);
                        }
                    }
                    _ => {}
                }
            }

            for succ in successors {
                let succ_start = block_starts[&succ];
                if succ_start <= start_pc {
                    // Back-edge detected.
                    for (_, int) in intervals.iter_mut() {
                        if int.start <= succ_start && int.end >= succ_start {
                            if int.end < end_pc {
                                int.end = end_pc;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
    }

    Liveness { intervals }
}

pub fn visit_operands<F: FnMut(Value)>(
    idata: &InstructionData,
    dfg: &veloc_ir::DataFlowGraph,
    mut f: F,
) {
    use InstructionData::*;
    match idata {
        Unary { arg, .. }
        | IntToPtr { arg }
        | PtrToInt { arg, .. }
        | PtrOffset { ptr: arg, .. }
        | Load { ptr: arg, .. } => f(*arg),
        Binary { args, .. } | IntCompare { args, .. } | FloatCompare { args, .. } => {
            f(args[0]);
            f(args[1]);
        }
        Store { ptr, value, .. } => {
            f(*ptr);
            f(*value);
        }
        StackStore { value, .. } => f(*value),
        Select {
            condition,
            then_val,
            else_val,
            ..
        } => {
            f(*condition);
            f(*then_val);
            f(*else_val);
        }
        Jump { dest } => {
            for &v in dfg.get_value_list(dfg.block_calls[*dest].args) {
                f(v);
            }
        }
        Br {
            condition,
            then_dest,
            else_dest,
        } => {
            f(*condition);
            for &v in dfg.get_value_list(dfg.block_calls[*then_dest].args) {
                f(v);
            }
            for &v in dfg.get_value_list(dfg.block_calls[*else_dest].args) {
                f(v);
            }
        }
        BrTable { index, table } => {
            f(*index);
            for &dest in &dfg.jump_tables[*table].targets {
                for &v in dfg.get_value_list(dfg.block_calls[dest].args) {
                    f(v);
                }
            }
        }
        Return { value } => {
            if let Some(v) = value {
                f(*v);
            }
        }
        Call { args, .. } => {
            for &v in dfg.get_value_list(*args) {
                f(v);
            }
        }
        CallIndirect { ptr, args, .. } => {
            f(*ptr);
            for &v in dfg.get_value_list(*args) {
                f(v);
            }
        }
        PtrIndex { ptr, index, .. } => {
            f(*ptr);
            f(*index);
        }
        _ => {}
    }
}
