use cranelift_entity::SecondaryMap;
use veloc_ir::{Block, Function, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LiveInterval {
    pub start: u32,
    pub end: u32,
}

/// Result of liveness analysis
pub struct Liveness {
    pub intervals: SecondaryMap<Value, LiveInterval>,
}

pub fn analyze_liveness(func: &Function) -> Liveness {
    let mut intervals: SecondaryMap<Value, LiveInterval> = SecondaryMap::new();
    let mut block_starts: SecondaryMap<Block, u32> = SecondaryMap::new();
    let mut block_ends: SecondaryMap<Block, u32> = SecondaryMap::new();
    let mut inst_pc = 0u32;

    // 1. Initial pass: linearly assign PC and find def/use
    for &block in &func.layout.block_order {
        block_starts[block] = inst_pc;

        for &param in &func.layout.blocks[block].params {
            intervals[param] = LiveInterval {
                start: inst_pc,
                end: inst_pc,
            };
        }

        for &inst in &func.layout.blocks[block].insts {
            // 处理多返回值：每个结果值设置生存期
            for &v in func.dfg.inst_results(inst) {
                intervals[v] = LiveInterval {
                    start: inst_pc,
                    end: inst_pc,
                };
            }

            inst.visit_operands(&func.dfg, |v| {
                if intervals[v].end != 0 || intervals[v].start != 0 {
                    intervals[v].end = inst_pc;
                }
            });
            inst_pc += 1;
        }
        block_ends[block] = inst_pc;
    }

    // 2. Multi-pass refinement for loops
    let mut changed = true;
    while changed {
        changed = false;
        for &block in &func.layout.block_order {
            let start_pc = block_starts[block];
            let end_pc = block_ends[block];

            let mut successors = Vec::new();
            if let Some(&last_inst) = func.layout.blocks[block].insts.last() {
                successors = func.dfg.analyze_successors(last_inst);
            }

            for succ in successors {
                let succ_start = block_starts[succ];
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
