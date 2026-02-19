use bitvec::prelude::*;
use cranelift_entity::{EntityRef, SecondaryMap};
use veloc_ir::{Block, Function, Inst, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveRange {
    pub start: u32,
    pub end: u32,
}

/// 完整的活跃区间（支持 Interval Holes）
#[derive(Debug, Clone, Default)]
pub struct LiveInterval {
    /// 必须保证：按 start 升序排列，且互相不重叠、不相邻
    pub ranges: Vec<LiveRange>,
}

impl LiveInterval {
    /// 向区间中添加一段新的活跃范围。通常在逆向扫描中使用。
    pub fn add_range(&mut self, mut start: u32, mut end: u32) {
        if start >= end {
            return;
        }

        let mut i = 0;
        while i < self.ranges.len() {
            let r = self.ranges[i];
            if start <= r.end && end >= r.start {
                // 存在重叠或相邻，进行合并
                start = start.min(r.start);
                end = end.max(r.end);
                self.ranges.remove(i);
                // 继续检查合并后的区间是否与后续区间重叠
                continue;
            }
            if end < r.start {
                // 新区间在当前区间之前且不重叠
                self.ranges.insert(i, LiveRange { start, end });
                return;
            }
            i += 1;
        }
        // 新区间在所有现有区间之后
        self.ranges.push(LiveRange { start, end });
    }

    pub fn start(&self) -> u32 {
        self.ranges.first().map(|r| r.start).unwrap_or(0)
    }

    pub fn end(&self) -> u32 {
        self.ranges.last().map(|r| r.end).unwrap_or(0)
    }

    pub fn conflicts_with(&self, other: &Self) -> bool {
        let mut i = 0;
        let mut j = 0;
        while i < self.ranges.len() && j < other.ranges.len() {
            let r1 = &self.ranges[i];
            let r2 = &other.ranges[j];
            if r1.start < r2.end && r2.start < r1.end { return true; }
            if r1.end < r2.end { i += 1; } else { j += 1; }
        }
        false
    }
}


pub struct Liveness {
    pub intervals: SecondaryMap<Value, LiveInterval>,
    pub block_starts: SecondaryMap<Block, u32>,
    pub block_ends: SecondaryMap<Block, u32>,
    pub inst_pcs: SecondaryMap<Inst, u32>,
}

pub fn analyze_liveness(func: &Function) -> Liveness {
    let entry = func.entry_block.expect("Function must have entry block");
    let rpo = func.layout.compute_rpo(entry);
    let num_values = func.dfg.values.len();
    let num_blocks = func.layout.blocks.len();
    let num_insts = func.dfg.instructions.len();

    let mut intervals: SecondaryMap<Value, LiveInterval> = SecondaryMap::with_capacity(num_values);
    let mut block_starts: SecondaryMap<Block, u32> = SecondaryMap::with_capacity(num_blocks);
    let mut block_ends: SecondaryMap<Block, u32> = SecondaryMap::with_capacity(num_blocks);
    let mut inst_pcs: SecondaryMap<Inst, u32> = SecondaryMap::with_capacity(num_insts);

    let mut def_pc: SecondaryMap<Value, u32> = SecondaryMap::with_capacity(num_values);
    let mut def_block: SecondaryMap<Value, Option<Block>> = SecondaryMap::with_capacity(num_values);
    let mut uses: SecondaryMap<Value, Vec<(u32, Block)>> = SecondaryMap::with_capacity(num_values);

    let mut inst_pc = 0u32;

    // 1. Pass: Assign PC and collect Defs/Uses
    for &block in &rpo {
        block_starts[block] = inst_pc;

        // Block parameters are defined at the start of the block
        for &param in &func.layout.blocks[block].params {
            def_pc[param] = inst_pc;
            def_block[param] = Some(block);
        }
        inst_pc += 2;

        for &inst in &func.layout.blocks[block].insts {
            let current_inst_pc = inst_pc;
            inst_pcs[inst] = current_inst_pc;

            // Record uses
            inst.visit_operands(&func.dfg, |v| {
                uses[v].push((current_inst_pc, block));
            });

            // Record defs (results are defined at inst_pc + 1)
            for &res in func.dfg.inst_results(inst) {
                def_pc[res] = current_inst_pc + 1;
                def_block[res] = Some(block);
            }
            inst_pc += 2;
        }
        block_ends[block] = inst_pc;
    }

    // 2. Pass: Compute live intervals for each variable using backward walk
    let mut live_in = bitvec![u64, Lsb0; 0; num_blocks];
    let mut worklist = Vec::with_capacity(num_blocks);

    let values_to_process: Vec<Value> = func.dfg.values.keys()
        .filter(|&v| def_block[v].is_some())
        .collect();

    for v in values_to_process {
        let v_def_block = def_block[v].unwrap();
        let v_def_pc = def_pc[v];

        // Ensure definition point is included
        intervals[v].add_range(v_def_pc, v_def_pc + 1);

        live_in.fill(false);
        worklist.clear();

        for &(use_pc, use_block) in &uses[v] {
            if use_block == v_def_block {
                intervals[v].add_range(v_def_pc, use_pc + 1);
            } else {
                intervals[v].add_range(block_starts[use_block], use_pc + 1);
                if !live_in[use_block.index()] {
                    live_in.set(use_block.index(), true);
                    for &pred in &func.layout.blocks[use_block].preds {
                        if !live_in[pred.index()] {
                            worklist.push(pred);
                        }
                    }
                }
            }
        }

        while let Some(b) = worklist.pop() {
            if live_in[b.index()] {
                continue;
            }
            live_in.set(b.index(), true);

            if b == v_def_block {
                intervals[v].add_range(v_def_pc, block_ends[b]);
            } else {
                intervals[v].add_range(block_starts[b], block_ends[b]);
                for &pred in &func.layout.blocks[b].preds {
                    if !live_in[pred.index()] {
                        worklist.push(pred);
                    }
                }
            }
        }
    }

    Liveness {
        intervals,
        block_starts,
        block_ends,
        inst_pcs,
    }
}


