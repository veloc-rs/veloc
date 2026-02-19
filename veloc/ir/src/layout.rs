use super::inst::Inst;
use super::types::{Block, Value};
use alloc::vec::Vec;
use cranelift_entity::{PrimaryMap, SecondaryMap};

#[derive(Debug, Clone)]
pub struct BlockData {
    pub params: Vec<Value>,
    pub preds: Vec<Block>,
    pub succs: Vec<Block>,
    pub insts: Vec<Inst>,
    pub is_sealed: bool,
}

#[derive(Debug, Clone)]
pub struct Layout {
    pub blocks: PrimaryMap<Block, BlockData>,
    pub block_order: Vec<Block>,
}

impl Layout {
    pub fn new() -> Self {
        Self {
            blocks: PrimaryMap::new(),
            block_order: Vec::new(),
        }
    }

    pub fn create_block(&mut self) -> Block {
        self.blocks.push(BlockData {
            params: Vec::new(),
            preds: Vec::new(),
            succs: Vec::new(),
            insts: Vec::new(),
            is_sealed: false,
        })
    }

    pub fn append_block(&mut self, block: Block) {
        self.block_order.push(block);
    }

    pub fn append_inst(&mut self, block: Block, inst: Inst) {
        self.blocks[block].insts.push(inst);
    }

    pub fn add_edge(&mut self, from: Block, to: Block) {
        if !self.blocks[from].succs.contains(&to) {
            self.blocks[from].succs.push(to);
        }
        if !self.blocks[to].preds.contains(&from) {
            self.blocks[to].preds.push(from);
        }
    }

    /// 计算从 entry 开始的后序遍历 (Post-Order)
    /// 返回的列表满足：如果 A 支配 B 且 A != B，则 B 在 A 之前出现（对于无环图）
    pub fn compute_post_order(&self, entry: Block) -> Vec<Block> {
        let mut post_order = Vec::with_capacity(self.blocks.len());
        let mut visited = SecondaryMap::<Block, bool>::with_capacity(self.blocks.len());
        let mut stack = Vec::new();

        stack.push((entry, false));

        while let Some((block, is_processed)) = stack.pop() {
            if is_processed {
                post_order.push(block);
                continue;
            }

            if visited[block] {
                continue;
            }

            visited[block] = true;
            // 重新压入当前块，标记为 is_processed=true
            // 这样它会在所有子节点处理完后被弹出并加入 post_order
            stack.push((block, true));

            // 对后继节点进行深度优先探索
            // 倒序压栈是为了在使用 pop 时能尽量保持和 succs 列表一致的探索顺序
            for &succ in self.blocks[block].succs.iter().rev() {
                if !visited[succ] {
                    stack.push((succ, false));
                }
            }
        }

        post_order
    }

    /// 计算从 entry 开始的逆后序遍历 (Reverse Post-Order)
    /// RPO 是数据流分析的最佳顺序，因为它保证了在大多数情况下 Def 在 Use 之前被访问
    pub fn compute_rpo(&self, entry: Block) -> Vec<Block> {
        let mut po = self.compute_post_order(entry);
        po.reverse();
        po
    }
}
