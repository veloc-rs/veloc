use alloc::vec::Vec;
use core::ops::{BitOr, BitOrAssign};
use hashbrown::{HashMap, HashSet};
use veloc_lir::{MachineFunction, Reg, UseDefChain};
use veloc_mir::Block;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
enum ChangeKind {
    InstOperands = 0,
    InstSemantics = 1,
    BlockLayout = 2,
    Cfg = 3,
    VregBanks = 4,
    SelectedOpcodes = 5,
    Regalloc = 6,
    StackFrame = 7,
    PhysicalRegs = 8,
    SymbolUses = 9,
    WholeFunction = 10,
}

const CHANGE_KIND_COUNT: usize = ChangeKind::WholeFunction as usize + 1;

/// 代码生成 pass 的变更集合。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ChangeSet {
    bits: u32,
}

impl ChangeSet {
    pub const NONE: Self = Self { bits: 0 };
    pub const INST_OPERANDS: Self = Self::single(ChangeKind::InstOperands);
    pub const INST_SEMANTICS: Self = Self::single(ChangeKind::InstSemantics);
    pub const BLOCK_LAYOUT: Self = Self::single(ChangeKind::BlockLayout);
    pub const CFG: Self = Self::single(ChangeKind::Cfg);
    pub const VREG_BANKS: Self = Self::single(ChangeKind::VregBanks);
    pub const SELECTED_OPCODES: Self = Self::single(ChangeKind::SelectedOpcodes);
    pub const REGALLOC: Self = Self::single(ChangeKind::Regalloc);
    pub const STACK_FRAME: Self = Self::single(ChangeKind::StackFrame);
    pub const PHYSICAL_REGS: Self = Self::single(ChangeKind::PhysicalRegs);
    pub const SYMBOL_USES: Self = Self::single(ChangeKind::SymbolUses);
    pub const WHOLE_FUNCTION: Self = Self::single(ChangeKind::WholeFunction);

    const fn single(kind: ChangeKind) -> Self {
        Self {
            bits: 1u32 << (kind as u32),
        }
    }

    pub const fn bits(self) -> u32 {
        self.bits
    }

    pub const fn is_empty(self) -> bool {
        self.bits == 0
    }

    pub const fn contains(self, other: Self) -> bool {
        (self.bits & other.bits) == other.bits
    }

    pub const fn intersects(self, other: Self) -> bool {
        (self.bits & other.bits) != 0
    }

    pub const fn normalized(self) -> Self {
        let mut bits = self.bits;
        if (bits & Self::CFG.bits) != 0 {
            bits |= Self::BLOCK_LAYOUT.bits;
        }
        if (bits & Self::WHOLE_FUNCTION.bits) != 0 {
            bits = Self::WHOLE_FUNCTION.bits
                | Self::CFG.bits
                | Self::BLOCK_LAYOUT.bits
                | Self::INST_OPERANDS.bits
                | Self::INST_SEMANTICS.bits
                | Self::VREG_BANKS.bits
                | Self::SELECTED_OPCODES.bits
                | Self::REGALLOC.bits
                | Self::STACK_FRAME.bits
                | Self::PHYSICAL_REGS.bits
                | Self::SYMBOL_USES.bits;
        }
        Self { bits }
    }

    fn kinds(self) -> impl Iterator<Item = ChangeKind> {
        [
            ChangeKind::InstOperands,
            ChangeKind::InstSemantics,
            ChangeKind::BlockLayout,
            ChangeKind::Cfg,
            ChangeKind::VregBanks,
            ChangeKind::SelectedOpcodes,
            ChangeKind::Regalloc,
            ChangeKind::StackFrame,
            ChangeKind::PhysicalRegs,
            ChangeKind::SymbolUses,
            ChangeKind::WholeFunction,
        ]
        .into_iter()
        .filter(move |kind| (self.normalized().bits & (1u32 << (*kind as u32))) != 0)
    }
}

impl BitOr for ChangeSet {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self {
            bits: self.bits | rhs.bits,
        }
        .normalized()
    }
}

impl BitOrAssign for ChangeSet {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = (*self | rhs).normalized();
    }
}

/// pass 执行效果。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PassEffect {
    pub change_set: ChangeSet,
}

impl PassEffect {
    pub const NONE: Self = Self {
        change_set: ChangeSet::NONE,
    };

    pub const fn new(change_set: ChangeSet) -> Self {
        Self { change_set }
    }
}

#[derive(Debug, Clone)]
pub struct AnalysisCache<T> {
    built_revision: u64,
    value: T,
}

impl<T> AnalysisCache<T> {
    fn new(built_revision: u64, value: T) -> Self {
        Self {
            built_revision,
            value,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CfgInfo {
    preds: HashMap<Block, Vec<Block>>,
    succs: HashMap<Block, Vec<Block>>,
}

impl CfgInfo {
    pub fn preds(&self, block: Block) -> &[Block] {
        self.preds.get(&block).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn succs(&self, block: Block) -> &[Block] {
        self.succs.get(&block).map(|v| v.as_slice()).unwrap_or(&[])
    }
}

#[derive(Debug, Clone, Default)]
pub struct DominatorTree {
    doms: HashMap<Block, HashSet<Block>>,
}

impl DominatorTree {
    pub fn dominates(&self, a: Block, b: Block) -> bool {
        self.doms.get(&b).is_some_and(|set| set.contains(&a))
    }
}

#[derive(Debug, Clone, Default)]
pub struct PostDominatorTree {
    post_doms: HashMap<Block, HashSet<Block>>,
}

impl PostDominatorTree {
    pub fn post_dominates(&self, a: Block, b: Block) -> bool {
        self.post_doms.get(&b).is_some_and(|set| set.contains(&a))
    }
}

#[derive(Debug, Clone, Default)]
pub struct LivenessInfo {
    live_in: HashMap<Block, HashSet<Reg>>,
    live_out: HashMap<Block, HashSet<Reg>>,
}

impl LivenessInfo {
    pub fn live_in(&self, block: Block) -> Option<&HashSet<Reg>> {
        self.live_in.get(&block)
    }

    pub fn live_out(&self, block: Block) -> Option<&HashSet<Reg>> {
        self.live_out.get(&block)
    }
}

#[derive(Debug, Clone, Default)]
pub struct LoopInfo {
    backedges: Vec<(Block, Block)>,
}

impl LoopInfo {
    pub fn backedges(&self) -> &[(Block, Block)] {
        &self.backedges
    }
}

#[derive(Debug, Clone, Default)]
pub struct RegisterPressure {
    pub per_block_max_live: HashMap<Block, usize>,
}

#[derive(Debug, Clone, Default)]
pub struct StackFrameSummary {
    pub local_size: u32,
    pub callee_saved_size: u32,
    pub total_size: u32,
    pub slot_count: usize,
}

/// 函数级分析上下文。
#[derive(Debug, Clone, Default)]
pub struct FunctionAnalysisCtx {
    revision: u64,
    last_changed_revision: [u64; CHANGE_KIND_COUNT],
    use_def: Option<AnalysisCache<UseDefChain>>,
    cfg: Option<AnalysisCache<CfgInfo>>,
    dominators: Option<AnalysisCache<DominatorTree>>,
    post_dominators: Option<AnalysisCache<PostDominatorTree>>,
    liveness: Option<AnalysisCache<LivenessInfo>>,
    loop_info: Option<AnalysisCache<LoopInfo>>,
    register_pressure: Option<AnalysisCache<RegisterPressure>>,
    stack_frame_summary: Option<AnalysisCache<StackFrameSummary>>,
}

impl FunctionAnalysisCtx {
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn apply(&mut self, change_set: ChangeSet) {
        let change_set = change_set.normalized();
        if change_set.is_empty() {
            return;
        }

        self.revision += 1;
        for kind in change_set.kinds() {
            self.last_changed_revision[kind as usize] = self.revision;
        }
    }

    fn is_cache_stale(&self, built_revision: u64, deps: ChangeSet) -> bool {
        deps.kinds()
            .any(|kind| self.last_changed_revision[kind as usize] > built_revision)
    }

    pub fn use_def<S>(&mut self, mfunc: &mut MachineFunction<S>) -> &UseDefChain {
        let deps = ChangeSet::INST_OPERANDS | ChangeSet::INST_SEMANTICS | ChangeSet::BLOCK_LAYOUT;
        let stale = self
            .use_def
            .as_ref()
            .is_none_or(|cache| self.is_cache_stale(cache.built_revision, deps));
        if stale {
            self.use_def = Some(AnalysisCache::new(self.revision, compute_use_def(mfunc)));
        }
        &self.use_def.as_ref().unwrap().value
    }

    pub fn cfg<S>(&mut self, mfunc: &MachineFunction<S>) -> &CfgInfo {
        let deps = ChangeSet::CFG;
        let stale = self
            .cfg
            .as_ref()
            .is_none_or(|cache| self.is_cache_stale(cache.built_revision, deps));
        if stale {
            self.cfg = Some(AnalysisCache::new(self.revision, compute_cfg(mfunc)));
        }
        &self.cfg.as_ref().unwrap().value
    }

    pub fn dominators<S>(&mut self, mfunc: &MachineFunction<S>) -> &DominatorTree {
        let deps = ChangeSet::CFG;
        let stale = self
            .dominators
            .as_ref()
            .is_none_or(|cache| self.is_cache_stale(cache.built_revision, deps));
        if stale {
            let cfg = self.cfg(mfunc).clone();
            self.dominators = Some(AnalysisCache::new(
                self.revision,
                compute_dominators(mfunc, &cfg),
            ));
        }
        &self.dominators.as_ref().unwrap().value
    }

    pub fn post_dominators<S>(&mut self, mfunc: &MachineFunction<S>) -> &PostDominatorTree {
        let deps = ChangeSet::CFG;
        let stale = self
            .post_dominators
            .as_ref()
            .is_none_or(|cache| self.is_cache_stale(cache.built_revision, deps));
        if stale {
            let cfg = self.cfg(mfunc).clone();
            self.post_dominators = Some(AnalysisCache::new(
                self.revision,
                compute_post_dominators(mfunc, &cfg),
            ));
        }
        &self.post_dominators.as_ref().unwrap().value
    }

    pub fn liveness<S>(&mut self, mfunc: &MachineFunction<S>) -> &LivenessInfo {
        let deps = ChangeSet::CFG | ChangeSet::INST_OPERANDS | ChangeSet::REGALLOC;
        let stale = self
            .liveness
            .as_ref()
            .is_none_or(|cache| self.is_cache_stale(cache.built_revision, deps));
        if stale {
            let cfg = self.cfg(mfunc).clone();
            self.liveness = Some(AnalysisCache::new(
                self.revision,
                compute_liveness(mfunc, &cfg),
            ));
        }
        &self.liveness.as_ref().unwrap().value
    }

    pub fn loop_info<S>(&mut self, mfunc: &MachineFunction<S>) -> &LoopInfo {
        let deps = ChangeSet::CFG;
        let stale = self
            .loop_info
            .as_ref()
            .is_none_or(|cache| self.is_cache_stale(cache.built_revision, deps));
        if stale {
            let cfg = self.cfg(mfunc).clone();
            let dom = self.dominators(mfunc).clone();
            self.loop_info = Some(AnalysisCache::new(
                self.revision,
                compute_loop_info(&cfg, &dom),
            ));
        }
        &self.loop_info.as_ref().unwrap().value
    }

    pub fn register_pressure<S>(&mut self, mfunc: &MachineFunction<S>) -> &RegisterPressure {
        let deps = ChangeSet::REGALLOC | ChangeSet::INST_OPERANDS | ChangeSet::BLOCK_LAYOUT;
        let stale = self
            .register_pressure
            .as_ref()
            .is_none_or(|cache| self.is_cache_stale(cache.built_revision, deps));
        if stale {
            let liveness = self.liveness(mfunc).clone();
            self.register_pressure = Some(AnalysisCache::new(
                self.revision,
                compute_register_pressure(mfunc, &liveness),
            ));
        }
        &self.register_pressure.as_ref().unwrap().value
    }

    pub fn stack_frame_summary<S>(&mut self, mfunc: &MachineFunction<S>) -> &StackFrameSummary {
        let deps = ChangeSet::STACK_FRAME | ChangeSet::REGALLOC;
        let stale = self
            .stack_frame_summary
            .as_ref()
            .is_none_or(|cache| self.is_cache_stale(cache.built_revision, deps));
        if stale {
            self.stack_frame_summary = Some(AnalysisCache::new(
                self.revision,
                StackFrameSummary {
                    local_size: mfunc.stack_frame.local_size,
                    callee_saved_size: mfunc.stack_frame.callee_saved_size,
                    total_size: mfunc.stack_frame.total_size,
                    slot_count: mfunc.stack_frame.slots.len(),
                },
            ));
        }
        &self.stack_frame_summary.as_ref().unwrap().value
    }
}

fn compute_use_def<S>(mfunc: &MachineFunction<S>) -> UseDefChain {
    let mut use_def = UseDefChain::default();
    for block in &mfunc.blocks {
        for &inst_id in &block.insts {
            let inst = &mfunc.dfg[inst_id];
            use_def.add_inst(inst_id, inst);
        }
    }
    use_def
}

/// 模块级分析上下文。
#[derive(Debug, Clone, Default)]
pub struct ModuleAnalysisCtx {
    revision: u64,
}

impl ModuleAnalysisCtx {
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn apply(&mut self, change_set: ChangeSet) {
        if !change_set.is_empty() {
            self.revision += 1;
        }
    }
}

fn compute_cfg<S>(mfunc: &MachineFunction<S>) -> CfgInfo {
    let mut preds: HashMap<Block, Vec<Block>> = HashMap::new();
    let mut succs: HashMap<Block, Vec<Block>> = HashMap::new();

    let mut block_order = Vec::new();
    for block in &mfunc.blocks {
        block_order.push(block.id);
        preds.entry(block.id).or_default();
        succs.entry(block.id).or_default();
    }

    for (index, block) in mfunc.blocks.iter().enumerate() {
        let mut block_succs = Vec::new();
        if let Some(&last_inst_id) = block.insts.last() {
            let inst = &mfunc.dfg[last_inst_id];
            if let Ok(br) = inst.as_branch() {
                block_succs.push(br.target);
            } else if let Ok(br) = inst.as_branch_cond() {
                block_succs.push(br.then_blk);
                block_succs.push(br.else_blk);
            } else if let Some(veloc_lir::InstExtra::BrTable(info)) = mfunc.inst_extra(last_inst_id)
            {
                for target in &info.targets {
                    block_succs.push(target.block);
                }
            } else if !matches!(inst.generic_opcode(), Some(veloc_lir::GenericOpcode::G_RET)) {
                if let Some(next) = block_order.get(index + 1).copied() {
                    block_succs.push(next);
                }
            }
        } else if let Some(next) = block_order.get(index + 1).copied() {
            block_succs.push(next);
        }

        block_succs.sort();
        block_succs.dedup();
        succs.insert(block.id, block_succs.clone());
        for succ in block_succs {
            preds.entry(succ).or_default().push(block.id);
        }
    }

    CfgInfo { preds, succs }
}

fn compute_dominators<S>(mfunc: &MachineFunction<S>, cfg: &CfgInfo) -> DominatorTree {
    let blocks: Vec<Block> = mfunc.blocks.iter().map(|b| b.id).collect();
    let Some(entry) = blocks.first().copied() else {
        return DominatorTree::default();
    };

    let mut doms: HashMap<Block, HashSet<Block>> = HashMap::new();
    let all_blocks: HashSet<Block> = blocks.iter().copied().collect();

    for &block in &blocks {
        if block == entry {
            doms.insert(block, [block].into_iter().collect());
        } else {
            doms.insert(block, all_blocks.clone());
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for &block in blocks.iter().skip(1) {
            let preds = cfg.preds(block);
            if preds.is_empty() {
                continue;
            }
            let mut new_set = all_blocks.clone();
            for pred in preds {
                if let Some(pred_doms) = doms.get(pred) {
                    new_set = new_set
                        .intersection(pred_doms)
                        .copied()
                        .collect::<HashSet<_>>();
                }
            }
            new_set.insert(block);
            if doms.get(&block) != Some(&new_set) {
                doms.insert(block, new_set);
                changed = true;
            }
        }
    }

    DominatorTree { doms }
}

fn compute_post_dominators<S>(mfunc: &MachineFunction<S>, cfg: &CfgInfo) -> PostDominatorTree {
    let blocks: Vec<Block> = mfunc.blocks.iter().map(|b| b.id).collect();
    let exits: Vec<Block> = blocks
        .iter()
        .copied()
        .filter(|block| cfg.succs(*block).is_empty())
        .collect();
    if blocks.is_empty() {
        return PostDominatorTree::default();
    }

    let all_blocks: HashSet<Block> = blocks.iter().copied().collect();
    let mut post_doms: HashMap<Block, HashSet<Block>> = HashMap::new();

    for &block in &blocks {
        if exits.contains(&block) {
            post_doms.insert(block, [block].into_iter().collect());
        } else {
            post_doms.insert(block, all_blocks.clone());
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for &block in &blocks {
            let succs = cfg.succs(block);
            if succs.is_empty() {
                continue;
            }
            let mut new_set = all_blocks.clone();
            for succ in succs {
                if let Some(succ_post_doms) = post_doms.get(succ) {
                    new_set = new_set
                        .intersection(succ_post_doms)
                        .copied()
                        .collect::<HashSet<_>>();
                }
            }
            new_set.insert(block);
            if post_doms.get(&block) != Some(&new_set) {
                post_doms.insert(block, new_set);
                changed = true;
            }
        }
    }

    PostDominatorTree { post_doms }
}

fn compute_liveness<S>(mfunc: &MachineFunction<S>, cfg: &CfgInfo) -> LivenessInfo {
    let mut block_uses: HashMap<Block, HashSet<Reg>> = HashMap::new();
    let mut block_defs: HashMap<Block, HashSet<Reg>> = HashMap::new();
    let mut live_in: HashMap<Block, HashSet<Reg>> = HashMap::new();
    let mut live_out: HashMap<Block, HashSet<Reg>> = HashMap::new();

    for block in &mfunc.blocks {
        let mut uses = HashSet::new();
        let mut defs = HashSet::new();
        for &inst_id in &block.insts {
            let inst = &mfunc.dfg[inst_id];
            for reg in inst.uses() {
                if !defs.contains(&reg) {
                    uses.insert(reg);
                }
            }
            for reg in inst.defs() {
                defs.insert(reg);
            }
        }
        block_uses.insert(block.id, uses);
        block_defs.insert(block.id, defs);
        live_in.insert(block.id, HashSet::new());
        live_out.insert(block.id, HashSet::new());
    }

    let mut changed = true;
    while changed {
        changed = false;
        for block in mfunc.blocks.iter().rev() {
            let mut out = HashSet::new();
            for succ in cfg.succs(block.id) {
                if let Some(succ_in) = live_in.get(succ) {
                    out.extend(succ_in.iter().copied());
                }
            }
            let mut new_in = block_uses.get(&block.id).cloned().unwrap_or_default();
            let defs = block_defs.get(&block.id).cloned().unwrap_or_default();
            for reg in &out {
                if !defs.contains(reg) {
                    new_in.insert(*reg);
                }
            }
            if live_out.get(&block.id) != Some(&out) {
                live_out.insert(block.id, out);
                changed = true;
            }
            if live_in.get(&block.id) != Some(&new_in) {
                live_in.insert(block.id, new_in);
                changed = true;
            }
        }
    }

    LivenessInfo { live_in, live_out }
}

fn compute_loop_info(cfg: &CfgInfo, dom: &DominatorTree) -> LoopInfo {
    let mut backedges = Vec::new();
    for (&block, succs) in &cfg.succs {
        for &succ in succs {
            if dom.dominates(succ, block) {
                backedges.push((block, succ));
            }
        }
    }
    LoopInfo { backedges }
}

fn compute_register_pressure<S>(
    mfunc: &MachineFunction<S>,
    liveness: &LivenessInfo,
) -> RegisterPressure {
    let mut per_block_max_live = HashMap::new();
    for block in &mfunc.blocks {
        let live = liveness.live_out(block.id).cloned().unwrap_or_default();
        per_block_max_live.insert(block.id, live.len());
    }
    RegisterPressure { per_block_max_live }
}

#[cfg(test)]
mod tests {
    use super::{ChangeSet, FunctionAnalysisCtx};
    use veloc_lir::stages::RawLir;
    use veloc_lir::{MachineBlock, MachineFunction, MachineInst, Writable};
    use veloc_mir::{Block, Type};

    #[test]
    fn changeset_cfg_implies_block_layout() {
        let normalized = ChangeSet::CFG.normalized();
        assert!(normalized.contains(ChangeSet::CFG));
        assert!(normalized.contains(ChangeSet::BLOCK_LAYOUT));
    }

    #[test]
    fn use_def_change_does_not_invalidate_cfg() {
        let mut mfunc = MachineFunction::<RawLir>::new("test".into());
        mfunc.blocks.push(MachineBlock::new(Block::from_u32(0)));
        let a = mfunc.alloc_vreg(Type::I64);
        let b = mfunc.alloc_vreg(Type::I64);
        let copy = mfunc.alloc_inst(MachineInst::build_copy(Writable(a), b));
        mfunc.append_inst_id_to_block(0, copy);

        let mut analyses = FunctionAnalysisCtx::default();
        let cfg_before = analyses.cfg(&mfunc) as *const _;
        analyses.apply(ChangeSet::INST_OPERANDS);
        let cfg_after = analyses.cfg(&mfunc) as *const _;
        assert_eq!(cfg_before, cfg_after);
    }

    #[test]
    fn cfg_change_invalidates_cfg_and_dependents() {
        let mut mfunc = MachineFunction::<RawLir>::new("test".into());
        mfunc.blocks.push(MachineBlock::new(Block::from_u32(0)));
        mfunc.blocks.push(MachineBlock::new(Block::from_u32(1)));
        let mut analyses = FunctionAnalysisCtx::default();
        let succs_before = analyses.cfg(&mfunc).succs(Block::from_u32(0)).len();
        analyses.apply(ChangeSet::CFG);
        mfunc.blocks.push(MachineBlock::new(Block::from_u32(2)));
        let succs_after = analyses.cfg(&mfunc).succs(Block::from_u32(1)).len();
        let dom = analyses.dominators(&mfunc);
        assert_eq!(succs_before, 1);
        assert_eq!(succs_after, 1);
        assert!(dom.dominates(Block::from_u32(0), Block::from_u32(2)));
    }

    #[test]
    fn stack_frame_change_does_not_invalidate_cfg() {
        let mut mfunc = MachineFunction::<RawLir>::new("test".into());
        mfunc.blocks.push(MachineBlock::new(Block::from_u32(0)));
        let mut analyses = FunctionAnalysisCtx::default();
        let cfg_before = analyses.cfg(&mfunc) as *const _;
        analyses.apply(ChangeSet::STACK_FRAME);
        let cfg_after = analyses.cfg(&mfunc) as *const _;
        assert_eq!(cfg_before, cfg_after);
    }
}
