use crate::target::arch::TargetInstructions;
use alloc::vec::Vec;
use core::ops::{BitOr, BitOrAssign};
use hashbrown::{HashMap, HashSet};
use veloc_lir::BlockId as Block;
use veloc_lir::{MachineFunction, Reg};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
enum ChangeKind {
    InstOperands = 0,
    InstSemantics = 1,
    BlockLayout = 2,
    Cfg = 3,
    SelectedOpcodes = 4,
    Regalloc = 5,
    StackFrame = 6,
    PhysicalRegs = 7,
    SymbolUses = 8,
    WholeFunction = 9,
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

    pub fn cfg(&mut self, mfunc: &MachineFunction, target: &dyn TargetInstructions) -> &CfgInfo {
        let deps = ChangeSet::CFG
            | ChangeSet::BLOCK_LAYOUT
            | ChangeSet::INST_OPERANDS
            | ChangeSet::INST_SEMANTICS
            | ChangeSet::SELECTED_OPCODES;
        let stale = self
            .cfg
            .as_ref()
            .is_none_or(|cache| self.is_cache_stale(cache.built_revision, deps));
        if stale {
            self.cfg = Some(AnalysisCache::new(
                self.revision,
                compute_cfg(mfunc, target),
            ));
        }
        &self.cfg.as_ref().unwrap().value
    }

    pub fn dominators(
        &mut self,
        mfunc: &MachineFunction,
        target: &dyn TargetInstructions,
    ) -> &DominatorTree {
        let deps = ChangeSet::CFG
            | ChangeSet::BLOCK_LAYOUT
            | ChangeSet::INST_OPERANDS
            | ChangeSet::INST_SEMANTICS
            | ChangeSet::SELECTED_OPCODES;
        let stale = self
            .dominators
            .as_ref()
            .is_none_or(|cache| self.is_cache_stale(cache.built_revision, deps));
        if stale {
            let cfg = self.cfg(mfunc, target).clone();
            self.dominators = Some(AnalysisCache::new(
                self.revision,
                compute_dominators(mfunc, &cfg),
            ));
        }
        &self.dominators.as_ref().unwrap().value
    }

    pub fn post_dominators(
        &mut self,
        mfunc: &MachineFunction,
        target: &dyn TargetInstructions,
    ) -> &PostDominatorTree {
        let deps = ChangeSet::CFG
            | ChangeSet::BLOCK_LAYOUT
            | ChangeSet::INST_OPERANDS
            | ChangeSet::INST_SEMANTICS
            | ChangeSet::SELECTED_OPCODES;
        let stale = self
            .post_dominators
            .as_ref()
            .is_none_or(|cache| self.is_cache_stale(cache.built_revision, deps));
        if stale {
            let cfg = self.cfg(mfunc, target).clone();
            self.post_dominators = Some(AnalysisCache::new(
                self.revision,
                compute_post_dominators(mfunc, &cfg),
            ));
        }
        &self.post_dominators.as_ref().unwrap().value
    }

    pub fn liveness(
        &mut self,
        mfunc: &MachineFunction,
        target: &dyn TargetInstructions,
    ) -> &LivenessInfo {
        let deps = ChangeSet::CFG
            | ChangeSet::BLOCK_LAYOUT
            | ChangeSet::INST_OPERANDS
            | ChangeSet::INST_SEMANTICS
            | ChangeSet::SELECTED_OPCODES
            | ChangeSet::REGALLOC;
        let stale = self
            .liveness
            .as_ref()
            .is_none_or(|cache| self.is_cache_stale(cache.built_revision, deps));
        if stale {
            let cfg = self.cfg(mfunc, target).clone();
            self.liveness = Some(AnalysisCache::new(
                self.revision,
                compute_liveness(mfunc, &cfg),
            ));
        }
        &self.liveness.as_ref().unwrap().value
    }

    pub fn loop_info(
        &mut self,
        mfunc: &MachineFunction,
        target: &dyn TargetInstructions,
    ) -> &LoopInfo {
        let deps = ChangeSet::CFG
            | ChangeSet::BLOCK_LAYOUT
            | ChangeSet::INST_OPERANDS
            | ChangeSet::INST_SEMANTICS
            | ChangeSet::SELECTED_OPCODES;
        let stale = self
            .loop_info
            .as_ref()
            .is_none_or(|cache| self.is_cache_stale(cache.built_revision, deps));
        if stale {
            let cfg = self.cfg(mfunc, target).clone();
            let dom = self.dominators(mfunc, target).clone();
            self.loop_info = Some(AnalysisCache::new(
                self.revision,
                compute_loop_info(&cfg, &dom),
            ));
        }
        &self.loop_info.as_ref().unwrap().value
    }

    pub fn register_pressure(
        &mut self,
        mfunc: &MachineFunction,
        target: &dyn TargetInstructions,
    ) -> &RegisterPressure {
        let deps = ChangeSet::CFG
            | ChangeSet::REGALLOC
            | ChangeSet::INST_OPERANDS
            | ChangeSet::INST_SEMANTICS
            | ChangeSet::SELECTED_OPCODES
            | ChangeSet::BLOCK_LAYOUT;
        let stale = self
            .register_pressure
            .as_ref()
            .is_none_or(|cache| self.is_cache_stale(cache.built_revision, deps));
        if stale {
            let liveness = self.liveness(mfunc, target).clone();
            self.register_pressure = Some(AnalysisCache::new(
                self.revision,
                compute_register_pressure(mfunc, &liveness),
            ));
        }
        &self.register_pressure.as_ref().unwrap().value
    }

    pub fn stack_frame_summary(&mut self, mfunc: &MachineFunction) -> &StackFrameSummary {
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

fn compute_cfg(mfunc: &MachineFunction, target: &dyn TargetInstructions) -> CfgInfo {
    let mut preds: HashMap<Block, Vec<Block>> = HashMap::new();
    let mut succs: HashMap<Block, Vec<Block>> = HashMap::new();

    let mut block_order = Vec::new();
    for block in mfunc.blocks() {
        block_order.push(block);
        preds.entry(block).or_default();
        succs.entry(block).or_default();
    }

    for (index, block) in mfunc.blocks().enumerate() {
        let mut block_succs = Vec::new();
        let mut falls_through = true;
        for id in mfunc.block_insts(block) {
            let inst = &mfunc.inst(id);
            let flow = target.control_flow(inst);
            if matches!(
                flow,
                veloc_lir::ControlFlow::Branch | veloc_lir::ControlFlow::Jump
            ) {
                for operand in inst.fields().iter() {
                    if let veloc_lir::InstField::Block(target) = operand {
                        block_succs.push(*target);
                    }
                }
                if let Some(veloc_lir::InstExtraRef::BrTable(info)) = mfunc.inst_extra(id) {
                    block_succs.extend(info.targets().map(|target| target.block));
                }
            }
            if matches!(
                flow,
                veloc_lir::ControlFlow::Jump
                    | veloc_lir::ControlFlow::Return
                    | veloc_lir::ControlFlow::Trap
            ) {
                falls_through = false;
                break;
            }
        }
        if falls_through {
            if let Some(next) = block_order.get(index + 1).copied() {
                block_succs.push(next);
            }
        }
        block_succs.sort();
        block_succs.dedup();
        succs.insert(block, block_succs.clone());
        for succ in block_succs {
            preds.entry(succ).or_default().push(block);
        }
    }

    CfgInfo { preds, succs }
}

fn compute_dominators(mfunc: &MachineFunction, cfg: &CfgInfo) -> DominatorTree {
    let blocks: Vec<Block> = mfunc.blocks().collect();
    let Some(entry) = mfunc.entry_block() else {
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

fn compute_post_dominators(mfunc: &MachineFunction, cfg: &CfgInfo) -> PostDominatorTree {
    let blocks: Vec<Block> = mfunc.blocks().collect();
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

fn compute_liveness(mfunc: &MachineFunction, cfg: &CfgInfo) -> LivenessInfo {
    let mut block_uses: HashMap<Block, HashSet<Reg>> = HashMap::new();
    let mut block_defs: HashMap<Block, HashSet<Reg>> = HashMap::new();
    let mut live_in: HashMap<Block, HashSet<Reg>> = HashMap::new();
    let mut live_out: HashMap<Block, HashSet<Reg>> = HashMap::new();

    for block in mfunc.blocks() {
        let mut uses = HashSet::new();
        let mut defs: HashSet<_> = mfunc.block_params(block).unwrap().iter().copied().collect();
        for inst_id in mfunc.block_insts(block) {
            let inst = &mfunc.inst(inst_id);
            for reg in inst.uses() {
                if !defs.contains(&reg) {
                    uses.insert(reg);
                }
            }
            for reg in inst.defs() {
                defs.insert(reg);
            }
        }
        block_uses.insert(block, uses);
        block_defs.insert(block, defs);
        live_in.insert(block, HashSet::new());
        live_out.insert(block, HashSet::new());
    }

    let mut changed = true;
    while changed {
        changed = false;
        for block in mfunc.blocks().rev() {
            let mut out = HashSet::new();
            for succ in cfg.succs(block) {
                if let Some(succ_in) = live_in.get(succ) {
                    out.extend(succ_in.iter().copied());
                }
            }
            let mut new_in = block_uses.get(&block).cloned().unwrap_or_default();
            let defs = block_defs.get(&block).cloned().unwrap_or_default();
            for reg in &out {
                if !defs.contains(reg) {
                    new_in.insert(*reg);
                }
            }
            if live_out.get(&block) != Some(&out) {
                live_out.insert(block, out);
                changed = true;
            }
            if live_in.get(&block) != Some(&new_in) {
                live_in.insert(block, new_in);
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

fn compute_register_pressure(mfunc: &MachineFunction, liveness: &LivenessInfo) -> RegisterPressure {
    let mut per_block_max_live = HashMap::new();
    for block in mfunc.blocks() {
        let live = liveness.live_out(block).cloned().unwrap_or_default();
        per_block_max_live.insert(block, live.len());
    }
    RegisterPressure { per_block_max_live }
}

#[cfg(test)]
mod tests {
    use super::{ChangeSet, FunctionAnalysisCtx};
    use crate::target::arch::TargetConfig;
    use crate::target::x86_64::X86_64TargetMachine;
    use veloc_lir::BlockId as Block;
    use veloc_lir::InstBuild;
    use veloc_lir::MachineFunction;
    use veloc_mir::Type;

    #[test]
    fn generic_control_comes_from_definitions_not_layout_or_last_instruction() {
        let mut f = MachineFunction::new("control".into());
        for _id in 0..4 {
            f.editor().create_block();
        }
        {
            let id = f.editor().writer().trap();
            f.editor().append_inst(veloc_lir::BlockId::from_u32(0), id);
            id
        };
        // Dead instructions cannot introduce successors after a trap.
        {
            let id = f.editor().writer().br(Block::from_u32(2));
            f.editor().append_inst(veloc_lir::BlockId::from_u32(0), id);
            id
        };
        {
            let id = f.editor().writer().brcond(
                veloc_lir::Reg::new_vreg(0),
                Block::from_u32(0),
                Block::from_u32(3),
            );
            f.editor().append_inst(veloc_lir::BlockId::from_u32(1), id);
            id
        };
        {
            let id = f.editor().writer().ret(&[]);
            f.editor().append_inst(veloc_lir::BlockId::from_u32(2), id);
            id
        };
        let target = X86_64TargetMachine::new(TargetConfig::default()).unwrap();
        let mut analyses = FunctionAnalysisCtx::default();
        let cfg = analyses.cfg(&f, &target);
        assert!(cfg.succs(Block::from_u32(0)).is_empty());
        assert_eq!(
            cfg.succs(Block::from_u32(1)),
            &[Block::from_u32(0), Block::from_u32(3)]
        );
        assert!(cfg.succs(Block::from_u32(2)).is_empty());
    }

    #[test]
    fn changeset_cfg_implies_block_layout() {
        let normalized = ChangeSet::CFG.normalized();
        assert!(normalized.contains(ChangeSet::CFG));
        assert!(normalized.contains(ChangeSet::BLOCK_LAYOUT));
    }

    #[test]
    fn selected_control_distinguishes_branch_fallthrough_and_terminal_transfer() {
        use crate::target::x86_64::inst::TargetInst;
        use veloc_lir::{InstField, MachineOpcode};
        let target = X86_64TargetMachine::new(TargetConfig::default()).unwrap();
        let mut f = MachineFunction::new("selected".into());
        for _id in 0..8 {
            f.editor().create_block();
        }
        let mut emit = |block, op: TargetInst, targets: &[u32]| {
            {
                let id = f.editor().writer().write(
                    MachineOpcode::Target(op.as_u32()),
                    &[],
                    &[],
                    &targets
                        .iter()
                        .map(|&b| InstField::Block(Block::from_u32(b)))
                        .collect::<alloc::vec::Vec<_>>(),
                );
                f.editor().append_inst(Block::from_u32(block), id);
                id
            };
        };
        emit(0, TargetInst::X86Ret, &[]);
        emit(0, TargetInst::X86Jmp, &[7]); // Dead after return.
        emit(1, TargetInst::X86Ud2, &[]);
        emit(2, TargetInst::X86Jmp, &[0]);
        emit(3, TargetInst::X86Je, &[0]);
        emit(3, TargetInst::X86Jmp, &[2]); // No edge to layout block 4.
        emit(4, TargetInst::X86Je, &[1]); // False path falls through to 5.
        emit(5, TargetInst::X86Call, &[]); // Calls return to the next instruction.
        emit(6, TargetInst::X86Ret, &[]);
        let mut analyses = FunctionAnalysisCtx::default();
        let cfg = analyses.cfg(&f, &target);
        assert!(cfg.succs(Block::from_u32(0)).is_empty());
        assert!(cfg.succs(Block::from_u32(1)).is_empty());
        assert_eq!(cfg.succs(Block::from_u32(2)), &[Block::from_u32(0)]);
        assert_eq!(
            cfg.succs(Block::from_u32(3)),
            &[Block::from_u32(0), Block::from_u32(2)]
        );
        assert_eq!(
            cfg.succs(Block::from_u32(4)),
            &[Block::from_u32(1), Block::from_u32(5)]
        );
        assert_eq!(cfg.succs(Block::from_u32(5)), &[Block::from_u32(6)]);
        assert!(cfg.succs(Block::from_u32(6)).is_empty());
        assert!(cfg.succs(Block::from_u32(7)).is_empty());
    }

    #[test]
    fn layout_and_opcode_changes_invalidate_control_analyses() {
        use crate::target::x86_64::inst::TargetInst;
        use veloc_lir::MachineOpcode;
        let target = X86_64TargetMachine::new(TargetConfig::default()).unwrap();
        let mut f = MachineFunction::new("layout".into());
        for _id in 0..3 {
            f.editor().create_block();
        }
        let mut analyses = FunctionAnalysisCtx::default();
        assert_eq!(
            analyses.cfg(&f, &target).succs(Block::from_u32(0)),
            &[Block::from_u32(1)]
        );
        f.editor()
            .move_block_before(Block::from_u32(2), Block::from_u32(1));
        analyses.apply(ChangeSet::BLOCK_LAYOUT);
        assert_eq!(
            analyses.cfg(&f, &target).succs(Block::from_u32(0)),
            &[Block::from_u32(2)]
        );
        {
            let id = f.editor().writer().write(
                MachineOpcode::Target(TargetInst::X86Ret.as_u32()),
                &[],
                &[],
                &[],
            );
            f.editor().append_inst(veloc_lir::BlockId::from_u32(0), id);
            id
        };
        analyses.apply(ChangeSet::SELECTED_OPCODES);
        assert!(
            analyses
                .cfg(&f, &target)
                .succs(Block::from_u32(0))
                .is_empty()
        );
    }

    #[test]
    fn branch_operand_change_invalidates_cfg_and_liveness() {
        let mut f = MachineFunction::new("control".into());
        for _id in 0..3 {
            f.editor().create_block();
        }
        let value = f.editor().alloc_vreg(Type::I64);
        let jump = {
            let id = f.editor().writer().br(Block::from_u32(1));
            f.editor().append_inst(veloc_lir::BlockId::from_u32(0), id);
            id
        };
        {
            let id = f.editor().writer().ret(&[value]);
            f.editor().append_inst(veloc_lir::BlockId::from_u32(1), id);
            id
        };
        {
            let id = f.editor().writer().ret(&[]);
            f.editor().append_inst(veloc_lir::BlockId::from_u32(2), id);
            id
        };
        let target = X86_64TargetMachine::new(TargetConfig::default()).unwrap();
        let mut analyses = FunctionAnalysisCtx::default();
        assert_eq!(
            analyses.cfg(&f, &target).succs(Block::from_u32(0)),
            &[Block::from_u32(1)]
        );
        assert!(
            analyses
                .liveness(&f, &target)
                .live_out(Block::from_u32(0))
                .unwrap()
                .contains(&value)
        );
        f.editor().rewriter(jump).br(Block::from_u32(2));
        analyses.apply(ChangeSet::INST_OPERANDS);
        assert_eq!(
            analyses.cfg(&f, &target).succs(Block::from_u32(0)),
            &[Block::from_u32(2)]
        );
        assert!(
            !analyses
                .liveness(&f, &target)
                .live_out(Block::from_u32(0))
                .unwrap()
                .contains(&value)
        );
    }

    #[test]
    fn cfg_change_invalidates_cfg_and_dependents() {
        let target = X86_64TargetMachine::new(TargetConfig::default()).unwrap();
        let mut mfunc = MachineFunction::new("test".into());
        mfunc.editor().create_block();
        mfunc.editor().create_block();
        let mut analyses = FunctionAnalysisCtx::default();
        let succs_before = analyses
            .cfg(&mfunc, &target)
            .succs(Block::from_u32(0))
            .len();
        analyses.apply(ChangeSet::CFG);
        mfunc.editor().create_block();
        let succs_after = analyses
            .cfg(&mfunc, &target)
            .succs(Block::from_u32(1))
            .len();
        let dom = analyses.dominators(&mfunc, &target);
        assert_eq!(succs_before, 1);
        assert_eq!(succs_after, 1);
        assert!(dom.dominates(Block::from_u32(0), Block::from_u32(2)));
    }

    #[test]
    fn stack_frame_change_does_not_invalidate_cfg() {
        let target = X86_64TargetMachine::new(TargetConfig::default()).unwrap();
        let mut mfunc = MachineFunction::new("test".into());
        mfunc.editor().create_block();
        let mut analyses = FunctionAnalysisCtx::default();
        let cfg_before = analyses.cfg(&mfunc, &target) as *const _;
        analyses.apply(ChangeSet::STACK_FRAME);
        let cfg_after = analyses.cfg(&mfunc, &target) as *const _;
        assert_eq!(cfg_before, cfg_after);
    }
}
