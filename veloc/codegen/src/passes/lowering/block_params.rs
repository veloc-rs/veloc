use crate::error::Result;
use crate::mir::{
    BrTableInfo, BrTableTarget, GenericOpcode, InstExtra, MachineFunction, MachineInst, Reg,
    Writable,
};
use crate::pipeline::stages::{LegalizedMir, RawMir};
use crate::pipeline::{ChangeSet, FunctionPassContext, PassEffect, StageTransformPass};
use alloc::vec::Vec;
use hashbrown::HashMap;
use smallvec::SmallVec;
use veloc_ir::{Block, Type};

/// 将分支边上的 block 参数显式化为 copy/中间块。
pub struct BlockParamLoweringPass;

/// 并行拷贝生成时的临时工作区。
struct ParallelCopyScratch {
    pending: Vec<Option<(Reg, Reg)>>,
    out: Vec<MachineInst>,
    dest_counts: HashMap<Reg, usize>,
    source_users: HashMap<Reg, SmallVec<[usize; 4]>>,
    ready: Vec<usize>,
}

impl ParallelCopyScratch {
    fn new() -> Self {
        Self {
            pending: Vec::new(),
            out: Vec::new(),
            dest_counts: HashMap::new(),
            source_users: HashMap::new(),
            ready: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.pending.clear();
        self.out.clear();
        self.dest_counts.clear();
        self.source_users.clear();
        self.ready.clear();
    }
}

impl BlockParamLoweringPass {
    pub fn run<S>(mfunc: &mut MachineFunction<S>) -> Result<()> {
        let mut scratch = ParallelCopyScratch::new();
        let original_blocks = mfunc.num_blocks();
        for block_idx in 0..original_blocks {
            mfunc.rewrite_block(block_idx, |cursor| -> Result<()> {
                let inst = cursor.current_inst_clone();
                match inst.generic_opcode() {
                    Some(GenericOpcode::G_BR) => Self::lower_branch(cursor, &mut scratch, &inst)?,
                    Some(GenericOpcode::G_BRCOND) => {
                        Self::lower_cond_branch(cursor, &mut scratch, &inst)?
                    }
                    Some(GenericOpcode::G_BRJT) => Self::lower_jump_table(cursor, &mut scratch)?,
                    _ => cursor.keep_current(),
                }
                Ok(())
            })?;
        }

        Ok(())
    }

    fn lower_branch<S>(
        cursor: &mut crate::mir::BlockRewriteCursor<'_, S>,
        scratch: &mut ParallelCopyScratch,
        inst: &MachineInst,
    ) -> Result<()> {
        let branch = inst.as_branch()?;
        let args = Self::branch_args(cursor);
        Self::emit_branch_arg_copies(cursor, scratch, branch.target, &args)?;
        cursor.keep_current();
        Ok(())
    }

    fn lower_cond_branch<S>(
        cursor: &mut crate::mir::BlockRewriteCursor<'_, S>,
        scratch: &mut ParallelCopyScratch,
        inst: &MachineInst,
    ) -> Result<()> {
        let branch = inst.as_branch_cond()?;
        let (then_args, else_args) = Self::cond_branch_args(cursor);

        if then_args.is_empty() && else_args.is_empty() {
            cursor.keep_current();
            return Ok(());
        }

        let then_blk = Self::redirect_edge(
            cursor.mfunc_mut().as_untyped_mut(),
            scratch,
            branch.then_blk,
            &then_args,
        )?;
        let else_blk = Self::redirect_edge(
            cursor.mfunc_mut().as_untyped_mut(),
            scratch,
            branch.else_blk,
            &else_args,
        )?;

        cursor.replace_current(MachineInst::build_br_cond(branch.cond, then_blk, else_blk));
        Ok(())
    }

    fn lower_jump_table<S>(
        cursor: &mut crate::mir::BlockRewriteCursor<'_, S>,
        scratch: &mut ParallelCopyScratch,
    ) -> Result<()> {
        let info = Self::jump_table_info(cursor);

        if info.targets.iter().all(|target| target.args.is_empty()) {
            cursor.keep_current();
            return Ok(());
        }

        let mut new_targets = Vec::with_capacity(info.targets.len());
        for target in info.targets {
            let block = Self::redirect_edge(
                cursor.mfunc_mut().as_untyped_mut(),
                scratch,
                target.block,
                &target.args,
            )?;
            new_targets.push(BrTableTarget {
                block,
                args: SmallVec::new(),
            });
        }

        cursor.set_current_extra(InstExtra::BrTable(BrTableInfo {
            targets: new_targets,
        }));
        cursor.keep_current();
        Ok(())
    }

    fn branch_args<S>(cursor: &crate::mir::BlockRewriteCursor<'_, S>) -> SmallVec<[Reg; 2]> {
        match cursor.current_extra() {
            Some(InstExtra::Branch(info)) => info.args.clone(),
            None => SmallVec::new(),
            Some(_) => panic!("expected branch extra on G_BR instruction"),
        }
    }

    fn cond_branch_args<S>(
        cursor: &'_ crate::mir::BlockRewriteCursor<'_, S>,
    ) -> (SmallVec<[Reg; 2]>, SmallVec<[Reg; 2]>) {
        match cursor.current_extra() {
            Some(InstExtra::BranchCond(info)) => (info.then_args.clone(), info.else_args.clone()),
            None => (SmallVec::new(), SmallVec::new()),
            Some(_) => panic!("expected branch-cond extra on G_BRCOND instruction"),
        }
    }

    fn jump_table_info<S>(cursor: &crate::mir::BlockRewriteCursor<'_, S>) -> BrTableInfo {
        match cursor.current_extra() {
            Some(InstExtra::BrTable(info)) => info.clone(),
            Some(_) => panic!("expected br_table extra on G_BRJT instruction"),
            None => panic!("missing br_table extra on G_BRJT instruction"),
        }
    }

    fn emit_branch_arg_copies<S>(
        cursor: &mut crate::mir::BlockRewriteCursor<'_, S>,
        scratch: &mut ParallelCopyScratch,
        target: Block,
        args: &[Reg],
    ) -> Result<()> {
        if args.is_empty() {
            return Ok(());
        }

        let copies =
            Self::build_edge_copies(cursor.mfunc_mut().as_untyped_mut(), scratch, target, args)?;
        cursor.emit_before_many(copies.iter().cloned());
        cursor.clear_current_extra();
        Ok(())
    }

    fn redirect_edge(
        mfunc: &mut MachineFunction,
        scratch: &mut ParallelCopyScratch,
        target: Block,
        args: &[Reg],
    ) -> Result<Block> {
        if args.is_empty() {
            Ok(target)
        } else {
            Self::create_edge_move_block(mfunc, scratch, target, args)
        }
    }

    fn build_edge_copies<'a>(
        mfunc: &mut MachineFunction,
        scratch: &'a mut ParallelCopyScratch,
        target: Block,
        args: &[Reg],
    ) -> Result<&'a [MachineInst]> {
        let params: SmallVec<[Reg; 4]> = mfunc
            .block_params(target)
            .unwrap_or_else(|| panic!("branch target {:?} not found in MIR", target))
            .iter()
            .copied()
            .collect();

        if params.len() != args.len() {
            panic!(
                "branch target {:?} expects {} arguments, got {}",
                target,
                params.len(),
                args.len()
            );
        }

        Ok(Self::build_parallel_copies(
            scratch,
            mfunc,
            params.as_slice(),
            args,
        ))
    }

    fn build_parallel_copies<'a>(
        scratch: &'a mut ParallelCopyScratch,
        mfunc: &mut MachineFunction,
        dsts: &[Reg],
        srcs: &[Reg],
    ) -> &'a [MachineInst] {
        scratch.clear();
        scratch.pending.reserve(dsts.len());
        scratch.out.reserve(dsts.len());
        scratch.dest_counts.reserve(dsts.len());
        scratch.source_users.reserve(dsts.len());
        scratch.ready.reserve(dsts.len());

        scratch
            .pending
            .extend(dsts.iter().copied().zip(srcs.iter().copied()).map(Some));

        for (index, copy) in scratch.pending.iter().enumerate() {
            let (dst, src) = copy.as_ref().copied().expect("pending copy must exist");
            *scratch.dest_counts.entry(dst).or_insert(0) += 1;
            scratch.source_users.entry(src).or_default().push(index);
        }

        for (index, copy) in scratch.pending.iter().enumerate() {
            let (dst, src) = copy.as_ref().copied().expect("pending copy must exist");
            if dst == src || scratch.dest_counts.get(&src).copied().unwrap_or(0) == 0 {
                scratch.ready.push(index);
            }
        }

        let mut remaining = scratch.pending.len();

        while remaining > 0 {
            if let Some(index) = scratch.ready.pop() {
                let Some((dst, src)) = scratch.pending[index].take() else {
                    continue;
                };

                remaining -= 1;
                if dst != src {
                    scratch
                        .out
                        .push(MachineInst::build_copy(Writable(dst), src));
                }

                let became_free = if let Some(count) = scratch.dest_counts.get_mut(&dst) {
                    *count -= 1;
                    *count == 0
                } else {
                    false
                };

                if became_free {
                    if let Some(users) = scratch.source_users.remove(&dst) {
                        for user_index in users {
                            if scratch.pending[user_index].is_some() {
                                scratch.ready.push(user_index);
                            }
                        }
                    }
                }

                continue;
            }

            let cycle_index = scratch
                .pending
                .iter()
                .position(|copy| copy.is_some())
                .expect("pending copy list should contain an unfinished entry");
            let cycle_src = scratch.pending[cycle_index]
                .as_ref()
                .expect("pending copy must exist")
                .1;
            let temp = mfunc.alloc_vreg(Self::copy_temp_type(mfunc, cycle_src));
            let save = MachineInst::build_copy(Writable(temp), cycle_src);
            scratch.out.push(save);

            if let Some(users) = scratch.source_users.remove(&cycle_src) {
                for user_index in users {
                    let Some((_, src)) = scratch.pending[user_index].as_mut() else {
                        continue;
                    };

                    *src = temp;
                    scratch
                        .source_users
                        .entry(temp)
                        .or_default()
                        .push(user_index);
                    scratch.ready.push(user_index);
                }
            }
        }

        &scratch.out
    }

    fn copy_temp_type(mfunc: &MachineFunction, cycle_src: Reg) -> Type {
        debug_assert!(
            cycle_src.is_vreg(),
            "block param lowering expects vreg sources before ABI lowering"
        );
        mfunc.vreg_data(cycle_src).ty
    }

    fn create_edge_move_block(
        mfunc: &mut MachineFunction,
        scratch: &mut ParallelCopyScratch,
        target: Block,
        args: &[Reg],
    ) -> Result<Block> {
        let block = mfunc.create_synthetic_block();
        let block_idx = mfunc
            .find_block_index(block)
            .expect("newly created synthetic block must exist");
        let insts = Self::build_edge_copies(mfunc, scratch, target, args)?;
        let br = mfunc.alloc_inst(MachineInst::build_br(target));
        for inst in insts.iter().cloned() {
            let inst_id = mfunc.alloc_inst(inst);
            mfunc.append_inst_id_to_block(block_idx, inst_id);
        }
        mfunc.append_inst_id_to_block(block_idx, br);
        Ok(block)
    }
}

impl StageTransformPass<RawMir, LegalizedMir> for BlockParamLoweringPass {
    fn name(&self) -> &'static str {
        "edge-args-lowered"
    }

    fn run(
        &self,
        mut mfunc: MachineFunction<RawMir>,
        _ctx: &mut FunctionPassContext<'_, RawMir>,
    ) -> Result<(MachineFunction<LegalizedMir>, PassEffect)> {
        Self::run(&mut mfunc)?;
        Ok((
            mfunc.into_stage(),
            PassEffect::new(ChangeSet::CFG | ChangeSet::INST_OPERANDS),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::BlockParamLoweringPass;
    use crate::mir::{
        BrTableInfo, BrTableTarget, InstExtra, MachineBlock, MachineFunction, MachineInst,
    };
    use alloc::vec;
    use smallvec::smallvec;
    use veloc_ir::{Block, Type};

    fn make_function() -> MachineFunction {
        MachineFunction::new("test".into())
    }

    fn push_block(mfunc: &mut MachineFunction, block: Block, params: &[crate::mir::Reg]) {
        let mut mblock = MachineBlock::new(block);
        mblock.params = params.to_vec();
        mfunc.blocks.push(mblock);
    }

    #[test]
    fn branch_without_args_is_left_unchanged() {
        let mut mfunc = make_function();
        let entry = Block::from_u32(0);
        let target = Block::from_u32(1);
        push_block(&mut mfunc, entry, &[]);
        push_block(&mut mfunc, target, &[]);

        let br = mfunc.alloc_inst(MachineInst::build_br(target));
        mfunc.append_inst_id_to_block(0, br);

        BlockParamLoweringPass::run(&mut mfunc).unwrap();

        assert_eq!(mfunc.blocks[0].insts, vec![br]);
        assert!(mfunc.inst_extra(br).is_none());
    }

    #[test]
    fn branch_with_args_emits_parallel_copies() {
        let mut mfunc = make_function();
        let entry = Block::from_u32(0);
        let target = Block::from_u32(1);
        let src = mfunc.alloc_vreg(Type::I64);
        let param = mfunc.alloc_vreg(Type::I64);
        push_block(&mut mfunc, entry, &[]);
        push_block(&mut mfunc, target, &[param]);

        let br = mfunc.alloc_inst(MachineInst::build_br(target));
        mfunc.set_inst_extra(
            br,
            InstExtra::Branch(crate::mir::BranchInfo {
                args: smallvec![src],
            }),
        );
        mfunc.append_inst_id_to_block(0, br);

        BlockParamLoweringPass::run(&mut mfunc).unwrap();

        assert_eq!(mfunc.blocks[0].insts.len(), 2);
        let copy = &mfunc.dfg[mfunc.blocks[0].insts[0]];
        let decoded = copy.as_unary_reg().unwrap();
        assert_eq!(decoded.dst, param);
        assert_eq!(decoded.src, src);
        assert_eq!(mfunc.blocks[0].insts[1], br);
        assert!(mfunc.inst_extra(br).is_none());
    }

    #[test]
    fn conditional_branch_creates_synthetic_block_for_argumentful_edge() {
        let mut mfunc = make_function();
        let entry = Block::from_u32(0);
        let then_blk = Block::from_u32(1);
        let else_blk = Block::from_u32(2);
        let cond = mfunc.alloc_vreg(Type::BOOL);
        let arg = mfunc.alloc_vreg(Type::I64);
        let param = mfunc.alloc_vreg(Type::I64);
        push_block(&mut mfunc, entry, &[]);
        push_block(&mut mfunc, then_blk, &[param]);
        push_block(&mut mfunc, else_blk, &[]);

        let br = mfunc.alloc_inst(MachineInst::build_br_cond(cond, then_blk, else_blk));
        mfunc.set_inst_extra(
            br,
            InstExtra::BranchCond(crate::mir::BranchCondInfo {
                then_args: smallvec![arg],
                else_args: smallvec![],
            }),
        );
        mfunc.append_inst_id_to_block(0, br);

        BlockParamLoweringPass::run(&mut mfunc).unwrap();

        assert_eq!(mfunc.num_blocks(), 4);
        let lowered = mfunc.dfg[br].as_branch_cond().unwrap();
        assert_ne!(lowered.then_blk, then_blk);
        assert_eq!(lowered.else_blk, else_blk);

        let synthetic_idx = mfunc.find_block_index(lowered.then_blk).unwrap();
        assert_eq!(mfunc.blocks[synthetic_idx].insts.len(), 2);
        let copied = mfunc.dfg[mfunc.blocks[synthetic_idx].insts[0]]
            .as_unary_reg()
            .unwrap();
        assert_eq!(copied.dst, param);
        assert_eq!(copied.src, arg);
        let synthetic_br = mfunc.dfg[mfunc.blocks[synthetic_idx].insts[1]]
            .as_branch()
            .unwrap();
        assert_eq!(synthetic_br.target, then_blk);
    }

    #[test]
    fn jump_table_rewrites_only_targets_with_arguments() {
        let mut mfunc = make_function();
        let entry = Block::from_u32(0);
        let target0 = Block::from_u32(1);
        let target1 = Block::from_u32(2);
        let idx = mfunc.alloc_vreg(Type::I32);
        let arg = mfunc.alloc_vreg(Type::I64);
        let param = mfunc.alloc_vreg(Type::I64);
        push_block(&mut mfunc, entry, &[]);
        push_block(&mut mfunc, target0, &[param]);
        push_block(&mut mfunc, target1, &[]);

        let jt = mfunc.alloc_inst(MachineInst::build_br_jt(idx));
        mfunc.set_inst_extra(
            jt,
            InstExtra::BrTable(BrTableInfo {
                targets: vec![
                    BrTableTarget {
                        block: target0,
                        args: smallvec![arg],
                    },
                    BrTableTarget {
                        block: target1,
                        args: smallvec![],
                    },
                ],
            }),
        );
        mfunc.append_inst_id_to_block(0, jt);

        BlockParamLoweringPass::run(&mut mfunc).unwrap();

        let Some(InstExtra::BrTable(info)) = mfunc.inst_extra(jt).cloned() else {
            panic!("expected br_table extra");
        };
        assert_eq!(info.targets.len(), 2);
        assert_ne!(info.targets[0].block, target0);
        assert!(info.targets[0].args.is_empty());
        assert_eq!(info.targets[1].block, target1);
        assert!(info.targets[1].args.is_empty());
    }

    #[test]
    fn parallel_copy_cycle_uses_temporary_vreg() {
        let mut mfunc = make_function();
        let entry = Block::from_u32(0);
        let target = Block::from_u32(1);
        let a = mfunc.alloc_vreg(Type::I64);
        let b = mfunc.alloc_vreg(Type::I64);
        push_block(&mut mfunc, entry, &[]);
        push_block(&mut mfunc, target, &[a, b]);

        let br = mfunc.alloc_inst(MachineInst::build_br(target));
        mfunc.set_inst_extra(
            br,
            InstExtra::Branch(crate::mir::BranchInfo {
                args: smallvec![b, a],
            }),
        );
        mfunc.append_inst_id_to_block(0, br);

        BlockParamLoweringPass::run(&mut mfunc).unwrap();

        assert_eq!(mfunc.blocks[0].insts.len(), 4);
        let save = mfunc.dfg[mfunc.blocks[0].insts[0]].as_unary_reg().unwrap();
        assert!(save.dst.is_vreg());
        assert_eq!(save.src, b);
        let copy0 = mfunc.dfg[mfunc.blocks[0].insts[1]].as_unary_reg().unwrap();
        assert_eq!(copy0.dst, a);
        assert_eq!(copy0.src, save.dst);
        let copy1 = mfunc.dfg[mfunc.blocks[0].insts[2]].as_unary_reg().unwrap();
        assert_eq!(copy1.dst, b);
        assert_eq!(copy1.src, a);
    }
}
