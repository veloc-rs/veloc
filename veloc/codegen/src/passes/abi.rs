use crate::error::Result;
use crate::pipeline::stages::{AbiLowered, LegalizedMir};
use crate::pipeline::{ChangeSet, FunctionPassContext, PassEffect, StageTransformPass};
use crate::target::arch::{CallLowering, TargetLowering};

/// 将 call/call_indirect/ret 降低为 ABI 相关指令序列。
struct CallAbiLowering<'a> {
    call_lowering: &'a dyn CallLowering,
    func_sig: &'a veloc_ir::Signature,
}

impl<'a> CallAbiLowering<'a> {
    fn new(call_lowering: &'a dyn CallLowering, func_sig: &'a veloc_ir::Signature) -> Self {
        Self {
            call_lowering,
            func_sig,
        }
    }

    fn run<S>(&self, mfunc: &mut crate::mir::MachineFunction<S>) -> Result<()> {
        let num_blocks = mfunc.num_blocks();

        for block_idx in 0..num_blocks {
            mfunc.rewrite_block(block_idx, |cursor| {
                let inst = cursor.current_inst_clone();
                match inst.opcode {
                    crate::mir::MachineOpcode::Generic(crate::mir::GenericOpcode::G_CALL)
                    | crate::mir::MachineOpcode::Generic(crate::mir::GenericOpcode::G_CALLIND) => {
                        self.call_lowering.lower_call(cursor.as_untyped_mut())?;
                    }
                    crate::mir::MachineOpcode::Generic(crate::mir::GenericOpcode::G_RET) => {
                        self.call_lowering
                            .lower_return(cursor.as_untyped_mut(), self.func_sig)?;
                    }
                    _ => cursor.keep_current(),
                }
                Ok(())
            })?;
        }

        Ok(())
    }
}

pub struct AbiLoweringPass<'a> {
    lowering: &'a dyn TargetLowering,
}

impl<'a> AbiLoweringPass<'a> {
    pub fn new(lowering: &'a dyn TargetLowering) -> Self {
        Self { lowering }
    }
}

impl<'a> StageTransformPass<LegalizedMir, AbiLowered> for AbiLoweringPass<'a> {
    fn name(&self) -> &'static str {
        "abi-lowered"
    }

    fn run(
        &self,
        mut mfunc: crate::mir::MachineFunction<LegalizedMir>,
        ctx: &mut FunctionPassContext<'_, LegalizedMir>,
    ) -> Result<(crate::mir::MachineFunction<AbiLowered>, PassEffect)> {
        let call_lowering = self.lowering.call_lowering();
        call_lowering.lower_formal_arguments(mfunc.as_untyped_mut(), ctx.func_sig)?;
        CallAbiLowering::new(call_lowering, ctx.func_sig).run(&mut mfunc)?;
        Ok((
            mfunc.into_stage(),
            PassEffect::new(ChangeSet::INST_SEMANTICS | ChangeSet::PHYSICAL_REGS),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::CallAbiLowering;
    use crate::mir::{MachineBlock, MachineFunction, MachineInst, SymbolId, Writable};
    use crate::target::arch::CallLowering;
    use alloc::vec;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use smallvec::smallvec;
    use veloc_ir::{CallConv, Signature, Type};

    #[derive(Default)]
    struct CountingCallLowering {
        call_count: AtomicUsize,
        ret_count: AtomicUsize,
    }

    impl CallLowering for CountingCallLowering {
        fn lower_formal_arguments(
            &self,
            _mfunc: &mut MachineFunction,
            _sig: &veloc_ir::Signature,
        ) -> Result<(), crate::error::Error> {
            Ok(())
        }

        fn lower_call(
            &self,
            cursor: &mut crate::mir::BlockRewriteCursor<'_, crate::pipeline::stages::Untyped>,
        ) -> Result<(), crate::error::Error> {
            self.call_count.fetch_add(1, Ordering::Relaxed);
            cursor.keep_current();
            Ok(())
        }

        fn lower_return(
            &self,
            cursor: &mut crate::mir::BlockRewriteCursor<'_, crate::pipeline::stages::Untyped>,
            _sig: &veloc_ir::Signature,
        ) -> Result<(), crate::error::Error> {
            self.ret_count.fetch_add(1, Ordering::Relaxed);
            cursor.keep_current();
            Ok(())
        }
    }

    #[test]
    fn lowers_only_calls_and_returns() {
        let lowering = CountingCallLowering::default();
        let sig = Signature::new(vec![], vec![], CallConv::SystemV);
        let mut mfunc = MachineFunction::<crate::pipeline::stages::Untyped>::new("test".into());
        mfunc
            .blocks
            .push(MachineBlock::new(veloc_ir::Block::from_u32(0)));

        let dst = mfunc.alloc_vreg(Type::I64);
        let arg = mfunc.alloc_vreg(Type::I64);
        let callee = SymbolId::from_u32(0);
        let direct = mfunc.alloc_inst(MachineInst::build_call([Writable(dst)], callee, [arg]));
        let indirect = mfunc.alloc_inst(MachineInst::build_call_indirect(
            [Writable(dst)],
            arg,
            [arg],
        ));
        let copy = mfunc.alloc_inst(MachineInst::build_copy(Writable(dst), arg));
        let ret = mfunc.alloc_inst(MachineInst::build_ret(smallvec![arg]));
        mfunc.blocks[0].insts.extend([direct, indirect, copy, ret]);

        CallAbiLowering::new(&lowering, &sig)
            .run(&mut mfunc)
            .unwrap();

        assert_eq!(lowering.call_count.load(Ordering::Relaxed), 2);
        assert_eq!(lowering.ret_count.load(Ordering::Relaxed), 1);
        assert_eq!(mfunc.blocks[0].insts, vec![direct, indirect, copy, ret]);
    }
}
