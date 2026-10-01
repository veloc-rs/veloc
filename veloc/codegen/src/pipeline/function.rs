//! Complete function compilation, including ownership-changing transitions.
use super::{FunctionPassContext, FunctionStage, PassSequence, run_function_pass};
use crate::analysis::FunctionAnalysisCtx;
use crate::isel::InstructionSelectionPass;
use crate::passes::{FrameFinalizePass, LegalizePass, PostIselOptimizePass, RemoveUnreachablePass};
use crate::target::TargetMachine;
use crate::{CodegenOptions, Error, Result};
use veloc_lir::MachineFunction;
use veloc_profile::{Metric, Profile};

/// Built once per module. Owns pass ordering; each run owns its analysis cache.
pub struct FunctionPipeline<'a> {
    target: &'a dyn TargetMachine,
    options: &'a CodegenOptions,
    profile: &'a Profile,
    prepare: PassSequence,
    pre_isel: PassSequence,
    post_isel: PassSequence,
    post_regalloc: PassSequence,
}
impl<'a> FunctionPipeline<'a> {
    pub fn new(
        target: &'a dyn TargetMachine,
        options: &'a CodegenOptions,
        profile: &'a Profile,
    ) -> Self {
        let config = target.pass_config();
        Self {
            target,
            options,
            profile,
            prepare: PassSequence::from_passes(config.prepare_passes()),
            pre_isel: PassSequence::from_passes(config.pre_isel_passes()),
            post_isel: PassSequence::from_passes(config.post_isel_passes()),
            post_regalloc: PassSequence::from_passes(config.post_regalloc_passes()),
        }
    }
    pub fn run(
        &self,
        function: MachineFunction,
        signature: &veloc_mir::Signature,
        symbols: &mut veloc_lir::SymbolTable,
    ) -> Result<MachineFunction> {
        let scope = self
            .profile
            .entity_scope("function", 0, || function.name.clone());
        let result = self.run_impl(function, signature, symbols);
        scope.result(&result);
        result
    }

    fn run_impl(
        &self,
        mut mfunc: MachineFunction,
        func_sig: &veloc_mir::Signature,
        symbols: &mut veloc_lir::SymbolTable,
    ) -> Result<MachineFunction> {
        use crate::verify::{verify, verify_allocated};
        let mut function_analyses =
            FunctionAnalysisCtx::default().with_profile(self.profile.clone());
        self.profile
            .record_lazy(Metric::count("initial_insts"), || {
                mfunc
                    .blocks()
                    .map(|b| mfunc.block_insts(b).count() as u64)
                    .sum()
            });
        self.profile.count("vregs", mfunc.vregs().len() as u64);
        super::dump_after("translated", &mfunc, self.options);
        self.verify_function("translated", &mfunc, verify)?;
        let mut ctx = FunctionPassContext::new(
            self.target,
            func_sig,
            symbols,
            self.options,
            self.profile,
            &mut function_analyses,
        );

        run_function_pass(
            &crate::passes::lowering::AbiLoweringPass::new(),
            &mut mfunc,
            &mut ctx,
        )?;
        self.maybe_dump_mfunc("abi-lowered", &mfunc);
        self.profile
            .measure("prepare", 0, || self.prepare.run(&mut mfunc, &mut ctx))?;
        run_function_pass(
            &LegalizePass::new(self.target.legalizer()),
            &mut mfunc,
            &mut ctx,
        )?;
        self.maybe_dump_mfunc("legalized", &mfunc);
        ctx.profile
            .record_lazy(Metric::count("legalized_insts"), || {
                mfunc
                    .blocks()
                    .map(|b| mfunc.block_insts(b).count() as u64)
                    .sum()
            });
        self.profile
            .measure("pre_isel", 0, || self.pre_isel.run(&mut mfunc, &mut ctx))?;
        // Target preparation may change the CFG. Selection always receives
        // reachable blocks only, even when optional optimizations are disabled.
        run_function_pass(&RemoveUnreachablePass, &mut mfunc, &mut ctx)?;
        self.maybe_dump_mfunc("pre-isel", &mfunc);
        run_function_pass(
            &InstructionSelectionPass::new(self.target.selector()),
            &mut mfunc,
            &mut ctx,
        )?;
        self.maybe_dump_mfunc("selected", &mfunc);
        ctx.profile
            .record_lazy(Metric::count("selected_insts"), || {
                mfunc
                    .blocks()
                    .map(|b| mfunc.block_insts(b).count() as u64)
                    .sum()
            });

        self.profile
            .measure("post_isel", 0, || self.post_isel.run(&mut mfunc, &mut ctx))?;
        self.maybe_dump_mfunc("post-isel-target", &mfunc);
        run_function_pass(
            &PostIselOptimizePass::new(self.target.post_isel()),
            &mut mfunc,
            &mut ctx,
        )?;
        self.maybe_dump_mfunc("post-isel-optimized", &mfunc);
        run_function_pass(&crate::passes::schedule::SchedulePass, &mut mfunc, &mut ctx)?;
        self.maybe_dump_mfunc("scheduled", &mfunc);

        // Allocation owns its exact input until its plan is materialized.
        let mut mfunc = self.profile.measure("regalloc", 0, || {
            let allocation = crate::regalloc::RegisterAllocator::new(self.target)
                .allocate(mfunc, ctx.function_analyses)?;
            allocation.materialize(self.target)
        })?;
        ctx.function_analyses
            .apply(crate::analysis::ChangeSet::WHOLE_FUNCTION);
        ctx.stage = FunctionStage::Allocated;
        self.verify_function("regalloc", &mfunc, verify_allocated)?;

        self.profile.measure("post_regalloc", 0, || {
            self.post_regalloc.run(&mut mfunc, &mut ctx)
        })?;
        self.maybe_dump_mfunc("post-regalloc", &mfunc);
        run_function_pass(
            &FrameFinalizePass::new(self.target.frame_lowering()),
            &mut mfunc,
            &mut ctx,
        )?;
        self.maybe_dump_mfunc("frame-finalized", &mfunc);
        ctx.profile.record_lazy(Metric::count("final_insts"), || {
            mfunc
                .blocks()
                .map(|b| mfunc.block_insts(b).count() as u64)
                .sum()
        });
        ctx.profile
            .count("stack_slots", mfunc.stack_frame.slots().len() as u64);

        self.maybe_dump_mfunc("final", &mfunc);
        super::dump_after("final", &mfunc, self.options);
        Ok(mfunc)
    }
    fn verify_function(
        &self,
        name: &'static str,
        mfunc: &MachineFunction,
        verify: fn(&MachineFunction, &dyn crate::target::TargetInstructions) -> Result<()>,
    ) -> Result<()> {
        if self.options.verify {
            self.profile
                .measure("verify", 0, || {
                    self.profile.measure(name, 0, || verify(mfunc, self.target))
                })
                .map_err(|e| Error::codegen(std::format!("{name}: {e}")))?;
        }
        self.maybe_dump_mfunc(name, mfunc);
        Ok(())
    }
    fn maybe_dump_mfunc(&self, stage: &str, mfunc: &MachineFunction) {
        use std::env;

        let filter = if self.options.dump_lir {
            Some(std::string::String::from("*"))
        } else {
            env::var("VELOC_DUMP_LIR").ok()
        };
        let Some(filter) = filter else {
            return;
        };

        if filter != "*" && filter != mfunc.name {
            return;
        }

        std::eprintln!("===== LIR {}: {} =====", stage, mfunc.name);
        std::eprintln!("{}", mfunc.format_for_dump());
    }
}
