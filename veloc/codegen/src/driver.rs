//! Code generation driver.
//!
//! 提供从 SSA IR 到机器码/目标文件的编译驱动。

use crate::error::{Error, Result};
use crate::object::ObjectFileBuilder;
use crate::passes::{
    BlockParamLoweringPass, FrameFinalizePass, InstructionSelectionPass, LegalizePass,
    PostIselOptimizePass, PreIselPass, RegisterAllocationPass,
};
use crate::pipeline::{
    CompiledFunction, CompiledModule, FunctionAnalysisCtx, FunctionPassContext, ModuleAnalysisCtx,
    ModulePassContext, ModulePassPipeline, StagePassPipeline, StageTransformPass,
};
use crate::target::arch::TargetMachine;
use crate::translate::IRTranslator;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use veloc_lir::stages::{
    LegalizedLir, PostIselOptimized, PreIselPrepared, PrologueEpilogueInserted, RawLir,
    RegAllocated, SelectedLir,
};
use veloc_lir::{MachineFunction, MachineModule};
use veloc_mir::{FuncId, Function, Module};

/// 代码生成统计信息
#[derive(Debug, Clone, Default)]
pub struct CodegenStats {
    /// 原始指令数
    pub initial_inst_count: usize,
    /// 合法化后指令数
    pub legalized_inst_count: usize,
    /// 指令选择后指令数
    pub selected_inst_count: usize,
    /// 指令融合后指令数
    pub combined_inst_count: usize,
    /// 寄存器分配后指令数
    pub final_inst_count: usize,
    /// 虚拟寄存器数量
    pub vreg_count: usize,
    /// 物理寄存器使用量
    pub preg_count: usize,
    /// 栈槽数量（溢出的虚拟寄存器）
    pub stack_slot_count: usize,
    /// 栈帧大小（字节）
    pub stack_frame_size: u32,
}

/// 代码生成选项
#[derive(Debug, Clone)]
pub struct CodegenOptions {
    /// 是否启用优化
    pub optimize: bool,
    /// 是否打印中间结果（调试用）
    pub dump_lir: bool,
    /// 目标优化级别
    pub opt_level: u8,
}

impl Default for CodegenOptions {
    fn default() -> Self {
        Self {
            optimize: true,
            dump_lir: false,
            opt_level: 2,
        }
    }
}

/// 代码生成驱动
///
/// 负责组织 typed LIR pipeline、target 相关扩展和最终发射。
///
/// # 示例
///
/// ```ignore
/// use veloc_codegen::{CodegenPipeline, TargetConfig, create_target_machine};
///
/// let config = TargetConfig::default();
/// let target = create_target_machine(config).unwrap();
/// let pipeline = CodegenPipeline::new(&*target);
///
/// let object = pipeline.compile_object(&module).unwrap();
/// ```
pub struct CodegenPipeline<'a> {
    target: &'a dyn TargetMachine,
    options: CodegenOptions,
}

impl<'a> CodegenPipeline<'a> {
    #[cfg(feature = "std")]
    fn maybe_dump_mfunc<S>(&self, stage: &str, mfunc: &MachineFunction<S>) {
        use std::env;

        let filter = if self.options.dump_lir {
            Some(alloc::string::String::from("*"))
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

    #[cfg(not(feature = "std"))]
    fn maybe_dump_mfunc<S>(&self, _stage: &str, _mfunc: &MachineFunction<S>) {}

    /// 创建新的代码生成驱动。
    pub fn new(target: &'a dyn TargetMachine) -> Self {
        Self {
            target,
            options: CodegenOptions::default(),
        }
    }

    /// 创建带显式选项的代码生成驱动。
    pub fn with_options(target: &'a dyn TargetMachine, options: CodegenOptions) -> Self {
        Self { target, options }
    }

    /// 编译整个模块并生成单个 relocatable object 文件。
    pub fn compile_object(&self, module: &veloc_mir::Module) -> Result<Vec<u8>> {
        let mut object = ObjectFileBuilder::new(self.target)?;
        let mut stats = CodegenStats::default();
        let mut module_analyses = ModuleAnalysisCtx::default();
        let compiled = self.compile_module_artifact(module, &mut stats, &mut module_analyses)?;

        for compiled_func in &compiled.functions {
            let func = module.get_function(compiled_func.func_id);
            if let Some(emitted) = &compiled_func.emitted {
                object.add_defined_function(func, emitted, &compiled.symbols)?;
            }
        }

        for (_, func) in &module.functions {
            if func.linkage == veloc_mir::Linkage::Import {
                object.add_undefined_function(func);
            }
        }

        object.finish()
    }

    /// 编译模块中的所有已定义函数并返回裸机器码。
    pub fn compile_functions(
        &self,
        module: &veloc_mir::Module,
    ) -> Result<BTreeMap<veloc_mir::FuncId, Vec<u8>>> {
        let mut stats = CodegenStats::default();
        let mut module_analyses = ModuleAnalysisCtx::default();
        let CompiledModule {
            symbols, functions, ..
        } = self.compile_module_artifact(module, &mut stats, &mut module_analyses)?;
        let mut results = BTreeMap::new();

        for compiled_func in functions {
            let emitted = compiled_func
                .emitted
                .ok_or_else(|| Error::missing_emitted_code(compiled_func.name.clone()))?;
            if let Some(reloc) = emitted.relocations.first() {
                let symbol = symbols.get(reloc.symbol).name.clone();
                return Err(Error::unexpected_relocation(symbol));
            }
            results.insert(compiled_func.func_id, emitted.data);
        }

        Ok(results)
    }

    fn compile_module_artifact(
        &self,
        module: &Module,
        stats: &mut CodegenStats,
        module_analyses: &mut ModuleAnalysisCtx,
    ) -> Result<CompiledModule> {
        let mmodule = self.translate_module(module)?;
        let mut compiled_functions = Vec::new();

        for (func_id, func) in &module.functions {
            if func.is_defined() {
                compiled_functions.push(self.compile_defined_function(
                    func_id,
                    module,
                    &mmodule,
                    func,
                    stats,
                    module_analyses,
                )?);
            }
        }

        let mut compiled = CompiledModule::new(
            mmodule.name.clone(),
            mmodule.symbols.clone(),
            compiled_functions,
        );
        self.run_module_pre_emit_passes(&mut compiled, stats, module_analyses)?;
        self.emit_compiled_functions(&mut compiled, stats)?;
        self.run_module_post_emit_passes(&mut compiled, stats, module_analyses)?;
        Ok(compiled)
    }

    fn translate_module(&self, module: &Module) -> Result<MachineModule> {
        IRTranslator::new(module).translate_module()
    }

    fn compile_defined_function(
        &self,
        func_id: FuncId,
        module: &Module,
        mmodule: &MachineModule,
        func: &Function,
        stats: &mut CodegenStats,
        module_analyses: &mut ModuleAnalysisCtx,
    ) -> Result<CompiledFunction> {
        let mfunc_id = mmodule
            .find_function_by_name(&func.name)
            .ok_or_else(|| Error::translated_function_not_found(func.name.clone()))?;
        let mfunc = mmodule.functions[mfunc_id].clone();
        let sig = module.get_signature(func.signature);
        let mut function_analyses = FunctionAnalysisCtx::default();

        stats.initial_inst_count = mfunc.blocks.iter().map(|b| b.insts.len()).sum();
        stats.vreg_count = mfunc.vregs.len();
        self.maybe_dump_mfunc("translated", &mfunc);

        let final_mfunc =
            self.run_function_pipeline(mfunc, sig, stats, &mut function_analyses, module_analyses)?;
        self.maybe_dump_mfunc("final", &final_mfunc);

        Ok(CompiledFunction {
            func_id,
            name: func.name.clone(),
            machine_function: final_mfunc,
            emitted: None,
        })
    }

    fn run_function_pipeline(
        &self,
        mfunc: MachineFunction<RawLir>,
        func_sig: &veloc_mir::Signature,
        stats: &mut CodegenStats,
        function_analyses: &mut FunctionAnalysisCtx,
        module_analyses: &mut ModuleAnalysisCtx,
    ) -> Result<MachineFunction<PrologueEpilogueInserted>> {
        let legalizer = self.target.target_legalizer();
        let selector = self.target.target_selector();
        let operand_lowering = self.target.target_operand_lowering();
        let target_post_isel = self.target.target_post_isel();
        let frame_lowering = self.target.target_frame_lowering();
        let pass_config = self.target.target_pass_config();

        let mut ctx = FunctionPassContext::<RawLir>::new(
            self.target,
            func_sig,
            &self.options,
            stats,
            function_analyses,
            module_analyses,
        );
        let mfunc = self.apply_stage_transform(&BlockParamLoweringPass, mfunc, &mut ctx)?;

        let mut ctx = ctx.into_stage::<LegalizedLir>();
        let mfunc = self.apply_stage_transform(
            &LegalizePass::new(legalizer, pass_config),
            mfunc,
            &mut ctx,
        )?;
        let mut ctx = ctx.into_stage::<PreIselPrepared>();
        let mfunc = self.apply_stage_transform(
            &PreIselPass::new(operand_lowering, pass_config),
            mfunc,
            &mut ctx,
        )?;

        let mfunc =
            self.apply_stage_transform(&InstructionSelectionPass::new(selector), mfunc, &mut ctx)?;
        let mut post_isel_pipeline = StagePassPipeline::<SelectedLir>::new();
        for pass in pass_config.post_isel_passes() {
            post_isel_pipeline.add_boxed_pass(pass);
        }
        let mut ctx = ctx.into_stage::<SelectedLir>();
        let mut mfunc = mfunc;
        self.run_stage_pipeline(
            "post-isel-target",
            &post_isel_pipeline,
            &mut mfunc,
            &mut ctx,
        )?;

        let mfunc = self.apply_stage_transform(
            &PostIselOptimizePass::new(target_post_isel, operand_lowering),
            mfunc,
            &mut ctx,
        )?;
        let mut ctx = ctx.into_stage::<PostIselOptimized>();
        let mfunc =
            self.apply_stage_transform(&RegisterAllocationPass::new(self.target), mfunc, &mut ctx)?;

        let mut post_regalloc = StagePassPipeline::<RegAllocated>::new();
        for pass in pass_config.post_regalloc_passes() {
            post_regalloc.add_boxed_pass(pass);
        }
        let mut ctx = ctx.into_stage::<RegAllocated>();
        let mut mfunc = mfunc;
        self.run_stage_pipeline("post-regalloc", &post_regalloc, &mut mfunc, &mut ctx)?;

        self.apply_stage_transform(&FrameFinalizePass::new(frame_lowering), mfunc, &mut ctx)
    }

    fn run_module_pre_emit_passes(
        &self,
        compiled: &mut CompiledModule,
        stats: &mut CodegenStats,
        module_analyses: &mut ModuleAnalysisCtx,
    ) -> Result<()> {
        let mut pipeline = ModulePassPipeline::new();
        for pass in self.target.target_pass_config().pre_emit_module_passes() {
            pipeline.add_boxed_pass(pass);
        }
        let mut ctx = ModulePassContext::new(self.target, &self.options, stats, module_analyses);
        let _ = pipeline.run(compiled, &mut ctx)?;
        Ok(())
    }

    fn run_module_post_emit_passes(
        &self,
        compiled: &mut CompiledModule,
        stats: &mut CodegenStats,
        module_analyses: &mut ModuleAnalysisCtx,
    ) -> Result<()> {
        let mut pipeline = ModulePassPipeline::new();
        for pass in self.target.target_pass_config().post_emit_module_passes() {
            pipeline.add_boxed_pass(pass);
        }
        let mut ctx = ModulePassContext::new(self.target, &self.options, stats, module_analyses);
        let _ = pipeline.run(compiled, &mut ctx)?;
        Ok(())
    }

    fn emit_compiled_functions(
        &self,
        compiled: &mut CompiledModule,
        stats: &mut CodegenStats,
    ) -> Result<()> {
        for func in &mut compiled.functions {
            if func.emitted.is_none() {
                func.emitted =
                    Some(self.emit_function_with_relocations(&func.machine_function, stats)?);
            }
        }
        Ok(())
    }

    fn apply_stage_transform<In, Out, P>(
        &self,
        pass: &P,
        mfunc: MachineFunction<In>,
        ctx: &mut FunctionPassContext<'_, In>,
    ) -> Result<MachineFunction<Out>>
    where
        P: StageTransformPass<In, Out>,
    {
        let stage_name = pass.name();
        let (mfunc, effect) = pass.run(mfunc, ctx)?;
        ctx.function_analyses.apply(effect.change_set);
        self.maybe_dump_mfunc(stage_name, &mfunc);
        Ok(mfunc)
    }

    fn run_stage_pipeline<S>(
        &self,
        stage_name: &str,
        pipeline: &StagePassPipeline<S>,
        mfunc: &mut MachineFunction<S>,
        ctx: &mut FunctionPassContext<'_, S>,
    ) -> Result<()> {
        let _ = pipeline.run(mfunc, ctx)?;
        self.maybe_dump_mfunc(stage_name, mfunc);
        Ok(())
    }

    fn emit_function_with_relocations(
        &self,
        mfunc: &MachineFunction<PrologueEpilogueInserted>,
        stats: &mut CodegenStats,
    ) -> Result<crate::EmittedCode> {
        let emitter = self.target.target_emitter();
        let mut output = crate::Emitter::new();

        for block in &mfunc.blocks {
            emitter.begin_block(&mut output, block, mfunc)?;
            for &inst_id in &block.insts {
                let inst = &mfunc.dfg[inst_id];
                emitter.emit_instruction(&mut output, inst, mfunc)?;
            }
        }

        emitter.finish_function(&mut output, mfunc)?;
        stats.stack_frame_size = mfunc.stack_frame.total_size;
        Ok(output.finish())
    }

    /// 获取编译选项的可变引用。
    pub fn options_mut(&mut self) -> &mut CodegenOptions {
        &mut self.options
    }

    /// 获取目标机器。
    pub fn target(&self) -> &dyn TargetMachine {
        self.target
    }
}
