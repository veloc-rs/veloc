//! Code generation driver.
//!
//! 提供从 SSA IR 到机器码/目标文件的编译驱动。

use crate::analysis::{FunctionAnalysisCtx, ModuleAnalysisCtx};
use crate::error::{Error, Result};
use crate::isel::InstructionSelectionPass;
use crate::object::ObjectFileBuilder;
use crate::passes::{FrameFinalizePass, LegalizePass, PostIselOptimizePass};
use crate::pipeline::{
    CompiledFunction, CompiledModule, FunctionPassContext, FunctionPassPipeline, ModulePassContext,
    ModulePassPipeline, run_function_pass,
};
use crate::target::TargetMachine;
use crate::translate::IRTranslator;
use std::collections::BTreeMap;
use std::vec::Vec;
use veloc_lir::{MachineFunction, MachineModule};
use veloc_mir::{FuncId, FunctionRef, Module};

/// 代码生成统计信息
#[derive(Debug, Clone, Default)]
pub struct CodegenStats {
    /// Number of local regions whose instruction order changed.
    pub scheduled_regions: usize,
    /// 原始指令数
    pub initial_inst_count: usize,
    /// 合法化后指令数
    pub legalized_inst_count: usize,
    /// 指令选择后指令数
    pub selected_inst_count: usize,
    /// Instruction count after frame finalization.
    pub final_inst_count: usize,
    /// Virtual registers immediately after MIR translation.
    pub vreg_count: usize,
    /// All stack objects, including locals, ABI areas and spills.
    pub stack_slot_count: usize,
    /// 栈帧大小（字节）
    pub stack_frame_size: u64,
    /// Total emitted code and embedded data bytes, excluding object metadata.
    pub code_bytes: usize,
    /// Per-function pass time, aggregated by name.
    pub pass_times: BTreeMap<std::string::String, core::time::Duration>,
}

/// 代码生成选项
#[derive(Debug, Clone)]
pub struct CodegenOptions {
    /// Validate SSA, selected and allocated invariants at pipeline boundaries.
    /// Enabled by default in debug builds; construction itself remains unchecked.
    pub verify: bool,
    /// 是否启用优化
    pub optimize: bool,
    /// Pass names whose output should be printed; `*` selects every pass.
    pub dump_after: Vec<std::string::String>,
    /// Restrict pass dumps to one function (None selects all functions).
    pub dump_function: Option<std::string::String>,
    /// 是否打印中间结果（调试用）
    pub dump_lir: bool,
    /// Collect aggregate pipeline counters. Disabled by default so production
    /// JIT compilation does not repeatedly scan every function for diagnostics.
    pub collect_stats: bool,
}

impl Default for CodegenOptions {
    fn default() -> Self {
        Self {
            verify: cfg!(debug_assertions),
            optimize: true,
            dump_after: Vec::new(),
            dump_function: None,
            dump_lir: false,
            collect_stats: false,
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

struct TargetFunctionPipelines {
    prepare: FunctionPassPipeline,
    pre_isel: FunctionPassPipeline,
    post_isel: FunctionPassPipeline,
    post_regalloc: FunctionPassPipeline,
}

impl TargetFunctionPipelines {
    fn new(config: &dyn crate::target::TargetPassConfig) -> Self {
        Self {
            prepare: FunctionPassPipeline::from_passes(config.prepare_passes()),
            pre_isel: FunctionPassPipeline::from_passes(config.pre_isel_passes()),
            post_isel: FunctionPassPipeline::from_passes(config.post_isel_passes()),
            post_regalloc: FunctionPassPipeline::from_passes(config.post_regalloc_passes()),
        }
    }
}

impl<'a> CodegenPipeline<'a> {
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
        self.compile_object_impl(module, &mut CodegenStats::default())
    }

    /// Compile and return diagnostics; ordinary compilation keeps them disabled by default.
    pub fn compile_object_with_stats(&self, module: &Module) -> Result<(Vec<u8>, CodegenStats)> {
        let mut options = self.options.clone();
        options.collect_stats = true;
        let pipeline = Self::with_options(self.target, options);
        let mut stats = CodegenStats::default();
        let object = pipeline.compile_object_impl(module, &mut stats)?;
        Ok((object, stats))
    }

    fn compile_object_impl(&self, module: &Module, stats: &mut CodegenStats) -> Result<Vec<u8>> {
        let mut object = ObjectFileBuilder::new(self.target)?;
        let mut module_analyses = ModuleAnalysisCtx::default();
        let compiled = self.compile_module_artifact(module, stats, &mut module_analyses)?;

        for compiled_func in &compiled.functions {
            let func = module.function(compiled_func.func_id);
            if let Some(emitted) = &compiled_func.emitted {
                object.add_defined_function(&func, emitted, &compiled.symbols)?;
            }
        }

        for (_, func) in module.functions() {
            if func.decl.linkage == veloc_mir::Linkage::Import {
                object.add_undefined_function(&func);
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
        let veloc_lir::MachineModule {
            name,
            symbols,
            functions,
        } = mmodule;
        let mut compiled_functions = Vec::new();
        let function_pipelines = TargetFunctionPipelines::new(self.target.pass_config());

        for ((func_id, func), (_, mfunc)) in module
            .functions()
            .filter(|(_, f)| f.body.is_some())
            .zip(functions.into_iter())
        {
            debug_assert_eq!(func.decl.name, mfunc.name);
            compiled_functions.push(self.compile_defined_function(
                func_id,
                &func,
                &module.signatures()[func.decl.signature],
                mfunc,
                stats,
                module_analyses,
                &function_pipelines,
            )?);
        }

        let mut compiled = CompiledModule::new(name, symbols, compiled_functions);
        self.run_module_pre_emit_passes(&mut compiled, stats, module_analyses)?;
        self.emit_compiled_functions(&mut compiled, stats)?;
        self.run_module_post_emit_passes(&mut compiled, stats, module_analyses)?;
        Ok(compiled)
    }

    fn translate_module(&self, module: &Module) -> Result<MachineModule> {
        IRTranslator::new(module, self.target.desc().data_layout).translate_module()
    }

    fn compile_defined_function(
        &self,
        func_id: FuncId,
        func: &FunctionRef,
        sig: &veloc_mir::Signature,
        mfunc: MachineFunction,
        stats: &mut CodegenStats,
        module_analyses: &mut ModuleAnalysisCtx,
        target_pipelines: &TargetFunctionPipelines,
    ) -> Result<CompiledFunction> {
        let mut function_analyses = FunctionAnalysisCtx::default();

        if self.options.collect_stats {
            stats.initial_inst_count += mfunc
                .blocks()
                .map(|b| mfunc.block_insts(b).count())
                .sum::<usize>();
            stats.vreg_count += mfunc.vregs().len();
        }
        self.maybe_dump_mfunc("translated", &mfunc);
        crate::pipeline::dump_after("translated", &mfunc, &self.options);

        let final_mfunc = self.run_function_pipeline(
            mfunc,
            sig,
            stats,
            &mut function_analyses,
            module_analyses,
            target_pipelines,
        )?;
        self.maybe_dump_mfunc("final", &final_mfunc);
        crate::pipeline::dump_after("final", &final_mfunc, &self.options);

        Ok(CompiledFunction {
            func_id,
            name: func.decl.name.clone(),
            machine_function: final_mfunc,
            emitted: None,
        })
    }

    fn run_function_pipeline(
        &self,
        mut mfunc: MachineFunction,
        func_sig: &veloc_mir::Signature,
        stats: &mut CodegenStats,
        function_analyses: &mut FunctionAnalysisCtx,
        module_analyses: &mut ModuleAnalysisCtx,
        target_pipelines: &TargetFunctionPipelines,
    ) -> Result<MachineFunction> {
        use crate::verify::{verify, verify_allocated, verify_selected};
        self.verify_function("translated", &mfunc, verify)?;
        let mut ctx = FunctionPassContext::new(
            self.target,
            func_sig,
            &self.options,
            stats,
            function_analyses,
            module_analyses,
        );

        run_function_pass(
            &crate::passes::lowering::AbiLoweringPass::new(),
            &mut mfunc,
            &mut ctx,
        )?;
        self.verify_function("abi-lowered", &mfunc, verify)?;
        target_pipelines.prepare.run(&mut mfunc, &mut ctx)?;
        run_function_pass(
            &LegalizePass::new(self.target.legalizer()),
            &mut mfunc,
            &mut ctx,
        )?;
        self.verify_function("legalized", &mfunc, verify)?;
        if ctx.options.collect_stats {
            ctx.stats.legalized_inst_count += mfunc
                .blocks()
                .map(|b| mfunc.block_insts(b).count())
                .sum::<usize>();
        }
        target_pipelines.pre_isel.run(&mut mfunc, &mut ctx)?;
        if self.options.verify {
            crate::passes::lowering::Legalizer::new(self.target.legalizer()).verify(&mfunc)?;
        }
        run_function_pass(
            &crate::passes::constraints::PreSelectOperandConstraintPass::new(
                self.target.operand_lowering(),
            ),
            &mut mfunc,
            &mut ctx,
        )?;
        self.verify_function("pre-isel", &mfunc, verify)?;
        run_function_pass(
            &InstructionSelectionPass::new(self.target.selector()),
            &mut mfunc,
            &mut ctx,
        )?;
        self.verify_function("selected", &mfunc, verify_selected)?;
        if ctx.options.collect_stats {
            ctx.stats.selected_inst_count += mfunc
                .blocks()
                .map(|b| mfunc.block_insts(b).count())
                .sum::<usize>();
        }

        target_pipelines.post_isel.run(&mut mfunc, &mut ctx)?;
        self.verify_function("post-isel-target", &mfunc, verify_selected)?;
        run_function_pass(
            &PostIselOptimizePass::new(self.target.post_isel()),
            &mut mfunc,
            &mut ctx,
        )?;
        self.verify_function("post-isel-optimized", &mfunc, verify_selected)?;
        run_function_pass(
            &crate::passes::constraints::PostSelectOperandConstraintPass::new(
                self.target.operand_lowering(),
            ),
            &mut mfunc,
            &mut ctx,
        )?;
        self.verify_function("operand-constraints", &mfunc, verify_selected)?;
        run_function_pass(&crate::passes::schedule::SchedulePass, &mut mfunc, &mut ctx)?;
        self.verify_function("scheduled", &mfunc, verify_selected)?;

        // Allocation owns its exact input until its plan is materialized.
        let start = self.options.collect_stats.then(std::time::Instant::now);
        let allocation = crate::regalloc::RegisterAllocator::new(self.target)
            .allocate(mfunc, ctx.function_analyses)?;
        let mut mfunc = allocation.materialize(self.target)?;
        if let Some(start) = start {
            *ctx.stats.pass_times.entry("regalloc".into()).or_default() += start.elapsed();
        }
        use crate::analysis::ChangeSet;
        ctx.function_analyses.apply(
            ChangeSet::REGALLOC
                | ChangeSet::PHYSICAL_REGS
                | ChangeSet::INST_OPERANDS
                | ChangeSet::INST_SEMANTICS
                | ChangeSet::CFG
                | ChangeSet::STACK_FRAME,
        );
        self.verify_function("regalloc", &mfunc, verify_allocated)?;

        target_pipelines.post_regalloc.run(&mut mfunc, &mut ctx)?;
        self.verify_function("post-regalloc", &mfunc, verify_allocated)?;
        run_function_pass(
            &FrameFinalizePass::new(self.target.frame_lowering()),
            &mut mfunc,
            &mut ctx,
        )?;
        self.verify_function("frame-finalized", &mfunc, verify_allocated)?;
        if ctx.options.collect_stats {
            ctx.stats.final_inst_count += mfunc
                .blocks()
                .map(|b| mfunc.block_insts(b).count())
                .sum::<usize>();
            ctx.stats.stack_slot_count += mfunc.stack_frame.slots().len();
        }

        Ok(mfunc)
    }

    fn run_module_pre_emit_passes(
        &self,
        compiled: &mut CompiledModule,
        stats: &mut CodegenStats,
        module_analyses: &mut ModuleAnalysisCtx,
    ) -> Result<()> {
        let mut pipeline = ModulePassPipeline::new();
        for pass in self.target.pass_config().pre_emit_module_passes() {
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
        for pass in self.target.pass_config().post_emit_module_passes() {
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

    fn verify_function(
        &self,
        name: &str,
        mfunc: &MachineFunction,
        verify: fn(&MachineFunction, &dyn crate::target::TargetInstructions) -> Result<()>,
    ) -> Result<()> {
        if self.options.verify {
            verify(mfunc, self.target).map_err(|e| Error::codegen(std::format!("{name}: {e}")))?;
        }
        self.maybe_dump_mfunc(name, mfunc);
        Ok(())
    }

    fn emit_function_with_relocations(
        &self,
        mfunc: &MachineFunction,
        stats: &mut CodegenStats,
    ) -> Result<crate::EmittedCode> {
        if Some(mfunc.entry_block()) != mfunc.blocks().next() {
            return Err(Error::codegen("function entry must be first at emission"));
        }
        let emitter = self.target.emitter();
        let start = self.options.collect_stats.then(std::time::Instant::now);
        let mut output = crate::Emitter::new();

        for block in mfunc.blocks() {
            emitter.begin_block(&mut output, block, mfunc)?;
            for inst_id in mfunc.block_insts(block) {
                let inst = &mfunc.inst(inst_id);
                if inst.is_generic() || inst.defs().chain(inst.uses()).any(|r| r.is_vreg()) {
                    return Err(Error::codegen(std::format!(
                        "unlowered instruction reached emission in {}: {:?}",
                        mfunc.name,
                        inst
                    )));
                }
                emitter.emit_instruction(&mut output, inst, mfunc)?;
            }
        }

        emitter.finish_function(&mut output, mfunc)?;
        if self.options.collect_stats {
            stats.stack_frame_size += mfunc
                .stack_frame
                .layout()
                .expect("finalized frame")
                .total_size as u64;
        }
        let emitted = output.finish()?;
        if let Some(start) = start {
            *stats.pass_times.entry("emit".into()).or_default() += start.elapsed();
        }
        if self.options.collect_stats {
            stats.code_bytes += emitted.data.len();
        }
        Ok(emitted)
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

#[cfg(test)]
mod memory_tests {
    use super::*;
    use std::string::ToString;
    use veloc_lir::MemoryKind;

    #[test]
    fn selection_rejects_a_changed_access_width_or_direction() {
        let module = veloc_mir::ModuleParser::new()
            .parse(
                r#"
local function access(ptr) -> i64
block0(v0: ptr):
  v1: i64 = load.volatile v0, offset=0
  return v1
"#,
            )
            .unwrap();
        module.validate().unwrap();
        let target = crate::create_target_machine(crate::TargetConfig::default()).unwrap();
        let pipeline = CodegenPipeline::new(&*target);
        let translated = pipeline.translate_module(&module).unwrap();
        let func = module.functions().next().unwrap().1;
        let sig = &module.signatures()[func.decl.signature];
        let target_pipelines = TargetFunctionPipelines::new(target.pass_config());
        for wrong_direction in [false, true] {
            let mut f = translated.functions.iter().next().unwrap().1.clone();
            let id = f
                .blocks()
                .flat_map(|b| f.block_insts(b))
                .find(|id| f.inst(*id).memory().is_some())
                .unwrap();
            let mut access = f.inst(id).memory().unwrap();
            if wrong_direction {
                access.kind = MemoryKind::Write;
            } else {
                access.bytes = 4;
            }
            f.editor().set_inst_memory(id, Some(access));
            let err = pipeline
                .run_function_pipeline(
                    f,
                    sig,
                    &mut CodegenStats::default(),
                    &mut FunctionAnalysisCtx::default(),
                    &mut ModuleAnalysisCtx::default(),
                    &target_pipelines,
                )
                .unwrap_err();
            assert!(
                err.to_string()
                    .contains("selection changed the memory access"),
                "{err}"
            );
        }
    }

    #[test]
    fn access_contracts_survive_the_complete_machine_pipeline() {
        let module = veloc_mir::ModuleParser::new()
            .parse(
                r#"
export function access(ptr, i64) -> i64

block0(v0: ptr, v1: i64):
  ss0: ptr = alloca size=8, align=8
  store.volatile.align8 v1, v0, offset=8
  v2: i64 = load.volatile.align8 v0, offset=8
  store v2, ss0, offset=0
  v3: i64 = load ss0, offset=0
  return v3
"#,
            )
            .unwrap();
        module.validate().unwrap();
        let target = crate::create_target_machine(crate::TargetConfig::default()).unwrap();
        for optimize in [false, true] {
            let pipeline = CodegenPipeline::with_options(
                &*target,
                CodegenOptions {
                    optimize,
                    ..Default::default()
                },
            );
            let compiled = pipeline
                .compile_module_artifact(
                    &module,
                    &mut CodegenStats::default(),
                    &mut ModuleAnalysisCtx::default(),
                )
                .unwrap();
            let f = &compiled.functions[0].machine_function;
            let accesses: Vec<_> = f
                .blocks()
                .flat_map(|b| f.block_insts(b))
                .filter_map(|id| f.inst(id).memory())
                .collect();
            assert_eq!(accesses.len(), 4);
            for (access, kind) in accesses.iter().zip([
                MemoryKind::Write,
                MemoryKind::Read,
                MemoryKind::Write,
                MemoryKind::Read,
            ]) {
                assert_eq!(access.kind, kind);
                assert_eq!(access.bytes, 8);
            }
            for access in &accesses[..2] {
                assert_eq!(access.alignment, 8);
                assert!(access.volatile);
                assert!(access.may_trap);
            }
            for access in &accesses[2..] {
                assert!(!access.volatile);
                assert!(!access.may_trap);
            }
        }
    }
}
