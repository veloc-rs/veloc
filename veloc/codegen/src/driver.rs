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
use veloc_profile::{Metric, Profile};

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
}

impl Default for CodegenOptions {
    fn default() -> Self {
        Self {
            verify: cfg!(debug_assertions),
            optimize: true,
            dump_after: Vec::new(),
            dump_function: None,
            dump_lir: false,
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
    profile: Profile,
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
            profile: Profile::default(),
        }
    }

    /// 创建带显式选项的代码生成驱动。
    pub fn with_options(target: &'a dyn TargetMachine, options: CodegenOptions) -> Self {
        Self {
            target,
            options,
            profile: Profile::default(),
        }
    }

    /// 编译整个模块并生成单个 relocatable object 文件。
    pub fn compile_object(&self, module: &veloc_mir::Module) -> Result<Vec<u8>> {
        self.profile
            .measure("codegen", 0, || self.compile_object_impl(module))
    }

    pub fn with_profile(mut self, profile: Profile) -> Self {
        self.profile = profile;
        self
    }

    fn compile_object_impl(&self, module: &Module) -> Result<Vec<u8>> {
        let mut object = ObjectFileBuilder::new(self.target)?;
        let mut module_analyses = ModuleAnalysisCtx::default();
        let compiled = self.compile_module_artifact(module, &mut module_analyses)?;

        self.profile.measure("object", 0, || {
            for compiled_func in &compiled.functions {
                let func = module.function(compiled_func.func_id);
                if let Some(emitted) = &compiled_func.emitted {
                    object.add_defined_function(&func, emitted, &compiled.symbols)?;
                }
            }

            // Referenced imports are created while emitting relocations. Unused
            // declarations must not require runtime symbol resolution.
            object.finish()
        })
    }

    /// 编译模块中的所有已定义函数并返回裸机器码。
    pub fn compile_functions(
        &self,
        module: &veloc_mir::Module,
    ) -> Result<BTreeMap<veloc_mir::FuncId, Vec<u8>>> {
        self.profile.measure("codegen", 0, || {
            let mut module_analyses = ModuleAnalysisCtx::default();
            let CompiledModule {
                symbols, functions, ..
            } = self.compile_module_artifact(module, &mut module_analyses)?;
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
        })
    }

    fn compile_module_artifact(
        &self,
        module: &Module,
        module_analyses: &mut ModuleAnalysisCtx,
    ) -> Result<CompiledModule> {
        let mmodule = self.translate_module(module)?;
        let veloc_lir::MachineModule {
            name,
            mut symbols,
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
            let scope = self
                .profile
                .entity_scope("function", 0, || mfunc.name.clone());
            let result = self.compile_defined_function(
                func_id,
                &func,
                &module.signatures()[func.decl.signature],
                mfunc,
                &mut symbols,
                module_analyses,
                &function_pipelines,
            );
            scope.result(&result);
            compiled_functions.push(result?);
        }

        let mut compiled = CompiledModule::new(name, symbols, compiled_functions);
        self.profile.measure("pre_emit", 0, || {
            self.run_module_pre_emit_passes(&mut compiled, module_analyses)
        })?;
        self.emit_compiled_functions(&mut compiled)?;
        self.profile.measure("post_emit", 0, || {
            self.run_module_post_emit_passes(&mut compiled, module_analyses)
        })?;
        Ok(compiled)
    }

    fn translate_module(&self, module: &Module) -> Result<MachineModule> {
        self.profile.measure("translate", 0, || {
            IRTranslator::new(module, self.target.desc().data_layout).translate_module()
        })
    }

    fn compile_defined_function(
        &self,
        func_id: FuncId,
        func: &FunctionRef,
        sig: &veloc_mir::Signature,
        mfunc: MachineFunction,
        symbols: &mut veloc_lir::SymbolTable,
        module_analyses: &mut ModuleAnalysisCtx,
        target_pipelines: &TargetFunctionPipelines,
    ) -> Result<CompiledFunction> {
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
        self.maybe_dump_mfunc("translated", &mfunc);
        crate::pipeline::dump_after("translated", &mfunc, &self.options);

        let final_mfunc = self.run_function_pipeline(
            mfunc,
            sig,
            symbols,
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
        symbols: &mut veloc_lir::SymbolTable,
        function_analyses: &mut FunctionAnalysisCtx,
        module_analyses: &mut ModuleAnalysisCtx,
        target_pipelines: &TargetFunctionPipelines,
    ) -> Result<MachineFunction> {
        use crate::verify::{verify, verify_allocated, verify_selected};
        self.verify_function("translated", &mfunc, verify)?;
        let mut ctx = FunctionPassContext::new(
            self.target,
            func_sig,
            symbols,
            &self.options,
            &self.profile,
            function_analyses,
            module_analyses,
        );

        run_function_pass(
            &crate::passes::lowering::AbiLoweringPass::new(),
            0,
            &mut mfunc,
            &mut ctx,
        )?;
        self.verify_function("abi-lowered", &mfunc, verify)?;
        self.profile.measure("prepare", 0, || {
            target_pipelines.prepare.run(&mut mfunc, &mut ctx)
        })?;
        run_function_pass(
            &LegalizePass::new(self.target.legalizer()),
            0,
            &mut mfunc,
            &mut ctx,
        )?;
        self.verify_function("legalized", &mfunc, verify)?;
        ctx.profile
            .record_lazy(Metric::count("legalized_insts"), || {
                mfunc
                    .blocks()
                    .map(|b| mfunc.block_insts(b).count() as u64)
                    .sum()
            });
        self.profile.measure("pre_isel", 0, || {
            target_pipelines.pre_isel.run(&mut mfunc, &mut ctx)
        })?;
        if self.options.verify {
            self.profile.measure("verify.legalized", 0, || {
                crate::passes::lowering::Legalizer::new(self.target.legalizer()).verify(&mfunc)
            })?;
        }
        self.verify_function("pre-isel", &mfunc, verify)?;
        run_function_pass(
            &InstructionSelectionPass::new(self.target.selector()),
            0,
            &mut mfunc,
            &mut ctx,
        )?;
        self.verify_function("selected", &mfunc, verify_selected)?;
        ctx.profile
            .record_lazy(Metric::count("selected_insts"), || {
                mfunc
                    .blocks()
                    .map(|b| mfunc.block_insts(b).count() as u64)
                    .sum()
            });

        self.profile.measure("post_isel", 0, || {
            target_pipelines.post_isel.run(&mut mfunc, &mut ctx)
        })?;
        self.verify_function("post-isel-target", &mfunc, verify_selected)?;
        run_function_pass(
            &PostIselOptimizePass::new(self.target.post_isel()),
            0,
            &mut mfunc,
            &mut ctx,
        )?;
        self.verify_function("post-isel-optimized", &mfunc, verify_selected)?;
        run_function_pass(
            &crate::passes::schedule::SchedulePass,
            0,
            &mut mfunc,
            &mut ctx,
        )?;
        self.verify_function("scheduled", &mfunc, verify_selected)?;

        // Allocation owns its exact input until its plan is materialized.
        let mut mfunc = self.profile.measure("regalloc", 0, || {
            let allocation = crate::regalloc::RegisterAllocator::new(self.target)
                .allocate(mfunc, ctx.function_analyses)?;
            allocation.materialize(self.target)
        })?;
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

        self.profile.measure("post_regalloc", 0, || {
            target_pipelines.post_regalloc.run(&mut mfunc, &mut ctx)
        })?;
        self.verify_function("post-regalloc", &mfunc, verify_allocated)?;
        run_function_pass(
            &FrameFinalizePass::new(self.target.frame_lowering()),
            0,
            &mut mfunc,
            &mut ctx,
        )?;
        self.verify_function("frame-finalized", &mfunc, verify_allocated)?;
        ctx.profile.record_lazy(Metric::count("final_insts"), || {
            mfunc
                .blocks()
                .map(|b| mfunc.block_insts(b).count() as u64)
                .sum()
        });
        ctx.profile
            .count("stack_slots", mfunc.stack_frame.slots().len() as u64);

        Ok(mfunc)
    }

    fn run_module_pre_emit_passes(
        &self,
        compiled: &mut CompiledModule,
        module_analyses: &mut ModuleAnalysisCtx,
    ) -> Result<()> {
        let mut pipeline = ModulePassPipeline::new();
        for pass in self.target.pass_config().pre_emit_module_passes() {
            pipeline.add_boxed_pass(pass);
        }
        let mut ctx =
            ModulePassContext::new(self.target, &self.options, &self.profile, module_analyses);
        let _ = pipeline.run(compiled, &mut ctx)?;
        Ok(())
    }

    fn run_module_post_emit_passes(
        &self,
        compiled: &mut CompiledModule,
        module_analyses: &mut ModuleAnalysisCtx,
    ) -> Result<()> {
        let mut pipeline = ModulePassPipeline::new();
        for pass in self.target.pass_config().post_emit_module_passes() {
            pipeline.add_boxed_pass(pass);
        }
        let mut ctx =
            ModulePassContext::new(self.target, &self.options, &self.profile, module_analyses);
        let _ = pipeline.run(compiled, &mut ctx)?;
        Ok(())
    }

    fn emit_compiled_functions(&self, compiled: &mut CompiledModule) -> Result<()> {
        for func in &mut compiled.functions {
            if func.emitted.is_none() {
                let scope = self.profile.entity_scope("emit", 0, || func.name.clone());
                let result = self.emit_function_with_relocations(&func.machine_function);
                scope.result(&result);
                func.emitted = Some(result?);
            }
        }
        Ok(())
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

    fn emit_function_with_relocations(
        &self,
        mfunc: &MachineFunction,
    ) -> Result<crate::EmittedCode> {
        if Some(mfunc.entry_block()) != mfunc.blocks().next() {
            return Err(Error::codegen("function entry must be first at emission"));
        }
        let emitter = self.target.emitter();
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
        self.profile.record_lazy(Metric::bytes("frame_size"), || {
            mfunc
                .stack_frame
                .layout()
                .expect("finalized frame")
                .total_size as u64
        });
        let emitted = output.finish()?;
        self.profile
            .record_lazy(Metric::bytes("code"), || emitted.data.len() as u64);
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
                    &mut translated.symbols.clone(),
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
                .compile_module_artifact(&module, &mut ModuleAnalysisCtx::default())
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
