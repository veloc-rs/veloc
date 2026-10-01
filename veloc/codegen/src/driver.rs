//! Code generation driver.
//!
//! 提供从 SSA IR 到机器码/目标文件的编译驱动。

use crate::error::{Error, Result};
use crate::object::ObjectFileBuilder;
use crate::pipeline::{
    CompiledFunction, CompiledModule, EmissionModule, EmittedFunction, FunctionPipeline,
    ModuleCodegenPass, ModulePassContext, ModulePassPipeline,
};
use crate::target::TargetMachine;
use crate::translate::{IRTranslator, TranslatedModule};
use std::vec::Vec;
use veloc_lir::{MachineFunction, MachineModule};
use veloc_mir::Module;
use veloc_profile::{Metric, Profile};

/// Optimization policy used when constructing codegen pipelines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptLevel {
    /// Run only the transformations required for correct code generation.
    None,
    /// Enable the standard optimization pipeline.
    #[default]
    Default,
}

/// 代码生成选项
#[derive(Debug, Clone)]
pub struct CodegenOptions {
    /// Validate SSA, selected and allocated invariants at pipeline boundaries.
    /// Enabled by default in debug builds; construction itself remains unchecked.
    pub verify: bool,
    /// Select the pass sequence at pipeline construction time.
    pub opt_level: OptLevel,
    /// Pass names whose output should be printed; `*` selects every pass.
    pub dump_after: Vec<std::string::String>,
    /// Restrict pass dumps to one function (None selects all functions).
    pub dump_function: Option<std::string::String>,
}

impl Default for CodegenOptions {
    fn default() -> Self {
        Self {
            verify: cfg!(debug_assertions),
            opt_level: OptLevel::Default,
            dump_after: Vec::new(),
            dump_function: None,
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
/// let pipeline = CodegenPipeline::new(&*target, Default::default());
///
/// let object = pipeline.compile_object(&module).unwrap();
/// ```
pub struct CodegenPipeline<'a> {
    target: &'a dyn TargetMachine,
    options: CodegenOptions,
    profile: Profile,
}

impl<'a> CodegenPipeline<'a> {
    /// Create a driver with explicit options; explicit dump settings override the environment.
    pub fn new(target: &'a dyn TargetMachine, mut options: CodegenOptions) -> Self {
        if options.dump_after.is_empty() {
            if let Ok(filter) = std::env::var("VELOC_DUMP_LIR") {
                options.dump_after.push("*".into());
                if filter != "*" && options.dump_function.is_none() {
                    options.dump_function = Some(filter);
                }
            }
        }
        Self {
            target,
            options,
            profile: Profile::default(),
        }
    }

    /// 编译整个模块并生成单个 relocatable object 文件。
    pub fn compile_object(&self, module: &Module) -> Result<Vec<u8>> {
        self.profile.measure("codegen", 0, || {
            let mut object = ObjectFileBuilder::new(self.target)?;
            let compiled = self.compile_emission_module(module)?;
            self.profile.measure("object", 0, || {
                for compiled_func in &compiled.functions {
                    let func = module.function(compiled_func.func_id);
                    object.add_defined_function(
                        &func,
                        compiled_func.symbol,
                        &compiled_func.emission,
                        &compiled.symbols,
                    );
                }
                // Only referenced imports need runtime symbol resolution.
                object.finish(&self.profile)
            })
        })
    }

    pub fn with_profile(mut self, profile: Profile) -> Self {
        self.profile = profile;
        self
    }

    fn compile_emission_module(&self, module: &Module) -> Result<EmissionModule> {
        let machines = self.compile_machine_module(module)?;
        let mut emission = self.emit_compiled_functions(machines)?;
        let name = emission.name.clone();
        self.run_module_passes(
            "post_emit",
            &mut emission,
            &name,
            self.target
                .pass_config()
                .post_emit_module_passes(self.options.opt_level),
        )?;
        Ok(emission)
    }

    fn compile_machine_module(&self, module: &Module) -> Result<CompiledModule> {
        let TranslatedModule {
            machine: mmodule,
            sources,
        } = self.translate_module(module)?;
        let MachineModule {
            name,
            mut symbols,
            functions,
        } = mmodule;
        let mut compiled_functions = Vec::new();
        let function_pipeline = FunctionPipeline::new(self.target, &self.options, &self.profile);

        for (machine_id, mfunc) in functions {
            let func_id = sources[machine_id];
            let func = module.function(func_id);
            let machine_function = function_pipeline.run(
                mfunc,
                &module.signatures()[func.decl.signature],
                &mut symbols,
            )?;
            compiled_functions.push(CompiledFunction {
                func_id,
                symbol: symbols.get_or_create_function(&func.decl.name, func.decl.linkage),
                name: func.decl.name.clone(),
                machine_function,
            });
        }

        let mut compiled = CompiledModule::new(name, symbols, compiled_functions);
        let name = compiled.name.clone();
        self.run_module_passes(
            "pre_emit",
            &mut compiled,
            &name,
            self.target
                .pass_config()
                .pre_emit_module_passes(self.options.opt_level),
        )?;
        Ok(compiled)
    }

    fn translate_module(&self, module: &Module) -> Result<TranslatedModule> {
        self.profile.measure("translate", 0, || {
            IRTranslator::new(module, self.target.desc().data_layout).translate_module()
        })
    }

    fn run_module_passes<M: core::fmt::Debug>(
        &self,
        stage: &'static str,
        module: &mut M,
        name: &str,
        passes: Vec<Box<dyn ModuleCodegenPass<M>>>,
    ) -> Result<()> {
        self.profile.measure(stage, 0, || {
            let pipeline = ModulePassPipeline::from_passes(passes);
            let mut ctx = ModulePassContext::new(self.target, &self.profile, name);
            pipeline.run(module, &mut ctx)
        })
    }

    fn emit_compiled_functions(&self, compiled: CompiledModule) -> Result<EmissionModule> {
        let mut functions = Vec::with_capacity(compiled.functions.len());
        for func in compiled.functions {
            let scope = self.profile.entity_scope("emit", 0, || func.name.clone());
            let result = self.emit_function(&func.machine_function);
            scope.result(&result);
            functions.push(EmittedFunction {
                func_id: func.func_id,
                symbol: func.symbol,
                name: func.name,
                emission: result?,
            });
        }
        Ok(EmissionModule {
            name: compiled.name,
            symbols: compiled.symbols,
            functions,
        })
    }

    fn emit_function(&self, mfunc: &MachineFunction) -> Result<crate::Emitter> {
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
        Ok(output)
    }
}
