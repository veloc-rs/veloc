//! Code generation driver.
//!
//! 提供从 SSA IR 到机器码/目标文件的编译驱动。

use crate::error::{Error, Result};
use crate::object::ObjectFileBuilder;
use crate::pipeline::{
    CompiledFunction, CompiledModule, EmissionModule, EmittedFunction, FunctionPipeline,
    ModulePassContext, ModulePassPipeline,
};
use crate::target::TargetMachine;
use crate::translate::IRTranslator;
use std::collections::BTreeMap;
use std::vec::Vec;
use veloc_lir::{MachineFunction, MachineModule};
use veloc_mir::Module;
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

impl<'a> CodegenPipeline<'a> {
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
        let compiled = self.compile_module_artifact(module)?;

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

            // Referenced imports are created while emitting relocations. Unused
            // declarations must not require runtime symbol resolution.
            object.finish(&self.profile)
        })
    }

    /// 编译模块中的所有已定义函数并返回裸机器码。
    pub fn compile_functions(
        &self,
        module: &veloc_mir::Module,
    ) -> Result<BTreeMap<veloc_mir::FuncId, Vec<u8>>> {
        self.profile.measure("codegen", 0, || {
            let EmissionModule {
                symbols, functions, ..
            } = self.compile_module_artifact(module)?;
            let mut results = BTreeMap::new();

            for compiled_func in functions {
                let emitted = compiled_func.emission.finish()?;
                if let Some(reloc) = emitted.relocations.first() {
                    let symbol = symbols.get(reloc.symbol).name.clone();
                    return Err(Error::unexpected_relocation(symbol));
                }
                self.profile
                    .record_lazy(Metric::bytes("code"), || emitted.data.len() as u64);
                results.insert(compiled_func.func_id, emitted.data);
            }

            Ok(results)
        })
    }

    fn compile_module_artifact(&self, module: &Module) -> Result<EmissionModule> {
        let machines = self.compile_machine_module(module)?;
        let mut emission = self.emit_compiled_functions(machines)?;
        self.profile.measure("post_emit", 0, || {
            self.run_module_post_emit_passes(&mut emission)
        })?;
        Ok(emission)
    }

    fn compile_machine_module(&self, module: &Module) -> Result<CompiledModule> {
        let mmodule = self.translate_module(module)?;
        let veloc_lir::MachineModule {
            name,
            mut symbols,
            functions,
        } = mmodule;
        let mut compiled_functions = Vec::new();
        let function_pipeline = FunctionPipeline::new(self.target, &self.options, &self.profile);

        for ((func_id, func), (_, mfunc)) in module
            .functions()
            .filter(|(_, f)| f.body.is_some())
            .zip(functions.into_iter())
        {
            debug_assert_eq!(func.decl.name, mfunc.name);
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
        self.profile.measure("pre_emit", 0, || {
            self.run_module_pre_emit_passes(&mut compiled)
        })?;
        Ok(compiled)
    }

    fn translate_module(&self, module: &Module) -> Result<MachineModule> {
        self.profile.measure("translate", 0, || {
            IRTranslator::new(module, self.target.desc().data_layout).translate_module()
        })
    }

    fn run_module_pre_emit_passes(&self, compiled: &mut CompiledModule) -> Result<()> {
        let mut pipeline = ModulePassPipeline::new();
        for pass in self.target.pass_config().pre_emit_module_passes() {
            pipeline.add_boxed_pass(pass);
        }
        let mut ctx =
            ModulePassContext::new(self.target, &self.options, &self.profile, &compiled.name);
        pipeline.run(compiled, &mut ctx)?;
        Ok(())
    }

    fn run_module_post_emit_passes(&self, compiled: &mut EmissionModule) -> Result<()> {
        let mut pipeline = ModulePassPipeline::new();
        for pass in self.target.pass_config().post_emit_module_passes() {
            pipeline.add_boxed_pass(pass);
        }
        let mut ctx =
            ModulePassContext::new(self.target, &self.options, &self.profile, &compiled.name);
        pipeline.run(compiled, &mut ctx)?;
        Ok(())
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
        let function_pipeline =
            FunctionPipeline::new(&*target, &pipeline.options, &pipeline.profile);
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
            let err = function_pipeline
                .run(f, sig, &mut translated.symbols.clone())
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
            let compiled = pipeline.compile_machine_module(&module).unwrap();
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
