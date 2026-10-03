use crate::pass::{FunctionPass, ModulePass, OptConfig, Pass};
use crate::passes::dce;
use alloc::boxed::Box;
use alloc::vec::Vec;
use veloc_analyzer::AnalysisManager;
use veloc_mir::{Module, function::FuncBody};
use veloc_profile::Profile;

/// 优化流程管理器。
pub struct PassManager {
    passes: Vec<Pass>,
    config: OptConfig,
    profile: Profile,
}

impl PassManager {
    pub fn new(config: OptConfig) -> Self {
        Self {
            passes: Vec::new(),
            config,
            profile: Profile::default(),
        }
    }

    pub fn new_o1() -> Self {
        Self::o1(
            OptConfig::default(),
            crate::passes::expression::Budget::DEFAULT,
        )
    }

    /// Native object pipeline. Immutable synthesized data and direct calls bind
    /// within this module; runtimes without module data support use `o1`.
    pub fn native_o1(layout: veloc_types::DataLayout) -> Self {
        let mut config = OptConfig::default();
        config.data_layout = Some(layout);
        let mut pm = Self::new(config.clone());
        pm.add_module_pass(crate::passes::affine::AffinePass);
        pm.add_function_pass(crate::passes::params::SimplifyParamsPass);
        pm.add_function_pass(crate::SimplifyPass);
        pm.add_function_pass(crate::passes::cfg::CfgPass);
        pm.add_function_pass(crate::SimplifyPass);
        pm.add_module_pass(crate::passes::inline::InlinePass);
        pm.add_function_pass(crate::passes::params::SimplifyParamsPass);
        pm.add_module_pass(crate::passes::inline::DevirtualizePass);
        pm.add_module_pass(crate::passes::inline::InlinePass);
        pm.add_module_pass(crate::passes::partial_inline::PartialInlinePass);
        pm.passes
            .append(&mut Self::o1(config, crate::passes::expression::Budget::DEFAULT).passes);
        pm.add_module_pass(crate::passes::bits::ReturnBitsPass);
        pm.add_function_pass(crate::SimplifyPass);
        pm.add_function_pass(dce::DcePass);
        pm
    }

    /// Prepare constants and local identities before memory forwarding, then
    /// spend the equality-search budget once on the resulting expressions.
    pub fn o1(config: OptConfig, budget: crate::passes::expression::Budget) -> Self {
        let mut pm = Self::new(config);
        pm.add_function_pass(crate::SimplifyPass);
        pm.add_function_pass(crate::passes::params::SimplifyParamsPass);
        pm.add_function_pass(crate::passes::cfg::CfgPass);
        pm.add_function_pass(crate::passes::params::SimplifyParamsPass);
        pm.add_function_pass(crate::SimplifyPass);
        pm.add_function_pass(crate::passes::MemoryPass);
        pm.add_function_pass(crate::passes::promote::PromotePass);
        pm.add_function_pass(crate::passes::params::SimplifyParamsPass);
        pm.add_function_pass(crate::ExpressionPass { budget });
        pm.add_function_pass(crate::passes::licm::LicmPass);
        pm.add_function_pass(crate::passes::strength::StrengthPass);
        pm.add_function_pass(crate::SimplifyPass);
        pm.add_function_pass(crate::passes::cse::CsePass);
        pm.add_function_pass(crate::passes::loop_memory::LoopMemoryPass);
        pm.add_function_pass(crate::passes::load_cse::LoadCsePass);
        pm.add_function_pass(crate::passes::rotate::RotatePass);
        // Expression extraction canonicalizes addresses before validity queries.
        pm.add_function_pass(crate::passes::memory_validity::MemoryValidityPass);
        pm.add_function_pass(crate::passes::bits::BitsPass);
        pm.add_function_pass(crate::SimplifyPass);
        pm.add_function_pass(dce::DcePass);
        pm.add_function_pass(crate::passes::cfg::CfgPass);
        pm.add_function_pass(crate::SimplifyPass);
        pm.add_function_pass(crate::passes::params::SimplifyParamsPass);
        pm.add_function_pass(dce::DcePass);
        pm
    }

    pub fn with_layout(mut self, layout: veloc_types::DataLayout) -> Self {
        self.config.data_layout = Some(layout);
        self
    }

    pub fn with_profile(mut self, profile: Profile) -> Self {
        self.profile = profile;
        self
    }

    pub fn config(&self) -> &OptConfig {
        &self.config
    }

    pub fn add_function_pass<P: FunctionPass + 'static>(&mut self, pass: P) {
        self.passes.push(Pass::Function(Box::new(pass)));
    }

    pub fn add_module_pass<P: ModulePass + 'static>(&mut self, pass: P) {
        self.passes.push(Pass::Module(Box::new(pass)));
    }

    /// 在整个模块上运行所有 Pass。
    pub fn run_on_module(&mut self, module: &mut Module) -> bool {
        let mut changed = false;
        let scope = self.profile.scope("optimizer", 0);

        for (position, pass) in self.passes.iter().enumerate() {
            let pass_scope = self.profile.scope(pass.name(), position as u32);

            match pass {
                Pass::Module(mp) => {
                    let pa = mp.run(module, &self.config, &self.profile);
                    if pa.changed() {
                        changed = true;
                    }
                }
                Pass::Function(fp) => {
                    let mut fp_changed = false;
                    for (id, func) in module.bodies_mut() {
                        let function_scope = self
                            .profile
                            .entity_scope("function", 0, || format!("{id:?}"));
                        let mut analyses =
                            AnalysisManager::new(func).with_profile(self.profile.clone());
                        let pa = fp.run(&mut analyses, &self.config, &self.profile);
                        function_scope.success();
                        if pa.changed() {
                            fp_changed = true;
                        }
                    }
                    if fp_changed {
                        changed = true;
                    }
                }
            };

            pass_scope.success();
        }

        scope.success();
        changed
    }

    /// 单独在某个函数上运行已注册的操作。
    pub fn run_on_function(&mut self, func: &mut FuncBody) -> bool {
        let mut changed = false;
        let scope = self.profile.scope("optimizer", 0);

        let mut analyses = AnalysisManager::new(func).with_profile(self.profile.clone());
        for (position, pass) in self.passes.iter().enumerate() {
            match pass {
                Pass::Function(fp) => {
                    let pass_scope = self.profile.scope(fp.name(), position as u32);

                    let pa = fp.run(&mut analyses, &self.config, &self.profile);
                    if pa.changed() {
                        changed = true;
                    }

                    pass_scope.success();

                    analyses.invalidate_with_preserved(|id| pa.is_preserved_id(id));
                }
                Pass::Module(_) => {}
            }
        }

        scope.success();
        changed
    }
}
