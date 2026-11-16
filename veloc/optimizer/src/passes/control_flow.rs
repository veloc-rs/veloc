//! Iterate branch threading and structural cleanup at pipeline boundaries.
//! Speculative condition merging runs separately before expression optimization.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use veloc_analyzer::AnalysisManager;

pub struct ControlFlowPass;
impl FunctionPass for ControlFlowPass {
    fn reuse_key(&self) -> Option<core::any::TypeId> {
        Some(core::any::TypeId::of::<Self>())
    }
    fn name(&self) -> &'static str {
        "ControlFlowPass"
    }
    fn run(
        &self,
        am: &mut AnalysisManager<'_>,
        config: &OptConfig,
        metrics: &Profile,
    ) -> PassOutcome {
        use super::{cfg::CfgPass, threading::BypassPass};
        let mut changed = am.function_mut().edit().remove_unreachable();
        loop {
            let mut round = false;
            // Keep specialization before merging: merging can hide a small
            // comparison/dispatch block that could have been bypassed.
            for pass in [&BypassPass as &dyn FunctionPass, &CfgPass] {
                let scope = metrics.scope(pass.name(), 0);
                round |= pass.run(am, config, metrics).changed();
                scope.success();
            }
            changed |= round;
            if !round {
                break;
            }
        }
        if changed {
            PassOutcome::Changed
        } else {
            PassOutcome::Unchanged
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::passes::{cfg::CfgPass, if_conversion::IfConversionPass, threading::BypassPass};
    use veloc_mir::{FuncId, InstView, Module, ModuleParser, Opcode};

    fn run(source: &str, pass: &impl FunctionPass) -> Module {
        let mut module = ModuleParser::new().parse(source).unwrap();
        module.validate().unwrap();
        let body = module.body_mut(FuncId(0)).unwrap();
        pass.run(
            &mut AnalysisManager::new(body),
            &OptConfig::default(),
            &Profile::default(),
        );
        module.validate().unwrap();
        module
    }

    fn count(module: &Module, opcode: Opcode) -> usize {
        let body = module.function(FuncId(0)).body.unwrap();
        body.layout()
            .block_order()
            .flat_map(|b| body.layout().block_insts(b))
            .filter(|&i| body.dfg().opcode(i) == opcode)
            .count()
    }

    #[test]
    fn cleanup_selects_the_exact_table_edge_including_default() {
        for (index, expected) in [(0, 10), (1, 20), (100, 30)] {
            let source = format!(
                "local function choose() -> i32\nblock0():\n  br-table i32({index}), [block1(i32(10)), block1(i32(20))], block1(i32(30))\nblock1(v0: i32):\n  return v0\n"
            );
            let mut module = run(&source, &CfgPass);
            let body = module.body_mut(FuncId(0)).unwrap();
            assert_eq!(body.layout().block_order().count(), 1);
            let ret = body.layout().last_inst(body.entry_block()).unwrap();
            let value = body.dfg().operands(ret)[0];
            assert_eq!(
                body.dfg().as_scalar_const(value).unwrap().to_bits(),
                expected
            );
            assert_eq!(
                CfgPass.run(
                    &mut AnalysisManager::new(body),
                    &OptConfig::default(),
                    &Profile::default()
                ),
                PassOutcome::Unchanged
            );
        }
    }

    #[test]
    fn cleanup_handles_reverse_layout_and_escaping_parameter_uses() {
        let source = "local function choose() -> i32
block0():
  jump block3(bool(true))
block1():
  return i32(7)
block2():
  br v0, block1(), block4()
block3(v0: bool):
  jump block2()
block4():
  return i32(9)
";
        let module = run(source, &CfgPass);
        let body = module.function(FuncId(0)).body.unwrap();
        assert_eq!(body.layout().block_order().count(), 1);
        assert_eq!(count(&module, Opcode::Br), 0);
    }

    #[test]
    fn cleanup_terminates_on_forwarding_cycles() {
        let mut module = run(
            "local function forever() -> void
block0():
  jump block1()
block1():
  jump block2()
block2():
  jump block1()
",
            &CfgPass,
        );
        let body = module.body_mut(FuncId(0)).unwrap();
        assert_eq!(body.layout().block_order().count(), 2);
        assert_eq!(
            CfgPass.run(
                &mut AnalysisManager::new(body),
                &OptConfig::default(),
                &Profile::default()
            ),
            PassOutcome::Unchanged
        );
    }

    #[test]
    fn incoming_constants_remain_edge_specific() {
        let module = run(
            "local function choose(v0: bool) -> i32
block0():
  br v0, block1(bool(true)), block1(bool(false))
block1(v1: bool):
  br v1, block2(), block3()
block2():
  return i32(11)
block3():
  return i32(22)
",
            &BypassPass,
        );
        let body = module.function(FuncId(0)).body.unwrap();
        let entry = body.layout().last_inst(body.entry_block()).unwrap();
        let view = body.dfg().inst(entry);
        let yes = view.successor(0).unwrap().block;
        let no = view.successor(1).unwrap().block;
        assert_ne!(yes, no);
        for (block, expected) in [(yes, 11), (no, 22)] {
            let ret = body.layout().last_inst(block).unwrap();
            assert_eq!(body.dfg().opcode(ret), Opcode::Return);
            assert_eq!(
                body.dfg()
                    .as_scalar_const(body.dfg().operands(ret)[0])
                    .unwrap()
                    .to_bits(),
                expected
            );
        }
    }

    #[test]
    fn comparison_bypass_uses_the_taken_edge_predicate() {
        let module = run(
            "local function compare(v0: i32, v1: i32) -> i32
block0():
  v2: bool = icmp lts v0, v1
  br v2, block1(), block3()
block1():
  v3: bool = icmp les v0, v1
  br v3, block2(), block3()
block2():
  return i32(1)
block3():
  return i32(0)
",
            &BypassPass,
        );
        let body = module.function(FuncId(0)).body.unwrap();
        let entry = body.layout().last_inst(body.entry_block()).unwrap();
        let target = body.dfg().inst(entry).successor(0).unwrap().block;
        assert_eq!(
            body.dfg().opcode(body.layout().first_inst(target).unwrap()),
            Opcode::Return
        );
    }

    #[test]
    fn cleanup_and_if_conversion_have_separate_responsibilities() {
        let source = "local function diamond(v0: bool, v1: i32) -> i32
block0():
  br v0, block1(), block2()
block1():
  v2: i32 = iadd v1, i32(1)
  jump block3(v2)
block2():
  v3: i32 = isub v1, i32(1)
  jump block3(v3)
block3(v4: i32):
  return v4
";
        let cleaned = run(source, &CfgPass);
        assert_eq!(count(&cleaned, Opcode::Br), 1);
        assert_eq!(count(&cleaned, Opcode::Select), 0);
        let converted = run(source, &IfConversionPass);
        assert_eq!(count(&converted, Opcode::Br), 0);
        assert_eq!(count(&converted, Opcode::Select), 1);
        let mut combined = converted;
        let body = combined.body_mut(FuncId(0)).unwrap();
        ControlFlowPass.run(
            &mut AnalysisManager::new(body),
            &OptConfig::default(),
            &Profile::default(),
        );
        combined.validate().unwrap();
        let body = combined.body_mut(FuncId(0)).unwrap();
        assert_eq!(body.layout().block_order().count(), 1);
        assert_eq!(
            ControlFlowPass.run(
                &mut AnalysisManager::new(body),
                &OptConfig::default(),
                &Profile::default()
            ),
            PassOutcome::Unchanged
        );
    }

    #[test]
    fn if_conversion_keeps_potentially_trapping_arms_conditional() {
        let module = run(
            "local function guarded(v0: bool, v1: i32) -> i32
block0():
  br v0, block1(), block2(i32(0))
block1():
  v2: i32 = idiv-s i32(42), v1
  jump block2(v2)
block2(v3: i32):
  return v3
",
            &IfConversionPass,
        );
        assert_eq!(count(&module, Opcode::Br), 1);
        assert_eq!(count(&module, Opcode::Select), 0);
        let body = module.function(FuncId(0)).body.unwrap();
        assert!(matches!(
            body.dfg()
                .inst(body.layout().last_inst(body.entry_block()).unwrap()),
            InstView::Br { .. }
        ));
    }
    #[test]
    fn generated_branch_rules_preserve_arguments_and_pending_facts() {
        use crate::rewrite::{self, Fact, Successors};
        let mut module = ModuleParser::new()
            .parse(
                "local function branch(v0: bool, v1: i32, v2: i32) -> i32
block0():
  v3: bool = select v0, bool(false), bool(true)
  br v3, block1(v1), block1(v2)
block1(v4: i32):
  return v4
",
            )
            .unwrap();
        module.validate().unwrap();
        let f = module.body_mut(FuncId(0)).unwrap();
        let inst = f.layout().last_inst(f.entry_block()).unwrap();
        let view = f.dfg().inst(inst);
        assert!(matches!(
            rewrite::successors(view, |_| Fact::Unknown),
            Successors::Pending
        ));
        assert!(matches!(
            rewrite::successors(view, |_| Fact::Varying),
            Successors::All
        ));
        crate::rewrite::find_rewrite(f, inst)
            .unwrap()
            .apply(f, inst);
        let view = f.dfg().inst(inst);
        assert_eq!(f.dfg().operands(inst)[0], f.params()[0]);
        assert_eq!(view.successor(0).unwrap().args, &[f.params()[2]]);
        assert_eq!(view.successor(1).unwrap().args, &[f.params()[1]]);
        // Sharing a destination does not make distinct argument lists equal.
        let edge = view.successor(1).unwrap();
        let dest = edge.block;
        let value = edge.args[0];
        assert!(crate::rewrite::find_rewrite(f, inst).is_none());
        f.edit().redirect_edge(
            veloc_mir::function::EdgeRef { inst, index: 0 },
            dest,
            &[value],
        );
        crate::rewrite::find_rewrite(f, inst)
            .unwrap()
            .apply(f, inst);
        assert_eq!(f.dfg().opcode(inst), Opcode::Jump);
        module.validate().unwrap();
    }
}
