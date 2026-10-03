//! Bounded bottom-up inlining of module-local definitions.
//! Recursion and stack allocation retain calls. Cloning uses the MIR schema,
//! keeping control edges, instruction properties and value mappings together.
use crate::{ModulePass, OptConfig, PreservedAnalyses, Profile};
use std::collections::{HashMap, HashSet};
use veloc_mir::{Block, FuncBody, FuncId, Inst, InstView, Module, Opcode, Successor};

pub struct InlinePass;
pub struct DevirtualizePass;

impl ModulePass for DevirtualizePass {
    fn name(&self) -> &'static str {
        "DevirtualizePass"
    }
    fn run(&self, module: &mut Module, _: &OptConfig, metrics: &Profile) -> PreservedAnalyses {
        let decls = module.decls().clone();
        let mut changed = 0;
        for (_, body) in module.bodies_mut() {
            let insts: Vec<_> = body
                .layout()
                .block_order()
                .flat_map(|b| body.layout().block_insts(b))
                .collect();
            for inst in insts {
                let InstView::CallIndirect { ptr, args, sig_id } = body.dfg().inst(inst) else {
                    continue;
                };
                let Some(source) = body.dfg().value_inst(ptr) else {
                    continue;
                };
                let InstView::FuncAddr { func_id } = body.dfg().inst(source) else {
                    continue;
                };
                if decls[func_id].signature != sig_id {
                    continue;
                }
                let args = args.to_vec();
                body.edit().replace_inst(inst, |w| w.call(func_id, &args));
                changed += 1;
            }
        }
        metrics.count("devirtualize.calls", changed);
        if changed == 0 {
            PreservedAnalyses::all()
        } else {
            PreservedAnalyses::none()
        }
    }
}

fn calls(body: &FuncBody) -> Vec<(Inst, FuncId)> {
    body.layout()
        .block_order()
        .flat_map(|b| body.layout().block_insts(b))
        .filter_map(|i| {
            if let InstView::Call { func_id, .. } = body.dfg().inst(i) {
                Some((i, func_id))
            } else {
                None
            }
        })
        .collect()
}
fn order(
    module: &Module,
    id: FuncId,
    seen: &mut HashSet<FuncId>,
    active: &mut HashSet<FuncId>,
    recursive: &mut HashSet<FuncId>,
    out: &mut Vec<FuncId>,
) {
    if active.contains(&id) {
        recursive.extend(active.iter().copied());
        return;
    }
    if !seen.insert(id) {
        return;
    }
    active.insert(id);
    if let Some(body) = module.function(id).body {
        for (_, callee) in calls(body) {
            order(module, callee, seen, active, recursive, out);
        }
    }
    active.remove(&id);
    out.push(id);
}
impl ModulePass for InlinePass {
    fn name(&self) -> &'static str {
        "InlinePass"
    }
    fn run(&self, module: &mut Module, _: &OptConfig, metrics: &Profile) -> PreservedAnalyses {
        let mut ordered = Vec::new();
        let mut seen = HashSet::new();
        let mut recursive = HashSet::new();
        for (id, _) in module.functions() {
            order(
                module,
                id,
                &mut seen,
                &mut HashSet::new(),
                &mut recursive,
                &mut ordered,
            );
        }
        let mut changed = 0;
        for id in ordered {
            let Some(body) = module.function(id).body else {
                continue;
            };
            let sites = calls(body);
            let original = body.dfg().inst_count();
            let mut growth = 0;
            for (site, callee) in sites {
                if recursive.contains(&callee)
                    || module.signatures()[module.decls()[callee].signature].variadic
                {
                    continue;
                }
                let Some(source) = module.function(callee).body else {
                    continue;
                };
                let size = source
                    .layout()
                    .block_order()
                    .map(|b| source.layout().block_insts(b).count())
                    .sum::<usize>();
                // Large bodies that retain calls lengthen many caller values
                // across ABI clobbers. Keep those boundaries; small wrappers
                // and callback dispatchers can still expose specialization.
                if size > 32 && !calls(source).is_empty() {
                    continue;
                }
                if size > 300
                    || growth + size > (original * 4).max(256).min(2000)
                    || source
                        .layout()
                        .block_order()
                        .flat_map(|b| source.layout().block_insts(b))
                        .any(|i| matches!(source.dfg().opcode(i), Opcode::Alloca))
                {
                    continue;
                }
                let source = source.clone();
                inline(module.body_mut(id).unwrap(), site, &source);
                growth += size;
                changed += 1;
            }
        }
        metrics.count("inline.calls", changed);
        if changed == 0 {
            PreservedAnalyses::all()
        } else {
            PreservedAnalyses::none()
        }
    }
}

pub(super) fn inline(target: &mut FuncBody, call: Inst, source: &FuncBody) {
    fn visit(source: &FuncBody, b: Block, seen: &mut HashSet<Block>, out: &mut Vec<Block>) {
        if !seen.insert(b) {
            return;
        }
        for &s in source.cfg().succs(b) {
            visit(source, s, seen, out);
        }
        out.push(b);
    }
    let mut ordered = Vec::new();
    visit(
        source,
        source.entry_block(),
        &mut HashSet::new(),
        &mut ordered,
    );
    let InstView::Call { args, .. } = target.dfg().inst(call) else {
        unreachable!()
    };
    let args = args.to_vec();
    let caller = target.layout().inst_block(call).unwrap();
    let continuation = target.edit().create_block();
    target.edit().append_block(continuation);
    let results = target.dfg().inst_results(call).to_vec();
    for result in results {
        let ty = target.dfg().value_type(result);
        let param = target.edit().append_block_param(continuation, ty);
        target.edit().replace_all_uses(result, param);
    }
    let mut tail = target.layout().next_inst(call);
    while let Some(inst) = tail {
        tail = target.layout().next_inst(inst);
        target.edit().move_to_end(inst, continuation);
    }
    target.edit().erase_inst(call);
    let mut blocks = HashMap::new();
    let mut values = HashMap::new();
    for &block in ordered.iter().rev() {
        let new = target.edit().create_block();
        target.edit().append_block(new);
        blocks.insert(block, new);
        for &param in source.dfg().block_params(block) {
            let mapped = target
                .edit()
                .append_block_param(new, source.dfg().value_type(param));
            values.insert(param, mapped);
        }
    }
    for (value, _) in source.dfg().values().iter() {
        if let Some(constant) = source.dfg().as_const(value) {
            let mapped = target.edit().constant(constant.clone());
            values.insert(value, mapped);
        }
    }
    target.edit().append_inst(
        caller,
        |w| {
            w.jump(Successor {
                block: blocks[&source.entry_block()],
                args: &args,
            })
        },
        &[],
    );
    // Dominating definitions precede uses in reverse postorder. Block parameters
    // are allocated above, so backedges can already reference their identities.
    for block in ordered.into_iter().rev() {
        for inst in source.layout().block_insts(block) {
            let operands: Vec<_> = source
                .dfg()
                .operands(inst)
                .iter()
                .map(|v| values[v])
                .collect();
            let view = source.dfg().inst(inst);
            if matches!(view, InstView::Return { .. }) {
                target.edit().append_inst(
                    blocks[&block],
                    |w| {
                        w.jump(Successor {
                            block: continuation,
                            args: &operands,
                        })
                    },
                    &[],
                );
            } else {
                let types: Vec<_> = source
                    .dfg()
                    .inst_results(inst)
                    .iter()
                    .map(|&v| source.dfg().value_type(v))
                    .collect();
                let new = target.edit().append_inst(
                    blocks[&block],
                    |w| w.import(view, &operands, |b| blocks[&b]),
                    &types,
                );
                for (&old, &new) in source
                    .dfg()
                    .inst_results(inst)
                    .iter()
                    .zip(target.dfg().inst_results(new))
                {
                    values.insert(old, new);
                }
            }
        }
    }
}
