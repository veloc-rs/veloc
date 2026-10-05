//! Bounded bottom-up inlining of module-local definitions.
//! Recursion and stack allocation retain calls. Cloning uses the MIR schema,
//! keeping control edges, instruction properties and value mappings together.
use crate::{ModulePass, OptConfig, PassOutcome, Profile};
use std::collections::{HashMap, HashSet};
use veloc_analyzer::{Dominators, graph::LoopInfo};
use veloc_mir::{Block, FuncBody, FuncId, Inst, InstView, Module, Opcode, Successor};
use veloc_types::TypeInfo;

veloc_policy::feature_set!(InlineFeatures {
    callee_insts,
    callee_blocks,
    callee_params,
    callee_calls,
    callee_loads,
    callee_stores,
    callee_branches,
    callee_loop_blocks,
    caller_insts,
    caller_blocks,
    constant_args,
    pointer_args,
    call_block_insts,
    call_loop_depth,
    growth,
    callee_multiplies,
    callee_to_caller,
    growth_to_caller,
    constant_arg_fraction,
    pointer_arg_fraction,
    callee_loop_fraction,
    callee_memory_density,
    callee_call_density,
});

pub struct InlinePass;
pub struct DevirtualizePass;

impl InlinePass {
    pub const POLICY_SCHEMA: veloc_policy::DecisionSchema = veloc_policy::DecisionSchema {
        name: "inline",
        version: 2,
        features: InlineFeatures::NAMES,
        actions: &["heuristic", "keep_call", "inline"],
        scope: "caller",
    };
}

impl ModulePass for DevirtualizePass {
    fn name(&self) -> &'static str {
        "DevirtualizePass"
    }
    fn run(&self, module: &mut Module, _: &OptConfig, metrics: &Profile) -> PassOutcome {
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
            PassOutcome::Unchanged
        } else {
            PassOutcome::Changed
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
    fn run(&self, module: &mut Module, config: &OptConfig, metrics: &Profile) -> PassOutcome {
        let policy = config
            .policy
            .as_deref()
            .filter(|p| p.enabled(&Self::POLICY_SCHEMA));
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
        let mut summaries = HashMap::new();
        let advice = policy.map(|p| p.session(&Self::POLICY_SCHEMA));
        for id in ordered {
            let Some(body) = module.function(id).body else {
                continue;
            };
            let sites = calls(body);
            // Capture original callsite depth once. Inlining may move a later
            // call into a continuation, while preserving its loop context.
            let depths: HashMap<Inst, u32> = if policy.is_some() && !sites.is_empty() {
                let dom = Dominators::compute(body.cfg(), body.entry_block());
                let loops = LoopInfo::compute(body.cfg(), &dom);
                sites
                    .iter()
                    .map(|&(inst, _)| (inst, loops.depth(body.layout().inst_block(inst).unwrap())))
                    .collect()
            } else {
                HashMap::new()
            };
            let original = body.dfg().inst_count();
            let mut growth = 0;
            if let Some(advice) = &advice {
                advice.reset_scope();
            }
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
                // Legality and the caller growth limit apply to every policy.
                if growth + size > (original * 4).max(256).min(2000)
                    || source
                        .layout()
                        .block_order()
                        .flat_map(|b| source.layout().block_insts(b))
                        .any(|i| matches!(source.dfg().opcode(i), Opcode::Alloca))
                {
                    continue;
                }
                let heuristic = size <= 300 && (size <= 32 || calls(source).is_empty());
                let action = advice
                    .as_ref()
                    .filter(|p| p.wants_features())
                    .map_or(0, |policy| {
                        let caller = module.function(id).body.unwrap();
                        // Bottom-up order means nonrecursive callees are complete.
                        // Their structural summary remains valid for every caller.
                        let summary = summaries
                            .entry(callee)
                            .or_insert_with(|| callee_features(source, size));
                        let features =
                            inline_features(caller, site, *summary, growth, depths[&site]);
                        metrics.count("inline.policy_decisions", 1);
                        policy.choose(&features.values(), if heuristic { 2 } else { 1 })
                    });
                let selected = match action {
                    0 => heuristic,
                    1 => false,
                    2 => true,
                    _ => unreachable!(),
                };
                if !selected {
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
            PassOutcome::Unchanged
        } else {
            PassOutcome::Changed
        }
    }
}

/// Structural features only: no symbol, source path, benchmark or instruction ID.
fn callee_features(callee: &FuncBody, size: usize) -> InlineFeatures {
    let mut features = InlineFeatures {
        callee_insts: size as f32,
        callee_blocks: callee.layout().block_order().count() as f32,
        callee_params: callee.params().len() as f32,
        ..Default::default()
    };
    let dom = Dominators::compute(callee.cfg(), callee.entry_block());
    let loops = LoopInfo::compute(callee.cfg(), &dom);
    for block in callee.layout().block_order() {
        features.callee_loop_blocks += f32::from(loops.depth(block) != 0);
        for inst in callee.layout().block_insts(block) {
            let opcode = callee.dfg().opcode(inst);
            features.callee_calls +=
                f32::from(matches!(opcode, Opcode::Call | Opcode::CallIndirect));
            features.callee_loads += f32::from(opcode.spec().memory_effect().may_read());
            features.callee_stores += f32::from(opcode.spec().memory_effect().may_write());
            features.callee_branches += f32::from(opcode.spec().is_terminator());
            features.callee_multiplies += f32::from(matches!(opcode, Opcode::IMul | Opcode::FMul));
        }
    }
    features
}

fn inline_features(
    caller: &FuncBody,
    site: Inst,
    mut features: InlineFeatures,
    growth: usize,
    depth: u32,
) -> InlineFeatures {
    features.caller_insts = caller
        .layout()
        .block_order()
        .map(|b| caller.layout().block_insts(b).count())
        .sum::<usize>() as f32;
    features.caller_blocks = caller.layout().block_order().count() as f32;
    let InstView::Call { args, .. } = caller.dfg().inst(site) else {
        unreachable!()
    };
    for &arg in args {
        features.constant_args += f32::from(caller.dfg().as_const(arg).is_some());
        features.pointer_args += f32::from(caller.dfg().value_type(arg).is_ptr());
    }
    features.call_block_insts = caller
        .layout()
        .block_insts(caller.layout().inst_block(site).unwrap())
        .count() as f32;
    features.call_loop_depth = depth as f32;
    features.growth = growth as f32;
    // Ratios expose specialization and code-growth relationships across program
    // sizes. Absolute sizes remain useful for code-size and cache costs.
    features.callee_to_caller = features.callee_insts / features.caller_insts.max(1.0);
    features.growth_to_caller = features.growth / features.caller_insts.max(1.0);
    features.constant_arg_fraction = features.constant_args / (args.len().max(1) as f32);
    features.pointer_arg_fraction = features.pointer_args / (args.len().max(1) as f32);
    features.callee_loop_fraction = features.callee_loop_blocks / features.callee_blocks.max(1.0);
    features.callee_memory_density =
        (features.callee_loads + features.callee_stores) / features.callee_insts.max(1.0);
    features.callee_call_density = features.callee_calls / features.callee_insts.max(1.0);
    features
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
    let mut values: HashMap<_, _> = source.params().iter().copied().zip(args).collect();
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
                args: &[],
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
