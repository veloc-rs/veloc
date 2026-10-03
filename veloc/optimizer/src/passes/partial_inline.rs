//! Inline a small early-return path while outlining the remaining body.
//! Entry loads needed by the continuation are passed to the outlined function.
//! Pure computations are retained there, avoiding work on the early-return path
//! without repeating memory accesses.
use crate::{ModulePass, OptConfig, PreservedAnalyses, Profile};
use std::collections::{HashMap, HashSet};
use veloc_mir::{
    Block, FuncBody, FuncId, Inst, InstView, Linkage, Module, Opcode, Signature, SuccessorData,
    Value,
};

pub struct PartialInlinePass;

struct Shape {
    discard_entry: Vec<Inst>,
    branch: Inst,
    cold: SuccessorData,
    captures: Vec<Value>,
}

impl ModulePass for PartialInlinePass {
    fn name(&self) -> &'static str {
        "PartialInlinePass"
    }
    fn run(&self, module: &mut Module, _: &OptConfig, metrics: &Profile) -> PreservedAnalyses {
        // New outlined functions are deliberately outside this worklist.
        let callers: Vec<_> = module.functions().map(|(id, _)| id).collect();
        let mut wrappers = HashMap::<FuncId, Option<FuncBody>>::new();
        let mut changed = 0;
        for caller in callers {
            let Some(body) = module.function(caller).body else {
                continue;
            };
            let sites: Vec<_> = body
                .layout()
                .block_order()
                .flat_map(|b| body.layout().block_insts(b))
                .filter_map(|i| match body.dfg().inst(i) {
                    InstView::Call { func_id, .. } if func_id != caller => Some((i, func_id)),
                    _ => None,
                })
                .collect();
            let mut growth = 0;
            for (site, callee) in sites {
                if !wrappers.contains_key(&callee) {
                    let wrapper = make_wrapper(module, callee);
                    wrappers.insert(callee, wrapper);
                }
                let Some(wrapper) = &wrappers[&callee] else {
                    continue;
                };
                let size = wrapper
                    .layout()
                    .block_order()
                    .map(|b| wrapper.layout().block_insts(b).count())
                    .sum::<usize>();
                if growth + size > 128 {
                    continue;
                }
                super::inline::inline(module.body_mut(caller).unwrap(), site, wrapper);
                growth += size;
                changed += 1;
            }
        }
        metrics.count("partial_inline.calls", changed);
        if changed == 0 {
            PreservedAnalyses::all()
        } else {
            PreservedAnalyses::none()
        }
    }
}

fn shape(f: &FuncBody) -> Option<Shape> {
    let entry = f.entry_block();
    if !f.cfg().preds(entry).is_empty() {
        return None;
    }
    let insts: Vec<_> = f.layout().block_insts(entry).collect();
    let (&branch, entry_insts) = insts.split_last()?;
    if entry_insts.len() > 8 || !entry_insts.iter().all(|&i| {
        let view = f.dfg().inst(i);
        view.can_speculate() || matches!(view, InstView::Load { flags, .. } if !flags.is_volatile() && flags.is_notrap())
    }) { return None }
    let InstView::Br {
        then_dest,
        else_dest,
        ..
    } = f.dfg().inst(branch)
    else {
        return None;
    };
    let early_return = |block: Block| {
        if block == entry || f.cfg().preds(block) != [entry] {
            return false;
        }
        let insts: Vec<_> = f.layout().block_insts(block).collect();
        let Some((&last, work)) = insts.split_last() else {
            return false;
        };
        matches!(f.dfg().inst(last), InstView::Return { .. })
            && work.len() <= 8
            && work.iter().all(|&i| f.dfg().inst(i).can_speculate())
    };
    let (fast, cold) = match (early_return(then_dest.block), early_return(else_dest.block)) {
        (true, false) => (then_dest.block, else_dest),
        (false, true) => (else_dest.block, then_dest),
        _ => return None,
    };
    if cold.block == entry {
        return None;
    }
    // Stack addresses cannot be transported across the new call boundary.
    // Small bodies are better handled by ordinary full inlining.
    let all: Vec<_> = f
        .layout()
        .block_order()
        .flat_map(|b| f.layout().block_insts(b))
        .collect();
    if all.len() <= 32
        || all.len() > 500
        || all.iter().any(|&i| f.dfg().opcode(i) == Opcode::Alloca)
    {
        return None;
    }
    let mut needed: HashSet<Value> = entry_insts
        .iter()
        .flat_map(|&i| f.dfg().inst_results(i).iter().copied())
        .filter(|&v| {
            cold.args.contains(&v)
                || f.dfg().uses(v).any(|s| {
                    f.layout()
                        .inst_block(s.inst())
                        .is_some_and(|b| b != entry && b != fast)
                })
        })
        .collect();
    let mut captures = Vec::new();
    let mut discard_entry = Vec::new();
    // Reverse entry order computes the dependency closure. Loads are captured;
    // their address calculations need not survive in the outlined function.
    for &inst in entry_insts.iter().rev() {
        let results = f.dfg().inst_results(inst);
        if !results.iter().any(|v| needed.contains(v)) {
            discard_entry.push(inst);
        } else if f.dfg().inst(inst).can_speculate() {
            needed.extend(f.dfg().operands(inst).iter().copied());
        } else {
            captures.extend(results.iter().filter(|v| needed.contains(v)).copied());
            discard_entry.push(inst);
        }
    }
    captures.reverse();
    if f.dfg().block_params(entry).len() + captures.len() > 8 {
        return None;
    }
    Some(Shape {
        discard_entry,
        branch,
        cold: SuccessorData::new(cold.block, cold.args),
        captures,
    })
}

fn make_wrapper(module: &mut Module, callee: FuncId) -> Option<FuncBody> {
    if module.signatures()[module.decls()[callee].signature].variadic {
        return None;
    }
    let source = module.function(callee).body?;
    let shape = shape(source)?;
    let mut wrapper = source.clone();
    let mut outlined = source.clone();
    let signature = &module.signatures()[module.decls()[callee].signature];
    let mut params = signature.params().to_vec();
    params.extend(shape.captures.iter().map(|&v| source.dfg().value_type(v)));
    let signature = Signature::new(params, signature.returns().to_vec(), signature.call_conv);
    let entry = outlined.entry_block();
    let mut replacements = HashMap::new();
    for &value in &shape.captures {
        let ty = outlined.dfg().value_type(value);
        let param = outlined.edit().append_block_param(entry, ty);
        outlined.edit().replace_all_uses(value, param);
        replacements.insert(value, param);
    }
    let args: Vec<_> = shape
        .cold
        .args
        .iter()
        .map(|v| replacements.get(v).copied().unwrap_or(*v))
        .collect();
    let cold = SuccessorData::new(shape.cold.block, &args);
    outlined
        .edit()
        .replace_inst(shape.branch, |w| w.jump(cold.as_view()));
    outlined.edit().remove_unreachable();
    outlined.edit().erase_insts(&shape.discard_entry);
    let mut suffix = 0;
    let name = loop {
        let name = format!("__veloc_outline_{}_{}", callee.0, suffix);
        if module.find_function(&name).is_none() && !module.globals().iter().any(|g| g.name == name)
        {
            break name;
        }
        suffix += 1;
    };
    let sig = module.intern_signature(signature);
    let helper = module.declare_function(name, sig, Linkage::Local);
    module.define_function(helper, outlined);

    // The wrapper retains the original entry and fast return. Its cold block
    // forwards the exact entry values to the outlined continuation.
    let insts: Vec<_> = wrapper.layout().block_insts(shape.cold.block).collect();
    let (&last, old_body) = insts.split_last().unwrap();
    let mut args = wrapper.dfg().block_params(wrapper.entry_block()).to_vec();
    args.extend_from_slice(&shape.captures);
    let returns = module.signatures()[sig].returns().to_vec();
    let call = wrapper
        .edit()
        .insert_before(last, |w| w.call(helper, &args), &returns);
    let results = wrapper.dfg().inst_results(call).to_vec();
    wrapper.edit().replace_inst(last, |w| w.ret(&results));
    wrapper.edit().remove_unreachable();
    wrapper.edit().erase_insts(old_body);
    Some(wrapper)
}
