//! Worklist constant propagation and root-local simplification before memory
//! optimization. Every CFG edge is considered; this pass does not prune branches.
use crate::{FunctionPass, OptConfig, PreservedAnalyses, Profile, evaluate};
use cranelift_entity::SecondaryMap;
use smallvec::SmallVec;
use std::collections::VecDeque;
use veloc_analyzer::AnalysisManager;
use veloc_mir::{
    Block, FuncBody, Inst, InstView, Int, IntCC, Opcode, Type, TypeInfo, Value, function::EdgeRef,
};

pub struct SimplifyPass;

impl FunctionPass for SimplifyPass {
    fn name(&self) -> &'static str {
        "SimplifyPass"
    }

    fn run(
        &self,
        am: &mut AnalysisManager<'_>,
        config: &OptConfig,
        metrics: &Profile,
    ) -> PreservedAnalyses {
        let mut work = Worklist::new(am.function());
        let func = am.function_mut();
        let mut folded = 0;
        let mut parameters = 0;
        while let Some(item) = work.pop() {
            match item {
                Item::Instruction(inst) => {
                    if func.layout().inst_block(inst).is_none() {
                        continue;
                    }
                    if func.dfg().inst(inst).is_terminator() {
                        func.dfg()
                            .inst(inst)
                            .visit_successors(|edge| work.block(edge.block));
                    } else if fold_pointer_roundtrip(func, inst, config, &mut work)
                        || fold_instruction(func, inst, &mut work)
                    {
                        folded += 1;
                    }
                }
                Item::Parameters(block) => {
                    if block == func.entry_block() || work.incoming[block].is_empty() {
                        continue;
                    }
                    let params = func.dfg().block_params(block).to_vec();
                    for (index, param) in params.into_iter().enumerate() {
                        if func.dfg().uses(param).next().is_none() {
                            continue;
                        }
                        let mut literal = None;
                        let mut agrees = true;
                        for &edge in &work.incoming[block] {
                            let mut argument = None;
                            let mut position = 0;
                            func.dfg().inst(edge.inst).visit_successors(|successor| {
                                if position == edge.index {
                                    argument = successor.args.get(index).copied();
                                }
                                position += 1;
                            });
                            let Some(argument) = argument else {
                                agrees = false;
                                break;
                            };
                            // A loop carrying the same parameter preserves the
                            // constant supplied by its other incoming edges.
                            if argument == param {
                                continue;
                            }
                            if func.dfg().as_const(argument).is_none()
                                || literal.is_some_and(|value| value != argument)
                            {
                                agrees = false;
                                break;
                            }
                            literal = Some(argument);
                        }
                        if agrees && let Some(value) = literal {
                            work.replace(func, param, value);
                            parameters += 1;
                        }
                    }
                }
            }
        }
        metrics.count("simplify.folded_insts", folded);
        metrics.count("simplify.constant_params", parameters);
        if folded + parameters == 0 {
            PreservedAnalyses::all()
        } else {
            if config.is_debug_enabled("simplify") {
                log::info!("Simplified {folded} instructions and {parameters} block parameters");
            }
            PreservedAnalyses::none()
        }
    }
}

/// Pointer/integer round trips are identities only at widths supported by the
/// target layout. Keeping this proof in MIR benefits every code generator.
fn fold_pointer_roundtrip(
    f: &mut FuncBody,
    inst: Inst,
    config: &OptConfig,
    work: &mut Worklist,
) -> bool {
    let Some(layout) = config.data_layout else {
        return false;
    };
    let (outer, arg) = match f.dfg().inst(inst) {
        InstView::PtrToInt { arg } => (Opcode::PtrToInt, arg),
        InstView::IntToPtr { arg } => (Opcode::IntToPtr, arg),
        _ => return false,
    };
    let Some(def) = f.dfg().value_inst(arg) else {
        return false;
    };
    let (inner, input) = match f.dfg().inst(def) {
        InstView::PtrToInt { arg } => (Opcode::PtrToInt, arg),
        InstView::IntToPtr { arg } => (Opcode::IntToPtr, arg),
        _ => return false,
    };
    let result = f.dfg().first_result(inst).unwrap();
    if f.dfg().value_type(input) != f.dfg().value_type(result) {
        return false;
    }
    let pointer_bits = u32::from(layout.pointer_size) * 8;
    let exact = match (outer, inner) {
        (Opcode::IntToPtr, Opcode::PtrToInt) => f
            .dfg()
            .value_type(arg)
            .element_bits()
            .is_some_and(|bits| bits >= pointer_bits),
        (Opcode::PtrToInt, Opcode::IntToPtr) => f
            .dfg()
            .value_type(input)
            .element_bits()
            .is_some_and(|bits| bits <= pointer_bits),
        _ => false,
    };
    if !exact {
        return false;
    }
    work.replace(f, result, input);
    f.edit().erase_inst(inst);
    true
}

fn fold_instruction(func: &mut FuncBody, inst: Inst, work: &mut Worklist) -> bool {
    if let InstView::Binary {
        opcode: Opcode::ISub,
        args: [lhs, rhs],
    } = func.dfg().inst(inst)
        && let Some(value) = func.dfg().as_scalar_const(*rhs)
    {
        let lhs = *lhs;
        let negated = Int::from_bits(value.ty(), value.to_bits().wrapping_neg()).unwrap();
        let rhs = func.edit().constant(negated.into());
        func.edit()
            .replace_inst(inst, |w| w.binary(Opcode::IAdd, [lhs, rhs]));
        work.instruction(inst);
        return true;
    }
    if let Some(access) = inst.memory_access(func.dfg())
        && let Some(address) = func.dfg().value_inst(access.ptr)
        && let InstView::PtrOffset { ptr, offset } = func.dfg().inst(address)
        && let Ok(combined) = u32::try_from(access.offset + i64::from(offset))
    {
        match func.dfg().inst(inst) {
            InstView::Load { flags, .. } => {
                func.edit()
                    .replace_inst(inst, |w| w.load(ptr, combined, flags));
            }
            InstView::Store { value, flags, .. } => {
                func.edit()
                    .replace_inst(inst, |w| w.store(ptr, value, combined, flags));
            }
            _ => return false,
        }
        work.instruction(inst);
        return true;
    }
    if let InstView::PtrIndex { ptr, index, imm_id } = func.dfg().inst(inst)
        && let Some(index) = func.dfg().as_scalar_const(index)
    {
        let offset = index
            .to_bits()
            .wrapping_mul(imm_id.scale as u64)
            .wrapping_add(imm_id.offset as u64) as i64;
        if let Ok(offset) = i32::try_from(offset) {
            func.edit()
                .replace_inst(inst, |w| w.ptr_offset(ptr, offset));
            work.instruction(inst);
            return true;
        }
    }
    if let InstView::PtrOffset { ptr, offset: 0 } = func.dfg().inst(inst) {
        let result = func.dfg().first_result(inst).unwrap();
        work.replace(func, result, ptr);
        func.edit().erase_inst(inst);
        return true;
    }
    if fold_casts_and_ranges(func, inst, work) {
        return true;
    }
    // Boolean materialization is common at language boundaries. Keep the
    // condition as SSA rather than turning it into control flow and back.
    if let veloc_mir::InstView::Ternary {
        opcode: veloc_mir::Opcode::Select,
        args: [condition, yes, no],
    } = func.dfg().inst(inst)
        && func
            .dfg()
            .as_scalar_const(*yes)
            .is_some_and(|v| v.to_bits() == 1)
        && func
            .dfg()
            .as_scalar_const(*no)
            .is_some_and(|v| v.to_bits() == 0)
    {
        let condition = *condition;
        let result = func.dfg().first_result(inst).unwrap();
        let ty = func.dfg().value_type(result);
        if ty == Type::BOOL {
            work.replace(func, result, condition);
            func.edit().erase_inst(inst);
            return true;
        } else if ty.is_integer() {
            func.edit()
                .replace_inst(inst, |w| w.unary(veloc_mir::Opcode::ExtendU, condition));
            for site in func.dfg().uses(result) {
                work.instruction(site.inst());
            }
            return true;
        }
    }
    if let veloc_mir::InstView::IntCompare {
        kind: veloc_mir::IntCC::Ne,
        args: [value, zero],
    } = func.dfg().inst(inst)
        && func
            .dfg()
            .as_scalar_const(*zero)
            .is_some_and(|v| v.to_bits() == 0)
        && let Some(source) = func.dfg().value_inst(*value)
        && let veloc_mir::InstView::Unary {
            opcode: veloc_mir::Opcode::ExtendU | veloc_mir::Opcode::ExtendS,
            arg,
        } = func.dfg().inst(source)
        && func.dfg().value_type(arg) == Type::BOOL
    {
        let result = func.dfg().first_result(inst).unwrap();
        work.replace(func, result, arg);
        func.edit().erase_inst(inst);
        return true;
    }
    let dfg = func.dfg();
    if !evaluate::can_reduce(dfg, inst) {
        return false;
    }
    let args: SmallVec<[Value; 3]> = dfg.operands(inst).into();
    let results: SmallVec<[Value; 2]> = dfg.inst_results(inst).into();
    let types: SmallVec<[Type; 2]> = results.iter().map(|&v| dfg.value_type(v)).collect();
    let view = dfg.inst(inst);
    let Some(folds) = evaluate::reduce(
        view.opcode(),
        &args,
        &types,
        &evaluate::properties(&view),
        |value| dfg.as_scalar_const(value),
    ) else {
        return false;
    };
    assert_eq!(folds.len(), results.len(), "fold result arity");
    for (value, fold) in results.into_iter().zip(folds) {
        let replacement = match fold {
            evaluate::Fold::Operand(index) => args[index],
            evaluate::Fold::Constant(value) => func.edit().constant(value.into()),
        };
        work.replace(func, value, replacement);
    }
    // Successful evaluation proves the operation cannot trap for these inputs.
    func.edit().erase_inst(inst);
    true
}

/// Trace only conversions that preserve the exact integer values zero and one.
fn boolean_value(f: &FuncBody, mut value: Value) -> Option<Value> {
    loop {
        if f.dfg().value_type(value) == Type::BOOL {
            return Some(value);
        }
        let source = f.dfg().value_inst(value)?;
        match f.dfg().inst(source) {
            InstView::Unary {
                opcode: Opcode::ExtendU | Opcode::Wrap,
                arg,
            } => value = arg,
            _ => return None,
        }
    }
}
fn fold_casts_and_ranges(f: &mut FuncBody, inst: Inst, work: &mut Worklist) -> bool {
    let view = f.dfg().inst(inst);
    let Some(result) = f.dfg().first_result(inst) else {
        return false;
    };
    let ty = f.dfg().value_type(result);
    // Equality survives modular subtraction. Ordered signed comparisons also
    // do when both operands were sign-extended from narrower integers: their
    // difference fits in the wider signed type, so overflow cannot change order.
    if let InstView::IntCompare {
        kind,
        args: [difference, zero],
    } = view
        && f.dfg()
            .as_scalar_const(*zero)
            .is_some_and(|c| c.to_bits() == 0)
        && let Some(source) = f.dfg().value_inst(*difference)
        && let InstView::Binary {
            opcode: Opcode::ISub,
            args: [left, right],
        } = f.dfg().inst(source)
    {
        let wide = f.dfg().value_type(*difference).element_bits().unwrap();
        let extended = |v: Value| {
            f.dfg().value_inst(v).is_some_and(|i| {
                matches!(f.dfg().inst(i), InstView::Unary { opcode: Opcode::ExtendS, arg }
                if f.dfg().value_type(arg).element_bits().is_some_and(|bits| bits < wide))
            })
        };
        if matches!(kind, IntCC::Eq | IntCC::Ne)
            || (kind.is_signed() && extended(*left) && extended(*right))
        {
            let args = [*left, *right];
            f.edit().replace_inst(inst, |w| w.int_compare(kind, args));
            work.instruction(inst);
            return true;
        }
    }
    if let InstView::Ternary {
        opcode: Opcode::Select,
        args: [condition, yes, no],
    } = view
    {
        let unwrap = |value: Value| match f.dfg().inst(f.dfg().value_inst(value)?) {
            InstView::Unary {
                opcode: Opcode::Wrap,
                arg,
            } => Some(arg),
            _ => None,
        };
        if let Some(input) = unwrap(*yes).or_else(|| unwrap(*no)) {
            let wide = f.dfg().value_type(input);
            let arms = [*yes, *no].map(|v| {
                if let Some(input) = unwrap(v).filter(|&input| f.dfg().value_type(input) == wide) {
                    Some(Ok(input))
                } else {
                    f.dfg().as_scalar_const(v).map(|c| Err(c.to_bits()))
                }
            });
            if let [Some(a), Some(b)] = arms {
                let condition = *condition;
                let mut widen = |v| match v {
                    Ok(v) => v,
                    Err(bits) => f
                        .edit()
                        .constant(Int::from_bits(wide, bits).unwrap().into()),
                };
                let a = widen(a);
                let b = widen(b);
                // Truncation distributes over selection, irrespective of the
                // discarded bits. Keep the comparison in its original width.
                let select = f.edit().insert_before(
                    inst,
                    |w| w.ternary(Opcode::Select, [condition, a, b]),
                    &[wide],
                );
                let selected = f.dfg().first_result(select).unwrap();
                f.edit()
                    .replace_inst(inst, |w| w.unary(Opcode::Wrap, selected));
                work.instruction(select);
                work.instruction(inst);
                return true;
            }
        }
    }
    if let InstView::Unary {
        opcode: Opcode::Wrap,
        arg,
    } = view
        && let Some(source) = f.dfg().value_inst(arg)
        && let InstView::Unary {
            opcode: op @ (Opcode::ExtendU | Opcode::ExtendS),
            arg: input,
        } = f.dfg().inst(source)
    {
        let from = f.dfg().value_type(input);
        if ty == from {
            work.replace(f, result, input);
            f.edit().erase_inst(inst);
            return true;
        }
        let op = if ty.element_bits() > from.element_bits() {
            op
        } else {
            Opcode::Wrap
        };
        f.edit().replace_inst(inst, |w| w.unary(op, input));
        work.instruction(inst);
        for site in f.dfg().uses(result) {
            work.instruction(site.inst());
        }
        return true;
    }
    let predicate = match view {
        InstView::IntCompare {
            kind: kind @ (IntCC::Eq | IntCC::Ne),
            args: [v, z],
        } if f
            .dfg()
            .as_scalar_const(*z)
            .is_some_and(|c| c.to_bits() == 0) =>
        {
            Some((*v, kind == IntCC::Eq))
        }
        InstView::Unary {
            opcode: Opcode::IEqz,
            arg,
        } => Some((arg, true)),
        _ => None,
    };
    // A zero test observes only whether the selected bits are set. Test their
    // original positions instead of shifting them down first. Require sole
    // users so the replacement actually removes the shift/mask computation.
    if let Some((value, _)) = predicate
        && f.dfg().uses(value).count() == 1
        && let Some(masked) = f.dfg().value_inst(value)
        && let InstView::Binary {
            opcode: Opcode::IAnd,
            args: [a, b],
        } = f.dfg().inst(masked)
    {
        for (shifted, mask) in [(*a, *b), (*b, *a)] {
            let Some(mask) = f.dfg().as_scalar_const(mask) else {
                continue;
            };
            let Some(shift) = f.dfg().value_inst(shifted) else {
                continue;
            };
            if f.dfg().uses(shifted).count() != 1 {
                continue;
            }
            let InstView::Binary {
                opcode: Opcode::IShrS | Opcode::IShrU,
                args: [input, amount],
            } = f.dfg().inst(shift)
            else {
                continue;
            };
            let Some(amount) = f.dfg().as_scalar_const(*amount) else {
                continue;
            };
            let input = *input;
            let input_ty = f.dfg().value_type(input);
            let bits = input_ty.element_bits().unwrap();
            let amount = amount.to_bits() % u64::from(bits);
            let full = u64::MAX >> (64 - bits);
            if amount == 0 || mask.to_bits() & !(full >> amount) != 0 {
                continue;
            }
            let mask = f.edit().constant(
                Int::from_bits(input_ty, mask.to_bits() << amount)
                    .unwrap()
                    .into(),
            );
            let test = f.edit().insert_before(
                inst,
                |w| w.binary(Opcode::IAnd, [input, mask]),
                &[input_ty],
            );
            work.instruction(test);
            let test = f.dfg().first_result(test).unwrap();
            f.edit().set_operand(inst, 0, test);
            work.instruction(inst);
            return true;
        }
    }
    if let Some((value, invert)) = predicate
        && let Some(boolean) = boolean_value(f, value)
    {
        if invert {
            let zero = f.edit().constant(
                veloc_mir::ScalarConst::from_bits(Type::BOOL, 0)
                    .unwrap()
                    .into(),
            );
            let one = f.edit().constant(
                veloc_mir::ScalarConst::from_bits(Type::BOOL, 1)
                    .unwrap()
                    .into(),
            );
            f.edit()
                .replace_inst(inst, |w| w.ternary(Opcode::Select, [boolean, zero, one]));
        } else {
            work.replace(f, result, boolean);
            f.edit().erase_inst(inst);
        }
        for site in f.dfg().uses(result) {
            work.instruction(site.inst());
        }
        return true;
    }
    // [low,high] is a contiguous interval in either signed or unsigned order.
    // Subtraction maps it to [0,high-low], tested using unsigned comparison.
    if let InstView::Binary {
        opcode: Opcode::IAnd,
        args: [a, b],
    } = view
    {
        let bound = |v| {
            let condition = boolean_value(f, v)?;
            let i = f.dfg().value_inst(condition)?;
            let InstView::IntCompare {
                kind,
                args: [value, bound],
            } = f.dfg().inst(i)
            else {
                return None;
            };
            let literal = f.dfg().as_scalar_const(*bound)?;
            Some((kind, *value, literal.to_bits(), *bound))
        };
        if let (Some(a), Some(b)) = (bound(*a), bound(*b)) {
            for (lower, upper) in [(a, b), (b, a)] {
                let signed = lower.0 == IntCC::GeS && upper.0 == IntCC::LeS;
                if !(signed || lower.0 == IntCC::GeU && upper.0 == IntCC::LeU) || lower.1 != upper.1
                {
                    continue;
                }
                let input_ty = f.dfg().value_type(lower.1);
                let bits = input_ty.element_bits().unwrap();
                let ordered = if signed {
                    IntCC::LeS.test(bits as u16, lower.2 as u128, upper.2 as u128)
                } else {
                    lower.2 <= upper.2
                };
                if !ordered {
                    continue;
                }
                let span = f.edit().constant(
                    Int::from_bits(input_ty, upper.2.wrapping_sub(lower.2))
                        .unwrap()
                        .into(),
                );
                let sub = f.edit().insert_before(
                    inst,
                    |w| w.binary(Opcode::ISub, [lower.1, lower.3]),
                    &[input_ty],
                );
                let sub = f.dfg().first_result(sub).unwrap();
                let compare = f.edit().insert_before(
                    inst,
                    |w| w.int_compare(IntCC::LeU, [sub, span]),
                    &[Type::BOOL],
                );
                let compare = f.dfg().first_result(compare).unwrap();
                f.edit()
                    .replace_inst(inst, |w| w.unary(Opcode::ExtendU, compare));
                work.instruction(inst);
                for site in f.dfg().uses(result) {
                    work.instruction(site.inst());
                }
                return true;
            }
        }
    }
    false
}

enum Item {
    Instruction(Inst),
    Parameters(Block),
}

struct Worklist {
    pending: VecDeque<Item>,
    instructions: SecondaryMap<Inst, bool>,
    blocks: SecondaryMap<Block, bool>,
    incoming: SecondaryMap<Block, Vec<EdgeRef>>,
}

impl Worklist {
    fn new(func: &FuncBody) -> Self {
        let mut work = Self {
            pending: VecDeque::new(),
            instructions: SecondaryMap::new(),
            blocks: SecondaryMap::new(),
            incoming: SecondaryMap::new(),
        };
        for block in func.layout().block_order() {
            work.block(block);
            for inst in func.layout().block_insts(block) {
                work.instruction(inst);
                let mut index = 0;
                func.dfg().inst(inst).visit_successors(|edge| {
                    work.incoming[edge.block].push(EdgeRef { inst, index });
                    index += 1;
                });
            }
        }
        work
    }

    fn instruction(&mut self, inst: Inst) {
        if !self.instructions[inst] {
            self.instructions[inst] = true;
            self.pending.push_back(Item::Instruction(inst));
        }
    }

    fn block(&mut self, block: Block) {
        if !self.blocks[block] {
            self.blocks[block] = true;
            self.pending.push_back(Item::Parameters(block));
        }
    }

    fn pop(&mut self) -> Option<Item> {
        let item = self.pending.pop_front()?;
        match item {
            Item::Instruction(inst) => self.instructions[inst] = false,
            Item::Parameters(block) => self.blocks[block] = false,
        }
        Some(item)
    }

    fn replace(&mut self, func: &mut FuncBody, old: Value, new: Value) {
        for site in func.dfg().uses(old) {
            self.instruction(site.inst());
        }
        func.edit().replace_all_uses(old, new);
    }
}
