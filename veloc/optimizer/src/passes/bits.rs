//! Backward demanded bits and forward known bits for scalar integer operations.
//! Unknown operations observe every input bit. Forward facts meet at block
//! parameters; module summaries can additionally describe direct call results.
use crate::{FunctionPass, OptConfig, PassOutcome, Profile};
use cranelift_entity::SecondaryMap;
use std::collections::VecDeque;
use veloc_analyzer::AnalysisManager;
use veloc_mir::{FuncBody, Inst, Opcode, ScalarConst, Value};
use veloc_types::TypeInfo;
mod facts;
use facts::{KnownBits, fact, known_bits, known_bits_with_returns};
mod memory;
mod returns;
pub use returns::ReturnBitsPass;

pub struct BitsPass;

impl FunctionPass for BitsPass {
    fn name(&self) -> &'static str {
        "BitsPass"
    }

    fn run(
        &self,
        am: &mut AnalysisManager<'_>,
        config: &OptConfig,
        metrics: &Profile,
    ) -> PassOutcome {
        let f = am.function_mut();
        let mut insts: Vec<_> = f
            .layout()
            .block_order()
            .flat_map(|b| f.layout().block_insts(b))
            .collect();
        let mut changed = if config
            .data_layout
            .is_some_and(|layout| layout.little_endian)
        {
            memory::narrow_stores(f, &insts)
        } else {
            0
        };
        if changed != 0 {
            insts = f
                .layout()
                .block_order()
                .flat_map(|b| f.layout().block_insts(b))
                .collect();
        }
        let demanded = demands(f, &insts);
        for &inst in &insts {
            if let veloc_mir::InstView::Unary {
                opcode: Opcode::ExtendS,
                arg,
            } = f.dfg().inst(inst)
                && matches!(
                    f.dfg().value_type(arg),
                    veloc_mir::Type::I8 | veloc_mir::Type::I16
                )
                && let Some(result) = f.dfg().first_result(inst)
                && demanded[result] & !width_mask(f, arg) == 0
            {
                // The narrow value's sign copies are unobserved. Zero extension
                // allows unsigned loads to supply the value directly and gives
                // subsequent bitwise simplification precise high-zero facts.
                f.edit()
                    .replace_inst(inst, |w| w.unary(Opcode::ExtendU, arg));
                changed += 1;
            }
        }
        for &inst in &insts {
            let veloc_mir::InstView::Load { ptr, offset, flags } = f.dfg().inst(inst) else {
                continue;
            };
            if flags.is_volatile() || !flags.is_notrap() {
                continue;
            }
            let result = f.dfg().first_result(inst).unwrap();
            let width = width_mask(f, result);
            let used = demanded[result];
            if width == 0 || used == 0 {
                continue;
            }
            let narrowed = if used <= 0xff && width > 0xff {
                Some(veloc_mir::Type::I8)
            } else if used <= 0xffff && width > 0xffff {
                Some(veloc_mir::Type::I16)
            } else if used <= 0xffff_ffff && width > 0xffff_ffff {
                Some(veloc_mir::Type::I32)
            } else {
                None
            };
            let Some(ty) = narrowed else { continue };
            // Low bits occupy the first bytes only on little-endian targets.
            // The pipeline supplies the target layout; absent layout is unknown.
            if !config
                .data_layout
                .as_ref()
                .is_some_and(|layout| layout.little_endian)
            {
                continue;
            }
            let load = f
                .edit()
                .insert_before(inst, |w| w.load(ptr, offset, flags), &[ty]);
            let value = f.dfg().first_result(load).unwrap();
            f.edit()
                .replace_inst(inst, |w| w.unary(Opcode::ExtendU, value));
            changed += 1;
        }
        // Bitwise constants only need to preserve bits observed by the users.
        for &inst in &insts {
            if !matches!(
                f.dfg().opcode(inst),
                Opcode::IAnd | Opcode::IOr | Opcode::IXor
            ) {
                continue;
            }
            let Some(result) = f.dfg().first_result(inst) else {
                continue;
            };
            let mask = demanded[result];
            if width_mask(f, result) == 0 || mask == 0 {
                continue;
            }
            let operands = f.dfg().operands(inst).to_vec();
            for (index, value) in operands.into_iter().enumerate() {
                let Some(bits) = constant(f, value) else {
                    continue;
                };
                if bits & mask != bits {
                    let ty = f.dfg().value_type(value);
                    let replacement = f
                        .edit()
                        .constant(ScalarConst::from_bits(ty, bits & mask).unwrap().into());
                    f.edit().set_operand(inst, index as u32, replacement);
                    changed += 1;
                }
            }
        }
        let facts = known_bits(f, &insts);
        // Apply consumers first so replacing a producer also updates the new
        // uses created by its consumers' replacements.
        for &inst in insts.iter().rev() {
            if simplify_conversions(f, inst, &facts) {
                changed += 1;
                continue;
            }
            let opcode = f.dfg().opcode(inst);
            if !matches!(opcode, Opcode::IAnd | Opcode::IOr | Opcode::IXor) {
                continue;
            }
            let result = f.dfg().first_result(inst).unwrap();
            // Facts describe complete values. Preserve every bit here: removing
            // a mask only in demanded bits could invalidate a downstream fact
            // that depended on its otherwise unobserved zero bits.
            let demand = width_mask(f, result);
            if demand == 0 {
                continue;
            }
            let [lhs, rhs] = *f.dfg().operands(inst) else {
                unreachable!()
            };
            for (value, other) in [(lhs, rhs), (rhs, lhs)] {
                let a = fact(f, &facts, value);
                let b = fact(f, &facts, other);
                let unchanged = match opcode {
                    Opcode::IAnd => (a.zero | b.one) & demand == demand,
                    Opcode::IOr => (a.one | b.zero) & demand == demand,
                    Opcode::IXor => b.zero & demand == demand,
                    _ => unreachable!(),
                };
                if unchanged {
                    f.edit().replace_all_uses(result, value);
                    changed += 1;
                    break;
                }
            }
        }
        metrics.count("bits.simplified", changed);
        if changed == 0 {
            PassOutcome::Unchanged
        } else {
            PassOutcome::Changed
        }
    }
}

fn simplify_conversions(
    f: &mut FuncBody,
    inst: Inst,
    facts: &SecondaryMap<Value, KnownBits>,
) -> bool {
    use veloc_mir::{InstView, Int, IntCC};
    match f.dfg().inst(inst) {
        InstView::Unary {
            opcode: op @ (Opcode::ExtendS | Opcode::ExtendU),
            arg,
        } => {
            let Some(def) = f.dfg().value_inst(arg) else {
                return false;
            };
            if let InstView::Unary {
                opcode: inner @ (Opcode::ExtendS | Opcode::ExtendU),
                arg: original,
            } = f.dfg().inst(def)
                && (inner == op || inner == Opcode::ExtendU)
            {
                f.edit().replace_inst(inst, |w| w.unary(inner, original));
                return true;
            }
            let InstView::Unary {
                opcode: Opcode::Wrap,
                arg: original,
            } = f.dfg().inst(def)
            else {
                return false;
            };
            let result = f.dfg().first_result(inst).unwrap();
            let from = f.dfg().value_type(original);
            let to = f.dfg().value_type(result);
            if to.element_bits() < from.element_bits() {
                return false;
            }
            let narrow = width_mask(f, arg);
            let high = width_mask(f, original) & !narrow;
            let known = fact(f, facts, original);
            let sign = (narrow >> 1) + 1;
            let unchanged = if op == Opcode::ExtendU {
                known.zero & high == high
            } else {
                let required = high | sign;
                known.zero & required == required || known.one & required == required
            };
            if !unchanged {
                return false;
            }
            if from == to {
                f.edit().replace_results(inst, &[original]);
            } else {
                f.edit().replace_inst(inst, |w| w.unary(op, original));
            }
            true
        }
        InstView::IntCompare { kind, args } => {
            let (kind, args) = (kind, *args);
            for index in 0..2 {
                let Some(def) = f.dfg().value_inst(args[index]) else {
                    continue;
                };
                let InstView::Unary {
                    opcode: Opcode::Wrap,
                    arg: original,
                } = f.dfg().inst(def)
                else {
                    continue;
                };
                let Some(literal) = f.dfg().as_scalar_const(args[1 - index]) else {
                    continue;
                };
                // A nonnegative value fitting the signed narrow domain has the
                // same order after either signed or unsigned promotion.
                let required = width_mask(f, original) & !(width_mask(f, args[index]) >> 1);
                if fact(f, facts, original).zero & required != required {
                    continue;
                }
                let signed = matches!(kind, IntCC::LtS | IntCC::LeS | IntCC::GtS | IntCC::GeS);
                let literal = Int::from_bits(literal.ty(), literal.to_bits()).unwrap();
                let bits = if signed {
                    literal.signed() as u64
                } else {
                    literal.to_bits()
                };
                let literal = Int::from_bits(f.dfg().value_type(original), bits).unwrap();
                let other = f.edit().constant(literal.into());
                let args = if index == 0 {
                    [original, other]
                } else {
                    [other, original]
                };
                f.edit().replace_inst(inst, |w| w.int_compare(kind, args));
                return true;
            }
            false
        }
        _ => false,
    }
}

fn width_mask(f: &FuncBody, value: Value) -> u64 {
    let ty = f.dfg().value_type(value);
    if ty.as_scalar().is_none() || !(ty.is_integer() || ty == veloc_mir::Type::BOOL) {
        return 0;
    }
    match ty.element_bits() {
        Some(bits @ 1..=64) => u64::MAX >> (64 - bits),
        _ => 0,
    }
}

fn constant(f: &FuncBody, value: Value) -> Option<u64> {
    (width_mask(f, value) != 0)
        .then(|| f.dfg().as_scalar_const(value).map(|c| c.to_bits()))
        .flatten()
}

fn shift(f: &FuncBody, args: &[Value], mask: u64) -> Option<u32> {
    let width = 64 - mask.leading_zeros();
    Some((constant(f, *args.get(1)?)? % u64::from(width)) as u32)
}

fn demands(f: &FuncBody, insts: &[Inst]) -> SecondaryMap<Value, u64> {
    let mut demanded = SecondaryMap::<Value, u64>::new();
    let mut pending: VecDeque<_> = insts.iter().rev().copied().collect();
    let mut queued = SecondaryMap::<Inst, bool>::new();
    for &inst in insts {
        queued[inst] = true;
    }
    while let Some(inst) = pending.pop_front() {
        queued[inst] = false;
        let args = f.dfg().operands(inst);
        let result = f.dfg().first_result(inst);
        let mask = result.map_or(0, |v| width_mask(f, v));
        let out = result.map_or(0, |v| demanded[v]);
        let opcode = f.dfg().opcode(inst);
        for (index, &value) in args.iter().enumerate() {
            let input = width_mask(f, value);
            if input == 0 {
                continue;
            }
            let bits = if mask == 0 {
                input
            } else {
                match opcode {
                    Opcode::IAnd => out & constant(f, args[1 - index]).unwrap_or(input),
                    Opcode::IOr => out & !constant(f, args[1 - index]).unwrap_or(0),
                    Opcode::IXor => out,
                    Opcode::Select if index != 0 => out,
                    Opcode::IAdd | Opcode::ISub | Opcode::IMul => {
                        if out == 0 {
                            0
                        } else {
                            u64::MAX >> out.leading_zeros()
                        }
                    }
                    Opcode::IShl | Opcode::IShrU | Opcode::IShrS if index == 0 => {
                        if let Some(amount) = shift(f, args, mask) {
                            match opcode {
                                Opcode::IShl => out >> amount,
                                Opcode::IShrU => out << amount,
                                Opcode::IShrS => {
                                    (out << amount)
                                        | if out & !(mask >> amount) != 0 {
                                            (mask >> 1) + 1
                                        } else {
                                            0
                                        }
                                }
                                _ => unreachable!(),
                            }
                        } else {
                            input
                        }
                    }
                    Opcode::Wrap | Opcode::ExtendU => out,
                    Opcode::ExtendS => {
                        out | if out & !input != 0 {
                            (input >> 1) + 1
                        } else {
                            0
                        }
                    }
                    _ => input,
                }
            } & input;
            if demanded[value] | bits == demanded[value] {
                continue;
            }
            demanded[value] |= bits;
            if let Some(def) = f.dfg().value_inst(value) {
                if !queued[def] {
                    queued[def] = true;
                    pending.push_back(def);
                }
            }
        }
    }
    demanded
}
