//! Monotone known-bit propagation through values, block arguments and calls.
use super::{constant, shift, width_mask};
use cranelift_entity::SecondaryMap;
use std::collections::VecDeque;
use veloc_mir::{FuncBody, Inst, Opcode, Value};

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct KnownBits {
    pub(super) zero: u64,
    pub(super) one: u64,
}

pub(super) fn fact(
    f: &FuncBody,
    facts: &SecondaryMap<Value, KnownBits>,
    value: Value,
) -> KnownBits {
    match constant(f, value) {
        Some(bits) => KnownBits {
            zero: !bits & width_mask(f, value),
            one: bits,
        },
        None => facts[value],
    }
}

pub(super) fn known_bits(f: &FuncBody, insts: &[Inst]) -> SecondaryMap<Value, KnownBits> {
    known_bits_with_returns(f, insts, &SecondaryMap::new())
}

pub(super) fn known_bits_with_returns(
    f: &FuncBody,
    insts: &[Inst],
    returns: &SecondaryMap<veloc_mir::FuncId, Vec<KnownBits>>,
) -> SecondaryMap<Value, KnownBits> {
    let mut facts = SecondaryMap::<Value, KnownBits>::new();
    let mut incoming = SecondaryMap::<Value, Vec<Value>>::new();
    for &inst in insts {
        f.dfg().inst(inst).visit_successors(|edge| {
            for (&param, &arg) in f.dfg().block_params(edge.block).iter().zip(edge.args) {
                incoming[param].push(arg);
            }
        });
    }
    let mut pending: VecDeque<_> = insts.iter().copied().collect();
    let mut queued = SecondaryMap::<Inst, bool>::new();
    for &inst in insts {
        queued[inst] = true;
    }
    while let Some(inst) = pending.pop_front() {
        queued[inst] = false;
        f.dfg().inst(inst).visit_successors(|edge| {
            for &param in f.dfg().block_params(edge.block) {
                let mask = width_mask(f, param);
                if mask == 0 {
                    continue;
                }
                let mut known = KnownBits {
                    zero: mask,
                    one: mask,
                };
                for &arg in &incoming[param] {
                    let input = fact(f, &facts, arg);
                    known.zero &= input.zero;
                    known.one &= input.one;
                }
                update_fact(f, param, known, &mut facts, &mut pending, &mut queued);
            }
        });
        if let veloc_mir::InstView::Call { func_id, .. } = f.dfg().inst(inst) {
            for (&result, &known) in f.dfg().inst_results(inst).iter().zip(&returns[func_id]) {
                update_fact(f, result, known, &mut facts, &mut pending, &mut queued);
            }
            continue;
        }
        let Some(result) = f.dfg().first_result(inst) else {
            continue;
        };
        let mask = width_mask(f, result);
        if mask == 0 {
            continue;
        }
        let args = f.dfg().operands(inst);
        let get = |index| fact(f, &facts, args[index]);
        let opcode = f.dfg().opcode(inst);
        let mut known = match opcode {
            Opcode::IAnd => KnownBits {
                zero: get(0).zero | get(1).zero,
                one: get(0).one & get(1).one,
            },
            Opcode::IOr => KnownBits {
                zero: get(0).zero & get(1).zero,
                one: get(0).one | get(1).one,
            },
            Opcode::IXor => KnownBits {
                zero: (get(0).zero & get(1).zero) | (get(0).one & get(1).one),
                one: (get(0).zero & get(1).one) | (get(0).one & get(1).zero),
            },
            Opcode::Select => KnownBits {
                zero: get(1).zero & get(2).zero,
                one: get(1).one & get(2).one,
            },
            Opcode::Wrap => get(0),
            Opcode::ExtendU => KnownBits {
                zero: get(0).zero | !width_mask(f, args[0]),
                one: get(0).one,
            },
            Opcode::ExtendS => {
                let input = width_mask(f, args[0]);
                let sign = (input >> 1) + 1;
                KnownBits {
                    zero: get(0).zero | if get(0).zero & sign != 0 { !input } else { 0 },
                    one: get(0).one | if get(0).one & sign != 0 { !input } else { 0 },
                }
            }
            Opcode::IShl | Opcode::IShrU | Opcode::IShrS => {
                let Some(amount) = shift(f, args, mask) else {
                    continue;
                };
                let source = get(0);
                if opcode == Opcode::IShl {
                    KnownBits {
                        zero: (source.zero << amount) | ((1u64 << amount) - 1),
                        one: source.one << amount,
                    }
                } else {
                    let high = mask & !(mask >> amount);
                    let sign = (mask >> 1) + 1;
                    KnownBits {
                        zero: (source.zero >> amount)
                            | if opcode == Opcode::IShrU || source.zero & sign != 0 {
                                high
                            } else {
                                0
                            },
                        one: (source.one >> amount)
                            | if opcode == Opcode::IShrS && source.one & sign != 0 {
                                high
                            } else {
                                0
                            },
                    }
                }
            }
            _ => continue,
        };
        known.zero &= mask;
        known.one &= mask;
        update_fact(f, result, known, &mut facts, &mut pending, &mut queued);
    }
    facts
}

fn update_fact(
    f: &FuncBody,
    value: Value,
    known: KnownBits,
    facts: &mut SecondaryMap<Value, KnownBits>,
    pending: &mut VecDeque<Inst>,
    queued: &mut SecondaryMap<Inst, bool>,
) {
    if known == facts[value] {
        return;
    }
    facts[value] = known;
    for user in f.dfg().uses(value) {
        let user = user.inst();
        if !queued[user] {
            queued[user] = true;
            pending.push_back(user);
        }
    }
}
