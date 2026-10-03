//! Narrow read/modify/write updates when the omitted bytes are unchanged.
use super::{constant, fact, known_bits, width_mask};
use veloc_mir::{FuncBody, Inst, InstView, Opcode, Type, TypeInfo, Value};

/// Strip casts that retain the requested low bits.
fn low_bits(f: &FuncBody, mut value: Value, bits: u32) -> Value {
    while let Some(inst) = f.dfg().value_inst(value) {
        if let InstView::Unary {
            opcode: Opcode::Wrap | Opcode::ExtendU | Opcode::ExtendS,
            arg,
        } = f.dfg().inst(inst)
            && f.dfg()
                .value_type(arg)
                .element_bits()
                .is_some_and(|n| n >= bits)
        {
            value = arg;
        } else {
            break;
        }
    }
    value
}

pub(super) fn narrow_stores(f: &mut FuncBody, insts: &[Inst]) -> u64 {
    let facts = known_bits(f, insts);
    let mut changed = 0;
    for &store in insts {
        let InstView::Store {
            ptr,
            offset,
            value,
            flags,
        } = f.dfg().inst(store)
        else {
            continue;
        };
        if flags.is_volatile() || !flags.is_notrap() {
            continue;
        }
        let width = width_mask(f, value);
        if width <= 0xff {
            continue;
        }
        let bits = f.dfg().value_type(value).element_bits().unwrap();
        let root = low_bits(f, value, bits);
        let Some(root) = f.dfg().value_inst(root) else {
            continue;
        };
        let InstView::Binary {
            opcode: Opcode::IOr,
            args: [a, b],
        } = f.dfg().inst(root)
        else {
            continue;
        };
        for (preserved, updated) in [(*a, *b), (*b, *a)] {
            let Some(and) = f.dfg().value_inst(preserved) else {
                continue;
            };
            let InstView::Binary {
                opcode: Opcode::IAnd,
                args: [a, b],
            } = f.dfg().inst(and)
            else {
                continue;
            };
            let (original, mask) = if let Some(mask) = constant(f, *b) {
                (*a, mask)
            } else if let Some(mask) = constant(f, *a) {
                (*b, mask)
            } else {
                continue;
            };
            let original = low_bits(f, original, bits);
            if width_mask(f, original) != width {
                continue;
            }
            let Some(load) = f.dfg().value_inst(original) else {
                continue;
            };
            let InstView::Load {
                ptr: p,
                offset: o,
                flags: read_flags,
            } = f.dfg().inst(load)
            else {
                continue;
            };
            if p != ptr || o != offset || read_flags.is_volatile() || !read_flags.is_notrap() {
                continue;
            }
            // An intervening write could have changed an omitted byte. The
            // original wide store would restore it; a narrow store would not.
            if f.layout().inst_block(load) != f.layout().inst_block(store) {
                continue;
            }
            let mut cursor = f.layout().next_inst(load);
            while let Some(inst) = cursor {
                if inst == store {
                    break;
                }
                let view = f.dfg().inst(inst);
                if view.has_volatile_access()
                    || view.memory_effect().may_write()
                    || view.memory_effect().may_free()
                {
                    break;
                }
                cursor = f.layout().next_inst(inst);
            }
            if cursor != Some(store) {
                continue;
            }
            let narrow = [Type::I8, Type::I16, Type::I32].into_iter().find(|&ty| {
                let low = u64::MAX >> (64 - ty.element_bits().unwrap());
                let high = width & !low;
                low < width && mask & width == high && fact(f, &facts, updated).zero & high == high
            });
            let Some(ty) = narrow else { continue };
            let replacement = extract_load(f, updated, ty).unwrap_or_else(|| {
                if f.dfg().value_type(updated) == ty {
                    return updated;
                }
                let cast = f
                    .edit()
                    .insert_before(store, |w| w.unary(Opcode::Wrap, updated), &[ty]);
                f.dfg().first_result(cast).unwrap()
            });
            f.edit()
                .replace_inst(store, |w| w.store(ptr, replacement, offset, flags));
            changed += 1;
            break;
        }
    }
    changed
}

/// Reading an aligned byte slice at the original load's position avoids a
/// wider read followed by shifts. Other users retain the original read.
fn extract_load(f: &mut FuncBody, value: Value, ty: Type) -> Option<Value> {
    let bits = ty.element_bits()?;
    let mask = u64::MAX >> (64 - bits);
    let mut value = low_bits(f, value, bits);
    if let Some(inst) = f.dfg().value_inst(value)
        && let InstView::Binary {
            opcode: Opcode::IAnd,
            args: [a, b],
        } = f.dfg().inst(inst)
    {
        if constant(f, *b).is_some_and(|c| c & mask == mask) {
            value = *a;
        } else if constant(f, *a).is_some_and(|c| c & mask == mask) {
            value = *b;
        }
    }
    let inst = f.dfg().value_inst(value)?;
    let InstView::Binary {
        opcode: Opcode::IShrU | Opcode::IShrS,
        args: [input, amount],
    } = f.dfg().inst(inst)
    else {
        return None;
    };
    let amount =
        (constant(f, *amount)? % u64::from(f.dfg().value_type(*input).element_bits()?)) as u32;
    if amount % 8 != 0 {
        return None;
    }
    let input = low_bits(f, *input, amount + bits);
    if f.dfg().value_type(input).element_bits()? < amount + bits {
        return None;
    }
    let inst = f.dfg().value_inst(input)?;
    let InstView::Load { ptr, offset, flags } = f.dfg().inst(inst) else {
        return None;
    };
    if flags.is_volatile() || !flags.is_notrap() {
        return None;
    }
    let offset = offset.checked_add(amount / 8)?;
    let flags = flags.with_alignment(1);
    let load = f
        .edit()
        .insert_before(inst, |w| w.load(ptr, offset, flags), &[ty]);
    f.dfg().first_result(load)
}
