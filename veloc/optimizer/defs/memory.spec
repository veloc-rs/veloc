import "../../defs/type_sets.spec";
import "../../mir/defs/formats.spec";

rule PtrOffset {
    case (ptr, 0) => ptr;
}

fn fits_u32(value: i64) -> bool {
    value = value >= 0 && value <= 0xffffffff;
}

// Preserve the memory operation and its existing SSA results.
rule Load {
    case (PtrOffset(ptr, inner), outer, flags)
        if fits_u32(i64(inner) + i64(outer))
        => Load(ptr, wrapping_cast(i64(inner) + i64(outer), u32), flags);
}

rule Store {
    case (PtrOffset(ptr, inner), value, outer, flags)
        if fits_u32(i64(inner) + i64(outer))
        => Store(ptr, value, wrapping_cast(i64(inner) + i64(outer), u32), flags);
}

// Keep the existing 64-bit modular address arithmetic. Widen first so the
// multiplication itself cannot overflow, then interpret the low 64 bits signed.
fn index_offset(index: u64, imm: PtrIndexImm) -> optional(i32) {
    value = checked_cast(
        wrapping_cast(i128(index) * i128(imm.scale) + i128(imm.offset), i64),
        i32);
}
rule PtrIndex {
    case (ptr, index, imm)
        => PtrOffset(ptr, index_offset(constant_bits(index), imm)?);
}
