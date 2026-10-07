import "../../mir/defs/formats.spec";

// Directed replacements preserve the root's position, SSA results and memory
// operation. Failed conversions leave the original instruction untouched.
rule(root: mir::Load) {
    case root(mir::PtrOffset(ptr, inner), outer, flags)
        => mir::Load(ptr, checked_cast(i64(inner) + i64(outer), u32)?, flags);
}
rule(root: mir::Store) {
    case root(mir::PtrOffset(ptr, inner), value, outer, flags)
        => mir::Store(ptr, value, checked_cast(i64(inner) + i64(outer), u32)?, flags);
}

// Keep the existing 64-bit modular address arithmetic. Widen first so the
// multiplication itself cannot overflow, then interpret the low 64 bits signed.
fn index_offset(index: u64, imm: PtrIndexImm) -> optional(i32) {
    value = checked_cast(
        wrapping_cast(i128(index) * i128(imm.scale) + i128(imm.offset), i64),
        i32);
}
rule(root: mir::PtrIndex) {
    case root(ptr, index, imm)
        => mir::PtrOffset(ptr, index_offset(constant_bits(index), imm)?);
}
