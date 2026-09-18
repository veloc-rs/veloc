import "predicates.spec";

extractor GPR8 { params = [reg]; pattern = is_i8(reg); }
extractor GPR16 { params = [reg]; pattern = is_i16(reg); }
extractor GPR32 { params = [reg]; pattern = is_i32(reg); }
extractor GPR64 { params = [reg]; pattern = is_i64(reg); }
extractor GPTR { params = [reg]; pattern = is_ptr(reg); }
extractor FPR32 { params = [reg]; pattern = is_f32(reg); }
extractor FPR64 { params = [reg]; pattern = is_f64(reg); }
extractor GPR32Like { params = [reg]; pattern = is_int32like(reg); }
extractor GPR64Like { params = [reg]; pattern = is_64like(reg); }
extractor BOOL { params = [reg]; pattern = is_bool(reg); }
select rule_24 {
    match = [bind(n, Constant::Constant { dst: GPR32Like(unused), imm: imm })];
    emit = X86Mov32Imm(imm);
    covers = [n];
    cost = 1;
}
select rule_25 {
    match = [bind(n, Constant::Constant { dst: GPR64(unused), imm: imm })];
    emit = X86Mov64Imm64(imm);
    covers = [n];
    cost = 1;
}
select rule_26 {
    match = [bind(n, UnaryReg::Copy { dst: GPR32(dst), src: GPR32(src) })];
    emit = X86Mov32(src);
    covers = [n];
    cost = 1;
}
select rule_27 {
    match = [bind(n, UnaryReg::Copy { dst: GPR64(dst), src: GPR64(src) })];
    emit = X86Mov64(src);
    covers = [n];
    cost = 1;
}
select rule_28 {
    match = [bind(n, UnaryReg::Copy { dst: GPTR(dst), src: GPTR(src) })];
    emit = X86Mov64(src);
    covers = [n];
    cost = 1;
}
select rule_29 {
    match = [bind(n, UnaryReg::Copy { dst: GPR64(dst), src: GPTR(src) })];
    emit = X86Mov64(src);
    covers = [n];
    cost = 1;
}
select rule_30 {
    match = [bind(n, UnaryReg::Copy { dst: GPTR(dst), src: GPR64(src) })];
    emit = X86Mov64(src);
    covers = [n];
    cost = 1;
}
select rule_31 {
    match = [bind(n, UnaryReg::Copy { dst: FPR32(dst), src: FPR32(src) })];
    emit = X86Movss(src);
    covers = [n];
    cost = 1;
}
select rule_32 {
    match = [bind(n, UnaryReg::Copy { dst: FPR64(dst), src: FPR64(src) })];
    emit = X86Movsd(src);
    covers = [n];
    cost = 1;
}
select rule_33 {
    match = [bind(n, UnaryReg::Inttoptr { dst: GPTR(dst), src: GPR64(src) })];
    emit = X86Mov64(src);
    covers = [n];
    cost = 1;
}
select rule_34 {
    match = [bind(n, UnaryReg::Ptrtoint { dst: GPR64(dst), src: GPTR(src) })];
    emit = X86Mov64(src);
    covers = [n];
    cost = 1;
}
select rule_35 {
    match = [bind(n, UnaryReg::Bitcast { dst: GPR32(dst), src: GPR32(src) })];
    emit = X86Mov32(src);
    covers = [n];
    cost = 1;
}
select rule_36 {
    match = [bind(n, UnaryReg::Bitcast { dst: GPR64(dst), src: GPR64(src) })];
    emit = X86Mov64(src);
    covers = [n];
    cost = 1;
}
select rule_37 {
    match = [bind(n, UnaryReg::Bitcast { dst: GPTR(dst), src: GPTR(src) })];
    emit = X86Mov64(src);
    covers = [n];
    cost = 1;
}
select rule_38 {
    match = [bind(n, UnaryReg::Bitcast { dst: GPR64(dst), src: GPTR(src) })];
    emit = X86Mov64(src);
    covers = [n];
    cost = 1;
}
select rule_39 {
    match = [bind(n, UnaryReg::Bitcast { dst: GPTR(dst), src: GPR64(src) })];
    emit = X86Mov64(src);
    covers = [n];
    cost = 1;
}
select rule_40 {
    match = [bind(n, UnaryReg::Bitcast { dst: FPR32(dst), src: FPR32(src) })];
    emit = X86Movss(src);
    covers = [n];
    cost = 1;
}
select rule_41 {
    match = [bind(n, UnaryReg::Bitcast { dst: FPR64(dst), src: FPR64(src) })];
    emit = X86Movsd(src);
    covers = [n];
    cost = 1;
}
select rule_42 {
    match = [bind(n, UnaryReg::Bitcast { dst: FPR32(dst), src: GPR32(src) })];
    emit = X86MovdToXmm(src);
    covers = [n];
    cost = 1;
}
select rule_43 {
    match = [bind(n, UnaryReg::Bitcast { dst: GPR32(dst), src: FPR32(src) })];
    emit = X86MovdFromXmm(src);
    covers = [n];
    cost = 1;
}
select rule_44 {
    match = [bind(n, UnaryReg::Bitcast { dst: FPR64(dst), src: GPR64(src) })];
    emit = X86MovqToXmm(src);
    covers = [n];
    cost = 1;
}
select rule_45 {
    match = [bind(n, UnaryReg::Bitcast { dst: GPR64(dst), src: FPR64(src) })];
    emit = X86MovqFromXmm(src);
    covers = [n];
    cost = 1;
}
select rule_46 {
    match = [bind(n, BinaryReg::PtrAdd { dst: dst, lhs: GPR64(x), rhs: GPR64(y) })];
    emit = X86Add64(y, x);
    covers = [n];
    cost = 1;
}
select rule_47 {
    match = [bind(n, BinaryReg::PtrAdd { dst: GPR64(dst), lhs: GPTR(x), rhs: GPR64(y) })];
    emit = X86Add64(y, x);
    covers = [n];
    cost = 1;
}
select rule_48 {
    match = [bind(n, BinaryReg::PtrAdd { dst: GPTR(dst), lhs: GPTR(x), rhs: GPR64(y) })];
    emit = X86Add64(y, x);
    covers = [n];
    cost = 1;
}
select rule_49 {
    match = [bind(n, UnaryReg::Trunc { dst: dst, src: src })];
    emit = X86Mov32(src);
    covers = [n];
    cost = 1;
}
select rule_50 {
    match = [bind(n, UnaryReg::Zext { dst: GPR64(dst), src: GPR32(src) })];
    emit = X86Mov32(src);
    covers = [n];
    cost = 1;
}
select rule_51 {
    match = [bind(n, UnaryReg::Zext { dst: GPR32(dst), src: GPR8(src) })];
    emit = X86Movzx8to32(src);
    covers = [n];
    cost = 1;
}
select rule_52 {
    match = [bind(n, UnaryReg::Zext { dst: GPR32(dst), src: GPR16(src) })];
    emit = X86Movzx16to32(src);
    covers = [n];
    cost = 1;
}
select rule_53 {
    match = [bind(n, UnaryReg::Zext { dst: GPR64(dst), src: GPR8(src) })];
    emit = X86Movzx8to32(src);
    covers = [n];
    cost = 1;
}
select rule_54 {
    match = [bind(n, UnaryReg::Zext { dst: GPR64(dst), src: GPR16(src) })];
    emit = X86Movzx16to32(src);
    covers = [n];
    cost = 1;
}
select rule_55 {
    match = [bind(n, UnaryReg::Sext { dst: GPR32(dst), src: GPR8(src) })];
    emit = X86Movsx8to32(src);
    covers = [n];
    cost = 1;
}
select rule_56 {
    match = [bind(n, UnaryReg::Sext { dst: GPR32(dst), src: GPR16(src) })];
    emit = X86Movsx16to32(src);
    covers = [n];
    cost = 1;
}
select rule_57 {
    match = [bind(n, UnaryReg::Sext { dst: GPR64(dst), src: GPR8(src) })];
    emit = X86Movsx8to64(src);
    covers = [n];
    cost = 1;
}
select rule_58 {
    match = [bind(n, UnaryReg::Sext { dst: GPR64(dst), src: GPR16(src) })];
    emit = X86Movsx16to64(src);
    covers = [n];
    cost = 1;
}
select rule_59 {
    match = [bind(n, UnaryReg::Sext { dst: GPR64(dst), src: GPR32(src) })];
    emit = X86Movsxd32to64(src);
    covers = [n];
    cost = 1;
}
select rule_60 {
    match = [bind(n, Call::Call { callee: target })];
    emit = X86Call(target);
    covers = [n];
    cost = 1;
}
select rule_61 {
    match = [bind(n, CallIndirect::Callind { callee: target })];
    emit = X86CallReg(target);
    covers = [n];
    cost = 1;
}
select rule_62 {
    match = [bind(n, BinaryReg::Add { dst: GPR32(dst), lhs: GPR32(x), rhs: GPR32(y) })];
    emit = X86Add32(y, x);
    covers = [n];
    cost = 1;
}
select rule_63 {
    match = [bind(n, BinaryReg::Sub { dst: GPR32(dst), lhs: GPR32(x), rhs: GPR32(y) })];
    emit = X86Sub32(y, x);
    covers = [n];
    cost = 1;
}
select rule_64 {
    match = [bind(n, BinaryReg::Add { dst: GPR64(dst), lhs: GPR64(x), rhs: GPR64(y) })];
    emit = X86Add64(y, x);
    covers = [n];
    cost = 1;
}
select rule_65 {
    match = [bind(n, BinaryReg::Sub { dst: GPR64(dst), lhs: GPR64(x), rhs: GPR64(y) })];
    emit = X86Sub64(y, x);
    covers = [n];
    cost = 1;
}
select rule_66 {
    match = [bind(n, Load::Load { dst: GPR64(dst), base: GPTR(base) })];
    emit = X86Load64(base, 0);
    covers = [n];
    cost = 1;
}
select rule_67 {
    match = [bind(n, Load::Load { dst: GPR64(dst), base: GPR64(base) })];
    emit = X86Load64(base, 0);
    covers = [n];
    cost = 1;
}
select rule_68 {
    match = [bind(n, LoadOffset::OffsetLoad { dst: GPR64(dst), base: GPTR(base), offset: off })];
    emit = X86Load64(base, off);
    covers = [n];
    cost = 1;
}
select rule_69 {
    match = [bind(n, LoadOffset::OffsetLoad { dst: GPR64(dst), base: GPR64(base), offset: off })];
    emit = X86Load64(base, off);
    covers = [n];
    cost = 1;
}
select rule_70 {
    match = [bind(n, Load::Load { dst: GPTR(dst), base: GPTR(base) })];
    emit = X86Load64(base, 0);
    covers = [n];
    cost = 1;
}
select rule_71 {
    match = [bind(n, Load::Load { dst: GPTR(dst), base: GPR64(base) })];
    emit = X86Load64(base, 0);
    covers = [n];
    cost = 1;
}
select rule_72 {
    match = [bind(n, LoadOffset::OffsetLoad { dst: GPTR(dst), base: GPTR(base), offset: off })];
    emit = X86Load64(base, off);
    covers = [n];
    cost = 1;
}
select rule_73 {
    match = [bind(n, LoadOffset::OffsetLoad { dst: GPTR(dst), base: GPR64(base), offset: off })];
    emit = X86Load64(base, off);
    covers = [n];
    cost = 1;
}
select rule_74 {
    match = [bind(n, Load::Load { dst: GPR32(dst), base: GPTR(base) })];
    emit = X86Load32(base, 0);
    covers = [n];
    cost = 1;
}
select rule_75 {
    match = [bind(n, Load::Load { dst: GPR32(dst), base: GPR64(base) })];
    emit = X86Load32(base, 0);
    covers = [n];
    cost = 1;
}
select rule_76 {
    match = [bind(n, LoadOffset::OffsetLoad { dst: GPR32(dst), base: GPTR(base), offset: off })];
    emit = X86Load32(base, off);
    covers = [n];
    cost = 1;
}
select rule_77 {
    match = [bind(n, LoadOffset::OffsetLoad { dst: GPR32(dst), base: GPR64(base), offset: off })];
    emit = X86Load32(base, off);
    covers = [n];
    cost = 1;
}
select rule_78 {
    match = [bind(n, Load::Load { dst: FPR32(dst), base: GPTR(base) })];
    emit = X86LoadF32(base, 0);
    covers = [n];
    cost = 1;
}
select rule_79 {
    match = [bind(n, Load::Load { dst: FPR32(dst), base: GPR64(base) })];
    emit = X86LoadF32(base, 0);
    covers = [n];
    cost = 1;
}
select rule_80 {
    match = [bind(n, LoadOffset::OffsetLoad { dst: FPR32(dst), base: GPTR(base), offset: off })];
    emit = X86LoadF32(base, off);
    covers = [n];
    cost = 1;
}
select rule_81 {
    match = [bind(n, LoadOffset::OffsetLoad { dst: FPR32(dst), base: GPR64(base), offset: off })];
    emit = X86LoadF32(base, off);
    covers = [n];
    cost = 1;
}
select rule_82 {
    match = [bind(n, Load::Load { dst: FPR64(dst), base: GPTR(base) })];
    emit = X86LoadF64(base, 0);
    covers = [n];
    cost = 1;
}
select rule_83 {
    match = [bind(n, Load::Load { dst: FPR64(dst), base: GPR64(base) })];
    emit = X86LoadF64(base, 0);
    covers = [n];
    cost = 1;
}
select rule_84 {
    match = [bind(n, LoadOffset::OffsetLoad { dst: FPR64(dst), base: GPTR(base), offset: off })];
    emit = X86LoadF64(base, off);
    covers = [n];
    cost = 1;
}
select rule_85 {
    match = [bind(n, LoadOffset::OffsetLoad { dst: FPR64(dst), base: GPR64(base), offset: off })];
    emit = X86LoadF64(base, off);
    covers = [n];
    cost = 1;
}
select rule_86 {
    match = [bind(n, Load::Load { dst: GPR16(dst), base: GPTR(base) })];
    emit = X86Load16U32(base, 0);
    covers = [n];
    cost = 1;
}
select rule_87 {
    match = [bind(n, Load::Load { dst: GPR16(dst), base: GPR64(base) })];
    emit = X86Load16U32(base, 0);
    covers = [n];
    cost = 1;
}
select rule_88 {
    match = [bind(n, LoadOffset::OffsetLoad { dst: GPR16(dst), base: GPTR(base), offset: off })];
    emit = X86Load16U32(base, off);
    covers = [n];
    cost = 1;
}
select rule_89 {
    match = [bind(n, LoadOffset::OffsetLoad { dst: GPR16(dst), base: GPR64(base), offset: off })];
    emit = X86Load16U32(base, off);
    covers = [n];
    cost = 1;
}
select rule_90 {
    match = [bind(n, Load::Load { dst: GPR8(dst), base: GPTR(base) })];
    emit = X86Load8U32(base, 0);
    covers = [n];
    cost = 1;
}
select rule_91 {
    match = [bind(n, Load::Load { dst: GPR8(dst), base: GPR64(base) })];
    emit = X86Load8U32(base, 0);
    covers = [n];
    cost = 1;
}
select rule_92 {
    match = [bind(n, LoadOffset::OffsetLoad { dst: GPR8(dst), base: GPTR(base), offset: off })];
    emit = X86Load8U32(base, off);
    covers = [n];
    cost = 1;
}
select rule_93 {
    match = [bind(n, LoadOffset::OffsetLoad { dst: GPR8(dst), base: GPR64(base), offset: off })];
    emit = X86Load8U32(base, off);
    covers = [n];
    cost = 1;
}
select rule_94 {
    match = [bind(n, Store::Store { src: GPR64(src), base: GPTR(base) })];
    emit = X86Store64(src, base, 0);
    covers = [n];
    cost = 1;
}
select rule_95 {
    match = [bind(n, Store::Store { src: GPR64(src), base: GPR64(base) })];
    emit = X86Store64(src, base, 0);
    covers = [n];
    cost = 1;
}
select rule_96 {
    match = [bind(n, StoreOffset::OffsetStore { src: GPR64(src), base: GPTR(base), offset: off })];
    emit = X86Store64(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_97 {
    match = [bind(n, StoreOffset::OffsetStore { src: GPR64(src), base: GPR64(base), offset: off })];
    emit = X86Store64(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_98 {
    match = [bind(n, Store::Store { src: GPTR(src), base: GPTR(base) })];
    emit = X86Store64(src, base, 0);
    covers = [n];
    cost = 1;
}
select rule_99 {
    match = [bind(n, Store::Store { src: GPTR(src), base: GPR64(base) })];
    emit = X86Store64(src, base, 0);
    covers = [n];
    cost = 1;
}
select rule_100 {
    match = [bind(n, StoreOffset::OffsetStore { src: GPTR(src), base: GPTR(base), offset: off })];
    emit = X86Store64(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_101 {
    match = [bind(n, StoreOffset::OffsetStore { src: GPTR(src), base: GPR64(base), offset: off })];
    emit = X86Store64(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_102 {
    match = [bind(n, Store::Store { src: GPR32(src), base: GPTR(base) })];
    emit = X86Store32(src, base, 0);
    covers = [n];
    cost = 1;
}
select rule_103 {
    match = [bind(n, Store::Store { src: GPR32(src), base: GPR64(base) })];
    emit = X86Store32(src, base, 0);
    covers = [n];
    cost = 1;
}
select rule_104 {
    match = [bind(n, StoreOffset::OffsetStore { src: GPR32(src), base: GPTR(base), offset: off })];
    emit = X86Store32(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_105 {
    match = [bind(n, StoreOffset::OffsetStore { src: GPR32(src), base: GPR64(base), offset: off })];
    emit = X86Store32(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_106 {
    match = [bind(n, Store::Store { src: FPR32(src), base: GPTR(base) })];
    emit = X86StoreF32(src, base, 0);
    covers = [n];
    cost = 1;
}
select rule_107 {
    match = [bind(n, Store::Store { src: FPR32(src), base: GPR64(base) })];
    emit = X86StoreF32(src, base, 0);
    covers = [n];
    cost = 1;
}
select rule_108 {
    match = [bind(n, StoreOffset::OffsetStore { src: FPR32(src), base: GPTR(base), offset: off })];
    emit = X86StoreF32(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_109 {
    match = [bind(n, StoreOffset::OffsetStore { src: FPR32(src), base: GPR64(base), offset: off })];
    emit = X86StoreF32(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_110 {
    match = [bind(n, Store::Store { src: FPR64(src), base: GPTR(base) })];
    emit = X86StoreF64(src, base, 0);
    covers = [n];
    cost = 1;
}
select rule_111 {
    match = [bind(n, Store::Store { src: FPR64(src), base: GPR64(base) })];
    emit = X86StoreF64(src, base, 0);
    covers = [n];
    cost = 1;
}
select rule_112 {
    match = [bind(n, StoreOffset::OffsetStore { src: FPR64(src), base: GPTR(base), offset: off })];
    emit = X86StoreF64(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_113 {
    match = [bind(n, StoreOffset::OffsetStore { src: FPR64(src), base: GPR64(base), offset: off })];
    emit = X86StoreF64(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_114 {
    match = [bind(n, Store::Store { src: GPR16(src), base: GPTR(base) })];
    emit = X86Store16(src, base, 0);
    covers = [n];
    cost = 1;
}
select rule_115 {
    match = [bind(n, Store::Store { src: GPR16(src), base: GPR64(base) })];
    emit = X86Store16(src, base, 0);
    covers = [n];
    cost = 1;
}
select rule_116 {
    match = [bind(n, StoreOffset::OffsetStore { src: GPR16(src), base: GPTR(base), offset: off })];
    emit = X86Store16(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_117 {
    match = [bind(n, StoreOffset::OffsetStore { src: GPR16(src), base: GPR64(base), offset: off })];
    emit = X86Store16(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_118 {
    match = [bind(n, Store::Store { src: GPR8(src), base: GPTR(base) })];
    emit = X86Store8(src, base, 0);
    covers = [n];
    cost = 1;
}
select rule_119 {
    match = [bind(n, Store::Store { src: GPR8(src), base: GPR64(base) })];
    emit = X86Store8(src, base, 0);
    covers = [n];
    cost = 1;
}
select rule_120 {
    match = [bind(n, StoreOffset::OffsetStore { src: GPR8(src), base: GPTR(base), offset: off })];
    emit = X86Store8(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_121 {
    match = [bind(n, StoreOffset::OffsetStore { src: GPR8(src), base: GPR64(base), offset: off })];
    emit = X86Store8(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_122 {
    match = [bind(n, StackLoad::StackLoad { dst: GPR64(dst), slot: slot })];
    emit = X86Load64Stack(dst, slot);
    covers = [n];
    cost = 1;
}
select rule_123 {
    match = [bind(n, StackLoad::StackLoad { dst: GPTR(dst), slot: slot })];
    emit = X86Load64Stack(dst, slot);
    covers = [n];
    cost = 1;
}
select rule_124 {
    match = [bind(n, StackLoad::StackLoad { dst: GPR32(dst), slot: slot })];
    emit = X86Load32Stack(dst, slot);
    covers = [n];
    cost = 1;
}
select rule_125 {
    match = [bind(n, StackLoad::StackLoad { dst: GPR16(dst), slot: slot })];
    emit = X86Load16U32Stack(dst, slot);
    covers = [n];
    cost = 1;
}
select rule_126 {
    match = [bind(n, StackLoad::StackLoad { dst: GPR8(dst), slot: slot })];
    emit = X86Load8U32Stack(dst, slot);
    covers = [n];
    cost = 1;
}
select rule_127 {
    match = [bind(n, StackLoad::StackLoad { dst: FPR32(dst), slot: slot })];
    emit = X86LoadF32Stack(dst, slot);
    covers = [n];
    cost = 1;
}
select rule_128 {
    match = [bind(n, StackLoad::StackLoad { dst: FPR64(dst), slot: slot })];
    emit = X86LoadF64Stack(dst, slot);
    covers = [n];
    cost = 1;
}
select rule_129 {
    match = [bind(n, StackStore::StackStore { src: GPR64(src), slot: slot })];
    emit = X86Store64Stack(src, slot);
    covers = [n];
    cost = 1;
}
select rule_130 {
    match = [bind(n, StackStore::StackStore { src: GPTR(src), slot: slot })];
    emit = X86Store64Stack(src, slot);
    covers = [n];
    cost = 1;
}
select rule_131 {
    match = [bind(n, StackStore::StackStore { src: GPR32(src), slot: slot })];
    emit = X86Store32Stack(src, slot);
    covers = [n];
    cost = 1;
}
select rule_132 {
    match = [bind(n, StackStore::StackStore { src: GPR16(src), slot: slot })];
    emit = X86Store16Stack(src, slot);
    covers = [n];
    cost = 1;
}
select rule_133 {
    match = [bind(n, StackStore::StackStore { src: GPR8(src), slot: slot })];
    emit = X86Store8Stack(src, slot);
    covers = [n];
    cost = 1;
}
select rule_134 {
    match = [bind(n, StackStore::StackStore { src: FPR32(src), slot: slot })];
    emit = X86StoreF32Stack(src, slot);
    covers = [n];
    cost = 1;
}
select rule_135 {
    match = [bind(n, StackStore::StackStore { src: FPR64(src), slot: slot })];
    emit = X86StoreF64Stack(src, slot);
    covers = [n];
    cost = 1;
}
select rule_136 {
    match = [bind(n, IndexedLoad::IndexedLoad { dst: GPR64(dst), base: GPTR(base), offset: off })];
    emit = X86Load64(base, off);
    covers = [n];
    cost = 1;
}
select rule_137 {
    match = [bind(n, IndexedLoad::IndexedLoad { dst: GPR64(dst), base: GPR64(base), offset: off })];
    emit = X86Load64(base, off);
    covers = [n];
    cost = 1;
}
select rule_138 {
    match = [bind(n, IndexedLoad::IndexedLoad { dst: GPTR(dst), base: GPTR(base), offset: off })];
    emit = X86Load64(base, off);
    covers = [n];
    cost = 1;
}
select rule_139 {
    match = [bind(n, IndexedLoad::IndexedLoad { dst: GPTR(dst), base: GPR64(base), offset: off })];
    emit = X86Load64(base, off);
    covers = [n];
    cost = 1;
}
select rule_140 {
    match = [bind(n, IndexedLoad::IndexedLoad { dst: GPR32(dst), base: GPTR(base), offset: off })];
    emit = X86Load32(base, off);
    covers = [n];
    cost = 1;
}
select rule_141 {
    match = [bind(n, IndexedLoad::IndexedLoad { dst: GPR32(dst), base: GPR64(base), offset: off })];
    emit = X86Load32(base, off);
    covers = [n];
    cost = 1;
}
select rule_142 {
    match = [bind(n, IndexedLoad::IndexedLoad { dst: FPR32(dst), base: GPTR(base), offset: off })];
    emit = X86LoadF32(base, off);
    covers = [n];
    cost = 1;
}
select rule_143 {
    match = [bind(n, IndexedLoad::IndexedLoad { dst: FPR32(dst), base: GPR64(base), offset: off })];
    emit = X86LoadF32(base, off);
    covers = [n];
    cost = 1;
}
select rule_144 {
    match = [bind(n, IndexedLoad::IndexedLoad { dst: FPR64(dst), base: GPTR(base), offset: off })];
    emit = X86LoadF64(base, off);
    covers = [n];
    cost = 1;
}
select rule_145 {
    match = [bind(n, IndexedLoad::IndexedLoad { dst: FPR64(dst), base: GPR64(base), offset: off })];
    emit = X86LoadF64(base, off);
    covers = [n];
    cost = 1;
}
select rule_146 {
    match = [bind(n, IndexedLoad::IndexedLoad { dst: GPR16(dst), base: GPTR(base), offset: off })];
    emit = X86Load16U32(base, off);
    covers = [n];
    cost = 1;
}
select rule_147 {
    match = [bind(n, IndexedLoad::IndexedLoad { dst: GPR16(dst), base: GPR64(base), offset: off })];
    emit = X86Load16U32(base, off);
    covers = [n];
    cost = 1;
}
select rule_148 {
    match = [bind(n, IndexedLoad::IndexedLoad { dst: GPR8(dst), base: GPTR(base), offset: off })];
    emit = X86Load8U32(base, off);
    covers = [n];
    cost = 1;
}
select rule_149 {
    match = [bind(n, IndexedLoad::IndexedLoad { dst: GPR8(dst), base: GPR64(base), offset: off })];
    emit = X86Load8U32(base, off);
    covers = [n];
    cost = 1;
}
select rule_150 {
    match = [bind(n, IndexedStore::IndexedStore { src: GPR64(src), base: GPTR(base), offset: off })];
    emit = X86Store64(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_151 {
    match = [bind(n, IndexedStore::IndexedStore { src: GPR64(src), base: GPR64(base), offset: off })];
    emit = X86Store64(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_152 {
    match = [bind(n, IndexedStore::IndexedStore { src: GPTR(src), base: GPTR(base), offset: off })];
    emit = X86Store64(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_153 {
    match = [bind(n, IndexedStore::IndexedStore { src: GPTR(src), base: GPR64(base), offset: off })];
    emit = X86Store64(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_154 {
    match = [bind(n, IndexedStore::IndexedStore { src: GPR32(src), base: GPTR(base), offset: off })];
    emit = X86Store32(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_155 {
    match = [bind(n, IndexedStore::IndexedStore { src: GPR32(src), base: GPR64(base), offset: off })];
    emit = X86Store32(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_156 {
    match = [bind(n, IndexedStore::IndexedStore { src: FPR32(src), base: GPTR(base), offset: off })];
    emit = X86StoreF32(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_157 {
    match = [bind(n, IndexedStore::IndexedStore { src: FPR32(src), base: GPR64(base), offset: off })];
    emit = X86StoreF32(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_158 {
    match = [bind(n, IndexedStore::IndexedStore { src: FPR64(src), base: GPTR(base), offset: off })];
    emit = X86StoreF64(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_159 {
    match = [bind(n, IndexedStore::IndexedStore { src: FPR64(src), base: GPR64(base), offset: off })];
    emit = X86StoreF64(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_160 {
    match = [bind(n, IndexedStore::IndexedStore { src: GPR16(src), base: GPTR(base), offset: off })];
    emit = X86Store16(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_161 {
    match = [bind(n, IndexedStore::IndexedStore { src: GPR16(src), base: GPR64(base), offset: off })];
    emit = X86Store16(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_162 {
    match = [bind(n, IndexedStore::IndexedStore { src: GPR8(src), base: GPTR(base), offset: off })];
    emit = X86Store8(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_163 {
    match = [bind(n, IndexedStore::IndexedStore { src: GPR8(src), base: GPR64(base), offset: off })];
    emit = X86Store8(src, base, off);
    covers = [n];
    cost = 1;
}
select rule_164 {
    match = [bind(n, BinaryReg::Mul { dst: GPR32(dst), lhs: GPR32(x), rhs: GPR32(y) })];
    emit = X86IMul32(y, x);
    covers = [n];
    cost = 1;
}
select rule_165 {
    match = [bind(n, BinaryReg::And { dst: GPR32(dst), lhs: GPR32(x), rhs: GPR32(y) })];
    emit = X86And32(y, x);
    covers = [n];
    cost = 1;
}
select rule_166 {
    match = [bind(n, BinaryReg::Or { dst: GPR32(dst), lhs: GPR32(x), rhs: GPR32(y) })];
    emit = X86Or32(y, x);
    covers = [n];
    cost = 1;
}
select rule_167 {
    match = [bind(n, BinaryReg::Xor { dst: GPR32(dst), lhs: GPR32(x), rhs: GPR32(y) })];
    emit = X86Xor32(y, x);
    covers = [n];
    cost = 1;
}
select rule_168 {
    match = [bind(n, BinaryReg::Mul { dst: GPR64(dst), lhs: GPR64(x), rhs: GPR64(y) })];
    emit = X86IMul64(y, x);
    covers = [n];
    cost = 1;
}
select rule_169 {
    match = [bind(n, BinaryReg::And { dst: GPR64(dst), lhs: GPR64(x), rhs: GPR64(y) })];
    emit = X86And64(y, x);
    covers = [n];
    cost = 1;
}
select rule_170 {
    match = [bind(n, BinaryReg::Or { dst: GPR64(dst), lhs: GPR64(x), rhs: GPR64(y) })];
    emit = X86Or64(y, x);
    covers = [n];
    cost = 1;
}
select rule_171 {
    match = [bind(n, BinaryReg::Xor { dst: GPR64(dst), lhs: GPR64(x), rhs: GPR64(y) })];
    emit = X86Xor64(y, x);
    covers = [n];
    cost = 1;
}
select rule_172 {
    match = [bind(n, BinaryReg::Shl { dst: GPR32(dst), lhs: GPR32(x), rhs: y })];
    emit = X86Shl32Cl(y, x);
    covers = [n];
    cost = 1;
}
select rule_173 {
    match = [bind(n, BinaryReg::Lshr { dst: GPR32(dst), lhs: GPR32(x), rhs: y })];
    emit = X86Shr32Cl(y, x);
    covers = [n];
    cost = 1;
}
select rule_174 {
    match = [bind(n, BinaryReg::Ashr { dst: GPR32(dst), lhs: GPR32(x), rhs: y })];
    emit = X86Sar32Cl(y, x);
    covers = [n];
    cost = 1;
}
select rule_175 {
    match = [bind(n, BinaryReg::Shl { dst: GPR64(dst), lhs: GPR64(x), rhs: y })];
    emit = X86Shl64Cl(y, x);
    covers = [n];
    cost = 1;
}
select rule_176 {
    match = [bind(n, BinaryReg::Lshr { dst: GPR64(dst), lhs: GPR64(x), rhs: y })];
    emit = X86Shr64Cl(y, x);
    covers = [n];
    cost = 1;
}
select rule_177 {
    match = [bind(n, BinaryReg::Ashr { dst: GPR64(dst), lhs: GPR64(x), rhs: y })];
    emit = X86Sar64Cl(y, x);
    covers = [n];
    cost = 1;
}
select rule_178 {
    match = [bind(n, ICmp::Icmp { dst: BOOL(dst), lhs: GPR32Like(x), rhs: GPR32Like(y), cc: CC::E })];
    temps = { bit: dst };
    emit = seq(X86Cmp32(x, y), X86Sete(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_179 {
    match = [bind(n, ICmp::Icmp { dst: BOOL(dst), lhs: GPR32Like(x), rhs: GPR32Like(y), cc: CC::NE })];
    temps = { bit: dst };
    emit = seq(X86Cmp32(x, y), X86Setne(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_180 {
    match = [bind(n, ICmp::Icmp { dst: BOOL(dst), lhs: GPR32Like(x), rhs: GPR32Like(y), cc: CC::L })];
    temps = { bit: dst };
    emit = seq(X86Cmp32(x, y), X86Setl(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_181 {
    match = [bind(n, ICmp::Icmp { dst: BOOL(dst), lhs: GPR32Like(x), rhs: GPR32Like(y), cc: CC::LE })];
    temps = { bit: dst };
    emit = seq(X86Cmp32(x, y), X86Setle(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_182 {
    match = [bind(n, ICmp::Icmp { dst: BOOL(dst), lhs: GPR32Like(x), rhs: GPR32Like(y), cc: CC::G })];
    temps = { bit: dst };
    emit = seq(X86Cmp32(x, y), X86Setg(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_183 {
    match = [bind(n, ICmp::Icmp { dst: BOOL(dst), lhs: GPR32Like(x), rhs: GPR32Like(y), cc: CC::GE })];
    temps = { bit: dst };
    emit = seq(X86Cmp32(x, y), X86Setge(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_184 {
    match = [bind(n, ICmp::Icmp { dst: BOOL(dst), lhs: GPR32Like(x), rhs: GPR32Like(y), cc: CC::B })];
    temps = { bit: dst };
    emit = seq(X86Cmp32(x, y), X86Setb(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_185 {
    match = [bind(n, ICmp::Icmp { dst: BOOL(dst), lhs: GPR32Like(x), rhs: GPR32Like(y), cc: CC::BE })];
    temps = { bit: dst };
    emit = seq(X86Cmp32(x, y), X86Setbe(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_186 {
    match = [bind(n, ICmp::Icmp { dst: BOOL(dst), lhs: GPR32Like(x), rhs: GPR32Like(y), cc: CC::A })];
    temps = { bit: dst };
    emit = seq(X86Cmp32(x, y), X86Seta(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_187 {
    match = [bind(n, ICmp::Icmp { dst: BOOL(dst), lhs: GPR32Like(x), rhs: GPR32Like(y), cc: CC::AE })];
    temps = { bit: dst };
    emit = seq(X86Cmp32(x, y), X86Setae(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_188 {
    match = [bind(n, ICmp::Icmp { dst: BOOL(dst), lhs: GPR64Like(x), rhs: GPR64Like(y), cc: CC::E })];
    temps = { bit: dst };
    emit = seq(X86Cmp64(x, y), X86Sete(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_189 {
    match = [bind(n, ICmp::Icmp { dst: BOOL(dst), lhs: GPR64Like(x), rhs: GPR64Like(y), cc: CC::NE })];
    temps = { bit: dst };
    emit = seq(X86Cmp64(x, y), X86Setne(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_190 {
    match = [bind(n, ICmp::Icmp { dst: BOOL(dst), lhs: GPR64Like(x), rhs: GPR64Like(y), cc: CC::L })];
    temps = { bit: dst };
    emit = seq(X86Cmp64(x, y), X86Setl(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_191 {
    match = [bind(n, ICmp::Icmp { dst: BOOL(dst), lhs: GPR64Like(x), rhs: GPR64Like(y), cc: CC::LE })];
    temps = { bit: dst };
    emit = seq(X86Cmp64(x, y), X86Setle(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_192 {
    match = [bind(n, ICmp::Icmp { dst: BOOL(dst), lhs: GPR64Like(x), rhs: GPR64Like(y), cc: CC::G })];
    temps = { bit: dst };
    emit = seq(X86Cmp64(x, y), X86Setg(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_193 {
    match = [bind(n, ICmp::Icmp { dst: BOOL(dst), lhs: GPR64Like(x), rhs: GPR64Like(y), cc: CC::GE })];
    temps = { bit: dst };
    emit = seq(X86Cmp64(x, y), X86Setge(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_194 {
    match = [bind(n, ICmp::Icmp { dst: BOOL(dst), lhs: GPR64Like(x), rhs: GPR64Like(y), cc: CC::B })];
    temps = { bit: dst };
    emit = seq(X86Cmp64(x, y), X86Setb(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_195 {
    match = [bind(n, ICmp::Icmp { dst: BOOL(dst), lhs: GPR64Like(x), rhs: GPR64Like(y), cc: CC::BE })];
    temps = { bit: dst };
    emit = seq(X86Cmp64(x, y), X86Setbe(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_196 {
    match = [bind(n, ICmp::Icmp { dst: BOOL(dst), lhs: GPR64Like(x), rhs: GPR64Like(y), cc: CC::A })];
    temps = { bit: dst };
    emit = seq(X86Cmp64(x, y), X86Seta(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_197 {
    match = [bind(n, ICmp::Icmp { dst: BOOL(dst), lhs: GPR64Like(x), rhs: GPR64Like(y), cc: CC::AE })];
    temps = { bit: dst };
    emit = seq(X86Cmp64(x, y), X86Setae(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_198 {
    match = [bind(n, FCmp::Fcmp { dst: BOOL(dst), lhs: FPR32(x), rhs: FPR32(y), cc: CC::A })];
    temps = { bit: dst };
    emit = seq(X86Ucomiss(x, y), X86Seta(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_199 {
    match = [bind(n, FCmp::Fcmp { dst: BOOL(dst), lhs: FPR32(x), rhs: FPR32(y), cc: CC::AE })];
    temps = { bit: dst };
    emit = seq(X86Ucomiss(x, y), X86Setae(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_200 {
    match = [bind(n, FCmp::Fcmp { dst: BOOL(dst), lhs: FPR64(x), rhs: FPR64(y), cc: CC::A })];
    temps = { bit: dst };
    emit = seq(X86Ucomisd(x, y), X86Seta(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_201 {
    match = [bind(n, FCmp::Fcmp { dst: BOOL(dst), lhs: FPR64(x), rhs: FPR64(y), cc: CC::AE })];
    temps = { bit: dst };
    emit = seq(X86Ucomisd(x, y), X86Setae(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_202 {
    match = [bind(n, UnaryReg::Ieqz { dst: BOOL(dst), src: GPR32Like(src) })];
    temps = { bit: dst };
    emit = seq(X86Test32(src, src), X86Sete(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_203 {
    match = [bind(n, UnaryReg::Ieqz { dst: BOOL(dst), src: GPR64Like(src) })];
    temps = { bit: dst };
    emit = seq(X86Test64(src, src), X86Sete(bit), X86Movzx8to32(dst, bit));
    covers = [n];
    cost = 1;
}
select rule_204 {
    match = [bind(n, Branch::Br { target: block })];
    emit = X86Jmp(block);
    covers = [n];
    cost = 1;
}
select rule_205 {
    match = [bind(n, BranchCond::Brcond { cond: cond, then_blk: then, else_blk: else })];
    emit = seq(X86Test32(cond, cond), X86Jne(then), X86Jmp(else));
    covers = [n];
    cost = 3;
}
select rule_206 {
    match = [bind(n, Return::Ret {  })];
    emit = X86Ret();
    covers = [n];
    cost = 1;
}
select rule_207 {
    match = [bind(n, Unreachable::Unreachable {  })];
    emit = X86Ud2();
    covers = [n];
    cost = 1;
}
select rule_208 {
    match = [bind(n, BinaryReg::Sdiv { dst: dst, lhs: lhs, rhs: GPR64(rhs) })];
    emit = seq(X86Mov64(reg(RAX), lhs), X86Cqo(), X86IDiv64(rhs), X86Mov64(dst, reg(RAX)));
    covers = [n];
    cost = 1;
}
select rule_209 {
    match = [bind(n, BinaryReg::Sdiv { dst: dst, lhs: lhs, rhs: GPR32(rhs) })];
    emit = seq(X86Mov32(reg(RAX), lhs), X86Cdq(), X86IDiv32(rhs), X86Mov32(dst, reg(RAX)));
    covers = [n];
    cost = 1;
}
select rule_210 {
    match = [bind(n, BinaryReg::Srem { dst: dst, lhs: lhs, rhs: GPR64(rhs) })];
    emit = seq(X86Mov64(reg(RAX), lhs), X86Cqo(), X86IDiv64(rhs), X86Mov64(dst, reg(RDX)));
    covers = [n];
    cost = 1;
}
select rule_211 {
    match = [bind(n, BinaryReg::Srem { dst: dst, lhs: lhs, rhs: GPR32(rhs) })];
    emit = seq(X86Mov32(reg(RAX), lhs), X86Cdq(), X86IDiv32(rhs), X86Mov32(dst, reg(RDX)));
    covers = [n];
    cost = 1;
}
select rule_212 {
    match = [bind(n, BinaryReg::Udiv { dst: dst, lhs: lhs, rhs: GPR64(rhs) })];
    emit = seq(X86Mov64(reg(RAX), lhs), X86Xor64(reg(RDX), reg(RDX), reg(RDX)), X86Div64(rhs), X86Mov64(dst, reg(RAX)));
    covers = [n];
    cost = 1;
}
select rule_213 {
    match = [bind(n, BinaryReg::Udiv { dst: dst, lhs: lhs, rhs: GPR32(rhs) })];
    emit = seq(X86Mov32(reg(RAX), lhs), X86Xor32(reg(RDX), reg(RDX), reg(RDX)), X86Div32(rhs), X86Mov32(dst, reg(RAX)));
    covers = [n];
    cost = 1;
}
select rule_214 {
    match = [bind(n, BinaryReg::Urem { dst: dst, lhs: lhs, rhs: GPR64(rhs) })];
    emit = seq(X86Mov64(reg(RAX), lhs), X86Xor64(reg(RDX), reg(RDX), reg(RDX)), X86Div64(rhs), X86Mov64(dst, reg(RDX)));
    covers = [n];
    cost = 1;
}
select rule_215 {
    match = [bind(n, BinaryReg::Urem { dst: dst, lhs: lhs, rhs: GPR32(rhs) })];
    emit = seq(X86Mov32(reg(RAX), lhs), X86Xor32(reg(RDX), reg(RDX), reg(RDX)), X86Div32(rhs), X86Mov32(dst, reg(RDX)));
    covers = [n];
    cost = 1;
}
select rule_216 {
    match = [bind(n, BinaryReg::Fadd { dst: FPR32(dst), lhs: FPR32(lhs), rhs: FPR32(rhs) })];
    emit = X86FAdd32(rhs, lhs);
    covers = [n];
    cost = 1;
}
select rule_217 {
    match = [bind(n, BinaryReg::Fadd { dst: FPR64(dst), lhs: FPR64(lhs), rhs: FPR64(rhs) })];
    emit = X86FAdd64(rhs, lhs);
    covers = [n];
    cost = 1;
}
select rule_218 {
    match = [bind(n, BinaryReg::Fsub { dst: FPR32(dst), lhs: FPR32(lhs), rhs: FPR32(rhs) })];
    emit = X86FSub32(rhs, lhs);
    covers = [n];
    cost = 1;
}
select rule_219 {
    match = [bind(n, BinaryReg::Fsub { dst: FPR64(dst), lhs: FPR64(lhs), rhs: FPR64(rhs) })];
    emit = X86FSub64(rhs, lhs);
    covers = [n];
    cost = 1;
}
select rule_220 {
    match = [bind(n, BinaryReg::Fmul { dst: FPR32(dst), lhs: FPR32(lhs), rhs: FPR32(rhs) })];
    emit = X86FMul32(rhs, lhs);
    covers = [n];
    cost = 1;
}
select rule_221 {
    match = [bind(n, BinaryReg::Fmul { dst: FPR64(dst), lhs: FPR64(lhs), rhs: FPR64(rhs) })];
    emit = X86FMul64(rhs, lhs);
    covers = [n];
    cost = 1;
}
select rule_222 {
    match = [bind(n, BinaryReg::Fdiv { dst: FPR32(dst), lhs: FPR32(lhs), rhs: FPR32(rhs) })];
    emit = X86FDiv32(rhs, lhs);
    covers = [n];
    cost = 1;
}
select rule_223 {
    match = [bind(n, BinaryReg::Fdiv { dst: FPR64(dst), lhs: FPR64(lhs), rhs: FPR64(rhs) })];
    emit = X86FDiv64(rhs, lhs);
    covers = [n];
    cost = 1;
}
select rule_224 {
    match = [bind(n, StackAddr::StackAddr { dst: GPTR(dst), slot: slot })];
    emit = X86LeaStack(dst, slot);
    covers = [n];
    cost = 1;
}
select rule_225 {
    match = [bind(n, Constant::Constant { dst: BOOL(dst), imm: imm })];
    emit = X86Mov32Imm(imm);
    covers = [n];
    cost = 1;
}
select rule_226 {
    match = [bind(n, UnaryReg::Zext { dst: GPR32(dst), src: BOOL(src) })];
    emit = X86Movzx8to32(dst, src);
    covers = [n];
    cost = 1;
}
select rule_227 {
    match = [bind(n, UnaryReg::Zext { dst: GPR64(dst), src: BOOL(src) })];
    emit = X86Movzx8to32(dst, src);
    covers = [n];
    cost = 1;
}
select rule_228 {
    match = [bind(n, BinaryReg::Rotl { dst: GPR32(dst), lhs: GPR32(x), rhs: y })];
    emit = X86Rol32Cl(y, x);
    covers = [n];
    cost = 1;
}
select rule_229 {
    match = [bind(n, BinaryReg::Rotl { dst: GPR64(dst), lhs: GPR64(x), rhs: y })];
    emit = X86Rol64Cl(y, x);
    covers = [n];
    cost = 1;
}
select rule_230 {
    match = [bind(n, BinaryReg::Rotr { dst: GPR32(dst), lhs: GPR32(x), rhs: y })];
    emit = X86Ror32Cl(y, x);
    covers = [n];
    cost = 1;
}
select rule_231 {
    match = [bind(n, BinaryReg::Rotr { dst: GPR64(dst), lhs: GPR64(x), rhs: y })];
    emit = X86Ror64Cl(y, x);
    covers = [n];
    cost = 1;
}
select rule_232 {
    match = [bind(n, BinaryReg::And { dst: BOOL(dst), lhs: BOOL(x), rhs: BOOL(y) })];
    emit = X86And32(y, x);
    covers = [n];
    cost = 1;
}
select rule_233 {
    match = [bind(n, BinaryReg::Or { dst: BOOL(dst), lhs: BOOL(x), rhs: BOOL(y) })];
    emit = X86Or32(y, x);
    covers = [n];
    cost = 1;
}
select rule_234 {
    match = [bind(n, BinaryReg::Xor { dst: BOOL(dst), lhs: BOOL(x), rhs: BOOL(y) })];
    emit = X86Xor32(y, x);
    covers = [n];
    cost = 1;
}
select rule_235 {
    match = [bind(n, UnaryReg::Sitofp { dst: FPR32(dst), src: GPR32(src) })];
    emit = X86I32ToF32(src);
    covers = [n];
    cost = 4;
}
select rule_236 {
    match = [bind(n, UnaryReg::Sitofp { dst: FPR32(dst), src: GPR64(src) })];
    emit = X86I64ToF32(src);
    covers = [n];
    cost = 4;
}
select rule_237 {
    match = [bind(n, UnaryReg::Sitofp { dst: FPR64(dst), src: GPR32(src) })];
    emit = X86I32ToF64(src);
    covers = [n];
    cost = 4;
}
select rule_238 {
    match = [bind(n, UnaryReg::Sitofp { dst: FPR64(dst), src: GPR64(src) })];
    emit = X86I64ToF64(src);
    covers = [n];
    cost = 4;
}
select rule_239 {
    match = [bind(n, UnaryReg::Fptosi { dst: GPR32(dst), src: FPR32(src) })];
    emit = X86F32ToI32(src);
    covers = [n];
    cost = 4;
}
select rule_240 {
    match = [bind(n, UnaryReg::Fptosi { dst: GPR64(dst), src: FPR32(src) })];
    emit = X86F32ToI64(src);
    covers = [n];
    cost = 4;
}
select rule_241 {
    match = [bind(n, UnaryReg::Fptosi { dst: GPR32(dst), src: FPR64(src) })];
    emit = X86F64ToI32(src);
    covers = [n];
    cost = 4;
}
select rule_242 {
    match = [bind(n, UnaryReg::Fptosi { dst: GPR64(dst), src: FPR64(src) })];
    emit = X86F64ToI64(src);
    covers = [n];
    cost = 4;
}
select rule_243 {
    match = [bind(n, UnaryReg::Fpext { dst: FPR64(dst), src: FPR32(src) })];
    emit = X86F32ToF64(src);
    covers = [n];
    cost = 4;
}
select rule_244 {
    match = [bind(n, UnaryReg::Fptrunc { dst: FPR32(dst), src: FPR64(src) })];
    emit = X86F64ToF32(src);
    covers = [n];
    cost = 4;
}
select rule_245 {
    match = [bind(n, UnaryReg::Fsqrt { dst: FPR32(dst), src: FPR32(src) })];
    emit = X86SqrtF32(src);
    covers = [n];
    cost = 4;
}
select rule_246 {
    match = [bind(n, UnaryReg::Fsqrt { dst: FPR64(dst), src: FPR64(src) })];
    emit = X86SqrtF64(src);
    covers = [n];
    cost = 4;
}
select rule_247 {
    match = [bind(n, UnaryReg::Ctpop { dst: GPR32(dst), src: GPR32(src) })];
    emit = X86Popcnt32(src);
    covers = [n];
    cost = 3;
}
select rule_248 {
    match = [bind(n, UnaryReg::Ctpop { dst: GPR64(dst), src: GPR64(src) })];
    emit = X86Popcnt64(src);
    covers = [n];
    cost = 3;
}
