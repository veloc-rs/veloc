import "../../../defs/type_sets.spec";

typeset SmallInt = Type::I8 | Type::I16 | Type::I32;
typeset WordOrPtr = Type::I64 | Type::PTR;

select(n: lir::Constant) {
    choose {
        case {
            require(type_is<SmallInt>(n.dst));
            replace(n, build(X86Mov32Imm(n.imm)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86Mov64Imm64(n.imm)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            replace(n, build(X86Mov32Imm(n.imm)));
        }
    }
}

select(n: lir::Copy) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(X86Mov32(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86Mov64(n.src)));
        }
        case {
            require(type_is<Type::PTR>(n.dst));
            replace(n, build(X86Mov64(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::PTR>(n.src));
            replace(n, build(X86Mov64(n.src)));
        }
        case {
            require(type_is<Type::PTR>(n.dst));
            require(type_is<Type::I64>(n.src));
            replace(n, build(X86Mov64(n.src)));
        }
        case {
            require(type_is<Type::F32>(n.dst));
            replace(n, build(X86Movss(n.src)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            replace(n, build(X86Movsd(n.src)));
        }
    }
}

select(n: lir::Inttoptr) {
    choose {
        case {
            require(type_is<Type::PTR>(n.dst));
            require(type_is<Type::I64>(n.src));
            replace(n, build(X86Mov64(n.src)));
        }
    }
}

select(n: lir::Ptrtoint) {
    choose {
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::PTR>(n.src));
            replace(n, build(X86Mov64(n.src)));
        }
    }
}

select(n: lir::Bitcast) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::I32>(n.src));
            replace(n, build(X86Mov32(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::I64>(n.src));
            replace(n, build(X86Mov64(n.src)));
        }
        case {
            require(type_is<Type::PTR>(n.dst));
            require(type_is<Type::PTR>(n.src));
            replace(n, build(X86Mov64(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::PTR>(n.src));
            replace(n, build(X86Mov64(n.src)));
        }
        case {
            require(type_is<Type::PTR>(n.dst));
            require(type_is<Type::I64>(n.src));
            replace(n, build(X86Mov64(n.src)));
        }
        case {
            require(type_is<Type::F32>(n.dst));
            require(type_is<Type::F32>(n.src));
            replace(n, build(X86Movss(n.src)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            require(type_is<Type::F64>(n.src));
            replace(n, build(X86Movsd(n.src)));
        }
        case {
            require(type_is<Type::F32>(n.dst));
            require(type_is<Type::I32>(n.src));
            replace(n, build(X86MovdToXmm(n.src)));
        }
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::F32>(n.src));
            replace(n, build(X86MovdFromXmm(n.src)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            require(type_is<Type::I64>(n.src));
            replace(n, build(X86MovqToXmm(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::F64>(n.src));
            replace(n, build(X86MovqFromXmm(n.src)));
        }
    }
}

select(n: lir::PtrAdd) {
    choose {
        case {
            require(type_is<Type::I64>(n.lhs));
            require(type_is<Type::I64>(n.rhs));
            replace(n, build(X86Add64(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::PTR>(n.lhs));
            require(type_is<Type::I64>(n.rhs));
            replace(n, build(X86Add64(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::PTR>(n.dst));
            require(type_is<Type::PTR>(n.lhs));
            require(type_is<Type::I64>(n.rhs));
            replace(n, build(X86Add64(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Trunc) {
    choose {
        case {
            replace(n, build(X86Mov32(n.src)));
        }
    }
}

select(n: lir::Zext) {
    choose {
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::I32>(n.src));
            replace(n, build(X86Mov32(n.src)));
        }
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::I8>(n.src));
            replace(n, build(X86Movzx8to32(n.src)));
        }
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::I16>(n.src));
            replace(n, build(X86Movzx16to32(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::I8>(n.src));
            replace(n, build(X86Movzx8to32(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::I16>(n.src));
            replace(n, build(X86Movzx16to32(n.src)));
        }
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::BOOL>(n.src));
            replace(n, build(X86Movzx8to32(n.dst, n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::BOOL>(n.src));
            replace(n, build(X86Movzx8to32(n.dst, n.src)));
        }
    }
}

select(n: lir::Sext) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::I8>(n.src));
            replace(n, build(X86Movsx8to32(n.src)));
        }
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::I16>(n.src));
            replace(n, build(X86Movsx16to32(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::I8>(n.src));
            replace(n, build(X86Movsx8to64(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::I16>(n.src));
            replace(n, build(X86Movsx16to64(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::I32>(n.src));
            replace(n, build(X86Movsxd32to64(n.src)));
        }
    }
}

select(n: lir::Call) {
    choose {
        case {
            replace(n, build(X86Call(n.callee)));
        }
    }
}

select(n: lir::Callind) {
    choose {
        case {
            replace(n, build(X86CallReg(n.callee)));
        }
    }
}

select(n: lir::Add) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(X86Add32(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86Add64(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Sub) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(X86Sub32(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86Sub64(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Load) {
    choose {
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::I8>(n.dst));
            require(matches(n.offset, 0));
            replace(n, build(X86Load8U32Stack(addr.slot)));
        }
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::I16>(n.dst));
            require(matches(n.offset, 0));
            replace(n, build(X86Load16U32Stack(addr.slot)));
        }
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::I32>(n.dst));
            require(matches(n.offset, 0));
            replace(n, build(X86Load32Stack(addr.slot)));
        }
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::I64>(n.dst));
            require(matches(n.offset, 0));
            replace(n, build(X86Load64Stack(addr.slot)));
        }
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::PTR>(n.dst));
            require(matches(n.offset, 0));
            replace(n, build(X86Load64Stack(addr.slot)));
        }
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::F32>(n.dst));
            require(matches(n.offset, 0));
            replace(n, build(X86LoadF32Stack(addr.slot)));
        }
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::F64>(n.dst));
            require(matches(n.offset, 0));
            replace(n, build(X86LoadF64Stack(addr.slot)));
        }
        case {
            let addr = def<lir::Add<Type::I64>>(n.base);
            require(type_is<WordOrPtr>(n.dst));
            replace(n, build(X86Load64Index(addr.lhs, addr.rhs, n.offset)));
        }
        case {
            let addr = def<lir::PtrAdd<Type::I64>>(n.base);
            require(type_is<WordOrPtr>(n.dst));
            require(type_is<Type::PTR>(addr.dst));
            require(type_is<Type::PTR>(addr.lhs));
            replace(n, build(X86Load64Index(addr.lhs, addr.rhs, n.offset)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86Load64(n.base, n.offset)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86Load64(n.base, n.offset)));
        }
        case {
            require(type_is<Type::PTR>(n.dst));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86Load64(n.base, n.offset)));
        }
        case {
            require(type_is<Type::PTR>(n.dst));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86Load64(n.base, n.offset)));
        }
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86Load32(n.base, n.offset)));
        }
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86Load32(n.base, n.offset)));
        }
        case {
            require(type_is<Type::F32>(n.dst));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86LoadF32(n.base, n.offset)));
        }
        case {
            require(type_is<Type::F32>(n.dst));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86LoadF32(n.base, n.offset)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86LoadF64(n.base, n.offset)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86LoadF64(n.base, n.offset)));
        }
        case {
            require(type_is<Type::I16>(n.dst));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86Load16U32(n.base, n.offset)));
        }
        case {
            require(type_is<Type::I16>(n.dst));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86Load16U32(n.base, n.offset)));
        }
        case {
            require(type_is<Type::I8>(n.dst));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86Load8U32(n.base, n.offset)));
        }
        case {
            require(type_is<Type::I8>(n.dst));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86Load8U32(n.base, n.offset)));
        }
    }
}

select(n: lir::Store) {
    choose {
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::I8>(n.src));
            require(matches(n.offset, 0));
            replace(n, build(X86Store8Stack(n.src, addr.slot)));
        }
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::I16>(n.src));
            require(matches(n.offset, 0));
            replace(n, build(X86Store16Stack(n.src, addr.slot)));
        }
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::I32>(n.src));
            require(matches(n.offset, 0));
            replace(n, build(X86Store32Stack(n.src, addr.slot)));
        }
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::I64>(n.src));
            require(matches(n.offset, 0));
            replace(n, build(X86Store64Stack(n.src, addr.slot)));
        }
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::PTR>(n.src));
            require(matches(n.offset, 0));
            replace(n, build(X86Store64Stack(n.src, addr.slot)));
        }
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::F32>(n.src));
            require(matches(n.offset, 0));
            replace(n, build(X86StoreF32Stack(n.src, addr.slot)));
        }
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::F64>(n.src));
            require(matches(n.offset, 0));
            replace(n, build(X86StoreF64Stack(n.src, addr.slot)));
        }
        case {
            let addr = def<lir::Add<Type::I64>>(n.base);
            require(type_is<WordOrPtr>(n.src));
            replace(n, build(X86Store64Index(n.src, addr.lhs, addr.rhs, n.offset)));
        }
        case {
            let addr = def<lir::PtrAdd<Type::I64>>(n.base);
            require(type_is<WordOrPtr>(n.src));
            require(type_is<Type::PTR>(addr.dst));
            require(type_is<Type::PTR>(addr.lhs));
            replace(n, build(X86Store64Index(n.src, addr.lhs, addr.rhs, n.offset)));
        }
        case {
            require(type_is<Type::I64>(n.src));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86Store64(n.src, n.base, n.offset)));
        }
        case {
            require(type_is<Type::I64>(n.src));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86Store64(n.src, n.base, n.offset)));
        }
        case {
            require(type_is<Type::PTR>(n.src));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86Store64(n.src, n.base, n.offset)));
        }
        case {
            require(type_is<Type::PTR>(n.src));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86Store64(n.src, n.base, n.offset)));
        }
        case {
            require(type_is<Type::I32>(n.src));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86Store32(n.src, n.base, n.offset)));
        }
        case {
            require(type_is<Type::I32>(n.src));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86Store32(n.src, n.base, n.offset)));
        }
        case {
            require(type_is<Type::F32>(n.src));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86StoreF32(n.src, n.base, n.offset)));
        }
        case {
            require(type_is<Type::F32>(n.src));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86StoreF32(n.src, n.base, n.offset)));
        }
        case {
            require(type_is<Type::F64>(n.src));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86StoreF64(n.src, n.base, n.offset)));
        }
        case {
            require(type_is<Type::F64>(n.src));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86StoreF64(n.src, n.base, n.offset)));
        }
        case {
            require(type_is<Type::I16>(n.src));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86Store16(n.src, n.base, n.offset)));
        }
        case {
            require(type_is<Type::I16>(n.src));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86Store16(n.src, n.base, n.offset)));
        }
        case {
            require(type_is<Type::I8>(n.src));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86Store8(n.src, n.base, n.offset)));
        }
        case {
            require(type_is<Type::I8>(n.src));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86Store8(n.src, n.base, n.offset)));
        }
    }
}

select(n: lir::Mul) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(X86IMul32(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86IMul64(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::And) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(X86And32(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86And64(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            replace(n, build(X86And32(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Or) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(X86Or32(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86Or64(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            replace(n, build(X86Or32(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Xor) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(X86Xor32(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86Xor64(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            replace(n, build(X86Xor32(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Shl) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(X86Shl32Cl(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86Shl64Cl(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Lshr) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(X86Shr32Cl(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86Shr64Cl(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Ashr) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(X86Sar32Cl(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86Sar64Cl(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Icmp) {
    choose {
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::E));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp32(n.lhs, n.rhs)), build(X86Sete(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::NE));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp32(n.lhs, n.rhs)), build(X86Setne(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::L));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp32(n.lhs, n.rhs)), build(X86Setl(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::LE));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp32(n.lhs, n.rhs)), build(X86Setle(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::G));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp32(n.lhs, n.rhs)), build(X86Setg(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::GE));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp32(n.lhs, n.rhs)), build(X86Setge(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::B));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp32(n.lhs, n.rhs)), build(X86Setb(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::BE));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp32(n.lhs, n.rhs)), build(X86Setbe(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::A));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp32(n.lhs, n.rhs)), build(X86Seta(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.lhs));
            require(matches(n.cc, CC::AE));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp32(n.lhs, n.rhs)), build(X86Setae(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::E));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp64(n.lhs, n.rhs)), build(X86Sete(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::NE));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp64(n.lhs, n.rhs)), build(X86Setne(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::L));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp64(n.lhs, n.rhs)), build(X86Setl(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::LE));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp64(n.lhs, n.rhs)), build(X86Setle(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::G));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp64(n.lhs, n.rhs)), build(X86Setg(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::GE));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp64(n.lhs, n.rhs)), build(X86Setge(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::B));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp64(n.lhs, n.rhs)), build(X86Setb(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::BE));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp64(n.lhs, n.rhs)), build(X86Setbe(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::A));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp64(n.lhs, n.rhs)), build(X86Seta(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.lhs));
            require(matches(n.cc, CC::AE));
            let bit = temp(Type::I8);
            replace(n, [build(X86Cmp64(n.lhs, n.rhs)), build(X86Setae(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
    }
}

// Ordered comparisons exclude NaN; inequality includes unordered inputs.
template FloatCompare(InputType: expr, Compare: ident, Condition: expr, Set: ident, Order: ident, Combine: ident) {
    select(n: lir::Fcmp) {
        choose {
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<InputType>(n.lhs));
            require(matches(n.cc, Condition));
            let bit = temp(Type::I8);
            let ordered_bit = temp(Type::I8);
            let predicate = temp(Type::I32);
            let ordered = temp(Type::I32);
            replace(n, [
                build(Compare(n.lhs, n.rhs)),
                build(Set(bit)),
                build(X86Movzx8to32(predicate, bit)),
                build(Order(ordered_bit)),
                build(X86Movzx8to32(ordered, ordered_bit)),
                build(Combine(n.dst, ordered, predicate)),
            ]);
        }
        }
    }
}
expand FloatCompare(Type::F32, X86Ucomiss, CC::E, X86Sete, X86Setnp, X86And32);
expand FloatCompare(Type::F32, X86Ucomiss, CC::NE, X86Setne, X86Setp, X86Or32);
expand FloatCompare(Type::F32, X86Ucomiss, CC::B, X86Setb, X86Setnp, X86And32);
expand FloatCompare(Type::F32, X86Ucomiss, CC::BE, X86Setbe, X86Setnp, X86And32);
expand FloatCompare(Type::F64, X86Ucomisd, CC::E, X86Sete, X86Setnp, X86And32);
expand FloatCompare(Type::F64, X86Ucomisd, CC::NE, X86Setne, X86Setp, X86Or32);
expand FloatCompare(Type::F64, X86Ucomisd, CC::B, X86Setb, X86Setnp, X86And32);
expand FloatCompare(Type::F64, X86Ucomisd, CC::BE, X86Setbe, X86Setnp, X86And32);

select(n: lir::Fcmp) {
    choose {
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<Type::F32>(n.lhs));
            require(matches(n.cc, CC::A));
            let bit = temp(Type::I8);
            replace(n, [build(X86Ucomiss(n.lhs, n.rhs)), build(X86Seta(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<Type::F32>(n.lhs));
            require(matches(n.cc, CC::AE));
            let bit = temp(Type::I8);
            replace(n, [build(X86Ucomiss(n.lhs, n.rhs)), build(X86Setae(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<Type::F64>(n.lhs));
            require(matches(n.cc, CC::A));
            let bit = temp(Type::I8);
            replace(n, [build(X86Ucomisd(n.lhs, n.rhs)), build(X86Seta(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<Type::F64>(n.lhs));
            require(matches(n.cc, CC::AE));
            let bit = temp(Type::I8);
            replace(n, [build(X86Ucomisd(n.lhs, n.rhs)), build(X86Setae(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
    }
}

// Select a value using false ^ ((true ^ false) & -bool(cond)).
template Select32(ResultType: expr, CondType: expr, Test: ident) {
    select(n: lir::Select) {
        choose {
        case {
            require(type_is<ResultType>(n.dst));
            require(type_is<CondType>(n.cond));
            let cond_byte = temp(Type::I8);
            let cond32 = temp(Type::I32);
            let zero = temp(Type::I32);
            let mask = temp(Type::I32);
            let diff = temp(Type::I32);
            let masked = temp(Type::I32);
            replace(n, [
                build(Test(n.cond, n.cond)),
                build(X86Setne(cond_byte)),
                build(X86Movzx8to32(cond32, cond_byte)),
                build(X86Mov32Imm(zero, 0)),
                build(X86Sub32(mask, cond32, zero)),
                build(X86Xor32(diff, n.v1, n.v2)),
                build(X86And32(masked, diff, mask)),
                build(X86Xor32(n.dst, n.v2, masked)),
            ]);
        }
        }
    }
}

template Select64(ResultType: expr, CondType: expr, Test: ident) {
    select(n: lir::Select) {
        choose {
        case {
            require(type_is<ResultType>(n.dst));
            require(type_is<CondType>(n.cond));
            let cond_byte = temp(Type::I8);
            let cond32 = temp(Type::I32);
            let zero = temp(Type::I64);
            let mask = temp(Type::I64);
            let diff = temp(Type::I64);
            let masked = temp(Type::I64);
            let wide = temp(Type::I64);
            replace(n, [
                build(Test(n.cond, n.cond)),
                build(X86Setne(cond_byte)),
                build(X86Movzx8to32(cond32, cond_byte)),
                build(X86Mov32(wide, cond32)),
                build(X86Mov64Imm32(zero, 0)),
                build(X86Sub64(mask, wide, zero)),
                build(X86Xor64(diff, n.v1, n.v2)),
                build(X86And64(masked, diff, mask)),
                build(X86Xor64(n.dst, n.v2, masked)),
            ]);
        }
        }
    }
}

template SelectF32(ResultType: expr, CondType: expr, Test: ident) {
    select(n: lir::Select) {
        choose {
        case {
            require(type_is<ResultType>(n.dst));
            require(type_is<CondType>(n.cond));
            let cond_byte = temp(Type::I8);
            let cond32 = temp(Type::I32);
            let zero = temp(Type::I32);
            let mask = temp(Type::I32);
            let diff = temp(Type::I32);
            let masked = temp(Type::I32);
            let true_bits = temp(Type::I32);
            let false_bits = temp(Type::I32);
            let result_bits = temp(Type::I32);
            replace(n, [
                build(Test(n.cond, n.cond)),
                build(X86Setne(cond_byte)),
                build(X86Movzx8to32(cond32, cond_byte)),
                build(X86MovdFromXmm(true_bits, n.v1)),
                build(X86MovdFromXmm(false_bits, n.v2)),
                build(X86Mov32Imm(zero, 0)),
                build(X86Sub32(mask, cond32, zero)),
                build(X86Xor32(diff, true_bits, false_bits)),
                build(X86And32(masked, diff, mask)),
                build(X86Xor32(result_bits, false_bits, masked)),
                build(X86MovdToXmm(n.dst, result_bits)),
            ]);
        }
        }
    }
}

template SelectF64(ResultType: expr, CondType: expr, Test: ident) {
    select(n: lir::Select) {
        choose {
        case {
            require(type_is<ResultType>(n.dst));
            require(type_is<CondType>(n.cond));
            let cond_byte = temp(Type::I8);
            let cond32 = temp(Type::I32);
            let zero = temp(Type::I64);
            let mask = temp(Type::I64);
            let diff = temp(Type::I64);
            let masked = temp(Type::I64);
            let wide = temp(Type::I64);
            let true_bits = temp(Type::I64);
            let false_bits = temp(Type::I64);
            let result_bits = temp(Type::I64);
            replace(n, [
                build(Test(n.cond, n.cond)),
                build(X86Setne(cond_byte)),
                build(X86Movzx8to32(cond32, cond_byte)),
                build(X86Mov32(wide, cond32)),
                build(X86MovqFromXmm(true_bits, n.v1)),
                build(X86MovqFromXmm(false_bits, n.v2)),
                build(X86Mov64Imm32(zero, 0)),
                build(X86Sub64(mask, wide, zero)),
                build(X86Xor64(diff, true_bits, false_bits)),
                build(X86And64(masked, diff, mask)),
                build(X86Xor64(result_bits, false_bits, masked)),
                build(X86MovqToXmm(n.dst, result_bits)),
            ]);
        }
        }
    }
}

expand Select32(SmallInt | Type::BOOL, SmallInt | Type::BOOL, X86Test32);
expand Select32(SmallInt | Type::BOOL, WordOrPtr, X86Test64);
expand Select64(WordOrPtr, SmallInt | Type::BOOL, X86Test32);
expand Select64(WordOrPtr, WordOrPtr, X86Test64);
expand SelectF32(Type::F32, SmallInt | Type::BOOL, X86Test32);
expand SelectF32(Type::F32, WordOrPtr, X86Test64);
expand SelectF64(Type::F64, SmallInt | Type::BOOL, X86Test32);
expand SelectF64(Type::F64, WordOrPtr, X86Test64);

select(n: lir::Ieqz) {
    choose {
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<SmallInt>(n.src));
            let bit = temp(Type::I8);
            replace(n, [build(X86Test32(n.src, n.src)), build(X86Sete(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
        case {
            require(type_is<Type::BOOL>(n.dst));
            require(type_is<WordOrPtr>(n.src));
            let bit = temp(Type::I8);
            replace(n, [build(X86Test64(n.src, n.src)), build(X86Sete(bit)), build(X86Movzx8to32(n.dst, bit))]);
        }
    }
}

select(n: lir::Br) {
    choose {
        case {
            replace(n, build(X86Jmp(n.target)));
        }
    }
}

select(n: lir::Brcond) {
    choose {
        case {
            replace(n, [build(X86Test32(n.cond, n.cond)), build(X86Jne(n.then_blk)), build(X86Jmp(n.else_blk))]);
        }
    }
}

select(n: lir::Ret) {
    choose {
        case {
            replace(n, build(X86Ret()));
        }
    }
}

select(n: lir::Trap) {
    choose {
        case {
            replace(n, build(X86Ud2()));
        }
    }
}

select(n: lir::Sdiv) {
    choose {
        case {
            require(type_is<Type::I64>(n.rhs));
            replace(n, [build(X86Mov64(reg(RAX), n.lhs)), build(X86Cqo()), build(X86IDiv64(n.rhs)), build(X86Mov64(n.dst, reg(RAX)))]);
        }
        case {
            require(type_is<Type::I32>(n.rhs));
            replace(n, [build(X86Mov32(reg(RAX), n.lhs)), build(X86Cdq()), build(X86IDiv32(n.rhs)), build(X86Mov32(n.dst, reg(RAX)))]);
        }
    }
}

select(n: lir::Srem) {
    choose {
        case {
            require(type_is<Type::I64>(n.rhs));
            replace(n, [build(X86Mov64(reg(RAX), n.lhs)), build(X86Cqo()), build(X86IDiv64(n.rhs)), build(X86Mov64(n.dst, reg(RDX)))]);
        }
        case {
            require(type_is<Type::I32>(n.rhs));
            replace(n, [build(X86Mov32(reg(RAX), n.lhs)), build(X86Cdq()), build(X86IDiv32(n.rhs)), build(X86Mov32(n.dst, reg(RDX)))]);
        }
    }
}

select(n: lir::Udiv) {
    choose {
        case {
            require(type_is<Type::I64>(n.rhs));
            replace(n, [build(X86Mov64(reg(RAX), n.lhs)), build(X86Xor64(reg(RDX), reg(RDX), reg(RDX))), build(X86Div64(n.rhs)), build(X86Mov64(n.dst, reg(RAX)))]);
        }
        case {
            require(type_is<Type::I32>(n.rhs));
            replace(n, [build(X86Mov32(reg(RAX), n.lhs)), build(X86Xor32(reg(RDX), reg(RDX), reg(RDX))), build(X86Div32(n.rhs)), build(X86Mov32(n.dst, reg(RAX)))]);
        }
    }
}

select(n: lir::Urem) {
    choose {
        case {
            require(type_is<Type::I64>(n.rhs));
            replace(n, [build(X86Mov64(reg(RAX), n.lhs)), build(X86Xor64(reg(RDX), reg(RDX), reg(RDX))), build(X86Div64(n.rhs)), build(X86Mov64(n.dst, reg(RDX)))]);
        }
        case {
            require(type_is<Type::I32>(n.rhs));
            replace(n, [build(X86Mov32(reg(RAX), n.lhs)), build(X86Xor32(reg(RDX), reg(RDX), reg(RDX))), build(X86Div32(n.rhs)), build(X86Mov32(n.dst, reg(RDX)))]);
        }
    }
}

select(n: lir::Fadd) {
    choose {
        case {
            require(type_is<Type::F32>(n.dst));
            replace(n, build(X86FAdd32(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            replace(n, build(X86FAdd64(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Fsub) {
    choose {
        case {
            require(type_is<Type::F32>(n.dst));
            replace(n, build(X86FSub32(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            replace(n, build(X86FSub64(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Fmul) {
    choose {
        case {
            require(type_is<Type::F32>(n.dst));
            replace(n, build(X86FMul32(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            replace(n, build(X86FMul64(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Fdiv) {
    choose {
        case {
            require(type_is<Type::F32>(n.dst));
            replace(n, build(X86FDiv32(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            replace(n, build(X86FDiv64(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::StackAddr) {
    choose {
        case {
            require(type_is<Type::PTR>(n.dst));
            replace(n, build(X86LeaStack(n.dst, n.slot)));
        }
    }
}

select(n: lir::Rotl) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(X86Rol32Cl(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86Rol64Cl(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Rotr) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(X86Ror32Cl(n.rhs, n.lhs)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86Ror64Cl(n.rhs, n.lhs)));
        }
    }
}

select(n: lir::Sitofp) {
    choose {
        case {
            require(type_is<Type::F32>(n.dst));
            require(type_is<Type::I32>(n.src));
            replace(n, build(X86I32ToF32(n.src)));
        }
        case {
            require(type_is<Type::F32>(n.dst));
            require(type_is<Type::I64>(n.src));
            replace(n, build(X86I64ToF32(n.src)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            require(type_is<Type::I32>(n.src));
            replace(n, build(X86I32ToF64(n.src)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            require(type_is<Type::I64>(n.src));
            replace(n, build(X86I64ToF64(n.src)));
        }
    }
}

select(n: lir::Fptosi) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::F32>(n.src));
            replace(n, build(X86F32ToI32(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::F32>(n.src));
            replace(n, build(X86F32ToI64(n.src)));
        }
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::F64>(n.src));
            replace(n, build(X86F64ToI32(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::F64>(n.src));
            replace(n, build(X86F64ToI64(n.src)));
        }
    }
}

select(n: lir::Fpext) {
    choose {
        case {
            require(type_is<Type::F64>(n.dst));
            require(type_is<Type::F32>(n.src));
            replace(n, build(X86F32ToF64(n.src)));
        }
    }
}

select(n: lir::Fptrunc) {
    choose {
        case {
            require(type_is<Type::F32>(n.dst));
            require(type_is<Type::F64>(n.src));
            replace(n, build(X86F64ToF32(n.src)));
        }
    }
}

select(n: lir::Fsqrt) {
    choose {
        case {
            require(type_is<Type::F32>(n.dst));
            replace(n, build(X86SqrtF32(n.src)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            replace(n, build(X86SqrtF64(n.src)));
        }
    }
}

select(n: lir::Ctpop) {
    choose {
        case {
            require(type_is<Type::I32>(n.dst));
            replace(n, build(X86Popcnt32(n.src)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            replace(n, build(X86Popcnt64(n.src)));
        }
    }
}
