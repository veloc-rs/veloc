import "../common.spec";

template Popcount(Opcode: ident, Wide: expr) {
    op Opcode(src: Value<GprValue>) -> (dst: Value<GprValue>) {
        requires = ["POPCNT"];
        encoding = Emission::legacy(
            Legacy { prefix: Prefix::F3, map: OpcodeMap::Map0F, opcode: 0xB8, wide: Wide },
            Form::ModRm(RegField::Register(dst), Rm::Register(src)),
            Immediate::None,
        );
        registers = { dst: GPR64, src: GPR64 };
        implicit = { clobbers: [EFLAGS] };
        schedule = { latency: 3 };
    }
}

expand Popcount(X86Popcnt32, false);

expand Asm(X86Popcnt32, "popcnt", [reg(dst, 32), reg(src, 32)]);

expand Popcount(X86Popcnt64, true);

expand Asm(X86Popcnt64, "popcnt", [reg(dst, 64), reg(src, 64)]);

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
