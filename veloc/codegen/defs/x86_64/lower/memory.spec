import "../common.spec";

fn memory(base: Reg, offset: i64) -> Rm {
    value = Rm::Memory(Address::BaseIndex(Memory {
        base: some(base), index: none, displacement: offset,
    }));
}

op X86Load8U32(base: Value<AddressValue>, off: i64, flags: MemFlags) -> (dst: Value<GprValue>) {
    schedule = Load;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Map0F, opcode: 0xB6, wide: false },
        Form::ModRm(RegField::Register(dst), memory(base, off)),
        Immediate::None,
    );
    registers = {
        dst: GPR64,
        base: GPR64,
    };
    memory = { kind: Read, bytes: 1 };
    assembly = {
        lines: [{ mnemonic: "movzx", operands: [reg(dst, 32), mem(base, off, 8)] }]
    };
}

op X86Load16U32(base: Value<AddressValue>, off: i64, flags: MemFlags) -> (dst: Value<GprValue>) {
    schedule = Load;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Map0F, opcode: 0xB7, wide: false },
        Form::ModRm(RegField::Register(dst), memory(base, off)),
        Immediate::None,
    );
    registers = {
        dst: GPR64,
        base: GPR64,
    };
    memory = { kind: Read, bytes: 2 };
    assembly = {
        lines: [{ mnemonic: "movzx", operands: [reg(dst, 32), mem(base, off, 16)] }]
    };
}

op X86Load32(base: Value<AddressValue>, off: i64, flags: MemFlags) -> (dst: Value<GprValue>) {
    schedule = Load;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0x8B, wide: false },
        Form::ModRm(RegField::Register(dst), memory(base, off)),
        Immediate::None,
    );
    registers = {
        dst: GPR64,
        base: GPR64,
    };
    memory = { kind: Read, bytes: 4 };
    assembly = {
        lines: [{ mnemonic: "mov", operands: [reg(dst, 32), mem(base, off, 32)] }]
    };
}

op X86Load64(base: Value<AddressValue>, off: i64, flags: MemFlags) -> (dst: Value<GprValue>) {
    schedule = Load;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0x8B, wide: true },
        Form::ModRm(RegField::Register(dst), memory(base, off)),
        Immediate::None,
    );
    registers = {
        dst: GPR64,
        base: GPR64,
    };
    memory = { kind: Read, bytes: 8 };
    assembly = {
        lines: [{ mnemonic: "mov", operands: [reg(dst, 64), mem(base, off, 64)] }]
    };
}

op X86LoadF32(base: Value<AddressValue>, off: i64, flags: MemFlags) -> (dst: Value<Type::F32>) {
    schedule = Load;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::F3, map: OpcodeMap::Map0F, opcode: 0x10, wide: false },
        Form::ModRm(RegField::Register(dst), memory(base, off)),
        Immediate::None,
    );
    registers = {
        dst: FPR128,
        base: GPR64,
    };
    memory = { kind: Read, bytes: 4 };
    assembly = {
        lines: [{ mnemonic: "movss", operands: [reg(dst, 128), mem(base, off, 32)] }]
    };
}

op X86LoadF64(base: Value<AddressValue>, off: i64, flags: MemFlags) -> (dst: Value<Type::F64>) {
    schedule = Load;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::F2, map: OpcodeMap::Map0F, opcode: 0x10, wide: false },
        Form::ModRm(RegField::Register(dst), memory(base, off)),
        Immediate::None,
    );
    registers = {
        dst: FPR128,
        base: GPR64,
    };
    memory = { kind: Read, bytes: 8 };
    assembly = {
        lines: [{ mnemonic: "movsd", operands: [reg(dst, 128), mem(base, off, 64)] }]
    };
}

op X86Store8(src: Value<GprValue>, base: Value<AddressValue>, off: i64, flags: MemFlags) -> () {
    schedule = Store;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0x88, wide: false },
        Form::ModRm(RegField::ByteRegister(src), memory(base, off)),
        Immediate::None,
    );
    registers = {
        src: GPR64,
        base: GPR64,
    };
    memory = { kind: Write, bytes: 1 };
    assembly = {
        lines: [{ mnemonic: "mov", operands: [mem(base, off, 8), reg(src, 8)] }]
    };
}

op X86Store16(src: Value<GprValue>, base: Value<AddressValue>, off: i64, flags: MemFlags) -> () {
    schedule = Store;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::P66, map: OpcodeMap::Primary, opcode: 0x89, wide: false },
        Form::ModRm(RegField::Register(src), memory(base, off)),
        Immediate::None,
    );
    registers = {
        src: GPR64,
        base: GPR64,
    };
    memory = { kind: Write, bytes: 2 };
    assembly = {
        lines: [{ mnemonic: "mov", operands: [mem(base, off, 16), reg(src, 16)] }]
    };
}

op X86Store32(src: Value<GprValue>, base: Value<AddressValue>, off: i64, flags: MemFlags) -> () {
    schedule = Store;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0x89, wide: false },
        Form::ModRm(RegField::Register(src), memory(base, off)),
        Immediate::None,
    );
    registers = {
        src: GPR64,
        base: GPR64,
    };
    memory = { kind: Write, bytes: 4 };
    assembly = {
        lines: [{ mnemonic: "mov", operands: [mem(base, off, 32), reg(src, 32)] }]
    };
}

op X86Store64(src: Value<GprValue>, base: Value<AddressValue>, off: i64, flags: MemFlags) -> () {
    schedule = Store;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0x89, wide: true },
        Form::ModRm(RegField::Register(src), memory(base, off)),
        Immediate::None,
    );
    registers = {
        src: GPR64,
        base: GPR64,
    };
    memory = { kind: Write, bytes: 8 };
    assembly = {
        lines: [{ mnemonic: "mov", operands: [mem(base, off, 64), reg(src, 64)] }]
    };
}

op X86StoreF32(src: Value<Type::F32>, base: Value<AddressValue>, off: i64, flags: MemFlags) -> () {
    schedule = Store;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::F3, map: OpcodeMap::Map0F, opcode: 0x11, wide: false },
        Form::ModRm(RegField::Register(src), memory(base, off)),
        Immediate::None,
    );
    registers = {
        src: FPR128,
        base: GPR64,
    };
    memory = { kind: Write, bytes: 4 };
    assembly = {
        lines: [{ mnemonic: "movss", operands: [mem(base, off, 32), reg(src, 128)] }]
    };
}

op X86StoreF64(src: Value<Type::F64>, base: Value<AddressValue>, off: i64, flags: MemFlags) -> () {
    schedule = Store;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::F2, map: OpcodeMap::Map0F, opcode: 0x11, wide: false },
        Form::ModRm(RegField::Register(src), memory(base, off)),
        Immediate::None,
    );
    registers = {
        src: FPR128,
        base: GPR64,
    };
    memory = { kind: Write, bytes: 8 };
    assembly = {
        lines: [{ mnemonic: "movsd", operands: [mem(base, off, 64), reg(src, 128)] }]
    };
}

op X86Load8U32Stack(slot: StackSlot, flags: MemFlags) -> (dst: Value<GprValue>) {
    schedule = Load;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Map0F, opcode: 0xB6, wide: false },
        Form::ModRm(RegField::Register(dst), Rm::Memory(slot)),
        Immediate::None,
    );
    registers = {
        dst: GPR64,
    };
    memory = { kind: Read, bytes: 1 };
    assembly = {
        lines: [{ mnemonic: "movzx", operands: [reg(dst, 32), stack(slot, 8)] }]
    };
}

op X86Load16U32Stack(slot: StackSlot, flags: MemFlags) -> (dst: Value<GprValue>) {
    schedule = Load;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Map0F, opcode: 0xB7, wide: false },
        Form::ModRm(RegField::Register(dst), Rm::Memory(slot)),
        Immediate::None,
    );
    registers = {
        dst: GPR64,
    };
    memory = { kind: Read, bytes: 2 };
    assembly = {
        lines: [{ mnemonic: "movzx", operands: [reg(dst, 32), stack(slot, 16)] }]
    };
}

op X86Load32Stack(slot: StackSlot, flags: MemFlags) -> (dst: Value<GprValue>) {
    schedule = Load;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0x8B, wide: false },
        Form::ModRm(RegField::Register(dst), Rm::Memory(slot)),
        Immediate::None,
    );
    registers = {
        dst: GPR64,
    };
    memory = { kind: Read, bytes: 4 };
    assembly = {
        lines: [{ mnemonic: "mov", operands: [reg(dst, 32), stack(slot, 32)] }]
    };
}

op X86Load64Stack(slot: StackSlot, flags: MemFlags) -> (dst: Value<GprValue>) {
    schedule = Load;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0x8B, wide: true },
        Form::ModRm(RegField::Register(dst), Rm::Memory(slot)),
        Immediate::None,
    );
    registers = {
        dst: GPR64,
    };
    memory = { kind: Read, bytes: 8 };
    assembly = {
        lines: [{ mnemonic: "mov", operands: [reg(dst, 64), stack(slot, 64)] }]
    };
}

op X86LoadF32Stack(slot: StackSlot, flags: MemFlags) -> (dst: Value<Type::F32>) {
    schedule = Load;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::F3, map: OpcodeMap::Map0F, opcode: 0x10, wide: false },
        Form::ModRm(RegField::Register(dst), Rm::Memory(slot)),
        Immediate::None,
    );
    registers = {
        dst: FPR128,
    };
    memory = { kind: Read, bytes: 4 };
    assembly = {
        lines: [{ mnemonic: "movss", operands: [reg(dst, 128), stack(slot, 32)] }]
    };
}

op X86LoadF64Stack(slot: StackSlot, flags: MemFlags) -> (dst: Value<Type::F64>) {
    schedule = Load;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::F2, map: OpcodeMap::Map0F, opcode: 0x10, wide: false },
        Form::ModRm(RegField::Register(dst), Rm::Memory(slot)),
        Immediate::None,
    );
    registers = {
        dst: FPR128,
    };
    memory = { kind: Read, bytes: 8 };
    assembly = {
        lines: [{ mnemonic: "movsd", operands: [reg(dst, 128), stack(slot, 64)] }]
    };
}

op X86Store8Stack(src: Value<GprValue>, slot: StackSlot, flags: MemFlags) -> () {
    schedule = Store;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0x88, wide: false },
        Form::ModRm(RegField::ByteRegister(src), Rm::Memory(slot)),
        Immediate::None,
    );
    registers = {
        src: GPR64,
    };
    memory = { kind: Write, bytes: 1 };
    assembly = {
        lines: [{ mnemonic: "mov", operands: [stack(slot, 8), reg(src, 8)] }]
    };
}

op X86Store16Stack(src: Value<GprValue>, slot: StackSlot, flags: MemFlags) -> () {
    schedule = Store;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::P66, map: OpcodeMap::Primary, opcode: 0x89, wide: false },
        Form::ModRm(RegField::Register(src), Rm::Memory(slot)),
        Immediate::None,
    );
    registers = {
        src: GPR64,
    };
    memory = { kind: Write, bytes: 2 };
    assembly = {
        lines: [{ mnemonic: "mov", operands: [stack(slot, 16), reg(src, 16)] }]
    };
}

op X86Store32Stack(src: Value<GprValue>, slot: StackSlot, flags: MemFlags) -> () {
    schedule = Store;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0x89, wide: false },
        Form::ModRm(RegField::Register(src), Rm::Memory(slot)),
        Immediate::None,
    );
    registers = {
        src: GPR64,
    };
    memory = { kind: Write, bytes: 4 };
    assembly = {
        lines: [{ mnemonic: "mov", operands: [stack(slot, 32), reg(src, 32)] }]
    };
}

op X86Store64Stack(src: Value<GprValue>, slot: StackSlot, flags: MemFlags) -> () {
    schedule = Store;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0x89, wide: true },
        Form::ModRm(RegField::Register(src), Rm::Memory(slot)),
        Immediate::None,
    );
    registers = {
        src: GPR64,
    };
    memory = { kind: Write, bytes: 8 };
    assembly = {
        lines: [{ mnemonic: "mov", operands: [stack(slot, 64), reg(src, 64)] }]
    };
}

op X86StoreF32Stack(src: Value<Type::F32>, slot: StackSlot, flags: MemFlags) -> () {
    schedule = Store;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::F3, map: OpcodeMap::Map0F, opcode: 0x11, wide: false },
        Form::ModRm(RegField::Register(src), Rm::Memory(slot)),
        Immediate::None,
    );
    registers = {
        src: FPR128,
    };
    memory = { kind: Write, bytes: 4 };
    assembly = {
        lines: [{ mnemonic: "movss", operands: [stack(slot, 32), reg(src, 128)] }]
    };
}

op X86StoreF64Stack(src: Value<Type::F64>, slot: StackSlot, flags: MemFlags) -> () {
    schedule = Store;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::F2, map: OpcodeMap::Map0F, opcode: 0x11, wide: false },
        Form::ModRm(RegField::Register(src), Rm::Memory(slot)),
        Immediate::None,
    );
    registers = {
        src: FPR128,
    };
    memory = { kind: Write, bytes: 8 };
    assembly = {
        lines: [{ mnemonic: "movsd", operands: [stack(slot, 64), reg(src, 128)] }]
    };
}

op X86LeaStack(slot: StackSlot) -> (dst: Value<GprValue>) {
    schedule = Address;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0x8D, wide: true },
        Form::ModRm(RegField::Register(dst), Rm::Memory(slot)),
        Immediate::None,
    );
    registers = {
        dst: GPR64,
    };
    assembly = {
        lines: [{ mnemonic: "lea", operands: [reg(dst, 64), stack(slot, 64)] }]
    };
}

fn indexed_memory(base: Reg, index: Reg, offset: i64) -> Rm {
    value = Rm::Memory(Address::BaseIndex(Memory {
        base: some(base), index: some(Index { reg: index, scale: Scale::One }), displacement: offset,
    }));
}

op X86Load64Index(base: Value<AddressValue>, index: Value<AddressValue>, off: i64, flags: MemFlags) -> (dst: Value<GprValue>) {
    schedule = Load;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0x8B, wide: true },
        Form::ModRm(RegField::Register(dst), indexed_memory(base, index, off)),
        Immediate::None,
    );
    registers = { dst: GPR64, base: GPR64, index: GPR64 };
    memory = { kind: Read, bytes: 8 };
    assembly = {
        lines: [{ mnemonic: "mov", operands: [reg(dst, 64), mem(base, index, off, 64)] }]
    };
}

op X86Store64Index(src: Value<GprValue>, base: Value<AddressValue>, index: Value<AddressValue>, off: i64, flags: MemFlags) -> () {
    schedule = Store;
    encoding = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: 0x89, wide: true },
        Form::ModRm(RegField::Register(src), indexed_memory(base, index, off)),
        Immediate::None,
    );
    registers = { src: GPR64, base: GPR64, index: GPR64 };
    memory = { kind: Write, bytes: 8 };
    assembly = {
        lines: [{ mnemonic: "mov", operands: [mem(base, index, off, 64), reg(src, 64)] }]
    };
}

select(n: lir::Load) {
    choose {
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::I8>(n.dst));
            require(matches(n.offset, 0));
            replace(n, build(X86Load8U32Stack(addr.slot, n.flags)));
        }
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::I16>(n.dst));
            require(matches(n.offset, 0));
            replace(n, build(X86Load16U32Stack(addr.slot, n.flags)));
        }
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::I32>(n.dst));
            require(matches(n.offset, 0));
            replace(n, build(X86Load32Stack(addr.slot, n.flags)));
        }
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::I64>(n.dst));
            require(matches(n.offset, 0));
            replace(n, build(X86Load64Stack(addr.slot, n.flags)));
        }
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::PTR>(n.dst));
            require(matches(n.offset, 0));
            replace(n, build(X86Load64Stack(addr.slot, n.flags)));
        }
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::F32>(n.dst));
            require(matches(n.offset, 0));
            replace(n, build(X86LoadF32Stack(addr.slot, n.flags)));
        }
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::F64>(n.dst));
            require(matches(n.offset, 0));
            replace(n, build(X86LoadF64Stack(addr.slot, n.flags)));
        }
        case {
            let addr = def<lir::Add<Type::I64>>(n.base);
            require(type_is<WordOrPtr>(n.dst));
            replace(n, build(X86Load64Index(addr.lhs, addr.rhs, n.offset, n.flags)));
        }
        case {
            let addr = def<lir::PtrAdd<Type::I64>>(n.base);
            require(type_is<WordOrPtr>(n.dst));
            require(type_is<Type::PTR>(addr.dst));
            require(type_is<Type::PTR>(addr.lhs));
            replace(n, build(X86Load64Index(addr.lhs, addr.rhs, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86Load64(n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::I64>(n.dst));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86Load64(n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::PTR>(n.dst));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86Load64(n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::PTR>(n.dst));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86Load64(n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86Load32(n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::I32>(n.dst));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86Load32(n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::F32>(n.dst));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86LoadF32(n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::F32>(n.dst));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86LoadF32(n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86LoadF64(n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::F64>(n.dst));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86LoadF64(n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::I16>(n.dst));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86Load16U32(n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::I16>(n.dst));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86Load16U32(n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::I8>(n.dst));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86Load8U32(n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::I8>(n.dst));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86Load8U32(n.base, n.offset, n.flags)));
        }
    }
}

select(n: lir::Store) {
    choose {
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::I8>(n.src));
            require(matches(n.offset, 0));
            replace(n, build(X86Store8Stack(n.src, addr.slot, n.flags)));
        }
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::I16>(n.src));
            require(matches(n.offset, 0));
            replace(n, build(X86Store16Stack(n.src, addr.slot, n.flags)));
        }
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::I32>(n.src));
            require(matches(n.offset, 0));
            replace(n, build(X86Store32Stack(n.src, addr.slot, n.flags)));
        }
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::I64>(n.src));
            require(matches(n.offset, 0));
            replace(n, build(X86Store64Stack(n.src, addr.slot, n.flags)));
        }
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::PTR>(n.src));
            require(matches(n.offset, 0));
            replace(n, build(X86Store64Stack(n.src, addr.slot, n.flags)));
        }
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::F32>(n.src));
            require(matches(n.offset, 0));
            replace(n, build(X86StoreF32Stack(n.src, addr.slot, n.flags)));
        }
        case {
            let addr = def<lir::StackAddr>(n.base);
            require(type_is<Type::F64>(n.src));
            require(matches(n.offset, 0));
            replace(n, build(X86StoreF64Stack(n.src, addr.slot, n.flags)));
        }
        case {
            let addr = def<lir::Add<Type::I64>>(n.base);
            require(type_is<WordOrPtr>(n.src));
            replace(n, build(X86Store64Index(n.src, addr.lhs, addr.rhs, n.offset, n.flags)));
        }
        case {
            let addr = def<lir::PtrAdd<Type::I64>>(n.base);
            require(type_is<WordOrPtr>(n.src));
            require(type_is<Type::PTR>(addr.dst));
            require(type_is<Type::PTR>(addr.lhs));
            replace(n, build(X86Store64Index(n.src, addr.lhs, addr.rhs, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::I64>(n.src));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86Store64(n.src, n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::I64>(n.src));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86Store64(n.src, n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::PTR>(n.src));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86Store64(n.src, n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::PTR>(n.src));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86Store64(n.src, n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::I32>(n.src));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86Store32(n.src, n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::I32>(n.src));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86Store32(n.src, n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::F32>(n.src));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86StoreF32(n.src, n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::F32>(n.src));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86StoreF32(n.src, n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::F64>(n.src));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86StoreF64(n.src, n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::F64>(n.src));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86StoreF64(n.src, n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::I16>(n.src));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86Store16(n.src, n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::I16>(n.src));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86Store16(n.src, n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::I8>(n.src));
            require(type_is<Type::PTR>(n.base));
            replace(n, build(X86Store8(n.src, n.base, n.offset, n.flags)));
        }
        case {
            require(type_is<Type::I8>(n.src));
            require(type_is<Type::I64>(n.base));
            replace(n, build(X86Store8(n.src, n.base, n.offset, n.flags)));
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
