import "../../../defs/type_sets.spec";

data_layout X86_64 {
    endian = little;
    pointer = { size: 8, align: 8 };
    types = [
        { types: Type::BOOL | Type::I8, size: 1, align: 1 },
        { types: Type::I16, size: 2, align: 2 },
        { types: Type::I32 | Type::F32, size: 4, align: 4 },
        { types: Type::I64 | Type::F64, size: 8, align: 8 },
        {
            types: Type::I8X16 | Type::I16X8 | Type::I32X4 | Type::I64X2 | Type::F32X4 | Type::F64X2,
            size: 16,
            align: 16,
        },
    ];
}
