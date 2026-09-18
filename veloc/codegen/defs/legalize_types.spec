type Type = rust("veloc_mir::Type") {
    const BOOL: Self;
    const I8: Self;
    const I16: Self;
    const I32: Self;
    const I64: Self;
    const F32: Self;
    const F64: Self;
    const PTR: Self;
}

typeset Narrow = Type::I8 | Type::I16;
typeset Word = Type::I32 | Type::I64;
typeset Float = Type::F32 | Type::F64;
typeset Scalar = Type::BOOL | Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::F32 | Type::F64 | Type::PTR;
typeset IntOrPtr = Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::PTR;
typeset WordValue = Type::BOOL | Type::I32 | Type::I64 | Type::F32 | Type::F64 | Type::PTR;
typeset Number = Type::I32 | Type::I64 | Type::F32 | Type::F64;
typeset WordOrPtr = Type::I32 | Type::I64 | Type::PTR;
typeset SmallInt = Type::I8 | Type::I16 | Type::I32;
