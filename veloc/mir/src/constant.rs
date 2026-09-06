use crate::{InstructionData, Type};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Constant {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
}

impl Constant {
    pub fn ty(self) -> Type {
        match self {
            Self::I8(_) => Type::I8,
            Self::I16(_) => Type::I16,
            Self::I32(_) => Type::I32,
            Self::I64(_) => Type::I64,
            Self::F32(_) => Type::F32,
            Self::F64(_) => Type::F64,
            Self::Bool(_) => Type::BOOL,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Constant::I8(v) => Some(*v as i64),
            Constant::I16(v) => Some(*v as i64),
            Constant::I32(v) => Some(*v as i64),
            Constant::I64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Constant::F32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Constant::F32(v) => Some(*v as f64),
            Constant::F64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Constant::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn bits(&self) -> u32 {
        match self {
            Constant::I8(_) => 8,
            Constant::I16(_) => 16,
            Constant::I32(_) | Constant::F32(_) => 32,
            Constant::I64(_) | Constant::F64(_) => 64,
            Constant::Bool(_) => 1,
        }
    }
}

impl From<Constant> for InstructionData {
    fn from(c: Constant) -> Self {
        use crate::constant::Constant;
        match c {
            Constant::I8(v) => InstructionData::Iconst { value: v as u64 },
            Constant::I16(v) => InstructionData::Iconst { value: v as u64 },
            Constant::I32(v) => InstructionData::Iconst { value: v as u64 },
            Constant::I64(v) => InstructionData::Iconst { value: v as u64 },
            Constant::F32(v) => InstructionData::Fconst {
                value: v.to_bits() as u64,
            },
            Constant::F64(v) => InstructionData::Fconst { value: v.to_bits() },
            Constant::Bool(v) => InstructionData::Bconst { value: v },
        }
    }
}
