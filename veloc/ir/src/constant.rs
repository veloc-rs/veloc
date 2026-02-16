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
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Constant::I8(v) => Some(*v as i64),
            Constant::I16(v) => Some(*v as i64),
            Constant::I32(v) => Some(*v as i64),
            Constant::I64(v) => Some(*v),
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
}
