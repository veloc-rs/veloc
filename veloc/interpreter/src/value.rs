use veloc_mir::Type;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(transparent)]
pub struct InterpreterValue(pub u64);

impl InterpreterValue {
    #[inline(always)]
    pub fn i32(v: i32) -> Self {
        Self(v as u32 as u64)
    }

    #[inline(always)]
    pub fn i64(v: i64) -> Self {
        Self(v as u64)
    }

    #[inline(always)]
    pub fn f32(v: f32) -> Self {
        Self(v.to_bits() as u64)
    }

    #[inline(always)]
    pub fn f64(v: f64) -> Self {
        Self(v.to_bits())
    }

    #[inline(always)]
    pub fn bool(v: bool) -> Self {
        Self(v as u64)
    }

    #[inline(always)]
    pub fn none() -> Self {
        Self(0)
    }

    #[inline(always)]
    pub fn unwrap_i32(self) -> i32 {
        self.0 as i32
    }

    #[inline(always)]
    pub fn unwrap_i64(self) -> i64 {
        self.0 as i64
    }

    #[inline(always)]
    pub fn unwrap_f32(self) -> f32 {
        f32::from_bits(self.0 as u32)
    }

    #[inline(always)]
    pub fn unwrap_f64(self) -> f64 {
        f64::from_bits(self.0)
    }

    #[inline(always)]
    pub fn unwrap_bool(self) -> bool {
        self.0 != 0
    }

    #[inline(always)]
    pub fn to_i64_bits(self) -> i64 {
        self.0 as i64
    }

    pub fn from_i64(v: i64, res_ty: Type) -> Self {
        match res_ty {
            Type::I8 => InterpreterValue::i32((v as i8) as i32),
            Type::I16 => InterpreterValue::i32((v as i16) as i32),
            Type::I32 => InterpreterValue::i32(v as i32),
            Type::I64 | Type::PTR => InterpreterValue::i64(v),
            Type::F32 => InterpreterValue::f32(f32::from_bits(v as u32)),
            Type::F64 => InterpreterValue::f64(f64::from_bits(v as u64)),
            Type::BOOL => InterpreterValue::bool(v != 0),
            _ => InterpreterValue::none(),
        }
    }
}
