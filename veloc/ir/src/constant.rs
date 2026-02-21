use crate::{InstructionData, IntCC, Opcode};

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

    pub fn bits(&self) -> u32 {
        match self {
            Constant::I8(_) => 8,
            Constant::I16(_) => 16,
            Constant::I32(_) | Constant::F32(_) => 32,
            Constant::I64(_) | Constant::F64(_) => 64,
            Constant::Bool(_) => 1,
        }
    }

    pub fn binary_op(self, other: Self, op: Opcode) -> Option<Self> {
        let l_val = self.as_i64()?;
        let r_val = other.as_i64()?;
        let bits = self.bits();

        if bits != other.bits() {
            return None;
        }

        let result = match op {
            Opcode::IAdd => l_val.wrapping_add(r_val),
            Opcode::ISub => l_val.wrapping_sub(r_val),
            Opcode::IMul => l_val.wrapping_mul(r_val),
            Opcode::IDivS => {
                if r_val == 0 || (l_val == i64::MIN && r_val == -1) {
                    return None;
                }
                l_val.wrapping_div(r_val)
            }
            Opcode::IDivU => {
                if r_val == 0 {
                    return None;
                }
                let l = truncate_u64(l_val as u64, bits);
                let r = truncate_u64(r_val as u64, bits);
                l.wrapping_div(r) as i64
            }
            Opcode::IAnd => l_val & r_val,
            Opcode::IOr => l_val | r_val,
            Opcode::IXor => l_val ^ r_val,
            Opcode::IShl => l_val.wrapping_shl((r_val as u32) % bits),
            Opcode::IShrS => {
                if bits == 32 {
                    (l_val as i32).wrapping_shr((r_val as u32) % 32) as i64
                } else {
                    l_val.wrapping_shr((r_val as u32) % 64)
                }
            }
            Opcode::IShrU => {
                let l = truncate_u64(l_val as u64, bits);
                l.wrapping_shr((r_val as u32) % bits) as i64
            }
            _ => return None,
        };

        Some(Self::from_i64(result, bits))
    }

    pub fn unary_op(self, op: Opcode) -> Option<Self> {
        let v = self.as_i64()?;
        let bits = self.bits();

        let result = match op {
            Opcode::INeg => v.wrapping_neg(),
            Opcode::IClz => {
                let v_u = truncate_u64(v as u64, bits);
                v_u.leading_zeros().saturating_sub(64 - bits) as i64
            }
            Opcode::ICtz => {
                let v_u = truncate_u64(v as u64, bits);
                v_u.trailing_zeros().min(bits) as i64
            }
            Opcode::IPopcnt => {
                let v_u = truncate_u64(v as u64, bits);
                v_u.count_ones() as i64
            }
            Opcode::IEqz => return Some(Constant::Bool(v == 0)),
            _ => return None,
        };

        Some(Self::from_i64(result, bits))
    }

    pub fn icmp(self, other: Self, kind: IntCC) -> Option<Self> {
        let l = self.as_i64()?;
        let r = other.as_i64()?;
        let bits = self.bits();

        if bits != other.bits() {
            return None;
        }

        let result = if kind.is_unsigned() {
            let lu = truncate_u64(l as u64, bits);
            let ru = truncate_u64(r as u64, bits);
            match kind {
                IntCC::Eq => lu == ru,
                IntCC::Ne => lu != ru,
                IntCC::LtU => lu < ru,
                IntCC::LeU => lu <= ru,
                IntCC::GtU => lu > ru,
                IntCC::GeU => lu >= ru,
                _ => return None,
            }
        } else {
            let ls = truncate_i64(l, bits);
            let rs = truncate_i64(r, bits);
            match kind {
                IntCC::Eq => ls == rs,
                IntCC::Ne => ls != rs,
                IntCC::LtS => ls < rs,
                IntCC::LeS => ls <= rs,
                IntCC::GtS => ls > rs,
                IntCC::GeS => ls >= rs,
                _ => return None,
            }
        };

        Some(Constant::Bool(result))
    }

    fn from_i64(val: i64, bits: u32) -> Self {
        let val = truncate_i64(val, bits);
        match bits {
            8 => Constant::I8(val as i8),
            16 => Constant::I16(val as i16),
            32 => Constant::I32(val as i32),
            64 => Constant::I64(val),
            _ => panic!("Unsupported bit width: {}", bits),
        }
    }
}

fn truncate_u64(val: u64, bits: u32) -> u64 {
    if bits == 64 {
        val
    } else {
        val & ((1u64 << bits) - 1)
    }
}

fn truncate_i64(val: i64, bits: u32) -> i64 {
    if bits == 64 {
        val
    } else {
        let mask = (1u64 << bits) - 1;
        let truncated = (val as u64) & mask;
        let sign_bit = 1u64 << (bits - 1);
        if truncated & sign_bit != 0 {
            (truncated | !mask) as i64
        } else {
            truncated as i64
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
