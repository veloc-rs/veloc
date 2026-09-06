use crate::semantics::{IntPredicate, Outcome, Sort, Value as SemanticValue};
use crate::{InstructionData, IntCC, Opcode, Type};
use alloc::vec::Vec;
use smallvec::SmallVec;

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

    /// Fold scalar results through the same typed graph exported to SMT.
    /// Unsupported types/effects are not evaluated speculatively.
    pub fn evaluate(
        op: Opcode,
        args: &[Self],
        results: &[Type],
        properties: &[IntPredicate],
    ) -> Option<Vec<Self>> {
        let input_types = args.iter().map(|c| c.ty()).collect::<SmallVec<[Type; 4]>>();
        op.validate_types(&input_types, results).ok()?;
        let sort = |ty: Type| {
            if ty == Type::BOOL {
                Some(Sort::Bool)
            } else if ty.is_scalar() && ty.is_integer() {
                Sort::bv(ty.min_bit_width()? as u16).ok()
            } else {
                None
            }
        };
        let inputs = input_types
            .iter()
            .map(|&ty| sort(ty))
            .collect::<Option<SmallVec<[Sort; 4]>>>()?;
        let outputs = results
            .iter()
            .map(|&ty| sort(ty))
            .collect::<Option<SmallVec<[Sort; 2]>>>()?;
        let function = op
            .spec()
            .semantics?
            .instantiate(&inputs, &outputs, properties)
            .ok()?;
        let values = args
            .iter()
            .map(|c| match c {
                Self::Bool(b) => Some(SemanticValue::Bool(*b)),
                _ => c.as_i64().map(|v| SemanticValue::Bv(v as u128)),
            })
            .collect::<Option<SmallVec<[SemanticValue; 4]>>>()?;
        let Outcome::Values(values) = function.execute(&values).ok()? else {
            // A constant trap must remain an executable instruction, not become
            // a fabricated constant (nor an error in the compiler itself).
            return None;
        };
        values
            .into_iter()
            .zip(results)
            .map(|(v, &ty)| match v {
                SemanticValue::Bool(b) => Some(Self::Bool(b)),
                SemanticValue::Bv(v) => Some(Self::from_i64(v as i64, ty.min_bit_width()?)),
            })
            .collect()
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

    pub fn binary_op(self, other: Self, op: Opcode) -> Option<Self> {
        Self::apply(op, &[self, other])
    }

    pub fn unary_op(self, op: Opcode) -> Option<Self> {
        Self::apply(op, &[self])
    }

    fn apply(op: Opcode, args: &[Self]) -> Option<Self> {
        let types = args.iter().map(|c| c.ty()).collect::<SmallVec<[Type; 2]>>();
        let crate::opspec::ResultTypes::Inferred(results) = op.infer_result_types(&types).ok()?
        else {
            return None;
        };
        if results.len() != 1 {
            return None;
        }
        Self::evaluate(op, args, &results, &[])?.first().copied()
    }

    pub fn icmp(self, other: Self, kind: IntCC) -> Option<Self> {
        Self::evaluate(
            Opcode::Icmp,
            &[self, other],
            &[Type::BOOL],
            &[kind.predicate()],
        )?
        .first()
        .copied()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_modular_operations_from_specs() {
        assert_eq!(
            Constant::I8(127).binary_op(Constant::I8(1), Opcode::IAdd),
            Some(Constant::I8(-128))
        );
        assert_eq!(
            Constant::I16(-32768).binary_op(Constant::I16(1), Opcode::ISub),
            Some(Constant::I16(32767))
        );
        assert_eq!(
            Constant::I32(i32::MAX).binary_op(Constant::I32(2), Opcode::IMul),
            Some(Constant::I32(-2))
        );
        assert_eq!(
            Constant::I64(i64::MIN).unary_op(Opcode::INeg),
            Some(Constant::I64(i64::MIN))
        );
        assert_eq!(
            Constant::I8(-1).binary_op(Constant::I8(15), Opcode::IAnd),
            Some(Constant::I8(15))
        );
        assert_eq!(
            Constant::I8(-128).binary_op(Constant::I8(15), Opcode::IOr),
            Some(Constant::I8(-113))
        );
        assert_eq!(
            Constant::I8(-1).binary_op(Constant::I8(15), Opcode::IXor),
            Some(Constant::I8(-16))
        );
    }

    #[test]
    fn folding_rejects_mismatched_types_and_arities() {
        assert_eq!(
            Constant::I8(1).binary_op(Constant::I16(2), Opcode::IAdd),
            None
        );
        assert_eq!(
            Constant::I8(1).binary_op(Constant::I8(2), Opcode::INeg),
            None
        );
        assert_eq!(Constant::I8(1).unary_op(Opcode::IAdd), None);
        assert_eq!(Constant::F32(1.0).unary_op(Opcode::INeg), None);
    }
}
