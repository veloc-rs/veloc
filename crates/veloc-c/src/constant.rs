//! Typed evaluation of C integer constant expressions under the LP64 model.
use crate::{
    ast::*,
    error::Result,
    types::{CType, Types, fail, qualifiers},
};

impl IntegerLiteral {
    pub(crate) fn signed_value(self) -> i64 {
        ((self.value << (64 - self.bits)) as i64) >> (64 - self.bits)
    }
    pub(crate) fn convert(self, bits: u16, signed: bool) -> Self {
        let value = if self.signed {
            self.signed_value() as u64
        } else {
            self.value
        };
        Self {
            value: value & (u64::MAX >> (64 - bits)),
            bits,
            signed,
        }
    }
    fn promote(self) -> Self {
        if self.bits < 32 {
            self.convert(32, true)
        } else {
            self
        }
    }
    fn ty(self) -> CType {
        CType::Int(self.bits, self.signed)
    }
}

impl Types {
    pub fn constant(&mut self, expr: &Expression) -> Result<i64> {
        let value = self.integer_constant(expr)?;
        Ok(if value.signed {
            value.signed_value()
        } else {
            value.value as i64
        })
    }

    pub fn integer_constant(&mut self, expr: &Expression) -> Result<IntegerLiteral> {
        use Expression::*;
        let boolean = |value| IntegerLiteral::int(i32::from(value));
        Ok(match expr {
            Integer(value) => *value,
            Char(c) => IntegerLiteral::int(*c as i32),
            Identifier(name) => match self.constants.get(name) {
                Some(&value) => IntegerLiteral::int(value as i32),
                None => return fail(format!("not an integer constant: {name}")),
            },
            Parenthesized(e) => self.integer_constant(e)?,
            Cast(ty, e) => {
                let value = self.integer_constant(e)?;
                match self.type_name(ty)?.plain() {
                    CType::Int(bits, signed) => value.convert(*bits, *signed),
                    CType::Bool => boolean(value.value != 0).convert(8, false),
                    CType::Pointer(_) => value.convert(64, false),
                    _ => return fail("expected an integer constant cast"),
                }
            }
            UnaryPlus(e) | UnaryMinus(e) | BitwiseNot(e) => {
                let mut value = self.integer_constant(e)?.promote();
                value.value = match expr {
                    UnaryMinus(_) => value.value.wrapping_neg(),
                    BitwiseNot(_) => !value.value,
                    _ => value.value,
                };
                value.convert(value.bits, value.signed)
            }
            LogicalNot(e) => boolean(self.integer_constant(e)?.value == 0),
            LogicalAnd(a, b) => boolean(
                self.integer_constant(a)?.value != 0 && self.integer_constant(b)?.value != 0,
            ),
            LogicalOr(a, b) => boolean(
                self.integer_constant(a)?.value != 0 || self.integer_constant(b)?.value != 0,
            ),
            Conditional(c, a, b) => {
                let condition = self.integer_constant(c)?.value != 0;
                let a_type = self.constant_type(a)?;
                let b_type = self.constant_type(b)?;
                let CType::Int(bits, signed) = CType::common(&a_type, &b_type) else {
                    return fail("expected integer conditional expression");
                };
                self.integer_constant(if condition { a } else { b })?
                    .convert(bits, signed)
            }
            ShiftLeft(a, b) | ShiftRight(a, b) => {
                let mut a = self.integer_constant(a)?.promote();
                let shift = self.integer_constant(b)?.value;
                if shift >= u64::from(a.bits) {
                    return fail("constant shift count is outside the operand width");
                }
                a.value = match expr {
                    ShiftLeft(..) => a.value << shift,
                    _ if a.signed => (a.signed_value() >> shift) as u64,
                    _ => a.value >> shift,
                };
                a.convert(a.bits, a.signed)
            }
            Add(a, b)
            | Subtract(a, b)
            | Multiply(a, b)
            | Divide(a, b)
            | Modulo(a, b)
            | BitwiseAnd(a, b)
            | BitwiseOr(a, b)
            | BitwiseXor(a, b)
            | Equal(a, b)
            | NotEqual(a, b)
            | LessThan(a, b)
            | LessThanOrEqual(a, b)
            | GreaterThan(a, b)
            | GreaterThanOrEqual(a, b) => {
                let a = self.integer_constant(a)?;
                let b = self.integer_constant(b)?;
                let CType::Int(bits, signed) = CType::common(&a.ty(), &b.ty()) else {
                    unreachable!()
                };
                let a = a.convert(bits, signed);
                let b = b.convert(bits, signed);
                let order = if signed {
                    a.signed_value().cmp(&b.signed_value())
                } else {
                    a.value.cmp(&b.value)
                };
                let value = match expr {
                    Add(..) => a.value.wrapping_add(b.value),
                    Subtract(..) => a.value.wrapping_sub(b.value),
                    Multiply(..) => a.value.wrapping_mul(b.value),
                    BitwiseAnd(..) => a.value & b.value,
                    BitwiseOr(..) => a.value | b.value,
                    BitwiseXor(..) => a.value ^ b.value,
                    Equal(..) => return Ok(boolean(order.is_eq())),
                    NotEqual(..) => return Ok(boolean(!order.is_eq())),
                    LessThan(..) => return Ok(boolean(order.is_lt())),
                    LessThanOrEqual(..) => return Ok(boolean(!order.is_gt())),
                    GreaterThan(..) => return Ok(boolean(order.is_gt())),
                    GreaterThanOrEqual(..) => return Ok(boolean(!order.is_lt())),
                    Divide(..) | Modulo(..) => {
                        if b.value == 0 {
                            return fail("division by zero in constant expression");
                        }
                        if signed {
                            (if matches!(expr, Divide(..)) {
                                a.signed_value().wrapping_div(b.signed_value())
                            } else {
                                a.signed_value().wrapping_rem(b.signed_value())
                            }) as u64
                        } else if matches!(expr, Divide(..)) {
                            a.value / b.value
                        } else {
                            a.value % b.value
                        }
                    }
                    _ => unreachable!(),
                };
                IntegerLiteral {
                    value,
                    bits,
                    signed,
                }
                .convert(bits, signed)
            }
            SizeofType(specs, decl) => {
                let base = self.specifiers(&qualifiers(specs))?;
                let ty = if let Some(d) = decl {
                    self.abstract_type(base, d)?
                } else {
                    base
                };
                IntegerLiteral {
                    value: self.layout(&ty)?.0 as u64,
                    bits: 64,
                    signed: false,
                }
            }
            _ => return fail(format!("expected an integer constant expression: {expr:?}")),
        })
    }

    fn constant_type(&mut self, expr: &Expression) -> Result<CType> {
        use Expression::*;
        Ok(match expr {
            Integer(value) => value.ty(),
            Char(_)
            | Identifier(_)
            | LogicalNot(_)
            | LogicalAnd(..)
            | LogicalOr(..)
            | Equal(..)
            | NotEqual(..)
            | LessThan(..)
            | LessThanOrEqual(..)
            | GreaterThan(..)
            | GreaterThanOrEqual(..) => CType::INT,
            Parenthesized(e) => self.constant_type(e)?,
            Cast(ty, _) => self.type_name(ty)?,
            UnaryPlus(e) | UnaryMinus(e) | BitwiseNot(e) | ShiftLeft(e, _) | ShiftRight(e, _) => {
                self.constant_type(e)?.promoted()
            }
            Conditional(_, a, b)
            | Add(a, b)
            | Subtract(a, b)
            | Multiply(a, b)
            | Divide(a, b)
            | Modulo(a, b)
            | BitwiseAnd(a, b)
            | BitwiseOr(a, b)
            | BitwiseXor(a, b) => CType::common(&self.constant_type(a)?, &self.constant_type(b)?),
            SizeofType(..) | SizeofExpression(_) => CType::SIZE,
            _ => return fail("expected integer constant expression type"),
        })
    }
}
