//! Checked predicates retain their structure until emission. Query filters and
//! capture dependencies are projections of this same tree.
use super::*;

pub(in crate::rules) enum Predicate {
    Value(usize),
    Number(u64),
    Type(TypeRef),
    Attribute(AttributeValue),
    PointerBits,
    Integer {
        slot: usize,
        signed: bool,
    },
    ShiftAmount(usize),
    LowMask(Box<Self>),
    TypeOf(Box<Self>),
    Bits(Box<Self>),
    IsConstant(usize),
    Constant {
        value: usize,
        bits: u64,
        equal: bool,
    },
    Binary(&'static str, Box<Self>, Box<Self>),
}

impl Predicate {
    pub(super) fn code(&self, types: &str, captures: &[usize]) -> String {
        let code = |p: &Self| p.code(types, captures);
        let value = |slot| capture_code(captures, slot);
        match self {
            Self::Value(slot) => value(*slot),
            Self::Number(n) => format!("{n}u64"),
            Self::Type(ty) => ty.code(types, captures),
            Self::Attribute(value) => value.code(),
            Self::PointerBits => "u64::from(cx.pointer_bits()?)".into(),
            Self::Integer { slot, signed } => {
                let value = value(*slot);
                if *signed {
                    format!("cx.constant({value})?.as_int()?.signed()")
                } else {
                    format!("cx.constant({value})?.as_int()?.to_bits()")
                }
            }
            Self::ShiftAmount(slot) => format!(
                "cx.constant({value})?.as_int()?.to_bits() % u64::from(cx.ty({value}).element_bits()?)",
                value = value(*slot)
            ),
            Self::LowMask(bits) => format!(
                "u64::MAX.checked_shr(64u32.checked_sub(u32::try_from({}).ok()?)?)?",
                code(bits)
            ),
            Self::TypeOf(value) => format!("cx.ty({})", code(value)),
            Self::Bits(ty) => format!("u64::from(({}).element_bits()?)", code(ty)),
            Self::IsConstant(slot) => format!("cx.constant({}).is_some()", value(*slot)),
            Self::Constant { value, bits, equal } => format!(
                "crate::evaluate::matches_constant(cx.constant({}), {bits}u64, {equal})",
                capture_code(captures, *value)
            ),
            Self::Binary(op @ ("+" | "-" | "*"), a, b) => {
                let method = match *op {
                    "+" => "checked_add",
                    "-" => "checked_sub",
                    _ => "checked_mul",
                };
                format!("({}).{method}({})?", code(a), code(b))
            }
            Self::Binary(op, a, b) => format!("({}) {op} ({})", code(a), code(b)),
        }
    }

    pub(super) fn captures(&self, slots: &mut BTreeSet<usize>) {
        match self {
            Self::Value(slot)
            | Self::IsConstant(slot)
            | Self::Integer { slot, .. }
            | Self::ShiftAmount(slot)
            | Self::Constant { value: slot, .. } => {
                slots.insert(*slot);
            }
            Self::Type(TypeRef {
                source: TypeSource::Value(slot),
                ..
            }) => {
                slots.insert(*slot);
            }
            Self::TypeOf(p) | Self::Bits(p) | Self::LowMask(p) => p.captures(slots),
            Self::Binary(_, a, b) => {
                a.captures(slots);
                b.captures(slots);
            }
            _ => {}
        }
    }

    pub(super) fn filters(&self, out: &mut Vec<Filter>) {
        match self {
            Self::Binary("&&", a, b) => {
                a.filters(out);
                b.filters(out);
            }
            Self::IsConstant(slot) => out.push(Filter::IsConstant(*slot)),
            Self::Constant { value, bits, equal } => {
                out.push(Filter::Constant(*value, *bits, *equal))
            }
            _ => self.required_facts(out),
        }
    }

    /// A query producing a Boolean need not be true just because its result is
    /// needed. Only partial numeric queries require known constants to evaluate.
    fn required_facts(&self, out: &mut Vec<Filter>) {
        match self {
            Self::Integer { slot, .. } | Self::ShiftAmount(slot) => {
                out.push(Filter::IsConstant(*slot))
            }
            Self::LowMask(value) => value.required_facts(out),
            Self::Binary("&&" | "||", _, _) => {}
            Self::Binary(_, a, b) => {
                a.required_facts(out);
                b.required_facts(out);
            }
            _ => {}
        }
    }
}

#[derive(PartialEq)]
pub(super) enum GuardType {
    Value,
    Type,
    Number,
    SignedNumber,
    Bool,
    Attribute(String),
}

impl Checker<'_> {
    pub(super) fn guard(&self, node: &Node) -> Result<(Predicate, GuardType), Error> {
        use GuardType as T;
        use Predicate as P;
        if let Some(n) = literal(node) {
            return Ok((P::Number(n), T::Number));
        }
        match &node.kind {
            Kind::Name(name) => {
                if let Some(&slot) = self.variables.get(name) {
                    return Ok((P::Value(slot), T::Value));
                }
                if let Some(a) = self.attributes.get(name) {
                    return Ok((P::Attribute(a.value.clone()), T::Attribute(a.ty.clone())));
                }
                Ok((P::Type(self.output_type(node)?), T::Type))
            }
            Kind::Call(name, args) if name == "pointer_bits" && args.is_empty() => {
                Ok((P::PointerBits, T::Number))
            }
            Kind::Call(name, args) if args.len() == 1 => {
                let (value, ty) = self.guard(&args[0])?;
                match (name.as_str(), ty, value) {
                    ("unsigned", T::Value, P::Value(slot)) => Ok((
                        P::Integer {
                            slot,
                            signed: false,
                        },
                        T::Number,
                    )),
                    ("signed", T::Value, P::Value(slot)) => {
                        Ok((P::Integer { slot, signed: true }, T::SignedNumber))
                    }
                    ("shift_amount", T::Value, P::Value(slot)) => {
                        Ok((P::ShiftAmount(slot), T::Number))
                    }
                    ("low_mask", T::Number, bits) => Ok((P::LowMask(Box::new(bits)), T::Number)),
                    ("type_of", T::Value, value) => Ok((P::TypeOf(Box::new(value)), T::Type)),
                    ("bits", T::Type, ty) => Ok((P::Bits(Box::new(ty)), T::Number)),
                    ("is_const", T::Value, P::Value(slot)) => Ok((P::IsConstant(slot), T::Bool)),
                    _ => Err(self.fail(node.offset, "invalid guard query")),
                }
            }
            Kind::Binary(op, a, b) => {
                let (left, at) = self.guard(a)?;
                let (right, bt) = if let T::Attribute(ty) = &at {
                    (
                        P::Attribute(self.attribute(b, ty)?),
                        T::Attribute(ty.clone()),
                    )
                } else {
                    self.guard(b)?
                };
                if matches!(*op, "==" | "!=")
                    && let (P::Value(value), P::Number(bits)) = (&left, &right)
                {
                    return Ok((
                        P::Constant {
                            value: *value,
                            bits: *bits,
                            equal: *op == "==",
                        },
                        T::Bool,
                    ));
                }
                let valid = match *op {
                    "&&" | "||" => at == T::Bool && bt == T::Bool,
                    "==" | "!=" => at == bt && at != T::Value,
                    "<" | ">" | "<=" | ">=" | "+" | "-" | "*" | "&" | "|" => {
                        at == bt && matches!(at, T::Number | T::SignedNumber)
                    }
                    _ => false,
                };
                if !valid {
                    return Err(self.fail(node.offset, "invalid guard operand types"));
                }
                let result = if matches!(*op, "+" | "-" | "*" | "&" | "|") {
                    at
                } else {
                    T::Bool
                };
                Ok((P::Binary(op, Box::new(left), Box::new(right)), result))
            }
            _ => Err(self.fail(node.offset, "unsupported guard")),
        }
    }
}
