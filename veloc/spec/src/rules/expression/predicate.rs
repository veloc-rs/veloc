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

/// The predicate tree and arithmetic are shared; each host supplies only the
/// value, constant and type queries that its interface can answer. Returning
/// None during generation keeps an unsupported guard on the e-graph path.
enum Context<'a> {
    Graph(&'a [usize]),
    Local {
        rule: &'a CheckedRule,
        order: &'a [usize],
    },
}

impl Context<'_> {
    fn value(&self, slot: usize) -> Option<String> {
        match self {
            Self::Graph(captures) => Some(capture_code(captures, slot)),
            Self::Local { rule, order } => {
                let PatternKind::Operation { args, .. } = &rule.pattern[0].kind else {
                    unreachable!()
                };
                let column = order[args.iter().position(|&arg| arg == slot)?];
                Some(format!("args[{column}]"))
            }
        }
    }

    fn constant(&self, slot: usize) -> Option<String> {
        let value = self.value(slot)?;
        Some(match self {
            Self::Graph(_) => format!("cx.constant({value})"),
            Self::Local { .. } => format!("constant({value})"),
        })
    }

    fn ty(&self, slot: usize, types: &str) -> Option<String> {
        match self {
            Self::Graph(_) => Some(format!("cx.ty({})", self.value(slot)?)),
            Self::Local { rule, .. } => {
                let source = &rule.pattern[slot].ty.source;
                if *source == rule.pattern[0].ty.source {
                    Some("ty".into())
                } else if let TypeSource::Exact(name) = source {
                    Some(name.replacen("Type", types, 1))
                } else {
                    // Operand identities carry no type query. Do not rely on a
                    // constant being available just to discover an input type.
                    None
                }
            }
        }
    }
}

impl Predicate {
    pub(super) fn code(&self, types: &str, captures: &[usize]) -> String {
        self.emit(types, &Context::Graph(captures))
            .expect("e-graph guard queries")
    }

    pub(super) fn local_code(
        &self,
        rule: &CheckedRule,
        types: &str,
        order: &[usize],
    ) -> Option<String> {
        self.emit(types, &Context::Local { rule, order })
    }

    fn emit(&self, types: &str, cx: &Context<'_>) -> Option<String> {
        let code = |p: &Self| p.emit(types, cx);
        Some(match self {
            Self::Value(slot) => cx.value(*slot)?,
            Self::Number(n) => format!("{n}u64"),
            Self::Type(ty) => match &ty.source {
                TypeSource::Value(slot) => cx.ty(*slot, types)?,
                TypeSource::Exact(name) => name.replacen("Type", types, 1),
            },
            Self::Attribute(value) => value.code(),
            Self::PointerBits => match cx {
                Context::Graph(_) => "u64::from(cx.pointer_bits()?)".into(),
                Context::Local { .. } => return None,
            },
            Self::Integer { slot, signed } => {
                let constant = cx.constant(*slot)?;
                if *signed {
                    format!("{constant}?.as_int()?.signed()")
                } else {
                    format!("{constant}?.as_int()?.to_bits()")
                }
            }
            Self::ShiftAmount(slot) => {
                let constant = cx.constant(*slot)?;
                // An integer constant carries its own width in either host.
                format!(
                    "{{ let c = {constant}?; c.as_int()?.to_bits() % u64::from(c.ty().element_bits()?) }}"
                )
            }
            Self::LowMask(bits) => format!(
                "u64::MAX.checked_shr(64u32.checked_sub(u32::try_from({}).ok()?)?)?",
                code(bits)?
            ),
            Self::TypeOf(value) => {
                let Self::Value(slot) = value.as_ref() else {
                    unreachable!("checked value type query")
                };
                cx.ty(*slot, types)?
            }
            Self::Bits(ty) => format!("u64::from(({}).element_bits()?)", code(ty)?),
            Self::IsConstant(slot) => format!("{}.is_some()", cx.constant(*slot)?),
            Self::Constant { value, bits, equal } => format!(
                "crate::rewrite::matches_constant({}, {bits}u64, {equal})",
                cx.constant(*value)?
            ),
            Self::Binary(op @ ("+" | "-" | "*"), a, b) => {
                let method = match *op {
                    "+" => "checked_add",
                    "-" => "checked_sub",
                    _ => "checked_mul",
                };
                format!("({}).{method}({})?", code(a)?, code(b)?)
            }
            Self::Binary(op, a, b) => format!("({}) {op} ({})", code(a)?, code(b)?),
        })
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
    pub(super) fn guard(&self, node: &Node) -> CheckResult<(Predicate, GuardType)> {
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
            Kind::Call(name, _)
                if !matches!(
                    name.as_str(),
                    "pointer_bits"
                        | "unsigned"
                        | "signed"
                        | "shift_amount"
                        | "low_mask"
                        | "type_of"
                        | "bits"
                        | "is_const"
                ) =>
            {
                Err(CheckError::Instruction)
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
