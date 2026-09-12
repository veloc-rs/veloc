//! Fold only definition-owned literals and expressions.
//! Foreign methods remain opaque and are evaluated by the generated Rust.
use super::*;
use crate::types::Types;

impl Expr {
    /// Fold defs-owned expressions without consulting foreign query tables.
    pub(crate) fn constant_node(&self, types: &Types, offset: usize) -> Option<Node> {
        Evaluator {
            types,
            locals: BTreeMap::new(),
        }
        .eval(self)
        .ok()?
        .node(types, offset)
    }

    pub(crate) fn literal_bool(&self, types: &Types) -> Option<bool> {
        let mut evaluator = Evaluator {
            types,
            locals: BTreeMap::new(),
        };
        match evaluator.eval(self).ok()? {
            Value::Bool(value) => Some(value),
            _ => None,
        }
    }

    pub(crate) fn type_only(&self, params: &[Param]) -> bool {
        match &self.kind {
            ExprKind::Query(Query::TypeOf, value) => matches!(&value.kind,
                ExprKind::Operand(name) if params.iter().any(|p| p.name == *name && p.kind == ParamKind::Value)),
            ExprKind::Constant(_)
            | ExprKind::Integer(_)
            | ExprKind::Type(_)
            | ExprKind::ResultType(_)
            | ExprKind::Results
            | ExprKind::Bound(_) => true,
            ExprKind::Unary(_, v)
            | ExprKind::Query(_, v)
            | ExprKind::Convert(v)
            | ExprKind::Some(v)
            | ExprKind::Try(v)
            | ExprKind::Field(v, _) => v.type_only(params),
            ExprKind::Binary(_, a, b) | ExprKind::Slice(a, b, _) | ExprKind::All(a, _, b) => {
                a.type_only(params) && b.type_only(params)
            }
            ExprKind::Record(fields) => fields.values().all(|v| v.type_only(params)),
            ExprKind::Rust(_, values) | ExprKind::Array(values) | ExprKind::Variant(_, values) => {
                values.iter().all(|v| v.type_only(params))
            }
            ExprKind::Matches(..)
            | ExprKind::Host(..)
            | ExprKind::Parameter(_)
            | ExprKind::Operand(_) => false,
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(super) enum Value {
    Int(i128),
    Bool(bool),
    Type(crate::types::Primitive, u32),
    Sequence(Vec<Value>),
    Record(String, BTreeMap<String, Value>),
    Variant(String, String, Vec<Value>),
    Optional(Option<Box<Value>>),
    Literal(data::Value),
}

impl Value {
    /// This only serializes values; it does not reimplement expression evaluation.
    fn node(self, types: &Types, offset: usize) -> Option<Node> {
        let kind = match self {
            Self::Int(n) => u32::try_from(n)
                .map(Kind::Number)
                .unwrap_or(Kind::Integer(n)),
            Self::Bool(v) => Kind::Name(v.to_string()),
            Self::Type(code, shape) => {
                let name = types
                    .exact
                    .iter()
                    .find(|(_, set)| {
                        set.0.len() == 1 && set.0.get(&code) == Some(&(1u32 << shape))
                    })?
                    .0;
                Kind::Name(name.clone())
            }
            Self::Record(name, fields) => Kind::Object(
                name,
                fields
                    .into_iter()
                    .map(|(name, value)| Some((name, value.node(types, offset)?)))
                    .collect::<Option<_>>()?,
            ),
            Self::Variant(_, name, args) => Kind::Call(
                name,
                args.into_iter()
                    .map(|v| v.node(types, offset))
                    .collect::<Option<_>>()?,
            ),
            Self::Sequence(values) => Kind::List(
                values
                    .into_iter()
                    .map(|v| v.node(types, offset))
                    .collect::<Option<_>>()?,
            ),
            Self::Optional(Some(v)) => Kind::Call("some".into(), vec![v.node(types, offset)?]),
            Self::Optional(None) => Kind::Name("none".into()),
            Self::Literal(data::Value::Empty(_)) => Kind::Name("empty".into()),
            Self::Literal(_) => unreachable!("structured literals were converted to values"),
        };
        Some(Node { offset, kind })
    }

    fn from_literal(value: &data::Value) -> Self {
        match value {
            data::Value::Bool(v) => Self::Bool(*v),
            data::Value::Number(v) => Self::Int(*v),
            data::Value::None => Self::Optional(None),
            data::Value::Some(v) => Self::Optional(Some(Box::new(Self::from_literal(v)))),
            data::Value::Record(name, fields) => Self::Record(
                name.clone(),
                fields
                    .iter()
                    .map(|(n, v)| (n.clone(), Self::from_literal(v)))
                    .collect(),
            ),
            data::Value::Variant(ty, n, args) => Self::Variant(
                ty.clone(),
                n.clone(),
                args.iter().map(Self::from_literal).collect(),
            ),
            _ => Self::Literal(value.clone()),
        }
    }
}

/// Unknown inputs cannot be folded; invalid operations reject a concrete instance.
#[derive(Debug)]
enum EvalError {
    Unknown,
    Invalid,
}
use EvalError::{Invalid, Unknown};

struct Evaluator<'a> {
    types: &'a Types,
    locals: BTreeMap<usize, Value>,
}

impl Evaluator<'_> {
    fn eval(&mut self, expr: &Expr) -> Result<Value, EvalError> {
        use ExprKind as E;
        use Value as V;
        // Singleton type domains are concrete even before an instruction exists.
        if expr.ty == Ty::named("Type")
            && let Some(set) = &expr.types
            && set.0.len() == 1
        {
            let (&code, &shapes) = set.0.first_key_value().unwrap();
            if shapes.count_ones() == 1 {
                return Ok(V::Type(code, shapes.trailing_zeros()));
            }
        }
        let value = match &expr.kind {
            E::Constant(v) => V::from_literal(v),
            E::Integer(v) => V::Int(*v),
            E::Type(name) => {
                let set = &self.types.exact[name];
                let (&code, &shapes) = set.0.first_key_value().expect("nonempty exact type");
                assert!(set.0.len() == 1 && shapes.count_ones() == 1, "exact type");
                V::Type(code, shapes.trailing_zeros())
            }
            E::Results | E::ResultType(_) | E::Query(Query::TypeOf, _) => return Err(Unknown),
            E::Bound(id) => self.locals.get(id).ok_or(Unknown)?.clone(),
            E::Query(Query::Len, value) => {
                let V::Sequence(values) = self.eval(value)? else {
                    unreachable!("checked sequence length")
                };
                V::Int(values.len() as i128)
            }
            E::Rust(..) => return Err(Unknown),
            E::Unary(op, value) => match (*op, self.eval(value)?) {
                ("!", V::Bool(value)) => V::Bool(!value),
                ("-", V::Int(value)) => V::Int(value.checked_neg().ok_or(Invalid)?),
                _ => unreachable!("checked unary expression"),
            },
            E::Binary(op, lhs, rhs) => {
                let lhs = self.eval(lhs)?;
                // Preserve runtime short circuiting, including partial queries.
                if (*op == "&&" && lhs == V::Bool(false)) || (*op == "||" && lhs == V::Bool(true)) {
                    return Ok(lhs);
                }
                let rhs = self.eval(rhs)?;
                match (*op, lhs, rhs) {
                    ("==", lhs, rhs) => V::Bool(lhs == rhs),
                    ("!=", lhs, rhs) => V::Bool(lhs != rhs),
                    ("&&", V::Bool(a), V::Bool(b)) => V::Bool(a && b),
                    ("||", V::Bool(a), V::Bool(b)) => V::Bool(a || b),
                    (op, V::Int(a), V::Int(b)) => match op {
                        "+" => V::Int(a.checked_add(b).ok_or(Invalid)?),
                        "-" => V::Int(a.checked_sub(b).ok_or(Invalid)?),
                        "*" => V::Int(a.checked_mul(b).ok_or(Invalid)?),
                        "&" => V::Int(a & b),
                        "|" => V::Int(a | b),
                        "<" => V::Bool(a < b),
                        "<=" => V::Bool(a <= b),
                        ">" => V::Bool(a > b),
                        ">=" => V::Bool(a >= b),
                        _ => unreachable!("checked integer expression"),
                    },
                    _ => unreachable!("checked binary expression"),
                }
            }
            E::Convert(value) => self.eval(value)?,
            E::Array(values) => V::Sequence(
                values
                    .iter()
                    .map(|v| self.eval(v))
                    .collect::<Result<_, _>>()?,
            ),
            E::Slice(sequence, index, prefix) => {
                let (V::Sequence(values), V::Int(index)) =
                    (self.eval(sequence)?, self.eval(index)?)
                else {
                    unreachable!("checked slice")
                };
                let index = usize::try_from(index).map_err(|_| Invalid)?;
                V::Sequence(
                    if *prefix {
                        values.get(..index).ok_or(Invalid)?
                    } else {
                        values.get(index..).ok_or(Invalid)?
                    }
                    .to_vec(),
                )
            }
            E::All(sequence, id, body) => {
                let values = match self.eval(sequence)? {
                    V::Sequence(values) => values,
                    V::Optional(value) => value.into_iter().map(|v| *v).collect(),
                    _ => unreachable!("checked all"),
                };
                let old = self.locals.get(id).cloned();
                let mut answer = true;
                for value in values {
                    self.locals.insert(*id, value);
                    match self.eval(body)? {
                        V::Bool(true) => {}
                        V::Bool(false) => {
                            answer = false;
                            break;
                        }
                        _ => unreachable!("checked predicate"),
                    }
                }
                if let Some(old) = old {
                    self.locals.insert(*id, old);
                } else {
                    self.locals.remove(id);
                }
                V::Bool(answer)
            }
            E::Record(fields) => V::Record(
                expr.ty.name().to_owned(),
                fields
                    .iter()
                    .map(|(n, v)| Ok((n.clone(), self.eval(v)?)))
                    .collect::<Result<_, _>>()?,
            ),
            E::Field(value, name) => {
                let V::Record(_, fields) = self.eval(value)? else {
                    unreachable!("checked record")
                };
                fields.get(name).expect("checked field").clone()
            }
            E::Variant(name, args) => V::Variant(
                expr.ty.name().to_owned(),
                name.clone(),
                args.iter()
                    .map(|v| self.eval(v))
                    .collect::<Result<_, _>>()?,
            ),
            E::Some(value) => V::Optional(Some(Box::new(self.eval(value)?))),
            E::Try(value) => {
                let V::Optional(value) = self.eval(value)? else {
                    unreachable!("checked optional")
                };
                *value.ok_or(Invalid)?
            }
            _ => return Err(Unknown),
        };
        if let V::Int(value) = &value
            && !expr.ty.fits(*value)
        {
            return Err(Invalid);
        }
        Ok(value)
    }
}
