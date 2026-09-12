//! Build-time evaluation of the same expressions emitted into type validators.
//! Only constraints independent of SSA identities, properties and host queries
//! are evaluated here. Undefined queries/arithmetic reject an instantiation.
use super::*;
use crate::types::Types;

impl Expr {
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
            ExprKind::Array(values) | ExprKind::Variant(_, values) => {
                values.iter().all(|v| v.type_only(params))
            }
            ExprKind::Matches(..)
            | ExprKind::Host(..)
            | ExprKind::Parameter(_)
            | ExprKind::Operand(_)
            | ExprKind::ResultValue(_) => false,
        }
    }

    pub(crate) fn accepts_types(
        &self,
        types: &Types,
        params: &[Param],
        codes: &[u8],
        inputs: usize,
        shape: u32,
    ) -> bool {
        assert!(
            self.type_only(params),
            "type evaluation requires a type-only expression"
        );
        let operands = params
            .iter()
            .filter(|p| p.kind == ParamKind::Value)
            .zip(codes)
            .map(|(p, code)| (p.name.as_str(), Value::Type(*code, shape)))
            .collect();
        let results = codes[inputs..]
            .iter()
            .map(|code| Value::Type(*code, shape))
            .collect();
        let mut evaluator = Evaluator {
            types,
            operands,
            results,
            locals: BTreeMap::new(),
        };
        matches!(evaluator.eval(self), Some(Value::Bool(true)))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum Value {
    Int(i128),
    Bool(bool),
    Type(u8, u32),
    Shape(u32),
    Bits(u32, bool),
    Sequence(Vec<Value>),
    Record(BTreeMap<String, Value>),
    Variant(String, Vec<Value>),
    Optional(Option<Box<Value>>),
    Literal(data::Value),
}

impl Value {
    fn literal(value: &data::Value) -> Self {
        match value {
            data::Value::Bool(v) => Self::Bool(*v),
            data::Value::Number(v) => Self::Int(*v),
            data::Value::None => Self::Optional(None),
            data::Value::Some(v) => Self::Optional(Some(Box::new(Self::literal(v)))),
            data::Value::Record(_, fields) => Self::Record(
                fields
                    .iter()
                    .map(|(n, v)| (n.clone(), Self::literal(v)))
                    .collect(),
            ),
            data::Value::Variant(_, n, args) => {
                Self::Variant(n.clone(), args.iter().map(Self::literal).collect())
            }
            _ => Self::Literal(value.clone()),
        }
    }
}

struct Evaluator<'a> {
    types: &'a Types,
    operands: BTreeMap<&'a str, Value>,
    results: Vec<Value>,
    locals: BTreeMap<usize, Value>,
}

impl Evaluator<'_> {
    fn eval(&mut self, expr: &Expr) -> Option<Value> {
        use ExprKind as E;
        use Value as V;
        let value = match &expr.kind {
            E::Constant(v) => V::literal(v),
            E::Integer(v) => V::Int(*v),
            E::Type(name) => {
                let set = &self.types.exact[name];
                let (&code, &shapes) = set.0.first_key_value()?;
                assert!(set.0.len() == 1 && shapes.count_ones() == 1, "exact type");
                V::Type(code, shapes.trailing_zeros())
            }
            E::Results => V::Sequence(self.results.clone()),
            E::ResultType(index) => self.results.get(*index)?.clone(),
            E::Bound(id) => self.locals.get(id)?.clone(),
            E::Query(Query::TypeOf, value) => {
                let E::Operand(name) = &value.kind else {
                    unreachable!("type-only SSA type query")
                };
                self.operands.get(name.as_str())?.clone()
            }
            E::Query(query, value) => {
                let value = self.eval(value)?;
                if let (Query::Len, V::Sequence(values)) = (query, &value) {
                    V::Int(values.len() as i128)
                } else {
                    let V::Type(code, shape) = value else {
                        unreachable!("checked type query")
                    };
                    let set = TypeSet::singleton(code, shape % 16, shape >= 16);
                    if let Some(known) = known_query(self.types, *query, &set) {
                        return self.eval(&known);
                    }
                    match query {
                        Query::Shape if shape != 0 => V::Shape(shape),
                        Query::BitSize => {
                            let scalar = self.types.scalars.iter().find(|s| s.code == code)?;
                            V::Bits(scalar.bits?.checked_shl(shape % 16)?, shape >= 16)
                        }
                        // Missing widths/shape are a failed requirement, just
                        // like the checked runtime queries emitted by Emitter.
                        _ => return None,
                    }
                }
            }
            E::Unary(op, value) => match (*op, self.eval(value)?) {
                ("!", V::Bool(value)) => V::Bool(!value),
                ("-", V::Int(value)) => V::Int(value.checked_neg()?),
                _ => unreachable!("checked unary expression"),
            },
            E::Binary(op, lhs, rhs) => {
                let lhs = self.eval(lhs)?;
                // Preserve runtime short circuiting, including partial queries.
                if (*op == "&&" && lhs == V::Bool(false)) || (*op == "||" && lhs == V::Bool(true)) {
                    return Some(lhs);
                }
                let rhs = self.eval(rhs)?;
                match (*op, lhs, rhs) {
                    ("==", lhs, rhs) => V::Bool(lhs == rhs),
                    ("!=", lhs, rhs) => V::Bool(lhs != rhs),
                    ("&&", V::Bool(a), V::Bool(b)) => V::Bool(a && b),
                    ("||", V::Bool(a), V::Bool(b)) => V::Bool(a || b),
                    (op, V::Int(a), V::Int(b)) => match op {
                        "+" => V::Int(a.checked_add(b)?),
                        "-" => V::Int(a.checked_sub(b)?),
                        "*" => V::Int(a.checked_mul(b)?),
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
            E::Array(values) => {
                V::Sequence(values.iter().map(|v| self.eval(v)).collect::<Option<_>>()?)
            }
            E::Slice(sequence, index, prefix) => {
                let (V::Sequence(values), V::Int(index)) =
                    (self.eval(sequence)?, self.eval(index)?)
                else {
                    unreachable!("checked slice")
                };
                let index = usize::try_from(index).ok()?;
                V::Sequence(
                    if *prefix {
                        values.get(..index)?
                    } else {
                        values.get(index..)?
                    }
                    .to_vec(),
                )
            }
            E::All(sequence, id, body) => {
                let V::Sequence(values) = self.eval(sequence)? else {
                    unreachable!("checked all")
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
                fields
                    .iter()
                    .map(|(n, v)| Some((n.clone(), self.eval(v)?)))
                    .collect::<Option<_>>()?,
            ),
            E::Field(value, name) => {
                let V::Record(fields) = self.eval(value)? else {
                    unreachable!("checked record")
                };
                fields.get(name)?.clone()
            }
            E::Variant(name, args) => V::Variant(
                name.clone(),
                args.iter().map(|v| self.eval(v)).collect::<Option<_>>()?,
            ),
            E::Some(value) => V::Optional(Some(Box::new(self.eval(value)?))),
            E::Try(value) => {
                let V::Optional(value) = self.eval(value)? else {
                    unreachable!("checked optional")
                };
                *value?
            }
            _ => unreachable!("non-type expression cannot reach type evaluation"),
        };
        if let V::Int(value) = &value
            && !expr.ty.fits(*value)
        {
            return None;
        }
        Some(value)
    }
}
