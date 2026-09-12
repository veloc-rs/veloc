//! Build-time evaluation of the same expressions emitted into type validators.
//! Only constraints independent of SSA identities, properties and host queries
//! are evaluated here. Undefined queries/arithmetic reject an instantiation.
use super::*;
use crate::types::Types;

impl Expr {
    /// Evaluate through the same typed interpreter as offline constraints.
    pub(crate) fn constant_node(&self, types: &Types, offset: usize) -> Option<Node> {
        Evaluator {
            types,
            operands: BTreeMap::new(),
            results: None,
            locals: BTreeMap::new(),
        }
        .eval(self)
        .ok()?
        .node(types, offset)
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

    pub(crate) fn offline_supported(&self) -> bool {
        match &self.kind {
            ExprKind::Rust(binding, args) => {
                binding.evaluation.is_some() && args.iter().all(Self::offline_supported)
            }
            ExprKind::Host(..) => false,
            ExprKind::Unary(_, v)
            | ExprKind::Query(_, v)
            | ExprKind::Convert(v)
            | ExprKind::Some(v)
            | ExprKind::Try(v)
            | ExprKind::Field(v, _) => v.offline_supported(),
            ExprKind::Binary(_, a, b)
            | ExprKind::Slice(a, b, _)
            | ExprKind::All(a, _, b)
            | ExprKind::Matches(a, b) => a.offline_supported() && b.offline_supported(),
            ExprKind::Record(fields) => fields.values().all(Self::offline_supported),
            ExprKind::Array(values) | ExprKind::Variant(_, values) => {
                values.iter().all(Self::offline_supported)
            }
            _ => true,
        }
    }

    pub(crate) fn accepts_types(
        &self,
        types: &Types,
        params: &[Param],
        values: &[(u8, u32)],
        inputs: usize,
    ) -> bool {
        // Called only for constraints classified as type-only during checking.
        let operands = params
            .iter()
            .filter(|p| p.kind == ParamKind::Value)
            .zip(values)
            .map(|(p, &(code, shape))| (p.name.as_str(), Value::Type(code, shape)))
            .collect();
        let results = values[inputs..]
            .iter()
            .map(|&(code, shape)| Value::Type(code, shape))
            .collect();
        let mut evaluator = Evaluator {
            types,
            operands,
            results: Some(results),
            locals: BTreeMap::new(),
        };
        matches!(evaluator.eval(self), Ok(Value::Bool(true)))
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(super) enum Value {
    Int(i128),
    Bool(bool),
    Type(u8, u32),
    Shape(veloc_types::Shape),
    Bits(veloc_types::TypeBits),
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
            Self::Literal(data::Value::Flags(_, members)) => Kind::List(
                members
                    .into_iter()
                    .map(|name| Node {
                        offset,
                        kind: Kind::Name(name),
                    })
                    .collect(),
            ),
            Self::Literal(data::Value::Empty(_)) => Kind::Name("empty".into()),
            Self::Literal(_) => unreachable!("structured literals were converted to values"),
            Self::Shape(_) | Self::Bits(_) => return None,
        };
        Some(Node { offset, kind })
    }

    pub(super) fn expression(self, ty: &Ty) -> Option<Expr> {
        let kind = match self {
            Self::Int(v) => ExprKind::Integer(v),
            Self::Bool(v) => return Some(Expr::boolean(v)),
            Self::Optional(Some(v)) => {
                let Ty::Optional(inner) = ty else {
                    unreachable!("typed optional query")
                };
                ExprKind::Some(Box::new(v.expression(inner)?))
            }
            // Keep absent/structured results as typed calls in generated Rust;
            // bare None would lose its type through fallible-expression expansion.
            _ => return None,
        };
        Some(Expr::new(ty.clone(), kind))
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
    operands: BTreeMap<&'a str, Value>,
    results: Option<Vec<Value>>,
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
            E::Results => V::Sequence(self.results.clone().ok_or(Unknown)?),
            E::ResultType(index) => self
                .results
                .as_ref()
                .ok_or(Unknown)?
                .get(*index)
                .ok_or(Invalid)?
                .clone(),
            E::Bound(id) => self.locals.get(id).ok_or(Unknown)?.clone(),
            E::Query(Query::TypeOf, value) => {
                let E::Operand(name) = &value.kind else {
                    return Err(Unknown);
                };
                self.operands.get(name.as_str()).ok_or(Unknown)?.clone()
            }
            E::Query(Query::Len, value) => {
                let V::Sequence(values) = self.eval(value)? else {
                    unreachable!("checked sequence length")
                };
                V::Int(values.len() as i128)
            }
            E::Rust(binding, args) => {
                let [arg] = args.as_slice() else {
                    return Err(Unknown);
                };
                let V::Type(code, shape) = self.eval(arg)? else {
                    return Err(Unknown);
                };
                binding
                    .evaluation
                    .as_ref()
                    .ok_or(Unknown)?
                    .evaluate(self.types, code, shape)
                    .ok_or(Invalid)?
            }
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

/// Convert a checked member of the defs type universe into shared concrete facts.
/// The scalar code and shape mask are representation adapters, not query semantics.
fn type_query(types: &Types, query: TypeQuery, code: u8, shape: u32) -> Option<Value> {
    use veloc_types::Type;
    let scalar = types.scalars.iter().find(|s| s.code == code)?;
    let element = scalar.ty;
    let ty = if shape == 0 {
        Type::scalar(element)
    } else {
        Type::vector(element, 1u16.checked_shl(shape % 16)?, shape >= 16)?
    };
    let optional = |value: Option<Value>| Value::Optional(value.map(Box::new));
    Some(match query {
        TypeQuery::ElementBits => optional(ty.element_bits().map(|v| Value::Int(v.into()))),
        TypeQuery::Lanes => optional(ty.lanes().map(|v| Value::Int(v.into()))),
        TypeQuery::MinBytes => optional(ty.min_bytes().map(|v| Value::Int(v.into()))),
        TypeQuery::BitSize => optional(ty.bit_size().map(Value::Bits)),
        TypeQuery::Shape => optional(ty.shape().map(Value::Shape)),
        TypeQuery::IsFixed => Value::Bool(ty.is_fixed()),
        TypeQuery::IsCallable => Value::Bool(ty.is_callable()),
        TypeQuery::IsOwned => Value::Bool(ty.is_owned()),
        TypeQuery::IsLocal => Value::Bool(ty.is_local()),
        TypeQuery::IsShared => Value::Bool(ty.is_shared()),
        TypeQuery::IsCompact => Value::Bool(true),
    })
}

impl RustEval {
    pub(super) fn evaluate(&self, types: &Types, code: u8, shape: u32) -> Option<Value> {
        match self {
            Self::Query(query) => type_query(types, *query, code, shape),
            // Membership is computed from the same checked set that generates
            // the runtime predicate, including user-defined predicates.
            Self::Predicate(set) => Some(Value::Bool(
                set.0
                    .get(&code)
                    .is_some_and(|mask| mask & (1 << shape) != 0),
            )),
        }
    }
}
