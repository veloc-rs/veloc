//! Typed, pure structural predicates. No opcode-specific rule names live here.
use std::collections::BTreeMap;
use std::fmt::Write;

use crate::Error;
use crate::comparisons::Comparison;
use crate::model::{Definitions, Op, ParamKind, Pattern};
use crate::records::PropertyType;
use crate::storage::Storage;
use crate::syntax::{Kind, Node};
use crate::type_set::TypeSet;
use crate::types::{ScalarKind, Types};

#[derive(Debug, Clone, PartialEq, Eq)]
enum Sort {
    Bool,
    Int,
    Type,
    Value,
    Enum(String),
    Record(String),
    Sequence(Box<Sort>),
}

#[derive(Debug, Clone)]
struct Term {
    sort: Sort,
    kind: TermKind,
    types: Option<TypeSet>,
}

#[derive(Debug, Clone)]
enum TermKind {
    Bool(bool),
    Int(i128),
    Param(usize),
    Bound(usize),
    Member(Box<Term>, String),
    Enum(String, String),
    ResultType(usize),
    Unary(&'static str, Box<Term>),
    Binary(&'static str, Box<Term>, Box<Term>),
    Query(Query, Box<Term>),
    All(Box<Term>, usize, Box<Term>),
}

#[derive(Debug, Clone, Copy)]
enum Query {
    TypeOf,
    Len,
    Lanes,
    MinBytes,
    IsFixed,
    IsPtr,
    IsVector,
    IsScalar,
}

pub(crate) struct Constraint {
    condition: Term,
    pub text: String,
}

impl Term {
    fn new(sort: Sort, kind: TermKind) -> Self {
        Self {
            sort,
            kind,
            types: None,
        }
    }
    fn boolean(value: bool) -> Self {
        Self::new(Sort::Bool, TermKind::Bool(value))
    }
    fn integer(value: i128) -> Self {
        Self::new(Sort::Int, TermKind::Int(value))
    }
}

pub(crate) fn check(
    source: &str,
    nodes: Vec<Node>,
    op: &Op,
    storage: &Storage,
    types: &Types,
    comparisons: &[Comparison],
) -> Result<Vec<Constraint>, Error> {
    let mut checker = Checker {
        source,
        op,
        storage,
        types,
        comparisons,
        locals: BTreeMap::new(),
        next_local: 0,
    };
    let mut constraints = Vec::new();
    for node in nodes {
        let (node, text) = if let Kind::Call(name, args) = &node.kind
            && name == "require"
        {
            let [
                condition,
                Node {
                    kind: Kind::Text(message),
                    ..
                },
            ] = args.as_slice()
            else {
                return Err(Error::at(
                    source,
                    node.offset,
                    "require expects a predicate and a diagnostic string",
                ));
            };
            (condition, message.clone())
        } else {
            (&node, describe(&node))
        };
        let condition = checker.term(node)?;
        checker.expect(node.offset, &condition, &Sort::Bool)?;
        if matches!(condition.kind, TermKind::Bool(false)) {
            return Err(Error::at(
                source,
                node.offset,
                format!("constraint is always false: {text}"),
            ));
        }
        constraints.push(Constraint { condition, text });
    }
    Ok(constraints)
}

struct Checker<'a> {
    source: &'a str,
    op: &'a Op,
    storage: &'a Storage,
    types: &'a Types,
    comparisons: &'a [Comparison],
    locals: BTreeMap<String, (usize, Sort)>,
    next_local: usize,
}

impl Checker<'_> {
    fn error(&self, offset: usize, message: impl Into<String>) -> Error {
        Error::at(self.source, offset, message)
    }
    fn expect(&self, offset: usize, term: &Term, sort: &Sort) -> Result<(), Error> {
        if &term.sort != sort {
            return Err(self.error(
                offset,
                format!("constraint expects {sort:?}, got {:?}", term.sort),
            ));
        }
        Ok(())
    }

    fn property(&self, offset: usize, name: &str) -> Result<Sort, Error> {
        Ok(match name {
            "bool" => Sort::Bool,
            "u8" | "u32" | "u64" | "i32" => Sort::Int,
            "Value" => Sort::Value,
            "Bytes" => Sort::Sequence(Box::new(Sort::Int)),
            name if self.storage.records.iter().any(|r| r.name == name) => {
                Sort::Record(name.into())
            }
            name if self.comparisons.iter().any(|c| c.name == name) => Sort::Enum(name.into()),
            _ => {
                return Err(self.error(
                    offset,
                    format!("unsupported constraint property type {name}"),
                ));
            }
        })
    }

    fn name(&self, offset: usize, name: &str) -> Result<Term, Error> {
        if name == "true" {
            return Ok(Term::boolean(true));
        }
        if name == "false" {
            return Ok(Term::boolean(false));
        }
        let mut path = name.split('.');
        let root = path.next().unwrap();
        let mut value = if let Some((id, sort)) = self.locals.get(root) {
            Term::new(sort.clone(), TermKind::Bound(*id))
        } else if let Some((index, param)) = self
            .op
            .params
            .iter()
            .enumerate()
            .find(|(_, p)| p.name == root)
        {
            let sort = match &param.kind {
                ParamKind::Value => Sort::Value,
                ParamKind::Values => Sort::Sequence(Box::new(Sort::Value)),
                ParamKind::Property(ty) => self.property(offset, ty)?,
                _ => {
                    return Err(
                        self.error(offset, "CFG destinations are not constraint expressions")
                    );
                }
            };
            Term::new(sort, TermKind::Param(index))
        } else if let Some(comparison) = self.comparisons.iter().find(|c| c.name == root) {
            let variant = path
                .next()
                .ok_or_else(|| self.error(offset, "expected an enum variant"))?;
            if !comparison.has_variant(variant) || path.next().is_some() {
                return Err(self.error(offset, format!("unknown comparison variant {name}")));
            }
            return Ok(Term::new(
                Sort::Enum(root.into()),
                TermKind::Enum(root.into(), variant.into()),
            ));
        } else {
            return Err(self.error(offset, format!("unknown constraint name {root}")));
        };
        for member in path {
            let Sort::Record(record) = &value.sort else {
                return Err(self.error(offset, "field access requires a record"));
            };
            let field = self
                .storage
                .records
                .iter()
                .find(|r| &r.name == record)
                .unwrap()
                .fields
                .iter()
                .find(|f| f.name == member)
                .ok_or_else(|| self.error(offset, format!("unknown field {record}.{member}")))?;
            let PropertyType::Named(ty) = &field.ty else {
                return Err(self.error(
                    offset,
                    "optional fields need an explicit presence operation",
                ));
            };
            value = Term::new(
                self.property(offset, ty)?,
                TermKind::Member(Box::new(value), member.into()),
            );
        }
        Ok(value)
    }

    fn term(&mut self, node: &Node) -> Result<Term, Error> {
        let offset = node.offset;
        Ok(match &node.kind {
            Kind::Name(name) => self.name(offset, name)?,
            Kind::Integer(value) => Term::integer(*value),
            Kind::Unary(op, value) => {
                let value = self.term(value)?;
                let sort = if *op == "!" { Sort::Bool } else { Sort::Int };
                self.expect(offset, &value, &sort)?;
                match (&value.kind, *op) {
                    (TermKind::Bool(value), "!") => Term::boolean(!value),
                    (TermKind::Int(value), "-") => Term::integer(
                        value
                            .checked_neg()
                            .ok_or_else(|| self.error(offset, "constraint arithmetic overflow"))?,
                    ),
                    _ => Term::new(sort, TermKind::Unary(op, Box::new(value))),
                }
            }
            Kind::Binary(op, lhs, rhs) => {
                let lhs = self.term(lhs)?;
                let rhs = self.term(rhs)?;
                self.expect(offset, &rhs, &lhs.sort)?;
                let sort = match *op {
                    "&&" | "||" => {
                        self.expect(offset, &lhs, &Sort::Bool)?;
                        Sort::Bool
                    }
                    "+" | "-" | "*" => {
                        self.expect(offset, &lhs, &Sort::Int)?;
                        Sort::Int
                    }
                    "<" | "<=" | ">" | ">=" => {
                        self.expect(offset, &lhs, &Sort::Int)?;
                        Sort::Bool
                    }
                    "==" | "!="
                        if matches!(
                            lhs.sort,
                            Sort::Bool | Sort::Int | Sort::Type | Sort::Enum(_)
                        ) =>
                    {
                        Sort::Bool
                    }
                    _ => return Err(self.error(offset, "unsupported constraint comparison")),
                };
                if let (TermKind::Int(a), TermKind::Int(b)) = (&lhs.kind, &rhs.kind) {
                    match *op {
                        "+" | "-" | "*" => Term::integer(
                            match *op {
                                "+" => a.checked_add(*b),
                                "-" => a.checked_sub(*b),
                                _ => a.checked_mul(*b),
                            }
                            .ok_or_else(|| self.error(offset, "constraint arithmetic overflow"))?,
                        ),
                        "==" => Term::boolean(a == b),
                        "!=" => Term::boolean(a != b),
                        "<" => Term::boolean(a < b),
                        "<=" => Term::boolean(a <= b),
                        ">" => Term::boolean(a > b),
                        ">=" => Term::boolean(a >= b),
                        _ => unreachable!("checked integer operation"),
                    }
                } else if let (TermKind::Bool(a), TermKind::Bool(b)) = (&lhs.kind, &rhs.kind) {
                    Term::boolean(match *op {
                        "&&" => *a && *b,
                        "||" => *a || *b,
                        "==" => a == b,
                        "!=" => a != b,
                        _ => unreachable!("checked boolean operation"),
                    })
                } else if let (TermKind::Enum(_, a), TermKind::Enum(_, b)) = (&lhs.kind, &rhs.kind)
                {
                    Term::boolean(if *op == "==" { a == b } else { a != b })
                } else {
                    // Only eliminate expressions whose evaluation is unreachable
                    // or whose constant value has no effects/errors of its own.
                    match (*op, &lhs.kind, &rhs.kind) {
                        ("&&", TermKind::Bool(false), _) | ("||", TermKind::Bool(true), _) => lhs,
                        ("&&", TermKind::Bool(true), _) | ("||", TermKind::Bool(false), _) => rhs,
                        ("&&", _, TermKind::Bool(true)) | ("||", _, TermKind::Bool(false)) => lhs,
                        _ => Term::new(sort, TermKind::Binary(op, Box::new(lhs), Box::new(rhs))),
                    }
                }
            }
            Kind::Call(name, args) if name == "result_type" => {
                let [
                    Node {
                        kind: Kind::Integer(index),
                        ..
                    },
                ] = args.as_slice()
                else {
                    return Err(self.error(offset, "result_type expects a constant result index"));
                };
                let patterns = self
                    .op
                    .signature
                    .results
                    .patterns()
                    .ok_or_else(|| self.error(offset, "result_type requires fixed results"))?;
                let index = usize::try_from(*index)
                    .ok()
                    .filter(|&i| i < patterns.len())
                    .ok_or_else(|| self.error(offset, "result_type index is out of bounds"))?;
                let mut term = Term::new(Sort::Type, TermKind::ResultType(index));
                term.types = self.possible(&patterns[index]);
                term
            }
            Kind::Call(name, args) if name == "all" => {
                let [
                    sequence,
                    Node {
                        kind: Kind::Lambda(name, body),
                        ..
                    },
                ] = args.as_slice()
                else {
                    return Err(self.error(offset, "all expects a sequence and |name| predicate"));
                };
                crate::model::identifier(self.source, offset, name)?;
                let sequence = self.term(sequence)?;
                let Sort::Sequence(element) = &sequence.sort else {
                    return Err(self.error(offset, "all expects a finite sequence"));
                };
                let id = self.next_local;
                self.next_local += 1;
                let previous = self.locals.insert(name.clone(), (id, *element.clone()));
                let body = self.term(body)?;
                self.expect(offset, &body, &Sort::Bool)?;
                if let Some(previous) = previous {
                    self.locals.insert(name.clone(), previous);
                } else {
                    self.locals.remove(name);
                }
                Term::new(
                    Sort::Bool,
                    TermKind::All(Box::new(sequence), id, Box::new(body)),
                )
            }
            Kind::Call(name, args) => {
                let query = match name.as_str() {
                    "type" => Query::TypeOf,
                    "len" => Query::Len,
                    "lanes" => Query::Lanes,
                    "min_bytes" => Query::MinBytes,
                    "is_fixed" => Query::IsFixed,
                    "is_ptr" => Query::IsPtr,
                    "is_vector" => Query::IsVector,
                    "is_scalar" => Query::IsScalar,
                    _ => {
                        return Err(
                            self.error(offset, format!("unknown constraint operation {name}"))
                        );
                    }
                };
                let [arg] = args.as_slice() else {
                    return Err(self.error(offset, "query expects one argument"));
                };
                let value = self.term(arg)?;
                let sort = match query {
                    Query::TypeOf => {
                        self.expect(offset, &value, &Sort::Value)?;
                        Sort::Type
                    }
                    Query::Len => {
                        if !matches!(value.sort, Sort::Sequence(_)) {
                            return Err(self.error(offset, "len expects a sequence"));
                        }
                        Sort::Int
                    }
                    Query::Lanes | Query::MinBytes => {
                        self.expect(offset, &value, &Sort::Type)?;
                        Sort::Int
                    }
                    _ => {
                        self.expect(offset, &value, &Sort::Type)?;
                        Sort::Bool
                    }
                };
                let known = if let Query::TypeOf = query {
                    if let TermKind::Param(index) = value.kind {
                        let index = operand_index(self.op, index);
                        let patterns = match &self.op.signature.operands {
                            crate::model::TypeList::Fixed(p)
                            | crate::model::TypeList::Variadic(p) => p,
                            _ => unreachable!("operand type prefix"),
                        };
                        self.possible(&patterns[index])
                    } else {
                        None
                    }
                } else {
                    None
                };
                if let Some(value) = value
                    .types
                    .as_ref()
                    .and_then(|set| self.known_query(query, set))
                {
                    value
                } else {
                    let mut term = Term::new(sort, TermKind::Query(query, Box::new(value)));
                    term.types = known;
                    term
                }
            }
            _ => return Err(self.error(offset, "expected a constraint expression")),
        })
    }

    // Conservative sets suffice: a predicate is folded only if every possible
    // type has the same answer. Dependent shape/element patterns may stay unknown.
    fn possible(&self, pattern: &Pattern) -> Option<TypeSet> {
        match pattern {
            Pattern::Class(set) | Pattern::Bind(_, set) | Pattern::ShapeOf(_, set) => {
                Some(set.clone())
            }
            Pattern::Exact(name) => self.types.exact.get(name).cloned(),
            Pattern::Same(slot) => {
                let patterns = match &self.op.signature.operands {
                    crate::model::TypeList::Fixed(p) | crate::model::TypeList::Variadic(p) => p,
                    _ => return None,
                };
                patterns
                    .iter()
                    .chain(self.op.signature.results.patterns().unwrap_or(&[]))
                    .find_map(|p| {
                        if let Pattern::Bind(id, set) = p
                            && id == slot
                        {
                            Some(set.clone())
                        } else {
                            None
                        }
                    })
            }
            _ => None,
        }
    }

    fn known_query(&self, query: Query, set: &TypeSet) -> Option<Term> {
        let mut answer = None;
        for (&code, &shapes) in &set.0 {
            let scalar = self.types.scalars.iter().find(|s| s.code == code)?;
            for bit in 0..32 {
                if shapes & (1 << bit) == 0 {
                    continue;
                }
                let value = match query {
                    Query::IsFixed => i128::from(bit > 0 && bit < 16),
                    Query::IsScalar => i128::from(bit == 0),
                    Query::IsVector => i128::from(bit != 0),
                    Query::IsPtr => i128::from(scalar.kind == ScalarKind::Pointer && bit == 0),
                    Query::Lanes => 1i128 << (bit % 16),
                    Query::MinBytes => i128::from(scalar.bits?.div_ceil(8)) << (bit % 16),
                    _ => return None,
                };
                if answer.is_some_and(|old| old != value) {
                    return None;
                }
                answer = Some(value);
            }
        }
        answer.map(|value| {
            if matches!(query, Query::Lanes | Query::MinBytes) {
                Term::integer(value)
            } else {
                Term::boolean(value != 0)
            }
        })
    }
}

fn operand_index(op: &Op, param: usize) -> usize {
    op.params[..param]
        .iter()
        .filter(|p| p.kind == ParamKind::Value)
        .count()
}

fn describe(node: &Node) -> String {
    match &node.kind {
        Kind::Name(name) => name.clone(),
        Kind::Integer(n) => n.to_string(),
        Kind::Text(text) => format!("{text:?}"),
        Kind::Unary(op, value) => format!("{op}({})", describe(value)),
        Kind::Binary(op, lhs, rhs) => format!("({} {op} {})", describe(lhs), describe(rhs)),
        Kind::Call(name, args) => format!(
            "{name}({})",
            args.iter().map(describe).collect::<Vec<_>>().join(", ")
        ),
        Kind::Lambda(name, body) => format!("|{name}| {}", describe(body)),
        _ => "invalid constraint".into(),
    }
}

struct Emitter<'a> {
    op: &'a Op,
    projections: BTreeMap<String, String>,
    error: String,
    storage_used: std::cell::Cell<bool>,
}

impl Emitter<'_> {
    fn operand(&self, term: &Term) -> String {
        let code = self.term(term);
        if matches!(
            term.kind,
            TermKind::Binary(..) | TermKind::Query(Query::Len, _)
        ) {
            format!("({code})")
        } else {
            code
        }
    }

    fn term(&self, term: &Term) -> String {
        let numeric = |code: String| {
            if term.sort == Sort::Int {
                format!("i128::from({code})")
            } else {
                code
            }
        };
        match &term.kind {
            TermKind::Bool(value) => value.to_string(),
            TermKind::Int(value) => format!("{value}i128"),
            TermKind::Enum(ty, value) => format!("crate::{ty}::{value}"),
            TermKind::Param(index) => {
                self.storage_used.set(true);
                numeric(self.projections[&self.op.params[*index].name].clone())
            }
            TermKind::Bound(id) => numeric(format!("_v{id}")),
            TermKind::Member(value, field) => numeric(format!("({}).{field}", self.term(value))),
            TermKind::ResultType(index) => format!("_results[{index}]"),
            TermKind::Unary("!", value) => format!("!({})", self.term(value)),
            TermKind::Unary("-", value) => format!(
                "({}).checked_neg().ok_or_else(|| {})?",
                self.term(value),
                self.error
            ),
            TermKind::Unary(_, _) => unreachable!("checked unary operator"),
            TermKind::Binary(op, lhs, rhs) => match *op {
                "+" | "-" | "*" => {
                    let (lhs, rhs) = (self.term(lhs), self.term(rhs));
                    let method = match *op {
                        "+" => "checked_add",
                        "-" => "checked_sub",
                        _ => "checked_mul",
                    };
                    format!("({lhs}).{method}({rhs}).ok_or_else(|| {})?", self.error)
                }
                _ => format!("{} {op} {}", self.operand(lhs), self.operand(rhs)),
            },
            TermKind::Query(query, value) => {
                if let Query::TypeOf = query
                    && let TermKind::Param(index) = value.kind
                {
                    return format!("_operands[{}]", operand_index(self.op, index));
                }
                let known_valid = value.types.is_some();
                let value = self.term(value);
                match query {
                    Query::TypeOf => format!("self.dfg.value_type({value})"),
                    Query::Len => format!("({value}).len() as i128"),
                    Query::Lanes if known_valid => format!("i128::from(({value}).lane_count())"),
                    Query::Lanes => format!(
                        "{{ let ty = {value}; if !ty.is_valid() {{ return Err({}); }} i128::from(ty.lane_count()) }}",
                        self.error
                    ),
                    Query::MinBytes => format!(
                        "i128::from(({value}).min_size_bytes().ok_or_else(|| {})?)",
                        self.error
                    ),
                    Query::IsFixed => {
                        format!("({value}).as_vector().is_some_and(|v| v.is_fixed())")
                    }
                    Query::IsPtr => format!("({value}).is_ptr()"),
                    Query::IsVector => format!("({value}).is_vector()"),
                    Query::IsScalar => format!("({value}).is_scalar()"),
                }
            }
            TermKind::All(sequence, id, body) => format!(
                "{{ let mut ok{id} = true; for &_v{id} in ({}).iter() {{ if !({}) {{ ok{id} = false; break; }} }} ok{id} }}",
                self.term(sequence),
                self.term(body)
            ),
        }
    }
}

pub(crate) fn generate(defs: &Definitions) -> String {
    let mut groups = BTreeMap::<String, Vec<&str>>::new();
    for op in &defs.ops {
        let format = defs
            .storage
            .formats
            .iter()
            .find(|f| f.name == op.format)
            .unwrap();
        let mut body = String::new();
        let mut storage_used = false;
        for constraint in &op.constraints {
            if matches!(constraint.condition.kind, TermKind::Bool(true)) {
                continue;
            }
            let error = format!("self.constraint_error(_inst, {:?})", constraint.text);
            let projections = crate::packing::projections(
                op,
                format,
                "&self.dfg",
                |name| {
                    format!(
                        "*_f{}",
                        format.fields.iter().position(|f| f.name == name).unwrap()
                    )
                },
                |value| format!("{value}.ok_or_else(|| {error})?"),
            )
            .into_iter()
            .collect();
            let emitter = Emitter {
                op,
                projections,
                error,
                storage_used: std::cell::Cell::new(false),
            };
            let condition = emitter.term(&constraint.condition);
            storage_used |= emitter.storage_used.get();
            writeln!(
                body,
                "                if !({condition}) {{ return Err({}); }}",
                emitter.error
            )
            .unwrap();
        }
        // Type-only predicates use already checked operand/result slices and
        // therefore work for alternate instruction layouts without reprojection.
        if storage_used {
            let fields = format
                .fields
                .iter()
                .enumerate()
                .map(|(i, f)| format!("{}: _f{i}", f.name))
                .collect::<Vec<_>>()
                .join(", ");
            let mismatch = if defs.storage.formats.len() + defs.storage.alternatives.len() == 1 {
                ""
            } else {
                " else { unreachable!(\"checked constraint storage\") }"
            };
            body.insert_str(
                0,
                &format!(
                    "                let InstructionData::{} {{ {fields} }} = data{mismatch};\n",
                    format.name
                ),
            );
        }
        groups.entry(body).or_default().push(&op.name);
    }
    let mut out = String::from(
        "// @generated by veloc-opgen. Edit the .ops definitions instead.\nimpl Function {\n    fn validate_constraints(&self, _inst: Inst, data: &InstructionData, _operands: &[Type], _results: &[Type]) -> Result<()> {\n        match data.opcode() {\n",
    );
    for (body, names) in groups {
        let names = names
            .iter()
            .map(|name| format!("Opcode::{name}"))
            .collect::<Vec<_>>()
            .join(" | ");
        writeln!(
            out,
            "            {names} => {{\n{body}                Ok(())\n            }},"
        )
        .unwrap();
    }
    out.push_str("        }\n    }\n}\n");
    out
}
