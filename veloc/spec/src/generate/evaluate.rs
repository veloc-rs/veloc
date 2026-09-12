//! Compile checked semantic recipes into straight-line scalar evaluators.
//!
//! Only legal scalar signatures representable by MIR Constant are emitted. Widths,
//! operand references and result layouts are resolved here, not during folding.

use std::fmt::Write;

use veloc_semantics::{BvOp, ComparisonRef, Conversion, IntPredicate, Sort, Step, TypeRef};

use crate::Error;
use crate::model::{Binding, Definitions, Semantic};
use crate::semantic::Instance;
use crate::types::{Primitive, Scalar};

pub(crate) struct Plan {
    operations: Vec<Operation>,
}

struct Operation {
    opcode: usize,
    cases: Vec<Case>,
    properties: Vec<String>,
}

struct Case {
    instance: Instance,
    scalars: Vec<usize>,
    variants: Vec<String>,
}

impl Plan {
    pub(crate) fn prepare(defs: &Definitions, source: &str) -> Result<Self, Error> {
        let mut operations = Vec::new();
        for (opcode, op) in defs.ops.iter().enumerate() {
            let Some(sem) = &op.semantics else { continue };
            let mut cases = Vec::new();
            for instance in &sem.instances {
                if !instance.scalar {
                    continue;
                }
                let scalars = instance
                    .codes
                    .iter()
                    .map(|code| {
                        defs.types
                            .scalars
                            .iter()
                            .position(|s| s.code == *code)
                            .expect("checked scalar code")
                    })
                    .collect::<Vec<_>>();
                let Some(variants) = scalars
                    .iter()
                    .map(|&i| constant(&defs.types.scalars[i]))
                    .collect::<Option<Vec<_>>>()
                else {
                    continue;
                };
                cases.push(Case {
                    instance: instance.clone(),
                    scalars,
                    variants,
                });
            }
            let properties = sem
                .properties
                .iter()
                .map(|property| {
                    op.bindings()
                        .iter()
                        .find_map(|(field, binding)| match binding {
                            Binding::Name(name) if name == property => Some(field.clone()),
                            _ => None,
                        })
                        .ok_or_else(|| {
                            Error::at(
                                source,
                                op.offset,
                                "semantic comparison properties require a direct storage field",
                            )
                        })
                })
                .collect::<Result<_, _>>()?;
            operations.push(Operation {
                opcode,
                cases,
                properties,
            });
        }
        Ok(Self { operations })
    }
}

pub(crate) fn generate(defs: &Definitions, plan: &Plan) -> String {
    let mut code = String::from(
        "// @generated from checked operation semantics.\n\
         #[allow(unused_variables, unreachable_patterns)]\n\
         pub fn evaluate(opcode: Opcode, args: &[ScalarConst], results: &[Type], properties: &[IntCC]) -> Option<Vec<ScalarConst>> {\n\
         match opcode {\n",
    );
    let mut supported = Vec::new();
    for prepared in &plan.operations {
        let op = &defs.ops[prepared.opcode];
        let sem = op.semantics.as_ref().expect("prepared semantic operation");
        let mut arms = String::new();
        for case in &prepared.cases {
            let instance = &case.instance;
            let variants = &case.variants;
            let scalars = case
                .scalars
                .iter()
                .map(|&i| &defs.types.scalars[i])
                .collect::<Vec<_>>();
            let inputs = sem.inputs as usize;
            let args = variants[..inputs]
                .iter()
                .enumerate()
                .map(|(i, _)| format!("a{i}"))
                .collect::<Vec<_>>()
                .join(", ");
            let results = scalars[inputs..]
                .iter()
                .map(|s| format!("Type::{}", s.exact()))
                .collect::<Vec<_>>()
                .join(", ");
            let properties = (0..sem.properties.len())
                .map(|i| format!("p{i}"))
                .collect::<Vec<_>>()
                .join(", ");
            let guard = scalars[..inputs]
                .iter()
                .enumerate()
                .map(|(i, s)| format!("a{i}.ty() == Type::{}", s.exact()))
                .collect::<Vec<_>>()
                .join(" && ");
            let guard = if guard.is_empty() {
                String::new()
            } else {
                format!(" if {guard}")
            };
            writeln!(arms, "([{args}], [{results}], [{properties}]){guard} => {{").unwrap();
            emit(defs, sem, instance, &variants[inputs..], &mut arms);
            arms.push_str("},\n");
        }
        if !arms.is_empty() {
            supported.push(format!("Opcode::{}", op.name));
            writeln!(
                code,
                "Opcode::{} => match (args, results, properties) {{\n{arms}_ => None,\n}},",
                op.name
            )
            .unwrap();
        }
    }
    code.push_str("_ => None,\n}\n}\n");
    let supported = if supported.is_empty() {
        "false".into()
    } else {
        format!("matches!(opcode, {})", supported.join(" | "))
    };
    writeln!(code, "/// Whether this opcode has a generated scalar constant evaluator.\npub const fn can_fold(opcode: Opcode) -> bool {{ {supported} }}").unwrap();
    code.push_str(&properties(defs, plan));
    code.push_str(&algebraic_rules(defs));
    code
}

fn properties(defs: &Definitions, plan: &Plan) -> String {
    let mut ops = String::from(
        "#[allow(unused_variables)] pub(crate) fn properties(data: &veloc_mir::InstView<'_>) -> smallvec::SmallVec<[IntCC; 1]> { match data.opcode() {\n",
    );
    for prepared in &plan.operations {
        let op = &defs.ops[prepared.opcode];
        let fields = &prepared.properties;
        if fields.is_empty() {
            continue;
        }
        writeln!(ops, "Opcode::{} => {{ let veloc_mir::InstView::{} {{ {}, .. }} = data else {{ unreachable!(\"checked semantic property layout\") }}; smallvec::smallvec![{}] }},", op.name, op.format, fields.join(", "), fields.iter().map(|f| format!("*{f}")).collect::<Vec<_>>().join(", ")).unwrap();
    }
    ops.push_str("_ => smallvec::smallvec![],\n} }\n");
    ops
}

fn constant(scalar: &Scalar) -> Option<String> {
    match scalar.ty {
        Primitive::Bool => Some("Bool".into()),
        Primitive::Int(bits) => match bits {
            bits @ (8 | 16 | 32 | 64) => Some(format!("I{bits}")),
            _ => None,
        },
        Primitive::Float(_) | Primitive::Ptr => None,
    }
}

fn width(sort: Sort) -> u16 {
    match sort {
        Sort::Bool => 1,
        Sort::Bv(width) => width.bits(),
    }
}

fn mask(sort: Sort) -> u128 {
    veloc_semantics::Width::new(width(sort)).unwrap().mask()
}

fn emit(
    defs: &Definitions,
    sem: &Semantic,
    instance: &Instance,
    results: &[String],
    code: &mut String,
) {
    let inputs = sem.inputs as usize;
    let sort = |ty: TypeRef| match ty {
        TypeRef::Input(i) => instance.sorts[i as usize],
        TypeRef::Result(i) => instance.sorts[inputs + i as usize],
        TypeRef::Fixed(sort) => sort,
    };
    let mut sorts = Vec::new();
    for (i, step) in sem.steps.iter().enumerate() {
        let (ty, expression) = match step {
            Step::Input(input) => {
                let ty = instance.sorts[*input as usize];
                (ty, format!("u128::from(a{input}.to_bits())"))
            }
            Step::Const { value, ty } => {
                let ty = sort(*ty);
                let value = value.eval(width(ty)).unwrap() & mask(ty);
                (ty, format!("{value}u128"))
            }
            Step::Apply { op, args } => {
                let ty = sorts[args[0] as usize];
                (
                    ty,
                    operation(
                        *op,
                        width(ty),
                        &args.iter().map(|i| format!("s{i}")).collect::<Vec<_>>(),
                    ),
                )
            }
            Step::Compare { kind, lhs, rhs } => {
                let bits = width(sorts[*lhs as usize]);
                let value = match kind {
                    ComparisonRef::Fixed(p) => comparison(*p, bits, *lhs, *rhs),
                    ComparisonRef::Property(i) => {
                        let predicates = defs
                            .comparisons
                            .iter()
                            .find(|c| c.name == "IntCC")
                            .expect("checked comparison type")
                            .integer_predicates()
                            .unwrap();
                        let arms = predicates
                            .into_iter()
                            .map(|(name, p)| {
                                format!("IntCC::{name} => {},", comparison(p, bits, *lhs, *rhs))
                            })
                            .collect::<String>();
                        format!("match p{i} {{ {arms} }}")
                    }
                };
                (Sort::Bool, value)
            }
            Step::Convert { kind, arg, to } => {
                let ty = sort(*to);
                let value = match kind {
                    Conversion::ZeroExtend | Conversion::Truncate => format!("s{arg}"),
                    Conversion::SignExtend => {
                        let shift = 128 - width(sorts[*arg as usize]);
                        format!("(((s{arg} << {shift}) as i128 >> {shift}) as u128)")
                    }
                };
                (ty, format!("{value} & {}u128", mask(ty)))
            }
            Step::Select { cond, yes, no } => (
                sorts[*yes as usize],
                format!("if s{cond} != 0 {{ s{yes} }} else {{ s{no} }}"),
            ),
        };
        // Pure unused recipe nodes are harmless. Keeping their names also preserves
        // stable step numbering; rustc can eliminate them without runtime graph work.
        writeln!(
            code,
            "#[allow(unused_variables)] let s{i}: u128 = {expression};"
        )
        .unwrap();
        sorts.push(ty);
    }
    for (guard, _) in &sem.traps {
        writeln!(code, "if s{guard} != 0 {{ return None; }}").unwrap();
    }
    let values = sem
        .outputs
        .iter()
        .zip(results)
        .map(|(i, variant)| {
            if variant == "Bool" {
                format!("ScalarConst::from(s{i} != 0)")
            } else {
                format!(
                    "ScalarConst::from(s{i} as {})",
                    variant.to_ascii_lowercase()
                )
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(code, "Some(alloc::vec![{values}])").unwrap();
}
fn comparison(p: IntPredicate, bits: u16, lhs: u16, rhs: u16) -> String {
    let operator = match p.outcomes() {
        0 => return "0u128".into(),
        1 => "<",
        2 => "==",
        3 => "<=",
        4 => ">",
        5 => "!=",
        6 => ">=",
        7 => return "1u128".into(),
        _ => unreachable!("checked comparison outcomes"),
    };
    let operand = |i| {
        if p.signed() && !matches!(p.outcomes(), 2 | 5) {
            format!("(s{i} ^ {}u128)", 1u128 << (bits - 1))
        } else {
            format!("s{i}")
        }
    };
    format!("u128::from({} {operator} {})", operand(lhs), operand(rhs))
}

fn algebraic_rules(defs: &Definitions) -> String {
    let mut code = String::from(
        "#[allow(unused_variables, unreachable_patterns)] fn algebraic(op: Opcode, args: &[Value; 2], constants: &[Option<ScalarConst>; 2]) -> Option<Replacement> { match op {\n",
    );
    for op in &defs.ops {
        let Some(sem) = &op.semantics else { continue };
        // Only reviewed primitive laws are currently available. No speculative
        // inference for effects, traps, or composed expressions.
        if sem.primitive().is_none()
            || (op.identity.is_none()
                && op.absorbing.is_none()
                && !op.traits.iter().any(|t| t == "IDEMPOTENT"))
        {
            continue;
        }
        writeln!(code, "Opcode::{} => {{", op.name).unwrap();
        for (value, absorbing) in [(op.identity, false), (op.absorbing, true)] {
            let Some(value) = value else { continue };
            let patterns = [
                ("I8", 8),
                ("I16", 16),
                ("I32", 32),
                ("I64", 64),
                ("Bool", 1),
            ]
            .into_iter()
            .map(|(variant, bits)| {
                let raw = value.eval(bits).unwrap();
                let ty = if variant == "Bool" { "BOOL" } else { variant };
                format!("(c.ty() == Type::{ty} && c.to_bits() == {raw}u64)")
            })
            .collect::<Vec<_>>()
            .join(" || ");
            for i in 0..2 {
                let result = if absorbing {
                    "Replacement::Constants(alloc::vec![c])".into()
                } else {
                    format!("Replacement::Value(args[{}])", 1 - i)
                };
                writeln!(
                    code,
                    "if let Some(c) = constants[{i}] && ({patterns}) {{ return Some({result}); }}"
                )
                .unwrap();
            }
        }
        if op.traits.iter().any(|t| t == "IDEMPOTENT") {
            code.push_str("if args[0] == args[1] { return Some(Replacement::Value(args[0])); }\n");
        }
        code.push_str("None\n},\n");
    }
    code.push_str("_ => None,\n}\n}\n");
    code
}

/// Concrete backend for the trusted bitvector vocabulary. The reference evaluator
/// and this backend are checked by offline differential tests; no runtime BvOp
/// dispatch or dependency on the semantic graph library is emitted.
fn operation(op: BvOp, bits: u16, args: &[String]) -> String {
    let x = &args[0];
    let y = args.get(1).map(String::as_str).unwrap_or("0u128");
    let mask = veloc_semantics::Width::new(bits).unwrap().mask();
    let signed = |v: &str| format!("((({v} << {}) as i128) >> {})", 128 - bits, 128 - bits);
    let signed_y = signed(y);
    let signed_y = &signed_y[1..signed_y.len() - 1];
    let body = match op {
        BvOp::Add => format!("{x}.wrapping_add({y})"),
        BvOp::Sub => format!("{x}.wrapping_sub({y})"),
        BvOp::Mul => format!("{x}.wrapping_mul({y})"),
        BvOp::Neg => format!("{x}.wrapping_neg()"),
        BvOp::And => format!("{x} & {y}"),
        BvOp::Or => format!("{x} | {y}"),
        BvOp::Xor => format!("{x} ^ {y}"),
        BvOp::Shl => format!("if {y} >= {bits} {{ 0 }} else {{ {x} << {y} }}"),
        BvOp::LShr => format!("if {y} >= {bits} {{ 0 }} else {{ {x} >> {y} }}"),
        BvOp::AShr => format!("({} >> {y}.min({})) as u128", signed(x), bits - 1),
        BvOp::UDiv => format!("{x}.checked_div({y}).unwrap_or({mask}u128)"),
        BvOp::URem => format!("if {y} == 0 {{ {x} }} else {{ {x} % {y} }}"),
        BvOp::SDiv => format!(
            "if {y} == 0 {{ if {} < 0 {{ 1 }} else {{ {mask}u128 }} }} else {{ {}.wrapping_div({signed_y}) as u128 }}",
            signed(x),
            signed(x)
        ),
        BvOp::SRem => format!(
            "if {y} == 0 {{ {x} }} else {{ {}.wrapping_rem({signed_y}) as u128 }}",
            signed(x)
        ),
        BvOp::Clz => format!("({x}.leading_zeros() - {}) as u128", 128 - bits),
        BvOp::Ctz => format!("{x}.trailing_zeros().min({bits}) as u128"),
        BvOp::Popcnt => format!("{x}.count_ones() as u128"),
    };
    format!("({body}) & {mask}u128")
}
