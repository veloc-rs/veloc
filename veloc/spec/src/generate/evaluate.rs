//! Compile checked semantic recipes into straight-line scalar evaluators.
//!
//! Only legal scalar signatures representable by MIR Constant are emitted. Widths,
//! operand references and result layouts are resolved here, not during folding.

use std::fmt::Write;

use veloc_semantics::{BvOp, ComparisonRef, Conversion, IntPredicate, Sort, Step, TypeRef};

use crate::Error;
use crate::model::{Binding, Definitions, Semantic, expr::Emitter};
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
                if !instance.scalar || instance.error.is_some() {
                    continue;
                }
                let scalars = instance
                    .kinds
                    .iter()
                    .map(|code| {
                        defs.types
                            .scalars
                            .iter()
                            .position(|s| s.ty == *code)
                            .expect("checked scalar kind")
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
         pub fn fold(dfg: &veloc_mir::dfg::DataFlowGraph, inst: veloc_mir::Inst, mut constant: impl FnMut(Value) -> Option<ScalarConst>) -> Option<smallvec::SmallVec<[ScalarConst; 2]>> {\n\
         let data = dfg.inst(inst);\n\
         match data.opcode() {\n",
    );
    let mut supported = Vec::new();
    let mut inputs_by_count = std::collections::BTreeMap::<usize, Vec<String>>::new();
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
            writeln!(arms, "([{args}], [{results}]){guard} => {{").unwrap();
            emit(sem, instance, &variants[inputs..], &mut arms);
            arms.push_str("},\n");
        }
        if !arms.is_empty() {
            supported.push(format!("Opcode::{} => true,", op.name));
            let inputs = sem.inputs as usize;
            inputs_by_count
                .entry(inputs)
                .or_default()
                .push(format!("Opcode::{}", op.name));
            let results = prepared.cases[0].instance.kinds.len() - inputs;
            let constraints = applicability(op);
            let args = (0..inputs)
                .map(|i| format!("constant(operands[{i}])?"))
                .collect::<Vec<_>>()
                .join(", ");
            let types = (0..results)
                .map(|i| format!("dfg.value_type(outputs[{i}])"))
                .collect::<Vec<_>>()
                .join(", ");
            let fields = prepared
                .properties
                .iter()
                .enumerate()
                .map(|(i, field)| format!("{field}: p{i}"))
                .collect::<Vec<_>>()
                .join(", ");
            let properties = if fields.is_empty() {
                String::new()
            } else {
                format!(
                    "let veloc_mir::InstView::{} {{ {fields}, .. }} = data else {{ unreachable!(\"semantic property layout\") }};\n",
                    op.format
                )
            };
            writeln!(
                code,
                "Opcode::{} => {{\nlet operands = dfg.operands(inst);\nlet outputs = dfg.inst_results(inst);\nassert_eq!(operands.len(), {inputs}, \"semantic operand count\");\nassert_eq!(outputs.len(), {results}, \"semantic result count\");\nlet args = [{args}];\nlet results = [{types}];\nfor (&value, constant) in operands.iter().zip(&args) {{ assert_eq!(dfg.value_type(value), constant.ty(), \"constant fact type\"); }}\n{properties}{constraints}match (&args, &results) {{\n{arms}_ => None,\n}}\n}},",
                op.name,
            )
            .unwrap();
        }
    }
    code.push_str("_ => None,\n}\n}\n");
    let supported = if supported.is_empty() {
        "false".into()
    } else {
        format!("match opcode {{ {} _ => false }}", supported.join("\n"))
    };
    writeln!(code, "/// Whether this opcode has a generated scalar constant evaluator.\npub const fn can_fold(opcode: Opcode) -> bool {{ {supported} }}").unwrap();
    // Keep scheduling requirements beside the operand reads emitted above.
    // This is readiness for concrete evaluation, not a second simplifier.
    code.push_str("/// Whether the facts required by the evaluator are available.\n\
        /// A ready instruction may still fail to fold, e.g. because it traps.\n\
        #[allow(unused_variables, unused_mut)]\n\
        pub fn ready(dfg: &veloc_mir::dfg::DataFlowGraph, inst: veloc_mir::Inst, mut known: impl FnMut(Value) -> bool) -> bool {\n\
        let operands = dfg.operands(inst);\n\
        match dfg.opcode(inst) {\n");
    for (count, ops) in inputs_by_count {
        let condition = if count == 0 {
            "true".to_owned()
        } else {
            (0..count)
                .map(|i| format!("known(operands[{i}])"))
                .collect::<Vec<_>>()
                .join(" && ")
        };
        writeln!(code, "{} => {{ debug_assert_eq!(operands.len(), {count}, \"semantic operand count\"); {condition} }},", ops.join(" | ")).unwrap();
    }
    code.push_str("_ => false,\n}\n}\n");
    code.push_str(&properties(defs, plan));
    code
}

/// Emit type-only applicability checks once per opcode. Previously these were
/// expanded over every scalar/vector combination as const assertions in the MIR
/// crate. Constant folding only sees concrete argument/result types, so checking
/// the same contract here is both smaller and keeps invalid input fallible.
fn applicability(op: &crate::model::Op) -> String {
    let constraints = op
        .constraints
        .iter()
        .filter(|constraint| constraint.type_only && !constraint.redundant())
        .collect::<Vec<_>>();
    if constraints.is_empty() {
        return String::new();
    }

    let mut emitter = Emitter::types(op, std::collections::BTreeMap::new(), "args", "results");
    for ty in emitter.operand_types.values_mut() {
        *ty = format!("({ty}).ty()");
    }
    constraints
        .into_iter()
        .map(|constraint| constraint.emit(&emitter, "return None"))
        .collect()
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

fn emit(sem: &Semantic, instance: &Instance, results: &[String], code: &mut String) {
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
                        format!("u128::from(p{i}.test({bits}, s{lhs}, s{rhs}))")
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
    writeln!(code, "Some(smallvec::smallvec![{values}])").unwrap();
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
