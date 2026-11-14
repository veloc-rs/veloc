//! Checked selection of existing successor occurrences, independent of CFG edits.
use super::{Binding, Definitions, Op, ParamKind, Pattern, Projection, TypeList};
use crate::{
    Error,
    syntax::{Kind, Node},
};
use std::fmt::Write;

pub(crate) enum Control {
    Jump,
    Branch,
    Table,
}

pub(crate) fn parse(source: &str, node: Node, op: &Op) -> Result<Control, Error> {
    let fail = || {
        Error::at(
            source,
            node.offset,
            "control requires jump(dest), branch(condition, then_dest, else_dest), or table(index, cases, default), in signature/storage order with bool/i32 selectors and no results",
        )
    };
    let Kind::Call(name, args) = &node.kind else {
        return Err(fail());
    };
    let (control, kinds, selector) = match name.as_str() {
        "jump" => (Control::Jump, vec![ParamKind::Successor], None),
        "branch" => (
            Control::Branch,
            vec![ParamKind::Value, ParamKind::Successor, ParamKind::Successor],
            Some("Type::BOOL"),
        ),
        "table" => (
            Control::Table,
            vec![
                ParamKind::Value,
                ParamKind::Successors,
                ParamKind::Successor,
            ],
            Some("Type::I32"),
        ),
        _ => return Err(fail()),
    };
    if args.len() != kinds.len()
        || op.params.len() != kinds.len()
        || op.signature.results.patterns() != Some(&[][..])
        || op
            .params
            .iter()
            .zip(&kinds)
            .zip(args)
            .any(|((p, kind), arg)| {
                p.moves || p.kind != *kind || !matches!(&arg.kind, Kind::Name(n) if *n == p.name)
            })
    {
        return Err(fail());
    }
    let TypeList::Variadic(patterns) = &op.signature.operands else {
        return Err(fail());
    };
    if let Some(selector) = selector {
        if !matches!(patterns.as_slice(), [Pattern::Exact(ty)] if ty == selector) {
            return Err(fail());
        }
    } else if !patterns.is_empty() {
        return Err(fail());
    }
    if !matches!(op.projection, Projection::Packed(_)) {
        return Err(fail());
    }
    Ok(control)
}

/// Verify that logical edge indices agree with visitation/storage order.
pub(crate) fn validate(
    source: &str,
    op: &Op,
    layout: &crate::storage::Format,
) -> Result<(), Error> {
    if op.control.is_none() {
        return Ok(());
    }
    if !op.constraints.is_empty() {
        return Err(Error::at(
            source,
            op.offset,
            "control selection does not support additional instruction constraints",
        ));
    }
    let mut order = Vec::new();
    for field in &layout.fields {
        match op.bindings().get(&field.name) {
            Some(Binding::Name(name)) => order.push(name.as_str()),
            Some(Binding::Table { cases, default }) => {
                order.push(cases);
                order.push(default);
            }
            _ => {
                return Err(Error::at(
                    source,
                    op.offset,
                    "control storage must preserve parameter order",
                ));
            }
        }
    }
    if order
        != op
            .params
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>()
    {
        return Err(Error::at(
            source,
            op.offset,
            "control storage must preserve parameter order",
        ));
    }
    Ok(())
}

pub(crate) fn generate(defs: &Definitions) -> String {
    let mut out = String::from(
        "/// Resolve successor knowledge using supplied facts; no IR edits.\n#[allow(unused_variables, unused_mut)]\npub(crate) fn successors(view: veloc_mir::InstView<'_>, mut fact: impl FnMut(veloc_mir::Value) -> Fact) -> Successors { match view.opcode() {\n",
    );
    for op in &defs.ops {
        let Some(control) = &op.control else {
            continue;
        };
        writeln!(out, "Opcode::{} => {{", op.name).unwrap();
        if matches!(control, Control::Jump) {
            out.push_str("Successors::All\n},\n");
            continue;
        }
        let format = defs
            .storage
            .formats
            .iter()
            .find(|f| f.name == op.format)
            .unwrap();
        let fields = format
            .fields
            .iter()
            .map(|f| format!("{}: field_{}", f.name, f.name))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            out,
            "let veloc_mir::InstView::{} {{ {fields} }} = view else {{ unreachable!() }};",
            op.format
        )
        .unwrap();
        let input = |name: &str| {
            op.inputs[name].emit(&|f| format!("field_{f}"), &|v| {
                format!("{v}.expect(\"validated control storage\")")
            })
        };
        writeln!(out, "let value = match fact({}) {{ Fact::Constant(value) => value, Fact::Unknown => return Successors::Pending, Fact::Varying => return Successors::All }};", input(&op.params[0].name)).unwrap();
        match control {
            Control::Branch => out.push_str("Successors::One(usize::from(value.to_bits() == 0))\n"),
            Control::Table => writeln!(
                out,
                "Successors::One(value.to_bits().min(({}).len() as u64) as usize)",
                input(&op.params[1].name)
            )
            .unwrap(),
            Control::Jump => unreachable!(),
        }
        out.push_str("},\n");
    }
    out.push_str("_ => Successors::All,\n} }\n");
    out
}
