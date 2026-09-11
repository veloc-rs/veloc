//! Checked access projections over logical operands, independent of opcode names.
use crate::{
    Error,
    model::{Definitions, Op, Param, ParamKind},
    syntax::{Kind, Node},
};
use std::collections::BTreeMap;
use std::fmt::Write;

pub(crate) struct Access {
    write: bool,
    ptr: String,
    offset: String,
    value: Option<String>,
}

pub(crate) fn check(source: &str, node: Node, params: &[Param]) -> Result<Access, Error> {
    let fail = || {
        Error::at(
            source,
            node.offset,
            "expected read(pointer, offset) or write(pointer, offset, value)",
        )
    };
    let Kind::Call(kind, args) = node.kind else {
        return Err(fail());
    };
    if !matches!(kind.as_str(), "read" | "write")
        || args.len() != if kind == "read" { 2 } else { 3 }
    {
        return Err(fail());
    }
    let names = args
        .iter()
        .map(|n| match &n.kind {
            Kind::Name(s) => Ok(s.clone()),
            _ => Err(fail()),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let param = |name: &str| params.iter().find(|p| p.name == name).map(|p| &p.kind);
    if param(&names[0]) != Some(&ParamKind::Value)
        || !matches!(param(&names[1]), Some(ParamKind::Property(t)) if matches!(t.as_str(), "u32" | "i32"))
        || (kind == "write" && param(&names[2]) != Some(&ParamKind::Value))
    {
        return Err(fail());
    }
    Ok(Access {
        write: kind == "write",
        ptr: names[0].clone(),
        offset: names[1].clone(),
        value: (kind == "write").then(|| names[2].clone()),
    })
}

impl Access {
    pub(crate) fn effect(&self) -> &'static str {
        if self.write { "WRITE" } else { "READ" }
    }
}

pub(crate) fn validate(source: &str, op: &Op) -> Result<(), Error> {
    if let Some(access) = &op.access {
        let pointer_index = op
            .params
            .iter()
            .filter(|p| p.kind == ParamKind::Value)
            .position(|p| p.name == access.ptr)
            .expect("checked pointer operand");
        if !matches!(op.signature.operands.patterns().and_then(|p| p.get(pointer_index)),
            Some(crate::model::Pattern::Exact(name)) if name == "PTR")
        {
            return Err(Error::at(
                source,
                op.offset,
                "access address must have type PTR",
            ));
        }
        if !access.write && op.signature.results.patterns().is_none_or(|p| p.len() != 1) {
            return Err(Error::at(
                source,
                op.offset,
                "read access requires exactly one result",
            ));
        }
    }
    Ok(())
}

pub(crate) fn generate(defs: &Definitions) -> String {
    let mut out = String::from(
        "impl InstructionView<'_> { pub fn memory_access(&self, dfg: &crate::dfg::DataFlowGraph, result: Option<crate::Type>) -> Option<crate::memory::Access> { let _ = (dfg, result); match self.opcode() {\n",
    );
    for op in &defs.ops {
        let Some(access) = &op.access else {
            continue;
        };
        let format = defs
            .storage
            .formats
            .iter()
            .find(|f| f.name == op.format)
            .unwrap();
        let fields = format
            .fields
            .iter()
            .enumerate()
            .map(|(i, f)| format!("{}: _f{i}", f.name))
            .collect::<Vec<_>>()
            .join(", ");
        let locals: BTreeMap<_, _> = crate::packing::projections(
            op,
            format,
            "dfg",
            |name| {
                format!(
                    "*_f{}",
                    format.fields.iter().position(|f| f.name == name).unwrap()
                )
            },
            |v| format!("{v}?"),
        )
        .into_iter()
        .collect();
        let ty = access.value.as_ref().map_or_else(
            || "result?".to_owned(),
            |v| format!("dfg.value_type({})", locals[v]),
        );
        let stored = access
            .value
            .as_ref()
            .map_or_else(|| "None".to_owned(), |v| format!("Some({})", locals[v]));
        writeln!(out, "crate::Opcode::{} => {{ let Self::{} {{ {fields} }} = self else {{ return None; }}; Some(crate::memory::Access {{ ptr: {}, offset: i64::from({}), ty: {ty}, stored: {stored}, flags: self.memory_flags().unwrap_or_default() }}) }},", op.name, format.name, locals[&access.ptr], locals[&access.offset]).unwrap();
    }
    out.push_str("_ => None, } } }\n");
    out
}
