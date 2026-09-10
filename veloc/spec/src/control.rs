//! Build-time control contracts. Lowering consumes
//! instruction views directly, without an intermediate runtime control enum.
use crate::{
    Error,
    model::{Binding, Op, Param, ParamKind, Pattern, TypeList, name},
    syntax::{Kind, Node},
};

pub(crate) struct Control {
    kind: String,
    args: Vec<String>,
}

impl Control {
    /// These effects follow from the lowering primitive, not from its spelling
    /// or a second declaration in every operation.
    pub(crate) fn traits(&self) -> &'static [&'static str] {
        if matches!(self.kind.as_str(), "tail_call" | "tail_call_value") {
            &["MAY_TRAP", "TERMINATOR"]
        } else {
            &["MAY_TRAP"]
        }
    }

    pub(crate) fn validate(&self, source: &str, op: &Op) -> Result<(), Error> {
        let fail = || {
            Error::at(
                source,
                op.offset,
                "control interface must bind all parameters directly and declare its result, terminator and effect contracts",
            )
        };
        let mut seen = std::collections::BTreeSet::new();
        if self.args.len() != op.params.len() || self.args.iter().any(|arg| !seen.insert(arg)) {
            return Err(fail());
        }
        for arg in &self.args {
            if !op
                .packing
                .values()
                .any(|b| matches!(b,Binding::Name(name) if name == arg))
            {
                return Err(fail());
            }
        }
        let create = matches!(self.kind.as_str(), "owned" | "local" | "shared");
        let expected = if create {
            TypeList::Fixed(vec![Pattern::Callable])
        } else if self.kind == "call" {
            TypeList::Signature
        } else {
            TypeList::Fixed(vec![])
        };
        if op.signature.results != expected
            || op.traits.iter().any(|t| t == "TERMINATOR")
                != matches!(self.kind.as_str(), "tail_call" | "tail_call_value")
            || !op.traits.iter().any(|t| t == "MAY_TRAP")
            || op.memory == "NONE"
            || op.semantics.is_some()
        {
            return Err(fail());
        }
        if matches!(self.kind.as_str(), "call" | "tail_call_value" | "drop")
            && !matches!(&op.signature.operands,TypeList::Fixed(p)|TypeList::Variadic(p) if p == &[Pattern::Callable])
        {
            return Err(fail());
        }
        Ok(())
    }
}

pub(crate) fn check(source: &str, node: Node, params: &[Param]) -> Result<Control, Error> {
    let error = || {
        Error::at(
            source,
            node.offset,
            "invalid control interface or parameter kinds",
        )
    };
    let Kind::Call(kind, nodes) = node.kind else {
        return Err(error());
    };
    let expected: &[&str] = match kind.as_str() {
        "owned" => &["FuncId", "values", "FuncId"],
        "local" | "shared" | "tail_call" => &["FuncId", "values"],
        "call" | "tail_call_value" => &["value", "values"],
        "drop" => &["value"],
        _ => return Err(error()),
    };
    if nodes.len() != expected.len() {
        return Err(error());
    }
    let args = nodes
        .into_iter()
        .map(|n| name(source, n))
        .collect::<Result<Vec<_>, _>>()?;
    for (arg, expected) in args.iter().zip(expected) {
        let Some(param) = params.iter().find(|p| p.name == *arg) else {
            return Err(error());
        };
        let matches = match &param.kind {
            ParamKind::Property(ty) => ty == expected,
            ParamKind::Value => *expected == "value",
            ParamKind::Values => *expected == "values",
            _ => false,
        };
        if !matches {
            return Err(error());
        }
    }
    Ok(Control { kind, args })
}
