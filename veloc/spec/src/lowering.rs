//! Build-time join of MIR recipes and reviewed generic-LIR primitive bindings.
use std::collections::HashSet;
use std::fmt::Write;

use veloc_semantics::BvOp;

use crate::{Definitions, Error};

/// Generate direct MIR-to-LIR dispatch, without runtime semantic identifiers.
///
/// Bindings assert that a generic LIR operation implements the scalar/per-lane
/// primitive with no extra observable effects. This is not a proof of target
/// legalization, predication, ABI behavior, or composed/trapping lowerings.
/// Callers must validate source types before using the generated mapping.
pub fn generate_lowering(defs: &Definitions, bindings: &[(BvOp, &str)]) -> Result<String, Error> {
    let mut primitives = HashSet::new();
    let mut targets = HashSet::new();
    for &(primitive, target) in bindings {
        if !primitives.insert(primitive) || !targets.insert(target) {
            return Err(Error::at("", 0, "ambiguous LIR primitive binding"));
        }
        if !target.starts_with("G_")
            || !target
                .bytes()
                .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
        {
            return Err(Error::at("", 0, "expected a generic LIR opcode identifier"));
        }
    }
    let mut code = String::from(
        "// @generated from MIR semantics and generic LIR bindings.\n\
         fn direct_lowering(opcode: veloc_mir::Opcode) -> Option<veloc_lir::GenericOpcode> {\n\
         match opcode {\n",
    );
    for op in &defs.ops {
        let Some(primitive) = op.semantics.as_ref().and_then(|sem| sem.primitive()) else {
            continue;
        };
        if let Some((_, target)) = bindings.iter().find(|(p, _)| *p == primitive) {
            writeln!(
                code,
                "veloc_mir::Opcode::{} => Some(veloc_lir::GenericOpcode::{target}),",
                op.name
            )
            .unwrap();
        }
    }
    code.push_str("_ => None,\n}\n}\n");
    Ok(code)
}
