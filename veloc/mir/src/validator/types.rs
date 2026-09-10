//! Module type structure and restrictions on where types may be used.
use crate::{CallableKind, ModuleData, Result, SigId, Type};
use alloc::{format, vec::Vec};

pub(super) fn validate(module: &ModuleData) -> Result<()> {
    for (id, sig) in &module.signatures {
        for (role, types) in [("parameter", &sig.params), ("return", &sig.returns)] {
            for (index, &ty) in types.iter().enumerate() {
                check_type(module, ty).map_err(|error| {
                    crate::Error::Message(format!("signature {id}, {role} {index}: {error}"))
                })?;
            }
        }
        check_returns(&sig.returns)
            .map_err(|error| crate::Error::Message(format!("signature {id}: {error}")))?;
    }
    check_cycles(module)?;
    for global in &module.globals {
        check_type(module, global.ty)
            .map_err(|error| crate::Error::Message(format!("global {}: {error}", global.name)))?;
        if global.ty.is_callable() {
            return Err(crate::Error::Message(format!(
                "global {}: callable globals require an ownership-aware storage model",
                global.name
            )));
        }
    }
    Ok(())
}

/// Check encoding and references, independently of how this type is used.
pub(super) fn check_type(module: &ModuleData, ty: Type) -> Result<()> {
    if !ty.is_valid() {
        return Err(crate::Error::Message("invalid value type".into()));
    }
    if let Some((id, _)) = ty.as_callable()
        && module.signatures.get(id).is_none()
    {
        return Err(crate::Error::Message(format!(
            "unknown callable signature {id}"
        )));
    }
    Ok(())
}

fn check_returns(returns: &[Type]) -> Result<()> {
    for (index, ty) in returns.iter().enumerate() {
        if matches!(ty.as_callable(), Some((_, CallableKind::Local))) {
            return Err(crate::Error::Message(format!(
                "return {index}: a borrowed callable cannot escape through a return"
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Visit {
    Unseen,
    Active,
    Done,
}

/// References have already been checked. An explicit DFS stack handles deeply
/// nested signatures without using the Rust call stack; shared tails are visited
/// once. Recursion needs an explicit recursive type model, not arbitrary ID cycles.
fn check_cycles(module: &ModuleData) -> Result<()> {
    let mut marks = vec![Visit::Unseen; module.signatures.len()];
    let mut stack: Vec<(SigId, usize)> = Vec::new();
    for (root, _) in &module.signatures {
        if !matches!(marks[root.0 as usize], Visit::Unseen) {
            continue;
        }
        marks[root.0 as usize] = Visit::Active;
        stack.push((root, 0));
        while let Some((id, next)) = stack.last_mut() {
            let sig = &module.signatures[*id];
            let ty = if *next < sig.params.len() {
                sig.params.get(*next)
            } else {
                sig.returns.get(*next - sig.params.len())
            };
            let Some(&ty) = ty else {
                marks[id.0 as usize] = Visit::Done;
                stack.pop();
                continue;
            };
            *next += 1;
            let Some((target, _)) = ty.as_callable() else {
                continue;
            };
            match marks[target.0 as usize] {
                Visit::Active => {
                    return Err(crate::Error::Message(format!(
                        "signature {id} references active signature {target}: recursive callable signature requires an explicit recursive type"
                    )));
                }
                Visit::Done => {}
                Visit::Unseen => {
                    marks[target.0 as usize] = Visit::Active;
                    stack.push((target, 0));
                }
            }
        }
    }
    Ok(())
}
