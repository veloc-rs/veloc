//! Module type structure and restrictions on where types may be used.
use crate::{CallableKind, ModuleData, Result, Type};
use alloc::format;
use veloc_types::TypeInfo;

pub(super) fn validate(module: &ModuleData) -> Result<()> {
    for (id, sig) in module.signatures.iter() {
        for (role, types) in [("parameter", sig.params()), ("return", sig.returns())] {
            for (index, &ty) in types.iter().enumerate() {
                check_type(module, ty).map_err(|error| {
                    crate::Error::Message(format!("signature {id}, {role} {index}: {error}"))
                })?;
            }
        }
        check_returns(sig.returns())
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

/// Shared iterative graph checking handles forward references and deep nesting.
fn check_cycles(module: &ModuleData) -> Result<()> {
    module
        .signatures
        .dependency_order()
        .map(|_| ())
        .map_err(|error| crate::Error::Message(alloc::format!("{error}")))
}
