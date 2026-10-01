//! Observation policy is shared by function and module runners.
use crate::{CodegenOptions, Error, Result};
use veloc_profile::{Observation, Profile};

pub(super) fn execute(
    profile: &Profile,
    name: &'static str,
    position: u32,
    entity: &str,
    operation: impl FnOnce() -> Result<()>,
) -> (Result<()>, Observation) {
    let scope = profile.entity_scope(name, position, || entity.into());
    let observation = scope.observation();
    let result = operation();
    // Formatting diagnostics and IR must not extend execution timing.
    scope.result(&result);
    let result = result.map_err(|e| Error::codegen(format!("{entity}/{name}#{position}: {e}")));
    if let Err(error) = &result {
        observation.remark(|| error.to_string());
    }
    (result, observation)
}

pub(crate) fn dump_after(
    name: &str,
    function: &veloc_lir::MachineFunction,
    options: &CodegenOptions,
) {
    if options.dump_after.iter().any(|p| p == "*" || p == name)
        && options
            .dump_function
            .as_deref()
            .is_none_or(|filter| filter == function.name)
    {
        std::eprintln!(
            "===== LIR after {name}: {} =====\n{}",
            function.name,
            function.format_for_dump()
        );
    }
}
