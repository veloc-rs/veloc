//! Architecture-independent descriptors for learned profitability models.
use super::{RegClass, TargetMachine};

/// Derive context from the selected target's register and scheduling models.
/// CPU names are provenance, not categorical inputs or dispatch conditions.
pub fn policy_context(target: &dyn TargetMachine) -> veloc_policy::Context {
    let desc = target.desc();
    let model = target.schedule_model();
    let mut context = veloc_policy::Context::default();
    for (name, value) in [
        ("arch", desc.arch.name()),
        ("cpu", target.config().cpu.as_str()),
        ("tune", target.config().tune.as_str()),
    ] {
        context.labels.insert(name.into(), value.into());
    }
    for (name, value) in [
        (
            "pointer_bits",
            f32::from(desc.data_layout.pointer_size) * 8.0,
        ),
        ("little_endian", f32::from(desc.data_layout.little_endian)),
        ("issue_width", model.issue_width as f32),
        ("resource_pools", model.resources.len() as f32),
        (
            "resource_units",
            model.resources.iter().map(|r| r.units).sum::<u32>() as f32,
        ),
        (
            "max_latency",
            model.classes.iter().map(|c| c.latency).max().unwrap_or(0) as f32,
        ),
        (
            "max_occupancy",
            model.classes.iter().map(|c| c.occupancy).max().unwrap_or(0) as f32,
        ),
    ] {
        context.features.insert(format!("target.{name}"), value);
    }
    for (name, class) in [
        ("gpr", RegClass::GPR),
        ("fpr", RegClass::FPR),
        ("vector", RegClass::VR),
        ("predicate", RegClass::PR),
    ] {
        let registers = desc.allocatable_regs_in_class(class);
        // Count each physical register once within an allocation class.
        let count = registers
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        context
            .features
            .insert(format!("target.{name}_registers"), count as f32);
    }
    context
}
