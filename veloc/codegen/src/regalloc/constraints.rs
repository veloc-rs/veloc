//! A single view of static machine constraints and per-call ABI requirements.
use crate::target::TargetInstructions;
use crate::{Error, Result};
use veloc_lir::{InstRef, MachineOpcode, OperandConstraint, OperandRef, Placement, Reg};

pub(crate) fn constraints<'a>(
    inst: InstRef<'a>,
    target: &dyn TargetInstructions,
) -> impl Iterator<Item = OperandConstraint> + 'a {
    let static_ = match inst.opcode() {
        MachineOpcode::Target(op) => target.instruction_metadata(op).constraints,
        _ => &[],
    };
    static_
        .iter()
        .chain(inst.constraints())
        .copied()
        .map(|mut c| {
            if let Placement::Registers([reg]) = c.placement {
                c.placement = Placement::Fixed(*reg);
            }
            c
        })
}

/// Validate references independently of the allocator. Physical operands must
/// satisfy constraints only after allocation; SSA operands retain their types.
pub(crate) fn validate(
    inst: InstRef<'_>,
    target: &dyn TargetInstructions,
    allocated: bool,
) -> Result<()> {
    for constraint in constraints(inst, target) {
        let value = *constraint
            .operand
            .get(inst.inputs(), inst.results())
            .ok_or_else(|| Error::codegen("constraint refers to a missing operand"))?;
        let accepts = match constraint.placement {
            Placement::Fixed(reg) => reg.is_preg() && (!allocated || value == reg),
            Placement::Registers(regs) => {
                !regs.is_empty()
                    && regs.iter().all(Reg::is_preg)
                    && (!allocated || regs.contains(&value))
            }
            Placement::Reuse(input) => {
                matches!(constraint.operand, OperandRef::Result(_))
                    && inst
                        .inputs()
                        .get(input)
                        .is_some_and(|&src| !allocated || value == src)
            }
        };
        if !accepts {
            return Err(Error::codegen("invalid operand placement constraint"));
        }
    }
    Ok(())
}
